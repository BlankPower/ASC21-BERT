# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import time
import builtins
import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
import numpy as np
import torch
import deepspeed
# from apex import amp
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
from turing.nvidia_modeling import BertForMultipleChoice, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertForMultipleChoice
from pytorch_pretrained_bert.optimization import BertAdam
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.utils import is_main_process
from tensorboardX import SummaryWriter

SUMMARY_WRITER_DIR_NAME = 'tensorboard'
all_step_time = 0.0

def setup_logging(logfile):
    if is_main_process():
        logging.basicConfig(filename=logfile, filemode='w',
                            format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info('logging started.\n')
    else:
        def print_none(*args, **kwargs):
            pass

        # 将内置print函数变为一个空函数，从而使非主进程的进程不会输出。         
        builtins.print = print_none


class RaceExample(object):
    """A single training/test example for the RACE dataset."""
    '''
    For RACE dataset:
    race_id: data id
    context_sentence: article
    start_ending: question
    ending_0/1/2/3: option_0/1/2/3
    label: true answer
    '''
    def __init__(self,
                 race_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label = None):
        self.race_id = race_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"id: {self.race_id}",
            f"article: {self.context_sentence}",
            f"question: {self.start_ending}",
            f"option_0: {self.endings[0]}",
            f"option_1: {self.endings[1]}",
            f"option_2: {self.endings[2]}",
            f"option_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)



class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

## paths is a list containing all paths
def read_race_examples(paths, debug=False):
    examples = []
    count = 0
    for path in paths:
        filenames = glob.glob(path+"/*txt")
        for filename in filenames:
            if debug:
                count += 1
                if count > 100:
                    break
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                article = data_raw['article']
                ## for each qn
                for i in range(len(data_raw['answers'])):
                    truth = ord(data_raw['answers'][i]) - ord('A')
                    question = data_raw['questions'][i]
                    options = data_raw['options'][i]
                    examples.append(
                        RaceExample(
                            race_id = filename+'-'+str(i),
                            context_sentence = article,
                            start_ending = question,

                            ending_0 = options[0],
                            ending_1 = options[1],
                            ending_2 = options[2],
                            ending_3 = options[3],
                            label = truth))
                
    return examples 



def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # RACE is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # The input will be like:
    # [CLS] Article [SEP] Question + Option [SEP]
    # for each option 
    # 
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    max_option_len = 0
    examples_iter = tqdm(examples, desc="Preprocessing: ", disable=False) if is_main_process() else examples
    for example_index, example in enumerate(examples_iter):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        
        features.append(
            InputFeatures(
                example_id = example.race_id,
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def get_summary_writer(name, base=".."):
    """Returns a tensorboard summary writer
    """
    return SummaryWriter(
        log_dir=os.path.join(base, SUMMARY_WRITER_DIR_NAME, name))


def write_summary_events(summary_writer, summary_events):
    for event in summary_events:
        summary_writer.add_scalar(event[0], event[1], event[2])

# from apex.multi_tensor_apply import multi_tensor_applier
# class GradientClipper:
#     """
#     Clips gradient norm of an iterable of parameters.
#     """
#     def __init__(self, max_grad_norm):
#         self.max_norm = max_grad_norm
#         if multi_tensor_applier.available:
#             import amp_C
#             self._overflow_buf = torch.cuda.IntTensor([0])
#             self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
#             self.multi_tensor_scale = amp_C.multi_tensor_scale
#         else:
#             raise RuntimeError('Gradient clipping requires cuda extensions')

#     def step(self, parameters):
#         l = [p.grad for p in parameters if p.grad is not None]
#         total_norm, _ = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [l], False)
#         total_norm = total_norm.item()
#         if (total_norm == float('inf')): return
#         clip_coef = self.max_norm / (total_norm + 1e-6)
#         if clip_coef < 1:
#             multi_tensor_applier(self.multi_tensor_scale, self._overflow_buf, [l, l], clip_coef)

def main():
    
    ete_start = time.time()
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--debug',
                        default=False,
                        action='store_true',
                        help='debug with small dataset')
    parser.add_argument('--job_name',
                        type=str,
                        default=None,
                        help='Output path for Tensorboard event files.')
    parser.add_argument('--print_steps',
                        type=int,
                        default=100,
                        help='Interval to print training details.')
    parser.add_argument('--deepspeed_transformer_kernel',
                        default=False,
                        action='store_true',
                        help='Use DeepSpeed transformer kernel to accelerate.')
    parser.add_argument(
        '--ckpt_type',
        type=str,
        default="DS",
        help="Checkpoint's type, DS - DeepSpeed, TF - Tensorflow, HF - Huggingface.")
    parser.add_argument("--model_file",
                        type=str,
                        default="0",
                        help="Path to the Pretrained BERT Encoder File.")
    parser.add_argument(
        "--origin_bert_config_file",
        type=str,
        default=None,
        help="The config json file corresponding to the non-DeepSpeed pre-trained BERT model."
    )
    parser.add_argument(
        '--preln',
        action='store_true',
        default=False,
        help=
        "Whether to display the breakdown of the wall-clock time for foraward, backward and step"
    )
    parser.add_argument('--gpus',
                        type=int,
                        default=2,
                        help="gpus that distributed training use")
    parser.add_argument(
        '--loss_plot_alpha',
        type=float,
        default=0.2,
        help='Alpha factor for plotting moving average of loss.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    deepspeed.init_distributed(dist_backend='nccl')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.train_batch_size = int(args.train_batch_size / 
                                args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, 
                                              do_lower_case=args.do_lower_case)
    
    # output directory
    suffix = args.job_name
    suffix += '_debug' if args.debug else ''
    args.output_dir = os.path.join(args.output_dir, suffix)
    os.makedirs(args.output_dir, exist_ok=True, mode=0o777)

    setup_logging(os.path.join(args.output_dir, 'train.log'))

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_dir = os.path.join(args.data_dir, 'train')
        train_examples = read_race_examples([train_dir+'/high', train_dir+'/middle'], args.debug)
        logging.info("train_examples: {}\n".format(len(train_examples)))
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model

    # model = BertForMultipleChoice.from_pretrained(args.bert_model,
    #     cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
    #     num_choices=4)
    # model.to(device)
    # if args.local_rank != -1:
        # model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    bert_model_config = {
        "vocab_size_or_config_json_file": 119547,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": args.dropout,
        "attention_probs_dropout_prob": args.dropout,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02
    }

    if args.ckpt_type == "DS":
        if args.preln:
            bert_config = BertConfigPreLN(**bert_model_config)
        else:
            bert_config = BertConfig(**bert_model_config)
        bert_config = BertConfig(**bert_model_config)
    else:
        # Models from Tensorflow and Huggingface are post-LN.
        if args.preln:
            raise ValueError(
                "Should NOT use --preln if the loading checkpoint doesn't use pre-layer-norm."
            )

        # Use the original bert config if want to load from non-DeepSpeed checkpoint.
        if args.origin_bert_config_file is None:
            raise ValueError(
                "--origin_bert_config_file is required for loading non-DeepSpeed checkpoint."
            )

        bert_config = BertConfig.from_json_file(args.origin_bert_config_file)

        if bert_config.vocab_size != len(tokenizer.vocab):
            raise ValueError("vocab size from original checkpoint mismatch.")

    bert_config.vocab_size = len(tokenizer.vocab)
    # Padding for divisibility by 8
    if bert_config.vocab_size % 8 != 0:
        vocab_diff = 8 - (bert_config.vocab_size % 8)
        bert_config.vocab_size += vocab_diff

    model = BertForMultipleChoice(bert_config, args, num_choices=4)

    print("VOCAB SIZE:", bert_config.vocab_size)
    if args.model_file != "0":
        logging.info(f"Loading Pretrained Bert Encoder from: {args.model_file}")

        if args.ckpt_type == "DS":
            checkpoint_state_dict = torch.load(
                args.model_file, map_location=torch.device("cpu"))
            if 'module' in checkpoint_state_dict:
                logging.info('Loading DeepSpeed v2.0 style checkpoint')
                model.load_state_dict(checkpoint_state_dict['module'],
                                      strict=False)
            elif 'model_state_dict' in checkpoint_state_dict:
                model.load_state_dict(
                    checkpoint_state_dict['model_state_dict'], strict=False)
            else:
                raise ValueError("Unable to find model state in checkpoint")
        else:
            from convert_bert_ckpt_to_deepspeed import convert_ckpt_to_deepspeed
            convert_ckpt_to_deepspeed(model, args.ckpt_type, args.model_file,
                                      vocab_diff,
                                      args.deepspeed_transformer_kernel)

        logging.info(f"Pretrained Bert Encoder Loaded from: {args.model_file}")

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # for p in param_optimizer:
    #     logging.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.deepspeed_transformer_kernel:
        no_decay = no_decay + [
            'attn_nw', 'attn_nb', 'norm_w', 'norm_b', 'attn_qkvb', 'attn_ob',
            'inter_b', 'output_b'
        ]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    model, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=optimizer_grouped_parameters,
        dist_init_required=True)
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() 
                              and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
    
    logging.info("device: {} ({}), n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
                device, torch.cuda.get_device_name(0), n_gpu, bool(args.local_rank != -1), args.fp16))
    logging.info("train batch size: {}, gradient accumulation steps: {}, GPUs: {}".format(
                args.train_batch_size, args.gradient_accumulation_steps, torch.distributed.get_world_size()))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
    # Prepare Summary writer
    if is_main_process() and args.job_name is not None:
        args.summary_writer = get_summary_writer(name=args.job_name,
                                                 base=args.output_dir)
    else:
        args.summary_writer = None

    logging.info("propagate deepspeed-config settings to client settings")
    if args.train_batch_size != model.train_micro_batch_size_per_gpu() // 4:
        raise ValueError("batch size error!")
    if args.gradient_accumulation_steps != model.gradient_accumulation_steps():
        raise ValueError("gradient accumulation steps error!")
    args.fp16 = model.fp16_enabled()
    args.print_steps = model.steps_per_print()
    if args.learning_rate != model.get_lr()[0]:
        raise ValueError("learning rate error")
    args.wall_clock_breakdown = model.wall_clock_breakdown()

    if args.local_rank != -1:
        t_total = num_train_steps // torch.distributed.get_world_size()
    else:
        t_total = num_train_steps

    global_step = 0
    train_start = time.time()
    if args.do_train:
        if not os.path.exists('./train_features.pt'):
            train_features = convert_examples_to_features(
                    train_examples, tokenizer, args.max_seq_length, True)
            torch.save(train_features, './train_features.pt')
        else:
            train_features = torch.load('./train_features.pt')
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data) 
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        sample_count = 0
        ema_loss = 0.
        global all_step_time
        ave_rounds = 20
        # scaler = GradScaler()
        for ep in range(int(args.num_train_epochs)):
            train_iter = tqdm(train_dataloader, disable=False) if is_main_process() else train_dataloader
            if is_main_process():
                train_iter.set_description("Trianing Epoch: {}/{}".format(ep+1, int(args.num_train_epochs)))
            for step, batch in enumerate(train_iter):
                start_time = time.time()
                bs_size = batch[0].size()[0]
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                ema_loss = args.loss_plot_alpha * ema_loss + (
                    1 - args.loss_plot_alpha) * loss.item()

                model.backward(loss)
                loss_item = loss.item() * args.gradient_accumulation_steps
                loss = None

                sample_count += (args.train_batch_size *
                                 torch.distributed.get_world_size())

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step

                    model.step()
                    global_step += 1

                    if is_main_process():
                        train_iter.set_postfix(loss=loss_item)
                        summary_events = [
                            (f'Train/Steps/lr', lr_this_step, global_step),
                            (f'Train/Samples/train_loss', loss_item,
                             sample_count),
                            (f'Train/Samples/lr', lr_this_step, sample_count),
                            (f'Train/Samples/train_ema_loss', ema_loss,
                             sample_count)
                        ]

                        if args.fp16 and hasattr(optimizer, 'cur_scale'):
                            summary_events.append(
                                (f'Train/Samples/scale', optimizer.cur_scale,
                                 sample_count))
                        write_summary_events(args.summary_writer,
                                             summary_events)
                        args.summary_writer.flush()                        

                    if is_main_process() and (step + 1) % args.print_steps == 0:
                        logging.info(f"bert_race_progress: step={global_step} lr={lr_this_step} loss={ema_loss}")
                else:
                    model.step()

                one_step_time = time.time() - start_time
                all_step_time += one_step_time
                if (step + 1) % (
                        ave_rounds) == 0 and torch.distributed.get_rank() == 0:
                    print(
                        ' At step {}, averaged throughput for {} rounds is: {} Samples/s'
                        .format(
                            step, ave_rounds,
                            bs_size * ave_rounds *
                            torch.distributed.get_world_size() /
                            all_step_time),
                        flush=True)
                    all_step_time = 0.0
            model.save_checkpoint(args.output_dir)

    finish_time = time.time()
    # Save a trained model
    if is_main_process():
        logging.info("ete_time: {}, training_time: {}".format(finish_time - ete_start, finish_time - train_start))
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        # torch.save(model_to_save.state_dict(), output_model_file)
        # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # model.save_checkpoint(args.output_dir)

if __name__ == "__main__":
    main()
