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

import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
# import apex
import deepspeed

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertForMultipleChoice
from turing.nvidia_modeling import BertForMultipleChoice, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.utils import is_main_process

def setup_logging(logfile):
    logging.basicConfig(filename=logfile, filemode='a+',
                        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('logging started.\n')

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
                 label = 4):
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
def read_race_examples(filename):
    examples = []
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
    for example_index, example in enumerate(examples):
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
        ## display some example
        # if example_index < 1:
        #     logging.info("*** Example ***")
        #     logging.info(f"race_id: {example.race_id}")
        #     for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
        #         logging.info(f"choice: {choice_idx}")
        #         logging.info(f"tokens: {' '.join(tokens)}")
        #         logging.info(f"input_ids: {' '.join(map(str, input_ids))}")
        #         logging.info(f"input_mask: {' '.join(map(str, input_mask))}")
        #         logging.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
        #     if is_training:
        #         logging.info(f"label: {label}")
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

    # Note: only truncate sequence A 
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        tokens_a.pop()
        # if len(tokens_a) > len(tokens_b):
        #     tokens_a.pop()
        # else:
        #     tokens_b.pop()

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

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def main():
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
    parser.add_argument("--checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint directory where the model chekckpoints exists.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--has_ans",
                        default=True,
                        action='store_true',
                        help="Whether has answer in the eval dataset")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--deepspeed_transformer_kernel',
                        default=False,
                        action='store_true',
                        help='Use DeepSpeed transformer kernel to accelerate.')
    parser.add_argument(
        '--ckpt_type',
        type=str,
        default="DS",
        help="Checkpoint's type, DS - DeepSpeed, TF - Tensorflow, HF - Huggingface.")
    parser.add_argument(
        '--ckpt_id',
        type=str,
        default="0",
        help="Checkpoint's ID.")
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
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    setup_logging(os.path.join(args.checkpoint, 'eval.log'))
    logging.info(f"----- checkpoint {args.ckpt_id} -----")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(0)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    # logging.info("device: {} ({}) n_gpu: {}".format(device, torch.cuda.get_device_name(0), n_gpu))

    # tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model

    # output_model_file = os.path.join(args.checkpoint, "pytorch_model.bin")
    # model_state_dict = torch.load(output_model_file)
    # model = BertForMultipleChoice.from_pretrained(args.bert_model,
    #     state_dict=model_state_dict,
    #     num_choices=4)
    # model.to(device)
    
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
        # if args.preln:
        #     bert_config = BertConfigPreLN(**bert_model_config)
        # else:
        #     bert_config = BertConfig(**bert_model_config)
        bert_config = BertConfig(**bert_model_config)
    else:
        # Models from Tensorflow and Huggingface are post-LN.
        # if args.preln:
        #     raise ValueError(
        #         "Should NOT use --preln if the loading checkpoint doesn't use pre-layer-norm."
        #     )

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

    # if args.preln:
    #     model = BertForQuestionAnsweringPreLN(bert_config, args)
    # else:
    #     model = BertForQuestionAnswering(bert_config, args)
    
    print("VOCAB SIZE:", bert_config.vocab_size)
    
    model = BertForMultipleChoice(bert_config, args, num_choices=4)

    print("VOCAB SIZE:", bert_config.vocab_size)
    if args.model_file != "0":
        # logging.info(f"Loading Pretrained Bert Encoder from: {args.model_file}")

        if args.ckpt_type == "DS":
            checkpoint_state_dict = torch.load(
                args.model_file, map_location=torch.device("cpu"))
            if 'module' in checkpoint_state_dict:
                # logging.info('Loading DeepSpeed v2.0 style checkpoint')
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

        # logging.info(f"Pretrained Bert Encoder Loaded from: {args.model_file}")

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

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

    model.load_checkpoint(args.checkpoint, args.ckpt_id)

    test_high = os.path.join(args.data_dir, 'high')
    test_middle = os.path.join(args.data_dir, 'middle')

    ## test high
    filenames = os.listdir(test_high)
    model.eval()
    logging.info("***** Running evaluation: test high *****")
    logging.info("  Num examples = %d", len(filenames))
    eval_iter = tqdm(filenames, disable=False)
    eval_answers = {}
    high_eval_accuracy = 0
    high_nb_eval_examples = 0
    for fid, filename in enumerate(eval_iter):
        file_path = os.path.join(test_high, filename)
        eval_examples = read_race_examples(file_path)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, True)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
        eval_answer = []
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            # eval_answer.extend(chr(np.argmax(logits, axis=1)+ord("A")))
            if args.has_ans:
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(logits, label_ids)
                # print(label_ids, np.argmax(logits, axis=1), tmp_eval_accuracy)
                high_eval_accuracy += tmp_eval_accuracy
                high_nb_eval_examples += input_ids.size(0)

        # eval_answers[filename] = eval_answer
    eval_accuracy = high_eval_accuracy / high_nb_eval_examples
    logging.info("high eval accuracy: {}".format(eval_accuracy))
    # result = json.dumps(eval_answers)

    ## test middle
    filenames = os.listdir(test_middle)
    model.eval()
    logging.info("***** Running evaluation: test middle *****")
    logging.info("  Num examples = %d", len(filenames))
    eval_iter = tqdm(filenames, disable=False)
    eval_answers = {}
    middle_eval_accuracy = 0
    middle_nb_eval_examples = 0
    for fid, filename in enumerate(eval_iter):
        file_path = os.path.join(test_middle, filename)
        eval_examples = read_race_examples(file_path)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, True)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)
        eval_answer = []
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            if args.has_ans:
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(logits, label_ids)
                # print(label_ids, np.argmax(logits, axis=1), tmp_eval_accuracy)
                middle_eval_accuracy += tmp_eval_accuracy
                middle_nb_eval_examples += input_ids.size(0)

    eval_accuracy = middle_eval_accuracy / middle_nb_eval_examples
    logging.info("middle eval accuracy: {}".format(eval_accuracy))

    # all test
    eval_accuracy = (middle_eval_accuracy + high_eval_accuracy) / (middle_nb_eval_examples + high_nb_eval_examples)
    logging.info("overall eval accuracy: {}".format(eval_accuracy))

    # output_eval_file = os.path.join(args.output_dir, "answers.json")
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir, exist_ok=True)

    # with open(output_eval_file, "w") as f:
    #     f.write(result)


if __name__ == "__main__":
    main()
