#~/bin/bash

#1: number of GPUs
#2: Model File Address
#3: RACE Data Directory Address
#4: Output Directory Address

NGPU_PER_NODE=${1:-8}
MODEL_FILE=${2:-"./bert-large-uncased/bert-large-uncased-whole-word-masking-pytorch_model.bin"}
RACE_DIR=${3:-"RACE"}
OUTPUT_DIR=${4:-"result"}
LR=${5:-0.00002}
SEED=${6:-12345}
MASTER_PORT=${7:-29500}
DROPOUT=${8:-0.1}
echo "lr is ${LR}"
echo "seed is $SEED"
echo "master port is $MASTER_PORT"
echo "dropout is ${DROPOUT}"

# Force deepspeed to run with only local node
NUM_NODES=1
HOSTFILE=/dev/null

NUM_TRAIN_EPOCHS=6
WARMUP_PROPORTION=0.2
NGPU=$((NGPU_PER_NODE*NUM_NODES))
EFFECTIVE_BATCH_SIZE=480
MAX_GPU_BATCH_SIZE=15
PER_GPU_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NGPU))
if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi
JOB_NAME="deepspeed_${NGPU}GPUs_${EFFECTIVE_BATCH_SIZE}batch_size_${NUM_TRAIN_EPOCHS}epochs"
config_json=dp_config.json

run_cmd="deepspeed \
       --master_port=${MASTER_PORT} \
       --hostfile ${HOSTFILE} \
       race_deepspeed.py \
       --bert_model ./bert-large-uncased \
       --do_train \
       --do_lower_case \
       --data_dir $RACE_DIR \
       --train_batch_size $PER_GPU_BATCH_SIZE \
       --learning_rate ${LR} \
       --num_train_epochs ${NUM_TRAIN_EPOCHS} \
       --max_seq_length 384 \
       --output_dir $OUTPUT_DIR \
       --job_name ${JOB_NAME} \
       --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
       --warmup_proportion ${WARMUP_PROPORTION} \
       --deepspeed \
       --fp16 \
       --deepspeed_config ${config_json} \
       --dropout ${DROPOUT} \
       --model_file ${MODEL_FILE} \
       --ckpt_type HF \
       --origin_bert_config_file ./bert-large-uncased/config.json \
       --deepspeed_transformer_kernel \
       --seed ${SEED}
       "
echo ${run_cmd}
eval ${run_cmd}
