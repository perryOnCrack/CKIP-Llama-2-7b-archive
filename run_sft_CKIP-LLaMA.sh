#!/bin/bash
output_dir="outputs" # output file
train_file="data/sample_train.jsonl" # training file
validation_file="data/sample_validation.jsonl" # validation file
export BELLE_DIR="./BELLE"
model_name_or_path="ckiplab/CKIP-Llama-2-7b" # or ckiplab/CKIP-Llama-2-7b-chat
cache_dir="hf_cache_dir"

mkdir -p ${output_dir}
mkdir -p ${cache_dir}

export CUDA_VISIBLE_DEVICES='0,1,2,3'
export WANDB_PROJECT="CKIP-Llama"
# export WANDB_ENTITY="WANDB_ENTITY"
# export WANDB_RESUME=allow
export PYTHONPATH="$BELLE_DIR/train"

# FT
torchrun --nproc_per_node 4 $BELLE_DIR/train/src/entry_point/sft_train.py \
    --llama \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --deepspeed $BELLE_DIR/train/configs/deepspeed_config.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 2 \
    --model_max_length 4096 \
    --save_total_limit 10 \
    --save_strategy "steps" \
    --evaluation_strategy "steps" \
    --save_steps 200 \
    --eval_steps 100 \
    --learning_rate 8e-6 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 20 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --use_flash_attention \
    | tee -a ${output_dir}/train.log
   # --resume_from_checkpoint ...
