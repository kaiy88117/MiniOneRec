#!/bin/bash
set -e

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_DISABLED=true
export WANDB_MODE=disabled

python sft.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --eval_file data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output_dir outputs/sft_long_tail_industrial_lt_large \
  --sample -1 \
  --seed 42 \
  --batch_size 64 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 5e-5 \
  --cutoff_len 512 \
  --category Industrial_and_Scientific \
  --sid_index_path data/Amazon/index/Industrial_and_Scientific.index.json \
  --item_meta_path data/Amazon/index/Industrial_and_Scientific.item.json \
  --dataset_mode sid_long_tail \
  --long_tail_head_weight 1.0 \
  --long_tail_middle_weight 1.5 \
  --long_tail_tail_weight 2.0 \
  --bf16 True