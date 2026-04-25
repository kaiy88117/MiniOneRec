#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE="data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"
EVAL_FILE="data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv"
SID_INDEX_PATH="data/Amazon/index/Industrial_and_Scientific.index.json"
ITEM_META_PATH="data/Amazon/index/Industrial_and_Scientific.item.json"

OUTPUT_DIR="outputs/debug_sft_sid_only_industrial_qwen25_05b"

python sft.py \
  --base_model "${MODEL_NAME}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --category "Industrial_and_Scientific" \
  --sid_index_path "${SID_INDEX_PATH}" \
  --item_meta_path "${ITEM_META_PATH}" \
  --dataset_mode "sid_only" \
  --sample 128 \
  --batch_size 16 \
  --micro_batch_size 2 \
  --num_epochs 1 \
  --learning_rate 5e-5 \
  --cutoff_len 512 \
  --wandb_project "" \
  --wandb_run_name "debug_sft_sid_only_industrial_qwen25_05b"