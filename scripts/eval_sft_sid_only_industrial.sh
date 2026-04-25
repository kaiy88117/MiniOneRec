#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache

export WANDB_DISABLED=true
export WANDB_MODE=disabled
CHECKPOINT_DIR="outputs/sft_sid_only_industrial_qwen25_05b/final_checkpoint"

TEST_FILE="data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"
INFO_FILE="data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"

RESULT_DIR="results"
RESULT_JSON="${RESULT_DIR}/sft_sid_only_industrial_qwen25_05b_test.json"

mkdir -p "${RESULT_DIR}"

python evaluate.py \
  --base_model "${CHECKPOINT_DIR}" \
  --test_data_path "${TEST_FILE}" \
  --info_file "${INFO_FILE}" \
  --category "Industrial_and_Scientific" \
  --result_json_data "${RESULT_JSON}" \
  --batch_size 4 \
  --K 0 \
  --max_new_tokens 32 \
  --num_beams 50 \
  --length_penalty 0.0

python calc.py \
  --path "${RESULT_JSON}" \
  --item_path "${INFO_FILE}"