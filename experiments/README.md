# MiniOneRec Reproduction Notes

## Project Goal

Reproduce the core MiniOneRec pipeline and improve the SFT stage with a long-tail-aware weighted loss for generative recommendation.

## Target Position

Tencent Search / Recommendation Algorithm Engineer

## Planned Dataset

Initial reproduction dataset:

- Amazon Reviews: Industrial & Scientific
- Backup dataset: Office Products

## Baseline Pipeline

1. Data preprocessing
2. Semantic ID construction
3. SFT training
4. Top-K recommendation evaluation

## Improvement Direction

Introduce a long-tail-aware / popularity-aware weighted loss in the SFT stage.

The goal is to reduce head item bias and improve long-tail item exposure while keeping the overall recommendation accuracy acceptable.

## Planned Metrics

Overall accuracy:

- HR@10
- NDCG@10

Generation quality:

- Invalid Rate

Long-tail recommendation:

- Coverage
- Tail Ratio@10
- Tail Recall@10

## Experiment Plan

### Stage 1: Baseline Reproduction

- Run official preprocessing script
- Run official SFT script
- Run official evaluation script
- Record baseline metrics

### Stage 2: Long-tail-aware Loss

- Count item frequency in training data
- Split items into head / middle / tail groups
- Assign higher loss weight to tail items
- Train with the same setup as baseline

### Stage 3: Ablation Study

Compare:

- Baseline
- Weighted loss with small tail weight
- Weighted loss with medium tail weight
- Weighted loss with large tail weight

Local SFT data pipeline checks passed.

- SidSFTDataset correctly reads Industrial_and_Scientific processed CSV.
- Input history comes from history_item_sid.
- Prediction target comes from item_sid.
- Prompt tokens are masked as -100 in labels.
- Only target semantic ID tokens and EOS participate in SFT loss.
- DataCollatorForSeq2Seq successfully pads a 4-sample batch.

## Baseline SFT Results: Sid-only Qwen2.5-0.5B on Industrial_and_Scientific

Cloud setup:

- Platform: AutoDL
- GPU: RTX 4090D 24GB
- Model: Qwen/Qwen2.5-0.5B-Instruct
- Dataset: Industrial_and_Scientific_5_2016-10-2018-11
- Training mode: SidSFTDataset only
- Epochs: 3
- Batch size: 64
- Micro batch size: 4
- Learning rate: 5e-5

Accuracy metrics:

| Metric | Value |
|---|---:|
| HR@10 | 0.1297 |
| NDCG@10 | 0.0913 |
| Invalid Rate | 0.0000 |

Long-tail metrics:

| Metric | Value |
|---|---:|
| Coverage@10 | 0.1851 |
| Tail Ratio@10 | 0.0213 |
| Tail Recall@10 | 0.0288 |

Observation:

The sid-only SFT baseline achieves a usable HR@10 and NDCG@10, but long-tail exposure is weak. Only about 2.13% of top-10 recommendation slots are tail items, and Tail Recall@10 is only about 2.88%. This supports the motivation for adding a long-tail-aware weighted loss in the SFT stage.

### Stage 4: Analysis

Analyze the trade-off between:

- HR@10 / NDCG@10
- Tail Ratio@10 / Tail Recall@10
- Coverage
- Invalid Rate