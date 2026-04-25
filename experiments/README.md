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

### Stage 4: Analysis

Analyze the trade-off between:

- HR@10 / NDCG@10
- Tail Ratio@10 / Tail Recall@10
- Coverage
- Invalid Rate