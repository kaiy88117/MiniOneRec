# Code Reading Notes

## 1. Data Processing

Related files:

- data/
- data.py
- convert_dataset.py
- convert_dataset_gpr.py

Questions:

- What is the raw input format?
- What is the processed training format?
- Where is item frequency stored or can be counted?
- How are user histories constructed?

## 2. SID Construction

Related files:

- rq/

Questions:

- How are item semantic IDs generated?
- What is the SID format?
- How are SIDs mapped back to items?

## 3. SFT Training

Related files:

- sft.py
- sft_gpr.py
- minionerec_trainer.py

Questions:

- Where is the model loaded?
- Where is the tokenizer extended with SID tokens?
- Where is the loss computed?
- Can loss weights be added at token level or sample level?

## Current Finding: SFT Loss Location

- `sft.py` uses the default `transformers.Trainer`, so the SFT baseline loss is computed by the model internally.
- `sft_gpr.py` defines `VAFT_Trainer`, which overrides `compute_loss`.
- `VAFT_Trainer.compute_loss` computes token-level CE loss with `reduction='none'`, averages it into sequence-level loss, and multiplies it by `final_value`.
- This provides a good reference implementation for long-tail-aware weighted loss.
- Next step: inspect dataset classes in `data.py` to determine whether each training sample contains target item information that can be mapped to item frequency.

## Current Finding: Dataset Fields for Long-tail Loss

- `SidSFTDataset` uses `history_item_sid` as the user interaction history and `item_sid` as the target output.
- `SidSFTDataset_GPR` additionally uses `history_item_id`, `item_id`, `item_sid`, and `item_features`.
- `SidSFTDataset_GPR` returns `final_value`, which is consumed by `VAFT_Trainer.compute_loss` in `sft_gpr.py`.
- This provides a reusable implementation pattern for long-tail-aware weighted loss.
- For long-tail loss, the preferred target identifier is `item_id`, because item frequency should be counted from item interactions rather than generated SID tokens.
- Next step: inspect the actual train CSV format and confirm whether `item_id` is available in the processed training file.

## Current Finding: Available Processed Amazon Data

- The repository already contains processed Amazon data under `data/Amazon`.
- Available categories:
  - `Industrial_and_Scientific`
  - `Office_Products`
- Available train / valid / test CSV files:
  - `data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv`
  - `data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv`
  - `data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv`
  - `data/Amazon/train/Office_Products_5_2016-10-2018-11.csv`
  - `data/Amazon/valid/Office_Products_5_2016-10-2018-11.csv`
  - `data/Amazon/test/Office_Products_5_2016-10-2018-11.csv`
- Available SID / item metadata files:
  - `data/Amazon/index/Industrial_and_Scientific.index.json`
  - `data/Amazon/index/Industrial_and_Scientific.item.json`
  - `data/Amazon/index/Office_Products.index.json`
  - `data/Amazon/index/Office_Products.item.json`
- Initial baseline will use `Industrial_and_Scientific`.

# Long-tail Analysis

## Dataset

- Category: Industrial_and_Scientific
- Train file: `data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv`

## Frequency Distribution

| Group | Item Ratio | Interaction Ratio |
|---|---:|---:|
| Head | 19.99% | 55.44% |
| Middle | 59.98% | 39.30% |
| Tail | 20.02% | 5.26% |

## Observation

The training data shows a clear long-tail distribution. The top 20% items contribute more than half of all training interactions, while the bottom 20% items contribute only about 5.26% of interactions.

This imbalance may cause the SFT model to overfit popular items and underfit tail items. Therefore, introducing a long-tail-aware weighted loss is reasonable for improving tail item exposure and tail recall.

## 4. Evaluation

Related files:

- evaluate.py
- calc.py
- LogitProcessor.py

Questions:

- How are HR@K and NDCG@K computed?
- How are invalid items detected?
- Where can Coverage, Tail Ratio@K, and Tail Recall@K be added?