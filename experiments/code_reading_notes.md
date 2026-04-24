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

## 4. Evaluation

Related files:

- evaluate.py
- calc.py
- LogitProcessor.py

Questions:

- How are HR@K and NDCG@K computed?
- How are invalid items detected?
- Where can Coverage, Tail Ratio@K, and Tail Recall@K be added?