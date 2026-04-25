import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import SidSFTDataset


def main():
    train_file = PROJECT_ROOT / "data" / "Amazon" / "train" / "Industrial_and_Scientific_5_2016-10-2018-11.csv"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print("=== Load Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("pad_token:", tokenizer.pad_token)
    print("pad_token_id:", tokenizer.pad_token_id)
    print("eos_token:", tokenizer.eos_token)
    print("eos_token_id:", tokenizer.eos_token_id)
    print()

    print("=== Build Dataset ===")
    dataset = SidSFTDataset(
        train_file=str(train_file),
        tokenizer=tokenizer,
        max_len=512,
        sample=8,
        test=False,
        seed=0,
        category="Industrial_and_Scientific",
        dedup=False,
    )

    samples = [dataset[i] for i in range(4)]

    print("raw sample lengths:")
    for i, sample in enumerate(samples):
        print(
            i,
            "input_ids:", len(sample["input_ids"]),
            "labels:", len(sample["labels"]),
            "valid_labels:", sum(x != -100 for x in sample["labels"]),
        )
    print()

    print("=== Build DataCollatorForSeq2Seq Batch ===")
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    batch = collator(samples)

    for key, value in batch.items():
        print(key, tuple(value.shape), value.dtype)

    print()
    print("labels ignore_index count:", int((batch["labels"] == -100).sum().item()))
    print("labels valid count:", int((batch["labels"] != -100).sum().item()))
    print("input padding count:", int((batch["input_ids"] == tokenizer.pad_token_id).sum().item()))
    print()

    print("=== Decode Each Target Label ===")
    for i in range(batch["labels"].shape[0]):
        valid_label_ids = batch["labels"][i][batch["labels"][i] != -100].tolist()
        print(f"[sample {i}] {tokenizer.decode(valid_label_ids)}")

    print()
    print("[OK] DataCollatorForSeq2Seq batch check passed.")


if __name__ == "__main__":
    main()