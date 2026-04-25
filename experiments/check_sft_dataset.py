import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import SidSFTDataset


def main():
    train_file = PROJECT_ROOT / "data" / "Amazon" / "train" / "Industrial_and_Scientific_5_2016-10-2018-11.csv"

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print("=== Load Tokenizer ===")
    print("model_name:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("pad_token:", tokenizer.pad_token)
    print("eos_token:", tokenizer.eos_token)
    print("vocab_size:", len(tokenizer))
    print()

    print("=== Build SidSFTDataset ===")
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

    print("dataset length:", len(dataset))
    print()

    print("=== Inspect Sample 0 ===")
    sample = dataset[0]

    for key, value in sample.items():
        print(key, type(value), len(value))

    input_ids = sample["input_ids"]
    labels = sample["labels"]
    attention_mask = sample["attention_mask"]

    valid_label_positions = [i for i, x in enumerate(labels) if x != -100]
    valid_label_ids = [labels[i] for i in valid_label_positions]

    print()
    print("input_ids length:", len(input_ids))
    print("attention_mask length:", len(attention_mask))
    print("labels length:", len(labels))
    print("valid label token count:", len(valid_label_ids))
    print("valid label positions:", valid_label_positions)
    print()

    print("=== Decoded Full Input ===")
    print(tokenizer.decode(input_ids))
    print()

    print("=== Decoded Target Label Tokens ===")
    print(tokenizer.decode(valid_label_ids))
    print()

    print("=== Tensor Dry-run Batch Shape ===")
    batch = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }

    for key, value in batch.items():
        print(key, tuple(value.shape), value.dtype)

    print()
    print("[OK] SidSFTDataset dry-run passed.")


if __name__ == "__main__":
    main()