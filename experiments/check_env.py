import os
import sys
import importlib
from pathlib import Path

import pandas as pd
import torch
import transformers


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_python_env():
    print("=== Python Environment ===")
    print("python:", sys.executable)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("transformers:", transformers.__version__)
    print()


def check_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"[OK] import {module_name}")
    except Exception as e:
        print(f"[FAIL] import {module_name}")
        print("      ", repr(e))


def check_project_imports():
    print("=== Project Imports ===")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    modules = [
        "data",
        "sft",
        "sft_gpr",
        "minionerec_trainer",
    ]

    for module_name in modules:
        check_import(module_name)

    print()


def check_data_files():
    print("=== Data Files ===")

    paths = {
        "train_file": PROJECT_ROOT / "data" / "Amazon" / "train" / "Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "eval_file": PROJECT_ROOT / "data" / "Amazon" / "valid" / "Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "test_file": PROJECT_ROOT / "data" / "Amazon" / "test" / "Industrial_and_Scientific_5_2016-10-2018-11.csv",
        "sid_index_path": PROJECT_ROOT / "data" / "Amazon" / "index" / "Industrial_and_Scientific.index.json",
        "item_meta_path": PROJECT_ROOT / "data" / "Amazon" / "index" / "Industrial_and_Scientific.item.json",
    }

    for name, path in paths.items():
        if path.exists():
            print(f"[OK] {name}: {path}")
        else:
            print(f"[FAIL] {name}: {path}")

    print()

    train_file = paths["train_file"]
    if train_file.exists():
        df = pd.read_csv(train_file)
        print("=== Train CSV Preview ===")
        print("shape:", df.shape)
        print("columns:", list(df.columns))
        print(df.head(3))
        print()

        required_columns = [
            "user_id",
            "history_item_title",
            "item_title",
            "history_item_id",
            "item_id",
            "history_item_sid",
            "item_sid",
        ]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print("[FAIL] missing columns:", missing)
        else:
            print("[OK] all required columns exist")

        print("unique items:", df["item_id"].nunique())
        print("interactions:", len(df))


def main():
    check_python_env()
    check_project_imports()
    check_data_files()


if __name__ == "__main__":
    main()