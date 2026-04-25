import argparse
import json
from collections import Counter

import pandas as pd


def build_item_frequency(train_file):
    df = pd.read_csv(train_file)
    item_counter = Counter(df["item_id"].astype(str).tolist())
    return item_counter


def split_head_middle_tail(item_counter, head_ratio=0.2, tail_ratio=0.2):
    items_sorted = sorted(item_counter.items(), key=lambda x: x[1], reverse=True)
    n_items = len(items_sorted)

    n_head = int(n_items * head_ratio)
    n_tail = int(n_items * tail_ratio)

    head_items = set(item for item, _ in items_sorted[:n_head])
    tail_items = set(item for item, _ in items_sorted[-n_tail:])
    middle_items = set(item for item, _ in items_sorted[n_head:-n_tail])

    return head_items, middle_items, tail_items


def load_sid_to_item_id(info_file):
    sid_to_item = {}

    with open(info_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            sid = parts[0].strip()
            item_id = parts[-1].strip()
            sid_to_item[sid] = item_id

    return sid_to_item


def normalize_sid(x):
    return str(x).strip().strip('"').strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    item_counter = build_item_frequency(args.train_file)
    head_items, middle_items, tail_items = split_head_middle_tail(item_counter)

    sid_to_item = load_sid_to_item_id(args.info_file)

    with open(args.pred_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    total_users = len(test_data)
    total_pred_slots = 0
    invalid_pred_count = 0

    recommended_items_at_k = set()
    tail_pred_count_at_k = 0
    valid_pred_count_at_k = 0

    tail_target_total = 0
    tail_target_hit_at_k = 0

    head_target_total = 0
    middle_target_total = 0

    for sample in test_data:
        target_sid = sample["output"]
        if isinstance(target_sid, list):
            target_sid = target_sid[0]
        target_sid = normalize_sid(target_sid)

        target_item = sid_to_item.get(target_sid)

        if target_item in tail_items:
            tail_target_total += 1
        elif target_item in head_items:
            head_target_total += 1
        elif target_item in middle_items:
            middle_target_total += 1

        preds = [normalize_sid(x) for x in sample["predict"][: args.topk]]
        total_pred_slots += len(preds)

        hit_tail_target = False

        for pred_sid in preds:
            pred_item = sid_to_item.get(pred_sid)

            if pred_item is None:
                invalid_pred_count += 1
                continue

            valid_pred_count_at_k += 1
            recommended_items_at_k.add(pred_item)

            if pred_item in tail_items:
                tail_pred_count_at_k += 1

            if target_item in tail_items and pred_item == target_item:
                hit_tail_target = True

        if hit_tail_target:
            tail_target_hit_at_k += 1

    coverage_at_k = len(recommended_items_at_k) / len(item_counter) if item_counter else 0.0
    tail_ratio_at_k = tail_pred_count_at_k / valid_pred_count_at_k if valid_pred_count_at_k else 0.0
    tail_recall_at_k = tail_target_hit_at_k / tail_target_total if tail_target_total else 0.0
    invalid_rate = invalid_pred_count / total_pred_slots if total_pred_slots else 0.0

    result = {
        "topk": args.topk,
        "total_users": total_users,
        "total_items": len(item_counter),
        "head_items": len(head_items),
        "middle_items": len(middle_items),
        "tail_items": len(tail_items),
        "head_target_total": head_target_total,
        "middle_target_total": middle_target_total,
        "tail_target_total": tail_target_total,
        "coverage_at_k": coverage_at_k,
        "tail_ratio_at_k": tail_ratio_at_k,
        "tail_recall_at_k": tail_recall_at_k,
        "invalid_count": invalid_pred_count,
        "invalid_rate": invalid_rate,
        "unique_recommended_items_at_k": len(recommended_items_at_k),
        "valid_pred_count_at_k": valid_pred_count_at_k,
        "tail_pred_count_at_k": tail_pred_count_at_k,
        "tail_target_hit_at_k": tail_target_hit_at_k,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()