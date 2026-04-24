import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="experiments/item_frequency_summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.train_file)

    if "item_id" not in df.columns:
        raise ValueError("Column 'item_id' not found in train_file.")

    freq = df["item_id"].value_counts().reset_index()
    freq.columns = ["item_id", "freq"]

    total_items = len(freq)
    total_interactions = int(freq["freq"].sum())

    # 按频次从高到低排序后，用 item 数量比例划分 head/middle/tail
    freq = freq.sort_values("freq", ascending=False).reset_index(drop=True)
    freq["rank"] = freq.index + 1
    freq["item_percentile"] = freq["rank"] / total_items

    def group_item(p):
        if p <= 0.2:
            return "head"
        elif p <= 0.8:
            return "middle"
        else:
            return "tail"

    freq["group"] = freq["item_percentile"].apply(group_item)

    summary = (
        freq.groupby("group")
        .agg(
            num_items=("item_id", "count"),
            total_interactions=("freq", "sum"),
            avg_freq=("freq", "mean"),
            min_freq=("freq", "min"),
            max_freq=("freq", "max"),
        )
        .reset_index()
    )

    summary["item_ratio"] = summary["num_items"] / total_items
    summary["interaction_ratio"] = summary["total_interactions"] / total_interactions

    print("Total items:", total_items)
    print("Total interactions:", total_interactions)
    print()
    print(summary)

    summary.to_csv(args.output_file, index=False)
    print(f"\nSaved summary to: {args.output_file}")


if __name__ == "__main__":
    main()