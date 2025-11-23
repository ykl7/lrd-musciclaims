#!/usr/bin/env python3
"""
Create a permanent train/dev split for the Qwen-VL judgement task.

Usage
------
$ python split_data.py \
      --input         ./data_collection/all_data_with_judge_without_fig_ref_v3.csv \
      --label_col     class \
      --out_dir       ./data_splits \
      --train_ratio   0.90 \
      --seed          42
"""

import argparse, hashlib, json, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

VALID_LABELS = {"SUPPORT", "CONTRADICT", "NEUTRAL"}

def hash_file(path, block_size=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(block_size)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="master CSV file")
    ap.add_argument("--label_col", default="class")
    ap.add_argument("--out_dir",  default="./data_splits")
    ap.add_argument("--train_ratio", type=float, default=0.90)
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    # --- Derived value ---
    dev_ratio = 1.0 - args.train_ratio
    if not (0 < args.train_ratio < 1):
        sys.exit("--train_ratio must be between 0 and 1")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"🔹 Loading {args.input} …")
    df = pd.read_csv(args.input)

    # ---- Minimal cleaning ----
    df["caption"] = df["caption"].fillna("")
    df[args.label_col] = df[args.label_col].str.strip().str.upper()
    df = df[df[args.label_col].isin(VALID_LABELS)].copy()
    print(f"  Remaining rows after cleaning: {len(df)}")

    # ---- Perform the two-way split ----
    train_df, dev_df = train_test_split(
        df,
        train_size=args.train_ratio,
        stratify=df[args.label_col],
        random_state=args.seed,
    )

    # ---- Save splits ----
    out_paths = {
        "train": os.path.join(args.out_dir, "train.csv"),
        "dev"  : os.path.join(args.out_dir, "dev.csv"),
    }
    train_df.to_csv(out_paths["train"], index=False)
    dev_df.to_csv(out_paths["dev"],     index=False)

    # ---- Save metadata ----
    meta = dict(
        source_file         = args.input,
        source_sha256_16hex = hash_file(args.input),
        label_column        = args.label_col,
        n_rows_total        = len(df),
        n_train             = len(train_df),
        n_dev               = len(dev_df),
        ratios              = dict(train=args.train_ratio, dev=round(dev_ratio, 4)),
        seed                = args.seed,
    )
    with open(os.path.join(args.out_dir, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"✅ Saved splits to {args.out_dir}")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
