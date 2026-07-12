"""
Cross-validation design for FREUID.

The competition is explicitly about generalization across document domains and across the
digital -> print-and-capture gap. Standard random K-fold massively over-estimates performance
here. We therefore use:

  * LODO  -- Leave-One-Document-type-Out: 5 folds, each holding out one of the 5 train types.
            Directly measures cross-document generalization (the stated objective).
  * A recapture stress split is applied at the dataset/aug level (see augment_recapture.py):
            the held-out fold is evaluated both clean and with simulated print-and-capture so we
            can watch the digital->physical gap, plus the 20 real recaptured rows.

This mirrors the leave-one-group-out lesson from past work: per-group holdout, and never pick
ensemble members by greedy in-fold search (it overfits) -- select by LODO-averaged FREUID.
"""
import os
import numpy as np
import pandas as pd


def resolve_image_path(data_dir, image_path):
    """Resolve a train_labels.csv image_path (e.g. 'train/<id>.jpeg') to the real file.

    The competition zip nests images one level deeper than the csv path implies
    (train/train/<id>.jpeg, public_test/public_test/<id>.jpeg).
    """
    base = os.path.basename(image_path)
    top = image_path.split("/")[0] if "/" in image_path else "train"
    candidates = [
        os.path.join(data_dir, image_path),                 # train/<id>.jpeg
        os.path.join(data_dir, top, top, base),             # train/train/<id>.jpeg (nested)
        os.path.join(data_dir, top, base),                  # train/<id>.jpeg (flat)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[1]  # best guess (nested) -- caller will raise on read if missing


def load_train_df(data_dir, csv_name="train_labels.csv"):
    """Load labels, attach resolved absolute paths and parsed metadata columns."""
    df = pd.read_csv(os.path.join(data_dir, csv_name))
    df = df[df["id"].notna()].reset_index(drop=True)

    df["abs_path"] = df["image_path"].apply(lambda p: resolve_image_path(data_dir, p))
    df["label"] = df["label"].astype(int)
    # is_digital may be bool or string "True"/"False"
    df["is_digital"] = df["is_digital"].astype(str).str.strip().str.lower().map(
        {"true": 1, "1": 1, "false": 0, "0": 0}
    ).fillna(1).astype(int)
    parts = df["type"].astype(str).str.split("/", n=1, expand=True)
    df["country"] = parts[0]
    df["doctype"] = parts[1] if parts.shape[1] > 1 else ""
    return df


def lodo_folds(df, group_col="type"):
    """Yield (fold_name, train_idx, val_idx) leaving one document group out at a time."""
    groups = sorted(df[group_col].unique())
    for g in groups:
        val_idx = df.index[df[group_col] == g].to_numpy()
        train_idx = df.index[df[group_col] != g].to_numpy()
        yield str(g), train_idx, val_idx


def summarize(df):
    """Quick console summary of the split-relevant structure."""
    print("rows:", len(df))
    print("label balance:\n", df["label"].value_counts(normalize=True).round(3).to_dict())
    print("is_digital (1=digital):\n", df["is_digital"].value_counts().to_dict())
    print("types:")
    g = df.groupby("type").agg(n=("label", "size"), fraud_rate=("label", "mean"))
    print(g.round(3).to_string())
    print("recaptured rows (is_digital=0):", int((df["is_digital"] == 0).sum()))


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "the-freuid-challenge-2026-ijcai-ecai"
    df = load_train_df(data_dir)
    summarize(df)
    print("\nLODO folds:")
    for name, tr, va in lodo_folds(df):
        print(f"  hold out {name:16s} train={len(tr):6d} val={len(va):6d} "
              f"val_fraud_rate={df.loc[va, 'label'].mean():.3f}")
    # verify a couple of resolved paths actually exist
    missing = (~df["abs_path"].apply(os.path.exists)).sum()
    print(f"\nresolved paths missing on disk: {missing} / {len(df)}")
