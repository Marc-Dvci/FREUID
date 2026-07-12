#!/usr/bin/env python3
"""Merge Docker-produced private scores into the frozen full Kaggle CSV, with hard checks."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def load_submission(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str})
    if list(frame.columns) != ["id", "label"]:
        raise ValueError(f"{name} must have exactly id,label columns; got {list(frame.columns)}")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"{name} contains missing or duplicate ids")
    values = pd.to_numeric(frame["label"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name} labels must be finite numeric values in [0,1]")
    frame["label"] = values
    return frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True,
                        help="full Kaggle CSV with already-frozen public scores")
    parser.add_argument("--private", type=Path, required=True,
                        help="private-only submission.csv emitted by the Docker image")
    parser.add_argument("--sample", type=Path, required=True,
                        help="official full sample_submission.csv")
    parser.add_argument("--out", type=Path, default=Path("submission_final.csv"))
    parser.add_argument("--expected-base-fill", type=float, default=0.5,
                        help="require private ids to still have this placeholder in --base")
    args = parser.parse_args()

    base = load_submission(args.base, "base")
    private = load_submission(args.private, "private")
    sample = load_submission(args.sample, "sample")
    if base["id"].tolist() != sample["id"].tolist():
        raise ValueError("base ids/order do not exactly match the official sample submission")

    base_ids = set(base["id"])
    private_ids = set(private["id"])
    unknown = private_ids - base_ids
    if unknown:
        raise ValueError(
            f"private output contains {len(unknown)} ids absent from sample, e.g. {sorted(unknown)[:3]}"
        )
    if not private_ids:
        raise ValueError("private output is empty")

    base_index = base.set_index("id")
    old_values = base_index.loc[private["id"], "label"].to_numpy(dtype=float)
    if not np.allclose(old_values, args.expected_base_fill, rtol=0, atol=1e-12):
        raise ValueError("private ids in base are not untouched placeholder values; refusing overwrite")

    private_map = private.set_index("id")["label"]
    replace_mask = base["id"].isin(private_ids)
    base.loc[replace_mask, "label"] = base.loc[replace_mask, "id"].map(private_map)
    if not np.array_equal(
        base.set_index("id").loc[private["id"], "label"].to_numpy(),
        private["label"].to_numpy(),
    ):
        raise AssertionError("private values changed during merge")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(base)} rows, replaced {replace_mask.sum()} private rows")
    print(f"sha256={sha256(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
