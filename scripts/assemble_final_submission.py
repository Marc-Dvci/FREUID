#!/usr/bin/env python3
"""Merge Docker-produced private scores into the frozen full Kaggle CSV, with hard checks."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FROZEN_BASE_SHA256 = "6a8d8ca4b58856e761e9aae4b65c18de47bf021512eabae2e818c11954529e6d"
OFFICIAL_SAMPLE_SHA256 = "c5350036e0d1262bd03652d418271ac58c4196b5a210f79da269e948a879a8ab"


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
    parser.add_argument("--public-image-dir", type=Path, required=True,
                        help="official flat public_test image directory")
    parser.add_argument("--out", type=Path, default=Path("submission_final.csv"))
    parser.add_argument("--expected-base-fill", type=float, default=0.5,
                        help="require private ids to still have this placeholder in --base")
    parser.add_argument("--expected-base-sha256", default=FROZEN_BASE_SHA256,
                        help="refuse any public base other than this frozen SHA-256")
    parser.add_argument("--expected-sample-sha256", default=OFFICIAL_SAMPLE_SHA256,
                        help="refuse an unexpected official sample file")
    args = parser.parse_args()

    if sha256(args.base).lower() != args.expected_base_sha256.lower():
        raise ValueError("base SHA-256 does not match the frozen public-base artifact")
    if sha256(args.sample).lower() != args.expected_sample_sha256.lower():
        raise ValueError("sample SHA-256 does not match the audited official file")

    base = load_submission(args.base, "base")
    private = load_submission(args.private, "private")
    sample = load_submission(args.sample, "sample")
    if base["id"].tolist() != sample["id"].tolist():
        raise ValueError("base ids/order do not exactly match the official sample submission")

    base_ids = set(base["id"])
    private_ids = set(private["id"])
    if not args.public_image_dir.is_dir():
        raise FileNotFoundError(f"public image directory not found: {args.public_image_dir}")
    public_paths = [
        path for path in args.public_image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    public_id_list = [path.stem for path in public_paths]
    if len(public_id_list) != len(set(public_id_list)):
        raise ValueError("public image directory contains duplicate filename stems")
    public_ids = set(public_id_list)
    unknown_public = public_ids - base_ids
    if unknown_public:
        raise ValueError(
            f"public images contain {len(unknown_public)} ids absent from sample, "
            f"e.g. {sorted(unknown_public)[:3]}"
        )
    expected_private_ids = base_ids - public_ids
    missing_private = expected_private_ids - private_ids
    extra_private = private_ids - expected_private_ids
    if missing_private or extra_private:
        raise ValueError(
            "private output ids do not exactly equal sample ids minus official public images: "
            f"missing={len(missing_private)}, extra={len(extra_private)}"
        )
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
