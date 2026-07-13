#!/usr/bin/env python3
"""Organizer-facing, no-network inference entrypoint for FREUID Challenge 2026."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader
from tqdm import tqdm

APP_SRC = Path("/app/src")
if not APP_SRC.is_dir():
    APP_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(APP_SRC))
from dataset import FREUIDDataset, get_transforms  # noqa: E402
from models import create_model  # noqa: E402


DATA_DIR = Path(os.environ.get("FREUID_DATA_DIR", "/data"))
OUTPUT_PATH = Path(os.environ.get("FREUID_SUBMISSION_PATH", "/submissions/submission.csv"))
MODEL_DIR = Path(os.environ.get("FREUID_MODEL_DIR", "/models"))
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODEL_NAMES = ("cnxb512_MAURITIUS-ID.pth", "dinov2b_full.pth")
# Conservative cross-domain hedge selected before private-test release. The legacy model is much
# stronger on the official real-recapture stress slice; DINOv2 contributes unseen-layout diversity.
MODEL_WEIGHTS = (0.75, 0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FREUID submission.csv")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--workers", type=int, default=0,
        help="0 is robust to Docker's small default /dev/shm; raise only with adequate shm",
    )
    parser.add_argument("--tta", action="store_true", help="opt-in horizontal-flip TTA (not frozen default)")
    return parser.parse_args()


def discover_images(data_dir: Path) -> pd.DataFrame:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    paths = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"no supported image files found directly under {data_dir}")
    ids = [p.stem for p in paths]
    duplicate_mask = pd.Series(ids).duplicated()
    duplicates = pd.Series(ids)[duplicate_mask].unique().tolist()
    if duplicates:
        raise ValueError(f"duplicate filename stems in /data, e.g. {duplicates[:3]}")
    return pd.DataFrame({"id": ids, "abs_path": [str(p) for p in paths]})


def rank_norm(values: np.ndarray) -> np.ndarray:
    """Average ranks in [0, 1]; ties must not acquire arbitrary filename-order signal."""
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1)


@torch.inference_mode()
def predict_checkpoint(
    checkpoint_path: Path,
    test_df: pd.DataFrame,
    batch_size: int,
    workers: int,
    device: str,
    tta: bool,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]
    image_size = int(model_args.get("img_size", 384))
    model = create_model(
        model_args["backbone"], pretrained=False,
        head=model_args.get("head", "linear"), img_size=image_size,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device, memory_format=torch.channels_last).eval()

    _, transform = get_transforms(image_size)
    dataset = FREUIDDataset(test_df, transform, is_test=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device == "cuda", persistent_workers=workers > 0,
    )

    chunks: list[np.ndarray] = []
    for images, _ in tqdm(loader, desc=checkpoint_path.name, mininterval=5):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(images)
            if tta:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2
        chunks.append(torch.sigmoid(logits.float()).cpu().numpy())

    del model, checkpoint, loader, dataset
    if device == "cuda":
        torch.cuda.empty_cache()
    scores = np.concatenate(chunks)
    if len(scores) != len(test_df) or not np.isfinite(scores).all():
        raise ValueError(f"invalid predictions from {checkpoint_path.name}")
    return scores


def validate_submission(frame: pd.DataFrame, expected_ids: list[str]) -> None:
    if list(frame.columns) != ["id", "label"]:
        raise ValueError(f"invalid columns: {list(frame.columns)}")
    if frame["id"].astype(str).tolist() != expected_ids:
        raise ValueError("output ids/order differ from discovered input images")
    labels = frame["label"].to_numpy(dtype=float)
    if not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 1)):
        raise ValueError("labels must be finite values in [0, 1]")


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("an NVIDIA CUDA GPU is required for full-test inference")
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    test_df = discover_images(args.data_dir.resolve())
    checkpoint_paths = [args.model_dir / name for name in MODEL_NAMES]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen checkpoint(s): {missing}")

    print(
        f"device={device} images={len(test_df)} models={len(checkpoint_paths)} "
        f"batch_size={args.batch_size} workers={args.workers} tta={args.tta}",
        file=sys.stderr,
    )
    ranked = []
    for checkpoint_path in checkpoint_paths:
        scores = predict_checkpoint(
            checkpoint_path, test_df, args.batch_size, args.workers,
            device, tta=args.tta,
        )
        ranked.append(rank_norm(scores))

    weights = np.asarray(MODEL_WEIGHTS, dtype=np.float64)
    if len(weights) != len(ranked) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("invalid frozen ensemble weights")
    labels = np.average(np.stack(ranked), axis=0, weights=weights)
    expected_ids = test_df["id"].astype(str).tolist()
    submission = pd.DataFrame({"id": expected_ids, "label": labels})
    validate_submission(submission, expected_ids)

    output_root = Path(os.environ.get("FREUID_OUTPUT_DIR", "/submissions")).resolve()
    output = args.output.resolve()
    if output.parent != output_root or output.name != "submission.csv":
        raise ValueError(f"output must be exactly {output_root / 'submission.csv'}")
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"wrote {len(submission)} rows to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
