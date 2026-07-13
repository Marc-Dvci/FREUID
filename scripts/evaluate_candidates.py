#!/usr/bin/env python3
"""Evaluate frozen checkpoints and rank ensembles on labeled CSV probes."""

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset import FREUIDDataset, get_transforms  # noqa: E402
from metrics import calculate_freuid_score  # noqa: E402
from models import create_model  # noqa: E402


DOC_RE = re.compile(r"([a-z]{2,3}\d{2})_\d{2}_\d{2}", re.I)


def rank_norm(values: np.ndarray) -> np.ndarray:
    """Average ranks in [0, 1]; ties must not acquire arbitrary row-order signal."""
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1)


def doc_id(path: str) -> str:
    match = DOC_RE.search(str(path).replace("\\", "/"))
    return match.group(1).lower() if match else "unknown"


def metric(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(x) for x in calculate_freuid_score(labels, scores))


@torch.inference_mode()
def predict_dual(
    checkpoint_path: Path, frame: pd.DataFrame, base_batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["args"]
    image_size = int(model_args.get("img_size", 384))
    batch_size = max(1, int(base_batch_size * (384 / image_size) ** 2))
    model = create_model(
        model_args["backbone"], pretrained=False,
        head=model_args.get("head", "linear"), img_size=image_size,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device, memory_format=torch.channels_last).eval()
    _, transform = get_transforms(image_size)
    loader = DataLoader(
        FREUIDDataset(frame, transform), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=device == "cuda",
    )

    raw_chunks: list[np.ndarray] = []
    tta_chunks: list[np.ndarray] = []
    for images, _ in tqdm(loader, desc=checkpoint_path.stem, mininterval=5):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(images)
            flip_logits = model(torch.flip(images, dims=[3]))
        raw_chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
        tta_chunks.append(torch.sigmoid(((logits + flip_logits) / 2).float()).cpu().numpy())

    del model, checkpoint, loader
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(raw_chunks), np.concatenate(tta_chunks)


def weight_grid(n: int, denominator: int = 4):
    for counts in itertools.product(range(denominator + 1), repeat=n):
        if sum(counts) != denominator:
            continue
        yield np.asarray(counts, dtype=float) / denominator


def evaluate_ensembles(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray], split_name: str
) -> pd.DataFrame:
    labels = frame["label"].to_numpy(dtype=int)
    docs = frame["doc"].to_numpy()
    names = list(predictions)
    ranked = {name: rank_norm(predictions[name]) for name in names}
    rows = []

    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            scores = np.mean([ranked[name] for name in subset], axis=0)
            fr, au, ap = metric(labels, scores)
            doc_scores = [metric(labels[docs == d], scores[docs == d])[0] for d in sorted(set(docs))]
            rows.append({
                "split": split_name, "method": "equal_rank", "members": "+".join(subset),
                "weights": "", "freuid": fr, "audet": au, "apcer01": ap,
                "doc_mean": float(np.mean(doc_scores)), "doc_worst": float(np.max(doc_scores)),
            })

    # Coarse weights across every candidate, including zeros. This subsumes subset search and
    # reveals whether a nominally weak member consistently helps the strict false-positive tail.
    for weights in weight_grid(len(names), denominator=4):
        if np.count_nonzero(weights) < 2:
            continue
        scores = sum(weights[i] * ranked[name] for i, name in enumerate(names))
        fr, au, ap = metric(labels, scores)
        doc_scores = [metric(labels[docs == d], scores[docs == d])[0] for d in sorted(set(docs))]
        rows.append({
            "split": split_name, "method": "weighted_rank", "members": "+".join(names),
            "weights": ",".join(f"{w:.2f}" for w in weights),
            "freuid": fr, "audet": au, "apcer01": ap,
            "doc_mean": float(np.mean(doc_scores)), "doc_worst": float(np.max(doc_scores)),
        })
    return pd.DataFrame(rows).sort_values(["freuid", "doc_mean", "doc_worst"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--ckpts", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation_cache"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split-name", default="probe")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    if not {"id", "abs_path", "label"}.issubset(frame.columns):
        raise ValueError("evaluation CSV requires id,abs_path,label")
    missing = frame.loc[~frame["abs_path"].map(os.path.exists), "abs_path"]
    if len(missing):
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing.iloc[0]}")
    frame = frame.reset_index(drop=True)
    frame["doc"] = frame["abs_path"].map(doc_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_predictions: dict[str, np.ndarray] = {}
    tta_predictions: dict[str, np.ndarray] = {}
    for checkpoint_path in args.ckpts:
        name = checkpoint_path.stem
        raw, tta = predict_dual(checkpoint_path, frame, args.batch_size, device)
        if not np.isfinite(raw).all() or not np.isfinite(tta).all():
            raise ValueError(f"non-finite predictions from {name}")
        raw_predictions[name] = raw
        tta_predictions[name] = tta
        fr0 = metric(frame["label"].to_numpy(), raw)
        fr1 = metric(frame["label"].to_numpy(), tta)
        print(f"{name}: raw={fr0} tta={fr1}")

    cache = frame[["id", "label", "doc"]].copy()
    for name, values in raw_predictions.items():
        cache[f"{name}__raw"] = values
        cache[f"{name}__tta"] = tta_predictions[name]
    cache.to_csv(args.out_dir / f"{args.split_name}_predictions.csv", index=False)

    raw_results = evaluate_ensembles(frame, raw_predictions, f"{args.split_name}_raw")
    tta_results = evaluate_ensembles(frame, tta_predictions, f"{args.split_name}_tta")
    results = pd.concat([raw_results, tta_results], ignore_index=True).sort_values(
        ["freuid", "doc_mean", "doc_worst"]
    )
    results.to_csv(args.out_dir / f"{args.split_name}_ensembles.csv", index=False)
    print("\nTop candidate ensembles:")
    print(results.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
