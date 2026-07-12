"""
Ensemble inference for FREUID submission — combines multiple checkpoints.

Supports:
  * Multiple checkpoints (glob patterns)
  * TTA (horizontal flip)
  * Geometric mean or rank-average ensembling
  * Fills missing test IDs (for private set not yet released) with 0.5

Usage:
  python src/infer_ensemble.py --ckpts checkpoints/cnxb384_full.pth checkpoints/dinov2b_full.pth checkpoints/fnoise_full.pth --tta --out submission_ensemble.csv
"""
import os, glob, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FREUIDDataset, get_transforms
from models import create_model

EPS = 1e-6


def build_test_df(img_dir):
    files = []
    for ext in ("*.jpeg", "*.jpg", "*.png"):
        files += glob.glob(os.path.join(img_dir, ext))
    ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return pd.DataFrame({"id": ids, "abs_path": files})


@torch.no_grad()
def predict_one(ckpt_path, test_df, batch_size, workers, device, tta=False):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    backbone = a["backbone"]
    img_size = a.get("img_size", 384)
    head = a.get("head", "linear")
    
    model = create_model(backbone, pretrained=False, head=head, img_size=img_size)
    model.load_state_dict(ck["model"])
    model = model.to(device, memory_format=torch.channels_last).eval()
    
    _, tf = get_transforms(img_size)
    ds = FREUIDDataset(test_df, tf, is_test=True, return_id=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    
    probs, ids = [], []
    for x, _, batch_ids in tqdm(dl, desc=f"  {os.path.basename(ckpt_path)}"):
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logit = model(x)
            if tta:
                logit = (logit + model(torch.flip(x, dims=[3]))) / 2
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        ids.extend(batch_ids)
    return np.asarray(ids), np.concatenate(probs)


def rank_norm(x):
    """Rank-normalize to [0, 1] — calibration-free common scale."""
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


def ensemble_predictions(per_model, method="rank"):
    """Combine multiple model predictions."""
    if len(per_model) == 1:
        return per_model[0]
    
    M = np.vstack(per_model)
    if method == "rank":
        # Rank-average: each model contributes equally regardless of calibration
        ranked = np.array([rank_norm(p) for p in M])
        return ranked.mean(axis=0)
    else:
        # Geometric mean
        p = np.clip(M, EPS, 1 - EPS)
        return np.exp(np.mean(np.log(p), axis=0))


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(sum([glob.glob(c) for c in args.ckpts], []))
    assert ckpts, f"no checkpoints matched {args.ckpts}"
    print(f"device: {device} | ckpts: {ckpts} | method: {args.method}")
    
    test_df = build_test_df(args.img_dir)
    print(f"Found {len(test_df)} test images in {args.img_dir}")
    
    per_model = []
    ids_ref = None
    for ck in ckpts:
        ids, probs = predict_one(ck, test_df, args.batch_size, args.workers, device, args.tta)
        order = np.argsort(ids)
        ids, probs = ids[order], probs[order]
        ids_ref = ids if ids_ref is None else ids_ref
        per_model.append(probs)
        print(f"  {os.path.basename(ck)}: mean={probs.mean():.4f} std={probs.std():.4f}")
    
    ens = ensemble_predictions(per_model, method=args.method)

    if args.normalize_per_template:
        # Rank within each document template, so every template contributes the same score
        # distribution to the single global operating point the FREUID score is read at.
        from template_norm import cluster_templates, normalize_per_template
        path_by_id = dict(zip(test_df["id"].astype(str), test_df["abs_path"]))
        paths = [path_by_id[i] for i in ids_ref]
        clusters, k = cluster_templates(paths)
        sizes = np.bincount(clusters)
        print(f"per-template normalization: k={k} (silhouette), cluster sizes={sorted(sizes, reverse=True)}")
        ens = normalize_per_template(ens, clusters, alpha=args.template_alpha)

    pred_map = dict(zip(ids_ref, ens))

    # Align to sample_submission
    sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    score_col = [c for c in sub.columns if c != "id"][0]
    sub[score_col] = sub["id"].map(pred_map).fillna(args.fill).astype(float)
    n_pred = sub["id"].isin(pred_map).sum()
    sub.to_csv(args.out, index=False)
    
    print(f"\nWrote {args.out}: {n_pred}/{len(sub)} predicted, {len(sub)-n_pred} filled with {args.fill}")
    print(f"Score stats: min={sub[score_col].min():.4f} mean={sub[score_col].mean():.4f} max={sub[score_col].max():.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="the-freuid-challenge-2026-ijcai-ecai")
    p.add_argument("--img_dir", default="the-freuid-challenge-2026-ijcai-ecai/public_test/public_test")
    p.add_argument("--ckpts", nargs="+", required=True, help="checkpoint glob(s)")
    p.add_argument("--out", default="submission_ensemble.csv")
    p.add_argument("--method", default="rank", choices=["rank", "geomean"])
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--fill", type=float, default=0.5)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--normalize_per_template", action="store_true",
                   help="rank scores within each discovered document template before pooling")
    p.add_argument("--template_alpha", type=float, default=1.0,
                   help="1.0 = pure within-template rank; 0.0 = global rank only")
    main(p.parse_args())
