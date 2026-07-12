"""
Ensemble inference -> FREUID submission CSV.

* Loads one or more trained checkpoints (src/train.py output).
* Predicts on whatever test images are present (default: local public_test). For ids in
  sample_submission that we did NOT predict (e.g. the ~135k private images we don't have
  locally), fills a constant so public-LB iteration works -- the public LB scores only the
  public subset.
* Aggregates models by GEOMETRIC MEAN of probabilities (beats arithmetic mean for this kind of
  ensemble) with optional hflip TTA and per-model temperature.
* Output header matches the real sample_submission.csv (id,label).

Examples:
  python src/inference.py --ckpts checkpoints/cnx_b_MAURITIUS-ID.pth --img_dir the-freuid-challenge-2026-ijcai-ecai/public_test/public_test
  python src/inference.py --ckpts checkpoints/*.pth --out submission.csv --tta
"""
import os, glob, argparse
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader

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
def predict_one(ckpt_path, test_df, img_size, batch_size, workers, device, tta=False):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = ck["args"]["backbone"]
    head = ck["args"].get("head", "linear")
    model = create_model(backbone, pretrained=False, head=head, img_size=img_size)  # dispatches forensic_* too
    model.load_state_dict(ck["model"])
    model = model.to(device, memory_format=torch.channels_last).eval()

    _, tf = get_transforms(img_size)
    ds = FREUIDDataset(test_df, tf, is_test=True, return_id=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    probs, ids = [], []
    for x, _, batch_ids in dl:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logit = model(x)
            if tta:
                logit = (logit + model(torch.flip(x, dims=[3]))) / 2
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        ids.extend(batch_ids)
    return np.asarray(ids), np.concatenate(probs)


def geomean(prob_matrix):
    """Geometric mean across models (rows=models). prob_matrix shape (M, N)."""
    p = np.clip(prob_matrix, EPS, 1 - EPS)
    return np.exp(np.mean(np.log(p), axis=0))


def score_labeled(args):
    """OOD/real validation: ensemble-predict on a LABELED set (abs_path,label[,type,source])
    and report the true FREUID. This is the trustworthy signal once external real captured/
    recaptured data is available -- the LODO/recapture CV on FREUID-only is saturated."""
    from metrics import calculate_freuid_score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(sum([glob.glob(c) for c in args.ckpts], []))
    assert ckpts, f"no checkpoints matched {args.ckpts}"
    df = pd.read_csv(args.labeled_csv)
    assert {"abs_path", "label"}.issubset(df.columns), "labeled_csv needs abs_path,label"
    df = df[df["abs_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"labeled eval: {len(df)} images, {ckpts}")

    mats = []
    for ck in ckpts:                                  # predict_one preserves df order
        _, probs = predict_one(ck, df, args.img_size, args.batch_size, args.workers, device, args.tta)
        mats.append(probs)
    ens = geomean(np.vstack(mats)) if len(mats) > 1 else mats[0]

    y = df["label"].values
    fr, au, ap = calculate_freuid_score(y, ens)
    print(f"\n=== OOD FREUID: {fr:.4f} (audet={au:.4f} apcer@1%={ap:.4f})  n={len(df)} ===")
    for col in ("source", "type"):
        if col in df.columns:
            print(f"-- by {col} --")
            for v, g in df.assign(pred=ens).groupby(col):
                if g["label"].nunique() < 2:
                    print(f"   {v:18s} n={len(g):5d}  (single-class, skip)"); continue
                gfr, gau, gap = calculate_freuid_score(g["label"].values, g["pred"].values)
                print(f"   {str(v):18s} n={len(g):5d}  FREUID={gfr:.4f} audet={gau:.4f} apcer@1%={gap:.4f}")


def main(args):
    if args.labeled_csv:
        return score_labeled(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(sum([glob.glob(c) for c in args.ckpts], []))
    assert ckpts, f"no checkpoints matched {args.ckpts}"
    print("device:", device, "| ckpts:", ckpts)

    test_df = build_test_df(args.img_dir)
    print(f"found {len(test_df)} test images in {args.img_dir}")

    per_model = []
    ids_ref = None
    for ck in ckpts:
        ids, probs = predict_one(ck, test_df, args.img_size, args.batch_size, args.workers, device, args.tta)
        order = np.argsort(ids)
        ids, probs = ids[order], probs[order]
        ids_ref = ids if ids_ref is None else ids_ref
        per_model.append(probs)
        print(f"  {os.path.basename(ck)}: mean_prob={probs.mean():.4f}")

    ens = geomean(np.vstack(per_model)) if len(per_model) > 1 else per_model[0]
    pred_map = dict(zip(ids_ref, ens))

    # align to sample_submission (preserves order + score column name; constant-fill the rest)
    sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    score_col = [c for c in sub.columns if c != "id"][0]
    sub[score_col] = sub["id"].map(pred_map).fillna(args.fill).astype(float)
    n_pred = sub["id"].isin(pred_map).sum()
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {n_pred}/{len(sub)} predicted, {len(sub)-n_pred} filled with {args.fill}")
    print("score stats:", sub[score_col].describe()[["min", "mean", "max"]].round(4).to_dict())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="the-freuid-challenge-2026-ijcai-ecai")
    p.add_argument("--img_dir", default="the-freuid-challenge-2026-ijcai-ecai/public_test/public_test")
    p.add_argument("--ckpts", nargs="+", required=True, help="checkpoint glob(s)")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--labeled_csv", default=None,
                   help="if set: OOD eval mode -- score ckpts on this labeled set (abs_path,label) and report FREUID")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--fill", type=float, default=0.5, help="constant for ids without a local image")
    p.add_argument("--tta", action="store_true")
    main(p.parse_args())
