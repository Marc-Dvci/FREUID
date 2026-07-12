"""
Evaluate OOF predictions + combine models for the LODO/OOD objective.

KEY INSIGHT about the FREUID metric: it is *rank/threshold-based* (AuDET = area under the DET
curve; APCER@1%BPCER = APCER at the threshold where BPCER=1%). Any strictly monotonic transform
of a single model's scores leaves the score UNCHANGED. Consequences:
  * Per-model temperature/Platt/isotonic calibration does NOT change a single model's FREUID --
    only the ranking matters. Calibration is only useful to (a) make multi-model blending
    meaningful and (b) satisfy the cosmetic "calibrated probabilities in [0,1]" request.
  * For ensembling, prefer RANK-averaging (calibration-free) or per-model rank-normalization
    before geometric mean -- so no single model's scale dominates.

This module: per-held-out-type FREUID breakdown (the private-LB proxy), the clean-vs-recapture
gap, score-separation diagnostics, and rank/geomean ensembling of several OOF files with
LODO-based (not greedy) member assessment.

Usage:
  python src/evaluate.py checkpoints/cnxb512_oof.csv
  python src/evaluate.py checkpoints/cnxb512_oof.csv checkpoints/forensic_oof.csv --method rank
"""
import sys, argparse
import numpy as np
import pandas as pd
from metrics import calculate_freuid_score


def per_fold_report(oof, name="model"):
    """oof: DataFrame[id,label,pred,fold]. fold == held-out document type for LODO."""
    rows = []
    fr, au, ap = calculate_freuid_score(oof["label"].values, oof["pred"].values)
    rows.append(("POOLED", len(oof), fr, au, ap))
    for f, g in oof.groupby("fold"):
        if g["label"].nunique() < 2:
            rows.append((f, len(g), float("nan"), float("nan"), float("nan")))
            continue
        ffr, fau, fap = calculate_freuid_score(g["label"].values, g["pred"].values)
        rows.append((f, len(g), ffr, fau, fap))
    print(f"\n=== {name}: FREUID by held-out type (lower=better) ===")
    print(f"{'fold/type':18s} {'n':>7s} {'FREUID':>8s} {'AuDET':>8s} {'APCER@1%':>9s}")
    for f, n, fr, au, ap in rows:
        print(f"{str(f):18s} {n:7d} {fr:8.4f} {au:8.4f} {ap:9.4f}")
    folds = [r for r in rows if r[0] != "POOLED" and not np.isnan(r[2])]
    if len(folds) > 1:
        spread = max(r[2] for r in folds) - min(r[2] for r in folds)
        print(f"cross-type spread (max-min FREUID): {spread:.4f}  "
              f"(small = robust to unseen types)")
    return rows


def separation(oof, name="model"):
    p = oof["pred"].values
    y = oof["label"].values
    print(f"[{name}] bona-fide score mean={p[y==0].mean():.3f} | "
          f"fraud score mean={p[y==1].mean():.3f} | "
          f"overlap(bona p>0.5)={np.mean(p[y==0] > 0.5):.3f}")


def rank_norm(x):
    """map to [0,1] by rank -> calibration-free common scale."""
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


def ensemble(oofs, method="rank"):
    """Align several OOF frames on id, combine preds. Returns combined DataFrame[id,label,pred,fold]."""
    base = oofs[0][["id", "label", "fold"]].copy()
    mats = []
    for o in oofs:
        o = o.set_index("id").reindex(base["id"])
        p = o["pred"].values
        mats.append(rank_norm(p) if method == "rank" else np.clip(p, 1e-6, 1 - 1e-6))
    M = np.vstack(mats)
    if method == "rank":
        base["pred"] = M.mean(axis=0)
    else:  # geometric mean
        base["pred"] = np.exp(np.mean(np.log(M), axis=0))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof", nargs="+", help="one or more OOF csvs from train.py")
    ap.add_argument("--method", default="rank", choices=["rank", "geomean"])
    a = ap.parse_args()

    oofs = [pd.read_csv(p) for p in a.oof]
    for path, o in zip(a.oof, oofs):
        per_fold_report(o, name=path)
        separation(o, name=path)

    if len(oofs) > 1:
        ens = ensemble(oofs, method=a.method)
        per_fold_report(ens, name=f"ENSEMBLE({a.method}, {len(oofs)} models)")
        separation(ens, name="ENSEMBLE")


if __name__ == "__main__":
    main()
