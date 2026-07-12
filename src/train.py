"""
Train FREUID detectors with LODO cross-validation.

Key choices (see plan):
  * Validation by Leave-One-Document-type-Out (cross-document generalization is the objective).
  * Recapture simulation on TRAIN (p_recapture) to close the digital->physical gap; each epoch
    the held-out fold is scored BOTH clean and with forced recapture -> watch the gap directly.
  * Model selection by the exact FREUID score (clean val), with recapture-val reported alongside.
  * AMP + gradient accumulation + optional grad checkpointing for a 12 GB GPU.
  * Out-of-fold predictions saved for ensemble stacking / calibration.

Examples:
  # fast smoke test on a tiny subset, one held-out type, 1 epoch
  python src/train.py --quick --epochs 1 --backbone convnextv2_nano.fcmae_ft_in22k_in1k
  # real run: hold out MAURITIUS/ID, 512px ConvNeXtV2-base
  python src/train.py --holdout MAURITIUS/ID --img_size 512 --epochs 6
  # full 5-fold LODO
  python src/train.py --lodo --img_size 512 --epochs 6
"""
import os, argparse, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cv import load_train_df, lodo_folds
from dataset import FREUIDDataset, get_transforms, get_recapture
from models import create_model
from metrics import calculate_freuid_score


def run_epoch(model, loader, criterion, optimizer, scaler, device, accum=1, train=True):
    model.train(train)
    losses, preds, labels = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        if train:
            optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tqdm(loader, desc="train" if train else "eval", leave=False)):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logit = model(x)
                loss = criterion(logit, y)
            if train:
                scaler.scale(loss / accum).backward()
                if (i + 1) % accum == 0:
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            preds.append(torch.sigmoid(logit.detach().float()).cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    return np.mean(losses), np.concatenate(preds), np.concatenate(labels)


def make_loaders(train_df, val_df, args):
    tr_tf, va_tf = get_transforms(args.img_size)
    recap = get_recapture(args.recapture_strength)
    train_ds = FREUIDDataset(train_df, tr_tf, recapture=recap, p_recapture=args.p_recapture)
    val_clean = FREUIDDataset(val_df, va_tf)
    val_recap = FREUIDDataset(val_df, va_tf, recapture=get_recapture(args.recapture_strength, force_macro=True),
                              p_recapture=1.0)
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.workers, pin_memory=True, drop_last=sh,
                                   persistent_workers=args.workers > 0)
    return dl(train_ds, True), dl(val_clean, False), dl(val_recap, False)


def train_fold(fold_name, train_df, val_df, args, device):
    train_loader, val_clean, val_recap = make_loaders(train_df, val_df, args)
    model = create_model(args.backbone, pretrained=not args.no_pretrained, head=args.head,
                         drop_rate=args.drop_rate, grad_checkpointing=args.grad_ckpt,
                         img_size=args.img_size)
    model = model.to(device, memory_format=torch.channels_last)

    pos_weight = None
    if args.pos_weight > 0:
        pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best = {"freuid": 1e9}
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        tl, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device,
                             accum=args.accum, train=True)
        _, vp, vy = run_epoch(model, val_clean, criterion, optimizer, scaler, device, train=False)
        fr, au, ap = calculate_freuid_score(vy, vp)
        # recapture stress eval (cheaper: subsample if large)
        _, rp, ry = run_epoch(model, val_recap, criterion, optimizer, scaler, device, train=False)
        rfr, rau, rap = calculate_freuid_score(ry, rp)
        scheduler.step()
        print(f"[{fold_name}] ep{epoch+1}/{args.epochs} train_loss={tl:.4f} | "
              f"CLEAN freuid={fr:.4f} (audet={au:.4f} apcer@1%={ap:.4f}) | "
              f"RECAP freuid={rfr:.4f} (audet={rau:.4f} apcer@1%={rap:.4f})")

        # select on the harder (recapture) score -- that is what the test set looks like
        sel = rfr if args.select_on == "recap" else fr
        if sel < best["freuid"]:
            best = {"freuid": sel, "clean_freuid": fr, "recap_freuid": rfr, "epoch": epoch + 1,
                    "oof_preds": vp, "oof_labels": vy, "val_ids": val_df["id"].to_numpy()}
            ckpt = os.path.join(args.save_dir, f"{args.tag}_{fold_name.replace('/','-')}.pth")
            torch.save({"model": model.state_dict(), "args": vars(args), "metric": best["freuid"]}, ckpt)
            print(f"   saved {ckpt} (select={sel:.4f})")
    return best


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| backbone:", args.backbone, "| img_size:", args.img_size)
    torch.backends.cudnn.benchmark = True
    df = load_train_df(args.data_dir)

    if args.quick:
        df = df.groupby("type", group_keys=False).apply(lambda g: g.sample(min(len(g), 200), random_state=0))
        df = df.reset_index(drop=True)
        args.epochs = max(1, args.epochs)
        print("QUICK mode: subsampled to", len(df), "rows")

    # external data is folded into TRAINING ONLY (validation stays FREUID-only so LODO/recapture
    # holdouts keep measuring real generalization). Build it with src/external_data.py.
    extra_df = None
    if args.extra_csv and os.path.exists(args.extra_csv):
        extra_df = pd.read_csv(args.extra_csv)
        assert {"abs_path", "label"}.issubset(extra_df.columns), "extra_csv needs abs_path,label"
        print(f"external training data: {len(extra_df)} rows from {args.extra_csv}")

    folds = list(lodo_folds(df)) if args.lodo else [
        (args.holdout, df.index[df["type"] != args.holdout].to_numpy(),
         df.index[df["type"] == args.holdout].to_numpy())]

    oof = []
    results = {}
    for name, tr_idx, va_idx in folds:
        train_part = df.loc[tr_idx]
        if extra_df is not None:
            train_part = pd.concat([train_part, extra_df], ignore_index=True)
        print(f"\n===== FOLD: hold out {name}  (train={len(train_part)} val={len(va_idx)}) =====")
        best = train_fold(name, train_part, df.loc[va_idx], args, device)
        results[name] = {k: best[k] for k in ("clean_freuid", "recap_freuid", "epoch")}
        oof.append(pd.DataFrame({"id": best["val_ids"], "label": best["oof_labels"],
                                 "pred": best["oof_preds"], "fold": name}))

    if oof:
        oof_df = pd.concat(oof, ignore_index=True)
        oof_path = os.path.join(args.save_dir, f"{args.tag}_oof.csv")
        oof_df.to_csv(oof_path, index=False)
        gfr, gau, gap = calculate_freuid_score(oof_df["label"].values, oof_df["pred"].values)
        print(f"\n=== OOF FREUID (clean, pooled): {gfr:.4f} (audet={gau:.4f} apcer@1%={gap:.4f}) ===")
        print("per-fold:", json.dumps(results, indent=2))
        print("saved OOF ->", oof_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="the-freuid-challenge-2026-ijcai-ecai")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--tag", default="cnx_b")
    p.add_argument("--backbone", default="convnextv2_base.fcmae_ft_in22k_in1k_384")
    p.add_argument("--head", default="linear", choices=["linear", "mlp"])
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--drop_rate", type=float, default=0.2)
    p.add_argument("--pos_weight", type=float, default=0.0)
    p.add_argument("--p_recapture", type=float, default=0.5)
    p.add_argument("--recapture_strength", default="medium", choices=["light", "medium", "heavy"])
    p.add_argument("--select_on", default="recap", choices=["recap", "clean"])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--extra_csv", default=None, help="normalized external data (abs_path,label); train-only")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--lodo", action="store_true", help="full 5-fold leave-one-type-out")
    mode.add_argument("--holdout", default="MAURITIUS/ID", help="single held-out type for fast iteration")
    p.add_argument("--quick", action="store_true", help="tiny subset smoke test")
    main(p.parse_args())
