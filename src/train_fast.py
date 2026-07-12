"""
Fast training pipeline for FREUID — optimised for a 5-hour budget on a single RTX 4070.

Key differences from train.py:
  * Trains on ALL data (no LODO hold-out) → no document type wasted.
  * Tiny random stratified validation split (5%) for loss/metric monitoring only.
  * Supports external data (--extra_csv) in training.
  * Supports multiple backbone configs in sequence (--configs).
  * Warmup + CosineAnnealing scheduler for stable fine-tuning.
  * Mixed-precision + channels-last + gradient accumulation.
  * Saves best checkpoint by validation FREUID (recapture-augmented).

Usage:
  # ConvNeXtV2-Base, 384px, 5 epochs, with external data
  python src/train_fast.py --backbone convnextv2_base.fcmae_ft_in22k_in1k_384 --img_size 384 --epochs 5 --tag cnxb384 --extra_csv external_train.csv
  
  # DINOv2 ViT-B, 384px, 5 epochs, with grad checkpointing
  python src/train_fast.py --backbone vit_base_patch14_reg4_dinov2.lvd142m --img_size 384 --epochs 5 --tag dinov2b --grad_ckpt --extra_csv external_train.csv
  
  # Forensic noise, 384px, 4 epochs (small backbone, fast)
  python src/train_fast.py --backbone forensic_noise:convnextv2_nano.fcmae_ft_in22k_in1k --img_size 384 --epochs 4 --tag fnoise --batch_size 16
"""
import os, argparse, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from cv import load_train_df
from dataset import FREUIDDataset, get_transforms, get_recapture
from models import create_model
from metrics import calculate_freuid_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def freuid_or_nan(labels, preds):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds, dtype=float)
    if len(labels) == 0 or np.unique(labels).size < 2:
        return float("nan"), float("nan"), float("nan")
    return calculate_freuid_score(labels, preds)


def fmt_score(name, vals):
    fr, au, ap = vals
    if not np.isfinite(fr):
        return f"{name}=n/a"
    return f"{name} freuid={fr:.4f} (audet={au:.4f} apcer@1%={ap:.4f})"


def stratified_cap(df, max_rows, seed):
    if not max_rows or max_rows <= 0 or len(df) <= max_rows:
        return df
    if "label" not in df.columns or df["label"].nunique() < 2:
        return df.sample(max_rows, random_state=seed).reset_index(drop=True)
    frac = max_rows / len(df)
    sampled = []
    for _, group in df.groupby("label"):
        n = max(1, int(round(len(group) * frac)))
        sampled.append(group.sample(min(len(group), n), random_state=seed))
    out = pd.concat(sampled, ignore_index=True).sample(frac=1.0, random_state=seed)
    return out.head(max_rows).reset_index(drop=True)


def load_eval_csv(path, name, max_rows, seed):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"{name}: missing {path}; skipping")
        return None
    df = pd.read_csv(path)
    assert {"abs_path", "label"}.issubset(df.columns), f"{name} needs abs_path,label"
    before = len(df)
    df = df[df["abs_path"].apply(os.path.exists)].reset_index(drop=True)
    df = stratified_cap(df, max_rows=max_rows, seed=seed)
    counts = df["label"].value_counts().to_dict() if len(df) else {}
    print(f"{name}: {len(df)}/{before} existing rows after cap | labels={counts}")
    return df


def run_epoch(model, loader, criterion, optimizer, scaler, device, accum=1, train=True, max_batches=0):
    model.train(train)
    losses, preds, labels = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        if train:
            optimizer.zero_grad(set_to_none=True)
        steps = 0
        for i, batch in enumerate(tqdm(
            loader,
            desc="train" if train else "eval",
            leave=False,
            mininterval=30,
            maxinterval=60,
            miniters=200,
            dynamic_ncols=False,
        )):
            steps += 1
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
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
            if max_batches and steps >= max_batches:
                break
        # flush remaining gradients
        if train and steps and steps % accum != 0:
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
    if not losses:
        return float("nan"), np.array([]), np.array([])
    return np.mean(losses), np.concatenate(preds), np.concatenate(labels)


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | backbone: {args.backbone} | img_size: {args.img_size} | tag: {args.tag}")
    torch.backends.cudnn.benchmark = True
    if args.limit_train_batches or args.limit_eval_batches:
        print(f"SMOKE LIMITS: train_batches={args.limit_train_batches or 'all'} "
              f"eval_batches={args.limit_eval_batches or 'all'}")
    
    # Load training data
    df = load_train_df(args.data_dir)
    print(f"FREUID training data: {len(df)} rows, {df['label'].value_counts().to_dict()}")
    real_recap_df = df[df["is_digital"] == 0].reset_index(drop=True)
    split_df = df
    if args.exclude_real_recapture_from_train and len(real_recap_df):
        split_df = df[df["is_digital"] != 0].reset_index(drop=True)
        print(f"Holding out {len(real_recap_df)} real recaptured FREUID rows from train/val stress eval")
    elif len(real_recap_df):
        print(f"Real recaptured FREUID rows remain in train/val split: {len(real_recap_df)}")
    
    # Stratified random split: 95% train, 5% val (just for monitoring)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=42)
    tr_idx, va_idx = next(sss.split(split_df, split_df["label"]))
    train_df = split_df.iloc[tr_idx].reset_index(drop=True)
    val_df = split_df.iloc[va_idx].reset_index(drop=True)
    
    # Add external data to training only
    if args.extra_csv and os.path.exists(args.extra_csv):
        extra_df = pd.read_csv(args.extra_csv)
        assert {"abs_path", "label"}.issubset(extra_df.columns), "extra_csv needs abs_path,label"
        # Filter to images that actually exist
        extra_df = extra_df[extra_df["abs_path"].apply(os.path.exists)].reset_index(drop=True)
        print(f"External data: {len(extra_df)} rows ({(extra_df.label==1).sum()} fraud / {(extra_df.label==0).sum()} bona)")
        train_df = pd.concat([train_df, extra_df], ignore_index=True)
    
    print(f"Final train: {len(train_df)} rows | Val: {len(val_df)} rows")
    print(f"Val types: {val_df['type'].value_counts().to_dict()}")
    print(f"Val labels: {val_df['label'].value_counts().to_dict()}")
    if len(real_recap_df):
        print(f"Real-recap stress labels: {real_recap_df['label'].value_counts().to_dict()}")
    
    # Build dataloaders
    tr_tf, va_tf = get_transforms(args.img_size)
    recap = get_recapture(args.recapture_strength)
    train_ds = FREUIDDataset(train_df, tr_tf, recapture=recap, p_recapture=args.p_recapture)
    val_clean = FREUIDDataset(val_df, va_tf)
    val_recap = FREUIDDataset(val_df, va_tf,
                              recapture=get_recapture(args.recapture_strength, force_macro=True),
                              p_recapture=1.0)
    
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    def dl(ds, sh, bs=args.batch_size):
        # Keep workers persistent only for the large training loader.  On Windows each worker
        # imports the full CUDA-enabled PyTorch runtime; retaining workers for all five loaders
        # can consume tens of gigabytes of committed memory and fail at the final stress loader
        # with WinError 1455.  Evaluation is a small fraction of the epoch, so zero workers is
        # the safest default and has negligible end-to-end cost.
        workers = args.workers if sh else args.eval_workers
        return DataLoader(
            ds, batch_size=bs, shuffle=sh, num_workers=workers, pin_memory=True, drop_last=sh,
            persistent_workers=sh and workers > 0, worker_init_fn=seed_worker,
            generator=generator)
    train_loader = dl(train_ds, True)
    val_clean_loader = dl(val_clean, False)
    val_recap_loader = dl(val_recap, False)

    ood_df = load_eval_csv(args.ood_csv, "OOD probe", args.ood_max, args.seed)
    eval_bs = args.eval_batch_size or args.batch_size
    ood_loader = dl(FREUIDDataset(ood_df, va_tf), False, eval_bs) if ood_df is not None and len(ood_df) else None
    real_recap_loader = None
    if len(real_recap_df):
        real_recap_loader = dl(FREUIDDataset(real_recap_df, va_tf), False, eval_bs)
    
    # Create model
    model = create_model(args.backbone, pretrained=not args.no_pretrained, head=args.head,
                         drop_rate=args.drop_rate, grad_checkpointing=args.grad_ckpt,
                         img_size=args.img_size)
    model = model.to(device, memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params/1e6:.1f}M")
    
    # Loss, optimizer, scheduler
    pos_weight = None
    if args.pos_weight > 0:
        pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # Warmup + cosine
    warmup_epochs = 1
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    
    os.makedirs(args.save_dir, exist_ok=True)
    best = {"selection": (float("inf"), float("inf")), "epoch": 0}
    history = []
    
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_start = time.time()
        
        # Train
        tl, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device,
                             accum=args.accum, train=True, max_batches=args.limit_train_batches)
        
        # Evaluate on clean val
        _, vp, vy = run_epoch(model, val_clean_loader, criterion, optimizer, scaler, device, train=False,
                              max_batches=args.limit_eval_batches)
        fr, au, ap = freuid_or_nan(vy, vp)
        
        # Evaluate on recapture-augmented val
        _, rp, ry = run_epoch(model, val_recap_loader, criterion, optimizer, scaler, device, train=False,
                              max_batches=args.limit_eval_batches)
        rfr, rau, rap = freuid_or_nan(ry, rp)

        ofr = oau = oap = float("nan")
        if ood_loader is not None:
            _, op, oy = run_epoch(model, ood_loader, criterion, optimizer, scaler, device, train=False,
                                  max_batches=args.limit_eval_batches)
            ofr, oau, oap = freuid_or_nan(oy, op)

        mfr = mau = map_ = float("nan")
        if real_recap_loader is not None:
            _, mp, my = run_epoch(model, real_recap_loader, criterion, optimizer, scaler, device, train=False,
                                  max_batches=args.limit_eval_batches)
            mfr, mau, map_ = freuid_or_nan(my, mp)
        
        scheduler.step()
        ep_time = time.time() - ep_start
        total_time = time.time() - t0

        record = {
            "epoch": epoch + 1,
            "train_loss": tl,
            "clean_freuid": fr, "clean_audet": au, "clean_apcer01": ap,
            "recap_freuid": rfr, "recap_audet": rau, "recap_apcer01": rap,
            "ood_freuid": ofr, "ood_audet": oau, "ood_apcer01": oap,
            "real_recap_freuid": mfr, "real_recap_audet": mau, "real_recap_apcer01": map_,
            "epoch_seconds": ep_time,
            "total_minutes": total_time / 60.0,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        pd.DataFrame(history).to_csv(os.path.join(args.save_dir, f"{args.tag}_history.csv"), index=False)
        
        print(f"[{args.tag}] ep{epoch+1}/{args.epochs} ({ep_time:.0f}s, total {total_time/60:.1f}min) "
              f"train_loss={tl:.4f} | "
              f"{fmt_score('CLEAN', (fr, au, ap))} | "
              f"{fmt_score('RECAP', (rfr, rau, rap))} | "
              f"{fmt_score('OOD', (ofr, oau, oap))} | "
              f"{fmt_score('REAL_RECAP', (mfr, mau, map_))}")
        
        # Select on OOD FREUID when available; use simulated recapture val as tie-break.
        if np.isfinite(ofr):
            selection = (ofr, rfr if np.isfinite(rfr) else float("inf"))
            selection_name = "ood"
        else:
            selection = (rfr if np.isfinite(rfr) else fr, fr if np.isfinite(fr) else float("inf"))
            selection_name = "recap"

        if selection < best["selection"]:
            best = {
                "selection": selection, "selection_name": selection_name,
                "clean_freuid": fr, "recap_freuid": rfr, "ood_freuid": ofr,
                "real_recap_freuid": mfr, "epoch": epoch + 1,
            }
            ckpt = os.path.join(args.save_dir, f"{args.tag}.pth")
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "metric": float(selection[0]),
                "selection_name": selection_name,
                "metrics": record,
            }, ckpt)
            print(f"   >>> BEST saved {ckpt} ({selection_name}_freuid={selection[0]:.4f})")
    
    total = time.time() - t0
    print(f"\n=== {args.tag} DONE in {total/60:.1f} min ===")
    print(f"Best: epoch {best['epoch']}, select={best.get('selection_name', 'n/a')}:{best['selection'][0]:.4f}, "
          f"clean_freuid={best.get('clean_freuid', float('nan')):.4f}, "
          f"recap_freuid={best.get('recap_freuid', float('nan')):.4f}, "
          f"ood_freuid={best.get('ood_freuid', float('nan')):.4f}, "
          f"real_recap_freuid={best.get('real_recap_freuid', float('nan')):.4f}")
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="the-freuid-challenge-2026-ijcai-ecai")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--tag", default="fast_cnxb", help="model tag for checkpoint naming")
    p.add_argument("--backbone", default="convnextv2_base.fcmae_ft_in22k_in1k_384")
    p.add_argument("--head", default="linear", choices=["linear", "mlp"])
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--drop_rate", type=float, default=0.3)
    p.add_argument("--pos_weight", type=float, default=0.0)
    p.add_argument("--p_recapture", type=float, default=0.5)
    p.add_argument("--recapture_strength", default="medium", choices=["light", "medium", "heavy"])
    p.add_argument("--val_frac", type=float, default=0.05, help="random val fraction for monitoring")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--eval_workers", type=int, default=0,
                   help="evaluation workers; 0 avoids Windows paging-file exhaustion")
    p.add_argument("--extra_csv", default=None, help="external data CSV (abs_path,label)")
    p.add_argument("--ood_csv", default=None, help="labeled OOD probe CSV (abs_path,label), used for checkpoint selection")
    p.add_argument("--ood_max", type=int, default=0, help="optional row cap for OOD probe eval; 0 = all rows")
    p.add_argument("--eval_batch_size", type=int, default=0, help="optional eval batch size; 0 = batch_size")
    p.add_argument("--limit_train_batches", type=int, default=0, help="smoke/debug only; 0 = full train epoch")
    p.add_argument("--limit_eval_batches", type=int, default=0, help="smoke/debug only; 0 = full eval")
    p.add_argument("--exclude_real_recapture_from_train", action="store_true",
                   help="hold FREUID is_digital=0 rows out of train/val and report them as a stress set")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
