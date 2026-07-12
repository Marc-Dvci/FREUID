"""
Run all 3 models sequentially and generate ensemble submission.
Designed for ~5 hour total budget on RTX 4070 (12GB VRAM).

Model 1: ConvNeXtV2-Base @ 384px — proven strong backbone (~2h)
Model 2: DINOv2 ViT-B  @ 384px — best cross-domain self-supervised features (~2h)
Model 3: Forensic Noise @ 384px — template-agnostic noise forensics (~40min)

Total: ~4.5-5h
"""
import argparse
import subprocess, sys, time, os

PYTHON = os.path.join(".venv", "Scripts", "python.exe")
COMMON = ["--data_dir", "the-freuid-challenge-2026-ijcai-ecai",
          "--save_dir", "checkpoints",
          "--extra_csv", "external_train.csv",
          "--exclude_real_recapture_from_train",
          "--eval_batch_size", "16",
          "--ood_max", "3000"]

OOD_CSV = "midv_holo_val.csv"
if os.path.exists(OOD_CSV):
    COMMON += ["--ood_csv", OOD_CSV]

MODELS = [
    {
        "tag": "cnxb384_full",
        "backbone": "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "img_size": "384",
        "epochs": "3",
        "batch_size": "12",
        "accum": "2",
        "lr": "2e-4",
        "extra_flags": ["--grad_ckpt", "--p_recapture", "0.35"],
    },
    {
        "tag": "dinov2b_full",
        "backbone": "vit_base_patch14_reg4_dinov2.lvd142m",
        "img_size": "392",
        "epochs": "3",
        # Fits in 6.1 GB without gradient checkpointing -> 30% faster than gc=ON at bs=10.
        # (ConvNeXt-V2-B keeps gc=ON: without it, it needs 12.3 GB and thrashes the 12 GB card.)
        "batch_size": "16",
        "accum": "2",
        "lr": "1e-4",
        "extra_flags": ["--p_recapture", "0.35"],
    },
    {
        "tag": "fnoise_full",
        "backbone": "forensic_noise:convnextv2_nano.fcmae_ft_in22k_in1k",
        "img_size": "384",
        "epochs": "3",
        "batch_size": "16",
        "accum": "2",
        "lr": "3e-4",
        "extra_flags": ["--p_recapture", "0.50"],
    },
]

def run_model(cfg, force=False):
    ckpt_path = os.path.join("checkpoints", f"{cfg['tag']}.pth")
    if os.path.exists(ckpt_path) and not force:
        print(f"\nSKIPPING {cfg['tag']}: checkpoint already exists at {ckpt_path} (use --force to retrain)")
        return True, "SKIPPED"

    cmd = [PYTHON, "src/train_fast.py",
           "--tag", cfg["tag"],
           "--backbone", cfg["backbone"],
           "--img_size", cfg["img_size"],
           "--epochs", cfg["epochs"],
           "--batch_size", cfg["batch_size"],
           "--accum", cfg["accum"],
           "--lr", cfg["lr"],
           "--workers", "8",
           ] + COMMON + cfg["extra_flags"]
    print(f"\n{'='*60}")
    print(f"STARTING: {cfg['tag']} ({cfg['backbone']})")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL (code {result.returncode})"
    print(f"\n{cfg['tag']}: {status} in {elapsed/60:.1f} min")
    return result.returncode == 0, status

def run_inference(include_legacy=False):
    ckpts = []
    for m in MODELS:
        path = os.path.join("checkpoints", f"{m['tag']}.pth")
        if os.path.exists(path):
            ckpts.append(path)
    if not ckpts:
        print("ERROR: No checkpoints found!")
        return
    
    # Keep the default ensemble compact for the organizer's 6-hour hidden-test
    # inference cap. The legacy LODO checkpoint is opt-in because it measured
    # poorly on OOD probes and adds runtime.
    existing = os.path.join("checkpoints", "cnxb512_MAURITIUS-ID.pth")
    if include_legacy and os.path.exists(existing):
        ckpts.append(existing)
    
    cmd = [PYTHON, "src/infer_ensemble.py",
           "--ckpts"] + ckpts + [
           "--tta", "--out", "submission_ensemble.csv",
           "--method", "rank"]
    print(f"\n{'='*60}")
    print(f"ENSEMBLE INFERENCE")
    print(f"Checkpoints: {ckpts}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, check=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="retrain even if a target checkpoint already exists")
    p.add_argument("--only", nargs="*", default=None, help="optional list of model tags to run")
    p.add_argument("--no_infer", action="store_true", help="skip ensemble inference after training")
    p.add_argument("--include_legacy", action="store_true",
                   help="include the old cnxb512 LODO checkpoint in ensemble inference")
    p.add_argument("--continue_on_error", action="store_true", help="continue training later models if one fails")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t_start = time.time()
    results = {}
    selected = [m for m in MODELS if not args.only or m["tag"] in set(args.only)]
    if not selected:
        raise SystemExit(f"No models selected. Known tags: {[m['tag'] for m in MODELS]}")

    for cfg in selected:
        ok, status = run_model(cfg, force=args.force)
        results[cfg["tag"]] = status
        if not ok and not args.continue_on_error:
            print("Stopping after failure. Use --continue_on_error to train remaining models.")
            break
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY:")
    for tag, status in results.items():
        print(f"  {tag}: {status}")
    print(f"Total training time: {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*60}")
    
    # Generate ensemble submission
    if not args.no_infer:
        run_inference(include_legacy=args.include_legacy)
