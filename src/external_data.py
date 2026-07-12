"""
External public datasets -> unified FREUID training schema.

Rules allow public, license-compatible, *cited* external data. This is the highest-leverage
lever for the two gaps the test set targets:
  * digital -> print-and-capture (recapture)  -> recaptured/presentation-attack datasets
  * cross-document generalization              -> many more countries / doc types / scripts

Every loader returns a DataFrame with the FREUID-compatible columns:
    id, abs_path, label (0=bona-fide,1=fraud), source, type, is_recapture
so external rows can be concatenated into TRAINING (never validation -- LODO/stress holdouts
must stay FREUID-only to measure real generalization).

Each loader is defensive: it globs images and infers the label from folder/filename conventions
(overridable), and returns empty if its root is absent -- so the module imports and the registry
prints even before anything is downloaded.

Usage:
    python src/external_data.py --print_registry
    python src/external_data.py --build --idnet_root D:/data/idnet --sidtd_root D:/data/sidtd \
        --recap_root D:/data/recaptured_id --out external_train.csv
"""
import os, glob, re, argparse, hashlib
import pandas as pd

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")

# ---- citable registry (feeds the technical report's "external resources" section) ----
REGISTRY = {
    "IDNet": dict(
        url="https://www.kaggle.com/datasets/chitreshkr/idnet-identity-document-analysis",
        paper="arXiv:2408.01690",
        license="research/non-commercial (verify on dataset page)",
        relevance="cross-document diversity (20 types, 10 US states + 10 EU countries) + 4 fraud "
                  "variants per sample (face morph, portrait swap, text alteration, combo)",
        note="~490 GB total -> subsample. Synthetic, DHS-sponsored, no real PII."),
    "SIDTD": dict(
        url="https://github.com/Oriolrt/SIDTD_Dataset",
        paper="Nature Scientific Data 2024 (s41597-024-04160-9)",
        license="research/non-commercial (verify)",
        relevance="forged ID/travel docs derived from MIDV-2020; common user-level forgeries",
        note="reals = MIDV2020 originals, fakes = altered versions."),
    "MIDV-Holo": dict(
        url="ftp://smartengines.com/midv-holo",
        paper="MIDV-Holo, ICDAR 2023, DOI:10.1007/978-3-031-41682-8_30",
        license="CC BY-SA 2.5 (bundled license.txt); Generated Photos attribution requested",
        relevance="screen-recapture, printed-photo, drawn-hologram, portrait-replacement attacks; "
                  "glare/recapture baselines -> directly targets the print-and-capture gap",
        note="video frames; sample frames per clip."),
    "MIDV-2020": dict(
        url="ftp://smartengines.com / project page",
        paper="MIDV-2020",
        license="research/non-commercial (verify)",
        relevance="phone-captured bona-fide IDs incl. under-represented scripts -> bona-fide "
                  "recapture realism + document diversity",
        note="bona-fide only (label 0); use as genuine-recapture negatives."),
    "Recaptured-ID": dict(
        url="https://www.kaggle.com/datasets/johnmageetud/recaptured-identity-documents",
        paper="-",
        license="verify on dataset page",
        relevance="recaptured identity documents -> presentation-attack / analog-hole realism",
        note="check whether labels are genuine-vs-recaptured or attack-vs-bonafide before mapping."),
    "DocTamper": dict(
        url="https://github.com/qcf-568/DocTamper",
        paper="DocTamper (170k tampered docs)",
        license="research/non-commercial (verify)",
        relevance="large-scale document tampering -> pretrain the forensic/DCT branch (manipulation)",
        note="tampered vs original; mostly for forensic-stream pretraining, not ID-specific."),
}


def _gid(path):
    return hashlib.md5(path.encode("utf-8")).hexdigest()[:16]


def _glob_images(root):
    files = []
    for ext in IMG_EXT:
        files += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return files


# default label heuristics by path substring (lowercased)
FRAUD_HINTS = ("fraud", "fake", "forg", "tamper", "manip", "morph", "swap", "alter",
               "attack", "spoof", "recaptur", "print", "screen", "photo")
BONA_HINTS = ("genuine", "real", "authentic", "bona", "original", "orig", "live", "template")


def _infer_label(path, fraud_hints=FRAUD_HINTS, bona_hints=BONA_HINTS):
    p = path.lower()
    f = any(h in p for h in fraud_hints)
    b = any(h in p for h in bona_hints)
    if f and not b:
        return 1
    if b and not f:
        return 0
    return None  # ambiguous -> caller decides


def load_folder_labeled(root, source, force_label=None, is_recapture=0,
                        fraud_hints=FRAUD_HINTS, bona_hints=BONA_HINTS, doc_type=""):
    """Generic loader: glob images under root, infer label from path (or force_label)."""
    if not root or not os.path.isdir(root):
        return pd.DataFrame(columns=["id", "abs_path", "label", "source", "type", "is_recapture"])
    rows = []
    for f in _glob_images(root):
        lab = force_label if force_label is not None else _infer_label(f, fraud_hints, bona_hints)
        if lab is None:
            continue  # skip ambiguous rather than mislabel
        rows.append(dict(id=_gid(f), abs_path=f, label=int(lab), source=source,
                         type=doc_type, is_recapture=is_recapture))
    df = pd.DataFrame(rows)
    print(f"  {source}: {len(df)} images ({(df.label==1).sum()} fraud / {(df.label==0).sum()} bona)"
          if len(df) else f"  {source}: 0 (root missing or no labelled images)")
    return df


def load_idnet(root):
    # IDNet: authentic vs {morph, portrait-swap, text-alter, combo}. Heuristic path labels.
    return load_folder_labeled(root, "IDNet")


def load_sidtd(root):
    """SIDTD: clips/Images/{reals,fakes}/<country>_id_<n>_frame_<f>.jpg.
    label: reals=0 (bona-fide, real MIDV2020 captures), fakes=1 (forged, different fraud methods
    than FREUID -> good cross-method OOD). country prefix -> doc type. Captured base -> is_recapture=1.
    """
    if not root or not os.path.isdir(root):
        return pd.DataFrame(columns=["id", "abs_path", "label", "source", "type", "is_recapture"])
    rows = []
    for f in _glob_images(root):
        pl = f.replace("\\", "/").lower()
        if "/fakes/" in pl or os.path.basename(os.path.dirname(f)).lower() == "fakes":
            lab = 1
        elif "/reals/" in pl or os.path.basename(os.path.dirname(f)).lower() == "reals":
            lab = 0
        else:
            continue
        country = os.path.basename(f).split("_")[0]
        rows.append(dict(id=_gid(f), abs_path=f, label=lab, source="SIDTD",
                         type=f"SIDTD/{country}", is_recapture=1))
    df = pd.DataFrame(rows)
    print(f"  SIDTD: {len(df)} images ({(df.label==1).sum() if len(df) else 0} fake / "
          f"{(df.label==0).sum() if len(df) else 0} real)")
    return df


def load_midv_holo(root):
    """MIDV-Holo: images/origins/{ID,passport} = bona-fide (real holograms, captured) -> 0;
    images/fraud/{copy_without_holo,photo_holo_copy,photo_replacement,pseudo_holo_copy} = real
    captured presentation attacks -> 1. All real captures (is_recapture=1). Unseen doc types
    (passports/European IDs) + real capture -> strong non-circular OOD probe.
    type = MIDV-Holo/<ID|passport>; attack subtype kept in 'source' suffix for breakdown.
    """
    if not root or not os.path.isdir(root):
        return pd.DataFrame(columns=["id", "abs_path", "label", "source", "type", "is_recapture"])
    rows = []
    for f in _glob_images(root):
        pl = f.replace("\\", "/").lower()
        if "/origins/" in pl:
            lab, sub = 0, "origins"
        elif "/fraud/" in pl:
            lab = 1
            seg = pl.split("/fraud/")[1].split("/")
            sub = seg[0] if seg else "fraud"
        else:
            continue
        doc = "passport" if "passport" in pl else ("ID" if "/id/" in pl else "doc")
        rows.append(dict(id=_gid(f), abs_path=f, label=lab, source=f"MIDV-Holo:{sub}",
                         type=f"MIDV-Holo/{doc}", is_recapture=1))
    df = pd.DataFrame(rows)
    print(f"  MIDV-Holo: {len(df)} images ({(df.label==1).sum() if len(df) else 0} attack / "
          f"{(df.label==0).sum() if len(df) else 0} bona)")
    return df


def load_midv2020(root):
    # bona-fide phone captures only
    return load_folder_labeled(root, "MIDV-2020", force_label=0, is_recapture=1)


def load_recaptured_id(root):
    return load_folder_labeled(root, "Recaptured-ID", is_recapture=1)


def load_doctamper(root):
    return load_folder_labeled(root, "DocTamper")


def combine_external(roots, balance=True, max_per_source=None):
    """Load every provided source and concatenate. roots: dict source->path."""
    loaders = {"IDNet": load_idnet, "SIDTD": load_sidtd, "MIDV-Holo": load_midv_holo,
               "MIDV-2020": load_midv2020, "Recaptured-ID": load_recaptured_id,
               "DocTamper": load_doctamper}
    dfs = []
    for src, root in roots.items():
        if root and src in loaders:
            df = loaders[src](root)
            if max_per_source and len(df) > max_per_source:
                df = df.sample(max_per_source, random_state=0)
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["id", "abs_path", "label", "source", "type", "is_recapture"])
    out = pd.concat(dfs, ignore_index=True).drop_duplicates("abs_path")
    if balance and len(out):
        # sample weights so each source contributes roughly equally (anti-domination)
        w = out.groupby("source")["label"].transform(lambda s: 1.0 / max(len(s), 1))
        out["sample_weight"] = (w / w.sum() * len(out)).clip(lower=1e-3)
    return out


def print_registry():
    for name, m in REGISTRY.items():
        print(f"\n## {name}")
        for k in ("url", "paper", "license", "relevance", "note"):
            print(f"   {k:10s}: {m[k]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--print_registry", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--idnet_root"); p.add_argument("--sidtd_root"); p.add_argument("--midv_holo_root")
    p.add_argument("--midv2020_root"); p.add_argument("--recap_root"); p.add_argument("--doctamper_root")
    p.add_argument("--max_per_source", type=int, default=None)
    p.add_argument("--out", default="external_train.csv")
    a = p.parse_args()

    if a.print_registry or not a.build:
        print_registry()
    if a.build:
        roots = {"IDNet": a.idnet_root, "SIDTD": a.sidtd_root, "MIDV-Holo": a.midv_holo_root,
                 "MIDV-2020": a.midv2020_root, "Recaptured-ID": a.recap_root,
                 "DocTamper": a.doctamper_root}
        df = combine_external(roots, max_per_source=a.max_per_source)
        df.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}: {len(df)} rows from sources "
              f"{sorted(df['source'].unique()) if len(df) else '[]'}")
