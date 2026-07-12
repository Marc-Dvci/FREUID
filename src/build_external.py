"""
Build the external training set and the OOD probe from MIDV-Holo.

The probe must measure what the private test actually asks for: detecting attacks the model has
never seen, on documents it has never seen. A random row-level split of MIDV-Holo does neither --
its frames come from ~20 documents filmed as short videos, so adjacent frames of the same document
land on both sides of the split and the probe reads as near-zero by memorisation.

We therefore split on two disjoint axes at once:

  * document identity -- 14 documents train, 6 held out (id08-id10, psp08-psp10);
  * attack subtype    -- `photo_replacement` and `copy_without_holo` train,
                         `pseudo_holo_copy` and `photo_holo_copy` held out.

Bona-fide `origins` frames appear on both sides but only for their own document identity, so no
image, clip or document is ever shared. The probe is thus: unseen documents x unseen attacks.

Usage:
  python src/build_external.py --midv_holo_root D:/data/external/midv_holo
"""
import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from external_data import load_midv_holo

HOLDOUT_DOCS = {"id08", "id09", "id10", "psp08", "psp09", "psp10"}
HOLDOUT_ATTACKS = {"pseudo_holo_copy", "photo_holo_copy"}

_CLIP_RE = re.compile(r"([a-z]{2,3}\d{2}_\d{2}_\d{2})")


def doc_identity(path):
    """'.../fraud/photo_replacement/ID/id08_04_01/img_0012.jpg' -> 'id08'."""
    m = _CLIP_RE.search(str(path).replace("\\", "/").lower())
    return m.group(1).split("_")[0] if m else None


def attack_subtype(source):
    """'MIDV-Holo:photo_replacement' -> 'photo_replacement'."""
    return str(source).split(":", 1)[1] if ":" in str(source) else str(source)


def main(args):
    df = load_midv_holo(args.midv_holo_root)
    if df.empty:
        raise SystemExit(f"no MIDV-Holo images under {args.midv_holo_root}")

    df["doc"] = df["abs_path"].map(doc_identity)
    df["attack"] = df["source"].map(attack_subtype)
    if df["doc"].isna().any():
        raise SystemExit(f"{int(df['doc'].isna().sum())} rows with unparseable document identity")

    held_doc = df["doc"].isin(HOLDOUT_DOCS)
    held_attack = df["attack"].isin(HOLDOUT_ATTACKS)

    # train: seen documents, seen attacks (bona-fide origins of those same documents included)
    train = df[~held_doc & ~held_attack].reset_index(drop=True)
    # probe: unseen documents, and for the attack class only unseen attack subtypes
    probe = df[held_doc & (held_attack | (df["label"] == 0))].reset_index(drop=True)

    for name, a, b in (("abs_path", set(train.abs_path), set(probe.abs_path)),
                       ("doc", set(train.doc), set(probe.doc))):
        overlap = a & b
        if overlap:
            raise SystemExit(f"LEAK: {len(overlap)} shared {name} between train and probe")

    train.drop(columns=["doc", "attack"]).to_csv(args.out_train, index=False)
    probe.drop(columns=["doc", "attack"]).to_csv(args.out_probe, index=False)

    def summary(name, d):
        print(f"\n{name}: {len(d)} rows | {(d.label == 0).sum()} bona / {(d.label == 1).sum()} attack")
        print(f"  documents: {sorted(d['doc'].unique())}")
        print(f"  attacks  : {sorted(d.loc[d.label == 1, 'attack'].unique())}")

    summary(f"train -> {args.out_train}", train)
    summary(f"probe -> {args.out_probe}", probe)
    print("\nno shared image, clip or document between the two.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--midv_holo_root", default="D:/data/external/midv_holo")
    p.add_argument("--out_train", default="external_train.csv")
    p.add_argument("--out_probe", default="midv_holo_val.csv")
    main(p.parse_args())
