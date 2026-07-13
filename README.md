# FREUID Challenge 2026 — frozen reproducibility package

Identity-document fraud detector for the [FREUID Challenge 2026
(IJCAI–ECAI)](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai).
Repository source is MIT-licensed. Dataset and pretrained-weight licenses are separate and documented
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Frozen method

The organizer-facing output is a pre-private-release, no-TTA weighted rank ensemble:

| Weight | Checkpoint | Backbone | Input | Parameters |
|---:|---|---|---:|---:|
| 0.75 | `cnxb512_MAURITIUS-ID.pth` | ConvNeXt-V2 Base FCMAE | 512 | 87,693,825 |
| 0.25 | `dinov2b_full.pth` | DINOv2 ViT-B/14 reg4 | 392 | 86,133,505 |

Each model emits a sigmoid fraud score. Scores are converted to average ranks over the mounted test
set (ties retain equal average rank), then combined as `0.75 * ConvNeXt + 0.25 * DINOv2`. Higher
means more likely fraudulent. The weights, checkpoints, preprocessing, and no-TTA policy are frozen.

The weighting is deliberately conservative. ConvNeXt is much stronger on the 20-row official
physical-recapture stress slice; DINOv2 provides limited diversity on unseen document layouts. The
one-shot run also produced a 384-pixel ConvNeXt and a forensic-noise model, but both became
near-constant on the disjoint unseen-attack probe and are excluded from inference.

Print/capture simulation (resampling, repeated JPEG, blur, moiré, halftone, glare, vignetting, paper
grain, colour cast, and sensor noise) is applied independently to both classes during training, so
capture appearance cannot become a fraud shortcut.

## Data and validation

- Official FREUID train: 69,352 images across five document types. Not redistributed.
- Selected DINOv2 external train: 15,179 MIDV-Holo frames (CC BY-SA 2.5), 14 documents, two attack
  families. External images are not redistributed.
- OOD probe: 6,428 MIDV-Holo frames from six disjoint documents and two unseen attack families, with
  zero image/clip/document overlap with external training.
- The 20 official real-recapture rows are excluded from fitting and used only as a stress slice.

Per-template score normalization was tested and rejected: it worsened the public score from 0.33816
to 0.54141 (lower is better). Horizontal-flip TTA was also rejected. These negative results remain
documented so they are not silently reintroduced.

## Obtain the frozen weights

The two selected `.pth` files are Git LFS objects, not runtime downloads:

```bash
git lfs install
git lfs pull
```

Verify them against [FROZEN_MANIFEST.json](FROZEN_MANIFEST.json) before building. A clone missing LFS
objects must not be used: Docker `COPY` should receive the actual 344–351 MB files, not pointer text.

## Organizer Docker

The image has an immutable PyTorch/CUDA base digest, fully pinned Python packages, and both weights
baked into `/models`. The default entrypoint requires CUDA, scans supported files directly under the
flat `/data` mount, and writes only `/submissions/submission.csv`.

```bash
docker build --pull -t freuid-repro:local .

mkdir -p out
docker run --rm --gpus all --network none --read-only \
  -v /absolute/path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local
```

Contract:

- input: `.jpeg`, `.jpg`, `.png`, `.webp`, `.bmp`, `.tif`, or `.tiff`, case-insensitive;
- ID: exact filename stem; duplicate stems fail nonzero;
- output: exactly `id,label`, one row per mounted image, finite labels in `[0,1]`;
- no network, runtime download, CSV input, subdirectory scan, or write outside `/submissions`.

The host entrypoint and repository inference path are parity-tested. The final audit records the
actual container run separately; do not claim container verification unless its Docker checks are
marked complete in [FINAL_AUDIT.md](FINAL_AUDIT.md).

## Training and local inference

Create an environment and install the research dependencies:

```bash
pip install -r requirements.txt
```

The pre-freeze one-shot training command was:

```bash
python run_all.py --force --no_infer --continue_on_error
```

It ran all three new candidates sequentially on one RTX 4070 for 292.8 minutes. The frozen DINOv2
checkpoint is epoch 2. The legacy ConvNeXt checkpoint's complete arguments are embedded in the
checkpoint and its training implementation remains in `src/train.py`.

Reproduce frozen public-directory inference without Docker:

```bash
python src/infer_ensemble.py \
  --ckpts checkpoints/cnxb512_MAURITIUS-ID.pth checkpoints/dinov2b_full.pth \
  --weights 0.75 0.25 --method rank --batch_size 16 --workers 0 \
  --out submission_frozen.csv
```

Do not add `--tta` or `--normalize_per_template`; neither is part of the frozen submission.

## Final private-row assembly

After release, run the unchanged container on the private images alone. Merge those rows into the
frozen full public-base CSV:

```bash
python scripts/assemble_final_submission.py \
  --base submission_frozen_rank_75legacy_25dino_20260713.csv \
  --private out/submission.csv \
  --sample the-freuid-challenge-2026-ijcai-ecai/sample_submission.csv \
  --public-image-dir the-freuid-challenge-2026-ijcai-ecai/public_test/public_test \
  --out submission_final.csv
```

The assembler refuses partial private output. It requires the private IDs to equal exactly the
official sample IDs minus the public-image stems, verifies that their base values remain untouched
`0.5` placeholders, preserves official order, and validates range/finiteness before emitting a hash.

## Hardware and timing

Training: NVIDIA RTX 4070 12 GB, PyTorch 2.6.0, CUDA 12.4, seed 42. Two-model public inference took
about 274 seconds for 7,821 images at conservative batch sizes; linear RTX 4070 extrapolation is
about 79 minutes for 134,997 private images, well below the six-hour A100 limit. The actual Docker
measurement, image digest, final CSV hash, and frozen commit are recorded in `FINAL_AUDIT.md`.
