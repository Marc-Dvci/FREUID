# FREUID Challenge 2026 — solution

Fraud detection on identity documents for the [FREUID Challenge 2026 (IJCAI-ECAI)](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai).

Licensed under the MIT License (see `LICENSE`).

## Method

An ensemble of three complementary single-logit detectors, rank-averaged, followed by a
**per-template score normalization** step at inference.

| Member | Backbone | Input | Role |
|---|---|---|---|
| `cnxb384` | ConvNeXt-V2 Base (`fcmae_ft_in22k_in1k_384`) | 384 | semantic / structural inconsistencies |
| `dinov2b` | DINOv2 ViT-B/14 reg4 (`lvd142m`) | 392 | cross-domain self-supervised features, unseen templates |
| `fnoise` | SRM + constrained Bayar stem → ConvNeXt-V2 Nano | 384 | template-agnostic noise forensics |

Two design points matter more than the backbones:

**Print-and-capture simulation.** The training set is ~99.97% fully digital while the test set
emphasises print-and-capture and screen recapture. `src/augment_recapture.py` simulates the analog
hole (down/up resample + double JPEG + blur, plus moiré, halftone, glare, vignette, paper grain,
colour cast, sensor noise). It is applied to **both classes**, so "looks recaptured" cannot become a
fraud shortcut.

**Per-template score normalization.** The FREUID score pools every document type into a single
global threshold, and APCER@1%BPCER is measured at a strict 1% BPCER operating point. Raw model
scores drift in scale from one document template to another, which alone is enough to wreck that
operating point: false alarms on one template push the global threshold up and hide attacks on
another. At inference we cluster the test images by template and rank-normalize scores *within*
each cluster before pooling. This is transductive over the test set only and uses no labels.

## Data

* **FREUID train set** (competition, 69,352 images, 5 document types). Not redistributed here.
* **External data** — public, license-compatible, cited. The registry lives in
  `src/external_data.py` (`python src/external_data.py --print_registry`); every source used in the
  final model is listed with its URL and license in the technical report.

Validation never uses external data. Model selection uses a held-out FREUID document type plus a
recapture-augmented split; see `src/cv.py`.

## Reproducing

```bash
pip install -r requirements.txt

# train the ensemble (single RTX 4070, ~6 h)
python run_all.py

# inference -> submission.csv
python src/infer_ensemble.py \
  --ckpts checkpoints/cnxb384_full.pth checkpoints/dinov2b_full.pth checkpoints/fnoise_full.pth \
  --tta --method rank --normalize_per_template --out submission.csv
```

## Docker (organizer no-network sandbox)

```bash
docker build -t freuid-repro:local .

docker run --rm --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local
```

Model weights are baked into the image; nothing is downloaded at runtime. The container reads a flat
directory of images from `/data` (row id = filename stem) and writes `/submissions/submission.csv`
with columns `id,label`, where `label` is a fraud score (higher = more likely fraudulent).

## Hardware

Trained on a single NVIDIA RTX 4070 (12 GB), PyTorch 2.6.0 + CUDA 12.4.
