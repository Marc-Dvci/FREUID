# FREUID Challenge 2026 — solution

Fraud detection on identity documents for the [FREUID Challenge 2026 (IJCAI-ECAI)](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai).

Licensed under the MIT License (see `LICENSE`).

## Method

An ensemble of three complementary single-logit detectors, rank-averaged.

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

**The operating point is what the metric is really about.** APCER is read at the threshold where
just 1% of bona-fides are rejected, so a heavy bona-fide false-positive tail is fatal: a handful of
genuine documents scored near 1.0 push that threshold to ~0.99, and every attack scoring below it
counts as missed. A single ConvNeXt baseline scores AuDET ≈ 0 yet APCER@1%BPCER ≈ 0.5 for exactly
this reason. Rank-averaging diverse members is the direct remedy — a false positive from one member
is demoted by the others — which is why the ensemble is the method rather than a nicety.

*Negative result, recorded so it isn't retried:* we tested **per-template score normalization**
(cluster the test set by document template, rank scores within each cluster, pool). The FREUID score
is purely rank-based, and per-template score-scale drift is real and measurable, so this looked
compelling. It is worse: public leaderboard 0.338 → 0.541. Rank-mapping cannot demote a confidently
mis-scored bona-fide out of the top of its own cluster, and equalizing templates implicitly assumes
equal fraud prevalence across them, which does not hold. The code remains in `src/template_norm.py`
behind `--normalize_per_template` (off by default).

## Data

* **FREUID train set** (competition, 69,352 images, 5 document types). Not redistributed here.
* **External data** — 15,179 MIDV-Holo frames under CC BY-SA 2.5, cited in the report. The registry lives in
  `src/external_data.py` (`python src/external_data.py --print_registry`); every source used in the
  final model is listed with its URL and license in the technical report.

The MIDV-Holo training and probe partitions share no image, video clip, or document identity, and
use disjoint attack families. Checkpoint selection uses the 6-document/2-unseen-attack probe with
forced-recapture FREUID validation as a tie-break. The 20 official real-recapture rows are excluded
from fitting and reported only as a small stress slice.

## Reproducing

```bash
git lfs install
git lfs pull

pip install -r requirements.txt

# train the ensemble (single RTX 4070, ~6 h)
python run_all.py

# inference -> submission.csv
python src/infer_ensemble.py \
  --ckpts checkpoints/cnxb384_full.pth checkpoints/dinov2b_full.pth checkpoints/fnoise_full.pth \
  --tta --method rank --out submission.csv
```

## Docker (organizer no-network sandbox)

```bash
docker build -t freuid-repro:local .

docker run --rm --gpus all --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local
```

Model weights are baked into the image; nothing is downloaded at runtime. The container reads a flat
directory of images from `/data` (row id = filename stem) and writes `/submissions/submission.csv`
with columns `id,label`, where `label` is a fraud score (higher = more likely fraudulent).

For the final Kaggle CSV, run the container on the private images alone and merge its output by id
with the frozen public-row CSV using `python scripts/assemble_final_submission.py`. This preserves
exact equality between organizer-side private-only reproduction and the selected Kaggle private rows.

## Hardware

Trained on a single NVIDIA RTX 4070 (12 GB), PyTorch 2.6.0 + CUDA 12.4.
