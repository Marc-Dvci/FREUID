# FREUID final audit — 2026-07-13

This is the evidence ledger for the reproducibility package. `PASS` means a retained command output,
hash, or artifact supports the claim. `BLOCKED` is not equivalent to pass and must be cleared before
the final Kaggle reply.

## Frozen selection

- **Policy:** `0.75 * average_rank(ConvNeXt) + 0.25 * average_rank(DINOv2)`.
- **TTA:** off. **Per-template normalization:** off.
- **Selection time recorded:** 2026-07-13 05:15 Europe/Paris.
- **Private release state:** last authenticated competition-file check at approximately 04:00 showed
  only `public_test/...`; another authenticated check is required before claiming publication was
  pre-release.
- **Manifest:** `FROZEN_MANIFEST.json`.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `checkpoints/cnxb512_MAURITIUS-ID.pth` | 350,933,114 | `aebf36acdf23dfe7a1542e9b07ba94e782910b547d6244f3fbb8d84083066510` |
| `checkpoints/dinov2b_full.pth` | 344,607,129 | `f333c566bef9d6c2e9be8ba4a5b5efb5e708492e71162a135856a5e08ee1e8ca` |
| local frozen public-base CSV | 5,545,975 | `6a8d8ca4b58856e761e9aae4b65c18de47bf021512eabae2e818c11954529e6d` |

The public base has 142,818 official rows: 7,821 scored public IDs and 134,997 untouched `0.5`
private placeholders. Columns are exactly `id,label`; IDs are unique and in official order; every
label is finite and in `[0,1]`.

## Training outcome — PASS

- One-shot command completed for all new candidates in 292.8 minutes on RTX 4070 12 GB.
- DINOv2 selected checkpoint: epoch 2, OOD-selection FREUID 0.974844 on the fixed stratified
  3,000-row cap; SHA above.
- New ConvNeXt-384 and forensic-noise checkpoints were near-constant on the unseen probe and are not
  copied into Docker or Git LFS.
- The first run's `WinError 1455` was traced to five persistent multi-worker loaders. Evaluation
  workers now default to zero; the exact five-loader smoke passed before restart.

Full 6,428-row unseen MIDV-Holo probe (lower is better):

| Candidate | FREUID | AuDET | APCER@1%BPCER |
|---|---:|---:|---:|
| DINOv2 raw | 0.977524 | 0.500522 | 0.988504 |
| legacy ConvNeXt raw | 0.982878 | 0.545075 | 0.991275 |
| frozen 0.75/0.25 rank | 0.986719 | 0.541910 | 0.993262 |

Official 20-row real-recapture stress slice:

| Candidate | FREUID | AuDET | APCER@1%BPCER |
|---|---:|---:|---:|
| legacy ConvNeXt raw | 0.105838 | 0.065476 | 0.142857 |
| DINOv2 raw | 0.675703 | 0.404762 | 0.777143 |
| frozen 0.75/0.25 rank | 0.148901 | 0.083333 | 0.205714 |

TTA worsened DINOv2 OOD to 0.993706 and is disabled. Per-template normalization worsened the public
leaderboard from 0.33816 to 0.54141 and is disabled. The ordinal double-`argsort` evaluator was found
to assign fake row-order signal to collapsed tied predictions; it was replaced with average ranks,
and the false ensemble result was discarded.

## Inference and CSV contract — PASS on host

- Real CUDA inference through `docker/prepare_submission.py` passed on four official images.
- Output had one row per image, exact filename-stem IDs, columns `id,label`, finite range-valid scores.
- Repository inference and organizer entrypoint produced numerically identical rows on the same set:
  `[0.25, 0.25, 0.8333333333333334, 0.6666666666666666]` after ID sorting.
- Empty directory, absent directory, and duplicate filename stems each failed nonzero with a clear
  exception.
- Local and Docker rank functions give equal average ranks for ties and identical weighted output.
- Python compilation passed for `src`, `docker`, `scripts`, and `run_all.py`.
- Public candidate schema/order/range validation passed; frozen-base SHA is above.

## Private assembly — PASS with synthetic full-ID rehearsal

- A 134,997-row synthetic private output whose IDs were exactly `sample - public` merged
  successfully into the 142,818-row base.
- Removing one private ID failed with `missing=1, extra=0`.
- Reloaded public and private numeric values were unchanged after CSV serialization.
- The assembler checks official order, duplicate/missing/unknown IDs, untouched private placeholders,
  finiteness, range, and exact private-set equality.

## Docker image — PASS (built and executed 2026-07-13)

Static/host checks completed:

- immutable base digest is pinned;
- Python dependency closure is version-pinned;
- only the two selected LFS weights enter the build context;
- entrypoint uses `pretrained=False`, disables CUDA cache, and has no HTTP/runtime download path;
- output path is restricted to exactly `/submissions/submission.csv`;
- local throughput extrapolates to about 79 minutes for 134,997 images on RTX 4070, below six hours.

Executed checks (`docker build --pull -t freuid-repro:local .`, exit 0):

- Image ID `sha256:4253468b407d569cec1799ccb22eeaf7596d5fde22f43bba993d241b85518d62`.
- Both baked weights re-verified inside the image by `sha256sum -c`: `OK`, `OK`.
- The runtime base ships no compiler, so the successful `pip install` proves the entire pinned
  closure resolved to prebuilt wheels with no source build.
- GPU run with `--gpus all --network none --read-only`, `/data:ro`, writable `/submissions`
  succeeded on the four-image official fixture and reproduced the host values exactly:
  `[0.25, 0.25, 0.8333333333333334, 0.6666666666666666]`. Container/host parity is therefore
  measured, not inferred.
- Negative tests inside the built image each exit nonzero with a clear message: empty `/data`
  (`no supported image files found`), duplicate stems (`duplicate filename stems in /data`), and
  missing weights (`missing frozen checkpoint(s)`).
- Egress is genuinely blocked under `--network none` (socket connect raises `OSError`).

Two defects were found only by executing the image; host-only checks had passed over both:

1. `stringzilla==4.6.2` publishes no cp311 manylinux x86_64 wheel, so pip fell back to a source
   build and the build died at `gcc: No such file or directory`. The image could never have been
   built by the organizers as previously pinned. Repinned to `4.6.1`, which ships that wheel and
   satisfies albucore (`stringzilla>=3.10.4`). It is a string-ops accelerator inside albucore and
   does not touch transform numerics; the parity run above confirms unchanged output.
2. Under `--read-only`, `timm -> torchvision -> torch._dynamo` calls `tempfile.gettempdir()` at
   import and aborted with `No usable temporary directory found`. The documented `--read-only`
   command in `README.md` would have failed as written. Fixed in-image with `VOLUME ["/tmp"]`, so
   bare `--read-only` now works with no extra caller flags and no persisted write outside
   `/submissions`.

## Publication and leaderboard — BLOCKED pending user authorization

- Current committed source baseline: `1024265776a0f865a06a49ddb2029e3bab6457b2`.
- Final source/weights/report changes are not yet committed or pushed.
- Git LFS attributes correctly select both final `.pth` files.
- Repository visibility was verified public: `https://github.com/Marc-Dvci/FREUID`.
- DINO-only and frozen 0.75/0.25 CSVs passed local contract checks, but Kaggle upload was rejected by
  the managed environment's external-action approval quota before any upload occurred.

Required before code freeze can be claimed:

1. Recheck competition file listing and record whether private images have appeared.
2. Commit selected LFS weights and all frozen source; record the new 40-character SHA.
3. Push and verify from a clean public clone with `git lfs pull`.
4. Submit frozen public candidate, wait for `COMPLETE`, and record its exact public score/date-time.

## Report and final reply — DRAFT / identity fields required

- LaTeX/BibTeX build passes; draft PDF is three pages.
- Report includes method, data/licenses, validation, results, inference, Docker contract, hashes,
  hardware, timing, dependencies, and randomness.
- Replace all `PENDING` fields with exact Kaggle team metadata, frozen SHA, and public result, then
  rebuild and hash the PDF.
- The Kaggle reply template must receive exact team name, usernames, selected submission label and
  time, repository URL, frozen SHA, report URL, captain username, and UTC date.
- Post exactly once to the pinned thread by 2026-07-15 23:59 AoE; do not post a draft or duplicate.
