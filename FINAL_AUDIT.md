# FREUID final audit — 2026-07-13

This is the evidence ledger for the reproducibility package. `PASS` means a retained command output,
hash, or artifact supports the claim. `BLOCKED` is not equivalent to pass and must be cleared before
the final Kaggle reply.

## Frozen selection

- **Policy:** `0.75 * average_rank(ConvNeXt) + 0.25 * average_rank(DINOv2)`.
- **TTA:** off. **Per-template normalization:** off.
- **Selection time recorded:** 2026-07-13 05:15 Europe/Paris.
- **Private release state:** UNVERIFIED — the Kaggle token is expired (HTTP 401), so the listing
  could not be re-checked. The last successful check (~04:00) showed only `public_test/...`. Do not
  claim publication was pre-release until this is confirmed. See the leaderboard section.
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

## Publication — PASS (pushed and verified from a clean clone 2026-07-13)

- Frozen source and both selected weights: `4db0ae5fd62c739a9398175c78e32add4f418a22`.
- Technical report (documentation only): `b92015e86bde97620d7bdb69ebbd9c2448efc903`, now `main`.
- Pushed to the public repository `https://github.com/Marc-Dvci/FREUID`.
- A clean public clone plus `git lfs pull` yields both `.pth` files as real objects that hash-match
  `FROZEN_MANIFEST.json`; the image builds from that clone and reproduces the host output exactly.
- The competition image tree, the organizer briefs, and internal working notes are not tracked.

## Leaderboard and pre-release timing — BLOCKED on expired Kaggle credentials

The Kaggle API token in `~/.kaggle/kaggle.json` (issued 2026-06-14) is rejected with HTTP 401 on
every endpoint, including an unauthenticated-looking competition list. It is expired or revoked, so
none of the following could be done and none may be claimed:

1. **Pre-release timing is UNVERIFIED.** The freeze rule requires the weights and source to be
   published before the organizers release the private images. The push above is recorded at
   2026-07-13 ~08:05 Europe/Paris, but the competition file listing could not be re-checked, so it
   is not established that this preceded private release. Do not assert "frozen pre-release" in the
   Kaggle reply until a listing check confirms it. If private images had already appeared, say so
   plainly rather than implying otherwise; the freeze content itself is unchanged either way.
2. The frozen public candidate is **not submitted**; there is no public score or `COMPLETE` status.
3. The submission label/date-time fields in the report and reply template remain `PENDING`.

Both candidate CSVs passed the full local contract check (142,818 rows, columns exactly `id,label`,
IDs unique and in official sample order, all labels finite and in `[0,1]`, 134,997 untouched `0.5`
private placeholders):

| Candidate | Bytes | SHA-256 |
|---|---:|---|
| `submission_frozen_rank_75legacy_25dino_20260713.csv` (frozen) | 5,545,975 | `6a8d8ca4b58856e761e9aae4b65c18de47bf021512eabae2e818c11954529e6d` |
| `submission_dinov2_raw_20260713.csv` (DINO-only) | 5,550,019 | `ce59e5ad5cee1aea2f4a07c3ab612356515d0a4541b9b48cc62270de2d914f62` |

The frozen candidate's hash matches `FROZEN_MANIFEST.json`. To clear this section, regenerate the
token at `https://www.kaggle.com/settings` ("Create New Token"), overwrite `~/.kaggle/kaggle.json`,
then re-run the listing check and submit the frozen candidate.

## Report and final reply — DRAFT / identity fields required

- LaTeX/BibTeX build passes; draft PDF is three pages.
- Report includes method, data/licenses, validation, results, inference, Docker contract, hashes,
  hardware, timing, dependencies, and randomness.
- Replace all `PENDING` fields with exact Kaggle team metadata, frozen SHA, and public result, then
  rebuild and hash the PDF.
- The Kaggle reply template must receive exact team name, usernames, selected submission label and
  time, repository URL, frozen SHA, report URL, captain username, and UTC date.
- Post exactly once to the pinned thread by 2026-07-15 23:59 AoE; do not post a draft or duplicate.
