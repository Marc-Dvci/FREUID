# FREUID final audit — 2026-07-13

This is the evidence ledger for the reproducibility package. `PASS` means a retained command output,
hash, or artifact supports the claim. `BLOCKED` is not equivalent to pass and must be cleared before
the final Kaggle reply.

## Frozen selection

- **Policy:** `0.75 * average_rank(ConvNeXt) + 0.25 * average_rank(DINOv2)`.
- **TTA:** off. **Per-template normalization:** off.
- **Selection time recorded:** 2026-07-13 05:15 Europe/Paris.
- **Private release state:** not yet released. The API listing could not be re-checked (token
  expired, HTTP 401); the entrant confirmed directly on Kaggle on 2026-07-13 that the private images
  were still unreleased at the time of the freeze push. The last successful API check (~04:00) also
  showed only `public_test/...`.
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
- full-test runtime measured at 131 minutes for all 134,997 private images on RTX 4070 (see below).

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

## Final private run — PASS (2026-07-13)

The organizers released the private images on 2026-07-13, after the source freeze. The frozen image
was run unchanged; no weight, architecture, checkpoint, training-code, or hyperparameter change was
made after release.

- Input: the 134,997 `private_test` images, extracted flat from the official archive. Their filename
  stems equal exactly `sample_submission` IDs minus the official public-image stems — 0 missing, 0
  extra, no duplicate stems. The archive's `sample_submission.csv` is byte-identical to the audited
  copy (`c5350036…a879a8ab`), so the official ID set and row order did not change.
- Command: `docker run --rm --gpus all --network none --read-only -v <private>:/data:ro
  -v <out>:/submissions freuid-repro:local`, image
  `sha256:4253468b407d569cec1799ccb22eeaf7596d5fde22f43bba993d241b85518d62`, defaults (batch 64,
  zero workers). Exit 0.
- **Measured wall clock: 7,871 s = 131 minutes** on one RTX 4070 (2026-07-13 16:17→18:28 UTC). This
  supersedes the earlier 79-minute linear extrapolation, which was optimistic; the run is bound by
  single-process image decoding, not the GPU. Still far below the six-hour A100 cap.
- Output: 134,997 rows, one per image, `id,label`, all finite and in `[0,1]`.

Merged with `scripts/assemble_final_submission.py` into the frozen public base. Independent
re-validation of the merged file (not reusing the assembler's own checks):

| Check | Result |
|---|---|
| Columns exactly `id,label`; 142,818 rows; IDs unique | PASS |
| Row order equals the official sample submission | PASS |
| All labels finite and in `[0,1]` | PASS |
| 7,821 public rows bit-identical to the frozen public base | PASS |
| 134,997 private rows bit-identical to the container output | PASS |
| Rows still holding the `0.5` placeholder | 0 |

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `submission_final.csv` | 7,551,061 | `239e31c498c7f3c9c2ceee47cc2ff36e47b5124abfdf9b6109611932eea63b38` |

Both halves have mean exactly 0.5000, the expected signature of per-set rank normalization. Public
and private ranks are normalized within their own mounted set, which is the frozen design: each
leaderboard split is scored only against its own rows, so the metric is unaffected.

## Leaderboard — BLOCKED on Kaggle credentials

Two Kaggle API tokens were tried for user `marcdonovici` (the 2026-06-14 one, and a replacement
issued 2026-07-13). Both are rejected with `{"code":401,"message":"Unauthenticated"}` on every
endpoint, including a plain competition list, and the second was confirmed by a direct HTTPS basic-
auth call that bypasses the CLI entirely. No `KAGGLE_USERNAME`/`KAGGLE_KEY` environment override is
set and the credentials file is well-formed, so this is the credential itself, not the tooling.
Consequently:

1. **Pre-release timing: confirmed by the entrant, not by API.** The freeze push is recorded at
   2026-07-13 ~08:05 Europe/Paris. The entrant confirmed directly on Kaggle that the private images
   were not yet released at that time, so the publication is pre-release. The API listing check
   could not corroborate it independently because the token is rejected.
2. `submission_final.csv` was uploaded through the Kaggle web UI by the entrant on 2026-07-13,
   labelled `submission_final.csv - 8:39 PM`. This is recorded on the entrant's report, not from an
   authenticated API check: the CLI could not confirm the `COMPLETE` status or read back any score.
   No leaderboard score is claimed anywhere in this package.
3. Report and reply now carry the submission label; no `PENDING` fields remain.

The upload is therefore a manual step for the entrant: upload `submission_final.csv`
(`239e31c4…a63b38`) through the Kaggle web UI, confirm the submission reaches `COMPLETE`, and copy
its label and date-time verbatim into `REPLY_TEMPLATE.txt` before posting the single allowed reply.

To restore API access, download `kaggle.json` from `https://www.kaggle.com/settings` ("Create New
Token") and place that file at `~/.kaggle/kaggle.json` unedited, rather than transcribing the key.

## Report and final reply — DRAFT / identity fields required

- LaTeX/BibTeX build passes; draft PDF is three pages.
- Report includes method, data/licenses, validation, results, inference, Docker contract, hashes,
  hardware, timing, dependencies, and randomness.
- Replace all `PENDING` fields with exact Kaggle team metadata, frozen SHA, and public result, then
  rebuild and hash the PDF.
- The Kaggle reply template must receive exact team name, usernames, selected submission label and
  time, repository URL, frozen SHA, report URL, captain username, and UTC date.
- Post exactly once to the pinned thread by 2026-07-15 23:59 AoE; do not post a draft or duplicate.
