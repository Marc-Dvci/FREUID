# FREUID reproducibility and disqualification-prevention checklist

Do not mark an unchecked item complete without retaining command output or an artifact hash in
`FINAL_AUDIT.md`.

## Code freeze and public package

- [x] Repository is public: `https://github.com/Marc-Dvci/FREUID` (GitHub API checked 2026-07-12).
- [x] Source is under the OSI-approved MIT license.
- [x] Final architecture, training code, and both selected weights are committed and pushed:
  frozen source `4db0ae5fd62c739a9398175c78e32add4f418a22`, report `b92015e8`.
  NOT YET CONFIRMED that this preceded private-image release — see the note below.
- [x] Two selected `.pth` files are present through Git LFS after a clean clone (`git lfs pull`)
  and both hash-match `FROZEN_MANIFEST.json`.
- [x] Frozen 40-character commit SHA is recorded in the report and Kaggle reply.
- [ ] No weight, architecture, checkpoint, training-code, or hyperparameter change is made after
  private-image release. Only inference, CSV merging, documentation, and packaging may change.

## Docker sandbox contract

- [x] Entrypoint discovers flat image files directly in `/data`; it requires no CSV or manifest.
- [x] `id` is exactly the filename stem; all organizer-listed extensions are case-insensitive.
- [x] Duplicate stems, missing weights, empty input, non-finite values, and row mismatches fail nonzero.
- [x] Output path and schema are exactly `/submissions/submission.csv` and `id,label`.
- [x] Scores are finite in `[0,1]`, with higher meaning more likely fraudulent.
- [x] Both selected weights are copied into the image; model creation uses `pretrained=False`.
- [x] Inference makes no HTTP/API calls and does not need a writable model cache.
- [x] Base image is pinned by immutable digest.
- [x] Image builds from a clean clone after `git lfs pull`, and its container output is identical
      to the host run on the official fixture.
- [x] Container succeeds with `--network none`, `/data:ro`, and writable `/submissions`.
- [x] Container succeeds with a read-only root filesystem (no writes outside `/submissions`).
      Required `VOLUME ["/tmp"]`: `torch._dynamo` calls `tempfile.gettempdir()` at import.
- [x] Container runs on NVIDIA GPU and fails clearly if required artifacts are missing.
- [x] Host entrypoint output has exactly one row per mounted image, no missing/extra ids, and correct fraud direction.
- [x] Host organizer entrypoint and local frozen pipeline are compared exactly on the same images.
- [x] Full-test runtime measured below the six-hour organizer cap with margin: 131 minutes for all
      134,997 private images on one RTX 4070 (the earlier 79-minute extrapolation was optimistic).

## Kaggle final submission

- [x] Per-template normalization was tested and rejected: public LB `0.33816 -> 0.54141`.
- [x] Frozen ensemble CSV submitted via the web UI (`submission_final.csv - 8:39 PM`, 2026-07-13).
  Status/score could not be read back: the Kaggle API returns 401 for both tokens.
- [x] Private images are inferred as one private-only set through the same unchanged Docker
  entrypoint: 134,997 images, exit 0, 131 minutes measured on RTX 4070.
- [x] `scripts/assemble_final_submission.py` rejects partial IDs and merges exact rows by id in rehearsal.
- [x] Final CSV schema, ids/order, finiteness, range, row count, and SHA-256 are recorded:
  `submission_final.csv`, 142,818 rows, `239e31c4...a63b38`. Public half bit-identical to the
  frozen base; private half bit-identical to the container output; zero `0.5` placeholders left.
- [ ] Final Kaggle submission is `COMPLETE`; label/date-time are copied verbatim into the reply.

## Report and one allowed discussion reply

- [x] Draft technical report contains method, data, licenses, pretrained weights, validation, results,
  inference, Docker reproduction, hardware, runtime, and seed.
- [x] Report PDF is committed and its URL resolves from the public frozen repository.
- [x] Reply contains Kaggle team, usernames, selected submission, repository URL, frozen SHA, report
  URL, captain signature, and UTC date.
- [ ] Exactly one reply is posted to the pinned Kaggle thread by 2026-07-15 23:59 AoE.
- [ ] The reply is not duplicated.
