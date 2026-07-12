# FREUID reproducibility and disqualification-prevention checklist

Do not mark an unchecked item complete without retaining command output or an artifact hash in
`FINAL_AUDIT.md`.

## Code freeze and public package

- [x] Repository is public: `https://github.com/Marc-Dvci/FREUID` (GitHub API checked 2026-07-12).
- [x] Source is under the OSI-approved MIT license.
- [ ] Final architecture, training code, and all three selected weights are committed before the
  organizers release private images on 2026-07-13.
- [ ] Three `.pth` files are present through Git LFS after a clean clone (`git lfs pull`).
- [ ] Frozen 40-character commit SHA is recorded in the report and Kaggle reply.
- [ ] No weight, architecture, checkpoint, training-code, or hyperparameter change is made after
  private-image release. Only inference, CSV merging, documentation, and packaging may change.

## Docker sandbox contract

- [x] Entrypoint discovers flat image files directly in `/data`; it requires no CSV or manifest.
- [x] `id` is exactly the filename stem; all organizer-listed extensions are case-insensitive.
- [x] Duplicate stems, missing weights, empty input, non-finite values, and row mismatches fail nonzero.
- [x] Output path and schema are exactly `/submissions/submission.csv` and `id,label`.
- [x] Scores are finite in `[0,1]`, with higher meaning more likely fraudulent.
- [x] All three weights are copied into the image; model creation uses `pretrained=False`.
- [x] Inference makes no HTTP/API calls and does not need a writable model cache.
- [ ] Base image is pinned by immutable digest.
- [ ] Image builds from a clean clone after `git lfs pull`.
- [ ] Container succeeds with `--network none`, `/data:ro`, and writable `/submissions`.
- [ ] Container succeeds with a read-only root filesystem (no writes outside `/submissions`).
- [ ] Container runs on NVIDIA GPU and fails clearly if required artifacts are missing.
- [ ] Output has exactly one row per mounted image, no missing/extra ids, and correct fraud direction.
- [ ] Docker output and local frozen-pipeline output are compared on the same images.
- [ ] Full-test runtime is measured/extrapolated below the six-hour organizer cap with margin.

## Kaggle final submission

- [x] Per-template normalization was tested and rejected: public LB `0.33816 -> 0.54141`.
- [ ] Frozen ensemble public-row CSV is submitted and its public score/status recorded.
- [ ] Private images are inferred as one private-only set through the same Docker entrypoint.
- [ ] `scripts/assemble_final_submission.py` merges those exact rows by id into the frozen full CSV.
- [ ] Final CSV schema, ids/order, finiteness, range, row count, and SHA-256 are recorded.
- [ ] Final Kaggle submission is `COMPLETE`; label/date-time are copied verbatim into the reply.

## Report and one allowed discussion reply

- [ ] Technical report contains method, data, licenses, pretrained weights, validation, results,
  inference, Docker reproduction, hardware, runtime, and seed.
- [ ] Report PDF is committed and its URL resolves from the public frozen repository.
- [ ] Reply contains Kaggle team, usernames, selected submission, repository URL, frozen SHA, report
  URL, captain signature, and UTC date.
- [ ] Exactly one reply is posted to the pinned Kaggle thread by 2026-07-15 23:59 AoE.
- [ ] The reply is not duplicated.
