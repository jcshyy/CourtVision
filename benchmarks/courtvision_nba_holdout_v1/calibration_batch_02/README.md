# NBA holdout calibration batch 02

This is a manually drafted, raw-video-only annotation batch for `nba_001` through
`nba_005`, with event corrections from a second reviewer. CourtVision predictions,
cached tracks, analyzer outputs, and prior benchmark scores were not used while
labeling it.

The batch remains separate from `../calibration` until a second reviewer verifies
it. Do not score or tune the analyzer against these clips before that review is
frozen.

## Contents

- `manifest.json`: clip, scene, and team definitions.
- `annotation_spec.json`: hand-reviewed possession intervals, conservative ball
  centers, and event records.
- `frames.jsonl`: 206 sampled frame records generated every 15 frames.
- `events.jsonl`: 22 events, including 21 second-reviewer-verified events and one
  remaining draft pass.
- `REVIEW_CHECKLIST.md`: short clip-by-clip verification guide.

Only 45 of the 206 samples are marked `visible` with a center. Another 8 are
explicitly `out_of_frame`; the remaining 153 are `uncertain` with no center. This
is deliberate: a center is present only when the raw frame makes the ball
visually separable, preventing body-locked guesses from becoming ground truth.

## Rebuild and validate

```powershell
.\.venv\Scripts\python.exe scripts\build_nba_holdout_annotation_batch.py --calibration-dir benchmarks\courtvision_nba_holdout_v1\calibration_batch_02
.\.venv\Scripts\python.exe scripts\validate_nba_holdout_annotations.py --calibration-dir benchmarks\courtvision_nba_holdout_v1\calibration_batch_02
.\.venv\Scripts\python.exe scripts\render_nba_holdout_annotation_review.py --calibration-dir benchmarks\courtvision_nba_holdout_v1\calibration_batch_02 --output-dir holdout_sources\previews\nba_holdout_annotation_review_batch_02_reviewed
```

The post-review sheets are in
`holdout_sources/previews/nba_holdout_annotation_review_batch_02_reviewed`.
