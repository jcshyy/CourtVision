# Temporal event benchmark v2

This directory contains the temporal evaluator, local calibration reports, and
a completed independent MultiSports pilot. The official gated data were accessed
only after the user accepted the license; no mirror or access bypass was used.

## Public benchmark

[MultiSports, ICCV 2021](https://github.com/MCG-NJU/MultiSports) is the preferred
independent benchmark. Its official [dataset card](https://huggingface.co/datasets/MCG-NJU/SportsAction)
documents basketball pass, 2-point/3-point shot, steals, rebounds, and pass-inbound
with frame-level person/action tubes. The official released `test_videos` list
means **validation**. Do not call it a held-out competition test result.

[FineSports, CVPR 2024](https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024)
is an alternative, but requires a signed release agreement emailed to its authors.

## Independent pilot

A deterministic 12/147-video sample of the released basketball validation split
was registered before inference. All 5,178 original 25 FPS frames were processed.
The primary 0.25-second micro F1 is 0.3918; 0.50-second sensitivity F1 is 0.5361.
See the [full report](../../docs/multisports-independent-validation.md) and the
machine-readable files in `results/2026-09-03`.

A post-change cache-identical replay improved primary micro F1 to **0.4286**
and sensitivity F1 to **0.5510**. Local possession-event F1 improved from
0.4000 to **0.7143**, while shot-window F1 remained 0.8750. Details are in the
[possession-continuity report](../../docs/possession-continuity-v2-results.md).

The actual annotation pickle was loaded with the restricted unpickler, and each
selected MP4's FPS, frame count and resolution were checked against official
metadata. `scripts/download_multisports.py`, `download_multisports_sample.py`,
`prepare_multisports_validation.py`, and `run_multisports_validation.py` preserve
the pinned revision, hashes, selection, missing-coverage behavior and audit log.

Prediction paths must be exact, for example
`runs/multisports/basketball/<official_clip_id>_analysis.json`. Run untrimmed clips
at their documented frame-sequence FPS. Missing predictions count as false
negatives and lower coverage; missing negative clips are also reported.

## Metric contract

- Zero-based inclusive truth action intervals; point labels use equal endpoints.
- Predicted pass/shot/throw-in release must fall inside an annotated action
  interval (or within a declared tolerance, default 0.25 seconds).
- One-to-one, maximum-cardinality class-aware matching penalizes duplicates.
- Wrong-class matches among unmatched events populate a confusion matrix but
  count as both a false positive and a false negative.
- Throw-in decisions are read from lifecycle diagnostics, separate from public
  pass totals. Rebounds/free throws remain outside the scored production scope.
- This temporal metric is **not** MultiSports' official spatial tube mAP.
- Source tubes do not supply exact release/rim/catch or team identity labels;
  do not invent timing MAE or team-accuracy claims from them.

The format is a JSON object with `name`, `scored_types`, `videos` (video_id,
fps, frame_count, split) and `events` (video_id, type, start_frame, end_frame).
Synthetic evaluator tests validate accounting only, not detector accuracy.
Keep public validation sealed from threshold tuning; use local reviewed clips
for calibration and report that limitation explicitly.

## Local calibration replay

`local_shot_windows.json` contains nine coarse, assistant-reviewed shot-action
windows from six BARD clips inspected during implementation. It is not a sealed
test set, exhaustive pass labeling, or independent human verification.
`scripts/compare_event_regression.py` also scores the eight existing verified
possession events with explicit release/catch boundaries in `courtvision_v1`.
That older set was candidate-reviewed and is not exhaustive ground truth.

See [the results report](../../docs/event-lifecycle-results.md) for exact metrics,
remaining misses, replay provenance, and commands. Machine-readable reports
are under `results/2026-09-03`.
