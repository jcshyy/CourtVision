# Shot/pass lifecycle implementation plan

Date: 2026-09-03. Scope: event logic and evaluation; retain E-BARD + WASB.

## Sequence and acceptance criteria

1. Capture the existing unit-test and six-clip BARD benchmark baseline without
   overwriting existing artifacts. Preserve unrelated working-tree changes.
2. Make possession events provisional until shot arbitration. A later outlet
   must not erase an earlier shot. Suppress a rebound acquisition interpreted as
   a pass/steal even when player IDs change. Preserve a genuine pre-rim catch.
3. Deduplicate passes using overlapping release/catch intervals and actor
   evidence; deduplicate shot hypotheses using their flight intervals, with
   separate attempts retained after a new control/release boundary. Resolve
   shooter identity from nearby stable pre-release control, not stale holders.
4. Represent shot-flight/post-rim phases in the inspectable lifecycle. Separate
   throw-ins only when independently supported by court-boundary evidence;
   never infer a made basket merely from an opponent controlling the ball.
5. Add event-level temporal one-to-one scoring, confusion and duplicate
   diagnostics, missing-coverage reporting, and a MultiSports annotation adapter.
   Do not relabel clip-count F1 as temporal accuracy or synthetic tests as video
   accuracy. Keep imported public validation data separate from calibration.
6. Run regression tests and replay local benchmark inputs through the current
   pipeline/cache with before/after reports. Investigate changed counts; report
   regressions and benchmark limitations rather than tuning to a target number.

## Required regression scenarios

- Shot followed by rebound/outlet retains the attempt and the later true pass.
- Shot flight cannot simultaneously become a pass/interception after ID churn.
- Stable catch before the rim preserves a lob; a subsequent shot remains separate.
- Overlapping hypotheses with different shooter IDs produce one attempt.
- New control and release after the rim allows a putback.
- Two-source copies of one pass merge; rapid distinct passes remain distinct.
- Discontinuities isolate event flights and terminal release candidates.
- Inbound evidence is separate from shot outcome; unknown evidence stays unknown.
- Temporal scoring penalizes wrong classes, duplicates and missing predictions.

## Public benchmark research

Primary choice: [MultiSports](https://github.com/MCG-NJU/MultiSports), ICCV 2021.
Its official [dataset card](https://huggingface.co/datasets/MCG-NJU/SportsAction)
documents frame-indexed action tubes, including basketball pass, 2/3-point shot,
offensive/defensive rebound, pass/dribble steal, and pass-inbound. Indices are
one-based; the released `test_videos` field is the validation split, not the
hidden competition test set. Access requires login and license acceptance
(CC BY-NC 4.0). No account agreement is accepted on the user's behalf.

Alternative: [FineSports](https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024),
CVPR 2024. It requires a signed release agreement emailed to the authors. It is
not an immediately runnable independent benchmark in this workspace.

Existing BARD labels remain useful for clip-level count regression only. The
uploaded rendered video is qualitative evidence, not clean detector input or
verified temporal ground truth. Do not run detectors on burned-in overlays and
claim a clean source-video evaluation.

## Model decision gate

Use candidate rejection and confusion diagnostics to attribute remaining errors.
Only revisit ball-model training/replacement if localization/coverage dominates
misses after lifecycle fixes. Consider an action classifier separately if valid
tracks remain ambiguous. No third detector is added in this change.
