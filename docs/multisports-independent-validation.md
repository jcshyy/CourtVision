# MultiSports independent validation — 2026-09-03

## Decision

Keep the current E-BARD + WASB hybrid for now. Do **not** add or replace the
ball model based on event F1 alone. The independent pilot shows two separable
problems:

1. Ball observation is weak around missed shots, so a basketball-specific ball
   model ablation is justified.
2. Possession continuity and event lifecycle logic remain the larger pass
   bottleneck, because many missed passes still contain visible ball evidence.

The sealed sample does not reproduce the user's exact shot-as-pass complaint.
At 0.25 seconds, no annotated shot was confused with a pass; three annotated
passes were confused with shots. It therefore validates generalization and
finds the opposite confusion mode, but it cannot replace a clean-source replay
of the reported CourtVision clip.

## Data and protocol

- Source: [MultiSports (ICCV 2021)](https://github.com/MCG-NJU/MultiSports),
  official gated [Hugging Face distribution](https://huggingface.co/datasets/MCG-NJU/SportsAction).
- Immutable revision: `01600ce7eabbf42a5ee7c82b82f49a11597b3a5f`.
- Official annotation SHA-256:
  `e6579fb0986713c0a681dd0abef2475eedb416915706b69c3e8580f501a46942`.
- Deterministic sample: lowest SHA-256 of a fixed seed plus official validation
  ID, selected before reading labels or predictions.
- Coverage: 12/147 validation videos, 5,178/72,179 frames (7.17%), and 58
  scored action tubes. Original 25 FPS and resolution were retained.
- Scored classes: pass, 2/3-point shot as `shot_attempt`, pass/dribble steal as
  `interception`, and pass-inbound as `throw_in`. Rebounds, free throws, and
  blocks remain outside the declared production event scope.
- Primary metric: one-to-one temporal event point-to-action-interval F1 with
  0.25-second tolerance. Sensitivity uses 0.50 seconds. This is **not** the
  official MultiSports spatial tube mAP.
- Pipeline: analysis-only, uncertain-team abstention allowed, E-BARD scene
  detector, hybrid E-BARD + WASB ball detector. No ground truth was used for
  team assignment.

Only the selected clip byte ranges were transferred from the official 15.5 GB
archive. The downloader validated HTTP ranges, TAR headers, clip lengths and a
SHA-256 for every retrieved MP4; OpenCV metadata matched the official frame
counts and resolutions. The whole TAR was not downloaded, so its published
whole-archive checksum was not independently recomputed.

## Results

| Metric | 0.25 s primary | 0.50 s sensitivity |
| --- | ---: | ---: |
| Micro precision / recall / F1 | 0.487 / 0.328 / **0.392** | 0.667 / 0.448 / **0.536** |
| Pass precision / recall / F1 | 0.563 / 0.243 / **0.340** | 0.813 / 0.351 / **0.491** |
| Shot precision / recall / F1 | 0.421 / 0.500 / **0.457** | 0.579 / 0.688 / **0.629** |
| Interception F1 | **0.500** | **0.500** |
| Throw-in F1 | **0.400** | **0.400** |

Primary totals are 19 TP, 20 FP and 39 FN over 58 truth events and 39
predictions. Approximate 95% Wilson intervals are broad because this is a pilot:
micro precision 0.339–0.638 and recall 0.221–0.456; pass recall 0.134–0.401;
shot recall 0.280–0.720. One duplicate prediction was observed (2.56%).

Primary confusion details:

- 24 passes were missed, 9 matched, 3 were called shots, and 1 was called a
  throw-in.
- 8 shots matched and 8 were missed; none was called a pass.
- Predictions without a nearby annotated event included 7 passes, 8 shots and
  1 interception. MultiSports' excluded action classes mean these are event
  metric false positives, not all necessarily bad ball detections.

The 0.50-second improvement is large. Four extra passes and three extra shots
match when tolerance doubles, indicating release localization/annotation
alignment is also material—not just class selection.

## Detector and possession evidence

Across all 5,178 frames, the hybrid pipeline marked the ball directly observed
in 46.68% and a controlled possession state in 35.86%. Within annotated event
intervals:

| Truth subset | Ball observed | Controlled possession |
| --- | ---: | ---: |
| Matched passes (9) | 71.11% | 68.89% |
| Missed passes (28) | 57.56% | 43.91% |
| Matched shots (8) | 57.96% | 28.03% |
| Missed shots (8) | 31.17% | 18.18% |

These are internal support rates, not ball-localization ground truth. Still,
they make the engineering priority clear: shot recall is strongly associated
with ball visibility, while more than half of missed-pass frames already have
an observed ball and almost half have controlled state. A better ball model may
help shots, but cannot by itself recover the pass lifecycle failures.

This aligns with published model design. [WASB](https://arxiv.org/abs/2311.05237)
already combines high-resolution features, position-aware training and temporal
consistency across multiple sports, so the current hybrid is not merely a
single-frame generic detector. Motion-aware alternatives such as
[TrackNetV4](https://arxiv.org/abs/2409.14543) are promising, but their reported
experiments are tennis and badminton rather than broadcast basketball. They
should be benchmarked on basketball ball coordinates before replacement.

## Implementation findings during validation

Independent data exposed two non-scoring infrastructure failures:

- Empty court-keypoint detections caused throw-in geometry to index an empty
  tensor. Production now abstains from endline geometry in that case, with a
  regression test.
- One 875-frame 1280×720 clip required about 2.3 GiB decoded memory, above the
  CLI's 2 GiB safety ceiling. A validation-only wrapper raised the ceiling to
  3 GiB for that clip; all 875 original frames were retained.

Both changes were registered after their failure and before retry, with exact
source/wrapper hashes. Neither changes models, event thresholds, scoring,
resolution or sampling. The protocol records ten outputs under the original
source snapshot and two under the crash-only amended snapshot.

## Next implementation gate

1. Annotate ball centers/visibility on a stratified basketball development set:
   release/rim/catch windows, occlusions, camera motion, and no-event negatives.
   Score E-BARD-only, WASB-only, the current hybrid, and one temporal candidate
   on localization recall, false positives and gap length—not event F1 alone.
2. Improve possession continuity on a development split: bridge short
   occlusions using trajectory plus hand/pose support, preserve player identity
   through switches, and expose an explicit uncertain transfer rather than
   forcing a pass/shot label.
3. Keep shot-flight evidence authoritative over provisional holder changes, but
   calibrate rim/launch evidence against the 11 external shot false positives.
4. Re-run this sealed 12-video pilot once after changes, then expand to the full
   147-video released validation split for a final estimate. Do not tune against
   these 12 clips now that their labels and results have been inspected.

Machine-readable artifacts are under
`benchmarks/event_sequences_v2/results/2026-09-03`: protocol, primary report,
sensitivity report, and detector/possession diagnostic summary.

Verification after the robustness change: 41/41 event-focused tests passed;
the full suite ran 364 tests with 362 passing. The same two unrelated web-demo
copy/configuration assertions documented in the lifecycle report remain failing.

## Post-change possession-continuity replay

After selecting and finalizing the next change entirely on local development
labels, the same frozen detector caches and 12 videos were replayed once:

| Metric | v1 | Possession continuity v2 |
| --- | ---: | ---: |
| Micro F1, 0.25 s | 0.3918 | **0.4286** |
| Pass F1, 0.25 s | 0.3396 | **0.4000** |
| Shot F1, 0.25 s | 0.4571 | **0.4706** |
| Micro F1, 0.50 s | 0.5361 | **0.5510** |

The change localizes releases at the last authoritative source-control frame
instead of a later provisional holder tail and supports conservative two-frame
interpolated possession gaps. Primary matched passes increased 9→11, pass→shot
confusions decreased 3→2, and total TP/FP/FN changed from 19/20/39 to
21/19/37. No post-result tuning was performed. See the
[implementation report](possession-continuity-v2-results.md).
