# E-BARD integration

CourtVision uses the NBA-trained E-BARD YOLOv8n checkpoint as its default
shared scene detector. One inference supplies player, referee, hoop, and
basketball detections to the downstream trackers.

## Reproducible assets

- Detector source: `GabrieleGiudici/E-BARD-detection-models`
- Upstream file: `BODD_yolov8n_0001.pt`
- Pinned revision: `3f4789c4431aa73269f60107a4ba0a5f86b7af8b`
- Local filename: `backend/models/ebard_yolov8n.pt`
- Detector SHA-256: `dfe3534d51bb21024d1a400c37f0c1fbf0c8b96ea9a56a5f3cb5454813bfd641`
- Team-attribution archive SHA-256: `f25c2bb8d8527992e68fc952d840f6c379b4ce24412d0dd764bdcc9c95134be7`
- Detection archive SHA-256: `4b0a5ef8fd25565714e6b36a7020bc68b1cc2765afdc82a3b3d7a099e5c2ab81`
- License: CC BY 4.0

The installers verify both downloads before materializing them. Model weights,
dataset images, runtime caches, and reports remain gitignored.

## Runtime behavior

E-BARD is selected by default and uses its published 0.25 confidence threshold.
`--scene-detector-backend` is the preferred CLI name;
`--player-detector-backend` remains a compatible alias. The legacy `current`
backend retains its 0.50 threshold and `Player`/`Ref` aliases.

Player-track and team-assignment cache identities include the selected backend,
so cached detections cannot leak between the two experiments.

Because E-BARD explicitly separates referees before tracking, E-BARD team
assignment uses two FashionCLIP team-color prompts. The current detector keeps
the additional referee prompt as a fallback.

## Measured results

The first comparison used 60 frames sampled at 10 FPS from
`det-vs-okc-0022401108_24.mp4`. This is an unlabeled agreement diagnostic:

| Measure | Current | E-BARD |
| --- | ---: | ---: |
| CPU inference FPS | 3.054 | 29.763 |
| Player detections | 582 | 603 |
| Referee detections | 182 | 186 |
| Mean players/frame | 9.700 | 10.050 |
| Mean player confidence | 0.8389 | 0.7713 |

The detectors matched 543 player boxes at IoU 0.50 with mean matched IoU
0.8481. These figures measure coverage and agreement, not accuracy; the later
annotated detection benchmark supplied the evidence used for the default.

FashionCLIP was then evaluated on all 1,530 crops in E-BARD's held-out test
split:

| Prompt strategy | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| E-BARD two-color prompt | 0.9229 | 0.9355 | 0.9232 |
| CourtVision team colors plus referee | 0.8987 | 0.9076 | 0.9075 |

The referee-aware prompt falsely assigned 32 known-player crops to `referee`.
The E-BARD detector makes that extra prompt unnecessary, so CourtVision disables
it only for E-BARD runs. The 92.29% zero-shot result is strong enough that a
FashionCLIP fine-tune is not currently justified; remaining observed errors are
more likely to come from crop quality, occlusion, and identity switches.

The ground-truth detector comparison then evaluated all 2,210 annotations in
E-BARD's 180-frame physical `yolo/test` split. Upstream's `data.yaml` aliases
this directory as `val`; the evaluator names the physical folder explicitly.

| Measure | Current 86M YOLO | E-BARD YOLOv8n |
| --- | ---: | ---: |
| CPU FPS | 3.461 | 24.709 |
| Overall mAP50 | 0.6237 | 0.8728 |
| Overall mAP50-95 | 0.3369 | 0.6062 |
| Basketball AP50 | 0.2976 | 0.6495 |
| Hoop AP50 | 0.3491 | 0.9244 |
| Player AP50 | 0.9473 | 0.9689 |
| Referee AP50 | 0.9009 | 0.9483 |

At the configured operating thresholds, E-BARD achieved 0.9128 micro F1 versus
0.8557 for the current checkpoint. Its existing 0.25 threshold was also the
best macro-F1 threshold in the benchmark sweep. E-BARD was 7.14 times faster
and improved AP for every shared class.

The normalized evaluator was checked against Ultralytics' native validation on
the E-BARD checkpoint. Native validation measured 0.8772 mAP50 and 0.6134
mAP50-95 versus 0.8728 and 0.6062 from the implementation-independent scorer;
the close agreement validates the comparison while allowing the seven-class
CourtVision output to be remapped fairly.

## Architecture decision

E-BARD is the primary scene detector. One lazy, memoized inference pass supplies
player and referee detections plus hoop context and basketball candidates to
both trackers. Player boxes feed identity tracking and team attribution.
Basketball boxes do not become final ball tracks directly:
their 0.5395 recall at confidence 0.25 is better than the current model's
0.3355, but still misses too many balls. E-BARD candidates must be fused with
WASB and passed through the existing adaptive-crop and temporal-selection
logic.

This removes the need to run the same 86M-parameter checkpoint once for players
and again for balls. E-BARD + WASB is now the CLI, local-demo, and AWS Batch
default; legacy checkpoints are requested only when their backends are selected
explicitly. Shared-inference and adaptive-ball caches carry separate versioned
identities. The implementation boundary remains modular:
the scene detector can be upgraded independently from identity tracking,
FashionCLIP team attribution, ball trajectory reconstruction, and event logic.

## Possession and shot-attempt architecture

The event layer now consumes an explicit possession timeline instead of
inferring every event from adjacent holder IDs. A shot candidate requires a
release, rising ball flight, and approach to an E-BARD hoop. Local rim
approaches split long possession segments so a block, retained offensive ball,
and a quick second attempt can be represented separately. A validated pass or
interception that resumes after the proposed rim contact preempts the shot
hypothesis.

The production contract deliberately stops at `shot_attempt`. It does not
publish makes, misses, rebounds, dead balls, or putback subtypes. Broadcast
ball tracking commonly disappears behind hands, players, the backboard, or the
net, so those outcome claims were not reliable enough for deployment. Stable
post-shot possession is retained only as an internal window boundary for
de-duplicating shot trajectories. Public shot events contain release, arc,
rim-approach, shooter/team, and confidence evidence without an `outcome` field.

`--event-only` regenerates player, ball, possession, and event overlays while
skipping unrelated court homography and speed/distance work. `--analysis-only`
also skips video encoding, which is useful for threshold sweeps.

## Shared-runtime verification

The implemented default was exercised on 40 frames from a separate 2025 NBA
broadcast clip. The cold-cache log contained one shared E-BARD full-frame pass;
both player tracking and the hybrid ball tracker consumed that result. A second
run hit the player, ball, adaptive, pose, court, and team caches and performed
no E-BARD inference.

On this clip, the raw hybrid pool contained 36 WASB temporal candidates and two
E-BARD full-frame basketball candidates. The fused tracker selected a ball on
33 of 40 frames. This illustrates why WASB remains part of the architecture:
E-BARD provides stronger semantic grounding, while WASB supplies dense small-
ball localization. The analysis manifest records the scene backend, ball
backend, semantic-ball source, WASB status, and whether inference was shared.

## Commands

```powershell
.\.venv\Scripts\python.exe scripts\prepare_ebard_model.py
.\.venv\Scripts\python.exe scripts\prepare_ebard_team_dataset.py
.\.venv\Scripts\python.exe scripts\evaluate_ebard_fashionclip.py
.\.venv\Scripts\python.exe scripts\prepare_ebard_detection_dataset.py
.\.venv\Scripts\python.exe scripts\evaluate_ebard_detection.py
.\.venv\Scripts\python.exe scripts\compare_player_detectors.py game.mp4
.\.venv\Scripts\python.exe main.py game.mp4 `
  --output-video output_videos\game-ebard-wasb.mp4
```

For AWS Batch, upload `ebard_yolov8n.pt` and the WASB, pose, and court weights
to the configured private model bucket. Set
`COURTVISION_PLAYER_DETECTOR_BACKEND=current` only for a legacy comparison.
