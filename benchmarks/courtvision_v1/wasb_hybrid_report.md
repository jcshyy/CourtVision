# CourtVision v1 baseline

Scored 201 verified frames across 5 videos using cached pipeline artifacts.

## Headline metrics

| Area | Metric | Result |
|---|---|---:|
| Ball | Observed detection recall on visible frames | 81.3% |
| Ball | Visible frames within 50 px | 87.3% |
| Ball | Median center error | 2.6 px |
| Ball | 90th-percentile center error | 4.9 px |
| Possession | Six-state accuracy | 40.8% |
| Possession | Controlled precision / recall / F1 | 75.5% / 61.2% / 67.6% |
| Team | End-to-end / conditional accuracy | 54.5% / 98.5% |
| Events | Precision / recall / F1 at ±15 frames | 50.0% / 60.0% / 54.5% |

## Largest measured failures

1. **possession_state** — Six-state possession accuracy is 40.8%; the pipeline currently emits only controlled or loose for this benchmark.
2. **possession_team_end_to_end** — End-to-end possession-team accuracy is 54.5% with 55.4% team coverage.
3. **events** — Event F1 is 54.5% at ±15 frames (6 TP, 6 FP, 4 FN).
4. **controlled_possession_recall** — Controlled-possession recall is 61.2%.
5. **ball_detection** — Only 81.3% of visible balls have an observed post-filter detection; the rest depend on interpolation.
6. **ball_localization_50px** — Only 87.3% of visible balls are localized within 50 pixels after interpolation.

## Root-cause signals

- 12 controlled-possession misses are attributed to `interpolated_ball_not_confirmable`.
- Team assignment is 98.5% correct when the pipeline actually identifies a holder; low end-to-end team accuracy is therefore a holder-coverage problem.
- Ball localization has a 2.6 px median but a 4.9 px p90, showing a severe wrong-object/interpolation tail.
- Shot event recall is 0.0%; the current event detector emits only passes and interceptions.

## Per-video frame metrics

| Video | Samples | Ball observed recall | Ball within 50 px | Controlled F1 | Team end-to-end |
|---|---:|---:|---:|---:|---:|
| video_1 | 9 | 50.0% | 75.0% | 50.0% | 40.0% |
| video_2 | 38 | 96.8% | 100.0% | 82.8% | 88.9% |
| video_3 | 58 | 89.4% | 89.4% | 73.2% | 62.2% |
| spursknicksclip | 59 | 77.5% | 87.5% | 53.8% | 24.3% |
| spursknicks2 | 37 | 62.5% | 70.8% | 53.3% | 53.3% |

## Event results

| Type | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| interception | 1 | 3 | 0 | 25.0% | 100.0% | 40.0% |
| pass | 5 | 3 | 2 | 62.5% | 71.4% | 66.7% |
| shot | 0 | 0 | 2 | 0.0% | 0.0% | 0.0% |

## Interpretation limits

- Team IDs are arbitrary clusters, so the scorer selects the better of the two possible per-video cluster mappings.
- Event labels contain reviewed pipeline candidates plus explicitly identified misses; they are not yet a complete play-by-play inventory.
- The current possession model does not predict in-flight, shot, dead-ball, or unknown states, so six-state scoring exposes those as unsupported classes.
- Interpolated ball positions count toward localization but not observed-detection recall.
