# CourtVision v1 baseline

Scored 201 verified frames across 5 videos using cached pipeline artifacts.

## Headline metrics

| Area | Metric | Result |
|---|---|---:|
| Ball | Observed detection recall on visible frames | 60.0% |
| Ball | Visible frames within 50 px | 66.7% |
| Ball | Median center error | 3.3 px |
| Ball | 90th-percentile center error | 176.5 px |
| Possession | Six-state accuracy | 33.3% |
| Possession | Controlled precision / recall / F1 | 87.9% / 47.9% / 62.0% |
| Team | End-to-end / conditional accuracy | 44.6% / 100.0% |
| Events | Precision / recall / F1 at ±15 frames | 100.0% / 80.0% / 88.9% |

## Largest measured failures

1. **possession_state** — Six-state possession accuracy is 33.3%; the pipeline currently emits only controlled or loose for this benchmark.
2. **possession_team_end_to_end** — End-to-end possession-team accuracy is 44.6% with 44.6% team coverage.
3. **controlled_possession_recall** — Controlled-possession recall is 47.9%.
4. **ball_detection** — Only 60.0% of visible balls have an observed post-filter detection; the rest depend on interpolation.
5. **ball_localization_50px** — Only 66.7% of visible balls are localized within 50 pixels after interpolation.
6. **events** — Event F1 is 88.9% at ±15 frames (8 TP, 0 FP, 2 FN).

## Root-cause signals

- 27 controlled-possession misses are attributed to `interpolated_ball_not_confirmable`.
- Team assignment is 100.0% correct when the pipeline actually identifies a holder; low end-to-end team accuracy is therefore a holder-coverage problem.
- Ball localization has a 3.3 px median but a 176.5 px p90, showing a severe wrong-object/interpolation tail.
- Shot event recall is 0.0%; the current event detector emits only passes and interceptions.

## Per-video frame metrics

| Video | Samples | Ball observed recall | Ball within 50 px | Controlled F1 | Team end-to-end |
|---|---:|---:|---:|---:|---:|
| video_1 | 9 | 62.5% | 50.0% | 57.1% | 40.0% |
| video_2 | 38 | 83.9% | 90.3% | 81.6% | 74.1% |
| video_3 | 58 | 53.2% | 70.2% | 50.9% | 35.1% |
| spursknicksclip | 59 | 57.5% | 55.0% | 60.4% | 35.1% |
| spursknicks2 | 37 | 45.8% | 54.2% | 52.2% | 40.0% |

## Event results

| Type | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| interception | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| pass | 7 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| shot | 0 | 0 | 2 | 0.0% | 0.0% | 0.0% |

## Interpretation limits

- Team IDs are arbitrary clusters, so the scorer selects the better of the two possible per-video cluster mappings.
- Event labels contain reviewed pipeline candidates plus explicitly identified misses; they are not yet a complete play-by-play inventory.
- The current possession model does not predict in-flight, shot, dead-ball, or unknown states, so six-state scoring exposes those as unsupported classes.
- Interpolated ball positions count toward localization but not observed-detection recall.
