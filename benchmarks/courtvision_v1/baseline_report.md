# CourtVision v1 baseline

Scored 201 verified frames across 5 videos using cached pipeline artifacts.

## Headline metrics

| Area | Metric | Result |
|---|---|---:|
| Ball | Observed detection recall on visible frames | 43.3% |
| Ball | Visible frames within 50 px | 69.3% |
| Ball | Median center error | 9.2 px |
| Ball | 90th-percentile center error | 368.1 px |
| Possession | Six-state accuracy | 19.9% |
| Possession | Controlled precision / recall / F1 | 69.8% / 24.8% / 36.6% |
| Team | End-to-end / conditional accuracy | 24.8% / 100.0% |
| Events | Precision / recall / F1 at ±15 frames | 66.7% / 66.7% / 66.7% |

## Largest measured failures

1. **possession_state** — Six-state possession accuracy is 19.9%; the pipeline currently emits only controlled or loose for this benchmark.
2. **controlled_possession_recall** — Controlled-possession recall is 24.8%.
3. **possession_team_end_to_end** — End-to-end possession-team accuracy is 24.8% with 24.8% team coverage.
4. **ball_detection** — Only 43.3% of visible balls have an observed post-filter detection; the rest depend on interpolation.
5. **events** — Event F1 is 66.7% at ±15 frames (6 TP, 3 FP, 3 FN).
6. **ball_localization_50px** — Only 69.3% of visible balls are localized within 50 pixels after interpolation.

## Root-cause signals

- 62 controlled-possession misses are attributed to `interpolated_ball_not_confirmable`.
- Team assignment is 100.0% correct when the pipeline actually identifies a holder; low end-to-end team accuracy is therefore a holder-coverage problem.
- Ball localization has a 9.2 px median but a 368.1 px p90, showing a severe wrong-object/interpolation tail.
- Shot event recall is 0.0%; the current event detector emits only passes and interceptions.

## Per-video frame metrics

| Video | Samples | Ball observed recall | Ball within 50 px | Controlled F1 | Team end-to-end |
|---|---:|---:|---:|---:|---:|
| video_1 | 9 | 50.0% | 62.5% | 0.0% | 0.0% |
| video_2 | 38 | 58.1% | 96.8% | 75.6% | 63.0% |
| video_3 | 58 | 36.2% | 83.0% | 33.3% | 21.6% |
| spursknicksclip | 59 | 32.5% | 35.0% | 0.0% | 0.0% |
| spursknicks2 | 37 | 54.2% | 66.7% | 34.5% | 33.3% |

## Event results

| Type | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| interception | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| pass | 5 | 2 | 1 | 71.4% | 83.3% | 76.9% |
| shot | 0 | 0 | 2 | 0.0% | 0.0% | 0.0% |

## Interpretation limits

- Team IDs are arbitrary clusters, so the scorer selects the better of the two possible per-video cluster mappings.
- Event labels contain reviewed pipeline candidates plus explicitly identified misses; they are not yet a complete play-by-play inventory.
- The current possession model does not predict in-flight, shot, dead-ball, or unknown states, so six-state scoring exposes those as unsupported classes.
- Interpolated ball positions count toward localization but not observed-detection recall.
