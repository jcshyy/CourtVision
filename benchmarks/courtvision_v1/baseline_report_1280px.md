# CourtVision v1 baseline

Scored 201 verified frames across 5 videos using cached pipeline artifacts.

## Headline metrics

| Area | Metric | Result |
|---|---|---:|
| Ball | Observed detection recall on visible frames | 56.7% |
| Ball | Visible frames within 50 px | 64.0% |
| Ball | Median center error | 3.4 px |
| Ball | 90th-percentile center error | 192.9 px |
| Possession | Six-state accuracy | 29.4% |
| Possession | Controlled precision / recall / F1 | 68.5% / 41.3% / 51.5% |
| Team | End-to-end / conditional accuracy | 34.7% / 97.7% |
| Events | Precision / recall / F1 at ±15 frames | 42.9% / 60.0% / 50.0% |

## Largest measured failures

1. **possession_state** — Six-state possession accuracy is 29.4%; the pipeline currently emits only controlled or loose for this benchmark.
2. **possession_team_end_to_end** — End-to-end possession-team accuracy is 34.7% with 35.5% team coverage.
3. **controlled_possession_recall** — Controlled-possession recall is 41.3%.
4. **events** — Event F1 is 50.0% at ±15 frames (6 TP, 8 FP, 4 FN).
5. **ball_detection** — Only 56.7% of visible balls have an observed post-filter detection; the rest depend on interpolation.
6. **ball_localization_50px** — Only 64.0% of visible balls are localized within 50 pixels after interpolation.

## Root-cause signals

- 28 controlled-possession misses are attributed to `interpolated_ball_not_confirmable`.
- Team assignment is 97.7% correct when the pipeline actually identifies a holder; low end-to-end team accuracy is therefore a holder-coverage problem.
- Ball localization has a 3.4 px median but a 192.9 px p90, showing a severe wrong-object/interpolation tail.
- Shot event recall is 0.0%; the current event detector emits only passes and interceptions.

## Per-video frame metrics

| Video | Samples | Ball observed recall | Ball within 50 px | Controlled F1 | Team end-to-end |
|---|---:|---:|---:|---:|---:|
| video_1 | 9 | 25.0% | 50.0% | 33.3% | 0.0% |
| video_2 | 38 | 74.2% | 74.2% | 60.9% | 51.9% |
| video_3 | 58 | 42.6% | 42.6% | 53.6% | 32.4% |
| spursknicksclip | 59 | 67.5% | 82.5% | 50.0% | 27.0% |
| spursknicks2 | 37 | 54.2% | 66.7% | 41.2% | 40.0% |

## Event results

| Type | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| interception | 1 | 3 | 0 | 25.0% | 100.0% | 40.0% |
| pass | 5 | 5 | 2 | 50.0% | 71.4% | 58.8% |
| shot | 0 | 0 | 2 | 0.0% | 0.0% | 0.0% |

## Interpretation limits

- Team IDs are arbitrary clusters, so the scorer selects the better of the two possible per-video cluster mappings.
- Event labels contain reviewed pipeline candidates plus explicitly identified misses; they are not yet a complete play-by-play inventory.
- The current possession model does not predict in-flight, shot, dead-ball, or unknown states, so six-state scoring exposes those as unsupported classes.
- Interpolated ball positions count toward localization but not observed-detection recall.
