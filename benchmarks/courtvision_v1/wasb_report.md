# CourtVision v1 baseline

Scored 201 verified frames across 5 videos using cached pipeline artifacts.

## Headline metrics

| Area | Metric | Result |
|---|---|---:|
| Ball | Observed detection recall on visible frames | 78.0% |
| Ball | Visible frames within 50 px | 86.7% |
| Ball | Median center error | 2.6 px |
| Ball | 90th-percentile center error | 5.0 px |
| Possession | Six-state accuracy | 34.3% |
| Possession | Controlled precision / recall / F1 | 72.1% / 51.2% / 59.9% |
| Team | End-to-end / conditional accuracy | 44.6% / 98.2% |
| Events | Precision / recall / F1 at ±15 frames | 45.5% / 50.0% / 47.6% |

## Largest measured failures

1. **possession_state** — Six-state possession accuracy is 34.3%; the pipeline currently emits only controlled or loose for this benchmark.
2. **possession_team_end_to_end** — End-to-end possession-team accuracy is 44.6% with 45.5% team coverage.
3. **events** — Event F1 is 47.6% at ±15 frames (5 TP, 6 FP, 5 FN).
4. **controlled_possession_recall** — Controlled-possession recall is 51.2%.
5. **ball_detection** — Only 78.0% of visible balls have an observed post-filter detection; the rest depend on interpolation.
6. **ball_localization_50px** — Only 86.7% of visible balls are localized within 50 pixels after interpolation.

## Root-cause signals

- 30 controlled-possession misses are attributed to `ball_missing`.
- Team assignment is 98.2% correct when the pipeline actually identifies a holder; low end-to-end team accuracy is therefore a holder-coverage problem.
- Ball localization has a 2.6 px median but a 5.0 px p90, showing a severe wrong-object/interpolation tail.
- Shot event recall is 0.0%; the current event detector emits only passes and interceptions.

## Per-video frame metrics

| Video | Samples | Ball observed recall | Ball within 50 px | Controlled F1 | Team end-to-end |
|---|---:|---:|---:|---:|---:|
| video_1 | 9 | 50.0% | 75.0% | 50.0% | 40.0% |
| video_2 | 38 | 87.1% | 100.0% | 75.5% | 70.4% |
| video_3 | 58 | 87.2% | 87.2% | 65.7% | 54.1% |
| spursknicksclip | 59 | 75.0% | 82.5% | 38.3% | 13.5% |
| spursknicks2 | 37 | 62.5% | 79.2% | 55.2% | 53.3% |

## Event results

| Type | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| interception | 1 | 3 | 0 | 25.0% | 100.0% | 40.0% |
| pass | 4 | 3 | 3 | 57.1% | 57.1% | 57.1% |
| shot | 0 | 0 | 2 | 0.0% | 0.0% | 0.0% |

## Interpretation limits

- Team IDs are arbitrary clusters, so the scorer selects the better of the two possible per-video cluster mappings.
- Event labels contain reviewed pipeline candidates plus explicitly identified misses; they are not yet a complete play-by-play inventory.
- The current possession model does not predict in-flight, shot, dead-ball, or unknown states, so six-state scoring exposes those as unsupported classes.
- Interpolated ball positions count toward localization but not observed-detection recall.
