# CourtVision v1 baseline

Scored 9 verified frames across 1 videos using cached pipeline artifacts.

## Headline metrics

| Area | Metric | Result |
|---|---|---:|
| Ball | Observed detection recall on visible frames | 50.0% |
| Ball | Visible frames within 50 px | 50.0% |
| Ball | Median center error | 3.9 px |
| Ball | 90th-percentile center error | 5.0 px |
| Possession | Six-state accuracy | 11.1% |
| Possession | Controlled precision / recall / F1 | 0.0% / 0.0% / 0.0% |
| Team | End-to-end / conditional accuracy | 0.0% / 0.0% |
| Events | Precision / recall / F1 at ±15 frames | 0.0% / 0.0% / 0.0% |

## Largest measured failures

1. **controlled_possession_recall** — Controlled-possession recall is 0.0%.
2. **possession_team_end_to_end** — End-to-end possession-team accuracy is 0.0% with 0.0% team coverage.
3. **events** — Event F1 is 0.0% at ±15 frames (0 TP, 0 FP, 0 FN).
4. **possession_state** — Six-state possession accuracy is 11.1%; the pipeline currently emits only controlled or loose for this benchmark.
5. **ball_detection** — Only 50.0% of visible balls have an observed post-filter detection; the rest depend on interpolation.
6. **ball_localization_50px** — Only 50.0% of visible balls are localized within 50 pixels after interpolation.

## Root-cause signals

- 3 controlled-possession misses are attributed to `ball_missing`.
- Team assignment is 0.0% correct when the pipeline actually identifies a holder; low end-to-end team accuracy is therefore a holder-coverage problem.
- Ball localization has a 3.9 px median but a 5.0 px p90, showing a severe wrong-object/interpolation tail.
- Shot event recall is 0.0%; the current event detector emits only passes and interceptions.

## Per-video frame metrics

| Video | Samples | Ball observed recall | Ball within 50 px | Controlled F1 | Team end-to-end |
|---|---:|---:|---:|---:|---:|
| video_1 | 9 | 50.0% | 50.0% | 0.0% | 0.0% |

## Event results

| Type | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|

## Interpretation limits

- Team IDs are arbitrary clusters, so the scorer selects the better of the two possible per-video cluster mappings.
- Event labels contain reviewed pipeline candidates plus explicitly identified misses; they are not yet a complete play-by-play inventory.
- The current possession model does not predict in-flight, shot, dead-ball, or unknown states, so six-state scoring exposes those as unsupported classes.
- Interpolated ball positions count toward localization but not observed-detection recall.
