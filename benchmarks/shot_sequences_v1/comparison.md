# Production shot-attempt benchmark

The production analyzer now answers one shot question: was a shot attempted?
It does not publish make, miss, rebound, dead-ball, or putback-subtype claims.

The baseline and current run use the same six locally available BARD clips,
containing nine ordered shot sequences. BARD supplies ordered clip-level
actions without timestamps, so this is per-clip shot-count F1 rather than
temporal localization F1.

| Metric | Baseline | Current |
| --- | ---: | ---: |
| True positives | 2 | 9 |
| Predicted attempts | 2 | 9 |
| Ground-truth attempts | 9 | 9 |
| Precision | 1.0000 | 1.0000 |
| Recall | 0.2222 | 1.0000 |
| **Shot-attempt F1** | **0.3636** | **1.0000** |

Ordinal shot coverage is complete for every style represented in this local
subset: one blocked shot, two close/midrange shots, four jump shots, one
layup, and one putback. The style names describe the source labels only; the
analyzer exports each result as the same plain `shot_attempt` event.

All 11 videos currently in `input_videos` were reprocessed. Their final
analysis manifests contain no `outcome` keys and no rebound or dead-ball
events. The other 13 clips in the 24-sequence annotation set are not included
in this measured result because their source videos are not present locally.

Raw production report: `shot_attempt_current6_report.json`.

The older `baseline_current6_report.json` and `tuned_current6_report.json`
remain as historical records from the retired outcome-resolution experiment.
