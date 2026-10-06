# Team possession timeline — 2026-10-05

Implemented locally; not deployed. **These are calibration diagnostics on
development clips, not an independent accuracy estimate.**

## Definition change

A team keeps possession during its own passes, matching standard basketball
scoring. Holderless frames between two confirmed holders on the same team are
published as that team's possession (`possessionSource: "team_gap_bridged"`).

The gap stays unknown when any of these is true:
- the two holders are on opposite teams;
- either holder's team is unknown;
- the gap exceeds `--team-possession-max-gap-seconds` (default 1.5 s);
- the gap contains a scene cut;
- the gap contains a shot-attempt release;
- a ball candidate from the opposite team appears in the gap.

This is a presentation layer only. `build_team_possession`
(`backend/app/analytics/possession_timeline.py`) runs after final events and
never feeds back into them. Holder-level possession diagnostics, events and
`players[].isHolder` are unchanged. The video overlay (`TeamBallControlDrawer`)
reads the same team timeline as the manifest.

## Measured comparison

Replay of six BARD shot clips and five `courtvision_v1` clips with cached
detectors (`scripts/replay_possession_benchmark.py`). The labelled frames are
the 201 verified `courtvision_v1` samples.

| Measure | Before | 1.5 s | 2.5 s | 4.0 s |
| --- | ---: | ---: | ---: | ---: |
| Controlled frames with the correct team (of 121) | 47 | **70** | 70 | 70 |
| Wrong team on controlled frames | 0 | 0 | 0 | 0 |
| Team on `shot` frames (of 32) | 2 | 2 | 2 | 2 |
| Team on `dead` frames (of 6) | 0 | 1 | 1 | 1 |
| Team on `in_flight` frames (of 22) | 4 | 12 | 12 | 12 |
| Team on `loose` frames (of 13) | 1 | 7 | 7 | 7 |
| Pass/steal F1 (8 events) | 0.7143 | 0.7143 | 0.7143 | 0.7143 |
| Shot-window F1 (9 windows) | 0.875 | 0.875 | 0.875 | 0.875 |

- **Events are unchanged:** event lists and possession diagnostics are
  byte-identical to the baseline for all 11 clips at every limit. Because
  MultiSports scores events only, its micro F1 (0.4286) is necessarily
  unchanged and was not rerun.
- **The gap limit doesn't matter here:** all three limits gave identical
  results, so the default is the most conservative (1.5 s).
- **The `in_flight` frames are expected.** Under the new definition, a
  same-team pass carries the passer's team.
- **The `loose` frames are one scramble:** `video_3` frames 76–81, a loose ball
  in the paint retained by the same team. This is consistent with the
  definition.
- **The `dead` frame is a whistle edge case:** `spursknicksclip` frame 345,
  where the clock appears stopped. The bridge carried the team one frame into
  the stoppage.

## Remaining controlled-frame misses (51)

| Cause | Frames |
| --- | ---: |
| Different holders, one holder's team unknown | 27 |
| Different holders, opposite teams | 12 |
| No holder on one side (clip edge) | 6 |
| Same holder, ball seen in gap | 3 |
| Same holder, other | 3 |

In 16 of these 51 frames, the combined E-BARD + WASB holder track already has a
holder with a known team. That is the target of the next step: extending
confirmed possession from the combined track.

## Known inconsistency

`backend/app/trusted_evidence.py` (optional AI summaries) still counts
possession from holder-only diagnostics. Its possession percentages can
therefore differ from the review page, which reads `possessionTeamId`.
