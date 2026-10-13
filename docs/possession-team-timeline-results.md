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

## AI evidence package

`backend/app/trusted_evidence.py` (optional AI summaries) now also counts
`team_gap_bridged` frames toward a team. To keep its stricter standard, a
bridged gap counts only when qualifying holder segments (at least `MIN_SUPPORT`
frames) of the same team sit on both sides. Per-player segments are unchanged.
The package's possession algorithm label is `semantic_contiguous_team_gap_v3`.
Extended (`fused_extension`) frames are not yet counted there.

## Step 2 — combined-track extension (2026-10-06)

`build_team_possession` now takes the combined E-BARD + WASB holder sequence
(`extension_acquisition`). Where the E-BARD-only path has no holder, a run of
frames where the combined track holds the **same player** that the E-BARD-only
path confirmed in the adjacent frame becomes `possessionSource:
"fused_extension"`. It never starts a possession, never changes player or
team, and never crosses a scene cut or shot release. Events are unchanged.

| Measure | Step 1 | Step 2 |
| --- | ---: | ---: |
| Controlled frames with the correct team (of 121) | 70 | **79** |
| Wrong team on controlled frames | 0 | 0 |
| Team on `shot` frames (of 32) | 2 | 4 |
| Team on `dead` frames (of 6) | 1 | 1 |
| Team on `in_flight` frames (of 22) | 12 | 13 |
| Team on `loose` frames (of 13) | 7 | 7 |
| Pass/steal F1 | 0.7143 | 0.7143 |
| Shot-window F1 | 0.875 | 0.875 |

The two additional `shot` frames are `video_3` frames 240 and 242. By then the
shot clock has reset to :24 and players are scrambling for the rebound; the
labeller kept the `shot` state through the scramble, and the combined track
saw a player gain control. This is a label-convention mismatch rather than a
clear wrong call, but it breaches the "must not increase" guardrail as
written.

Bug fixed in this step: the shot-release boundary read a top-level
`release_frame` key that shot events do not have, so it fell back to the
rim-arrival `frame_index`. It now reads `evidence.release_frame`. Re-running
the replay after the fix produced identical results on these clips.

Remaining controlled-frame misses: 42, of which 22 involve a holder whose team
is unknown and 11 sit between holders on opposite teams.

## Step 3 — "ball seen, no holder" inspection (2026-10-10)

The plan expected 12–18 misses where the ball is seen but no holder is
assigned. Of the 42 remaining misses only about 9 carry a holder-model reason
of that kind (`candidate_building`, `candidate_switch_building`,
`ambiguous_candidates`), and none is `no_credible_candidate`, so the
distance-threshold change was not made.

Three presentation-layer rules were simulated on the Step 2 replay:

| Rule | Correct | Side effects | Outcome |
| --- | ---: | --- | --- |
| Back-fill the candidate frames before a holder is confirmed | +1 | +4 `shot`, +1 `dead` | Rejected |
| Keep the team while the candidate is a teammate | 0 | +1 `loose` | Rejected |
| Ignore an opposite-team candidate when the same player holds again within 0.2 s | +2 | none | Implemented |

Implemented rule: `TEAM_POSSESSION_FLICKER_SECONDS = 0.2`. A wider limit
(about 0.3 s) gains one more controlled frame but also names a team on a
`loose` frame and on 20 unlabelled frames.

Replay `same_holder_flicker` against Step 2:

| Measure | Step 2 | Step 3 |
| --- | ---: | ---: |
| Controlled frames with the correct team (of 121) | 79 | **81** |
| Wrong team on controlled frames | 0 | 0 |
| Team on `shot` / `dead` / `in_flight` / `loose` frames | 4 / 1 / 13 / 7 | 4 / 1 / 13 / 7 |
| Pass/steal F1 | 0.7143 | 0.7143 |
| Shot-window F1 | 0.875 | 0.875 |

Events and possession diagnostics are identical for all 11 clips; three
frames changed in total (`video_2` 140, `spursknicks2` 75, `spursknicksclip`
140).

Remaining controlled-frame misses: 40. 22 involve a holder whose team is
unknown (mostly `spursknicksclip` track 4), 11 sit between opposite-team
holders — 8 of them `video_3` frames 202–210, where the pipeline's shot
release (frame 201) precedes the labelled end of control — and 6 are clip
starts.

## Unknown-team holders — inspection (2026-10-10)

The 22 misses in the "a holder's team is unknown" bucket were checked with an
oracle: the four unknown holder tracks were forced to a team inside
`build_team_possession` only (replays `oracle_a`, `oracle_b`).

- **Ceiling is about +5 frames, not 22.** The best forcing gave 85 correct
  (from 81) with one wrong-team call; 17 of the 22 frames stayed missed. Those
  17 have no ball observation for long stretches (`ball_missing`), so no
  holder exists to give a team to.
- **The tracks are identity switches.** Per-frame jersey votes change in
  blocks over time: `spursknicksclip` track 4 votes team 1 up to about frame
  180 and team 2 from about frame 216; track 24 flips at frame 329; `video_2`
  tracks 1 and 3 flip similarly. A single team per track is wrong for part of
  each track, which is why the forced team produced a wrong call
  (`spursknicksclip` frame 285). Abstaining on these tracks is currently the
  correct behaviour.
- **Side effects of forcing:** +1 `shot`, +1 `dead`, +1 `loose` and +6
  `in_flight` frames with a team. Events were identical.

Of the 40 remaining misses, 32 have no observed ball in the pipeline; the
labeller marked the ball visible in 19 of those.

## Ball detection — inspection and confidence sweep (2026-10-10)

Stage of loss for the 25 missed controlled frames where the label marks the
ball visible (ball candidates captured per stage, 40 px match radius):

| Pipeline state | Frames |
| --- | ---: |
| Neither E-BARD nor WASB has a candidate at the ball | 5 |
| Only WASB has it | 12 |
| Both have it, no holder confirmed | 7 |
| Only E-BARD has it | 1 |

E-BARD single-frame recall on the 150 labelled visible-ball frames, by
confidence band, with detections away from the label over all 201 frames:

| Confidence | Ball near label | Detections elsewhere |
| --- | ---: | ---: |
| ≥ 0.25 (current) | 90 | 137 |
| 0.15–0.25 | +17 | +95 |
| 0.10–0.15 | +5 | +90 |
| 0.05–0.10 | +10 | +175 |

Sweep of the E-BARD ball confidence, replaying all 11 clips with cached WASB
candidates and player tracks (experiment harness only; no code change). The
0.25 run reproduced Step 3 exactly.

| Measure | 0.25 | 0.15 | 0.10 |
| --- | ---: | ---: | ---: |
| Controlled frames with the correct team (of 121) | 81 | 83 | 86 |
| Wrong team on controlled frames | 0 | 0 | 0 |
| Team on `shot` / `dead` frames | 4 / 1 | 4 / 1 | 5 / 1 |
| Team on `in_flight` / `loose` frames | 13 / 7 | 19 / 7 | 18 / 2 |
| Pass/steal F1 | 0.7143 | 0.7143 | 0.5 |
| Shot-window F1 | 0.875 | 0.875 | 0.75 |

- **0.10 is rejected:** both event scores fall.
- **0.15 is not adopted:** the scored F1s hold, but events change in 6 of 11
  clips. It adds a false shot at `video_3` frame 60 (the labelled event there
  is an interception, release 46, catch 82; shots on that clip are not in the
  scored shot windows) and drops a pass candidate in `bard_gsw_mem_21`. Across
  all frames it names a team on 268 more frames and 65 fewer, with no team
  flips.
- **Separate finding:** the pipeline's shot release in `video_3` is frame 201;
  the verified label is 219.

## Close-out check (2026-10-11)

Possession work is closed at 81 of 121 controlled frames with the correct
team and 0 wrong-team calls on the development clips.

Frozen MultiSports replay of the current code
(`runs/multisports-possession-closeout-2026-10-11`, the 12 pinned videos,
coverage 1.0) against the recorded baseline (`multisports-independent-v2`):

| Measure | Baseline | Current |
| --- | ---: | ---: |
| Micro F1 at 0.25 s | 0.4286 | 0.449 |
| Micro F1 at 0.5 s | 0.551 | 0.551 |
| Pass F1 at 0.25 s | 0.4 | 0.4364 |
| Shot-attempt F1 at 0.25 s | 0.4706 | 0.4706 |

The difference is one pass that now lands inside the 0.25 s tolerance (22 true
positives against 21). It comes from event-side changes made since the
baseline was recorded, such as the occlusion bridge; the team possession
timeline never feeds events. MultiSports scores events only, so it confirms
no event regression and says nothing about possession accuracy on new video.
