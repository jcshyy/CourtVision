# Possession continuity v2 results — 2026-09-03

## Outcome

The next event-logic change is implemented and improves both development and
independent benchmarks without replacing the E-BARD + WASB ball models.

| Evaluation | Before | After |
| --- | ---: | ---: |
| Local possession-event F1 (0.25 s) | 0.4000 | **0.7143** |
| Local possession precision / recall | 0.4286 / 0.3750 | **0.8333 / 0.6250** |
| Local shot-window F1 | 0.8750 | **0.8750** |
| MultiSports micro F1 (0.25 s) | 0.3918 | **0.4286** |
| MultiSports pass F1 (0.25 s) | 0.3396 | **0.4000** |
| MultiSports shot F1 (0.25 s) | 0.4571 | **0.4706** |
| MultiSports micro F1 (0.50 s) | 0.5361 | **0.5510** |

The local possession totals changed from 3 TP / 4 FP / 5 FN to
**5 TP / 1 FP / 3 FN**. On MultiSports, totals changed from
19 TP / 20 FP / 39 FN to **21 TP / 19 FP / 37 FN**. At the primary tolerance,
truth passes confused as shots decreased from three to two; matched passes
increased from nine to eleven.

## What changed

- A causal possession bridge may now retain up to two interpolated frames only
  when observed endpoints share a ball-track segment, the intermediate centers
  follow the endpoint trajectory, and every center remains near the same
  player. It cannot create a holder or bridge a competing takeover chain.
- Transfer release time now uses the last authoritative source-control frame.
  Provisional `switch_pending`, missing/interpolated-gap, retrospective, and
  offline-bridged frames no longer move the public release timestamp later.
- The previous segment tail remains available as `holder_tail_frame`, with an
  explicit `release_localization_reason`, so the adjustment is inspectable.
- The corrected release is consistently used for flight credibility, team
  lookup, pass/steal classification, and shot reconciliation. This rejected an
  early close-range takeover that was not the later annotated steal.

The main measured gain came from release localization rather than the new
two-frame bridge. In `video_2`, three reviewed pass releases now fall within the
0.25-second window instead of one. This is consistent with the earlier
MultiSports result, where doubling tolerance produced a large score increase.

## Evaluation discipline

Implementation selection and debugging used only the local calibration clips.
After the code and local result were finalized, MultiSports was replayed once
against the previously frozen 12-video sample, using the same detector, pose,
team, ball and court caches. No thresholds or logic were changed after viewing
that post-change result. The independent result therefore measures an
event-logic A/B on identical upstream model outputs, although the small sample
and previously viewed v1 results remain important limitations.

The full suite ran 368 tests: **366 passed and the same two unrelated web-demo
assertions failed**. Event-focused tests pass, including linear/nonlinear
two-frame gaps, authoritative/provisional release tails, lifecycle arbitration,
temporal matching, and duplicate/coverage accounting.

Machine-readable reports are in
`benchmarks/event_sequences_v2/results/2026-09-03` with the
`possession_continuity_v2_*` and `multisports_v2_*` prefixes.
