# CourtVision benchmark v1

This benchmark is an annotation-ready, versioned evaluation set for the five
videos in `input_videos/`. It keeps human ground truth in `annotations.jsonl`
and pipeline output in `suggestions.jsonl`; suggestions must never be counted
as verified labels.

## Build the review set

```powershell
.\.venv\Scripts\python.exe scripts\build_labeled_benchmark.py
```

This extracts regular samples every 15 frames plus dense samples around known
event candidates. It also records video hashes and metadata in
`dataset.json`. Existing verified annotations are preserved.

## Annotate

```powershell
.\.venv\Scripts\python.exe scripts\annotate_benchmark.py
```

Controls:

- Left click: mark the visible ball center.
- `v`: visible ball, keeping the clicked center.
- `o`: occluded ball.
- `x`: ball out of frame.
- `u`: uncertain ball visibility.
- `1`: Team A controls the ball.
- `2`: Team B controls the ball.
- `l`: loose ball.
- `f`: ball in flight.
- `s`: shot in progress.
- `d`: dead ball.
- `n`: possession unknown.
- `Enter`: mark the record verified and advance.
- `b`: go back.
- `q`: save and quit.

Team names and jersey descriptions appear in `manifest.json`. A visible ball
requires a center coordinate before a record can be verified.

## Validate

```powershell
.\.venv\Scripts\python.exe scripts\validate_benchmark.py
```

Validation fails for malformed records, invalid coordinates, unknown label
values, missing images, or a visible verified ball without a center.

## Score the cached pipeline

```powershell
.\.venv\Scripts\python.exe scripts\score_benchmark.py
```

The scorer recreates the current post-filter ball track, holder timeline, team
control, and pass/interception events from each video's cached pipeline
artifacts. It writes `baseline_report.json` for machine-readable regression
tracking and `baseline_report.md` for review.

Ball metrics separate observed detections from interpolated positions. Team
accuracy uses the better of the two possible per-video cluster-ID mappings,
because automatically discovered cluster numbers have no fixed semantic order.
Events are matched one-to-one by video and type, with a default tolerance of 15
frames; use `--event-tolerance` to change it.

## Review event candidates

```powershell
.\.venv\Scripts\python.exe scripts\annotate_events.py
```

The event reviewer opens the source video at the first pending candidate.

- `a` / `d`: previous / next video frame.
- `r`: set the release frame; `Shift+R`: clear it.
- `c`: set the catch or secured-control frame; `Shift+C`: clear it.
- `p`: pass; `t`: steal; `i`: interception.
- `o`: offensive rebound; `e`: defensive rebound.
- `s`: shot; `f`: deflection; `k`: dead ball; `u`: unknown change.
- `1` / `2`: set the originating team to Team A / Team B.
- `3` / `4`: set the receiving team to Team A / Team B.
- `0`: clear both teams.
- `v`: verify and advance; `j`: reject and advance. Both skip resolved events.
- `[` / `]`: previous / next candidate; `q`: save and quit.

The header shows semantic team names for the current video. Candidate rejection
is appropriate when the proposed event did not happen. This tool reviews the
nine proposed candidates; events missed entirely by the pipeline must still be
added separately for a recall-complete event benchmark.

Selecting `shot` clears catch and receiving-team fields. Selecting a rebound
clears release and originating-team fields. This prevents values left over from
a previous event type from being saved as verified ground truth.

## Label definitions

Ball visibility is one of `visible`, `occluded`, `out_of_frame`, or
`uncertain`. Possession state is one of `controlled`, `loose`, `in_flight`,
`shot`, `dead`, or `unknown`. A controlled state must name `team_a` or
`team_b`; other states must not claim a controlling team.

Event annotations live in `events.jsonl` and use `pass`, `steal`,
`interception`, `offensive_rebound`, `defensive_rebound`, `shot`, `deflection`,
`dead_ball`, or `unknown_change`. Release/catch frame fields may be null until
verified.
