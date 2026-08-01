# Holdout annotation calibration

This directory contains a first-pass, human-reviewable annotation draft for
three calibration clips. The draft was created from the sealed source videos
only. CourtVision predictions, caches, and analyzer outputs were not used.

The calibration clips are:

- `nba_015`: NBA broadcast, assisted three-point shot, and an inbound sequence.
- `ncaa_m_001`: several source-game excerpts and hard broadcast cuts.
- `fixed_001`: wide fixed-camera 3x3 footage with a small ball.

`frames.jsonl` samples every 15 frames. `events.jsonl` records the visible
event sequence and cut boundaries. Frame records remain `draft`; event records
verified by the second reviewer are marked `verified`, while provisional
release/catch boundaries remain `draft`.

Important conventions:

- Ball centers use source-video pixel coordinates.
- `uncertain` is used instead of inventing a center when the ball cannot be
  located confidently in the raw frame.
- Team IDs are scoped to a `scene_id`. They do not carry across a hard cut.
- `controlled` possession requires a scene-local team ID.
- `shot`, `in_flight`, `loose`, `dead`, and `unknown` do not claim a team.
- Event confidence is annotation-review confidence, not model confidence.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\validate_nba_holdout_annotations.py
.\.venv\Scripts\python.exe scripts\render_nba_holdout_annotation_review.py
```

The rendered review sheets are written under the ignored
`holdout_sources/previews/nba_holdout_annotation_review/` directory.
Use `REVIEW_CHECKLIST.md` to verify the event boundaries and the low-confidence
fixed-camera team grouping before the remaining holdout clips are labeled.
