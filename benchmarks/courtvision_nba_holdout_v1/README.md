# CourtVision NBA Holdout v1

This is a sealed, NBA-weighted evaluation set for ball localization, possession,
team association, and basketball-event interpretation.

## Composition

- 24 NBA clips from 24 distinct BARD validation games.
- 2 NCAA clips: one men's Division I and one women's Division I.
- 1 NBA G League clip.
- 1 FIBA Europe Cup clip.
- 1 Italian women's professional-league clip.
- 1 indoor fixed-camera 3x3 stress clip.

The 24 NBA clips are copied in full. The five BASKET clips are fixed 25-second
windows from long source videos. Their source paths and start times were selected
and recorded before CourtVision was run on any member of this holdout. The
fixed-camera TrackID3x3 source is copied in full.

## Sealing rules

1. Do not run CourtVision on these clips until the frame and event annotations
   have been frozen and independently reviewed.
2. Never use this set to select thresholds, models, features, or heuristics.
3. Never copy CourtVision predictions into the ground-truth annotations.
4. Report this holdout only as a whole and by the predefined cohorts in
   `source_selection.json`.
5. If a clip must be replaced, create a new benchmark version. Do not silently
   modify this version.
6. Keep analyzer outputs and cached stubs outside `holdout_videos/nba_sealed_v1`.

The existing files in `holdout_videos/` are development stress clips and are not
members of this sealed set because CourtVision has already processed them.

## Build

The BARD and TrackID3x3 source repositories/data are stored under
`holdout_sources/`, which is intentionally ignored by Git.

```powershell
.\.venv\Scripts\python.exe scripts\build_nba_holdout.py
.\.venv\Scripts\python.exe scripts\validate_nba_holdout.py
```

Use `--require-complete` with the validator for a release-gating check. A normal
validation allows access-gated source slots to remain pending while still
checking all materialized clips.

## Licensing

- BARD: CC BY 4.0. Its validation videos originate from official NBA play pages.
- BASKET: Apache-2.0 dataset card plus the access conditions displayed by the
  gated Hugging Face repository.
- TrackID3x3 dataset: CC BY 4.0.

The dataset licenses do not replace any separate rights that may be required for
commercial use of underlying broadcast footage. Confirm commercial video rights
before using this benchmark as a production-release gate.

## Annotation scope

Ground truth must eventually include:

- ball visibility and observed center;
- controlled possession, team, and holder;
- pass release and reception;
- shots and rebounds;
- steals, interceptions, deflections, turnovers, and loose balls;
- camera cuts, replays, graphics, and frames that should be treated as unknown.

BARD's clip-level event labels are preserved only as source metadata. They are
not a substitute for the frame-accurate CourtVision annotations.
