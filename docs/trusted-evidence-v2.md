# Trusted Evidence Package v2

`backend.app.trusted_evidence.build_trusted_evidence_v2(analysis)` is a pure,
provider-independent transformation. It does not mutate the manifest, read secrets,
call OpenAI, or use wall-clock time. `validate_trusted_evidence_v2` validates its
output. The canonical schema is `SCHEMA` in that module; the adjacent JSON Schema
is its portable export. Version is `2.0`; the enclosing analysis manifest retains
version 1 for existing consumers. `gameSummary.evidence` is now the v2 object.

## Contract

| Field | Meaning |
| --- | --- |
| `coverage` | Duration, source/sample/usable counts, usable ratio, known/unknown team-possession frames, calibrated frames and ratio, known/unknown team observations and known ratio, omitted raw event count. Usable means a sampled frame containing track observations, not a certified image-quality score. |
| `possession` | Explicit eligible-frame denominator, Team 1/2 counts and shares, unknown count/share, stable contiguous holder segments with opaque track/team references and supporting counts. Shares are fractions, not percentages. |
| `tracks` | Opaque IDs, conservative team assignment, observed duration, continuity ratio and run count, calibrated observations, measurement quality/reasons, optional measured distance/time, average/p95/sustained-peak speed and peak time. |
| `events` | Content-derived stable IDs, type, candidate status, timestamp/frame, bounded replay window, optional release/catch frames, source/destination track/team IDs, support count and machine-readable uncertainty reasons. |
| `sequences` | Stable references to possession segments and events at release/catch boundaries. Relation is `temporal_adjacency_only`, never causation. |
| `provenance` | Schema/algorithm versions, FPS, seconds/meters/meters-per-second units, thresholds, calibration status and verification policy. |
| `supportedClaims`, `summaryOptions`, `limitations` | Deterministic explanations available to the model. Each supported claim has its exact evidence references. |

## Input and measurement rules

The manifest writer now exports `evidenceFrames` from existing player tracks, so
observation coverage survives event-only processing without court projection.
Existing `frames` continue to drive the tactical UI. Older manifests fall back to
`frames`; absent calibration and diagnostic metadata never become inferred metrics.
`distanceMeters` is an existing pipeline interval distance, not a recomputation of
court geometry. Raw scene and tactical discontinuities are exported together.
Fallback homographies and unavailable projection frames are excluded from calibrated
measurements. Calibration here is pipeline geometry availability, not an accuracy
certification.

Possession uses the semantic timeline. The eligible denominator contains sampled
frames with explicit possession diagnostics, including unknown and loose states.
Missing diagnostic frames are excluded from that denominator but remain unknown in
whole-clip coverage. Provisional holder carries, control runs shorter than three
consecutive frames, absent tracks and unknown teams do not contribute to team shares.
Runs break at missing frames, holder/team changes and discontinuities.

Movement only includes consecutive, calibrated track intervals without a cut.
Negative/non-finite distances and speeds above 12 m/s are excluded. At least 0.5
seconds of accepted measurement is required. Distance is the sum of accepted
pipeline intervals; average speed uses accepted elapsed time; p95 uses nearest rank
on accepted interval speeds. Sustained peak uses the maximum mean over a contiguous
window of at least 0.5 seconds, with an end-of-window timestamp (earliest on ties).
It is not the pipeline's displacement-based overlay speed. No distance is added
across a gap, and short windows never become zero-speed observations. Partial
measurements are never described as total clip distance.

## Validation and safety

The dependency-free structural validator enforces the exported JSON Schema subset:
object fields, required fields, no extra fields, arrays, types, enums, finite numbers,
and ranges. Semantic validation checks clip bounds, duplicate IDs, track/event/sequence
references, denominator identities, shares, coverage ratios, speed ceilings,
measurement support, replay bounds, event timestamps and contradiction rules.

Malformed source dimensions and frame IDs fail validation. Unsupported/out-of-bounds
events, reversed release/catch boundaries, conflicting team semantics, track/team
contradictions and conflicting descriptions of a release are excluded. Duplicates
are removed, events are sorted deterministically, and at most 200 are included.
Omissions are counted and disclosed. Missing actor references become null with
`missing_track_reference`; insufficient event support remains visibly uncertain.

Current uncertainty codes: `verification_gate_not_available`, `insufficient_support`,
`missing_track_reference`, `unknown_team_assignment`, `tracking_discontinuity`,
`team_assignment_inconsistent`, `speed_outlier_filtered`, and
`insufficient_calibrated_measurements`.

There is no calibrated event verification gate in the current pipeline. The schema
therefore accepts only `candidate`; incoming `verified` labels cannot promote an
event. A future verified tier needs a versioned gate and explicit tests. Court zones
are omitted because orientation/zone confidence is not established. Confidence tiers,
identities, scores, winners, shot outcomes, formations, schemes and causes are omitted.

The OpenAI output schema requires `claim`, `evidenceIds`, and nullable `caveat` for
each insight. The model selects/orders exact supplied explanations. Runtime validation
rejects rewritten claims, unrelated but valid evidence IDs, invented numbers, unsupported
overall summaries and caveats. This intentionally limits prose flexibility to make
unsupported-claim prevention enforceable; regex-only semantic checking is insufficient.
Provider failures remain isolated from video processing. No new API route is required:
the existing batch enrichment and manifest delivery carry the v2 report.

## UI and compatibility

The AI clip summary sits in normal document flow below the replay workspace, with a
persistent summary jump link. It displays overall summary, referenced findings,
coverage and limitations. Replay buttons seek to the supplied pre-event window and
update the video, timeline and tactical court together. Keyboard buttons have 44px
minimum height, wrapping labels and existing focus behavior. Desktop/mobile layouts
retain separate video, court, header actions and event rundown. Legacy v1 reports
still render. Disabled/missing summaries stay hidden; failures show a recovery message.

## Verification

- Full suite: `.venv/Scripts/python -m unittest discover -s tests -q`.
- Focused v2 tests: `.venv/Scripts/python -m unittest discover -s tests -p test_trusted_evidence.py -v`.
- JS syntax: `node --check web/app.js` (static vanilla JS; no frontend compilation step).
- Python syntax: `.venv/Scripts/python -m compileall -q backend/app/trusted_evidence.py backend/app/game_summary.py main.py`.
- Ruff on touched Python files; mypy on the two summary/evidence modules with
  `--follow-imports=skip --ignore-missing-imports` (untyped bodies are outside this static check;
  runtime schema tests validate the dynamic JSON contract).
- Layout detector: `node .agents/skills/impeccable/scripts/detect.mjs --json --scope layout web/app.js web/styles.css`.
- Browser fixture: `node tmp/evidence-v2-ui-check.cjs`, desktop 1440x1000 and mobile
  390x844; screenshots `tmp/evidence-v2-desktop.png` and `tmp/evidence-v2-mobile.png`.
- Build: `docker build -f Dockerfile.api -t courtvision-api:evidence-v2 .`.
- Whitespace review: `git diff --check`.

No live paid model call, model-weight inference run, push or deployment was performed.
The existing pipeline's tracking/geometry accuracy remains a measurement limitation;
this package validates consistency and support, not basketball ground truth.

Final verification results: 396 full-suite tests passed; 15 focused v2 tests passed.
Ruff passed on all six touched Python files. Scoped mypy passed on both evidence/
summary modules. JavaScript/Python syntax, desktop/mobile browser checks, layout
scan (zero findings), whitespace review and the final local API Docker build passed.

Files changed for this implementation: `backend/app/trusted_evidence.py`,
`backend/app/game_summary.py`, `main.py`, `tests/test_trusted_evidence.py`,
`tests/test_game_summary.py`, `tests/test_web_demo.py`, `web/app.js`,
`web/styles.css`, `docs/trusted-evidence-v2.md`, and
`docs/trusted-evidence-v2.schema.json`. Temporary check tools, fixtures, screenshots
and logs are under `tmp/`. The working tree had extensive pre-existing changes;
those were preserved. Import ordering was normalized in the touched Python files.

## Replay-first summaries

The `replay_review_v1_gap_5s` explanation algorithm replaces frame-count and speed
recitations with concise action reviews. The model selects at most three moments.
Each moment includes a candidate action, readable timestamp, available team/opaque
track context, and a replay start cue. Track IDs appear as `T11`, never as identities.
The overall summary introduces the review without repeating the full moment text.
Frame counts and measurement details remain in the package; the UI places coverage
and limitations in a closed, keyboard-accessible disclosure.

A pass-to-shot narrative requires adjacent event candidates no more than five seconds
apart, at least three pass-support frames, a known matching team, the pass receiver
matching the shot's source track, and a single contiguous possession segment from
catch through shot release. The segment must have full per-frame support. Tracking
cuts and unknown control split segments and prevent this link. The sequence references
both event IDs and the supporting possession segment, and validation rechecks the
relationship. Wording is "followed by", never "caused", "led to", or a completed pass.
Nearby-player counts, defensive pressure and shot outcomes are still not inferred.

The current sample explanation is: "Review the shot-attempt candidate near 3.3s,
associated with Team 1 track T11. Start the replay at 1.3s to inspect the buildup and
release. The shot outcome is not established." The deterministic preview is saved
separately from the previous live AI result; regenerate the summary to use the new
wording. No new provider call is needed to build or inspect that preview.
