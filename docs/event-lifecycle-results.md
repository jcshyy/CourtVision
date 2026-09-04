# Shot/pass lifecycle results — 2026-09-03

Implemented locally; not deployed. The [implementation plan](event-lifecycle-plan.md)
keeps E-BARD + WASB and changes temporal event interpretation, not model weights.

## What changed

- Passes/interceptions remain provisional until shot-flight arbitration. Later
  rebounds/outlets no longer delete earlier attempts; ID churn cannot turn one
  shot flight into a simultaneous pass/steal.
- Stable pre-rim catches preserve lobs. High passes and passes underneath the
  basket cannot qualify solely through 2D proximity to the hoop.
- Holder-grounded releases take priority; rim-seeded recovery handles missed
  holder transitions. Overlapping hypotheses, including a two-frame split seam,
  merge while established rebound control protects a new release.
- Shooter/team attribution abstains when nearby release evidence is inadequate.
  Internal states expose flight, unresolved post-shot play, control, and reasons
  for suppressed hypotheses. No made/missed basket or rebound outcome is claimed.
- Supported endline throw-ins are excluded from public pass totals. Missing
  geometry causes abstention. Event-only modes infer only needed court frames.
- Added temporal one-to-one scoring, wrong-class/duplicate diagnostics, coverage
  accounting, and an authorized-local-file MultiSports annotation adapter.

## Measured comparison

| Evaluation | Archived comparator | New replay |
| --- | ---: | ---: |
| Six BARD clips, nine coarse shot windows: F1 | 0.6667 | **0.8750** |
| Same shot windows: TP / FP / FN | 6 / 3 / 3 | **7 / 0 / 2** |
| Five local clips, eight possession events: F1 | 0.4000 | **0.7143** |
| Same possession events: TP / FP / FN | 3 / 4 / 5 | **5 / 1 / 3** |
| Legacy six-clip shot-count F1 | 1.0000 | **0.8750** |

Both temporal evaluations have 100% prediction-file coverage. The legacy count
metric scores only the six available inputs; its other 13 listed videos remain
missing and were excluded by that older evaluator. Do not call this full BARD
coverage. The new temporal scorer does penalize missing selected videos.

**These are calibration diagnostics, not an independent accuracy estimate.**
The shot windows were added during this implementation after inspecting source
frames; they describe coarse shooting motion, not exact release timestamps.
The older possession labels were candidate-reviewed and are not exhaustive.
Matching uses release time for passes/shots and catch time for interceptions,
with 0.25-second tolerance. Possession recall is now 0.6250 after authoritative
release localization; see the [v2 results](possession-continuity-v2-results.md).

The archived comparator is the explicit August 21
`output_videos/<video_id>_ebard_wasb_analysis.json`, not a fresh execution of
pre-change code. The candidate reruns current event logic on existing detector,
pose and team caches. Original FPS/resolution are retained (BARD clips: 60 FPS);
`--allow-uncertain-teams` matches the archived jobs and their `cfcdf3f38738`
assignment cache. Sparse court inference runs when its full cache is absent.
This is not a cold detector-speed test or a strict all-else-equal model A/B.
Machine-readable reports record prediction and current event-source SHA-256s.

## Why the old perfect count score was misleading

- `bard_gsw_mem_9`: the archived shot at release frame 118 was a pass; the real
  jumper is later. The replay localizes the shot release at frame 216.
- `bard_det_okc_24`: the archived second shot around frame 478 was post-rebound
  dribbling. Removing that false positive does not recover the actual putback
  around frames 340–385. The old total was right for the wrong reason.
- `bard_gsw_mem_21`: the replay retains the actual shot at release 328 instead
  of promoting an earlier high transfer; ordinary earlier passes remain.
- The five other inputs exposed high-pass and duplicate-rim-window regressions
  during development. They were corrected before the reported final replay.

The two remaining shot misses are the early blocked drive in
`bard_bkn_hou_366` (frames 55–90) and the putback in `bard_det_okc_24`
(340–385). Current tracks do not provide adequate shooting-height/hand evidence
for the first or sufficient pre-rim observed flight for the second. These need
targeted track/pose review and more labels, not relaxed gates solely to recover
the old count score.

## Verification

- Full unit suite: **368 run, 366 passed, 2 failed, 0 errors, 0 skipped**.
- The two failures are in unchanged web-demo files:
  `test_landing_exposes_honest_public_preview_entry` expects public preview to
  be enabled, and `test_public_demo_is_labeled_as_preprocessed_experimental_analysis`
  expects older page copy. No unrelated web behavior was changed.
- All event, lifecycle, temporal-scorer, pipeline and contract tests passed.
  New coverage includes ID switches, rebound/outlet arbitration, high lobs,
  pass chains, shot-window seams, quick putbacks, cuts, inbound abstention,
  evaluator matching/coverage, and restricted annotation loading.
- Eleven clean local source clips were replayed. The user's rendered
  `courtvision-93060089.mp4` was qualitative evidence, not detector input:
  burned-in overlays would contaminate a new detection run. A clean original
  and matching analysis JSON are still needed for an exact before/after of it.

## Independent benchmark validation

[MultiSports (ICCV 2021)](https://github.com/MCG-NJU/MultiSports) was downloaded
from its official gated distribution after the user accepted the license. A
deterministically preregistered 12/147-video released-validation pilot completed
at 100% coverage: primary 0.25-second event F1 **0.3918**, pass F1 **0.3396**,
and shot F1 **0.4571**. The 0.50-second sensitivity F1 is **0.5361**. This is a
custom temporal event metric, not official tube mAP, and the sample is not
guaranteed disjoint from upstream model training.

The exact shot-as-pass complaint did not appear in this pilot: no truth shot was
called a pass. The v2 replay reduced truth passes called shots from three to two.
Missed-shot intervals
had much lower internal ball-observation support than matched shots (31.17% vs
57.96%), but missed passes still had 57.56% visible-ball support. The resulting
decision is to keep the hybrid model while benchmarking ball alternatives on
ball-coordinate labels and improving possession continuity in parallel. See the
[independent validation report](multisports-independent-validation.md).

[FineSports (CVPR 2024)](https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024)
is an alternative, but requires a signed agreement emailed to the authors.
No third-party mirror, gated-access workaround or external message was used.
The new metric evaluates event timing/type; it is not official spatial tube mAP.

## Reproduce and next decision

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe scripts/run_event_regression.py tmp/event-lifecycle-final
.\.venv\Scripts\python.exe scripts/run_event_regression.py tmp/event-lifecycle-final --video-id video_1 --video-id video_2 --video-id video_3 --video-id spursknicksclip --video-id spursknicks2
.\.venv\Scripts\python.exe scripts/compare_event_regression.py tmp/event-lifecycle-final tmp/event-lifecycle-comparison
```

Saved compact reports: `benchmarks/event_sequences_v2/results/2026-09-03`.
Full candidate manifests: `tmp/event-lifecycle-final`. Public-data preparation
commands and limitations are in the [benchmark README](../benchmarks/event_sequences_v2/README.md).

Do not add a third ball detector on event F1 alone. The sealed MultiSports pilot
shows both missing ball observations and lifecycle failures. First label ball
coordinates/visibility on a separate basketball development set, benchmark the
hybrid against temporal alternatives, and improve possession continuity. Neither
model replacement nor production deployment is justified by this 12-video pilot.
The post-change replay improved primary micro F1 from 0.3918 to **0.4286** and
pass F1 from 0.3396 to **0.4000** without changing upstream detector caches.
