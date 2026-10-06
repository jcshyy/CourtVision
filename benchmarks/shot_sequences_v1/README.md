# CourtVision shot sequences v1

This benchmark contains 24 ordered shot sequences from 19 official BARD
validation clips. It is deliberately split by clip into 13 calibration and 11
holdout sequences so threshold tuning cannot see the holdout clip.

The source labels retain makes, misses, offensive/defensive rebounds, layups,
jump shots, blocked shots, and putbacks for future research. The production
evaluator intentionally scores only whether a shot was attempted. BARD supplies
ordered clip-level action labels rather than event timestamps, so the evaluator
reports per-clip shot-count F1 and does not present style coverage as temporal
F1.

The initial six-clip input subset and the tuned pipeline are compared in
`comparison.md`. The remaining 13 annotated clips are ready for future
holdout processing, but are not included in the reported F1 because their
source videos are not present in `input_videos`.

Regenerate the annotations:

```powershell
.\.venv\Scripts\python.exe scripts\build_shot_sequence_benchmark.py
```

Score an analysis directory:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_shot_sequences.py output_videos `
  --output benchmarks\shot_sequences_v1\report.json
```
