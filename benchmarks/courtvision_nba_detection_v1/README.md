# CourtVision NBA Detection Benchmark v1

This is a separate, image-level regression benchmark for the raw CourtVision ball detector. It does not extend `courtvision_v1`'s temporal ball tracks, possession labels, or event labels.

## Data

- Source: [Roboflow Universe basketball-player-detection-3 v6](https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo/dataset/6)
- License: CC BY 4.0
- Published test split only: 94 images at 640x640
- Ball ground truth: 88 physical `ball` boxes. The 8 larger `ball-in-basket` regions are excluded because they overlap a physical ball label and describe basket state rather than a second ball.
- The downloaded images are installed under `data/` and intentionally gitignored. Source metadata and integrity hashes live in `manifest.json`.

The source contains nearby frames from a small number of NBA game segments. Its test split is useful for repeatable detector regression checks, but correlated frames mean it is not an independent estimate of performance across teams, arenas, broadcasts, or seasons.

The current detector checkpoint reports training on `Basketball-Players-17`. Those original training images are not present locally, so exact and perceptual overlap with this Roboflow dataset is not yet known. Until that audit is possible, results must be marked **overlap status unknown**, not contamination-free.

## Install and validate

Download the COCO archive from the source page and extract it so the test annotation file is:

```text
benchmarks/courtvision_nba_detection_v1/data/test/_annotations.coco.json
```

Then run:

```powershell
.\.venv\Scripts\python.exe scripts\validate_nba_detection_benchmark.py
```

The validator checks the source annotation hash, expected image and target-box counts, image dimensions, bounding-box bounds, and presence of every test image.

## Score the current detector

```powershell
.\.venv\Scripts\python.exe scripts\score_nba_detection_benchmark.py
```

The scorer uses the production checkpoint, `imgsz=640`, and the model's `Ball` class. It reports AP50, AP50:95, precision/recall/F1 at confidence 0.25, matched center error, recall by COCO object-size bucket, negative-frame false positives, and top-candidate accuracy. Results are written to `baseline_report.json` and `baseline_report.md` by default.

It also writes `error_analysis_test.json`. Render its per-frame misses and false positives with:

```powershell
.\.venv\Scripts\python.exe scripts\render_nba_detection_errors.py
```

Tune the detector confidence on the published validation split, never on test:

```powershell
.\.venv\Scripts\python.exe scripts\score_nba_detection_benchmark.py --split valid
```

## Prepare targeted fine-tuning data

The builder exports only the physical `ball` class from the published train and validation splits, retains negative frames, and verifies that no test filename enters the training tree. `ball-in-basket` regions are deliberately excluded because those frames already contain a physical ball box:

```powershell
.\.venv\Scripts\python.exe scripts\build_nba_ball_training_set.py
```

The generated `training_yolo/` directory is gitignored. On a CUDA machine, a conservative first experiment is:

```powershell
.\.venv\Scripts\yolo.exe detect train model=backend/models/ball_detector_model.pt data=benchmarks/courtvision_nba_detection_v1/training_dataset.yaml imgsz=640 epochs=30 batch=16 device=0 project=runs name=nba_ball_ft_v1
```

Choose epochs and confidence thresholds using validation results only. Score the 94 test images only after the experiment is fixed.

## Score adaptive inference

The adaptive scorer preserves chronological order within each game segment and runs the production player, pose, hoop, and ball models:

```powershell
.\.venv\Scripts\python.exe scripts\score_nba_adaptive_detector.py --split valid
.\.venv\Scripts\python.exe scripts\score_nba_adaptive_detector.py --split test
```

The production policy keeps full-frame confidence at 0.25 and requires 0.50 confidence from spatially gated adaptive crops. Results are saved in `adaptive_valid_report.json` and `adaptive_report.json`.
