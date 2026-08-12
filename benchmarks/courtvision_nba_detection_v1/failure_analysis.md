# Ball Detector Failure Analysis

## Label audit correction

The source test split contains 88 physical `ball` boxes and 8 larger `ball-in-basket` regions. Every `ball-in-basket` region appears on a frame that also has a physical ball box and covers the ball/rim state region. It is not a second basketball. The benchmark and training export therefore use only the 88 `ball` boxes.

## Production threshold (`conf=0.25`)

| Metric | Result |
| --- | ---: |
| AP50 | 0.699 |
| mAP50:95 | 0.367 |
| Precision | 0.632 |
| Recall | 0.761 |
| F1 | 0.691 |
| True positives | 67 |
| False positives | 39 |
| False negatives | 21 |

All 88 physical balls are in COCO's small-object bucket at 640x640.

Miss review at IoU 0.50:

- 11: no plausible detector candidate (`conf>=0.001`, IoU below 0.10)
- 6: candidate exists but is poorly localized (best IoU from 0.10 to 0.49)
- 4: correctly localized candidate is below confidence 0.25

False-positive review:

- 37: background/person/crowd details with no meaningful overlap with the ball
- 2: localization or duplicate errors near the true ball

The contact sheets in `review/` draw ground truth in green and predictions in red.

## Validation-selected confidence

The published validation split selects `conf=0.50` for maximum raw-detector F1:

| Split / confidence | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation / 0.25 | 0.458 | 0.682 | 0.548 | 60 | 71 | 28 |
| Validation / 0.50 | 0.647 | 0.625 | 0.636 | 55 | 30 | 33 |
| Test / 0.25 | 0.632 | 0.761 | 0.691 | 67 | 39 | 21 |
| Test / 0.50 | 0.750 | 0.682 | 0.714 | 60 | 20 | 28 |

The validation-selected threshold removes 19 test false positives but also removes 7 true ball candidates. CourtVision's temporal tracker can reject candidates across time and needs high candidate recall, so the production tracking threshold should remain at 0.25 until a temporal benchmark proves that 0.50 improves complete tracks rather than only single-frame F1. The 0.50 result is useful for single-frame detection mode.

## Fine-tuning set

The generated one-class YOLO export contains:

- Train: 464 images, 432 ball boxes, 34 negative frames
- Validation: 96 images, 88 ball boxes, 8 negative frames
- Test: excluded from the generated training tree
- Exact filename overlap with test: 0
- Exact SHA256 image-content overlap with test: 0

Adjacent frames from the same game segments can still occur across published splits. This makes the data appropriate for targeted adaptation and regression testing, but not a broad generalization claim.

## Recommended experiment

Fine-tune the existing checkpoint for 30 epochs at 640px on a CUDA machine, choosing the final epoch and confidence using validation only. Then rescore the sealed test split and the existing temporal `courtvision_v1` clips before replacing the production checkpoint.

Local training was not started because this environment reports no CUDA device; CPU training of the current large checkpoint would be slow and would not be a useful controlled iteration.

## Adaptive 640px ROI inference

CourtVision now retains the 640px full-frame pass at confidence 0.25. Frames whose selected full-frame detection is missing or below 0.55 receive up to four additional 640px passes over:

- the velocity-projected ball location;
- confident player wrists from the existing pose model; and
- hoops detected by the existing ball-model checkpoint.

Each ROI is spatially gated around its trajectory, wrist, or rim anchor. Validation selected confidence 0.50 for adaptive detections while leaving full-frame detections at 0.25. Duplicate full-frame and ROI boxes are merged before the existing temporal candidate lattice runs.

Sealed test comparison at the production full-frame threshold:

| Metric | Full frame | Full + adaptive | Change |
| --- | ---: | ---: | ---: |
| AP50 | 0.699 | 0.725 | +0.025 |
| mAP50:95 | 0.367 | 0.389 | +0.022 |
| Precision | 0.632 | 0.636 | +0.004 |
| Recall | 0.761 | 0.795 | +0.034 |
| F1 | 0.691 | 0.707 | +0.016 |
| True positives | 67 | 70 | +3 |
| False positives | 39 | 40 | +1 |
| False negatives | 21 | 18 | -3 |
| Top-candidate accuracy | 0.693 | 0.750 | +0.057 |

The test set required 118 adaptive crops across 94 frames and produced 17 accepted adaptive candidates. The adaptive pass therefore averaged 1.26 additional model crops per frame rather than running four crops on every frame.
