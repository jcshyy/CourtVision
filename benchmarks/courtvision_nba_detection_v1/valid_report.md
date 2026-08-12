# NBA Ball Detection Baseline

- Model: `backend\models\ball_detector_model.pt`
- Input size: 640
- Test images: 96
- Ball boxes: 88
- Training overlap status: **unknown**

| Metric | Value |
| --- | ---: |
| AP50 | 0.639 |
| mAP50:95 | 0.365 |
| Precision @ 0.25 | 0.458 |
| Recall @ 0.25 | 0.682 |
| F1 @ 0.25 | 0.548 |
| Top-candidate accuracy | 0.625 |
| Mean center error | 1.00 px |
| Negative-frame false positives | 8 |

## Recall by object size

- small: 0.682 (88 boxes)

> This is a correlated-frame regression set from a small number of game segments. It is not a broad generalization estimate. Exact overlap with the checkpoint's original training images has not yet been audited.
