# NBA Ball Detection Baseline

- Model: `backend\models\ball_detector_model.pt`
- Input size: 640
- Test images: 94
- Ball boxes: 88
- Training overlap status: **unknown**

| Metric | Value |
| --- | ---: |
| AP50 | 0.699 |
| mAP50:95 | 0.367 |
| Precision @ 0.50 | 0.750 |
| Recall @ 0.50 | 0.682 |
| F1 @ 0.50 | 0.714 |
| Top-candidate accuracy | 0.636 |
| Mean center error | 0.91 px |
| Negative-frame false positives | 2 |

## Recall by object size

- small: 0.682 (88 boxes)

> This is a correlated-frame regression set from a small number of game segments. It is not a broad generalization estimate. Exact overlap with the checkpoint's original training images has not yet been audited.
