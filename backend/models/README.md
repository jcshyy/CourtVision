# Runtime model assets

Model weights in this directory are intentionally gitignored.

Install the default E-BARD YOLOv8n scene detector with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_ebard_model.py
```

`ebard_yolov8n.pt` is the E-BARD `BODD_yolov8n_0001.pt` checkpoint from
revision `3f4789c4431aa73269f60107a4ba0a5f86b7af8b`. It is distributed under
CC BY 4.0 by the E-BARD authors. The installer pins and verifies its SHA-256.

- Source: https://huggingface.co/GabrieleGiudici/E-BARD-detection-models
- License: https://creativecommons.org/licenses/by/4.0/

The default analyzer also requires `wasb_basketball_torchscript.pt`,
`yolo11n-pose.pt`, and `court_keypoint_detector.pt`. The legacy
`player_detector.pt` and `ball_detector_model.pt` files are loaded only when
the `current` scene backend is explicitly selected.
