import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import (
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    EBARD_YOLO_DETECTOR_PATH,
    PLAYER_DETECTOR_PATH,
    PLAYER_POSE_DETECTOR_PATH,
    WASB_BALL_DETECTOR_PATH,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Check CourtVision runtime assets.")
    parser.add_argument("--check-models", action="store_true")
    parser.add_argument(
        "--check-ebard",
        action="store_true",
        help="Compatibility flag; E-BARD is required by --check-models.",
    )
    parser.add_argument(
        "--check-legacy",
        action="store_true",
        help="Also require the former player and ball YOLO checkpoints.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when PyTorch cannot access an NVIDIA CUDA device.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    imports = {}
    for module_name in (
        "cv2",
        "numpy",
        "pandas",
        "supervision",
        "torch",
        "ultralytics",
    ):
        try:
            module = __import__(module_name)
            imports[module_name] = getattr(module, "__version__", "available")
        except ImportError as error:
            imports[module_name] = f"missing: {error}"

    model_paths = [
        PLAYER_DETECTOR_PATH,
        BALL_DETECTOR_PATH,
        COURT_KEYPOINT_DETECTOR_PATH,
        EBARD_YOLO_DETECTOR_PATH,
        PLAYER_POSE_DETECTOR_PATH,
        WASB_BALL_DETECTOR_PATH,
    ]
    models = {
        path.name: {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
        }
        for path in model_paths
    }
    accelerator = _accelerator_details()
    result = {
        "python": sys.version,
        "imports": imports,
        "accelerator": accelerator,
        "models": models,
    }
    print(json.dumps(result, indent=2))

    imports_ok = all(not value.startswith("missing:") for value in imports.values())
    required_models = {
        COURT_KEYPOINT_DETECTOR_PATH.name,
        EBARD_YOLO_DETECTOR_PATH.name,
        PLAYER_POSE_DETECTOR_PATH.name,
        WASB_BALL_DETECTOR_PATH.name,
    }
    if args.check_legacy:
        required_models.update(
            {PLAYER_DETECTOR_PATH.name, BALL_DETECTOR_PATH.name}
        )
    models_ok = all(models[name]["exists"] for name in required_models)
    if (
        not imports_ok
        or (args.check_models and not models_ok)
        or (args.require_cuda and not accelerator["cuda_available"])
    ):
        raise SystemExit(1)


def _accelerator_details():
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "device": "unavailable"}

    cuda_available = torch.cuda.is_available()
    return {
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
    }


if __name__ == "__main__":
    main()
