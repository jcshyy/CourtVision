from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"
MODELS_DIR = BACKEND_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output_videos"
STUBS_DIR = BACKEND_ROOT / "stubs"

PLAYER_DETECTOR_PATH = MODELS_DIR / "player_detector.pt"
EBARD_YOLO_DETECTOR_PATH = MODELS_DIR / "ebard_yolov8n.pt"
PLAYER_POSE_DETECTOR_PATH = MODELS_DIR / "yolo11n-pose.pt"
BALL_DETECTOR_PATH = MODELS_DIR / "ball_detector_model.pt"
WASB_BALL_DETECTOR_PATH = MODELS_DIR / "wasb_basketball_torchscript.pt"
COURT_KEYPOINT_DETECTOR_PATH = MODELS_DIR / "court_keypoint_detector.pt"
