from backend.app.config import BALL_DETECTOR_PATH
from backend.app.detection.yolo_detector import YoloDetector


class BallDetector(YoloDetector):
    """Detects the basketball in frames."""

    def __init__(self, model_path=BALL_DETECTOR_PATH):
        super().__init__(
            model_path,
            target_class_names=["ball", "basketball"],
        )
