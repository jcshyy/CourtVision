from backend.app.config import PLAYER_DETECTOR_PATH
from backend.app.detection.yolo_detector import YoloDetector


class PlayerDetector(YoloDetector):
    """Detects basketball players in frames."""

    def __init__(self, model_path=PLAYER_DETECTOR_PATH):
        super().__init__(
            model_path,
            target_class_names=["player", "players"],
        )
