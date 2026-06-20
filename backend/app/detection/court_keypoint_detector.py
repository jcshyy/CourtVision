from pathlib import Path

from ultralytics import YOLO

from backend.app.config import COURT_KEYPOINT_DETECTOR_PATH
from backend.app.utils import load_cache, save_cache


class CourtKeypointDetector:
    """Runs a YOLO keypoint model against basketball court video frames."""

    def __init__(self, model_path: str | Path = COURT_KEYPOINT_DETECTOR_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Court keypoint model not found at {self.model_path}. "
                "Add court_keypoint_detector.pt to backend/models."
            )

        self.model = YOLO(str(self.model_path))

    def predict_frames(self, frames, confidence: float = 0.5, batch_size: int = 20):
        keypoints_per_frame = []

        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            results = self.model.predict(batch, conf=confidence)
            keypoints_per_frame.extend(result.keypoints for result in results)

        return keypoints_per_frame

    def get_court_keypoints(self, frames, read_from_cache=False, cache_path=None):
        court_keypoints = (
            load_cache(cache_path, enabled=read_from_cache) if cache_path else None
        )
        if court_keypoints is not None and len(court_keypoints) == len(frames):
            return court_keypoints

        court_keypoints = self.predict_frames(frames, confidence=0.5, batch_size=20)

        if cache_path:
            save_cache(cache_path, court_keypoints)

        return court_keypoints
