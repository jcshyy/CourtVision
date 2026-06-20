from pathlib import Path

from ultralytics import YOLO


class YoloDetector:
    """Shared wrapper for YOLO detection models."""

    def __init__(self, model_path: str | Path, target_class_names=None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.target_class_names = target_class_names or []
        self.target_class_ids = self._resolve_class_ids(self.target_class_names)

    def predict_frame(self, frame, confidence: float = 0.25):
        return self.model.predict(
            frame,
            conf=confidence,
            classes=self.target_class_ids or None,
        )[0]

    def predict_frames(self, frames, confidence: float = 0.25, batch_size: int = 20):
        results = []

        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            results.extend(
                self.model.predict(
                    batch,
                    conf=confidence,
                    classes=self.target_class_ids or None,
                )
            )

        return results

    def _resolve_class_ids(self, target_class_names):
        normalized_targets = {name.lower() for name in target_class_names}
        return [
            class_id
            for class_id, class_name in self.model.names.items()
            if class_name.lower() in normalized_targets
        ]
