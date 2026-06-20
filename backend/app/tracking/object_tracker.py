from pathlib import Path

from ultralytics import YOLO


class ObjectTracker:
    """Runs YOLO tracking with a configurable tracker backend."""

    def __init__(self, model_path: str | Path, target_class_names=None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Tracking model not found at {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.target_class_names = target_class_names or []
        self.target_class_ids = self._resolve_class_ids(self.target_class_names)

    def track_video(
        self,
        video_path: str | Path,
        confidence: float = 0.25,
        tracker_config: str = "bytetrack.yaml",
        save: bool = False,
    ):
        return self.model.track(
            source=str(video_path),
            conf=confidence,
            classes=self.target_class_ids or None,
            tracker=tracker_config,
            persist=True,
            save=save,
        )

    def track_frames(
        self,
        frames,
        confidence: float = 0.25,
        tracker_config: str = "bytetrack.yaml",
    ):
        tracked_results = []

        for frame in frames:
            result = self.model.track(
                source=frame,
                conf=confidence,
                classes=self.target_class_ids or None,
                tracker=tracker_config,
                persist=True,
                verbose=False,
            )[0]
            tracked_results.append(result)

        return tracked_results

    def _resolve_class_ids(self, target_class_names):
        normalized_targets = {name.lower() for name in target_class_names}
        return [
            class_id
            for class_id, class_name in self.model.names.items()
            if class_name.lower() in normalized_targets
        ]
