import supervision as sv
from ultralytics import YOLO

from backend.app.config import PLAYER_DETECTOR_PATH
from backend.app.utils import load_cache, save_cache


class PlayerTracker:
    """Tracks players across video frames using the reference YOLO + ByteTrack flow."""

    def __init__(self, model_path=PLAYER_DETECTOR_PATH):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for start in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[start : start + batch_size],
                conf=0.5,
            )
            detections.extend(detections_batch)

        return detections

    def get_object_tracks(self, frames, read_from_cache=False, cache_path=None):
        tracks = load_cache(cache_path, enabled=read_from_cache) if cache_path else None
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for detection in detections:
            class_names_inv = {value: key for key, value in detection.names.items()}
            detections_sv = sv.Detections.from_ultralytics(detection)
            detections_with_tracks = self.tracker.update_with_detections(detections_sv)
            frame_tracks = {}

            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inv["Player"]:
                    frame_tracks[track_id] = {"bbox": bbox}

            tracks.append(frame_tracks)

        if cache_path:
            save_cache(cache_path, tracks)

        return tracks

    def track_frames(self, frames):
        return self.get_object_tracks(frames)
