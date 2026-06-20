import pandas as pd
import supervision as sv
from ultralytics import YOLO

from backend.app.config import BALL_DETECTOR_PATH
from backend.app.utils import load_cache, save_cache


class BallTracker:
    """Detects and filters basketball positions using the reference repo flow."""

    def __init__(self, model_path=BALL_DETECTOR_PATH):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for start in range(0, len(frames), batch_size):
            detections.extend(
                self.model.predict(
                    frames[start : start + batch_size],
                    conf=0.5,
                )
            )

        return detections

    def get_object_tracks(self, frames, read_from_cache=False, cache_path=None):
        tracks = load_cache(cache_path, enabled=read_from_cache) if cache_path else None
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for detection in detections:
            class_names_inv = {value: key for key, value in detection.names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            frame_tracks = {}
            chosen_bbox = None
            max_confidence = 0

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                confidence = frame_detection[2]

                if class_id == class_names_inv["Ball"] and max_confidence < confidence:
                    chosen_bbox = bbox
                    max_confidence = confidence

            if chosen_bbox is not None:
                frame_tracks[1] = {"bbox": chosen_bbox}

            tracks.append(frame_tracks)

        if cache_path:
            save_cache(cache_path, tracks)

        return tracks

    def remove_wrong_detections(self, ball_positions):
        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for index in range(len(ball_positions)):
            current_box = ball_positions[index].get(1, {}).get("bbox", [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                last_good_frame_index = index
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get(
                "bbox", []
            )
            frame_gap = index - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            if _top_left_distance(last_good_box, current_box) > adjusted_max_distance:
                ball_positions[index] = {}
            else:
                last_good_frame_index = index

        return ball_positions

    def interpolate_positions(self, ball_positions):
        rows = [frame.get(1, {}).get("bbox", []) for frame in ball_positions]
        positions = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"])
        positions = positions.interpolate()
        positions = positions.bfill()

        return [{1: {"bbox": row}} for row in positions.to_numpy().tolist()]


def _top_left_distance(previous_box, current_box):
    return (
        (previous_box[0] - current_box[0]) ** 2
        + (previous_box[1] - current_box[1]) ** 2
    ) ** 0.5
