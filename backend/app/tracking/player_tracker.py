from collections import Counter

import supervision as sv
from ultralytics import YOLO

from backend.app.config import PLAYER_DETECTOR_PATH
from backend.app.utils import load_cache, save_cache


PLAYER_TRACKING_ALGORITHM_VERSION = "v5_repeated_referee_filter"
REFEREE_OVERLAP_IOU_THRESHOLD = 0.8
REFEREE_CONFIDENCE_TOLERANCE = 0.1
MINIMUM_REFEREE_OVERLAP_OBSERVATIONS = 3


class PlayerTracker:
    """Tracks players across video frames using the reference YOLO + ByteTrack flow."""

    def __init__(self, model_path=PLAYER_DETECTOR_PATH):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.cache_filename = (
            f"player_track_{PLAYER_TRACKING_ALGORITHM_VERSION}.pkl"
        )

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for start in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[start : start + batch_size],
                conf=0.5,
                verbose=False,
            )
            detections.extend(detections_batch)

        return detections

    def get_object_tracks(self, frames, read_from_cache=False, cache_path=None):
        tracks = load_cache(cache_path, enabled=read_from_cache) if cache_path else None
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        detections = self.detect_frames(frames)
        tracks = []
        player_observation_counts = Counter()
        referee_overlap_counts = Counter()

        for detection in detections:
            class_names_inv = {value: key for key, value in detection.names.items()}
            detections_sv = sv.Detections.from_ultralytics(detection)
            player_class_id = class_names_inv["Player"]
            referee_class_id = class_names_inv.get("Ref")
            player_detections = detections_sv[
                detections_sv.class_id == player_class_id
            ]
            referee_detections = detections_sv[
                detections_sv.class_id
                == (referee_class_id if referee_class_id is not None else -999)
            ]

            detections_with_tracks = self.tracker.update_with_detections(
                player_detections
            )
            frame_tracks = {}

            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                confidence = frame_detection[2]
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == player_class_id:
                    frame_tracks[track_id] = {"bbox": bbox}
                    player_observation_counts[track_id] += 1
                    if _is_duplicate_referee_detection(
                        bbox,
                        confidence,
                        referee_detections,
                    ):
                        referee_overlap_counts[track_id] += 1

            tracks.append(frame_tracks)

        referee_track_ids = {
            track_id
            for track_id, overlap_count in referee_overlap_counts.items()
            if _is_persistent_referee_track(
                overlap_count,
                player_observation_counts[track_id],
            )
        }
        if referee_track_ids:
            tracks = [
                {
                    track_id: track
                    for track_id, track in frame_tracks.items()
                    if track_id not in referee_track_ids
                }
                for frame_tracks in tracks
            ]

        if cache_path:
            save_cache(cache_path, tracks)

        return tracks

    def track_frames(self, frames):
        return self.get_object_tracks(frames)


def _is_duplicate_referee_detection(player_bbox, player_confidence, referee_detections):
    """Reject a Player box when the model also identifies the same person as Ref."""
    for referee_bbox, referee_confidence in zip(
        referee_detections.xyxy,
        referee_detections.confidence,
    ):
        if (
            _bbox_iou(player_bbox, referee_bbox) >= REFEREE_OVERLAP_IOU_THRESHOLD
            and referee_confidence
            >= player_confidence - REFEREE_CONFIDENCE_TOLERANCE
        ):
            return True
    return False


def _is_persistent_referee_track(overlap_count, observation_count):
    """Require repeated referee evidence before deleting an entire track."""
    if observation_count <= 0:
        return False
    return overlap_count >= MINIMUM_REFEREE_OVERLAP_OBSERVATIONS


def _bbox_iou(first, second):
    intersection_x1 = max(float(first[0]), float(second[0]))
    intersection_y1 = max(float(first[1]), float(second[1]))
    intersection_x2 = min(float(first[2]), float(second[2]))
    intersection_y2 = min(float(first[3]), float(second[3]))
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0,
        float(first[3]) - float(first[1]),
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0,
        float(second[3]) - float(second[1]),
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0
