import pandas as pd
import numpy as np
import supervision as sv
from ultralytics import YOLO

from backend.app.config import BALL_DETECTOR_PATH
from backend.app.utils import load_cache, save_cache

BALL_TRACKING_CACHE_VERSION = "v4_conservative_multisignal"


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

    def get_object_tracks(
        self,
        frames,
        read_from_cache=False,
        cache_path=None,
        player_tracks=None,
    ):
        tracks = load_cache(cache_path, enabled=read_from_cache) if cache_path else None
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        detections = self.detect_frames(frames)
        tracks = []

        previous_bbox = None
        for frame_index, detection in enumerate(detections):
            class_names_inv = {value: key for key, value in detection.names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            frame_tracks = {}
            ball_detections = []

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                confidence = frame_detection[2]

                if class_id == class_names_inv["Ball"]:
                    ball_detections.append(
                        {"bbox": bbox, "confidence": float(confidence)}
                    )

            chosen = _select_ball_detection(
                ball_detections,
                (
                    player_tracks[frame_index]
                    if player_tracks is not None
                    else {}
                ),
                previous_bbox,
            )

            if chosen is not None:
                frame_tracks[1] = {
                    "bbox": chosen["bbox"],
                    "confidence": chosen["confidence"],
                    "candidate_count": len(ball_detections),
                }
                previous_bbox = chosen["bbox"]

            tracks.append(frame_tracks)

        if cache_path:
            save_cache(cache_path, tracks)

        return tracks

    def remove_wrong_detections(self, ball_positions, player_tracks=None):
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

            camera_motion = _median_player_motion(
                player_tracks,
                last_good_frame_index,
                index,
            )
            if (
                _camera_adjusted_top_left_distance(
                    last_good_box,
                    current_box,
                    camera_motion,
                )
                > adjusted_max_distance
            ):
                ball_positions[index] = {}
            else:
                last_good_frame_index = index

        return ball_positions

    def interpolate_positions(self, ball_positions):
        rows = [frame.get(1, {}).get("bbox", []) for frame in ball_positions]
        observed = [bool(row) for row in rows]
        confidences = [frame.get(1, {}).get("confidence") for frame in ball_positions]
        positions = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"])
        positions = positions.interpolate()
        positions = positions.bfill()

        return [
            {1: {
                "bbox": row,
                "confidence": confidences[index],
                "interpolated": not observed[index],
            }}
            for index, row in enumerate(positions.to_numpy().tolist())
        ]


def _top_left_distance(previous_box, current_box):
    return (
        (previous_box[0] - current_box[0]) ** 2
        + (previous_box[1] - current_box[1]) ** 2
    ) ** 0.5


def _select_ball_detection(detections, players, previous_bbox):
    if not detections:
        return None
    highest_confidence = max(detections, key=lambda item: item["confidence"])
    if _minimum_bbox_distance(highest_confidence["bbox"], players) <= 80:
        return highest_confidence

    minimum_alternative_confidence = max(
        0.5,
        highest_confidence["confidence"] - 0.35,
    )
    supported = [
        detection
        for detection in detections
        if detection["confidence"] >= minimum_alternative_confidence
        and _minimum_bbox_distance(detection["bbox"], players) <= 50
    ]
    if not supported:
        return highest_confidence

    def fallback_score(detection):
        continuity = 0.5
        if previous_bbox is not None:
            continuity = max(
                0.0,
                1.0 - _bbox_center_distance(previous_bbox, detection["bbox"]) / 150.0,
            )
        return 0.75 * detection["confidence"] + 0.25 * continuity

    return max(supported, key=fallback_score)


def _minimum_bbox_distance(ball_bbox, players):
    if not players:
        return 80.0
    center_x = (ball_bbox[0] + ball_bbox[2]) / 2
    center_y = (ball_bbox[1] + ball_bbox[3]) / 2
    distances = []
    for player in players.values():
        bbox = player.get("bbox")
        if not bbox:
            continue
        closest_x = min(max(center_x, bbox[0]), bbox[2])
        closest_y = min(max(center_y, bbox[1]), bbox[3])
        distances.append(
            ((center_x - closest_x) ** 2 + (center_y - closest_y) ** 2) ** 0.5
        )
    return min(distances, default=80.0)


def _bbox_center_distance(first_bbox, second_bbox):
    first_center = (
        (first_bbox[0] + first_bbox[2]) / 2,
        (first_bbox[1] + first_bbox[3]) / 2,
    )
    second_center = (
        (second_bbox[0] + second_bbox[2]) / 2,
        (second_bbox[1] + second_bbox[3]) / 2,
    )
    return (
        (first_center[0] - second_center[0]) ** 2
        + (first_center[1] - second_center[1]) ** 2
    ) ** 0.5


def _camera_adjusted_top_left_distance(previous_box, current_box, camera_motion):
    dx = current_box[0] - previous_box[0] - camera_motion[0]
    dy = current_box[1] - previous_box[1] - camera_motion[1]
    return (dx**2 + dy**2) ** 0.5


def _median_player_motion(player_tracks, previous_index, current_index):
    if player_tracks is None:
        return 0.0, 0.0
    previous_players = player_tracks[previous_index]
    current_players = player_tracks[current_index]
    displacements = []
    for player_id in previous_players.keys() & current_players.keys():
        previous_bbox = previous_players[player_id].get("bbox")
        current_bbox = current_players[player_id].get("bbox")
        if not previous_bbox or not current_bbox:
            continue
        displacements.append(
            (
                current_bbox[0] - previous_bbox[0],
                current_bbox[1] - previous_bbox[1],
            )
        )
    if len(displacements) < 3:
        return 0.0, 0.0
    return tuple(np.median(np.asarray(displacements), axis=0).tolist())
