from copy import deepcopy
from pathlib import Path

import cv2
from ultralytics import YOLO

from backend.app.config import PLAYER_POSE_DETECTOR_PATH
from backend.app.utils import load_cache, save_cache


PLAYER_POSE_CACHE_VERSION = "v4_normalized_candidate_crops"


class PlayerPoseDetector:
    """Caches COCO person poses and associates them with tracked players."""

    def __init__(self, model_path=PLAYER_POSE_DETECTOR_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Player pose model not found at {self.model_path}. "
                "Add yolo11n-pose.pt to backend/models."
            )
        self.model = YOLO(str(self.model_path))
        self.cache_filename = f"player_pose_{PLAYER_POSE_CACHE_VERSION}.pkl"

    def detect_frames(self, frames):
        detections = []
        for start in range(0, len(frames), 20):
            detections.extend(self.model.predict(
                frames[start : start + 20],
                conf=0.25,
                imgsz=640,
                classes=[0],
                verbose=False,
            ))
        return detections

    def get_player_poses(
        self,
        frames,
        player_tracks,
        *,
        ball_tracks=None,
        read_from_cache=False,
        cache_path=None,
    ):
        if len(frames) != len(player_tracks):
            raise ValueError("Video frames and player tracks must align")
        if ball_tracks is not None and len(frames) != len(ball_tracks):
            raise ValueError("Video frames and ball tracks must align")
        cached = (
            load_cache(cache_path, enabled=read_from_cache)
            if cache_path
            else None
        )
        if cached is not None and len(cached) == len(frames):
            return cached

        if ball_tracks is None:
            results = self.detect_frames(frames)
            poses = [
                _match_player_poses(frame_tracks, _pose_detections(result))
                for frame_tracks, result in zip(player_tracks, results)
            ]
        else:
            requests, crops = _candidate_player_crops(
                frames,
                player_tracks,
                ball_tracks,
            )
            results = self.detect_frames(crops)
            poses = [{} for _ in frames]
            for request, result in zip(requests, results):
                pose = _select_crop_pose(
                    _pose_detections(result),
                    request["local_player_bbox"],
                )
                if pose is None:
                    continue
                poses[request["frame_index"]][request["player_id"]] = (
                    _translate_pose(
                        pose,
                        request["origin"],
                        request["scale"],
                        request["padding"],
                    )
                )
        if cache_path:
            save_cache(cache_path, poses)
        return poses


def attach_player_poses(player_tracks, pose_frames):
    """Return player tracks enriched with matched pose data."""
    if len(player_tracks) != len(pose_frames):
        raise ValueError("Player tracks and pose frames must align")
    enriched = deepcopy(player_tracks)
    for frame_tracks, frame_poses in zip(enriched, pose_frames):
        for player_id, pose in frame_poses.items():
            if player_id in frame_tracks:
                frame_tracks[player_id]["pose"] = deepcopy(pose)
    return enriched


def _pose_detections(result):
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or keypoints.xy is None:
        return []
    xy = keypoints.xy.cpu().tolist()
    confidence_tensor = getattr(keypoints, "conf", None)
    confidence = (
        confidence_tensor.cpu().tolist()
        if confidence_tensor is not None
        else [[1.0] * len(points) for points in xy]
    )
    return [
        {
            "bbox": list(bbox),
            "confidence": float(box_confidence),
            "keypoints_xy": points,
            "keypoint_confidences": point_confidence,
        }
        for bbox, box_confidence, points, point_confidence in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            xy,
            confidence,
        )
    ]


def _candidate_player_crops(frames, player_tracks, ball_tracks):
    requests = []
    crops = []
    for frame_index, (frame, players, ball_frame) in enumerate(
        zip(frames, player_tracks, ball_tracks)
    ):
        candidates = ball_frame.get(1, {}).get("candidates")
        if candidates is None:
            bbox = ball_frame.get(1, {}).get("bbox")
            candidates = [{"bbox": bbox}] if bbox else []
        candidate_boxes = [
            candidate.get("bbox")
            for candidate in candidates
            if candidate.get("bbox")
        ]
        if not candidate_boxes:
            continue
        frame_height, frame_width = frame.shape[:2]
        for player_id, player in players.items():
            player_bbox = player.get("bbox")
            if not player_bbox:
                continue
            player_height = float(player_bbox[3]) - float(player_bbox[1])
            if player_height <= 0 or not any(
                _bbox_to_bbox_distance(candidate_bbox, player_bbox)
                <= 0.35 * player_height
                for candidate_bbox in candidate_boxes
            ):
                continue
            crop_bbox = _expanded_crop_bbox(
                player_bbox,
                frame_width,
                frame_height,
            )
            x1, y1, x2, y2 = crop_bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pose_crop, scale, padding = _normalize_pose_crop(crop)
            scale_x, scale_y = scale
            padding_x, padding_y = padding
            requests.append({
                "frame_index": frame_index,
                "player_id": player_id,
                "origin": (x1, y1),
                "scale": scale,
                "padding": padding,
                "local_player_bbox": [
                    (float(player_bbox[0]) - x1) * scale_x + padding_x,
                    (float(player_bbox[1]) - y1) * scale_y + padding_y,
                    (float(player_bbox[2]) - x1) * scale_x + padding_x,
                    (float(player_bbox[3]) - y1) * scale_y + padding_y,
                ],
            })
            crops.append(pose_crop)
    return requests, crops


def _expanded_crop_bbox(player_bbox, frame_width, frame_height):
    width = float(player_bbox[2]) - float(player_bbox[0])
    height = float(player_bbox[3]) - float(player_bbox[1])
    return (
        max(0, int(float(player_bbox[0]) - 0.20 * width)),
        max(0, int(float(player_bbox[1]) - 0.10 * height)),
        min(frame_width, int(float(player_bbox[2]) + 0.20 * width + 1)),
        min(frame_height, int(float(player_bbox[3]) + 0.05 * height + 1)),
    )


def _normalize_pose_crop(crop, output_width=192, output_height=640):
    height, width = crop.shape[:2]
    resized = cv2.resize(
        crop,
        (output_width, output_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized, (output_width / width, output_height / height), (0, 0)


def _bbox_to_bbox_distance(first, second):
    dx = max(
        float(second[0]) - float(first[2]),
        float(first[0]) - float(second[2]),
        0.0,
    )
    dy = max(
        float(second[1]) - float(first[3]),
        float(first[1]) - float(second[3]),
        0.0,
    )
    return (dx * dx + dy * dy) ** 0.5


def _select_crop_pose(pose_detections, local_player_bbox):
    if not pose_detections:
        return None
    return max(
        pose_detections,
        key=lambda pose: (
            0.70 * _intersection_over_smaller_box(
                local_player_bbox,
                pose.get("bbox"),
            )
            + 0.30 * float(pose.get("confidence", 0.0))
        ),
    )


def _translate_pose(pose, origin, scale=1.0, padding=(0, 0)):
    offset_x, offset_y = origin
    padding_x, padding_y = padding
    scale_x, scale_y = scale if isinstance(scale, tuple) else (scale, scale)

    def translate(point):
        return [
            (float(point[0]) - padding_x) / scale_x + offset_x,
            (float(point[1]) - padding_y) / scale_y + offset_y,
        ]

    translated = deepcopy(pose)
    bbox_start = translate(pose["bbox"][:2])
    bbox_end = translate(pose["bbox"][2:])
    translated["bbox"] = [*bbox_start, *bbox_end]
    translated["keypoints_xy"] = [
        translate(point)
        if point[0] > 0 and point[1] > 0
        else [0.0, 0.0]
        for point in pose["keypoints_xy"]
    ]
    return translated


def _match_player_poses(player_tracks, pose_detections, minimum_match=0.30):
    """Greedily perform a one-to-one overlap match between tracks and poses."""
    matches = []
    for player_id, player in player_tracks.items():
        player_bbox = player.get("bbox")
        if not player_bbox:
            continue
        for pose_index, pose in enumerate(pose_detections):
            overlap = _intersection_over_smaller_box(player_bbox, pose.get("bbox"))
            if overlap >= minimum_match:
                matches.append((overlap, player_id, pose_index))

    assigned_players = set()
    assigned_poses = set()
    result = {}
    for _, player_id, pose_index in sorted(matches, reverse=True):
        if player_id in assigned_players or pose_index in assigned_poses:
            continue
        assigned_players.add(player_id)
        assigned_poses.add(pose_index)
        result[player_id] = pose_detections[pose_index]
    return result


def _intersection_over_smaller_box(first, second):
    if not first or not second:
        return 0.0
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller > 0 else 0.0
