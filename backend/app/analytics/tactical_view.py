from copy import deepcopy
import logging
import math
import statistics

import cv2
import numpy as np

from backend.app.analytics.homography import Homography
from backend.app.utils.geometry import euclidean_distance


logger = logging.getLogger(__name__)


class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 300
        self.height = 161
        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15
        self.min_keypoint_confidence = 0.35
        self.duplicate_source_distance = 28
        self.duplicate_target_distance = 18
        self.ransac_reprojection_threshold = 12
        self.max_reprojection_error = 20
        self.max_homography_fallback_frames = 12
        self.last_diagnostics = {}
        self.key_points = [
            (0, 0),
            (0, int((0.91 / self.actual_height_in_meters) * self.height)),
            (0, int((5.18 / self.actual_height_in_meters) * self.height)),
            (0, int((10 / self.actual_height_in_meters) * self.height)),
            (0, int((14.1 / self.actual_height_in_meters) * self.height)),
            (0, int(self.height)),
            (int(self.width / 2), self.height),
            (int(self.width / 2), 0),
            (
                int((5.79 / self.actual_width_in_meters) * self.width),
                int((5.18 / self.actual_height_in_meters) * self.height),
            ),
            (
                int((5.79 / self.actual_width_in_meters) * self.width),
                int((10 / self.actual_height_in_meters) * self.height),
            ),
            (self.width, int(self.height)),
            (self.width, int((14.1 / self.actual_height_in_meters) * self.height)),
            (self.width, int((10 / self.actual_height_in_meters) * self.height)),
            (self.width, int((5.18 / self.actual_height_in_meters) * self.height)),
            (self.width, int((0.91 / self.actual_height_in_meters) * self.height)),
            (self.width, 0),
            (
                int(
                    ((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters)
                    * self.width
                ),
                int((5.18 / self.actual_height_in_meters) * self.height),
            ),
            (
                int(
                    ((self.actual_width_in_meters - 5.79) / self.actual_width_in_meters)
                    * self.width
                ),
                int((10 / self.actual_height_in_meters) * self.height),
            ),
        ]

    def validate_keypoints(self, keypoints_list):
        keypoints_list = deepcopy(keypoints_list)

        for frame_index, frame_keypoints in enumerate(keypoints_list):
            frame_points = _keypoint_points(frame_keypoints)
            if not frame_points:
                continue

            confidences = _keypoint_confidences(frame_keypoints, len(frame_points))
            detected_indices = [
                index
                for index, keypoint in enumerate(frame_points)
                if (
                    keypoint[0] > 0
                    and keypoint[1] > 0
                    and confidences[index] >= self.min_keypoint_confidence
                )
            ]

            invalid_keypoints = {
                index
                for index, keypoint in enumerate(frame_points)
                if (
                    keypoint[0] <= 0
                    or keypoint[1] <= 0
                    or confidences[index] < self.min_keypoint_confidence
                )
            }

            if len(detected_indices) < 2:
                for index in invalid_keypoints:
                    _zero_keypoint(frame_keypoints, index)
                continue

            for left_position, left_index in enumerate(detected_indices):
                if left_index in invalid_keypoints:
                    continue

                for right_index in detected_indices[left_position + 1 :]:
                    if right_index in invalid_keypoints:
                        continue

                    source_distance = euclidean_distance(
                        frame_points[left_index],
                        frame_points[right_index],
                    )
                    target_distance = euclidean_distance(
                        self.key_points[left_index],
                        self.key_points[right_index],
                    )

                    if (
                        source_distance <= self.duplicate_source_distance
                        and target_distance >= self.duplicate_target_distance
                    ):
                        if confidences[left_index] < confidences[right_index]:
                            invalid_keypoints.add(left_index)
                        else:
                            invalid_keypoints.add(right_index)

            ransac_indices = [
                index for index in detected_indices if index not in invalid_keypoints
            ]
            if len(ransac_indices) >= 4:
                source_points = np.array(
                    [frame_points[index] for index in ransac_indices],
                    dtype=np.float32,
                )
                target_points = np.array(
                    [self.key_points[index] for index in ransac_indices],
                    dtype=np.float32,
                )
                try:
                    matrix, mask = cv2.findHomography(
                        source_points,
                        target_points,
                        cv2.RANSAC,
                        self.ransac_reprojection_threshold,
                    )
                    if matrix is None or mask is None:
                        invalid_keypoints.update(ransac_indices)
                    else:
                        projected_points = cv2.perspectiveTransform(
                            source_points.reshape(-1, 1, 2),
                            matrix,
                        ).reshape(-1, 2)
                        errors = np.linalg.norm(projected_points - target_points, axis=1)

                        for point_index, keypoint_index in enumerate(ransac_indices):
                            if (
                                mask[point_index][0] == 0
                                or errors[point_index] > self.max_reprojection_error
                            ):
                                invalid_keypoints.add(keypoint_index)
                except cv2.error:
                    invalid_keypoints.update(ransac_indices)

            for index in invalid_keypoints:
                _zero_keypoint(keypoints_list[frame_index], index)

        return keypoints_list

    def transform_players_to_tactical_view(
        self,
        keypoints_list,
        player_tracks,
    ):
        tactical_player_positions = []
        last_good_homography = None
        fallback_frames = 0
        previous_tactical_positions = {}
        diagnostic_frames = {
            "missing_keypoints": [],
            "insufficient_keypoints": [],
            "rejected_homography": [],
            "fallback_used": [],
            "homography_unavailable": [],
            "temporal_discontinuity": [],
        }

        for frame_index, (frame_keypoints, frame_tracks) in enumerate(
            zip(keypoints_list, player_tracks)
        ):
            tactical_positions = {}
            frame_keypoints = _keypoint_points(frame_keypoints)

            if frame_keypoints is None or len(frame_keypoints) == 0:
                diagnostic_frames["missing_keypoints"].append(frame_index)
                tactical_player_positions.append(tactical_positions)
                continue

            valid_indices = [
                index
                for index, keypoint in enumerate(frame_keypoints)
                if keypoint[0] > 0 and keypoint[1] > 0
            ]

            if len(valid_indices) < 4:
                diagnostic_frames["insufficient_keypoints"].append(frame_index)
                homography = None
            else:
                source_points = np.array(
                    [frame_keypoints[index] for index in valid_indices],
                    dtype=np.float32,
                )
                target_points = np.array(
                    [self.key_points[index] for index in valid_indices],
                    dtype=np.float32,
                )

                try:
                    homography = Homography(
                        source_points,
                        target_points,
                        method=cv2.RANSAC,
                        ransac_reproj_threshold=self.ransac_reprojection_threshold,
                    )
                    if homography.inlier_count < 4 or not _homography_is_consistent(
                        homography,
                        source_points,
                        target_points,
                        self.max_reprojection_error,
                    ):
                        diagnostic_frames["rejected_homography"].append(frame_index)
                        homography = None
                except (ValueError, cv2.error):
                    diagnostic_frames["rejected_homography"].append(frame_index)
                    homography = None

            if homography is not None:
                last_good_homography = homography
                fallback_frames = 0
            elif (
                last_good_homography is not None
                and fallback_frames < self.max_homography_fallback_frames
            ):
                homography = last_good_homography
                fallback_frames += 1
                diagnostic_frames["fallback_used"].append(frame_index)
            else:
                diagnostic_frames["homography_unavailable"].append(frame_index)
                tactical_player_positions.append(tactical_positions)
                continue

            try:
                for player_id, player_data in frame_tracks.items():
                    player_position = np.array([_foot_position(player_data["bbox"])])
                    tactical_position = homography.transform_points(player_position)

                    if (
                        tactical_position[0][0] < 0
                        or tactical_position[0][0] > self.width
                        or tactical_position[0][1] < 0
                        or tactical_position[0][1] > self.height
                    ):
                        continue

                    tactical_positions[player_id] = tactical_position[0].tolist()

            except (ValueError, cv2.error):
                pass

            if _tactical_view_discontinuity(
                previous_tactical_positions,
                tactical_positions,
                self.width,
                self.height,
                self.actual_width_in_meters,
                self.actual_height_in_meters,
            ):
                diagnostic_frames["temporal_discontinuity"].append(frame_index)

            previous_tactical_positions = tactical_positions

            tactical_player_positions.append(tactical_positions)

        self.last_diagnostics = diagnostic_frames
        _log_homography_diagnostics(diagnostic_frames)
        return tactical_player_positions


def _keypoint_points(frame_keypoints):
    points = frame_keypoints.xy.cpu().tolist()
    return points[0] if points else []


def _keypoint_confidences(frame_keypoints, point_count):
    confidences = [1.0] * point_count

    confidence_tensor = getattr(frame_keypoints, "conf", None)
    if confidence_tensor is None:
        return confidences

    confidence_values = confidence_tensor.cpu().tolist()
    if not confidence_values:
        return confidences

    confidence_values = confidence_values[0]
    for index, confidence in enumerate(confidence_values[:point_count]):
        confidences[index] = float(confidence)

    return confidences


def _zero_keypoint(frame_keypoints, index):
    data = getattr(frame_keypoints, "data", None)
    if data is not None and len(data.shape) >= 3 and index < data.shape[1]:
        data[0][index] *= 0

    for attribute in ("xy", "xyn"):
        points = getattr(frame_keypoints, attribute, None)
        if points is not None and len(points.shape) >= 3 and index < points.shape[1]:
            points[0][index] *= 0

    confidences = getattr(frame_keypoints, "conf", None)
    if (
        confidences is not None
        and len(confidences.shape) >= 2
        and index < confidences.shape[1]
    ):
        confidences[0][index] *= 0


def _homography_is_consistent(homography, source_points, target_points, max_error):
    projected_points = homography.transform_points(source_points)
    errors = np.linalg.norm(projected_points - target_points, axis=1)
    return np.median(errors) <= max_error and np.max(errors) <= max_error * 2


def _log_homography_diagnostics(diagnostic_frames):
    failures = {
        name: frames
        for name, frames in diagnostic_frames.items()
        if frames
    }
    if not failures:
        return

    details = ", ".join(
        f"{name}={len(frames)} (sample frames: {frames[:5]})"
        for name, frames in failures.items()
    )
    logger.warning("Tactical-view homography diagnostics: %s", details)


def _foot_position(bbox):
    x1, _, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def _tactical_view_discontinuity(
    previous_positions,
    current_positions,
    width_pixels,
    height_pixels,
    width_meters,
    height_meters,
    minimum_common_tracks=2,
    maximum_median_court_fraction=0.25,
):
    common_tracks = previous_positions.keys() & current_positions.keys()
    if len(common_tracks) < minimum_common_tracks:
        return False

    displacements = []
    for player_id in common_tracks:
        previous_x, previous_y = previous_positions[player_id]
        current_x, current_y = current_positions[player_id]
        displacements.append(
            math.hypot(
                (current_x - previous_x) * width_meters / width_pixels,
                (current_y - previous_y) * height_meters / height_pixels,
            )
        )

    court_diagonal = math.hypot(width_meters, height_meters)
    return (
        statistics.median(displacements)
        > court_diagonal * maximum_median_court_fraction
    )
