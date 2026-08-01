from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from backend.app.utils.geometry import bbox_center, euclidean_distance


@dataclass(frozen=True)
class HolderFrameState:
    holder_id: int | None
    state: str
    confidence: float
    candidate_id: int | None
    reason: str
    ball_confidence: float | None
    distance_to_holder: float | None
    frames_since_confirmed: int

    def to_dict(self):
        return asdict(self)


class BallHolderStateModel:
    """Small temporal state machine over ball detections and player tracks."""

    def __init__(
        self,
        *,
        confirmation_frames=3,
        max_missing_frames=3,
        maximum_distance=50.0,
        minimum_score=0.42,
        ambiguity_margin=0.08,
        minimum_gap_ball_confidence=0.65,
    ):
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.maximum_distance = float(maximum_distance)
        self.minimum_score = float(minimum_score)
        self.ambiguity_margin = float(ambiguity_margin)
        self.minimum_gap_ball_confidence = float(minimum_gap_ball_confidence)

    def process(self, player_tracks, ball_tracks):
        if len(player_tracks) != len(ball_tracks):
            raise ValueError("Player and ball tracks must have the same frame count")

        states = []
        holder_id = None
        holder_confidence = 0.0
        pending_id = None
        pending_frames = 0
        pending_gap_frames = 0
        pending_relative_positions = []
        missing_frames = 0
        frames_since_confirmed = 0
        previous_ball_center = None
        last_observed_ball_confidence = None

        for players, frame_ball in zip(player_tracks, ball_tracks):
            ball = frame_ball.get(1, {})
            bbox = ball.get("bbox")
            observed = bool(bbox)
            interpolated = bool(ball.get("interpolated", False))
            ball_confidence = _optional_float(ball.get("confidence"))

            if not observed:
                missing_frames += 1
                if pending_id is not None:
                    missing_candidate_id = pending_id
                    pending_id, pending_frames, pending_gap_frames = None, 0, 0
                    frames_since_confirmed += 1
                    states.append(
                        HolderFrameState(
                            None, "unknown", 0.0, missing_candidate_id,
                            "pending_candidate_ball_missing", ball_confidence, None,
                            frames_since_confirmed,
                        ).to_dict()
                    )
                elif (
                    holder_id is not None
                    and missing_frames <= self.max_missing_frames
                    and frames_since_confirmed <= self.max_missing_frames
                    and last_observed_ball_confidence is not None
                    and last_observed_ball_confidence
                    >= self.minimum_gap_ball_confidence
                ):
                    frames_since_confirmed += 1
                    decay = max(0.35, 1.0 - missing_frames / (self.max_missing_frames + 1))
                    states.append(HolderFrameState(
                        holder_id, "confirmed", round(holder_confidence * decay, 4),
                        None, "brief_ball_gap", ball_confidence,
                        _distance_to_player(previous_ball_center, players.get(holder_id)),
                        frames_since_confirmed,
                    ).to_dict())
                else:
                    holder_id = None
                    holder_confidence = 0.0
                    frames_since_confirmed += 1
                    states.append(
                        HolderFrameState(
                            None, "unknown", 0.0, None, "ball_missing",
                            ball_confidence, None, frames_since_confirmed,
                        ).to_dict()
                    )
                continue

            ball_center = bbox_center(bbox)
            ranked = self._rank_candidates(
                players, bbox, ball_center, ball_confidence, holder_id,
                previous_ball_center, interpolated,
                hand_pose_player_id=ball.get("hand_pose_player_id"),
                hand_pose_supported=bool(ball.get("hand_pose_supported", False)),
            )
            previous_ball_center = ball_center

            if interpolated:
                missing_frames += 1
                candidate_id = ranked[0][0] if ranked else pending_id
                candidate_score = ranked[0][1] if ranked else 0.0
                if pending_id is not None:
                    pending_gap_frames += 1
                    if pending_gap_frames > self.max_missing_frames:
                        pending_id, pending_frames, pending_gap_frames = None, 0, 0
                    frames_since_confirmed += 1
                    states.append(HolderFrameState(
                        None, "loose", round(candidate_score, 4),
                        pending_id or candidate_id,
                        "interpolated_ball_not_confirmable", ball_confidence, None,
                        frames_since_confirmed,
                    ).to_dict())
                elif (
                    holder_id is not None
                    and missing_frames <= self.max_missing_frames
                    and frames_since_confirmed <= self.max_missing_frames
                    and last_observed_ball_confidence is not None
                    and last_observed_ball_confidence
                    >= self.minimum_gap_ball_confidence
                    and ranked
                    and ranked[0][0] == holder_id
                    and ranked[0][1] >= self.minimum_score
                ):
                    frames_since_confirmed += 1
                    decay = max(0.35, 1.0 - missing_frames / (self.max_missing_frames + 1))
                    states.append(HolderFrameState(
                        holder_id, "confirmed", round(holder_confidence * decay, 4),
                        holder_id, "brief_interpolated_gap", ball_confidence,
                        round(ranked[0][2], 3), frames_since_confirmed,
                    ).to_dict())
                else:
                    if missing_frames > self.max_missing_frames:
                        holder_id, holder_confidence = None, 0.0
                    frames_since_confirmed += 1
                    states.append(HolderFrameState(
                        None, "loose", round(candidate_score, 4), candidate_id,
                        "interpolated_ball_not_confirmable", ball_confidence, None,
                        frames_since_confirmed,
                    ).to_dict())
                continue

            missing_frames = 0
            pending_gap_frames = 0
            last_observed_ball_confidence = ball_confidence

            if not ranked or ranked[0][1] < self.minimum_score:
                pending_id, pending_frames, pending_gap_frames = None, 0, 0
                frames_since_confirmed += 1
                states.append(HolderFrameState(
                    None, "loose", ranked[0][1] if ranked else 0.0,
                    ranked[0][0] if ranked else None, "no_credible_candidate",
                    ball_confidence, None, frames_since_confirmed,
                ).to_dict())
                continue

            candidate_id, candidate_score, candidate_distance = ranked[0]
            ambiguous = len(ranked) > 1 and candidate_score - ranked[1][1] < self.ambiguity_margin
            if ambiguous and holder_id not in (candidate_id, ranked[1][0]):
                pending_id, pending_frames, pending_gap_frames = None, 0, 0
                frames_since_confirmed += 1
                states.append(HolderFrameState(
                    None, "loose", round(candidate_score, 4), candidate_id,
                    "ambiguous_candidates", ball_confidence, None,
                    frames_since_confirmed,
                ).to_dict())
                continue

            if candidate_id == holder_id:
                pending_id, pending_frames, pending_gap_frames = None, 0, 0
                holder_confidence = candidate_score
                frames_since_confirmed = 0
                states.append(HolderFrameState(
                    holder_id, "confirmed", round(candidate_score, 4), candidate_id,
                    "holder_reinforced", ball_confidence, round(candidate_distance, 3), 0,
                ).to_dict())
                continue

            if candidate_id == pending_id:
                pending_frames += 1
            else:
                pending_id, pending_frames = candidate_id, 1
                pending_relative_positions = []
            relative_position = _player_relative_position(
                ball_center,
                players.get(candidate_id, {}).get("bbox"),
            )
            if relative_position is not None:
                pending_relative_positions.append((
                    relative_position,
                    bool(
                        ball.get("hand_pose_supported", False)
                        and ball.get("hand_pose_player_id") == candidate_id
                    ),
                ))

            required = self.confirmation_frames
            if (
                pending_frames >= required
                and _looks_like_unsupported_flythrough(
                    pending_relative_positions
                )
            ):
                holder_id = None
                holder_confidence = 0.0
                pending_id, pending_frames, pending_gap_frames = None, 0, 0
                frames_since_confirmed += 1
                states.append(HolderFrameState(
                    None, "loose", round(candidate_score, 4), candidate_id,
                    "airborne_candidate_not_confirmed", ball_confidence,
                    None, frames_since_confirmed,
                ).to_dict())
                continue
            if holder_id is None and pending_frames >= required:
                holder_id = candidate_id
                holder_confidence = candidate_score
                pending_id, pending_frames, pending_gap_frames = None, 0, 0
                frames_since_confirmed = 0
                reason = "initial_holder_confirmed"
            elif holder_id is not None and pending_frames >= required:
                holder_id = candidate_id
                holder_confidence = candidate_score
                pending_id, pending_frames, pending_gap_frames = None, 0, 0
                frames_since_confirmed = 0
                reason = "holder_switch_confirmed"
            elif (
                holder_id is not None
                and pending_frames == 1
                and ball_confidence is not None
                and ball_confidence >= self.minimum_gap_ball_confidence
            ):
                frames_since_confirmed += 1
                states.append(HolderFrameState(
                    holder_id, "confirmed", round(holder_confidence * 0.85, 4),
                    candidate_id, "switch_pending", ball_confidence,
                    _rounded_distance(ball_center, players.get(holder_id)),
                    frames_since_confirmed,
                ).to_dict())
                continue
            elif holder_id is not None:
                frames_since_confirmed += 1
                states.append(HolderFrameState(
                    None, "loose", round(candidate_score, 4), candidate_id,
                    "candidate_switch_building", ball_confidence, None,
                    frames_since_confirmed,
                ).to_dict())
                continue
            else:
                frames_since_confirmed += 1
                states.append(HolderFrameState(
                    None, "candidate", round(candidate_score, 4), candidate_id,
                    "candidate_building", ball_confidence, None,
                    frames_since_confirmed,
                ).to_dict())
                continue

            states.append(HolderFrameState(
                holder_id, "confirmed", round(holder_confidence, 4), holder_id,
                reason, ball_confidence, round(candidate_distance, 3), 0,
            ).to_dict())

        return states

    def _rank_candidates(
        self, players, ball_bbox, ball_center, ball_confidence, holder_id,
        previous_ball_center, interpolated, *, hand_pose_player_id=None,
        hand_pose_supported=False,
    ):
        ranked = []
        motion = (
            euclidean_distance(ball_center, previous_ball_center)
            if previous_ball_center is not None else 0.0
        )
        motion_quality = max(0.0, 1.0 - motion / 120.0)
        detection_quality = (
            0.35 if interpolated
            else (0.7 if ball_confidence is None else ball_confidence)
        )
        for player_id, player in players.items():
            player_bbox = player.get("bbox")
            if not player_bbox:
                continue
            distance = _bbox_distance(ball_center, player_bbox)
            containment = _containment_ratio(player_bbox, ball_bbox)
            inside = _point_inside(ball_center, player_bbox)
            proximity = max(0.0, 1.0 - distance / self.maximum_distance)
            score = (
                0.40 * proximity
                + 0.25 * min(1.0, containment)
                + 0.12 * float(inside)
                + 0.13 * max(0.0, min(1.0, detection_quality))
                + 0.10 * motion_quality
            )
            # Pose is deliberately a relative tie-breaker rather than a
            # requirement. A confident ball-to-hand association can resolve
            # overlapping player boxes; missing/weak pose data changes no
            # score. Penalizing only the competing candidates also avoids the
            # score-cap saturation that occurs when the ball is inside both.
            if hand_pose_supported:
                score += 0.04 if player_id == hand_pose_player_id else -0.10
            if player_id == holder_id:
                score += 0.10
            ranked.append((player_id, min(1.0, score), distance))
        return sorted(ranked, key=lambda item: (-item[1], item[2], item[0]))


def _containment_ratio(player_bbox, ball_bbox):
    px1, py1, px2, py2 = player_bbox
    bx1, by1, bx2, by2 = ball_bbox
    width, height = bx2 - bx1, by2 - by1
    if width <= 0 or height <= 0:
        return 0.0
    intersection = max(0.0, min(px2, bx2) - max(px1, bx1)) * max(
        0.0, min(py2, by2) - max(py1, by1)
    )
    return intersection / (width * height)


def _bbox_distance(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    closest = (min(max(x, x1), x2), min(max(y, y1), y2))
    return euclidean_distance(point, closest)


def _player_relative_position(point, bbox):
    if not bbox:
        return None
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    if width <= 0 or height <= 0:
        return None
    return (
        (float(point[0]) - float(bbox[0])) / width,
        (float(point[1]) - float(bbox[1])) / height,
    )


def _looks_like_unsupported_flythrough(samples):
    if len(samples) < 4 or any(hand_supported for _, hand_supported in samples):
        return False
    positions = [position for position, _ in samples]
    net_x = positions[-1][0] - positions[0][0]
    net_y = positions[-1][1] - positions[0][1]
    net_distance = (net_x ** 2 + net_y ** 2) ** 0.5
    if net_distance < 0.28:
        return False
    unit_x, unit_y = net_x / net_distance, net_y / net_distance
    aligned = 0
    for start, end in zip(positions, positions[1:]):
        progress = (
            (end[0] - start[0]) * unit_x
            + (end[1] - start[1]) * unit_y
        )
        aligned += progress >= -0.02
    return aligned / (len(positions) - 1) >= 0.75


def _point_inside(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _distance_to_player(point, player):
    if point is None or not player or not player.get("bbox"):
        return None
    return round(_bbox_distance(point, player["bbox"]), 3)


def _rounded_distance(point, player):
    return _distance_to_player(point, player)


def _optional_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None
