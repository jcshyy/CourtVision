import math

from backend.app.analytics.ball_holder_state import BallHolderStateModel
from backend.app.utils.geometry import bbox_center, euclidean_distance


class BallAquisitionDetector:
    """Detects ball possession using the reference repo's bbox heuristics."""

    def __init__(self, fps=30, minimum_possession_seconds=11 / 30):
        if fps <= 0 or minimum_possession_seconds <= 0:
            raise ValueError("FPS and possession duration must be positive")
        self.possession_threshold = 50
        self.min_frames = max(1, math.ceil(fps * minimum_possession_seconds))
        self.state_confirmation_frames = max(2, round(fps * 0.1))
        self.containment_threshold = 0.8

    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        ball_center_x = ball_center[0]
        ball_center_y = ball_center[1]

        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1

        output_points = []
        if ball_center_y > y1 and ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))

        if ball_center_x > x1 and ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))

        output_points += [
            (x1 + width // 2, y1),
            (x2, y1),
            (x1, y1),
            (x2, y1 + height // 2),
            (x1, y1 + height // 2),
            (x1 + width // 2, y1 + height // 2),
            (x2, y2),
            (x1, y2),
            (x1 + width // 2, y2),
            (x1 + width // 2, y1 + height // 3),
        ]
        return output_points

    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        intersection_x1 = max(px1, bx1)
        intersection_y1 = max(py1, by1)
        intersection_x2 = min(px2, bx2)
        intersection_y2 = min(py2, by2)

        if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
            return 0.0

        intersection_area = (intersection_x2 - intersection_x1) * (
            intersection_y2 - intersection_y1
        )
        ball_area = (bx2 - bx1) * (by2 - by1)

        return intersection_area / ball_area

    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        key_points = self.get_key_basketball_player_assignment_points(
            player_bbox,
            ball_center,
        )
        return min(euclidean_distance(ball_center, point) for point in key_points)

    def find_best_candidate_for_possession(
        self,
        ball_center,
        player_tracks_frame,
        ball_bbox,
    ):
        high_containment_players = []
        regular_distance_players = []

        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get("bbox", [])
            if not player_bbox:
                continue

            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            min_distance = self.find_minimum_distance_to_ball(
                ball_center,
                player_bbox,
            )

            if containment > self.containment_threshold:
                high_containment_players.append((player_id, min_distance))
            else:
                regular_distance_players.append((player_id, min_distance))

        if high_containment_players:
            best_candidate = min(high_containment_players, key=lambda item: item[1])
            return best_candidate[0]

        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda item: item[1])
            if best_candidate[1] < self.possession_threshold:
                return best_candidate[0]

        return -1

    def detect_ball_possession(self, player_tracks, ball_tracks):
        states = self.detect_holder_states(player_tracks, ball_tracks)
        return [state["holder_id"] if state["holder_id"] is not None else -1 for state in states]

    def detect_holder_states(self, player_tracks, ball_tracks):
        model = BallHolderStateModel(
            confirmation_frames=self.state_confirmation_frames,
            max_missing_frames=max(1, round(self.state_confirmation_frames)),
            maximum_distance=self.possession_threshold,
        )
        return model.process(player_tracks, ball_tracks)

    def detect_candidates(self, player_tracks, ball_tracks):
        """Return the unconfirmed closest-player candidate for diagnostics."""
        candidates = [-1] * len(ball_tracks)

        for frame_num in range(len(ball_tracks)):
            ball_info = ball_tracks[frame_num].get(1, {})
            if not ball_info:
                continue

            ball_bbox = ball_info.get("bbox", [])
            if not ball_bbox:
                continue

            ball_center = bbox_center(ball_bbox)
            best_player_id = self.find_best_candidate_for_possession(
                ball_center,
                player_tracks[frame_num],
                ball_bbox,
            )
            candidates[frame_num] = best_player_id

        return candidates

    def detect_acquisitions(self, player_tracks, ball_tracks):
        return self.detect_ball_possession(player_tracks, ball_tracks)


def summarize_acquisition_segments(ball_acquisitions):
    segments = []
    current_player_id = -1
    start_frame = None

    for frame_index, player_id in enumerate(ball_acquisitions):
        if player_id == current_player_id:
            continue

        if current_player_id != -1:
            segments.append(
                {
                    "player_id": current_player_id,
                    "start_frame": start_frame,
                    "end_frame": frame_index - 1,
                }
            )

        current_player_id = player_id
        start_frame = frame_index if player_id != -1 else None

    if current_player_id != -1:
        segments.append(
            {
                "player_id": current_player_id,
                "start_frame": start_frame,
                "end_frame": len(ball_acquisitions) - 1,
            }
        )

    return [
        (
            segment["player_id"],
            segment["start_frame"],
            segment["end_frame"],
            segment["end_frame"] - segment["start_frame"] + 1,
        )
        for segment in segments
    ]


def clean_acquisition_timeline(ball_acquisitions, *_, **__):
    return ball_acquisitions


BallAcquisitionDetector = BallAquisitionDetector
