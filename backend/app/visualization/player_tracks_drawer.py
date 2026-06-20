from backend.app.visualization.drawing_utils import draw_ellipse, draw_triangle


class PlayerTracksDrawer:
    """Draws tracked player positions on video frames."""

    def __init__(self, default_color=(0, 255, 0)):
        self.default_color = default_color

    def draw(
        self,
        frames,
        tracked_results,
        team_assignments=None,
        team_colors=None,
        ball_acquisitions=None,
    ):
        output_frames = [frame.copy() for frame in frames]
        team_assignments = team_assignments or [{} for _ in output_frames]
        team_colors = team_colors or {}
        ball_acquisitions = ball_acquisitions or [-1 for _ in output_frames]

        for frame, result, frame_teams, acquired_player_id in zip(
            output_frames,
            tracked_results,
            team_assignments,
            ball_acquisitions,
        ):
            for track_id, bbox in _extract_tracks(result):
                team_id = frame_teams.get(track_id)
                color = team_colors.get(team_id, self.default_color)
                draw_ellipse(frame, bbox, color, track_id)
                if track_id == acquired_player_id:
                    draw_triangle(frame, bbox, (0, 0, 255))

        return output_frames


def _extract_tracks(result):
    if isinstance(result, dict):
        return [
            (track_id, player["bbox"])
            for track_id, player in result.items()
            if player.get("bbox") is not None
        ]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    return list(zip(_extract_track_ids(boxes), boxes.xyxy.cpu().tolist()))


def _extract_track_ids(boxes):
    if boxes.id is None:
        return [None] * len(boxes)

    return [int(track_id) for track_id in boxes.id.cpu().tolist()]
