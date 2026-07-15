import cv2
import numpy as np


class TeamBallControlDrawer:
    """Draw cumulative control over frames with a known holder and known team."""

    def __init__(self, *_, **__):
        pass

    def get_team_ball_control(self, player_assignment, ball_acquisition):
        team_ball_control = []

        for assignment_frame, acquisition_frame in zip(
            player_assignment,
            ball_acquisition,
        ):
            if acquisition_frame == -1 or acquisition_frame is None:
                team_ball_control.append(-1)
                continue

            team_id = assignment_frame.get(acquisition_frame)
            if team_id not in (1, 2):
                team_ball_control.append(-1)
                continue

            team_ball_control.append(team_id)

        return np.array(team_ball_control)

    def draw(self, video_frames, player_assignment, ball_acquisition=None):
        if ball_acquisition is None:
            team_ball_control = np.array(player_assignment)
        else:
            team_ball_control = self.get_team_ball_control(
                player_assignment,
                ball_acquisition,
            )

        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            output_video_frames.append(
                self.draw_frame(frame, frame_num, team_ball_control)
            )

        return output_video_frames

    def draw_frame(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2
        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.60)
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.99)
        rect_y2 = int(frame_height * 0.90)
        text_x = int(frame_width * 0.63)
        text_y1 = int(frame_height * 0.80)
        text_y2 = int(frame_height * 0.88)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        team_1, team_2 = self.get_control_percentages(
            team_ball_control_till_frame
        )

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1 * 100:.2f}%",
            (text_x, text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2 * 100:.2f}%",
            (text_x, text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

        return frame

    def get_control_percentages(self, team_ball_control):
        """Exclude no-holder/unknown frames so the two displayed shares sum to 1."""
        team_1_num_frames = team_ball_control[team_ball_control == 1].shape[0]
        team_2_num_frames = team_ball_control[team_ball_control == 2].shape[0]
        known_possession_frames = team_1_num_frames + team_2_num_frames
        if known_possession_frames == 0:
            return 0.0, 0.0
        return (
            team_1_num_frames / known_possession_frames,
            team_2_num_frames / known_possession_frames,
        )


def build_team_ball_control(ball_acquisitions, team_assignments, max_carry_frames=None):
    return TeamBallControlDrawer().get_team_ball_control(
        team_assignments,
        ball_acquisitions,
    )


def build_team_ball_control_from_events(frame_count, events, initial_team_id):
    team_ball_control = []
    current_team_id = initial_team_id if initial_team_id is not None else -1
    events_by_frame = {}

    for event in events:
        events_by_frame.setdefault(event["frame_index"], []).append(event)

    for frame_index in range(frame_count):
        for event in events_by_frame.get(frame_index, []):
            current_team_id = event["to_team_id"]
        team_ball_control.append(current_team_id)

    return team_ball_control


def infer_initial_control_team(ball_acquisitions, team_assignments):
    team_ball_control = build_team_ball_control(ball_acquisitions, team_assignments)
    team_1_count = team_ball_control[team_ball_control == 1].shape[0]
    team_2_count = team_ball_control[team_ball_control == 2].shape[0]

    if team_1_count == 0 and team_2_count == 0:
        return None

    return 1 if team_1_count >= team_2_count else 2
