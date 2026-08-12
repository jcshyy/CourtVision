import cv2
import numpy as np


DEFAULT_TEAM_COLORS = {
    1: (255, 80, 0),
    2: (0, 215, 255),
}


class TeamBallControlDrawer:
    """Draw a compact upper-right estimate based only on known team holders."""

    def __init__(self, team_colors=None, *_, **__):
        self.team_colors = team_colors or DEFAULT_TEAM_COLORS

    def get_team_ball_control(self, player_assignment, ball_acquisition):
        team_ball_control = []
        for assignment_frame, acquisition_frame in zip(player_assignment, ball_acquisition):
            if acquisition_frame == -1 or acquisition_frame is None:
                team_ball_control.append(-1)
                continue
            team_id = assignment_frame.get(acquisition_frame)
            team_ball_control.append(team_id if team_id in (1, 2) else -1)
        return np.array(team_ball_control)

    def draw(self, video_frames, player_assignment, ball_acquisition=None):
        team_ball_control = (
            np.array(player_assignment)
            if ball_acquisition is None
            else self.get_team_ball_control(player_assignment, ball_acquisition)
        )
        return [
            self.draw_frame(frame, frame_num, team_ball_control)
            for frame_num, frame in enumerate(video_frames)
        ]

    def draw_frame(self, frame, frame_num, team_ball_control):
        frame_height, frame_width = frame.shape[:2]
        scale = max(0.72, min(1.15, frame_width / 1280))
        margin = max(12, round(frame_width * 0.018))
        panel_width = min(round(frame_width * 0.30), round(370 * scale))
        panel_height = round(112 * scale)
        x2 = frame_width - margin
        x1 = x2 - panel_width
        y1 = margin
        y2 = y1 + panel_height

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 14, 17), cv2.FILLED)
        cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (92, 103, 112), 1)

        team_1, team_2 = self.get_control_percentages(team_ball_control[: frame_num + 1])
        known = team_1 + team_2 > 0
        pad = round(15 * scale)
        cv2.putText(frame, "POSSESSION EST.", (x1 + pad, y1 + round(25 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale, (214, 220, 224), 1, cv2.LINE_AA)

        if not known:
            cv2.putText(frame, "NO RELIABLE HOLDER", (x1 + pad, y1 + round(66 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (245, 247, 248), 1, cv2.LINE_AA)
            return frame

        label_y = y1 + round(57 * scale)
        cv2.putText(frame, f"T1  {team_1 * 100:.0f}%", (x1 + pad, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.54 * scale, self.team_colors[1], 1, cv2.LINE_AA)
        right_label = f"T2  {team_2 * 100:.0f}%"
        size, _ = cv2.getTextSize(right_label, cv2.FONT_HERSHEY_SIMPLEX, 0.54 * scale, 1)
        cv2.putText(frame, right_label, (x2 - pad - size[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.54 * scale, self.team_colors[2], 1, cv2.LINE_AA)

        bar_x1 = x1 + pad
        bar_x2 = x2 - pad
        bar_y1 = y1 + round(75 * scale)
        bar_y2 = bar_y1 + max(7, round(10 * scale))
        split_x = bar_x1 + round((bar_x2 - bar_x1) * team_1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (48, 55, 61), cv2.FILLED)
        if split_x > bar_x1:
            cv2.rectangle(frame, (bar_x1, bar_y1), (split_x, bar_y2), self.team_colors[1], cv2.FILLED)
        if split_x < bar_x2:
            cv2.rectangle(frame, (split_x, bar_y1), (bar_x2, bar_y2), self.team_colors[2], cv2.FILLED)
        return frame

    def get_control_percentages(self, team_ball_control):
        """Exclude no-holder/unknown frames so the displayed shares sum to one."""
        team_1_num_frames = team_ball_control[team_ball_control == 1].shape[0]
        team_2_num_frames = team_ball_control[team_ball_control == 2].shape[0]
        known_possession_frames = team_1_num_frames + team_2_num_frames
        if known_possession_frames == 0:
            return 0.0, 0.0
        return team_1_num_frames / known_possession_frames, team_2_num_frames / known_possession_frames


def build_team_ball_control(ball_acquisitions, team_assignments, max_carry_frames=None):
    return TeamBallControlDrawer().get_team_ball_control(team_assignments, ball_acquisitions)


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
