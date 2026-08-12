import cv2


DEFAULT_TEAM_COLORS = {
    1: (255, 80, 0),
    2: (0, 215, 255),
}


class PassInterceptionDrawer:
    """Draw compact cumulative event totals without obscuring the play."""

    def __init__(self, event_display_frames=15, team_colors=None):
        self.event_display_frames = max(1, int(event_display_frames))
        self.team_colors = team_colors or DEFAULT_TEAM_COLORS

    def get_stats(self, passes, interceptions):
        team1_passes = sum(value == 1 for value in passes)
        team2_passes = sum(value == 2 for value in passes)
        team1_interceptions = sum(value == 1 for value in interceptions)
        team2_interceptions = sum(value == 2 for value in interceptions)
        return (
            team1_passes,
            team2_passes,
            team1_interceptions,
            team2_interceptions,
        )

    def draw(self, video_frames, passes, interceptions):
        return [
            self.draw_frame(frame, frame_num, passes, interceptions)
            for frame_num, frame in enumerate(video_frames)
        ]

    def draw_frame(self, frame, frame_num, passes, interceptions):
        frame_height, frame_width = frame.shape[:2]
        scale = max(0.72, min(1.15, frame_width / 1280))
        margin = max(12, round(frame_width * 0.018))
        panel_width = min(round(frame_width * 0.34), round(420 * scale))
        panel_height = round(112 * scale)
        x1 = margin
        y1 = frame_height - margin - panel_height
        x2 = x1 + panel_width
        y2 = y1 + panel_height

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 14, 17), cv2.FILLED)
        cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (92, 103, 112), 1)
        cv2.rectangle(frame, (x1, y1), (x1 + round(4 * scale), y2), (47, 107, 255), cv2.FILLED)

        totals = self.get_stats(
            passes[: frame_num + 1], interceptions[: frame_num + 1]
        )
        recent_event = self.get_recent_event(frame_num, passes, interceptions)
        pad = round(15 * scale)
        header_y = y1 + round(24 * scale)
        self._text(frame, "EVENTS", (x1 + pad, header_y), 0.48 * scale, (214, 220, 224), 1)

        if recent_event:
            label = f"{recent_event['type'].upper()}  T{recent_event['team_id']}"
            self._right_text(
                frame,
                label,
                (x2 - pad, header_y),
                0.43 * scale,
                self.team_colors[recent_event["team_id"]],
                1,
            )

        row_y = [y1 + round(58 * scale), y1 + round(91 * scale)]
        rows = [
            (1, totals[0], totals[2]),
            (2, totals[1], totals[3]),
        ]
        for y, (team_id, pass_count, interception_count) in zip(row_y, rows):
            color = self.team_colors[team_id]
            cv2.circle(frame, (x1 + pad + round(4 * scale), y - round(5 * scale)), round(4 * scale), color, cv2.FILLED, cv2.LINE_AA)
            self._text(frame, f"TEAM {team_id}", (x1 + pad + round(16 * scale), y), 0.5 * scale, (245, 247, 248), 1)
            self._right_text(frame, f"P {pass_count}   INT {interception_count}", (x2 - pad, y), 0.5 * scale, (245, 247, 248), 1)

        return frame

    @staticmethod
    def _text(frame, text, origin, scale, color, thickness):
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _right_text(frame, text, right_baseline, scale, color, thickness):
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        PassInterceptionDrawer._text(
            frame,
            text,
            (right_baseline[0] - size[0], right_baseline[1]),
            scale,
            color,
            thickness,
        )

    def get_recent_event(self, frame_num, passes, interceptions):
        start = max(0, frame_num - self.event_display_frames + 1)
        for event_frame in range(frame_num, start - 1, -1):
            if event_frame < len(passes) and passes[event_frame] in (1, 2):
                return {"type": "pass", "team_id": passes[event_frame], "frame_index": event_frame}
            if event_frame < len(interceptions) and interceptions[event_frame] in (1, 2):
                return {"type": "interception", "team_id": interceptions[event_frame], "frame_index": event_frame}
        return None
