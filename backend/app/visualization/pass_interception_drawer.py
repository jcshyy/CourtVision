import cv2


class PassInterceptionDrawer:
    """Draws cumulative pass and interception totals using repo-style arrays."""

    def __init__(self, event_display_frames=15):
        self.event_display_frames = max(1, int(event_display_frames))

    def get_stats(self, passes, interceptions):
        team1_passes = []
        team2_passes = []
        team1_interceptions = []
        team2_interceptions = []

        for frame_num, (pass_frame, interception_frame) in enumerate(
            zip(passes, interceptions)
        ):
            if pass_frame == 1:
                team1_passes.append(frame_num)
            elif pass_frame == 2:
                team2_passes.append(frame_num)

            if interception_frame == 1:
                team1_interceptions.append(frame_num)
            elif interception_frame == 2:
                team2_interceptions.append(frame_num)

        return (
            len(team1_passes),
            len(team2_passes),
            len(team1_interceptions),
            len(team2_interceptions),
        )

    def draw(self, video_frames, passes, interceptions):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            output_video_frames.append(
                self.draw_frame(frame, frame_num, passes, interceptions)
            )

        return output_video_frames

    def draw_frame(self, frame, frame_num, passes, interceptions):
        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2
        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.16)
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.55)
        rect_y2 = int(frame_height * 0.90)
        text_x = int(frame_width * 0.19)
        heading_y = int(frame_height * 0.785)
        text_y1 = int(frame_height * 0.83)
        text_y2 = int(frame_height * 0.88)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        (
            team1_passes,
            team2_passes,
            team1_interceptions,
            team2_interceptions,
        ) = self.get_stats(passes[: frame_num + 1], interceptions[: frame_num + 1])

        recent_event = self.get_recent_event(frame_num, passes, interceptions)
        heading = "Cumulative events"
        if recent_event is not None:
            heading += f" | {recent_event['type'].upper()} - Team {recent_event['team_id']}"
        cv2.putText(
            frame,
            heading,
            (text_x, heading_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            font_thickness,
        )

        cv2.putText(
            frame,
            f"Team 1 - Passes: {team1_passes} Interceptions: {team1_interceptions}",
            (text_x, text_y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )
        cv2.putText(
            frame,
            f"Team 2 - Passes: {team2_passes} Interceptions: {team2_interceptions}",
            (text_x, text_y2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

        return frame

    def get_recent_event(self, frame_num, passes, interceptions):
        start = max(0, frame_num - self.event_display_frames + 1)
        for event_frame in range(frame_num, start - 1, -1):
            if passes[event_frame] in (1, 2):
                return {
                    "type": "pass",
                    "team_id": passes[event_frame],
                    "frame_index": event_frame,
                }
            if interceptions[event_frame] in (1, 2):
                return {
                    "type": "interception",
                    "team_id": interceptions[event_frame],
                    "frame_index": event_frame,
                }
        return None
