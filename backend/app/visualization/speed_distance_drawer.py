import cv2


class SpeedAndDistanceDrawer:
    """Draw readable speed chips; distance remains optional for diagnostic renders."""

    def __init__(self, show_distance=False, minimum_speed=0.5):
        self.show_distance = bool(show_distance)
        self.minimum_speed = float(minimum_speed)

    def draw(self, video_frames, player_tracks, player_distances_per_frame, player_speed_per_frame):
        output_video_frames = []
        total_distances = {}
        for frame, frame_tracks, player_distance, player_speed in zip(
            video_frames, player_tracks, player_distances_per_frame, player_speed_per_frame
        ):
            output_frame = frame.copy()
            for player_id, distance in player_distance.items():
                total_distances[player_id] = total_distances.get(player_id, 0) + distance
            for player_id, player in frame_tracks.items():
                speed = player_speed.get(player_id)
                if speed is None or speed < self.minimum_speed:
                    continue
                distance = total_distances.get(player_id)
                label = f"{speed:.1f} km/h"
                if self.show_distance and distance is not None:
                    label += f"  {distance:.1f} m"
                self._draw_chip(output_frame, player["bbox"], label)
            output_video_frames.append(output_frame)
        return output_video_frames

    @staticmethod
    def _draw_chip(frame, bbox, label):
        frame_height, frame_width = frame.shape[:2]
        scale = max(0.42, min(0.62, frame_width / 2200))
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        pad_x, pad_y = 6, 4
        chip_width = text_width + pad_x * 2
        chip_height = text_height + baseline + pad_y * 2
        center_x = int((bbox[0] + bbox[2]) / 2)
        x1 = max(2, min(frame_width - chip_width - 2, center_x - chip_width // 2))
        preferred_y = int(bbox[3]) + 32
        y1 = preferred_y if preferred_y + chip_height < frame_height - 2 else max(2, int(bbox[1]) - chip_height - 6)
        x2, y2 = x1 + chip_width, y1 + chip_height
        cv2.rectangle(frame, (x1, y1), (x2, y2), (8, 10, 12), cv2.FILLED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (224, 229, 232), 1)
        cv2.putText(
            frame,
            label,
            (x1 + pad_x, y2 - pad_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
