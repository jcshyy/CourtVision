import cv2


class SpeedAndDistanceDrawer:
    def draw(
        self,
        video_frames,
        player_tracks,
        player_distances_per_frame,
        player_speed_per_frame,
    ):
        output_video_frames = []
        total_distances = {}

        for frame, frame_tracks, player_distance, player_speed in zip(
            video_frames,
            player_tracks,
            player_distances_per_frame,
            player_speed_per_frame,
        ):
            output_frame = frame.copy()

            for player_id, distance in player_distance.items():
                if player_id not in total_distances:
                    total_distances[player_id] = 0
                total_distances[player_id] += distance

            for player_id, player in frame_tracks.items():
                x1, _, x2, y2 = player["bbox"]
                position = [int((x1 + x2) / 2), int(y2) + 40]
                distance = total_distances.get(player_id)
                speed = player_speed.get(player_id)

                if speed is not None:
                    cv2.putText(
                        output_frame,
                        f"{speed:.2f} km/h",
                        position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )

                if distance is not None:
                    cv2.putText(
                        output_frame,
                        f"{distance:.2f} m",
                        (position[0], position[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )

            output_video_frames.append(output_frame)

        return output_video_frames
