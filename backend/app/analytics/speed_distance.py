from backend.app.utils.geometry import euclidean_distance
import statistics


class SpeedAndDistanceCalculator:
    def __init__(
        self,
        width_in_pixels,
        height_in_pixels,
        width_in_meters,
        height_in_meters,
    ):
        self.width_in_pixels = width_in_pixels
        self.height_in_pixels = height_in_pixels
        self.width_in_meters = width_in_meters
        self.height_in_meters = height_in_meters

    def smooth_positions(
        self,
        tactical_player_positions,
        discontinuity_frames=None,
        window_radius=2,
    ):
        """Apply a centered median within continuous track/homography segments."""
        if window_radius < 0:
            raise ValueError("Window radius must be non-negative")
        discontinuity_frames = set(discontinuity_frames or [])
        output = [dict(frame) for frame in tactical_player_positions]
        segments = {}

        for frame_index, frame_positions in enumerate(tactical_player_positions):
            for player_id in frame_positions:
                player_segments = segments.setdefault(player_id, [[]])
                current_segment = player_segments[-1]
                if (
                    current_segment
                    and (
                        frame_index != current_segment[-1] + 1
                        or frame_index in discontinuity_frames
                    )
                ):
                    current_segment = []
                    player_segments.append(current_segment)
                current_segment.append(frame_index)

        for player_id, player_segments in segments.items():
            for segment in player_segments:
                for position_index, frame_index in enumerate(segment):
                    start = max(0, position_index - window_radius)
                    end = min(len(segment), position_index + window_radius + 1)
                    neighbors = [
                        tactical_player_positions[neighbor_frame][player_id]
                        for neighbor_frame in segment[start:end]
                    ]
                    output[frame_index][player_id] = [
                        statistics.median(position[0] for position in neighbors),
                        statistics.median(position[1] for position in neighbors),
                    ]
        return output

    def calculate_distance(
        self,
        tactical_player_positions,
        discontinuity_frames=None,
    ):
        previous_players_position = {}
        output_distances = []
        discontinuity_frames = set(discontinuity_frames or [])

        for frame_number, tactical_player_position_frame in enumerate(
            tactical_player_positions
        ):
            output_distances.append({})
            if frame_number in discontinuity_frames:
                previous_players_position = {}

            for player_id, current_player_position in tactical_player_position_frame.items():
                if player_id in previous_players_position:
                    previous_frame, previous_position = previous_players_position[
                        player_id
                    ]
                    if frame_number == previous_frame + 1:
                        meter_distance = self.calculate_meter_distance(
                            previous_position,
                            current_player_position,
                        )
                        output_distances[frame_number][player_id] = meter_distance

                previous_players_position[player_id] = (
                    frame_number,
                    current_player_position,
                )

        return output_distances

    def calculate_meter_distance(self, previous_pixel_position, current_pixel_position):
        previous_pixel_x, previous_pixel_y = previous_pixel_position
        current_pixel_x, current_pixel_y = current_pixel_position

        previous_meter_x = (
            previous_pixel_x * self.width_in_meters / self.width_in_pixels
        )
        previous_meter_y = (
            previous_pixel_y * self.height_in_meters / self.height_in_pixels
        )
        current_meter_x = current_pixel_x * self.width_in_meters / self.width_in_pixels
        current_meter_y = (
            current_pixel_y * self.height_in_meters / self.height_in_pixels
        )

        meter_distance = euclidean_distance(
            (current_meter_x, current_meter_y),
            (previous_meter_x, previous_meter_y),
        )

        return meter_distance

    def calculate_speed(
        self,
        distances,
        fps=30,
        tactical_player_positions=None,
        window_seconds=0.5,
    ):
        if fps <= 0:
            raise ValueError("FPS must be positive")
        if window_seconds <= 0:
            raise ValueError("Speed window must be positive")
        speeds = []
        window_size = max(2, round(fps * window_seconds))

        for frame_index in range(len(distances)):
            speeds.append({})

            for player_id in distances[frame_index].keys():
                consecutive_distances = []
                for index in range(frame_index, -1, -1):
                    if player_id not in distances[index]:
                        break
                    consecutive_distances.append(distances[index][player_id])
                    if len(consecutive_distances) == window_size:
                        break

                if len(consecutive_distances) == window_size:
                    elapsed_seconds = window_size / fps
                    distance_for_speed = sum(consecutive_distances)
                    start_frame = frame_index - window_size
                    if (
                        tactical_player_positions is not None
                        and start_frame >= 0
                        and player_id
                        in tactical_player_positions[start_frame]
                    ):
                        distance_for_speed = self.calculate_meter_distance(
                            tactical_player_positions[start_frame][player_id],
                            tactical_player_positions[frame_index][player_id],
                        )
                    speeds[frame_index][player_id] = (
                        distance_for_speed / elapsed_seconds * 3.6
                    )
                else:
                    speeds[frame_index][player_id] = 0

        return speeds
