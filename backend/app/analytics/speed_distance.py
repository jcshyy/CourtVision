from backend.app.utils.geometry import euclidean_distance


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

    def calculate_distance(self, tactical_player_positions):
        previous_players_position = {}
        output_distances = []

        for frame_number, tactical_player_position_frame in enumerate(
            tactical_player_positions
        ):
            output_distances.append({})

            for player_id, current_player_position in tactical_player_position_frame.items():
                if player_id in previous_players_position:
                    previous_position = previous_players_position[player_id]
                    meter_distance = self.calculate_meter_distance(
                        previous_position,
                        current_player_position,
                    )
                    output_distances[frame_number][player_id] = meter_distance

                previous_players_position[player_id] = current_player_position

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

        return meter_distance * 0.4

    def calculate_speed(self, distances, fps=30):
        speeds = []
        window_size = 5

        for frame_index in range(len(distances)):
            speeds.append({})

            for player_id in distances[frame_index].keys():
                start_frame = max(0, frame_index - (window_size * 3) + 1)
                total_distance = 0
                frames_present = 0
                last_frame_present = None

                for index in range(start_frame, frame_index + 1):
                    if player_id in distances[index]:
                        if last_frame_present is not None:
                            total_distance += distances[index][player_id]
                            frames_present += 1
                        last_frame_present = index

                if frames_present >= window_size:
                    time_in_seconds = frames_present / fps
                    time_in_hours = time_in_seconds / 3600
                    speeds[frame_index][player_id] = (
                        (total_distance / 1000) / time_in_hours
                        if time_in_hours > 0
                        else 0
                    )
                else:
                    speeds[frame_index][player_id] = 0

        return speeds
