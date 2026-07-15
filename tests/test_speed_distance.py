import unittest

from backend.app.analytics.speed_distance import SpeedAndDistanceCalculator


class SpeedDistanceTests(unittest.TestCase):
    def setUp(self):
        self.calculator = SpeedAndDistanceCalculator(280, 150, 28, 15)

    def test_meter_scale_does_not_apply_video_specific_multiplier(self):
        self.assertEqual(
            self.calculator.calculate_meter_distance((0, 0), (10, 0)),
            1.0,
        )

    def test_track_gap_does_not_add_jump_distance(self):
        distances = self.calculator.calculate_distance(
            [
                {7: (0, 0)},
                {7: (10, 0)},
                {},
                {7: (100, 0)},
                {7: (110, 0)},
            ]
        )

        self.assertEqual(distances[1][7], 1.0)
        self.assertNotIn(7, distances[3])
        self.assertEqual(distances[4][7], 1.0)

    def test_global_transform_discontinuity_resets_distance(self):
        distances = self.calculator.calculate_distance(
            [
                {7: (10, 0)},
                {7: (20, 0)},
                {7: (270, 0)},
                {7: (280, 0)},
            ],
            discontinuity_frames=[2],
        )

        self.assertEqual(distances[1][7], 1.0)
        self.assertNotIn(7, distances[2])
        self.assertEqual(distances[3][7], 1.0)

    def test_speed_uses_actual_fps(self):
        distances = [{}] + [{7: 1.0} for _ in range(10)]

        at_ten_fps = self.calculator.calculate_speed(distances, fps=10)
        at_twenty_fps = self.calculator.calculate_speed(distances, fps=20)

        self.assertEqual(at_ten_fps[-1][7], 36.0)
        self.assertEqual(at_twenty_fps[-1][7], 72.0)

    def test_speed_window_does_not_cross_track_gap(self):
        distances = [{7: 1.0}] * 4 + [{}] + [{7: 1.0}] * 4
        speeds = self.calculator.calculate_speed(distances, fps=10)

        self.assertEqual(speeds[-1][7], 0)

    def test_speed_uses_window_displacement_to_resist_coordinate_jitter(self):
        positions = [
            {7: [10, 20]},
            {7: [11, 20]},
            {7: [9, 20]},
            {7: [11, 20]},
            {7: [9, 20]},
            {7: [10, 20]},
        ]
        distances = self.calculator.calculate_distance(positions)

        speeds = self.calculator.calculate_speed(
            distances,
            fps=10,
            tactical_player_positions=positions,
        )

        self.assertEqual(speeds[-1][7], 0)

    def test_speed_rejects_invalid_fps(self):
        with self.assertRaisesRegex(ValueError, "FPS"):
            self.calculator.calculate_speed([], fps=0)

    def test_position_median_rejects_single_frame_homography_jitter(self):
        positions = [
            {7: [10, 20]},
            {7: [11, 20]},
            {7: [200, 140]},
            {7: [13, 20]},
            {7: [14, 20]},
        ]

        smoothed = self.calculator.smooth_positions(positions)

        self.assertEqual(smoothed[2][7], [13, 20])

    def test_position_smoothing_does_not_cross_discontinuity(self):
        positions = [
            {7: [10, 20]},
            {7: [11, 20]},
            {7: [280, 20]},
            {7: [281, 20]},
        ]

        smoothed = self.calculator.smooth_positions(
            positions,
            discontinuity_frames=[2],
        )

        self.assertLess(smoothed[1][7][0], 20)
        self.assertGreater(smoothed[2][7][0], 270)


if __name__ == "__main__":
    unittest.main()
