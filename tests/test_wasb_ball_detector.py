import unittest

import numpy as np

from backend.app.detection.wasb_ball_detector import (
    _affine_transform,
    _merge_window_candidates,
    _window_indices,
)


class WasbBallDetectorHelpersTest(unittest.TestCase):
    def test_window_indices_cover_every_frame_with_dense_inference(self):
        windows = _window_indices(6, step=1)

        self.assertEqual(
            windows,
            [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)],
        )
        self.assertEqual(set().union(*map(set, windows)), set(range(6)))

    def test_short_clip_repeats_last_available_frame(self):
        self.assertEqual(_window_indices(1, step=1), [(0, 0, 0)])
        self.assertEqual(_window_indices(2, step=1), [(0, 1, 1)])

    def test_affine_transform_round_trip_restores_frame_point(self):
        shape = (720, 1280, 3)
        forward = _affine_transform(shape)
        inverse = _affine_transform(shape, inverse=True)
        point = np.asarray([932.5, 411.25, 1.0], dtype=np.float32)

        model_point = forward @ point
        restored = inverse @ np.asarray([*model_point, 1.0], dtype=np.float32)

        np.testing.assert_allclose(restored, point[:2], atol=1e-3)

    def test_overlapping_window_candidates_are_deduplicated(self):
        candidates = [
            {
                "bbox": [95.0, 95.0, 105.0, 105.0],
                "confidence": 0.90,
                "wasb_component_score": 4.0,
            },
            {
                "bbox": [98.0, 96.0, 108.0, 106.0],
                "confidence": 0.80,
                "wasb_component_score": 3.0,
            },
            {
                "bbox": [195.0, 195.0, 205.0, 205.0],
                "confidence": 0.70,
                "wasb_component_score": 2.0,
            },
        ]

        merged = _merge_window_candidates(candidates)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["confidence"], 0.90)
        self.assertEqual(merged[0]["wasb_window_support"], 2)
        self.assertEqual(merged[1]["wasb_window_support"], 1)


if __name__ == "__main__":
    unittest.main()
