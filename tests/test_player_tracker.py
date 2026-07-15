import unittest

import numpy as np

from backend.app.tracking.player_tracker import (
    _bbox_iou,
    _is_duplicate_referee_detection,
)


class _Detections:
    def __init__(self, boxes, confidence):
        self.xyxy = np.array(boxes, dtype=float)
        self.confidence = np.array(confidence, dtype=float)


class PlayerTrackerRefereeFilterTests(unittest.TestCase):
    def test_duplicate_player_and_referee_box_is_rejected(self):
        referees = _Detections([[10, 10, 50, 90]], [0.82])

        self.assertTrue(
            _is_duplicate_referee_detection(
                [10.5, 10, 50.5, 90],
                0.88,
                referees,
            )
        )

    def test_nearby_player_is_not_rejected(self):
        referees = _Detections([[10, 10, 50, 90]], [0.95])

        self.assertFalse(
            _is_duplicate_referee_detection(
                [45, 10, 85, 90],
                0.80,
                referees,
            )
        )

    def test_weak_referee_hypothesis_does_not_override_player(self):
        referees = _Detections([[10, 10, 50, 90]], [0.60])

        self.assertFalse(
            _is_duplicate_referee_detection(
                [10, 10, 50, 90],
                0.90,
                referees,
            )
        )

    def test_iou_boundary_is_scale_independent(self):
        self.assertAlmostEqual(
            _bbox_iou([0, 0, 10, 10], [0, 0, 8, 10]),
            _bbox_iou([0, 0, 100, 100], [0, 0, 80, 100]),
        )


if __name__ == "__main__":
    unittest.main()
