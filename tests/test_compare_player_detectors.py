import unittest

from scripts.compare_player_detectors import box_iou, greedy_matches


class ComparePlayerDetectorsTests(unittest.TestCase):
    def test_box_iou(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_greedy_matches_do_not_reuse_a_detection(self):
        current = [{"bbox": [0, 0, 10, 10]}]
        ebard = [
            {"bbox": [0, 0, 10, 10]},
            {"bbox": [0, 0, 9, 10]},
        ]

        matches = greedy_matches(current, ebard)

        self.assertEqual(matches, [1.0])


if __name__ == "__main__":
    unittest.main()
