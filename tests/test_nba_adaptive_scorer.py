import unittest

from scripts.score_nba_adaptive_detector import (
    candidate_predictions,
    default_output_name,
    is_adaptive,
    score_center_hits,
)


class NbaAdaptiveScorerTests(unittest.TestCase):
    def test_center_hit_credits_small_point_box_inside_stretched_truth(self):
        truths = {1: [{"id": 1, "bbox": [100, 100, 111, 120], "area": 220}]}
        # An 8 px WASB-style square centered on the ball has IoU < 0.5 but should hit.
        predictions = {1: [{"image_id": 1, "bbox": [101.5, 106, 109.5, 114], "confidence": 0.9}]}
        metrics = score_center_hits(predictions, truths, confidence=0.25)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_center_hit_matches_each_truth_once(self):
        truths = {1: [{"id": 1, "bbox": [0, 0, 10, 10], "area": 100}], 2: []}
        predictions = {
            1: [
                {"image_id": 1, "bbox": [4, 4, 6, 6], "confidence": 0.9},
                {"image_id": 1, "bbox": [3, 3, 5, 5], "confidence": 0.8},
                {"image_id": 1, "bbox": [3, 3, 5, 5], "confidence": 0.1},
            ],
            2: [{"image_id": 2, "bbox": [0, 0, 4, 4], "confidence": 0.5}],
        }
        metrics = score_center_hits(predictions, truths, confidence=0.25)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 2)
        self.assertEqual(metrics["false_negatives"], 0)

    def test_candidates_are_mapped_back_to_annotation_coordinates(self):
        scale = (640 / 1280, 640 / 720)
        [prediction] = candidate_predictions(
            scale,
            7,
            [{"bbox": [1280, 720, 1280, 720], "confidence": 0.4, "detection_source": "wasb_temporal"}],
        )
        self.assertEqual(prediction["bbox"], [640.0, 640.0, 640.0, 640.0])
        self.assertEqual(prediction["image_id"], 7)
        self.assertFalse(is_adaptive(prediction))
        self.assertTrue(is_adaptive({"detection_source": "adaptive_player_hand"}))

    def test_legacy_default_output_name_is_unchanged(self):
        self.assertEqual(default_output_name("test", "current", "yolo"), "adaptive_report.json")
        self.assertEqual(
            default_output_name("valid", "ebard", "hybrid", (1280, 720)),
            "adaptive_valid_report_ebard_hybrid_1280x720.json",
        )


if __name__ == "__main__":
    unittest.main()
