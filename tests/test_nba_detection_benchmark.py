import unittest

from scripts.score_nba_detection_benchmark import (
    average_precision,
    box_iou,
    build_error_analysis,
    score_predictions,
    sweep_confidence_thresholds,
    xywh_to_xyxy,
)


class NbaDetectionBenchmarkTests(unittest.TestCase):
    def test_xywh_conversion_and_iou(self):
        self.assertEqual(xywh_to_xyxy([10, 20, 30, 40]), [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_average_precision_penalizes_high_confidence_false_positive(self):
        truths = {1: [{"id": 1, "bbox": [0, 0, 10, 10], "area": 100}]}
        clean = {1: [{"image_id": 1, "bbox": [0, 0, 10, 10], "confidence": 0.8}]}
        noisy = {1: [
            {"image_id": 1, "bbox": [20, 20, 30, 30], "confidence": 0.9},
            {"image_id": 1, "bbox": [0, 0, 10, 10], "confidence": 0.8},
        ]}
        self.assertEqual(average_precision(clean, truths, 0.5), 1.0)
        self.assertLess(average_precision(noisy, truths, 0.5), 1.0)

    def test_fixed_threshold_metrics_include_negative_frames(self):
        truths = {
            1: [{"id": 1, "bbox": [0, 0, 10, 10], "area": 100}],
            2: [],
        }
        predictions = {
            1: [{"image_id": 1, "bbox": [0, 0, 10, 10], "confidence": 0.9}],
            2: [{"image_id": 2, "bbox": [2, 2, 8, 8], "confidence": 0.7}],
        }
        metrics = score_predictions(predictions, truths, confidence=0.25)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["negative_frame_false_positives"], 1)
        self.assertEqual(metrics["recall_by_size"]["small"], 1.0)

    def test_error_analysis_separates_low_confidence_misses_and_background_fps(self):
        coco = {"images": [{"id": 1, "file_name": "frame.jpg"}]}
        truths = {1: [{"id": 1, "bbox": [0, 0, 10, 10], "area": 100}]}
        predictions = {1: [
            {"image_id": 1, "bbox": [0, 0, 10, 10], "confidence": 0.2},
            {"image_id": 1, "bbox": [20, 20, 30, 30], "confidence": 0.8},
        ]}
        analysis = build_error_analysis(coco, predictions, truths, confidence=0.25)
        self.assertEqual(analysis["miss_reason_counts"], {"low_confidence": 1})
        self.assertEqual(analysis["false_positive_reason_counts"], {"background": 1})

    def test_confidence_sweep_can_select_a_lower_threshold(self):
        truths = {1: [{"id": 1, "bbox": [0, 0, 10, 10], "area": 100}]}
        predictions = {1: [
            {"image_id": 1, "bbox": [0, 0, 10, 10], "confidence": 0.2},
        ]}
        rows = sweep_confidence_thresholds(predictions, truths)
        self.assertEqual(max(rows, key=lambda row: row["f1"])["confidence"], 0.05)


if __name__ == "__main__":
    unittest.main()
