import tempfile
import unittest
import zipfile
from hashlib import sha256 as hashlib_sha256
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_ebard_detection import (
    canonical_class_map,
    class_average_precision,
    detection_metrics,
    yolo_box_to_xyxy,
)
from scripts.prepare_ebard_detection_dataset import (
    dataset_is_ready,
    install_dataset,
)


class EbardDetectionEvaluationTests(unittest.TestCase):
    def test_class_vocabulary_is_normalized_across_both_models(self):
        class_map = canonical_class_map(
            {0: "Ball", 1: "Hoop", 2: "Players", 3: "Ref"}
        )

        self.assertEqual(
            class_map,
            {0: "basketball", 1: "hoop", 2: "player", 3: "referee"},
        )

    def test_yolo_box_is_converted_to_pixel_coordinates(self):
        self.assertEqual(
            yolo_box_to_xyxy([0.5, 0.5, 0.25, 0.5], 200, 100),
            [75.0, 25.0, 125.0, 75.0],
        )

    def test_operating_metrics_count_true_false_and_missed_detections(self):
        samples = [
            {
                "ground_truth": [
                    {"class": "player", "bbox": [0, 0, 10, 10]},
                    {"class": "referee", "bbox": [20, 20, 30, 30]},
                ]
            }
        ]
        predictions = [
            [
                {"class": "player", "bbox": [0, 0, 10, 10], "confidence": 0.9},
                {"class": "player", "bbox": [40, 40, 50, 50], "confidence": 0.8},
            ]
        ]

        metrics = detection_metrics(samples, predictions, 0.5)

        self.assertEqual(metrics["per_class"]["player"]["tp"], 1)
        self.assertEqual(metrics["per_class"]["player"]["fp"], 1)
        self.assertEqual(metrics["per_class"]["referee"]["fn"], 1)
        self.assertEqual(metrics["micro"]["tp"], 1)
        self.assertEqual(metrics["micro"]["fp"], 1)
        self.assertEqual(metrics["micro"]["fn"], 1)

    def test_perfect_ranked_detection_has_unit_average_precision(self):
        samples = [
            {
                "ground_truth": [
                    {"class": "basketball", "bbox": [0, 0, 10, 10]}
                ]
            }
        ]
        predictions = [
            [
                {
                    "class": "basketball",
                    "bbox": [0, 0, 10, 10],
                    "confidence": 0.9,
                }
            ]
        ]

        self.assertEqual(
            class_average_precision(samples, predictions, "basketball", 0.5),
            1.0,
        )


class EbardDetectionInstallerTests(unittest.TestCase):
    def test_verified_yolo_subset_is_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "all.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("qwen/ignored.json", "{}")
                bundle.writestr("yolo/data.yaml", "names: []\n")
                for split in ("train", "val", "test"):
                    bundle.writestr(f"yolo/{split}/images/frame.jpg", b"fixture")
                    bundle.writestr(f"yolo/{split}/labels/frame.txt", "")
            expected = hashlib_sha256(archive.read_bytes()).hexdigest()
            output = root / "installed"

            with patch(
                "scripts.prepare_ebard_detection_dataset.DATASET_SHA256",
                expected,
            ):
                installed = install_dataset(output, archive=archive)

            self.assertEqual(installed, output)
            self.assertTrue(dataset_is_ready(output))
            self.assertFalse((output / "qwen").exists())

    def test_incomplete_existing_target_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "installed"
            output.mkdir()

            with self.assertRaisesRegex(RuntimeError, "exists but is incomplete"):
                install_dataset(output)


if __name__ == "__main__":
    unittest.main()
