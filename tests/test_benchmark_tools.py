import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_labeled_benchmark import _sample_frames
from scripts.annotate_events import event_is_complete, next_pending_index, set_event_type
from scripts.validate_benchmark import validate
from scripts.score_benchmark import (
    classification_summary,
    match_events,
    optimal_team_mapping,
)


class BenchmarkToolTests(unittest.TestCase):
    def test_sampling_includes_regular_final_and_focus_context_frames(self):
        self.assertEqual(
            _sample_frames(11, 5, [6], 1),
            [0, 5, 6, 7, 10],
        )

    def test_validator_accepts_minimal_verified_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "frames" / "clip").mkdir(parents=True)
            (root / "frames" / "clip" / "000000.jpg").write_bytes(b"image")
            (root / "dataset.json").write_text(json.dumps({
                "videos": [{
                    "id": "clip", "frame_count": 1, "width": 100, "height": 50,
                }]
            }), encoding="utf-8")
            record = {
                "video_id": "clip", "frame_index": 0,
                "image_path": "frames/clip/000000.jpg",
                "review_status": "verified",
                "ball": {"visibility": "visible", "center_px": [10, 20]},
                "possession": {"state": "controlled", "team": "team_a"},
            }
            (root / "annotations.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            (root / "events.jsonl").write_text("", encoding="utf-8")
            validate(root)

    def test_event_completion_depends_on_event_semantics(self):
        self.assertTrue(event_is_complete({
            "event_type": "pass", "release_frame": 4, "catch_frame": 8,
        }))
        self.assertFalse(event_is_complete({
            "event_type": "pass", "release_frame": None, "catch_frame": 8,
        }))
        self.assertTrue(event_is_complete({
            "event_type": "shot", "release_frame": 4, "catch_frame": None,
        }))
        self.assertFalse(event_is_complete({
            "event_type": "shot", "release_frame": 4, "catch_frame": 8,
        }))
        self.assertTrue(event_is_complete({
            "event_type": "defensive_rebound", "release_frame": None,
            "catch_frame": 9,
        }))

    def test_changing_event_type_clears_irrelevant_fields(self):
        event = {
            "event_type": "pass", "release_frame": 4, "catch_frame": 8,
            "from_team": "team_a", "to_team": "team_b",
        }
        set_event_type(event, "shot")
        self.assertEqual(event["release_frame"], 4)
        self.assertIsNone(event["catch_frame"])
        self.assertIsNone(event["to_team"])
        set_event_type(event, "defensive_rebound")
        self.assertIsNone(event["release_frame"])
        self.assertIsNone(event["from_team"])

    def test_next_pending_skips_resolved_events_and_wraps(self):
        events = [
            {"review_status": "pending"},
            {"review_status": "verified"},
            {"review_status": "pending"},
        ]
        self.assertEqual(next_pending_index(events, 0), 2)
        self.assertEqual(next_pending_index(events, 2), 0)
        events[0]["review_status"] = "verified"
        self.assertEqual(next_pending_index(events, 2), len(events))

    def test_classification_summary_reports_supported_class_metrics(self):
        summary = classification_summary(
            [("controlled", "controlled"), ("controlled", "loose"), ("loose", "loose")],
            ["controlled", "loose"],
        )
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertAlmostEqual(summary["per_class"]["controlled"]["recall"], 0.5)
        self.assertEqual(summary["confusion"]["controlled"]["loose"], 1)

    def test_optimal_team_mapping_handles_arbitrary_cluster_ids(self):
        mapping = optimal_team_mapping([
            (1, "team_b"), (1, "team_b"), (2, "team_a"),
        ])
        self.assertEqual(mapping, {1: "team_b", 2: "team_a"})

    def test_event_matching_is_type_aware_and_one_to_one(self):
        truth = [
            {"video_id": "clip", "event_type": "pass", "catch_frame": 10},
            {"video_id": "clip", "event_type": "shot", "release_frame": 20},
        ]
        predictions = [
            {"video_id": "clip", "type": "pass", "catch_frame": 12},
            {"video_id": "clip", "type": "pass", "catch_frame": 13},
        ]
        result = match_events(truth, predictions, tolerance=5)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["unmatched_truth"], [1])
        self.assertEqual(result["unmatched_predictions"], [1])


if __name__ == "__main__":
    unittest.main()
