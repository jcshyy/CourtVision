import io
import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.evaluate_event_sequences import evaluate, _match
from scripts.prepare_multisports_events import AnnotationUnpickler, convert


class EventScoringTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.truth = {"name": "synthetic accounting test", "scored_types": ["pass", "shot_attempt", "throw_in"],
                      "videos": [{"video_id": "clip", "fps": 30, "frame_count": 120, "split": "validation"}],
                      "events": [{"video_id": "clip", "type": "shot_attempt", "start_frame": 30, "end_frame": 36}]}

    def write(self, events, fps=30, decisions=None):
        report = {"source": {"fps": fps}, "events": events,
                  "diagnostics": {"shotAttemptTimeline": {"arbitration": decisions or []}}}
        (self.root / "clip_analysis.json").write_text(json.dumps(report))

    def test_wrong_class_is_fp_and_fn_not_success(self):
        self.write([{"type": "pass", "frameIndex": 33}])
        result = evaluate(self.truth, self.root)
        self.assertEqual((result["micro"]["tp"], result["micro"]["fp"], result["micro"]["fn"]), (0, 1, 1))
        self.assertEqual(result["confusion"]["shot_attempt"]["pass"], 1)

    def test_duplicate_is_counted_once_and_penalized(self):
        self.write([{"type": "shot_attempt", "frameIndex": 33}] * 2)
        result = evaluate(self.truth, self.root)
        self.assertEqual(result["micro"]["tp"], 1)
        self.assertEqual(result["micro"]["fp"], 1)
        self.assertEqual(result["duplicate_predictions"], 1)

    def test_missing_prediction_is_not_removed_from_recall(self):
        result = evaluate(self.truth, self.root)
        self.assertEqual(result["micro"]["fn"], 1)
        self.assertEqual(result["coverage"], 0)
        self.assertEqual(result["missing_videos"], ["clip"])

    def test_wrong_timing_is_not_saved_by_matching_clip_count(self):
        self.write([{"type": "shot_attempt", "frameIndex": 90}])
        result = evaluate(self.truth, self.root)
        self.assertEqual(result["micro"]["f1"], 0)

    def test_resampled_prediction_uses_seconds_and_release_evidence(self):
        self.write([{"type": "shot_attempt", "frameIndex": 60,
                     "evidence": {"release_frame": 16}}], fps=15)
        self.assertEqual(evaluate(self.truth, self.root)["micro"]["tp"], 1)

    def test_false_event_in_negative_video_counts(self):
        self.truth["events"] = []
        self.write([{"type": "pass", "frameIndex": 90}])
        self.assertEqual(evaluate(self.truth, self.root)["micro"]["fp"], 1)

    def test_throw_in_can_be_scored_without_public_pass_count(self):
        self.truth["events"][0]["type"] = "throw_in"
        self.write([], decisions=[{"reason": "throw_in", "event": {
            "type": "pass", "release_frame": 33, "frame_index": 60}}])
        self.assertEqual(evaluate(self.truth, self.root)["by_class"]["throw_in"]["tp"], 1)

    def test_maximum_matching_handles_ambiguous_nearest_neighbor(self):
        truth = [{"type": "pass", "start_seconds": 1, "end_seconds": 1},
                 {"type": "pass", "start_seconds": 1.3, "end_seconds": 1.3}]
        predictions = [{"type": "pass", "time_seconds": 1.1}, {"type": "pass", "time_seconds": 0.8}]
        self.assertEqual(len(_match(truth, predictions, 0.25)), 2)

    def test_invalid_intervals_and_unsafe_paths_fail_closed(self):
        self.truth["events"][0]["end_frame"] = 150
        with self.assertRaises(ValueError):
            evaluate(self.truth, self.root)
        self.truth["events"] = []
        self.truth["videos"][0]["video_id"] = "../outside"
        with self.assertRaises(ValueError):
            evaluate(self.truth, self.root)


class MultiSportsAdapterTests(unittest.TestCase):
    def data(self):
        return {"labels": ["basketball pass", "basketball 3-point shot", "basketball pass-inbound", "basketball screen"],
                "train_videos": [["basketball/train"]], "test_videos": [["basketball/val"]],
                "nframes": {"basketball/val": 100}, "gttubes": {"basketball/val": {
                    0: [np.array([[1, 0, 0, 1, 1], [3, 0, 0, 1, 1]])],
                    1: [np.array([[20, 0, 0, 1, 1], [40, 0, 0, 1, 1]])],
                    2: [np.array([[60, 0, 0, 1, 1]])], 3: [np.array([[80, 0, 0, 1, 1]])]}}}

    def test_labels_frame_base_and_validation_split(self):
        result = convert(self.data(), {"basketball/val": 25})
        self.assertEqual([e["type"] for e in result["events"]], ["pass", "shot_attempt", "throw_in"])
        self.assertEqual(result["events"][0]["start_frame"], 0)
        self.assertEqual(result["events"][0]["end_frame"], 2)
        self.assertEqual(result["videos"][0]["split"], "validation")
        self.assertEqual(result["ignored_source_labels"], {"basketball screen": 1})

    def test_numpy_only_pickle_roundtrip(self):
        for protocol in (2, 4, 5):
            if protocol == 2:
                continue  # Legacy byte codecs intentionally require conversion to JSON.
            restored = AnnotationUnpickler(io.BytesIO(pickle.dumps(self.data(), protocol=protocol))).load()
            self.assertEqual(len(convert(restored, {"basketball/val": 25})["events"]), 3)

    def test_pickle_cannot_resolve_arbitrary_globals(self):
        payload = b"cos\nsystem\n."
        with self.assertRaises(pickle.UnpicklingError):
            AnnotationUnpickler(io.BytesIO(payload)).load()

    def test_missing_fps_is_not_guessed(self):
        with self.assertRaises(KeyError):
            convert(self.data(), {})


if __name__ == "__main__":
    unittest.main()
