import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_nba_holdout_annotation_batch import build
from scripts.validate_nba_holdout_annotations import validate


def _fixture():
    manifest = {
        "courtvision_predictions_used": False,
        "clips": [
            {
                "id": "clip",
                "width": 100,
                "height": 80,
                "frame_count": 16,
                "fps": 30.0,
                "scenes": [
                    {
                        "id": "scene_a",
                        "start_frame": 0,
                        "end_frame": 15,
                        "teams": {
                            "scene_a_team_a": {"name": "A"},
                            "scene_a_team_b": {"name": "B"},
                        },
                    }
                ],
            }
        ],
    }
    frames = [
        {
            "video_id": "clip",
            "frame_index": frame,
            "scene_id": "scene_a",
            "ball": {
                "visibility": "visible",
                "center_px": [50, 40],
                "confidence": "high",
            },
            "possession": {
                "state": "controlled",
                "team_id": "scene_a_team_a",
                "holder": "player",
            },
            "review_status": "draft",
        }
        for frame in (0, 15)
    ]
    events = [
        {
            "video_id": "clip",
            "scene_id": "scene_a",
            "event_type": "pass",
            "start_frame": 1,
            "end_frame": 8,
            "release_frame": 2,
            "catch_frame": 7,
            "from_team_id": "scene_a_team_a",
            "to_team_id": "scene_a_team_a",
            "confidence": "high",
            "review_status": "draft",
        }
    ]
    return manifest, frames, events


def _write_fixture(directory, manifest, frames, events):
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "frames.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in frames),
        encoding="utf-8",
    )
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in events),
        encoding="utf-8",
    )


class NBAHoldoutAnnotationToolTests(unittest.TestCase):
    def test_batch_builder_expands_segments_and_ball_overrides(self):
        manifest, _, _ = _fixture()
        manifest["sampling_interval_frames"] = 15
        spec = {
            "clips": {
                "clip": {
                    "possession_segments": [
                        {
                            "start_frame": 0,
                            "end_frame": 7,
                            "state": "controlled",
                            "team_id": "scene_a_team_a",
                            "holder": "player",
                        },
                        {"start_frame": 8, "end_frame": 15, "state": "unknown"},
                    ],
                    "ball_overrides": {
                        "15": {
                            "visibility": "visible",
                            "center_px": [25, 30],
                            "confidence": "high",
                        }
                    },
                }
            },
            "events": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (root / "annotation_spec.json").write_text(
                json.dumps(spec), encoding="utf-8"
            )
            counts = build(root)
            records = [
                json.loads(line)
                for line in (root / "frames.jsonl").read_text().splitlines()
            ]
        self.assertEqual(counts, {"frames": 2, "events": 0})
        self.assertEqual(records[0]["possession"]["state"], "controlled")
        self.assertEqual(records[1]["possession"]["state"], "unknown")
        self.assertEqual(records[1]["ball"]["center_px"], [25, 30])

    def test_valid_calibration(self):
        manifest, frames, events = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root, manifest, frames, events)
            summary = validate(root)
        self.assertEqual(summary["frame_draft"], 2)
        self.assertEqual(summary["event_type_pass"], 1)

    def test_visible_ball_center_must_be_inside_frame(self):
        manifest, frames, events = _fixture()
        frames[0]["ball"]["center_px"] = [100, 40]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root, manifest, frames, events)
            with self.assertRaisesRegex(ValueError, "invalid visible ball center"):
                validate(root)

    def test_controlled_team_must_be_scene_local(self):
        manifest, frames, events = _fixture()
        frames[0]["possession"]["team_id"] = "other_scene_team"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root, manifest, frames, events)
            with self.assertRaisesRegex(ValueError, "scene-local team"):
                validate(root)

    def test_pass_endpoints_must_be_ordered(self):
        manifest, frames, events = _fixture()
        events[0]["release_frame"] = 7
        events[0]["catch_frame"] = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root, manifest, frames, events)
            with self.assertRaisesRegex(ValueError, "ordered release/catch"):
                validate(root)

    def test_prediction_use_must_be_explicitly_false(self):
        manifest, frames, events = _fixture()
        manifest["courtvision_predictions_used"] = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root, manifest, frames, events)
            with self.assertRaisesRegex(ValueError, "predictions were not used"):
                validate(root)


if __name__ == "__main__":
    unittest.main()
