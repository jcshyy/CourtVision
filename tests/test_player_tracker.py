import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

from backend.app.tracking.player_tracker import (
    PlayerTracker,
    _bbox_iou,
    _is_duplicate_referee_detection,
    _is_persistent_referee_track,
    player_detector_settings,
    player_tracking_algorithm_version,
    resolve_player_class_ids,
)


class _Detections:
    def __init__(self, boxes, confidence):
        self.xyxy = np.array(boxes, dtype=float)
        self.confidence = np.array(confidence, dtype=float)


class PlayerTrackerRefereeFilterTests(unittest.TestCase):
    def test_class_resolution_supports_current_and_ebard_labels(self):
        self.assertEqual(
            resolve_player_class_ids({0: "Ball", 4: "Player", 5: "Ref"}),
            (4, 5),
        )
        self.assertEqual(
            resolve_player_class_ids(
                {0: "basketball", 1: "hoop", 2: "player", 3: "referee"}
            ),
            (2, 3),
        )

    def test_class_resolution_requires_player_label(self):
        with self.assertRaisesRegex(ValueError, "does not expose a player class"):
            resolve_player_class_ids({0: "basketball", 1: "hoop"})

    def test_detector_backends_have_isolated_tracking_cache_versions(self):
        self.assertEqual(
            player_tracking_algorithm_version("current"),
            "v5_repeated_referee_filter",
        )
        self.assertEqual(
            player_tracking_algorithm_version("ebard"),
            "v5_repeated_referee_filter_ebard",
        )

    def test_unknown_detector_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown player detector backend"):
            player_detector_settings("mystery")

    def test_ebard_backend_uses_its_threshold_and_cache(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "ebard.pt"
            model_path.write_bytes(b"weights")
            with patch(
                "backend.app.tracking.player_tracker.YOLO",
                return_value=MagicMock(),
            ), patch("backend.app.tracking.player_tracker.sv.ByteTrack"):
                tracker = PlayerTracker(
                    model_path=model_path,
                    detector_backend="ebard",
                )

        self.assertEqual(tracker.confidence, 0.25)
        self.assertEqual(
            tracker.cache_filename,
            "player_track_v5_repeated_referee_filter_ebard.pkl",
        )

    def test_player_tracks_consume_shared_scene_provider(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "ebard.pt"
            model_path.write_bytes(b"weights")
            model = MagicMock()
            byte_tracker = MagicMock()
            byte_tracker.update_with_detections.return_value = []
            provider = MagicMock(
                return_value=[MagicMock(names={2: "player", 3: "referee"})]
            )
            detections = MagicMock()
            detections.class_id = np.asarray([], dtype=int)
            with patch(
                "backend.app.tracking.player_tracker.YOLO",
                return_value=model,
            ), patch(
                "backend.app.tracking.player_tracker.sv.ByteTrack",
                return_value=byte_tracker,
            ), patch(
                "backend.app.tracking.player_tracker.sv.Detections.from_ultralytics",
                return_value=detections,
            ):
                tracker = PlayerTracker(
                    model_path=model_path,
                    detector_backend="ebard",
                )
                tracks = tracker.get_object_tracks(
                    [object()],
                    detections_provider=provider,
                )

        provider.assert_called_once_with()
        model.predict.assert_not_called()
        self.assertEqual(tracks, [{}])

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

    def test_single_referee_overlap_does_not_delete_long_player_track(self):
        self.assertFalse(_is_persistent_referee_track(1, 64))

    def test_persistent_referee_overlap_deletes_track(self):
        self.assertTrue(_is_persistent_referee_track(8, 10))

    def test_three_referee_overlaps_delete_even_long_track(self):
        self.assertTrue(_is_persistent_referee_track(3, 255))

    def test_short_ambiguous_track_is_preserved(self):
        self.assertFalse(_is_persistent_referee_track(2, 2))


if __name__ == "__main__":
    unittest.main()
