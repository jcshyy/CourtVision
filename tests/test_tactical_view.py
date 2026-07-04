import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _FakeHomography:
    def __init__(self, source, target, **kwargs):
        self.inlier_count = len(source)

    def transform_points(self, points):
        return [_Point(point) for point in points]


class _Point(list):
    def tolist(self):
        return list(self)


fake_cv2 = types.ModuleType("cv2")
fake_cv2.RANSAC = 8
fake_cv2.error = RuntimeError

fake_numpy = types.ModuleType("numpy")
fake_numpy.float32 = float
fake_numpy.array = lambda value, dtype=None: value

fake_analytics = types.ModuleType("backend.app.analytics")
fake_analytics.__path__ = []
fake_homography = types.ModuleType("backend.app.analytics.homography")
fake_homography.Homography = _FakeHomography

fake_geometry = types.ModuleType("backend.app.utils.geometry")
fake_geometry.euclidean_distance = lambda first, second: 0

MODULE_PATH = (
    Path(__file__).parents[1] / "backend" / "app" / "analytics" / "tactical_view.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "courtvision_tactical_view",
    MODULE_PATH,
)
tactical_view = importlib.util.module_from_spec(MODULE_SPEC)
with patch.dict(
    sys.modules,
    {
        "cv2": fake_cv2,
        "numpy": fake_numpy,
        "backend.app.analytics": fake_analytics,
        "backend.app.analytics.homography": fake_homography,
        "backend.app.utils.geometry": fake_geometry,
    },
):
    MODULE_SPEC.loader.exec_module(tactical_view)


class _TensorLike:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class _Keypoints:
    def __init__(self, points):
        self.xy = _TensorLike([points])


class TacticalViewHomographyTests(unittest.TestCase):
    def setUp(self):
        self.converter = tactical_view.TacticalViewConverter("unused.png")
        self.player_tracks = [{1: {"bbox": [140, 80, 160, 100]}}]

    def test_insufficient_keypoints_are_logged_with_frame_index(self):
        keypoints = [_Keypoints([[10, 10], [20, 20], [30, 30]])]

        with self.assertLogs(
            "courtvision_tactical_view",
            level="WARNING",
        ) as logs:
            positions = self.converter.transform_players_to_tactical_view(
                keypoints,
                self.player_tracks,
            )

        self.assertEqual(positions, [{}])
        self.assertIn("insufficient_keypoints=1", logs.output[0])
        self.assertIn("homography_unavailable=1", logs.output[0])
        self.assertIn("sample frames: [0]", logs.output[0])

    def test_rejected_homography_is_logged_and_not_used(self):
        valid_points = [[0, 0] for _ in range(18)]
        for index in (8, 9, 12, 13, 16, 17):
            valid_points[index] = list(self.converter.key_points[index])

        with patch.object(
            tactical_view,
            "_homography_is_consistent",
            return_value=False,
        ):
            with self.assertLogs(
                "courtvision_tactical_view",
                level="WARNING",
            ) as logs:
                positions = self.converter.transform_players_to_tactical_view(
                    [_Keypoints(valid_points)],
                    self.player_tracks,
                )

        self.assertEqual(positions, [{}])
        self.assertIn("rejected_homography=1", logs.output[0])
        self.assertIn("homography_unavailable=1", logs.output[0])

    def test_last_good_homography_fallback_is_bounded_and_logged(self):
        valid_points = [[0, 0] for _ in range(18)]
        for index in (8, 9, 12, 13, 16, 17):
            valid_points[index] = list(self.converter.key_points[index])

        insufficient_points = [[0, 0] for _ in range(18)]
        for index in (8, 9, 12):
            insufficient_points[index] = list(self.converter.key_points[index])

        fallback_count = self.converter.max_homography_fallback_frames
        keypoints = [_Keypoints(valid_points)] + [
            _Keypoints(insufficient_points)
            for _ in range(fallback_count + 1)
        ]
        player_tracks = self.player_tracks * len(keypoints)

        with patch.object(
            tactical_view,
            "_homography_is_consistent",
            return_value=True,
        ):
            with self.assertLogs(
                "courtvision_tactical_view",
                level="WARNING",
            ) as logs:
                positions = self.converter.transform_players_to_tactical_view(
                    keypoints,
                    player_tracks,
                )

        self.assertIn(1, positions[0])
        self.assertTrue(all(1 in frame for frame in positions[1 : fallback_count + 1]))
        self.assertEqual(positions[-1], {})
        self.assertIn(f"fallback_used={fallback_count}", logs.output[0])
        self.assertIn("homography_unavailable=1", logs.output[0])


if __name__ == "__main__":
    unittest.main()
