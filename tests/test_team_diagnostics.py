import importlib.util
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.cache_paths import video_cache_dir
from backend.app.team_assignment import diagnostics
from backend.app.team_assignment.diagnostics import _pixel_decision_summary


fake_cv2 = types.ModuleType("cv2")
fake_cv2.COLOR_BGR2RGB = 1
fake_cv2.COLOR_BGR2GRAY = 2
fake_cv2.CV_64F = 3
fake_cv2.cvtColor = lambda image, conversion: image
fake_cv2.Laplacian = lambda image, depth: types.SimpleNamespace(var=lambda: 42.5)

fake_pil = types.ModuleType("PIL")
fake_pil.Image = types.SimpleNamespace(fromarray=lambda image: image)

fake_transformers = types.ModuleType("transformers")
fake_transformers.CLIPModel = types.SimpleNamespace(from_pretrained=lambda name: None)
fake_transformers.CLIPProcessor = types.SimpleNamespace(from_pretrained=lambda name: None)

TEAM_ASSIGNER_PATH = (
    Path(__file__).parents[1]
    / "backend"
    / "app"
    / "team_assignment"
    / "team_assigner.py"
)
TEAM_ASSIGNER_SPEC = importlib.util.spec_from_file_location(
    "backend.app.team_assignment.team_assigner",
    TEAM_ASSIGNER_PATH,
)
team_assigner = importlib.util.module_from_spec(TEAM_ASSIGNER_SPEC)
with patch.dict(
    sys.modules,
    {
        "cv2": fake_cv2,
        "PIL": fake_pil,
        "transformers": fake_transformers,
        "backend.app.team_assignment.team_assigner": team_assigner,
    },
):
    TEAM_ASSIGNER_SPEC.loader.exec_module(team_assigner)


class FakeCrop:
    shape = (30, 30, 3)
    size = 2700

    def reshape(self, *shape):
        return self

    def tolist(self):
        return [(0, 0, 220)] * 900


class FakeFrame:
    shape = (720, 1280, 3)

    def __getitem__(self, key):
        return FakeCrop()


class SmallCrop(FakeCrop):
    shape = (29, 30, 3)
    size = 2610

    def tolist(self):
        return [(0, 0, 220)] * 870


class SmallCropFrame(FakeFrame):
    def __getitem__(self, key):
        return SmallCrop()


class HalfScaleCrop(FakeCrop):
    shape = (15, 15, 3)
    size = 675

    def tolist(self):
        return [(0, 0, 220)] * 225


class HalfScaleFrame(FakeFrame):
    shape = (360, 640, 3)

    def __getitem__(self, key):
        return HalfScaleCrop()


class TeamDiagnosticTests(unittest.TestCase):
    def _cached_video_and_tracks(self, root):
        video = root / "video.mp4"
        video.write_bytes(b"diagnostic-video")
        cache_dir = video_cache_dir(root / "stubs", video)
        cache_dir.mkdir(parents=True)
        tracks = [{22: {"bbox": [0, 0, 60, 80]}}] * 3
        with (cache_dir / "player_track_stubs.pkl").open("wb") as file:
            pickle.dump(tracks, file)
        return video, tracks

    def test_observation_exposes_current_acceptance_and_crop_metrics(self):
        observation = team_assigner._jersey_observation(
            FakeFrame(),
            [-5, 10, 55, 90],
        )

        self.assertFalse(observation["accepted"])
        self.assertEqual(observation["rejection_reason"], "edge_clipped")
        self.assertTrue(observation["edge_clipped"])
        self.assertEqual(observation["torso_width"], 30)
        self.assertEqual(observation["torso_height"], 30)
        self.assertEqual(observation["total_pixels"], 900)
        self.assertEqual(observation["visible_pixels"], 900)
        self.assertEqual(observation["filtered_pixels"], 900)
        self.assertEqual(observation["blur_variance"], 42.5)
        self.assertEqual(observation["visible_fraction"], 1.0)
        self.assertIsNone(observation["feature"])

    def test_invalid_bbox_is_rejected_with_reason(self):
        observation = team_assigner._jersey_observation(
            FakeFrame(),
            [20, 20, 20, 50],
        )

        self.assertFalse(observation["accepted"])
        self.assertEqual(observation["rejection_reason"], "invalid_bbox")
        self.assertIsNone(observation["feature"])

    def test_small_torso_is_rejected_despite_valid_color_pixels(self):
        observation = team_assigner._jersey_observation(
            SmallCropFrame(),
            [0, 0, 60, 80],
        )

        self.assertFalse(observation["accepted"])
        self.assertEqual(observation["rejection_reason"], "torso_too_small")
        self.assertEqual(observation["torso_area"], 870)
        self.assertEqual(observation["minimum_torso_area"], 900)
        self.assertIsNone(observation["feature"])

    def test_equivalent_downscaled_torso_keeps_same_quality_decision(self):
        full_size = team_assigner._jersey_observation(
            FakeFrame(),
            [0, 0, 60, 80],
        )
        half_size = team_assigner._jersey_observation(
            HalfScaleFrame(),
            [0, 0, 30, 40],
        )

        self.assertTrue(full_size["accepted"])
        self.assertTrue(half_size["accepted"])
        self.assertEqual(full_size["minimum_torso_area"], 900)
        self.assertEqual(half_size["minimum_torso_area"], 225)
        self.assertAlmostEqual(
            full_size["torso_frame_area_fraction"],
            half_size["torso_frame_area_fraction"],
        )

    def test_team_distances_keep_invalid_feature_explicit(self):
        distances = team_assigner._team_distances(
            None,
            {1: (1, 2, 3), 2: (4, 5, 6)},
        )

        self.assertEqual(distances, {1: None, 2: None})
        self.assertIsNone(
            team_assigner._nearest_team(
                None,
                {1: (1, 2, 3), 2: (4, 5, 6)},
            )
        )

    def test_decision_summary_exposes_unenforced_quality_and_weak_margins(self):
        observations = [
            {
                "frame": 24,
                "accepted": True,
                "rejection_reason": None,
                "bootstrap_selected": True,
                "distance_margin": 3.0,
                "nearest_team": 1,
                "computed_team": 1,
                "cached_team": 1,
                "prototype_distances": {1: 20.0, 2: 23.0},
            },
            {
                "frame": 26,
                "accepted": False,
                "rejection_reason": "no_visible_pixels",
                "bootstrap_selected": False,
                "distance_margin": None,
                "nearest_team": None,
                "computed_team": 1,
                "cached_team": 1,
                "prototype_distances": {1: None, 2: None},
            },
            {
                "frame": 100,
                "accepted": True,
                "rejection_reason": None,
                "bootstrap_selected": True,
                "distance_margin": 40.0,
                "nearest_team": 2,
                "computed_team": 2,
                "cached_team": 2,
                "prototype_distances": {1: 70.0, 2: 30.0},
            },
        ]

        summary = _pixel_decision_summary(observations)

        self.assertEqual(
            summary["acceptance_policy"],
            "minimum_torso_area_and_nonempty_feature_with_normalized_prototype_margin",
        )
        self.assertNotIn("distance_margin", summary["measured_but_not_rejected"])
        self.assertEqual(summary["accepted_observation_count"], 2)
        self.assertEqual(summary["rejected_observation_count"], 1)
        self.assertEqual(
            summary["rejection_reason_counts"],
            {"no_visible_pixels": 1},
        )
        self.assertEqual(summary["bootstrap_frames"], [24, 100])
        self.assertEqual(
            summary["nearest_team_transitions"],
            [
                {"frame": 24, "nearest_team": 1},
                {"frame": 100, "nearest_team": 2},
            ],
        )
        self.assertEqual(
            summary["weakest_accepted_margins"][0],
            {
                "frame": 24,
                "distance_margin": 3.0,
                "nearest_team": 1,
                "prototype_distances": {1: 20.0, 2: 23.0},
            },
        )

    def test_pixel_diagnostic_returns_automatic_confidence_failure(self):
        discovery = {
            "status": "needs_team_colors",
            "reason": "insufficient_cluster_support",
            "prototypes": None,
            "confidence": {"cluster_support": [1, 3]},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video, tracks = self._cached_video_and_tracks(root)
            with patch.dict(
                sys.modules,
                {
                    "cv2": fake_cv2,
                    "backend.app.team_assignment.team_assigner": team_assigner,
                },
            ), patch.object(
                diagnostics,
                "_read_video",
                return_value=[FakeFrame()] * len(tracks),
            ), patch.object(
                team_assigner,
                "_discover_team_colors_result",
                return_value=discovery,
            ):
                result = diagnostics.diagnose_team_track(
                    video,
                    root / "stubs",
                    22,
                )

        self.assertEqual(result["status"], "needs_team_colors")
        self.assertEqual(result["assignment_mode"], "automatic")
        self.assertEqual(result["discovery_result"], discovery)
        self.assertEqual(result["visible_frame_range"], [0, 2])
        self.assertEqual(result["observations"], [])

    def test_pixel_diagnostic_accepts_guided_team_colors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video, tracks = self._cached_video_and_tracks(root)
            with patch.dict(
                sys.modules,
                {
                    "cv2": fake_cv2,
                    "backend.app.team_assignment.team_assigner": team_assigner,
                },
            ), patch.object(
                diagnostics,
                "_read_video",
                return_value=[FakeFrame()] * len(tracks),
            ):
                result = diagnostics.diagnose_team_track(
                    video,
                    root / "stubs",
                    22,
                    team_1_color="#C8102E",
                    team_2_color="#FFFFFF",
                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["assignment_mode"], "user_colors")
        self.assertEqual(
            result["normalized_team_colors"],
            ("#C8102E", "#FFFFFF"),
        )
        self.assertIsNone(result["discovery_result"])
        self.assertEqual(
            [observation["computed_team"] for observation in result["observations"]],
            [1, 1, 1],
        )


if __name__ == "__main__":
    unittest.main()
