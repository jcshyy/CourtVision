import importlib.util
import builtins
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


fake_cv2 = types.ModuleType("cv2")
fake_cv2.COLOR_BGR2RGB = 1
fake_cv2.cvtColor = lambda image, conversion: image
fake_cv2.mean = lambda image: (0, 0, 0, 0)

fake_pil = types.ModuleType("PIL")
fake_pil.Image = types.SimpleNamespace(fromarray=lambda image: image)

fake_transformers = types.ModuleType("transformers")
fake_transformers.CLIPModel = types.SimpleNamespace(from_pretrained=lambda name: None)
fake_transformers.CLIPProcessor = types.SimpleNamespace(from_pretrained=lambda name: None)

MODULE_PATH = (
    Path(__file__).parents[1]
    / "backend"
    / "app"
    / "team_assignment"
    / "team_assigner.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("courtvision_team_assigner", MODULE_PATH)
team_assigner = importlib.util.module_from_spec(MODULE_SPEC)
with patch.dict(
    sys.modules,
    {
        "cv2": fake_cv2,
        "PIL": fake_pil,
        "transformers": fake_transformers,
    },
):
    MODULE_SPEC.loader.exec_module(team_assigner)


def _synthetic_jersey_pixels(jersey_color, number_color):
    return (
        [jersey_color] * 700
        + [number_color] * 100
        + [(35, 70, 105)] * 120  # Dark skin-like pixels.
        + [(20, 20, 20)] * 120  # Shadow/background pixels.
    )


def _discovery_result(prototypes=((10, 20, 30), (210, 220, 230))):
    return {
        "status": "confident",
        "reason": None,
        "prototypes": prototypes,
        "confidence": {
            "candidate_observation_count": 8,
            "accepted_observation_count": 8,
            "accepted_observation_fraction": 1.0,
            "observed_track_count": 4,
            "eligible_track_count": 4,
            "prototype_separation": 346.41,
            "cluster_support": [2, 2],
            "track_observation_agreement": 1.0,
        },
    }


class TeamDiscoveryTests(unittest.TestCase):
    def test_automatic_assignment_module_does_not_import_clip_dependencies(self):
        module_spec = importlib.util.spec_from_file_location(
            "courtvision_team_assigner_without_clip",
            MODULE_PATH,
        )
        module = importlib.util.module_from_spec(module_spec)
        real_import = builtins.__import__

        def reject_clip_dependencies(name, *args, **kwargs):
            if name == "PIL" or name.startswith("transformers"):
                raise AssertionError(
                    f"Automatic assignment imported optional dependency {name}"
                )
            return real_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {"cv2": fake_cv2}), patch(
            "builtins.__import__",
            side_effect=reject_clip_dependencies,
        ):
            module_spec.loader.exec_module(module)

        self.assertTrue(module.TeamAssigner().use_discovered_colors)

    def test_configured_labels_report_missing_transformers_actionably(self):
        assigner = team_assigner.TeamAssigner("white jersey", "dark jersey")
        real_import = builtins.__import__

        def reject_transformers(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_transformers):
            with self.assertRaisesRegex(
                RuntimeError,
                "Install backend/requirements.txt",
            ):
                assigner.load_model()

    def test_default_assignment_discovers_colors_without_loading_clip(self):
        assigner = team_assigner.TeamAssigner()
        frames = [object()]
        tracks = [{10: {"bbox": [0, 0, 10, 20]}, 20: {"bbox": [0, 0, 10, 20]}}]

        with patch.object(
            team_assigner,
            "_discover_team_colors_result",
            return_value=_discovery_result(),
        ), patch.object(assigner, "load_model") as load_model, patch.object(
            assigner,
            "get_player_jersey_color",
            side_effect=[(12, 22, 32), (205, 215, 225)],
        ):
            assignments = assigner.get_player_teams_across_frames(frames, tracks)

        self.assertEqual(assignments, [{10: 1, 20: 2}])
        self.assertEqual(assigner.team_colors, {1: (10, 20, 30), 2: (210, 220, 230)})
        self.assertEqual(
            assigner.assignment_metadata["discovery_confidence"],
            _discovery_result()["confidence"],
        )
        load_model.assert_not_called()

    def test_indistinct_samples_return_needs_team_colors(self):
        assigner = team_assigner.TeamAssigner()

        with patch.object(
            team_assigner,
            "_discover_team_colors_result",
            return_value={
                "status": "needs_team_colors",
                "reason": "insufficient_distinct_team_prototypes",
                "prototypes": None,
                "confidence": {
                    "eligible_track_count": 1,
                    "prototype_separation": None,
                    "cluster_support": [],
                },
            },
        ), patch.object(assigner, "load_model") as load_model, self.assertLogs(
            "courtvision_team_assigner",
            level="WARNING",
        ):
            with self.assertRaises(team_assigner.NeedsTeamColorsError) as raised:
                assigner.get_player_teams_across_frames(
                    [object()],
                    [{7: {"bbox": [0, 0, 10, 20]}}],
                )

        self.assertEqual(raised.exception.result["status"], "needs_team_colors")
        self.assertEqual(
            raised.exception.result["reason"],
            "insufficient_distinct_team_prototypes",
        )
        self.assertEqual(
            raised.exception.result["discovery_confidence"]["eligible_track_count"],
            1,
        )
        load_model.assert_not_called()

    def test_user_colors_are_normalized_and_skip_automatic_discovery(self):
        assigner = team_assigner.TeamAssigner(
            team_1_color="ffffff",
            team_2_color=" #c8102e ",
        )
        frames = [object()] * 3
        tracks = [{7: {"bbox": [0, 0, 10, 20]}}] * 3

        with patch.object(
            team_assigner,
            "_discover_team_colors_result",
        ) as discover, patch.object(
            assigner,
            "get_player_jersey_color",
            return_value=team_assigner._color_prototype("#C8102E"),
        ):
            assignments = assigner.get_player_teams_across_frames(frames, tracks)

        self.assertTrue(all(frame[7] == 2 for frame in assignments))
        self.assertEqual(assigner.assignment_mode, "user_colors")
        self.assertEqual(assigner.normalized_team_colors, ("#FFFFFF", "#C8102E"))
        self.assertEqual(
            assigner.assignment_metadata,
            {
                "algorithm_version": "v11",
                "assignment_mode": "user_colors",
                "team_colors": ["#FFFFFF", "#C8102E"],
                "discovery_confidence": None,
            },
        )
        discover.assert_not_called()

    def test_user_colors_require_a_valid_distinct_pair(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            team_assigner.TeamAssigner(team_1_color="#FFFFFF")
        with self.assertRaisesRegex(ValueError, "#RRGGBB"):
            team_assigner.TeamAssigner(
                team_1_color="not-a-color",
                team_2_color="#FF0000",
            )
        with self.assertRaisesRegex(ValueError, "sufficiently distinct"):
            team_assigner.TeamAssigner(
                team_1_color="#FFFFFF",
                team_2_color="#FEFEFE",
            )

    def test_user_colors_reject_shadow_filtered_swatch_actionably(self):
        with self.assertRaisesRegex(
            ValueError,
            r"bright enough.*#202020",
        ):
            team_assigner.TeamAssigner(
                team_1_color="#F0F0F0",
                team_2_color="#202020",
            )

    def test_assignment_cache_identity_changes_only_with_assignment_inputs(self):
        automatic_a = team_assigner.TeamAssigner()
        automatic_b = team_assigner.TeamAssigner()
        guided_a = team_assigner.TeamAssigner(
            team_1_color="#FFFFFF",
            team_2_color="#C8102E",
        )
        guided_b = team_assigner.TeamAssigner(
            team_1_color="#C8102E",
            team_2_color="#FFFFFF",
        )

        self.assertEqual(automatic_a.cache_filename, automatic_b.cache_filename)
        self.assertNotEqual(automatic_a.cache_filename, guided_a.cache_filename)
        self.assertNotEqual(guided_a.cache_filename, guided_b.cache_filename)
        self.assertIn("user_colors", guided_a.cache_filename)
        self.assertNotEqual(guided_a.metadata_filename, guided_b.metadata_filename)

    def test_assignment_cache_hit_restores_discovery_confidence(self):
        assigner = team_assigner.TeamAssigner()
        confidence = _discovery_result()["confidence"]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / assigner.cache_filename
            team_assigner.save_cache(cache_path, [{7: 1}])
            cache_path.with_name(assigner.metadata_filename).write_text(
                json.dumps(
                    {
                        **assigner.assignment_metadata,
                        "discovery_confidence": confidence,
                    }
                ),
                encoding="utf-8",
            )

            assignments = assigner.get_player_teams_across_frames(
                [object()],
                [{7: {"bbox": [0, 0, 10, 20]}}],
                read_from_cache=True,
                cache_path=cache_path,
            )

        self.assertEqual(assignments, [{7: 1}])
        self.assertEqual(assigner.assignment_metadata["discovery_confidence"], confidence)

    def test_color_clustering_rejects_one_indistinct_team(self):
        colors = [(100, 100, 100), (102, 101, 100), (99, 100, 101), (101, 99, 100)]

        self.assertIsNone(team_assigner._cluster_team_colors(colors))

    def test_color_clustering_finds_two_distinct_teams(self):
        colors = [(10, 20, 30), (12, 18, 31), (200, 210, 220), (202, 208, 218)]

        clusters = team_assigner._cluster_team_colors(colors)

        self.assertEqual(set(clusters), {(11, 19, 30), (201, 209, 219)})

    def test_hsv_jersey_features_separate_red_and_white_despite_contamination(self):
        red = _synthetic_jersey_pixels((0, 0, 220), (240, 240, 240))
        white = _synthetic_jersey_pixels((220, 220, 220), (0, 0, 160))
        dim_red = _synthetic_jersey_pixels((0, 0, 130), (180, 180, 180))

        red_feature = team_assigner._jersey_feature(red)
        white_feature = team_assigner._jersey_feature(white)
        dim_red_feature = team_assigner._jersey_feature(dim_red)

        self.assertLess(
            team_assigner._color_distance(red_feature, dim_red_feature),
            team_assigner._color_distance(red_feature, white_feature),
        )
        self.assertGreater(red_feature[2], white_feature[2] + 100)

    def test_discovery_requires_multiple_observations_per_tracked_player(self):
        frames = [object()] * 3
        tracks = [
            {
                1: {"bbox": [0, 0, 10, 20]},
                2: {"bbox": [0, 0, 10, 20]},
                3: {"bbox": [0, 0, 10, 20]},
                4: {"bbox": [0, 0, 10, 20]},
                99: {"bbox": [0, 0, 10, 20]},
            },
            {
                1: {"bbox": [0, 0, 10, 20]},
                2: {"bbox": [0, 0, 10, 20]},
                3: {"bbox": [0, 0, 10, 20]},
                4: {"bbox": [0, 0, 10, 20]},
            },
            {
                1: {"bbox": [0, 0, 10, 20]},
                2: {"bbox": [0, 0, 10, 20]},
                3: {"bbox": [0, 0, 10, 20]},
                4: {"bbox": [0, 0, 10, 20]},
            },
        ]
        observations = [
            (250, 0, 250, 220),
            (248, 2, 248, 215),
            (0, 0, 5, 230),
            (2, 0, 4, 225),
            (120, 120, 120, 120),  # One-off noisy track; must be ignored.
            (245, 3, 247, 190),
            (250, 0, 250, 205),
            (3, 0, 5, 200),
            (0, 2, 3, 210),
            (252, 0, 249, 235),
            (246, 2, 251, 225),
            (1, 1, 4, 240),
            (2, 0, 6, 215),
        ]

        with patch.object(team_assigner, "_jersey_color", side_effect=observations):
            clusters = team_assigner._discover_team_colors(frames, tracks)

        self.assertIsNotNone(clusters)
        self.assertTrue(any(color[2] > 200 for color in clusters))
        self.assertTrue(any(color[2] < 20 for color in clusters))

    def test_discovery_confidence_rejects_single_track_cluster(self):
        frames = [object(), object()]
        tracks = [
            {
                1: {"bbox": [0, 0, 10, 20]},
                2: {"bbox": [0, 0, 10, 20]},
                3: {"bbox": [0, 0, 10, 20]},
                4: {"bbox": [0, 0, 10, 20]},
            }
        ] * 2
        team_one = (10, 20, 30)
        team_two = (210, 220, 230)

        with patch.object(
            team_assigner,
            "_jersey_color",
            side_effect=[
                team_one,
                team_one,
                team_one,
                team_two,
                team_one,
                team_one,
                team_one,
                team_two,
            ],
        ):
            result = team_assigner._discover_team_colors_result(frames, tracks)

        self.assertEqual(result["status"], "needs_team_colors")
        self.assertEqual(result["reason"], "insufficient_cluster_support")
        self.assertEqual(sorted(result["confidence"]["cluster_support"]), [1, 3])
        self.assertEqual(result["confidence"]["eligible_track_count"], 4)
        self.assertEqual(result["confidence"]["accepted_observation_fraction"], 1.0)
        self.assertGreater(result["confidence"]["prototype_separation"], 30)
        self.assertEqual(result["confidence"]["track_observation_agreement"], 1.0)

    def test_discovery_confidence_rejects_unstable_track_observations(self):
        frames = [object()] * 5
        tracks = [
            {
                1: {"bbox": [0, 0, 10, 20]},
                2: {"bbox": [0, 0, 10, 20]},
                3: {"bbox": [0, 0, 10, 20]},
                4: {"bbox": [0, 0, 10, 20]},
            }
        ] * 5
        low = (0, 0, 0)
        high = (100, 0, 0)
        observations = []
        for frame_index in range(5):
            observations.extend(
                [
                    low if frame_index < 3 else high,
                    low if frame_index < 3 else high,
                    high if frame_index < 3 else low,
                    high if frame_index < 3 else low,
                ]
            )

        with patch.object(
            team_assigner,
            "_jersey_color",
            side_effect=observations,
        ):
            result = team_assigner._discover_team_colors_result(frames, tracks)

        self.assertEqual(result["status"], "needs_team_colors")
        self.assertEqual(result["reason"], "unstable_track_observations")
        self.assertEqual(sorted(result["confidence"]["cluster_support"]), [2, 2])
        self.assertEqual(result["confidence"]["track_observation_agreement"], 0.6)

    def test_discovery_agreement_boundary_is_accepted(self):
        frames = [object()] * 5
        tracks = [
            {
                1: {"bbox": [0, 0, 10, 20]},
                2: {"bbox": [0, 0, 10, 20]},
                3: {"bbox": [0, 0, 10, 20]},
                4: {"bbox": [0, 0, 10, 20]},
            }
        ] * 5
        low = (0, 0, 0)
        high = (100, 0, 0)
        observations = []
        for frame_index in range(5):
            observations.extend(
                [
                    high if frame_index == 4 else low,
                    high if frame_index == 4 else low,
                    low if frame_index == 4 else high,
                    low if frame_index == 4 else high,
                ]
            )

        with patch.object(
            team_assigner,
            "_jersey_color",
            side_effect=observations,
        ):
            result = team_assigner._discover_team_colors_result(frames, tracks)

        self.assertEqual(result["status"], "confident")
        self.assertEqual(result["confidence"]["track_observation_agreement"], 0.8)

    def test_single_noisy_refresh_does_not_flip_player_team(self):
        assigner = team_assigner.TeamAssigner()
        assigner.team_colors = {1: (10, 20, 30), 2: (210, 220, 230)}

        with patch.object(
            assigner,
            "get_player_jersey_color",
            side_effect=[(10, 20, 30), (210, 220, 230)],
        ):
            initial_team = assigner.get_player_team(object(), [0, 0, 10, 20], 7)
            refreshed_team = assigner.get_player_team(
                object(),
                [0, 0, 10, 20],
                7,
                refresh=True,
            )

        self.assertEqual(initial_team, 1)
        self.assertEqual(refreshed_team, 1)
        self.assertEqual(assigner.player_team_votes[7], [1, 2])

    def test_invalid_observation_keeps_new_track_unknown(self):
        assigner = team_assigner.TeamAssigner()
        assigner.team_colors = {1: (10, 20, 30), 2: (210, 220, 230)}

        with patch.object(assigner, "get_player_jersey_color", return_value=None):
            team = assigner.get_player_team(object(), [0, 0, 10, 20], 7)

        self.assertEqual(team, -1)
        self.assertEqual(assigner.player_team_votes[7], [])

    def test_ambiguous_prototype_margin_keeps_new_track_unknown(self):
        assigner = team_assigner.TeamAssigner()
        assigner.team_colors = {1: (0, 0, 0), 2: (100, 0, 0)}

        with patch.object(
            assigner,
            "get_player_jersey_color",
            return_value=(52, 0, 0),
        ):
            assigned_team = assigner.get_player_team(
                object(), [0, 0, 10, 20], 7
            )

        self.assertEqual(assigned_team, -1)
        self.assertEqual(assigner.player_team_votes[7], [])

    def test_prototype_margin_boundary_and_team_order_are_equivalent(self):
        color = (55, 0, 0)
        forward = {1: (0, 0, 0), 2: (100, 0, 0)}
        reversed_teams = {1: (100, 0, 0), 2: (0, 0, 0)}

        self.assertEqual(
            team_assigner._confident_nearest_team(color, forward),
            2,
        )
        self.assertEqual(
            team_assigner._confident_nearest_team(color, reversed_teams),
            1,
        )
        self.assertIsNone(
            team_assigner._confident_nearest_team((54, 0, 0), forward)
        )

    def test_offline_track_ignores_ambiguous_observations(self):
        assigner = team_assigner.TeamAssigner()
        frames = [object()] * 5
        tracks = [{7: {"bbox": [0, 0, 10, 20]}}] * 5

        with patch.object(
            team_assigner,
            "_discover_team_colors_result",
            return_value=_discovery_result(((0, 0, 0), (100, 0, 0))),
        ), patch.object(
            assigner,
            "get_player_jersey_color",
            side_effect=[
                (49, 0, 0),
                (51, 0, 0),
                (52, 0, 0),
                (90, 0, 0),
                (95, 0, 0),
            ],
        ):
            assignments = assigner.get_player_teams_across_frames(frames, tracks)

        self.assertTrue(all(frame[7] == 2 for frame in assignments))

    def test_invalid_refresh_preserves_established_track_team(self):
        assigner = team_assigner.TeamAssigner()
        assigner.team_colors = {1: (10, 20, 30), 2: (210, 220, 230)}

        with patch.object(
            assigner,
            "get_player_jersey_color",
            side_effect=[(210, 220, 230), None],
        ):
            initial_team = assigner.get_player_team(object(), [0, 0, 10, 20], 7)
            refreshed_team = assigner.get_player_team(
                object(),
                [0, 0, 10, 20],
                7,
                refresh=True,
            )

        self.assertEqual(initial_team, 2)
        self.assertEqual(refreshed_team, 2)
        self.assertEqual(assigner.player_team_votes[7], [2])

    def test_initial_assignment_uses_multiple_views_before_first_frame(self):
        assigner = team_assigner.TeamAssigner(initial_observations=3)
        frames = [object()] * 4
        tracks = [{22: {"bbox": [0, 0, 10, 20]}}] * 4

        with patch.object(
            team_assigner,
            "_discover_team_colors_result",
            return_value=_discovery_result(),
        ), patch.object(
            assigner,
            "get_player_jersey_color",
            side_effect=[
                (10, 20, 30),       # Poor first view suggests team 1.
                (210, 220, 230),    # Clearer views identify team 2.
                (205, 215, 225),
                (215, 225, 235),
            ],
        ):
            assignments = assigner.get_player_teams_across_frames(frames, tracks)

        self.assertTrue(all(frame[22] == 2 for frame in assignments))
        self.assertEqual(assigner.player_team_votes[22], [2])

    def test_sustained_refresh_votes_can_change_player_team(self):
        assigner = team_assigner.TeamAssigner()
        assigner.team_colors = {1: (10, 20, 30), 2: (210, 220, 230)}

        with patch.object(
            assigner,
            "get_player_jersey_color",
            side_effect=[
                (10, 20, 30),
                (210, 220, 230),
                (210, 220, 230),
            ],
        ):
            teams = [
                assigner.get_player_team(object(), [0, 0, 10, 20], 7),
                assigner.get_player_team(
                    object(),
                    [0, 0, 10, 20],
                    7,
                    refresh=True,
                ),
                assigner.get_player_team(
                    object(),
                    [0, 0, 10, 20],
                    7,
                    refresh=True,
                ),
            ]

        self.assertEqual(teams, [1, 1, 2])

    def test_offline_track_assignment_backfills_complete_track_evidence(self):
        assigner = team_assigner.TeamAssigner()
        frames = [object()] * 101
        tracks = [{7: {"bbox": [0, 0, 10, 20]}}] * 101

        with patch.object(
            team_assigner,
            "_discover_team_colors_result",
            return_value=_discovery_result(),
        ), patch.object(
            assigner,
            "get_player_jersey_color",
            side_effect=[
                (10, 20, 30),
                (10, 20, 30),
                (10, 20, 30),
                *[(210, 220, 230)] * 98,
            ],
        ):
            assignments = assigner.get_player_teams_across_frames(frames, tracks)

        self.assertEqual(assignments[0][7], 2)
        self.assertEqual(assignments[49][7], 2)
        self.assertEqual(assignments[50][7], 2)
        self.assertEqual(assignments[100][7], 2)

    def test_complete_track_evidence_is_equivalent_when_team_ids_are_swapped(self):
        frames = [object()] * 5
        tracks = [{7: {"bbox": [0, 0, 10, 20]}}] * 5
        observations = [(12, 22, 32), (205, 215, 225), (210, 220, 230),
                        (208, 218, 228), (212, 222, 232)]

        results = []
        for prototypes in (
            {1: (10, 20, 30), 2: (210, 220, 230)},
            {1: (210, 220, 230), 2: (10, 20, 30)},
        ):
            assigner = team_assigner.TeamAssigner()
            with patch.object(
                team_assigner,
                "_discover_team_colors_result",
                return_value=_discovery_result(tuple(prototypes.values())),
            ), patch.object(
                assigner,
                "get_player_jersey_color",
                side_effect=observations,
            ):
                results.append(assigner.get_player_teams_across_frames(frames, tracks))

        self.assertTrue(all(frame[7] == 2 for frame in results[0]))
        self.assertTrue(all(frame[7] == 1 for frame in results[1]))


if __name__ == "__main__":
    unittest.main()
