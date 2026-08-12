import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from main import events_to_overlay_arrays, filter_ball_tracks_with_pose, write_analysis_manifest


class _PoseDetector:
    def get_player_poses(self, frames, player_tracks, **kwargs):
        return [{7: {"keypoints_xy": [[1, 1]], "keypoint_confidences": [0.9]}}]


class _BallTracker:
    def __init__(self):
        self.saw_pose = False
        self.adaptive_saw_pose = False
        self.detector_backend = "yolo"

    def enhance_tracks_with_adaptive_crops(
        self,
        frames,
        tracks,
        player_tracks,
        **kwargs,
    ):
        self.adaptive_saw_pose = "pose" in player_tracks[0][7]
        self.adaptive_cache_path = kwargs.get("cache_path")
        return tracks

    def remove_wrong_detections(
        self,
        tracks,
        player_tracks=None,
        discontinuity_frames=None,
    ):
        self.saw_pose = "pose" in player_tracks[0][7]
        self.discontinuity_frames = discontinuity_frames
        return tracks

    def interpolate_positions(self, tracks, discontinuity_frames=None):
        self.interpolation_discontinuities = discontinuity_frames
        return tracks

    def build_semantic_tracks(
        self,
        tracks,
        player_tracks,
        *,
        fused_tracks,
        discontinuity_frames=None,
    ):
        self.semantic_saw_pose = "pose" in player_tracks[0][7]
        self.semantic_fused_tracks = fused_tracks
        return [{1: {"bbox": [8, 8, 10, 10], "semantic_track": True}}]


class MainPipelineTests(unittest.TestCase):
    def test_render_keeps_compact_event_huds_but_not_tactical_view_in_video(self):
        source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")

        self.assertIn("PassInterceptionDrawer(", source)
        self.assertIn("TeamBallControlDrawer(", source)
        self.assertNotIn("TacticalViewDrawer(", source)

    def test_events_are_mapped_to_overlay_arrays_without_changing_type(self):
        passes, interceptions = events_to_overlay_arrays(
            [
                {"type": "pass", "frame_index": 1, "to_team_id": 1},
                {"type": "interception", "frame_index": 3, "to_team_id": 2},
            ],
            5,
        )

        self.assertEqual(passes, [-1, 1, -1, -1, -1])
        self.assertEqual(interceptions, [-1, -1, -1, 2, -1])

    def test_pose_is_attached_before_ball_candidate_filtering(self):
        tracker = _BallTracker()
        players, ball = filter_ball_tracks_with_pose(
            [object()],
            [{7: {"bbox": [0, 0, 10, 20]}}],
            [{1: {"bbox": [2, 2, 4, 4]}}],
            _PoseDetector(),
            tracker,
            adaptive_cache_path=Path("adaptive.pkl"),
            discontinuity_frames=[3],
        )

        self.assertTrue(tracker.saw_pose)
        self.assertTrue(tracker.adaptive_saw_pose)
        self.assertEqual(tracker.adaptive_cache_path, Path("adaptive.pkl"))
        self.assertIn("pose", players[0][7])
        self.assertEqual(ball[0][1]["bbox"], [2, 2, 4, 4])
        self.assertEqual(tracker.discontinuity_frames, [3])
        self.assertEqual(tracker.interpolation_discontinuities, [3])

    def test_hybrid_pipeline_returns_separate_semantic_track(self):
        tracker = _BallTracker()
        tracker.detector_backend = "hybrid"

        players, fused, semantic = filter_ball_tracks_with_pose(
            [object()],
            [{7: {"bbox": [0, 0, 10, 20]}}],
            [{1: {"bbox": [2, 2, 4, 4]}}],
            _PoseDetector(),
            tracker,
            include_semantic_track=True,
        )

        self.assertIn("pose", players[0][7])
        self.assertEqual(fused[0][1]["bbox"], [2, 2, 4, 4])
        self.assertTrue(semantic[0][1]["semantic_track"])
        self.assertTrue(tracker.semantic_saw_pose)
        self.assertIs(tracker.semantic_fused_tracks, fused)

    def test_analysis_manifest_preserves_candidates_and_unknowns(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "analysis.json"
            write_analysis_manifest(
                output,
                fps=10,
                frame_count=2,
                court_width=300,
                court_height=161,
                tactical_player_positions=[{7: [150.25, 80.5], 8: [80, 40]}, {}],
                player_assignment=[{7: np.int64(1), 8: np.int64(-1)}, {}],
                ball_acquisition=[np.int64(7), -1],
                events=[
                    {
                        "type": "pass",
                        "frame_index": 1,
                        "from_team_id": 1,
                        "to_team_id": None,
                        "gap_frames": 2,
                    }
                ],
                tactical_diagnostics={"fallback_used": [1]},
                assignment_metadata={"discovery_confidence": None},
            )

            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(payload["beta"])
        self.assertEqual(payload["source"]["durationSeconds"], 0.2)
        self.assertEqual(payload["events"][0]["status"], "unknown")
        self.assertEqual(payload["events"][0]["timeSeconds"], 0.1)
        self.assertTrue(payload["frames"][0]["players"][0]["isHolder"])
        self.assertEqual(payload["frames"][0]["players"][0]["teamId"], 1)
        self.assertEqual(payload["frames"][0]["possessionTeamId"], 1)
        self.assertIsNone(payload["frames"][0]["players"][1]["teamId"])
