import unittest

from main import filter_ball_tracks_with_pose


class _PoseDetector:
    def get_player_poses(self, frames, player_tracks, **kwargs):
        return [{7: {"keypoints_xy": [[1, 1]], "keypoint_confidences": [0.9]}}]


class _BallTracker:
    def __init__(self):
        self.saw_pose = False

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


class MainPipelineTests(unittest.TestCase):
    def test_pose_is_attached_before_ball_candidate_filtering(self):
        tracker = _BallTracker()
        players, ball = filter_ball_tracks_with_pose(
            [object()],
            [{7: {"bbox": [0, 0, 10, 20]}}],
            [{1: {"bbox": [2, 2, 4, 4]}}],
            _PoseDetector(),
            tracker,
            discontinuity_frames=[3],
        )

        self.assertTrue(tracker.saw_pose)
        self.assertIn("pose", players[0][7])
        self.assertEqual(ball[0][1]["bbox"], [2, 2, 4, 4])
        self.assertEqual(tracker.discontinuity_frames, [3])
        self.assertEqual(tracker.interpolation_discontinuities, [3])
