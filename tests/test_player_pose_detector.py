import unittest

import numpy as np

from backend.app.detection.player_pose_detector import (
    _candidate_player_crops,
    _match_player_poses,
    _translate_pose,
    attach_player_poses,
)


class PlayerPoseDetectorTests(unittest.TestCase):
    def test_overlap_matching_is_one_to_one(self):
        players = {
            4: {"bbox": [0, 0, 50, 100]},
            8: {"bbox": [60, 0, 110, 100]},
        }
        poses = [
            {"bbox": [2, 2, 48, 98], "confidence": 0.9},
            {"bbox": [62, 2, 108, 98], "confidence": 0.8},
        ]

        matched = _match_player_poses(players, poses)

        self.assertEqual(matched[4], poses[0])
        self.assertEqual(matched[8], poses[1])

    def test_pose_attachment_does_not_mutate_raw_tracks(self):
        tracks = [{4: {"bbox": [0, 0, 50, 100]}}]
        poses = [{4: {"bbox": [0, 0, 50, 100], "confidence": 0.9}}]

        enriched = attach_player_poses(tracks, poses)

        self.assertNotIn("pose", tracks[0][4])
        self.assertEqual(enriched[0][4]["pose"], poses[0][4])

    def test_pose_attachment_requires_aligned_frames(self):
        with self.assertRaisesRegex(ValueError, "must align"):
            attach_player_poses([{}], [])

    def test_pose_crops_include_only_players_near_ball_candidates(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        players = [{
            4: {"bbox": [20, 20, 80, 180]},
            8: {"bbox": [220, 20, 280, 180]},
        }]
        ball_tracks = [{1: {"candidates": [{
            "bbox": [45, 80, 55, 90],
            "confidence": 0.8,
        }]}}]

        requests, crops = _candidate_player_crops(
            [frame],
            players,
            ball_tracks,
        )

        self.assertEqual([request["player_id"] for request in requests], [4])
        self.assertEqual(crops[0].shape, (640, 192, 3))

    def test_pose_coordinates_are_restored_to_source_frame(self):
        pose = {
            "bbox": [20, 40, 60, 120],
            "keypoints_xy": [[30, 80]],
        }

        translated = _translate_pose(
            pose,
            origin=(100, 200),
            scale=(2.0, 4.0),
        )

        self.assertEqual(translated["bbox"], [110.0, 210.0, 130.0, 230.0])
        self.assertEqual(translated["keypoints_xy"], [[115.0, 220.0]])


if __name__ == "__main__":
    unittest.main()
