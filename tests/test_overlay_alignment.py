import unittest
from unittest.mock import patch

import numpy as np

from backend.app.visualization.pass_interception_drawer import (
    PassInterceptionDrawer,
)
from backend.app.visualization.team_ball_control_drawer import TeamBallControlDrawer
from backend.app.visualization.speed_distance_drawer import SpeedAndDistanceDrawer


class OverlayAlignmentTests(unittest.TestCase):
    def test_team_control_drawer_preserves_every_frame(self):
        frames = [object(), object(), object()]
        drawer = TeamBallControlDrawer()
        with patch.object(drawer, "draw_frame", side_effect=lambda frame, *_: frame):
            output = drawer.draw(frames, np.array([-1, 1, 2]))

        self.assertEqual(output, frames)

    def test_pass_drawer_preserves_every_frame(self):
        frames = [object(), object(), object()]
        drawer = PassInterceptionDrawer()
        with patch.object(drawer, "draw_frame", side_effect=lambda frame, *_: frame):
            output = drawer.draw(frames, [-1] * 3, [-1] * 3)

        self.assertEqual(output, frames)

    def test_pass_drawer_labels_only_recent_events(self):
        drawer = PassInterceptionDrawer(event_display_frames=3)
        passes = [-1, 1, -1, -1, -1]
        interceptions = [-1] * 5

        self.assertEqual(
            drawer.get_recent_event(3, passes, interceptions),
            {"type": "pass", "team_id": 1, "frame_index": 1},
        )
        self.assertIsNone(drawer.get_recent_event(4, passes, interceptions))

    def test_possession_percentages_exclude_unknown_frames(self):
        percentages = TeamBallControlDrawer().get_control_percentages(
            np.array([-1, -1, 1, 1, 1, 2])
        )

        self.assertEqual(percentages, (0.75, 0.25))

    def test_no_known_possession_has_zero_percentages(self):
        self.assertEqual(
            TeamBallControlDrawer().get_control_percentages(np.array([-1, -1])),
            (0.0, 0.0),
        )

    def test_speed_overlay_hides_distance_by_default(self):
        self.assertFalse(SpeedAndDistanceDrawer().show_distance)
        self.assertTrue(SpeedAndDistanceDrawer(show_distance=True).show_distance)


if __name__ == "__main__":
    unittest.main()
