import json
import tempfile
import unittest
from pathlib import Path

from backend.app.analytics import build_team_possession
from main import write_analysis_manifest


def _assignments(frame_count, teams):
    return [dict(teams) for _ in range(frame_count)]


class TeamPossessionTests(unittest.TestCase):
    # Players 7 and 8 are team 1; player 9 is team 2; player 5 has no team.
    TEAMS = {7: 1, 8: 1, 9: 2, 5: None}

    def build(self, acquisition, **kwargs):
        options = {"fps": 10, "max_gap_seconds": 1.0, **kwargs}
        return build_team_possession(
            acquisition,
            _assignments(len(acquisition), self.TEAMS),
            **options,
        )

    def test_same_team_pass_keeps_team_possession(self):
        result = self.build([7, 7, -1, -1, -1, 8, 8])

        self.assertEqual([frame["team_id"] for frame in result], [1] * 7)
        self.assertEqual(
            [frame["source"] for frame in result],
            ["holder", "holder", "team_gap_bridged", "team_gap_bridged",
             "team_gap_bridged", "holder", "holder"],
        )

    def test_gaps_that_must_stay_unknown(self):
        cases = {
            "opposite teams": ([7, -1, -1, 9], {}),
            "unknown-team holder": ([7, -1, 5, -1, 8], {}),
            "gap longer than limit": ([7] + [-1] * 11 + [8], {}),
            "scene cut in gap": ([7, -1, -1, 8], {"discontinuity_frames": [2]}),
            "shot release in gap": ([7, -1, -1, 8], {"shot_release_frames": [1]}),
            "opposite-team candidate": (
                [7, -1, -1, 8],
                {"holder_states": [{}, {"candidate_id": 9}, {}, {}]},
            ),
        }
        for name, (acquisition, kwargs) in cases.items():
            with self.subTest(name):
                result = self.build(acquisition, **kwargs)
                gap = [
                    frame for index, frame in enumerate(result)
                    if acquisition[index] == -1
                ]
                self.assertTrue(all(frame["team_id"] is None for frame in gap))

    def test_same_team_candidate_does_not_block_bridge(self):
        result = self.build(
            [7, -1, -1, 8],
            holder_states=[{}, {"candidate_id": 8}, {"candidate_id": None}, {}],
        )

        self.assertEqual([frame["team_id"] for frame in result], [1, 1, 1, 1])

    def test_holderless_clip_edges_stay_unknown(self):
        result = self.build([-1, -1, 7, 7, -1])

        self.assertEqual([frame["team_id"] for frame in result], [None, None, 1, 1, None])

    def test_manifest_publishes_team_possession_and_source(self):
        acquisition = [7, -1, 8]
        assignments = _assignments(3, self.TEAMS)
        team_possession = self.build(acquisition)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            write_analysis_manifest(
                path, fps=10, frame_count=3, court_width=300, court_height=160,
                tactical_player_positions=[{}, {}, {}],
                player_assignment=assignments,
                ball_acquisition=acquisition,
                events=[], tactical_diagnostics={}, assignment_metadata={},
                team_possession=team_possession,
            )
            frames = json.loads(path.read_text(encoding="utf-8"))["frames"]

        self.assertEqual([frame["possessionTeamId"] for frame in frames], [1, 1, 1])
        self.assertEqual(
            [frame["possessionSource"] for frame in frames],
            ["holder", "team_gap_bridged", "holder"],
        )


if __name__ == "__main__":
    unittest.main()
