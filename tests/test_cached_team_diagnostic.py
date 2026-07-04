import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.cache_paths import video_cache_dir
from backend.app.team_assignment.diagnostics import diagnose_team_track
from scripts.diagnose_team_track import parse_args


class CachedTeamDiagnosticTests(unittest.TestCase):
    def test_cli_accepts_guided_colors_for_pixel_diagnostic(self):
        with patch.object(
            sys,
            "argv",
            [
                "diagnose_team_track",
                "video.mp4",
                "22",
                "--team-1-color",
                "#C8102E",
                "--team-2-color",
                "#FFFFFF",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.team_1_color, "#C8102E")
        self.assertEqual(args.team_2_color, "#FFFFFF")

    def test_cli_rejects_guided_colors_in_cache_only_mode(self):
        with patch.object(
            sys,
            "argv",
            [
                "diagnose_team_track",
                "video.mp4",
                "22",
                "--cache-only",
                "--team-1-color",
                "#C8102E",
                "--team-2-color",
                "#FFFFFF",
            ],
        ), self.assertRaises(SystemExit):
            parse_args()

    def test_cache_only_reports_track_timeline_without_video_decoder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            video.write_bytes(b"diagnostic-video")
            cache_dir = video_cache_dir(root / "stubs", video)
            cache_dir.mkdir(parents=True)

            tracks = [
                {22: {"bbox": [1, 2, 11, 22]}},
                {},
                {22: {"bbox": [3, 4, 13, 24]}},
                {22: {"bbox": [5, 6, 15, 26]}},
            ]
            assignments = [
                {22: 1},
                {},
                {22: 1},
                {22: 2},
            ]
            with (cache_dir / "player_track_stubs.pkl").open("wb") as file:
                pickle.dump(tracks, file)
            with (cache_dir / "player_assignment_stub_v3.pkl").open("wb") as file:
                pickle.dump(assignments, file)

            result = diagnose_team_track(
                video,
                root / "stubs",
                22,
                cache_only=True,
            )

        self.assertEqual(result["mode"], "cache_only")
        self.assertFalse(result["pixel_diagnostics_available"])
        self.assertEqual(result["visible_frame_count"], 3)
        self.assertEqual(result["visible_frame_range"], [0, 3])
        self.assertEqual(
            result["assignment_transitions"],
            [{"frame": 0, "team": 1}, {"frame": 3, "team": 2}],
        )
        self.assertEqual(
            result["decision_summary"],
            {
                "cached_assignment_transitions": [
                    {"frame": 0, "team": 1},
                    {"frame": 3, "team": 2},
                ],
                "unassigned_visible_frames": [],
            },
        )
        self.assertEqual(
            [observation["frame"] for observation in result["observations"]],
            [0, 2, 3],
        )

    def test_cache_only_rejects_missing_track(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            video.write_bytes(b"diagnostic-video")
            cache_dir = video_cache_dir(root / "stubs", video)
            cache_dir.mkdir(parents=True)
            with (cache_dir / "player_track_stubs.pkl").open("wb") as file:
                pickle.dump([{}], file)

            with self.assertRaisesRegex(ValueError, "Track 22"):
                diagnose_team_track(
                    video,
                    root / "stubs",
                    22,
                    cache_only=True,
                )


if __name__ == "__main__":
    unittest.main()
