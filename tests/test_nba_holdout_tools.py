import json
import tempfile
import unittest

from scripts.extract_nba_holdout_annotation_frames import sampled_indices
from pathlib import Path

from scripts.build_nba_holdout import validate_selection


def _selection():
    clips = [
        {
            "id": f"nba_{index:03d}",
            "cohort": "nba",
            "source": "bard",
            "source_path": f"clip_{index}.mp4",
            "game_id": f"game_{index}",
            "coverage_tags": [],
            "copy_mode": "full",
        }
        for index in range(24)
    ]
    for clip_id, cohort in (
        ("ncaa_m_001", "ncaa_mens"),
        ("ncaa_w_001", "ncaa_womens"),
        ("gleague_001", "nba_g_league"),
        ("fiba_001", "fiba"),
        ("women_pro_001", "womens_pro"),
    ):
        clips.append({
            "id": clip_id,
            "cohort": cohort,
            "source": "basket",
            "source_path": f"{clip_id}.mp4",
            "coverage_tags": [],
            "copy_mode": "window",
            "start_seconds": 10,
            "duration_seconds": 25,
        })
    clips.append({
        "id": "fixed_001",
        "cohort": "fixed_camera",
        "source": "trackid3x3",
        "source_path": "fixed.mp4",
        "coverage_tags": [],
        "copy_mode": "full",
    })
    return {
        "selection_policy": {
            "target_clip_count": 30,
            "nba_clip_count": 24,
            "non_nba_clip_count": 6,
        },
        "known_seen_source_games": [],
        "clips": clips,
    }


class NBAHoldoutToolTests(unittest.TestCase):
    def test_annotation_sampling_matches_validator_interval(self):
        self.assertEqual(sampled_indices(31, 15), [0, 15, 30])
        self.assertEqual(sampled_indices(30, 15), [0, 15])

    def test_valid_selection_has_expected_composition(self):
        validate_selection(_selection())

    def test_selection_rejects_duplicate_nba_game(self):
        selection = _selection()
        selection["clips"][1]["game_id"] = selection["clips"][0]["game_id"]
        with self.assertRaisesRegex(ValueError, "source games are not unique"):
            validate_selection(selection)

    def test_selection_rejects_previously_seen_game(self):
        selection = _selection()
        selection["known_seen_source_games"] = [selection["clips"][0]["game_id"]]
        with self.assertRaisesRegex(ValueError, "already seen"):
            validate_selection(selection)

    def test_selection_rejects_invalid_window(self):
        selection = _selection()
        selection["clips"][24]["duration_seconds"] = 0
        with self.assertRaisesRegex(ValueError, "invalid duration_seconds"):
            validate_selection(selection)


if __name__ == "__main__":
    unittest.main()
