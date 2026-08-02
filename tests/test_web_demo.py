import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from backend.app.utils import probe_video


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
ASSET_ROOT = WEB_ROOT / "assets"


class WebDemoTests(unittest.TestCase):
    def test_preprocessed_demo_assets_share_a_timeline(self):
        video_path = ASSET_ROOT / "courtvision-demo-tactical-fixed.webm"
        analysis_path = ASSET_ROOT / "courtvision-demo-analysis.json"

        self.assertTrue(video_path.is_file())
        self.assertTrue(analysis_path.is_file())

        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        video = probe_video(video_path)
        video_duration = video["frame_count"] / video["fps"]

        self.assertEqual(analysis["schemaVersion"], 1)
        self.assertTrue(analysis["beta"])
        self.assertAlmostEqual(
            video_duration,
            analysis["source"]["durationSeconds"],
            delta=1 / video["fps"],
        )
        self.assertTrue(analysis["events"])
        self.assertTrue(
            all(event["type"] in {"pass", "interception"} for event in analysis["events"])
        )
        self.assertTrue(
            all(
                0 <= event["timeSeconds"] <= analysis["source"]["durationSeconds"]
                for event in analysis["events"]
            )
        )

        sync_samples = [
            (0, 1),
            (38, 1),
            (60, 9),
            (102, 1),
            (103, 1),
            (104, 1),
            (110, 1),
            (159, 7),
            (160, 1),
            (162, 1),
        ]
        mirror_ranges = analysis["court"]["mirrorXFrameRanges"]
        capture = cv2.VideoCapture(str(video_path))
        display_colors = {
            1: np.array([255, 80, 0], dtype=np.int16),
            2: np.array([0, 215, 255], dtype=np.int16),
        }
        for frame_index, player_id in sync_samples:
            tracked_player = next(
                player
                for player in analysis["frames"][frame_index]["players"]
                if player["id"] == player_id
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            read_ok, video_frame = capture.read()
            self.assertTrue(read_ok)

            mirror_x = any(
                first_frame <= frame_index <= last_frame
                for first_frame, last_frame in mirror_ranges
            )
            court_x = (
                analysis["court"]["width"] - tracked_player["x"]
                if mirror_x
                else tracked_player["x"]
            )
            video_x = 20 + round(court_x)
            video_y = 40 + round(tracked_player["y"])
            marker_patch = video_frame[
                video_y - 4 : video_y + 5,
                video_x - 4 : video_x + 5,
            ].astype(np.int16)
            minimum_color_error = np.linalg.norm(
                marker_patch - display_colors[tracked_player["teamId"]],
                axis=2,
            ).min()
            self.assertLess(minimum_color_error, 55, msg=f"frame {frame_index}")
        capture.release()

    def test_public_demo_is_labeled_as_preprocessed_experimental_analysis(self):
        landing = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
        demo = (WEB_ROOT / "demo.html").read_text(encoding="utf-8")
        client = (WEB_ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn("Sample CourtVision analysis", landing)
        self.assertIn("Preprocessed sample", landing)
        self.assertIn("OKC vs. Timberwolves game sample", landing)
        self.assertNotIn("Preprocessed experimental output", landing)
        self.assertNotIn("Synthetic interface demo", landing)
        self.assertIn(
            "demo.html?embedded=1&amp;v=video-2-tactical-fixed-4", landing
        )
        self.assertIn("app.js?v=video-2-tactical-fixed-4", demo)
        self.assertIn(
            'videoUrl: "assets/courtvision-demo-tactical-fixed.webm"', client
        )
        self.assertIn("${tacticalDockMarkup(analysis)}", client)
        self.assertIn("analysis.court?.mirrorXFrameRanges", client)
        self.assertNotIn('permanentDemo ? "" : tacticalDockMarkup(analysis)', client)
        self.assertIn('analysisUrl: "assets/courtvision-demo-analysis.json"', client)


if __name__ == "__main__":
    unittest.main()
