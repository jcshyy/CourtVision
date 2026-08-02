import json
from pathlib import Path
import unittest

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
            "demo.html?embedded=1&amp;v=video-2-tactical-fixed-3", landing
        )
        self.assertIn("app.js?v=video-2-tactical-fixed-3", demo)
        self.assertIn(
            'videoUrl: "assets/courtvision-demo-tactical-fixed.webm"', client
        )
        self.assertIn("${tacticalDockMarkup(analysis)}", client)
        self.assertNotIn('permanentDemo ? "" : tacticalDockMarkup(analysis)', client)
        self.assertIn('analysisUrl: "assets/courtvision-demo-analysis.json"', client)


if __name__ == "__main__":
    unittest.main()
