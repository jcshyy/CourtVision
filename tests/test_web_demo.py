import json
from pathlib import Path
import unittest

from backend.app.utils import probe_video


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
ASSET_ROOT = WEB_ROOT / "assets"


class WebDemoTests(unittest.TestCase):
    def test_preprocessed_demo_assets_share_a_timeline(self):
        video_path = ASSET_ROOT / "courtvision-demo-updated.mp4"
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
        self.assertEqual(video["frame_count"], analysis["source"]["frameCount"])
        self.assertEqual(video["frame_count"], len(analysis["frames"]))
        self.assertTrue(analysis["events"])
        self.assertTrue(
            all(
                event["type"] in {"pass", "interception", "shot_attempt"}
                for event in analysis["events"]
            )
        )
        self.assertIn("shot_attempt", {event["type"] for event in analysis["events"]})
        self.assertTrue(all("outcome" not in event.get("evidence", {}) for event in analysis["events"]))
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
            "demo.html?embedded=1&amp;v=official-preview-2", landing
        )
        self.assertIn("app.js?v=official-preview-2", demo)
        self.assertIn(
            'videoUrl: "assets/courtvision-demo-updated.mp4"', client
        )
        self.assertIn("${tacticalDockMarkup(analysis)}", client)
        self.assertIn("${summaryDockMarkup(analysis)}", client)
        self.assertIn("Ball control estimate", client)
        self.assertIn('role="tablist" aria-label="Replay inspector view"', client)
        self.assertIn('inspectorTab: "court"', client)
        self.assertIn('id="new-analysis"', client)
        self.assertIn("Analyze another clip?", client)
        self.assertIn("No reliable event candidate", client)
        self.assertIn("No pass, interception, or shot attempt met the review threshold", client)
        self.assertIn("Shot-attempt candidate", client)
        self.assertIn("Ball observations", client)
        self.assertIn("Measured release path", client)
        self.assertIn("Rim proximity signal", client)
        self.assertNotIn("Shot outcome", client)
        self.assertIn("Recent analyses", client)
        self.assertIn("You can reopen the previous result under Recent analyses until it expires.", client)
        self.assertIn("targetFps: 30", client)
        self.assertIn("maxWidth: 1280", client)
        self.assertIn("analysis.court?.mirrorXFrameRanges", client)
        self.assertNotIn('permanentDemo ? "" : tacticalDockMarkup(analysis)', client)
        self.assertIn('analysisUrl: "assets/courtvision-demo-analysis.json"', client)

    def test_landing_exposes_live_public_analysis_entry(self):
        root = Path(__file__).resolve().parents[1]
        landing = (root / "web" / "index.html").read_text(encoding="utf-8")
        config = (root / "web" / "config.js").read_text(encoding="utf-8")
        client = (root / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="analyze"', landing)
        self.assertIn("Analyze video", landing)
        self.assertIn("The review desk and live analysis are open.", landing)
        self.assertIn("Public analysis is live.", landing)
        self.assertIn("authConnected: true", config)
        self.assertIn("publicPreview: false", config)
        self.assertIn("analysisAvailable: true", config)
        self.assertNotIn("waiting on GPU capacity", landing)
        self.assertIn("This video was not uploaded", client)
        self.assertIn("Analysis capacity pending", client)
        self.assertIn("targetFps: 30", config)
        self.assertIn("maxWidth: 1280", config)

    def test_upload_rundown_wraps_long_filenames(self):
        root = Path(__file__).resolve().parents[1]
        styles = (root / "web" / "styles.css").read_text(encoding="utf-8")
        client = (root / "web" / "app.js").read_text(encoding="utf-8")

        self.assertIn(".selected-file h3", styles)
        self.assertIn("overflow-wrap: anywhere", styles)
        self.assertIn('title="${escapeHtml(state.selectedFile.name)}"', client)


if __name__ == "__main__":
    unittest.main()
