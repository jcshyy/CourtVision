import copy
import json
import unittest

from backend.app import game_summary
from backend.app.trusted_evidence import build_trusted_evidence_v2 as build
from backend.app.trusted_evidence import validate_trusted_evidence_v2 as validate


def fixture():
    return {
        "source": {"durationSeconds": 2, "fps": 10, "frameCount": 20},
        "frames": [
            {
                "frameIndex": i,
                "courtCalibrated": True,
                "players": [{"id": 1, "teamId": 1, "distanceMeters": 0.2}],
            }
            for i in range(20)
        ],
        "diagnostics": {
            "possessionTimeline": {
                "semantic": {
                    "frames": [
                        {
                            "frame_index": i,
                            "holder_id": 1 if i < 10 else None,
                            "state": "controlled" if i < 10 else "unknown",
                        }
                        for i in range(20)
                    ]
                }
            }
        },
        "events": [
            {
                "type": "pass",
                "frameIndex": 8,
                "timeSeconds": 0.8,
                "fromTeamId": 1,
                "toTeamId": 1,
                "evidence": {
                    "from_player_id": 1,
                    "to_player_id": 1,
                    "release_frame": 7,
                    "catch_frame": 8,
                    "possession_evidence": {
                        "source_support_frames": 3,
                        "receiver_support_frames": 3,
                    },
                },
            }
        ],
    }


class TrustedEvidenceTests(unittest.TestCase):
    def test_manifest_exports_observations_without_court_projection(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from main import write_analysis_manifest
        with TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            write_analysis_manifest(path, fps=10, frame_count=2, court_width=300,
                court_height=161, tactical_player_positions=[{}, {}],
                player_assignment=[{7: 1}, {7: 1}], ball_acquisition=[7, 7],
                events=[], tactical_diagnostics={"event_only": ["court_projection_skipped"]},
                assignment_metadata={}, player_tracks=[{7: {}}, {7: {}}], discontinuity_frames=[1])
            result = build(json.loads(path.read_text()))
        self.assertEqual(result["tracks"][0]["observedFrames"], 2)
        self.assertEqual(result["tracks"][0]["continuousRuns"], 2)
        self.assertNotIn("distanceMeters", result["tracks"][0])

    def test_short_control_runs_remain_unknown(self):
        a = fixture()
        a["diagnostics"]["possessionTimeline"]["semantic"]["frames"] = [
            {"frame_index": i, "holder_id": 1, "state": "controlled"} for i in range(2)]
        result = build(a)
        self.assertEqual(result["possession"]["unknownShare"], 1)
        self.assertEqual(result["possession"]["segments"], [])

    def test_sparse_data_omits_metrics(self):
        a = fixture()
        a["frames"] = []
        a["events"] = []
        p = build(a)
        self.assertEqual(p["tracks"], [])
        self.assertIsNone(p["possession"]["unknownShare"])
        self.assertEqual(p["coverage"]["unknownPossessionFrames"], 20)

    def test_unknown_denominator(self):
        p = build(fixture())
        self.assertEqual(p["possession"]["eligibleFrames"], 20)
        self.assertEqual(p["possession"]["unknownShare"], 0.5)
        self.assertEqual(p["possession"]["team1Share"], 0.5)
        self.assertEqual(p["possession"]["segments"][0]["supportFrames"], 10)

    def test_discontinuities_break_windows_and_segments(self):
        a = fixture()
        a["measurements"] = {"discontinuityFrames": list(range(1, 20, 3))}
        p = build(a)
        self.assertGreater(p["tracks"][0]["continuousRuns"], 1)
        self.assertNotIn("sustainedPeakSpeedMps", p["tracks"][0])
        self.assertTrue(
            all(s["supportFrames"] <= 3 for s in p["possession"]["segments"])
        )

    def test_outlier_not_in_distance_or_speed(self):
        a = fixture()
        a["frames"][10]["players"][0]["distanceMeters"] = 100
        p = build(a)["tracks"][0]
        self.assertAlmostEqual(p["distanceMeters"], 3.6)
        self.assertAlmostEqual(p["averageSpeedMps"], 2)
        self.assertIn("speed_outlier_filtered", p["uncertaintyReasons"])

    def test_uncalibrated_no_metrics(self):
        a = fixture()
        for f in a["frames"]:
            f.pop("courtCalibrated")
        self.assertNotIn("distanceMeters", build(a)["tracks"][0])

    def test_malformed_reference_rejected(self):
        p = build(fixture())
        p["events"][0]["fromTrackId"] = "track-999"
        with self.assertRaises(ValueError):
            validate(p)
        p = build(fixture())
        p["sequences"][0]["evidenceIds"] = ["missing"]
        with self.assertRaises(ValueError):
            validate(p)

    def test_contradiction_omitted(self):
        a = fixture()
        a["events"][0]["toTeamId"] = 2
        self.assertEqual(build(a)["events"], [])
        a = fixture()
        a["events"][0]["evidence"]["release_frame"] = 9
        self.assertEqual(build(a)["events"], [])

    def test_conflicting_release_omits_both(self):
        a = fixture()
        e = copy.deepcopy(a["events"][0])
        e["type"] = "shot_attempt"
        a["events"].append(e)
        self.assertEqual(build(a)["events"], [])

    def test_deterministic_and_order_independent(self):
        a = fixture()
        before = copy.deepcopy(a)
        p = build(a)
        self.assertEqual(p, build(a))
        self.assertEqual(a, before)
        a["frames"].reverse()
        self.assertEqual(p, build(a))
        json.dumps(p, allow_nan=False)

    def test_bad_numbers_bounds_structure(self):
        for value in (float("nan"), float("inf"), -1, True):
            a = fixture()
            a["source"]["fps"] = value
            with self.assertRaises(ValueError):
                build(a)
        p = build(fixture())
        p["events"][0]["timeSeconds"] = 9
        with self.assertRaises(ValueError):
            validate(p)
        p = build(fixture())
        p["tracks"][0]["identity"] = "Name"
        with self.assertRaises(ValueError):
            validate(p)

    def test_no_verification_by_input_status(self):
        a = fixture()
        a["events"][0]["status"] = "verified"
        self.assertEqual(build(a)["events"][0]["status"], "candidate")

    def test_model_cannot_invent_claim_even_with_valid_reference(self):
        p = build(fixture())
        report = {
            "summary": p["summaryOptions"][0],
            "limitations": [],
            "tacticalInsights": [
                {
                    "claim": "Track 1 averages 99 meters per second.",
                    "evidenceIds": ["track-1"],
                    "caveat": None,
                }
            ],
        }
        with self.assertRaises(ValueError):
            game_summary._validate(report, p)
        report["tacticalInsights"] = [{**p["supportedClaims"][0], "caveat": None}]
        self.assertEqual(game_summary._validate(report, p), report)

    def test_legacy_manifest_no_inferred_calibration(self):
        from pathlib import Path

        p = build(
            json.loads(Path("web/assets/courtvision-demo-analysis.json").read_text())
        )
        self.assertEqual(p["coverage"]["calibratedFrameCount"], 0)


if __name__ == "__main__":
    unittest.main()
