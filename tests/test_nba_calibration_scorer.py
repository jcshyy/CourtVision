import unittest

from scripts.score_nba_calibration import _canonical_truth, _score_events


class NBACalibrationScorerTests(unittest.TestCase):
    def test_only_verified_supported_events_are_scored(self):
        events = [
            {"video_id": "clip", "event_type": "pass", "review_status": "verified"},
            {"video_id": "clip", "event_type": "steal", "review_status": "verified"},
            {"video_id": "clip", "event_type": "shot", "review_status": "verified"},
            {"video_id": "clip", "event_type": "pass", "review_status": "draft"},
        ]

        result = _canonical_truth(events, {"clip"})

        self.assertEqual([event["event_type"] for event in result], ["pass", "interception"])

    def test_per_video_score_does_not_match_across_clips(self):
        truth = [
            {"video_id": "a", "event_type": "pass", "catch_frame": 10},
        ]
        predictions = [
            {"video_id": "b", "type": "pass", "frame_index": 10},
        ]

        result = _score_events(truth, predictions, tolerance=15)

        self.assertEqual(result["true_positive"], 0)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 1)


if __name__ == "__main__":
    unittest.main()
