import unittest
from unittest.mock import patch

from backend.app.analytics.ball_acquisition import BallAcquisitionDetector
from backend.app.analytics.ball_holder_state import BallHolderStateModel
from backend.app.analytics.pass_interception import (
    PassInterceptionDetector,
    events_from_arrays,
)
from backend.app.tracking.ball_tracker import BallTracker, _select_ball_detection


class BallAcquisitionTests(unittest.TestCase):
    def test_confirmation_duration_scales_with_fps(self):
        self.assertEqual(BallAcquisitionDetector(fps=30).min_frames, 11)
        self.assertEqual(BallAcquisitionDetector(fps=15).min_frames, 6)

    def test_nearest_high_containment_player_wins(self):
        detector = BallAcquisitionDetector()
        players = {
            1: {"bbox": [0, 0, 80, 80]},
            2: {"bbox": [45, 45, 65, 65]},
        }

        self.assertEqual(
            detector.find_best_candidate_for_possession(
                (50, 50),
                players,
                [48, 48, 52, 52],
            ),
            2,
        )

    def test_confirmed_possession_is_backfilled_to_run_start(self):
        detector = BallAcquisitionDetector()
        detector.min_frames = 3
        player_tracks = [{7: {"bbox": [0, 0, 100, 100]}}] * 4
        ball_tracks = [{1: {"bbox": [40, 40, 50, 50]}}] * 4

        with patch.object(
            detector,
            "find_best_candidate_for_possession",
            return_value=7,
        ):
            possession = detector.detect_ball_possession(
                player_tracks,
                ball_tracks,
            )

        self.assertEqual(possession, [-1, -1, 7, 7])

    def test_missing_ball_breaks_confirmation_run(self):
        detector = BallAcquisitionDetector()
        detector.min_frames = 3
        player_tracks = [{7: {"bbox": [0, 0, 100, 100]}}] * 5
        ball_tracks = [
            {1: {"bbox": [40, 40, 50, 50]}},
            {1: {"bbox": [40, 40, 50, 50]}},
            {},
            {1: {"bbox": [40, 40, 50, 50]}},
            {1: {"bbox": [40, 40, 50, 50]}},
        ]

        with patch.object(
            detector,
            "find_best_candidate_for_possession",
            return_value=7,
        ):
            possession = detector.detect_ball_possession(
                player_tracks,
                ball_tracks,
            )

        self.assertEqual(possession, [-1] * 5)

    def test_ball_motion_filter_compensates_for_camera_pan(self):
        ball_tracks = [
            {1: {"bbox": [100, 100, 104, 104]}},
            {1: {"bbox": [140, 100, 144, 104]}},
        ]
        player_tracks = [
            {
                1: {"bbox": [0, 0, 10, 20]},
                2: {"bbox": [50, 0, 60, 20]},
                3: {"bbox": [100, 0, 110, 20]},
            },
            {
                1: {"bbox": [40, 0, 50, 20]},
                2: {"bbox": [90, 0, 100, 20]},
                3: {"bbox": [140, 0, 150, 20]},
            },
        ]

        filtered = BallTracker.remove_wrong_detections(
            None,
            ball_tracks,
            player_tracks=player_tracks,
        )

        self.assertIn(1, filtered[1])

    def test_ball_selection_can_prefer_player_supported_detection(self):
        detections = [
            {"bbox": [300, 10, 305, 15], "confidence": 0.95},
            {"bbox": [105, 105, 110, 110], "confidence": 0.65},
        ]
        players = {7: {"bbox": [90, 80, 140, 180]}}

        selected = _select_ball_detection(detections, players, None)

        self.assertEqual(selected["bbox"], [105, 105, 110, 110])

    def test_ball_selection_uses_continuity_during_airborne_motion(self):
        detections = [
            {"bbox": [300, 10, 305, 15], "confidence": 0.9},
            {"bbox": [130, 100, 135, 105], "confidence": 0.68},
            {"bbox": [210, 100, 215, 105], "confidence": 0.7},
        ]
        players = {
            1: {"bbox": [120, 80, 145, 180]},
            2: {"bbox": [200, 80, 225, 180]},
        }

        selected = _select_ball_detection(
            detections,
            players,
            previous_bbox=[100, 100, 105, 105],
        )

        self.assertEqual(selected["bbox"], [130, 100, 135, 105])

class BallHolderStateTests(unittest.TestCase):
    def setUp(self):
        self.model = BallHolderStateModel(
            confirmation_frames=2,
            max_missing_frames=2,
        )
        self.players = {
            10: {"bbox": [0, 0, 30, 60]},
            20: {"bbox": [70, 0, 100, 60]},
        }

    @staticmethod
    def ball(x, confidence=0.9, interpolated=False):
        return {1: {
            "bbox": [x - 2, 28, x + 2, 32],
            "confidence": None if interpolated else confidence,
            "interpolated": interpolated,
        }}

    def acquisitions(self, states):
        return [state["holder_id"] if state["holder_id"] is not None else -1 for state in states]

    def test_one_frame_candidate_flicker_does_not_change_holder(self):
        states = self.model.process(
            [self.players] * 5,
            [self.ball(15), self.ball(15), self.ball(85), self.ball(15), self.ball(15)],
        )
        self.assertEqual(self.acquisitions(states), [-1, 10, 10, 10, 10])
        self.assertEqual(states[2]["reason"], "switch_pending")
        self.assertEqual(states[2]["candidate_id"], 20)

    def test_brief_missing_detection_preserves_holder(self):
        states = self.model.process(
            [self.players] * 5,
            [self.ball(15), self.ball(15), {}, {}, self.ball(15)],
        )
        self.assertEqual(self.acquisitions(states), [-1, 10, 10, 10, 10])
        self.assertEqual(states[2]["reason"], "brief_ball_gap")
        self.assertGreater(states[2]["confidence"], 0)

    def test_sustained_new_candidate_switches_holder(self):
        states = self.model.process(
            [self.players] * 4,
            [self.ball(15), self.ball(15), self.ball(85), self.ball(85)],
        )
        self.assertEqual(self.acquisitions(states), [-1, 10, 10, 20])
        self.assertEqual(states[3]["reason"], "holder_switch_confirmed")

    def test_interpolated_ball_cannot_confirm_receiver(self):
        model = BallHolderStateModel(confirmation_frames=3, max_missing_frames=2)
        states = model.process(
            [self.players] * 6,
            [
                self.ball(15),
                self.ball(15),
                self.ball(15),
                self.ball(85),
                self.ball(85),
                self.ball(85, interpolated=True),
            ],
        )

        self.assertEqual(self.acquisitions(states), [-1, -1, 10, 10, -1, -1])
        self.assertEqual(states[5]["state"], "loose")
        self.assertEqual(states[5]["reason"], "interpolated_ball_not_confirmable")

    def test_real_detection_after_flight_confirms_catch(self):
        model = BallHolderStateModel(confirmation_frames=3, max_missing_frames=2)
        states = model.process(
            [self.players] * 7,
            [
                self.ball(15),
                self.ball(15),
                self.ball(15),
                self.ball(85),
                self.ball(85),
                self.ball(85, interpolated=True),
                self.ball(85),
            ],
        )

        self.assertEqual(states[4]["state"], "loose")
        self.assertEqual(states[5]["state"], "loose")
        self.assertEqual(states[6]["holder_id"], 20)
        self.assertEqual(states[6]["reason"], "holder_switch_confirmed")

    def test_interpolated_gap_can_preserve_but_not_create_holder(self):
        states = self.model.process(
            [self.players] * 4,
            [
                self.ball(15),
                self.ball(15),
                self.ball(15, interpolated=True),
                self.ball(15),
            ],
        )

        self.assertEqual(self.acquisitions(states), [-1, 10, 10, 10])
        self.assertEqual(states[2]["reason"], "brief_interpolated_gap")

    def test_ambiguous_ball_between_players_is_loose(self):
        close_players = {
            10: {"bbox": [20, 0, 50, 60]},
            20: {"bbox": [50, 0, 80, 60]},
        }
        states = self.model.process([close_players], [self.ball(50)])
        self.assertIsNone(states[0]["holder_id"])
        self.assertEqual(states[0]["state"], "loose")
        self.assertEqual(states[0]["reason"], "ambiguous_candidates")

    def test_confirmed_team_change_can_be_interception(self):
        states = self.model.process(
            [self.players] * 4,
            [self.ball(15), self.ball(15), self.ball(85), self.ball(85)],
        )
        acquisitions = self.acquisitions(states)
        assignments = [{10: 1, 20: 2}] * 4
        result = PassInterceptionDetector().detect_interceptions(acquisitions, assignments)
        self.assertEqual(result[3], 2)

    def test_same_team_confirmed_change_can_be_pass(self):
        states = self.model.process(
            [self.players] * 4,
            [self.ball(15), self.ball(15), self.ball(85), self.ball(85)],
        )
        acquisitions = self.acquisitions(states)
        assignments = [{10: 1, 20: 1}] * 4
        result = PassInterceptionDetector().detect_passes(acquisitions, assignments)
        self.assertEqual(result[3], 1)


class PassInterceptionTests(unittest.TestCase):
    def test_long_unknown_gap_does_not_emit_event(self):
        detector = PassInterceptionDetector(max_holder_gap_frames=2)
        acquisitions = [10, -1, -1, -1, 20]
        assignments = [{10: 1}, {}, {}, {}, {20: 2}]

        self.assertEqual(detector.detect_passes(acquisitions, assignments), [-1] * 5)
        self.assertEqual(
            detector.detect_interceptions(acquisitions, assignments),
            [-1] * 5,
        )

    def test_event_records_release_catch_and_gap(self):
        acquisitions = [10, -1, 20]
        events = events_from_arrays(
            [-1, -1, 1],
            [-1, -1, -1],
            acquisitions,
        )

        self.assertEqual(events[0]["release_frame"], 0)
        self.assertEqual(events[0]["catch_frame"], 2)
        self.assertEqual(events[0]["gap_frames"], 1)

    def test_recent_known_holder_team_fills_unknown_release_frame(self):
        detector = PassInterceptionDetector(team_lookup_frames=4)
        acquisitions = [10, 10, -1, 20]
        assignments = [
            {10: 1},
            {10: -1},
            {},
            {20: 1},
        ]

        self.assertEqual(
            detector.detect_passes(acquisitions, assignments),
            [-1, -1, -1, 1],
        )

    def test_interception_requires_observed_loose_ball_when_states_are_given(self):
        detector = PassInterceptionDetector()
        acquisitions = [10, -1, 20]
        assignments = [{10: 1}, {}, {20: 2}]
        interpolated_only = [
            {"state": "confirmed", "ball_confidence": 0.9},
            {"state": "loose", "ball_confidence": None},
            {"state": "confirmed", "ball_confidence": 0.8},
        ]

        self.assertEqual(
            detector.detect_interceptions(
                acquisitions,
                assignments,
                holder_states=interpolated_only,
            ),
            [-1, -1, -1],
        )

    def test_observed_loose_ball_allows_confirmed_interception(self):
        detector = PassInterceptionDetector()
        acquisitions = [10, -1, 20]
        assignments = [{10: 1}, {}, {20: 2}]
        observed_loose = [
            {"state": "confirmed", "ball_confidence": 0.9},
            {"state": "loose", "ball_confidence": 0.7},
            {"state": "confirmed", "ball_confidence": 0.8},
        ]

        self.assertEqual(
            detector.detect_interceptions(
                acquisitions,
                assignments,
                holder_states=observed_loose,
            ),
            [-1, -1, 2],
        )

    def test_brief_intermediate_holder_collapses_to_final_interception(self):
        detector = PassInterceptionDetector(transient_control_frames=8)
        acquisitions = [10, -1, 20, 20, -1, 30]
        assignments = [
            {10: 2},
            {},
            {20: 1},
            {20: 1},
            {},
            {30: 1},
        ]
        states = [
            {"state": "confirmed", "ball_confidence": 0.9},
            {"state": "loose", "ball_confidence": 0.7},
            {"state": "confirmed", "ball_confidence": 0.8},
            {"state": "confirmed", "ball_confidence": 0.8},
            {"state": "loose", "ball_confidence": 0.7},
            {"state": "confirmed", "ball_confidence": 0.9},
        ]

        cleaned = detector.clean_transient_control_chains(
            acquisitions,
            assignments,
            holder_states=states,
        )

        self.assertEqual(cleaned, [10, -1, -1, -1, -1, 30])
        self.assertEqual(
            detector.detect_passes(cleaned, assignments),
            [-1] * 6,
        )
        self.assertEqual(
            detector.detect_interceptions(
                cleaned,
                assignments,
                holder_states=states,
            ),
            [-1, -1, -1, -1, -1, 1],
        )


if __name__ == "__main__":
    unittest.main()
