import copy
import unittest
from unittest.mock import patch

from backend.app.analytics.event_lifecycle import finalize_ball_events, _endline, _is_throw_in
from backend.app.analytics.pass_interception import merge_corroborated_pass_events
from backend.app.analytics.possession_timeline import PossessionTimelineBuilder
from backend.app.analytics.shot_rebound import (
    ShotReboundDetector, ShotReboundTimeline, _same_shot_flight,
    _terminal_release_candidates, reconcile_shot_events,
    _shooting_hand_support,
)


def transfer(start=2, end=12, source=10, receiver=20, kind="pass"):
    return {"type": kind, "release_frame": start, "catch_frame": end,
            "frame_index": end, "from_player_id": source, "to_player_id": receiver,
            "from_team_id": 1, "to_team_id": 1, "gap_frames": end - start - 1}


def shot_timeline():
    return ShotReboundTimeline(
        [{"frame_index": i, "state": "possession"} for i in range(30)],
        [{"sequence_id": 1, "release_frame": 2, "holder_release_frame": 2,
          "rim_frame": 10, "pending_end_frame": 16, "shooter_id": 10,
          "post_shot_control_frame": 16, "evidence": {"rim_approach_frame": 10}}],
        [{"type": "shot_attempt", "frame_index": 6, "release_frame": 2, "sequence_id": 1}], [])


class LifecycleTests(unittest.TestCase):
    def test_empty_model_keypoints_abstain_from_endline_geometry(self):
        class EmptyTensor:
            def cpu(self):
                return self

            def tolist(self):
                return []

        class EmptyKeypoints:
            xy = EmptyTensor()
            conf = EmptyTensor()

        self.assertIsNone(_endline(EmptyKeypoints(), "left"))

    def test_rim_seed_recovers_shot_without_holder_transition(self):
        balls = [{1: {"bbox": [x - 3, y - 3, x + 3, y + 3],
                      "rim_regions": [{"bbox": [100, 40, 120, 60]}]}}
                 for x, y in [(10, 160), (30, 130), (50, 90), (70, 50), (100, 45), (115, 55), (120, 90)]]
        players = [{10: {"bbox": [0, 70, 40, 170]}} for _ in balls]
        timeline = PossessionTimelineBuilder().build([-1] * len(balls))
        result = ShotReboundDetector().detect(timeline, [{}] * len(balls), balls, players)
        self.assertEqual(len(result.events), 1)
        self.assertIsNone(result.events[0]["from_player_id"])

    def test_high_hand_evidence_requires_two_observed_frames(self):
        points = [[0, 0] for _ in range(17)]
        points[9] = [20, 80]
        player = {"bbox": [0, 70, 40, 170], "pose": {"keypoints_xy": points, "keypoint_confidences": [0.9] * 17}}
        observations = [{"frame": i, "center": [20, 80]} for i in range(2)]
        self.assertIsNone(_shooting_hand_support(observations[:1], [{10: player}] * 2))
        self.assertEqual(_shooting_hand_support(observations, [{10: player}] * 2)["player_id"], 10)

    def test_id_switch_does_not_turn_descending_shot_into_pass(self):
        result = reconcile_shot_events([transfer(source=99)], shot_timeline())
        self.assertEqual([e["type"] for e in result], ["shot_attempt"])

    def test_defensive_rebound_is_not_an_interception(self):
        result = reconcile_shot_events([transfer(kind="interception")], shot_timeline())
        self.assertEqual([e["type"] for e in result], ["shot_attempt"])

    def test_rebound_outlet_preserves_attempt_and_outlet(self):
        result = reconcile_shot_events([transfer(16, 22, 20, 30)], shot_timeline())
        self.assertEqual([e["type"] for e in result], ["shot_attempt", "pass"])

    def test_post_rim_takeover_before_stable_control_is_suppressed(self):
        timeline = shot_timeline()
        result = reconcile_shot_events([transfer(11, 16, 99, 20)], timeline)
        self.assertEqual(len(result), 1)
        self.assertEqual(timeline.arbitration[0]["reason"], "shot_flight_or_rebound_acquisition")

    def test_stable_lob_catch_before_rim_is_pass(self):
        result = reconcile_shot_events([transfer(2, 7)], shot_timeline())
        self.assertEqual([e["type"] for e in result], ["pass"])

    def test_midflight_holder_churn_cannot_veto_shot(self):
        result = reconcile_shot_events([transfer(6, 9)], shot_timeline())
        self.assertEqual([e["type"] for e in result], ["shot_attempt"])

    def test_stable_catch_of_high_pass_refutes_weak_rim_hypothesis(self):
        timeline = shot_timeline()
        timeline.sequences[0]["evidence"]["rim_distance_player_heights"] = 1.4
        event = transfer(6, 10, 99, 20)
        event["possession_evidence"] = {"receiver_support_frames": 6}
        result = reconcile_shot_events([event], timeline)
        self.assertEqual([e["type"] for e in result], ["pass"])

    def test_stable_rebound_cannot_refute_close_rim_flight(self):
        timeline = shot_timeline()
        timeline.sequences[0]["evidence"]["rim_distance_player_heights"] = 0.1
        event = transfer(6, 11, 99, 20)
        event["possession_evidence"] = {"receiver_support_frames": 6}
        result = reconcile_shot_events([event], timeline)
        self.assertEqual([e["type"] for e in result], ["shot_attempt"])

    def test_separate_pass_into_shooter_is_retained(self):
        result = reconcile_shot_events([transfer(0, 2, 20, 10)], shot_timeline())
        self.assertEqual([e["type"] for e in result], ["pass", "shot_attempt"])

    def test_overlap_dedupe_ignores_shooter_identity(self):
        self.assertTrue(_same_shot_flight(4, 12, shot_timeline().sequences[0]))

    def test_partial_overlap_rim_seeds_are_one_flight(self):
        self.assertTrue(_same_shot_flight(9, 20, shot_timeline().sequences[0]))

    def test_two_frame_rim_window_seam_is_one_flight(self):
        self.assertTrue(_same_shot_flight(12, 24, shot_timeline().sequences[0]))

    def test_new_control_protects_quick_putback_from_seam_merge(self):
        sequence = shot_timeline().sequences[0]
        sequence["post_shot_control_frame"] = 11
        self.assertFalse(_same_shot_flight(12, 24, sequence))

    def test_putback_after_new_control_is_distinct(self):
        self.assertFalse(_same_shot_flight(17, 24, shot_timeline().sequences[0]))

    def test_same_pair_overlapping_flights_merge_despite_delayed_catch(self):
        players = [{10: {"bbox": [0, 0, 50, 100]}, 20: {"bbox": [300, 0, 350, 100]}}
                   for _ in range(80)]
        result = merge_corroborated_pass_events([transfer(2, 40)], [transfer(5, 55)], players)
        self.assertEqual(len(result), 1)

    def test_rapid_distinct_passes_are_not_merged(self):
        players = [{10: {"bbox": [0, 0, 50, 100]}, 20: {"bbox": [300, 0, 350, 100]},
                    30: {"bbox": [600, 0, 650, 100]}} for _ in range(30)]
        result = merge_corroborated_pass_events([transfer(2, 12)], [transfer(12, 18, 20, 30)], players)
        self.assertEqual(len(result), 2)

    def test_repeated_pass_same_pair_with_separate_flights_is_not_merged(self):
        result = merge_corroborated_pass_events([transfer(2, 8), transfer(9, 14)], [], None)
        self.assertEqual(len(result), 2)

    def test_cross_cut_pass_is_excluded(self):
        result = finalize_ball_events([transfer()], [], ShotReboundTimeline([], [], [], []),
                                      None, discontinuity_frames={8})
        self.assertEqual(result, [])

    def test_terminal_release_is_bounded_per_scene(self):
        timeline = PossessionTimelineBuilder().build(
            [10, 10, -1, -1, -1, 20, 20, -1, -1, -1], discontinuity_frames={5})
        candidates = _terminal_release_candidates(timeline, 10)
        self.assertEqual([c["transition_frame"] for c in candidates], [4, 9])

    @staticmethod
    def inbound_fixture():
        points = [[0, 0] for _ in range(18)]
        points[0], points[5], points[8] = [100, 100], [100, 500], [300, 250]
        keys = [{"points": copy.deepcopy(points), "confidence": [0.95] * 18} for _ in range(12)]
        players = [{10: {"bbox": [40, 180, 80, 280]},
                    20: {"bbox": [200, 180, 240, 280]}} for _ in range(12)]
        balls = [{1: {"bbox": [55, 205, 65, 215]}} for _ in range(12)]
        return players, balls, keys

    def test_inbound_is_excluded_from_pass_totals_without_make_claim(self):
        players, balls, keys = self.inbound_fixture()
        timeline = ShotReboundTimeline([], [], [], [])
        result = finalize_ball_events([transfer(4, 10)], [], timeline, players,
                                      ball_tracks=balls, court_keypoints=keys)
        self.assertEqual(result, [])
        self.assertEqual(timeline.arbitration[0]["reason"], "throw_in")

    def test_unknown_court_does_not_infer_inbound_from_possession(self):
        players, balls, keys = self.inbound_fixture()
        self.assertFalse(_is_throw_in(transfer(4, 10), players, balls, None, 3))
        keys[3]["confidence"] = [0.1] * 18
        self.assertFalse(_is_throw_in(transfer(4, 10), players, balls, keys, 3))

    def test_normal_inbounds_pass_is_not_throw_in(self):
        players, balls, keys = self.inbound_fixture()
        for frame in players:
            frame[10]["bbox"] = [110, 180, 150, 280]
        self.assertFalse(_is_throw_in(transfer(4, 10), players, balls, keys, 3))

    def test_unobserved_ball_does_not_infer_throw_in(self):
        players, balls, keys = self.inbound_fixture()
        self.assertFalse(_is_throw_in(transfer(4, 10), players, [{}] * 12, keys, 3))

    def test_below_rim_pass_trajectory_is_not_shot(self):
        balls = [{1: {"bbox": [x - 3, y - 3, x + 3, y + 3],
                      "rim_regions": [{"bbox": [100, 40, 120, 60]}]}}
                 for x, y in [(10, 160), (30, 150), (50, 140), (70, 130), (90, 125)]]
        players = [{10: {"bbox": [0, 70, 40, 170]}} for _ in balls]
        evidence = ShotReboundDetector()._shot_evidence(balls, players, 10, 0, 4)
        self.assertFalse(evidence["confirmed"])
        self.assertEqual(evidence["reason"], "below_rim_transfer_without_shooting_release")

    def test_pass_directly_under_basket_still_requires_shooting_height(self):
        balls = [{1: {"bbox": [x - 3, y - 3, x + 3, y + 3],
                      "rim_regions": [{"bbox": [100, 40, 120, 60]}]}}
                 for x, y in [(10, 160), (40, 135), (75, 115), (100, 100), (110, 95)]]
        players = [{10: {"bbox": [0, 70, 40, 170]}} for _ in balls]
        evidence = ShotReboundDetector()._shot_evidence(balls, players, 10, 0, 4)
        self.assertFalse(evidence["confirmed"])
        self.assertEqual(evidence["reason"], "below_rim_transfer_without_shooting_release")


if __name__ == "__main__":
    unittest.main()
