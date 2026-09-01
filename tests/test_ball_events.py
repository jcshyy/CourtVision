import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

from backend.app.analytics.ball_acquisition import BallAcquisitionDetector
from backend.app.analytics.ball_holder_state import BallHolderStateModel
from backend.app.analytics.pass_interception import (
    PassInterceptionDetector,
    build_event_team_hints,
    events_from_arrays,
    merge_corroborated_pass_events,
)
from backend.app.analytics.possession_timeline import PossessionTimelineBuilder
from backend.app.analytics.shot_rebound import (
    ShotReboundDetector,
    ShotReboundTimeline,
    _find_stable_control,
    _rim_candidate_windows,
    reconcile_shot_events,
)
from backend.app.tracking.ball_tracker import (
    BALL_DETECTION_BATCH_SIZE,
    BALL_DETECTION_IMAGE_SIZE,
    BALL_TRACKING_CACHE_VERSION,
    BallTracker,
    _adaptive_crop_requests,
    _merge_ball_candidates,
    _named_detections,
    _prune_sustained_body_locked_candidates,
    _reject_uncertain_observations,
    _select_ball_detection,
    _select_ball_track,
)


class BallDetectorConfigurationTests(unittest.TestCase):
    @patch("backend.app.tracking.ball_tracker.sv.Detections.from_ultralytics")
    def test_ebard_basketball_label_is_routed_as_ball(self, from_ultralytics):
        result = MagicMock()
        result.names = {0: "basketball", 1: "hoop"}
        from_ultralytics.return_value = [
            (np.asarray([10, 20, 30, 40]), None, 0.8, 0, None, None)
        ]

        detections = _named_detections(result, "Ball")

        self.assertEqual(detections[0]["bbox"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(detections[0]["confidence"], 0.8)

    @patch("backend.app.tracking.ball_tracker._named_detections")
    def test_shared_semantic_provider_replaces_full_frame_yolo_pass(
        self,
        named_detections,
    ):
        semantic_model = MagicMock()
        provider = MagicMock(return_value=[object()])
        named_detections.side_effect = lambda _result, name: (
            [{"bbox": [10, 10, 20, 20], "confidence": 0.8}]
            if name == "Ball"
            else []
        )
        tracker = BallTracker(
            detector_backend="yolo",
            semantic_model=semantic_model,
            semantic_detector_backend="ebard",
        )

        tracks = tracker.get_object_tracks(
            [object()],
            player_tracks=[{}],
            detections_provider=provider,
        )

        provider.assert_called_once_with()
        semantic_model.predict.assert_not_called()
        self.assertEqual(tracks[0][1]["bbox"], [10, 10, 20, 20])
        self.assertEqual(
            tracks[0][1]["detection_source"],
            "ebard_full_frame",
        )
        self.assertEqual(tracker.cache_version, "v7_ebard_shared_scene")

    @patch("backend.app.tracking.ball_tracker.YOLO")
    def test_ball_detection_uses_default_resolution_and_batch_size(self, yolo):
        model = MagicMock()
        model.predict.side_effect = [["a", "b", "c", "d"], ["e"]]
        yolo.return_value = model
        frames = [object() for _ in range(BALL_DETECTION_BATCH_SIZE + 1)]

        detections = BallTracker().detect_frames(frames)

        self.assertEqual(detections, ["a", "b", "c", "d", "e"])
        self.assertEqual(BALL_DETECTION_IMAGE_SIZE, 640)
        self.assertEqual(BALL_TRACKING_CACHE_VERSION, "v6_adaptive_roi_full_pass")
        self.assertEqual(
            model.predict.call_args_list,
            [
                call(
                    frames[:BALL_DETECTION_BATCH_SIZE],
                    conf=0.25,
                    imgsz=640,
                    verbose=False,
                ),
                call(
                    frames[BALL_DETECTION_BATCH_SIZE:],
                    conf=0.25,
                    imgsz=640,
                    verbose=False,
                ),
            ],
        )

    def test_adaptive_crops_cover_prediction_hands_and_rim_on_uncertain_frame(self):
        frames = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(2)]
        ball_tracks = [
            {1: {"bbox": [95, 95, 105, 105], "confidence": 0.8}},
            {1: {
                "bbox": [105, 95, 115, 105],
                "confidence": 0.3,
                "rim_regions": [{"bbox": [500, 100, 530, 120], "confidence": 0.9}],
            }},
        ]
        points = [[1, 1] for _ in range(17)]
        points[9], points[10] = [300, 400], [380, 400]
        players = [{}, {7: {
            "bbox": [260, 250, 420, 580],
            "pose": {
                "keypoints_xy": points,
                "keypoint_confidences": [0.9] * 17,
            },
        }}]

        requests = _adaptive_crop_requests(frames, ball_tracks, players)

        self.assertEqual({request["frame_index"] for request in requests}, {1})
        self.assertEqual(
            {request["source"] for request in requests},
            {"adaptive_predicted", "adaptive_player_hand", "adaptive_rim"},
        )
        self.assertLessEqual(len(requests), 4)

    def test_candidate_merge_keeps_best_duplicate_and_distinct_roi_candidate(self):
        merged = _merge_ball_candidates(
            [{
                "bbox": [100, 100, 110, 110],
                "confidence": 0.4,
                "detection_source": "full_frame",
            }],
            [
                {
                    "bbox": [99, 99, 111, 111],
                    "confidence": 0.8,
                    "detection_source": "adaptive_predicted",
                },
                {
                    "bbox": [300, 300, 310, 310],
                    "confidence": 0.6,
                    "detection_source": "adaptive_rim",
                },
            ],
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["detection_source"], "adaptive_predicted")
        self.assertEqual(merged[1]["detection_source"], "adaptive_rim")

    @patch.object(BallTracker, "interpolate_positions")
    @patch.object(BallTracker, "remove_wrong_detections")
    def test_semantic_track_excludes_wasb_candidates(
        self,
        remove_wrong_detections,
        interpolate_positions,
    ):
        remove_wrong_detections.side_effect = lambda _, tracks, **kwargs: tracks
        interpolate_positions.side_effect = lambda _, tracks, **kwargs: tracks
        raw = [{1: {
            "raw_candidates": [
                {
                    "bbox": [10, 10, 20, 20],
                    "confidence": 0.9,
                    "detection_source": "wasb_temporal",
                },
                {
                    "bbox": [30, 30, 40, 40],
                    "confidence": 0.7,
                    "detection_source": "full_frame",
                },
            ],
        }}]

        semantic = BallTracker.build_semantic_tracks(raw, [{}])

        candidates = semantic[0][1]["raw_candidates"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["detection_source"], "full_frame")

    @patch.object(BallTracker, "interpolate_positions")
    @patch.object(BallTracker, "remove_wrong_detections")
    def test_close_wasb_rescue_refines_but_cannot_confirm_semantic_track(
        self,
        remove_wrong_detections,
        interpolate_positions,
    ):
        semantic = [{1: {
            "bbox": [100, 100, 110, 110],
            "confidence": None,
            "interpolated": True,
            "position_source": "interpolated",
        }}]
        remove_wrong_detections.return_value = [{}]
        interpolate_positions.return_value = semantic
        fused = [{1: {
            "bbox": [104, 102, 114, 112],
            "confidence": 0.9,
            "interpolated": False,
            "detection_source": "wasb_temporal",
        }}]

        result = BallTracker.build_semantic_tracks(
            [{1: {"semantic_raw_candidates": []}}],
            [{}],
            fused_tracks=fused,
        )

        info = result[0][1]
        self.assertEqual(info["bbox"], fused[0][1]["bbox"])
        self.assertTrue(info["interpolated"])
        self.assertFalse(info["semantic_confirmable"])
        self.assertEqual(info["position_source"], "wasb_guarded_rescue")

    @patch.object(BallTracker, "interpolate_positions")
    @patch.object(BallTracker, "remove_wrong_detections")
    def test_consecutive_hand_supported_wasb_rescues_become_confirmable(
        self,
        remove_wrong_detections,
        interpolate_positions,
    ):
        semantic = [
            {1: {
                "bbox": [100 + frame, 100, 110 + frame, 110],
                "confidence": None,
                "interpolated": True,
                "position_source": "interpolated",
            }}
            for frame in range(3)
        ]
        remove_wrong_detections.return_value = [{} for _ in semantic]
        interpolate_positions.return_value = semantic
        fused = [
            {1: {
                "bbox": [101 + frame, 100, 111 + frame, 110],
                "confidence": 0.9,
                "interpolated": False,
                "detection_source": "wasb_temporal",
            }}
            for frame in range(3)
        ]
        points = [[1, 1] for _ in range(17)]
        points[7], points[9] = [104, 105], [106, 105]
        players = [{7: {
            "bbox": [80, 50, 130, 150],
            "pose": {
                "keypoints_xy": points,
                "keypoint_confidences": [0.9] * 17,
            },
        }} for _ in semantic]

        result = BallTracker.build_semantic_tracks(
            [{1: {"semantic_raw_candidates": []}} for _ in semantic],
            players,
            fused_tracks=fused,
        )

        self.assertTrue(all(
            frame[1]["semantic_confirmable"] for frame in result
        ))
        self.assertTrue(all(
            not frame[1]["interpolated"] for frame in result
        ))
        self.assertTrue(all(
            frame[1]["position_source"] == "wasb_hand_confirmed"
            for frame in result
        ))

    @patch.object(BallTracker, "interpolate_positions")
    @patch.object(BallTracker, "remove_wrong_detections")
    def test_airborne_wasb_sequence_cannot_confirm_possession(
        self,
        remove_wrong_detections,
        interpolate_positions,
    ):
        semantic = [
            {1: {
                "bbox": [100 + 10 * frame, 100, 110 + 10 * frame, 110],
                "confidence": None,
                "interpolated": True,
                "position_source": "interpolated",
            }}
            for frame in range(3)
        ]
        remove_wrong_detections.return_value = [{} for _ in semantic]
        interpolate_positions.return_value = semantic
        fused = [
            {1: {
                "bbox": [100 + 10 * frame, 100, 110 + 10 * frame, 110],
                "confidence": 0.9,
                "interpolated": False,
                "detection_source": "wasb_temporal",
            }}
            for frame in range(3)
        ]
        points = [[1, 1] for _ in range(17)]
        points[7], points[9] = [10, 10], [15, 10]
        players = [{7: {
            "bbox": [0, 0, 50, 100],
            "pose": {
                "keypoints_xy": points,
                "keypoint_confidences": [0.9] * 17,
            },
        }} for _ in semantic]

        result = BallTracker.build_semantic_tracks(
            [{1: {"semantic_raw_candidates": []}} for _ in semantic],
            players,
            fused_tracks=fused,
        )

        self.assertTrue(all(frame[1]["interpolated"] for frame in result))
        self.assertTrue(all(
            not frame[1]["semantic_confirmable"] for frame in result
        ))

    @patch("backend.app.tracking.ball_tracker._adaptive_crop_requests")
    @patch("backend.app.tracking.ball_tracker._named_detections")
    @patch("backend.app.tracking.ball_tracker.YOLO")
    def test_adaptive_pass_maps_crop_detection_back_to_frame(
        self,
        yolo,
        named_detections,
        crop_requests,
    ):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        crop_requests.return_value = [{
            "frame_index": 0,
            "bbox": [100, 200, 300, 400],
            "source": "adaptive_player_hand",
        }]
        named_detections.return_value = [{
            "bbox": [10, 20, 20, 30],
            "confidence": 0.8,
        }]
        yolo.return_value.predict.return_value = [object()]
        tracker = BallTracker()

        enhanced = tracker.enhance_tracks_with_adaptive_crops(
            [frame],
            [{1: {"raw_candidates": [], "rim_regions": []}}],
            [{}],
        )

        candidates = enhanced[0][1]["raw_candidates"]
        self.assertEqual(candidates[0]["bbox"], [110, 220, 120, 230])
        self.assertEqual(candidates[0]["detection_source"], "adaptive_player_hand")
        self.assertEqual(enhanced[0][1]["adaptive_crop_count"], 1)
        predict_call = yolo.return_value.predict.call_args
        self.assertEqual(predict_call.kwargs["conf"], 0.50)
        self.assertEqual(predict_call.kwargs["imgsz"], 640)

    @patch("backend.app.tracking.ball_tracker._adaptive_crop_requests")
    @patch("backend.app.tracking.ball_tracker._named_detections")
    @patch("backend.app.tracking.ball_tracker.YOLO")
    def test_adaptive_pass_rejects_detection_far_from_roi_anchor(
        self,
        yolo,
        named_detections,
        crop_requests,
    ):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        crop_requests.return_value = [{
            "frame_index": 0,
            "bbox": [100, 200, 300, 400],
            "source": "adaptive_player_hand",
            "support_center": [250, 350],
            "support_radius": 20,
        }]
        named_detections.return_value = [{
            "bbox": [10, 20, 20, 30],
            "confidence": 0.9,
        }]
        yolo.return_value.predict.return_value = [object()]

        enhanced = BallTracker().enhance_tracks_with_adaptive_crops(
            [frame],
            [{1: {"raw_candidates": [], "rim_regions": []}}],
            [{}],
        )

        self.assertEqual(enhanced[0][1]["raw_candidates"], [])
        self.assertEqual(enhanced[0][1]["adaptive_candidates_added"], 0)


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

        self.assertEqual(possession, [7, 7, 7, 7])

    def test_same_holder_is_bridged_across_short_compatible_gap(self):
        states = [
            {"holder_id": 7, "state": "confirmed", "confidence": 0.8,
             "candidate_id": 7, "reason": "holder_reinforced"},
            {"holder_id": None, "state": "unknown", "confidence": 0.0,
             "candidate_id": None, "reason": "ball_missing"},
            {"holder_id": None, "state": "loose", "confidence": 0.5,
             "candidate_id": 7, "reason": "interpolated_ball_not_confirmable"},
            {"holder_id": 7, "state": "confirmed", "confidence": 0.8,
             "candidate_id": 7, "reason": "holder_reinforced"},
        ]

        from backend.app.analytics.ball_holder_state import (
            _bridge_short_same_holder_gaps,
        )

        result = _bridge_short_same_holder_gaps(states, maximum_gap=2)

        self.assertEqual([state["holder_id"] for state in result], [7] * 4)
        self.assertEqual(result[1]["reason"], "same_holder_gap_bridged")

    def test_gap_with_competing_candidate_is_not_bridged(self):
        states = [
            {"holder_id": 7, "state": "confirmed", "confidence": 0.8,
             "candidate_id": 7, "reason": "holder_reinforced"},
            {"holder_id": None, "state": "loose", "confidence": 0.5,
             "candidate_id": 8, "reason": "no_credible_candidate"},
            {"holder_id": 7, "state": "confirmed", "confidence": 0.8,
             "candidate_id": 7, "reason": "holder_reinforced"},
        ]

        from backend.app.analytics.ball_holder_state import (
            _bridge_short_same_holder_gaps,
        )

        result = _bridge_short_same_holder_gaps(states, maximum_gap=2)

        self.assertIsNone(result[1]["holder_id"])

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

    def test_ball_interpolation_fills_only_short_bounded_gaps(self):
        ball_tracks = [
            {},
            {1: {"bbox": [0, 0, 4, 4], "confidence": 0.9}},
            {},
            {},
            {1: {"bbox": [6, 0, 10, 4], "confidence": 0.8}},
            {},
        ]

        interpolated = BallTracker.interpolate_positions(
            None,
            ball_tracks,
            max_gap_frames=2,
        )

        self.assertEqual(interpolated[0], {})
        self.assertEqual(interpolated[5], {})
        self.assertEqual(interpolated[2][1]["bbox"], [2.0, 0.0, 6.0, 4.0])
        self.assertEqual(interpolated[3][1]["bbox"], [4.0, 0.0, 8.0, 4.0])
        self.assertTrue(interpolated[2][1]["interpolated"])
        self.assertIsNone(interpolated[2][1]["confidence"])
        self.assertEqual(interpolated[2][1]["interpolation_gap_frames"], 2)
        self.assertEqual(interpolated[1][1]["position_source"], "observed")
        self.assertFalse(interpolated[1][1]["interpolated"])

    def test_ball_interpolation_leaves_long_gap_unknown(self):
        ball_tracks = [
            {1: {"bbox": [0, 0, 4, 4]}},
            {},
            {},
            {},
            {1: {"bbox": [8, 0, 12, 4]}},
        ]

        interpolated = BallTracker.interpolate_positions(
            None,
            ball_tracks,
            max_gap_frames=2,
        )

        self.assertEqual(interpolated[1:4], [{}, {}, {}])

    def test_ball_interpolation_does_not_cross_scene_cut(self):
        ball_tracks = [
            {1: {"bbox": [0, 0, 4, 4], "confidence": 0.8}},
            {},
            {},
            {1: {"bbox": [30, 0, 34, 4], "confidence": 0.8}},
        ]

        interpolated = BallTracker.interpolate_positions(
            None,
            ball_tracks,
            discontinuity_frames=[2],
        )

        self.assertEqual(interpolated[1:3], [{}, {}])

    def test_ball_interpolation_rejects_negative_gap_limit(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            BallTracker.interpolate_positions(None, [], max_gap_frames=-1)

    def test_ball_interpolation_does_not_cross_quarantined_candidate_chain(self):
        rejection = {
            "candidate_index": 0,
            "bbox": [20, 20, 24, 24],
            "confidence": 0.8,
            "reason": "persistent_competing_takeover_chain",
        }
        ball_tracks = [
            {1: {"bbox": [0, 0, 4, 4], "track_segment_id": 0}},
            {1: {
                "raw_candidates": [{
                    "candidate_index": 0,
                    "bbox": [20, 20, 24, 24],
                    "confidence": 0.8,
                }],
                "candidate_rejections": [rejection],
            }},
            {1: {"bbox": [4, 0, 8, 4], "track_segment_id": 0}},
        ]

        interpolated = BallTracker.interpolate_positions(None, ball_tracks)

        self.assertNotIn("bbox", interpolated[1][1])
        self.assertEqual(interpolated[1][1]["candidate_rejections"], [rejection])
        self.assertEqual(len(interpolated[1][1]["raw_candidates"]), 1)

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

    def test_ball_sequence_prefers_supported_track_over_static_distractor(self):
        detection_frames = [
            [
                {"bbox": [300, 10, 310, 25], "confidence": 0.95},
                {"bbox": [105, 105, 115, 120], "confidence": 0.55},
            ],
            [
                {"bbox": [300, 10, 310, 25], "confidence": 0.94},
                {"bbox": [110, 105, 120, 120], "confidence": 0.58},
            ],
            [
                {"bbox": [300, 10, 310, 25], "confidence": 0.96},
                {"bbox": [115, 105, 125, 120], "confidence": 0.61},
            ],
        ]
        players = [{7: {"bbox": [90, 80, 150, 180]}}] * 3

        selected = _select_ball_track(detection_frames, players)

        self.assertEqual(
            [frame[1]["bbox"] for frame in selected],
            [[105, 105, 115, 120], [110, 105, 120, 120], [115, 105, 125, 120]],
        )
        self.assertEqual(selected[0][1]["candidate_count"], 2)
        self.assertEqual(len(selected[0][1]["candidates"]), 2)

    def test_ball_sequence_can_leave_unsupported_distractor_unknown(self):
        detection_frames = [
            [{"bbox": [300, 10, 310, 25], "confidence": 0.95}],
            [{"bbox": [300, 10, 310, 25], "confidence": 0.95}],
        ]
        players = [{7: {"bbox": [0, 100, 40, 200]}}] * 2

        selected = _select_ball_track(detection_frames, players)

        self.assertNotIn("bbox", selected[0][1])
        self.assertNotIn("bbox", selected[1][1])
        self.assertEqual(selected[0][1]["candidate_count"], 1)

    def test_ball_sequence_requires_aligned_player_tracks(self):
        with self.assertRaisesRegex(ValueError, "must align"):
            _select_ball_track([[]], [])

    def test_ball_sequence_marks_track_reset_after_long_gap(self):
        detections = [
            [{"bbox": [10, 20, 16, 26], "confidence": 0.8}],
            *([[]] * 8),
            [{"bbox": [90, 20, 96, 26], "confidence": 0.8}],
        ]

        tracks = _select_ball_track(
            detections,
            [{} for _ in detections],
            max_observation_gap=8,
        )

        self.assertEqual(tracks[0][1]["track_segment_id"], 0)
        self.assertEqual(tracks[9][1]["track_segment_id"], 1)
        self.assertTrue(tracks[9][1]["tracking_discontinuity"])

    def test_cached_candidates_without_player_tracks_remain_supported(self):
        cached = [
            {1: {"candidates": [{
                "bbox": [10 + frame, 20, 16 + frame, 26],
                "confidence": 0.8,
            }]}}
            for frame in range(2)
        ]

        filtered = BallTracker.remove_wrong_detections(None, cached)

        self.assertEqual(len(filtered), len(cached))

    def test_sustained_upper_body_detection_is_rejected(self):
        tracks = [
            {1: {
                "bbox": [45, 35, 55, 45],
                "confidence": 0.8,
                "player_distance": 0.0,
            }}
            for _ in range(20)
        ]
        candidates = [[{
            "bbox": [45, 35, 55, 45],
            "confidence": 0.8,
        }]] * 20
        players = [{7: {"bbox": [0, 0, 100, 200]}}] * 20

        filtered = _reject_uncertain_observations(tracks, candidates, players)

        self.assertTrue(all(not frame[1].get("bbox") for frame in filtered))
        self.assertTrue(all(
            frame[1]["observation_rejection"]["reason"]
            == "sustained_central_upper_body_lock"
            for frame in filtered
        ))

    def test_brief_upper_body_detection_is_preserved(self):
        tracks = [
            {1: {
                "bbox": [45, 35, 55, 45],
                "confidence": 0.8,
                "player_distance": 0.0,
            }}
            for _ in range(2)
        ]
        candidates = [[{
            "bbox": [45, 35, 55, 45],
            "confidence": 0.8,
        }]] * 2
        players = [{7: {"bbox": [0, 0, 100, 200]}}] * 2

        filtered = _reject_uncertain_observations(tracks, candidates, players)

        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(1 in frame for frame in filtered))

    def test_five_frame_upper_body_lock_is_rejected(self):
        tracks = [
            {1: {
                "bbox": [45, 35, 55, 45],
                "confidence": 0.8,
                "player_distance": 0.0,
            }}
            for _ in range(5)
        ]
        candidates = [[{
            "bbox": [45, 35, 55, 45],
            "confidence": 0.8,
        }]] * 5
        players = [{7: {"bbox": [0, 0, 100, 200]}}] * 5

        filtered = _reject_uncertain_observations(tracks, candidates, players)

        self.assertTrue(all(not frame[1].get("bbox") for frame in filtered))
        self.assertTrue(all(frame[1]["observation_rejected"] for frame in filtered))

    def test_hand_supported_upper_body_ball_is_preserved(self):
        tracks = [
            {1: {
                "bbox": [45, 35, 55, 45],
                "confidence": 0.8,
                "player_distance": 0.0,
            }}
            for _ in range(5)
        ]
        candidates = [[{
            "bbox": [45, 35, 55, 45],
            "confidence": 0.8,
        }]] * 5
        keypoints = [[1, 1] for _ in range(17)]
        keypoints[7] = [35, 60]
        keypoints[9] = [50, 40]
        players = [{7: {
            "bbox": [0, 0, 100, 200],
            "pose": {
                "keypoints_xy": keypoints,
                "keypoint_confidences": [0.9] * 17,
            },
        }}] * 5

        filtered = _reject_uncertain_observations(tracks, candidates, players)

        self.assertTrue(all(1 in frame for frame in filtered))
        self.assertTrue(all(frame[1]["hand_pose_supported"] for frame in filtered))

    def test_low_confidence_wrist_does_not_exempt_body_lock(self):
        tracks = [
            {1: {
                "bbox": [45, 35, 55, 45],
                "confidence": 0.8,
                "player_distance": 0.0,
            }}
            for _ in range(3)
        ]
        candidates = [[{
            "bbox": [45, 35, 55, 45],
            "confidence": 0.8,
        }]] * 3
        keypoints = [[1, 1] for _ in range(17)]
        keypoints[7] = [35, 60]
        keypoints[9] = [50, 40]
        confidences = [0.9] * 17
        confidences[9] = 0.2
        players = [{7: {
            "bbox": [0, 0, 100, 200],
            "pose": {
                "keypoints_xy": keypoints,
                "keypoint_confidences": confidences,
            },
        }}] * 3

        filtered = _reject_uncertain_observations(tracks, candidates, players)

        self.assertTrue(all(not frame[1].get("bbox") for frame in filtered))
        self.assertTrue(all(
            frame[1]["observation_rejection"]["bbox"] == [45, 35, 55, 45]
            for frame in filtered
        ))

    def test_single_hand_pose_frame_does_not_exempt_body_lock(self):
        tracks = [
            {1: {
                "bbox": [45, 35, 55, 45],
                "confidence": 0.8,
                "player_distance": 0.0,
            }}
            for _ in range(3)
        ]
        candidates = [[{
            "bbox": [45, 35, 55, 45],
            "confidence": 0.8,
        }]] * 3
        keypoints = [[1, 1] for _ in range(17)]
        keypoints[7] = [35, 60]
        keypoints[9] = [50, 40]
        pose = {
            "keypoints_xy": keypoints,
            "keypoint_confidences": [0.9] * 17,
        }
        players = [
            {7: {"bbox": [0, 0, 100, 200]}},
            {7: {"bbox": [0, 0, 100, 200], "pose": pose}},
            {7: {"bbox": [0, 0, 100, 200]}},
        ]

        filtered = _reject_uncertain_observations(tracks, candidates, players)

        self.assertTrue(all(not frame[1].get("bbox") for frame in filtered))
        self.assertTrue(all(frame[1]["position_source"] == "rejected" for frame in filtered))

    def test_unsupported_selection_with_supported_competitor_is_rejected(self):
        selected_bbox = [200, 40, 210, 50]
        supported_bbox = [45, 90, 55, 100]
        tracks = [{1: {
            "bbox": selected_bbox,
            "confidence": 0.8,
            "player_distance": 100.0,
        }}]
        candidates = [[
            {"bbox": selected_bbox, "confidence": 0.8},
            {"bbox": supported_bbox, "confidence": 0.7},
        ]]
        players = [{7: {"bbox": [0, 0, 100, 200]}}]

        filtered = _reject_uncertain_observations(tracks, candidates, players)

        self.assertNotIn("bbox", filtered[0][1])
        self.assertEqual(
            filtered[0][1]["observation_rejection"]["reason"],
            "unsupported_selection_with_supported_competitor",
        )

    def test_body_locked_candidate_chain_survives_player_id_changes(self):
        detections = []
        players = []
        for frame in range(20):
            false_bbox = [45 + frame, 8, 51 + frame, 14]
            true_bbox = [10 + 12 * frame, 80, 16 + 12 * frame, 86]
            detections.append([
                {"bbox": false_bbox, "confidence": 0.8},
                {"bbox": true_bbox, "confidence": 0.7},
            ])
            player_id = 7 if frame % 2 == 0 else 8
            players.append({
                player_id: {"bbox": [frame, 0, 100 + frame, 200]},
            })

        pruned = _prune_sustained_body_locked_candidates(detections, players)

        self.assertEqual(
            [[candidate["bbox"] for candidate in frame] for frame in pruned],
            [[[10 + 12 * frame, 80, 16 + 12 * frame, 86]] for frame in range(20)],
        )

    def test_short_overhead_ball_candidate_is_not_pruned(self):
        detections = [[{
            "bbox": [45 + frame, 8, 51 + frame, 14],
            "confidence": 0.8,
        }] for frame in range(12)]
        players = [
            {7: {"bbox": [frame, 0, 100 + frame, 200]}}
            for frame in range(12)
        ]

        pruned = _prune_sustained_body_locked_candidates(detections, players)

        self.assertEqual(pruned, detections)

    def test_pose_locked_lower_body_object_is_reselected_to_free_candidate(self):
        keypoints = [[1, 1] for _ in range(17)]
        keypoints[7] = [40, 60]
        keypoints[9] = [50, 50]
        pose = {
            "keypoints_xy": keypoints,
            "keypoint_confidences": [0.9] * 17,
        }
        cached = []
        players = []
        expected = []
        for frame in range(6):
            locked = [45 + frame, 150, 55 + frame, 160]
            free = [110 + frame, 50, 120 + frame, 60]
            cached.append({1: {"candidates": [
                {"bbox": locked, "confidence": 0.8},
                {"bbox": free, "confidence": 0.6},
            ]}})
            players.append({7: {
                "bbox": [frame, 0, 100 + frame, 200],
                "pose": pose,
            }})
            expected.append(free)

        filtered = BallTracker.remove_wrong_detections(
            None,
            cached,
            player_tracks=players,
        )

        self.assertEqual(
            [frame[1].get("bbox") for frame in filtered],
            expected,
        )
        self.assertTrue(
            all(frame[1]["pose_locked_candidates_removed"] == 1 for frame in filtered)
        )

    def test_moving_lower_body_ball_is_not_pose_locked(self):
        keypoints = [[1, 1] for _ in range(17)]
        keypoints[7] = [40, 60]
        keypoints[9] = [50, 50]
        pose = {
            "keypoints_xy": keypoints,
            "keypoint_confidences": [0.9] * 17,
        }
        y_positions = [105, 145, 110, 150, 115, 155]
        cached = [
            {1: {"candidates": [{
                "bbox": [45 + frame, y, 55 + frame, y + 10],
                "confidence": 0.8,
            }]}}
            for frame, y in enumerate(y_positions)
        ]
        players = [{7: {
            "bbox": [frame, 0, 100 + frame, 200],
            "pose": pose,
        }} for frame in range(6)]

        filtered = BallTracker.remove_wrong_detections(
            None,
            cached,
            player_tracks=players,
        )

        self.assertTrue(all(1 in frame for frame in filtered))
        self.assertTrue(
            all(frame[1]["pose_locked_candidates_removed"] == 0 for frame in filtered)
        )


    def test_persistent_competing_chain_is_quarantined_before_takeover(self):
        cached = []
        players = []
        for frame in range(14):
            true_x = 10 + 10 * frame
            frame_candidates = []
            if frame < 8:
                frame_candidates.append({
                    "bbox": [true_x, 45, true_x + 6, 51],
                    "confidence": 0.95,
                })
            if frame < 8 or frame >= 10:
                frame_candidates.append({
                    "bbox": [175, 45, 181, 51],
                    "confidence": 0.50 if frame < 8 else 0.90,
                })
            cached.append({1: {"candidates": frame_candidates}})
            players.append({
                1: {"bbox": [true_x - 10, 0, true_x + 20, 100]},
                2: {"bbox": [160, 0, 200, 100]},
            })

        filtered = BallTracker.remove_wrong_detections(
            None,
            cached,
            player_tracks=players,
        )

        self.assertEqual(
            [filtered[frame][1].get("bbox") for frame in range(8)],
            [[10 + 10 * frame, 45, 16 + 10 * frame, 51] for frame in range(8)],
        )
        self.assertTrue(all(
            filtered[frame][1].get("bbox") is None
            for frame in range(10, 14)
        ))
        self.assertEqual(filtered[10][1]["raw_candidate_count"], 1)
        self.assertEqual(filtered[10][1]["takeover_chain_candidates_removed"], 1)
        rejection = filtered[10][1]["candidate_rejections"][0]
        self.assertEqual(rejection["reason"], "persistent_competing_takeover_chain")
        self.assertEqual(rejection["bbox"], [175, 45, 181, 51])
        self.assertGreater(
            rejection["chain_evidence"]["prediction_error_player_heights"],
            0.45,
        )

    def test_player_locked_chain_without_competing_ball_is_preserved(self):
        cached = [
            {1: {"candidates": [{
                "bbox": [45, 45, 51, 51],
                "confidence": 0.9,
            }]}}
            for _ in range(8)
        ]
        players = [{7: {"bbox": [0, 0, 100, 100]}} for _ in cached]

        filtered = BallTracker.remove_wrong_detections(
            None,
            cached,
            player_tracks=players,
        )

        self.assertTrue(all(frame[1].get("bbox") for frame in filtered))
        self.assertTrue(all(
            frame[1]["takeover_chain_candidates_removed"] == 0
            for frame in filtered
        ))

    def test_takeover_chain_does_not_link_across_discontinuity(self):
        cached = []
        players = []
        for frame in range(12):
            true_x = 10 + 10 * frame
            candidates = []
            if frame < 6:
                candidates.extend([
                    {
                        "bbox": [true_x, 45, true_x + 6, 51],
                        "confidence": 0.95,
                    },
                    {
                        "bbox": [175, 45, 181, 51],
                        "confidence": 0.50,
                    },
                ])
            elif frame >= 8:
                candidates.append({
                    "bbox": [175, 45, 181, 51],
                    "confidence": 0.90,
                })
            cached.append({1: {"candidates": candidates}})
            players.append({
                1: {"bbox": [true_x - 10, 0, true_x + 20, 100]},
                2: {"bbox": [160, 0, 200, 100]},
            })

        filtered = BallTracker.remove_wrong_detections(
            None,
            cached,
            player_tracks=players,
            discontinuity_frames=[8],
        )

        self.assertTrue(all(
            filtered[frame][1].get("bbox") == [175, 45, 181, 51]
            for frame in range(8, 12)
        ))
        self.assertTrue(all(
            filtered[frame][1]["takeover_chain_candidates_removed"] == 0
            for frame in range(8, 12)
        ))


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

    def test_low_confidence_switch_does_not_carry_stale_holder(self):
        states = self.model.process(
            [self.players] * 3,
            [self.ball(15), self.ball(15), self.ball(85, confidence=0.4)],
        )

        self.assertEqual(self.acquisitions(states), [-1, 10, -1])
        self.assertEqual(states[2]["reason"], "candidate_switch_building")

    def test_brief_missing_detection_preserves_holder(self):
        states = self.model.process(
            [self.players] * 5,
            [self.ball(15), self.ball(15), {}, {}, self.ball(15)],
        )
        self.assertEqual(self.acquisitions(states), [-1, 10, 10, 10, 10])
        self.assertEqual(states[2]["reason"], "brief_ball_gap")
        self.assertGreater(states[2]["confidence"], 0)

    def test_low_confidence_detection_does_not_carry_holder_through_gap(self):
        states = self.model.process(
            [self.players] * 3,
            [self.ball(15, confidence=0.6), self.ball(15, confidence=0.6), {}],
        )

        self.assertEqual(self.acquisitions(states), [-1, 10, -1])
        self.assertEqual(states[2]["reason"], "ball_missing")

    def test_stale_holder_is_not_carried_into_later_missing_frame(self):
        states = self.model.process(
            [self.players] * 7,
            [
                self.ball(15),
                self.ball(15),
                self.ball(200),
                self.ball(200),
                self.ball(200),
                self.ball(200),
                {},
            ],
        )

        self.assertEqual(states[-1]["holder_id"], None)
        self.assertEqual(states[-1]["reason"], "ball_missing")

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

    def test_hand_pose_evidence_resolves_overlapping_player_boxes(self):
        close_players = {
            10: {"bbox": [20, 0, 55, 60]},
            20: {"bbox": [45, 0, 80, 60]},
        }
        ball = self.ball(50)
        ball[1]["hand_pose_supported"] = True
        ball[1]["hand_pose_player_id"] = 20

        states = self.model.process([close_players] * 2, [ball, ball])

        self.assertEqual(states[-1]["holder_id"], 20)
        self.assertEqual(states[-1]["reason"], "initial_holder_confirmed")

    def test_missing_hand_pose_does_not_reduce_candidate_score(self):
        states = self.model.process(
            [self.players] * 2,
            [self.ball(15), self.ball(15)],
        )

        self.assertEqual(states[-1]["holder_id"], 10)

    def test_unsupported_ball_flying_through_player_is_not_control(self):
        players = [{10: {"bbox": [0, 0, 100, 100]}}] * 4
        balls = [
            {1: {"bbox": [48, y - 2, 52, y + 2], "confidence": 0.9}}
            for y in (85, 72, 58, 42)
        ]
        model = BallHolderStateModel(confirmation_frames=4)

        states = model.process(players, balls)

        self.assertIsNone(states[-1]["holder_id"])
        self.assertEqual(states[-1]["reason"], "airborne_candidate_not_confirmed")

    def test_hand_supported_motion_can_confirm_control(self):
        players = [{10: {"bbox": [0, 0, 100, 100]}}] * 4
        balls = [
            {1: {"bbox": [48, y - 2, 52, y + 2], "confidence": 0.9}}
            for y in (85, 72, 58, 42)
        ]
        for ball in balls:
            ball[1]["hand_pose_supported"] = True
            ball[1]["hand_pose_player_id"] = 10
        model = BallHolderStateModel(confirmation_frames=4)

        states = model.process(players, balls)

        self.assertEqual(states[-1]["holder_id"], 10)

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
    @staticmethod
    def shot_fixture(*, rebound_team=1):
        acquisitions = [10, 10, -1, -1, -1, -1, -1, 20, 20, 20]
        assignments = [
            {10: 1, 20: rebound_team}
            for _ in acquisitions
        ]
        centers = [
            (25, 180), (30, 180), (50, 145), (75, 105), (105, 78),
            (130, 70), (140, 105), (160, 150), (165, 165), (165, 165),
        ]
        balls = []
        for center_x, center_y in centers:
            balls.append({1: {
                "bbox": [center_x - 5, center_y - 5, center_x + 5, center_y + 5],
                "confidence": 0.9,
                "interpolated": False,
                "rim_regions": [{
                    "bbox": [115, 62, 145, 78],
                    "confidence": 0.9,
                }],
            }})
        players = [
            {
                10: {"bbox": [0, 80, 60, 220]},
                20: {"bbox": [135, 80, 195, 220]},
            }
            for _ in acquisitions
        ]
        timeline = PossessionTimelineBuilder(minimum_catch_frames=3).build(
            acquisitions,
            ball_tracks=balls,
        )
        return timeline, assignments, balls, players

    def test_shot_release_emits_attempt_without_outcome_or_rebound(self):
        timeline, assignments, balls, players = self.shot_fixture()

        result = ShotReboundDetector().detect(
            timeline,
            assignments,
            balls,
            players,
        )

        self.assertEqual(
            [event["type"] for event in result.events],
            ["shot_attempt"],
        )
        self.assertEqual(result.events[0]["frame_index"], 3)
        self.assertEqual(result.events[0]["release_frame"], 1)
        self.assertNotIn("outcome", result.events[0])
        self.assertNotIn("shot_subtype", result.events[0])
        self.assertEqual(result.frames[1]["state"], "shot_attempt")
        self.assertTrue(all(
            result.frames[frame]["state"] == "shot_in_flight"
            for frame in range(2, 6)
        ))
        self.assertEqual(result.frames[6]["state"], "possession")
        self.assertEqual(result.frames[7]["state"], "possession")
        self.assertEqual(result.sequences[0]["status"], "confirmed_attempt")
        self.assertEqual(result.sequences[0]["post_shot_control_frame"], 7)
        self.assertNotIn("outcome", result.sequences[0])

    def test_post_shot_team_does_not_change_attempt_only_contract(self):
        timeline, assignments, balls, players = self.shot_fixture(
            rebound_team=2
        )

        result = ShotReboundDetector().detect(
            timeline,
            assignments,
            balls,
            players,
        )

        self.assertEqual([event["type"] for event in result.events], [
            "shot_attempt",
        ])

    def test_downward_rim_crossing_does_not_publish_make_claim(self):
        timeline, assignments, balls, players = self.shot_fixture()
        balls[6][1]["bbox"] = [125, 78, 135, 88]

        result = ShotReboundDetector().detect(
            timeline,
            assignments,
            balls,
            players,
        )

        self.assertEqual([event["type"] for event in result.events], [
            "shot_attempt",
        ])
        self.assertNotIn("outcome", result.events[0])
        self.assertNotIn("made_evidence", result.sequences[0])
        self.assertEqual(result.sequences[0]["status"], "confirmed_attempt")

    def test_missing_post_shot_ball_does_not_publish_outcome(self):
        timeline, assignments, balls, players = self.shot_fixture()
        acquisitions = [10, 10, -1, -1, -1, -1, -1, -1, -1, -1]
        for frame in range(6, len(balls)):
            balls[frame] = {1: {
                "rim_regions": [{"bbox": [115, 62, 145, 78]}],
            }}
        timeline = PossessionTimelineBuilder(minimum_catch_frames=3).build(
            acquisitions,
            ball_tracks=balls,
        )

        result = ShotReboundDetector().detect(
            timeline,
            assignments,
            balls,
            players,
            discontinuity_frames={9},
        )

        self.assertEqual([event["type"] for event in result.events], [
            "shot_attempt",
        ])
        self.assertEqual(result.sequences[0]["status"], "confirmed_attempt")
        self.assertNotIn("outcome", result.sequences[0])

    def test_tracking_loss_without_stoppage_still_emits_attempt_only(self):
        timeline, assignments, balls, players = self.shot_fixture()
        acquisitions = [10, 10, -1, -1, -1, -1, -1, -1, -1, -1]
        for frame in range(6, len(balls)):
            balls[frame] = {1: {
                "rim_regions": [{"bbox": [115, 62, 145, 78]}],
            }}
        timeline = PossessionTimelineBuilder(minimum_catch_frames=3).build(
            acquisitions,
            ball_tracks=balls,
        )

        result = ShotReboundDetector().detect(
            timeline,
            assignments,
            balls,
            players,
        )

        self.assertEqual(result.sequences[0]["status"], "confirmed_attempt")
        self.assertEqual([event["type"] for event in result.events], [
            "shot_attempt",
        ])

    def test_sustained_low_confidence_post_shot_ball_does_not_add_event(self):
        timeline, assignments, balls, players = self.shot_fixture()
        acquisitions = [10, 10, -1, -1, -1, -1, -1, -1, -1, -1]
        for frame in range(6, len(balls)):
            balls[frame][1]["confidence"] = 0.1
        timeline = PossessionTimelineBuilder(minimum_catch_frames=3).build(
            acquisitions,
            ball_tracks=balls,
        )

        result = ShotReboundDetector().detect(
            timeline,
            assignments,
            balls,
            players,
            discontinuity_frames={9},
        )

        self.assertEqual([event["type"] for event in result.events], [
            "shot_attempt",
        ])
        self.assertNotIn("dead_ball_frame", result.sequences[0])

    def test_shot_without_stable_post_control_is_still_confirmed_attempt(self):
        timeline, assignments, balls, players = self.shot_fixture()
        timeline = PossessionTimelineBuilder(minimum_catch_frames=3).build(
            [10, 10, -1, -1, -1, -1, -1, 20, -1, -1],
            ball_tracks=balls,
        )

        result = ShotReboundDetector().detect(
            timeline,
            assignments,
            balls,
            players,
        )

        self.assertEqual([event["type"] for event in result.events], [
            "shot_attempt",
        ])
        self.assertEqual(result.sequences[0]["status"], "confirmed_attempt")
        self.assertIsNone(result.sequences[0]["post_shot_control_frame"])
        self.assertEqual(result.frames[-1]["state"], "possession")

    def test_control_run_spanning_rim_is_not_an_instant_rebound(self):
        acquisitions = [7, 7, -1, -1, 9, 9, 9, 9, 9, 9]

        self.assertIsNone(_find_stable_control(acquisitions, 6, 9, 3))

        acquisitions[7:] = [-1, 8, 8]
        self.assertIsNone(_find_stable_control(acquisitions, 6, 9, 3))

    def test_two_attempts_remain_plain_shot_attempts(self):
        acquisitions = [
            10, 10, -1, -1, -1, -1, -1,
            20, 20, 20, 20, -1, -1, -1, -1, -1, -1, -1,
        ]
        centers = [
            (25, 180), (30, 180), (50, 145), (75, 105), (105, 78),
            (130, 70), (140, 105), (160, 150), (165, 165), (165, 165),
            (165, 165), (160, 145), (150, 105), (140, 78), (130, 70),
            (125, 105), (120, 150), (120, 165),
        ]
        balls = [{1: {
            "bbox": [x - 5, y - 5, x + 5, y + 5],
            "confidence": 0.9,
            "interpolated": False,
            "rim_regions": [{"bbox": [115, 62, 145, 78]}],
        }} for x, y in centers]
        players = [{
            10: {"bbox": [0, 80, 60, 220]},
            20: {"bbox": [135, 80, 195, 220]},
        } for _ in acquisitions]
        assignments = [{10: 1, 20: 1} for _ in acquisitions]
        timeline = PossessionTimelineBuilder(minimum_catch_frames=3).build(
            acquisitions,
            ball_tracks=balls,
        )

        result = ShotReboundDetector().detect(
            timeline,
            assignments,
            balls,
            players,
        )

        shots = [event for event in result.events if event["type"] == "shot_attempt"]
        self.assertEqual(len(shots), 2)
        self.assertTrue(all("outcome" not in shot for shot in shots))
        self.assertTrue(all("shot_subtype" not in shot for shot in shots))

    def test_long_possession_is_split_at_distinct_rim_approaches(self):
        balls = []
        players = []
        for frame in range(60):
            center = (400, 300)
            if 14 <= frame <= 16:
                center = (130 + abs(15 - frame) * 10, 75)
            elif 44 <= frame <= 46:
                center = (130 + abs(45 - frame) * 10, 75)
            balls.append({1: {
                "bbox": [
                    center[0] - 5,
                    center[1] - 5,
                    center[0] + 5,
                    center[1] + 5,
                ],
                "rim_regions": [{"bbox": [115, 62, 145, 78]}],
            }})
            players.append({10: {"bbox": [0, 80, 60, 220]}})

        windows = _rim_candidate_windows(
            balls,
            players,
            10,
            0,
            59,
            20,
            2.0,
        )

        self.assertEqual([window[2] for window in windows], [15, 45])

    def test_horizontal_pass_does_not_enter_rebound_pending(self):
        timeline, assignments, balls, players = self.shot_fixture()
        for frame, ball in enumerate(balls):
            ball[1]["bbox"] = [20 + frame * 20, 115, 30 + frame * 20, 125]
            ball[1]["rim_regions"] = [{"bbox": [500, 60, 530, 80]}]

        result = ShotReboundDetector().detect(
            timeline,
            assignments,
            balls,
            players,
        )

        self.assertEqual(result.events, [])
        self.assertFalse(any(
            frame["state"] == "rebound_pending" for frame in result.frames
        ))

    def test_shot_sequence_replaces_covered_pass_interpretation(self):
        timeline, assignments, balls, players = self.shot_fixture()
        shot_result = ShotReboundDetector().detect(
            timeline,
            assignments,
            balls,
            players,
        )
        possession_events = [{
            "type": "pass",
            "frame_index": 7,
            "release_frame": 1,
            "from_player_id": 10,
            "to_player_id": 20,
            "from_team_id": 1,
            "to_team_id": 1,
        }]

        reconciled = reconcile_shot_events(possession_events, shot_result)

        self.assertEqual(
            [event["type"] for event in reconciled],
            ["shot_attempt"],
        )

    def test_stable_catch_before_rim_keeps_lob_as_pass(self):
        shot_timeline = ShotReboundTimeline(
            frames=[
                {"frame_index": frame, "state": "rebound_pending",
                 "sequence_id": 1}
                for frame in range(7)
            ],
            sequences=[{
                "sequence_id": 1,
                "shooter_id": 10,
                "release_frame": 2,
                "holder_release_frame": 0,
                "pending_end_frame": 6,
                "evidence": {"rim_approach_frame": 6},
            }],
            events=[{
                "type": "shot_attempt",
                "frame_index": 4,
                "sequence_id": 1,
            }],
            candidates=[],
        )
        pass_event = {
            "type": "pass",
            "frame_index": 5,
            "catch_frame": 5,
            "release_frame": 0,
            "from_player_id": 10,
        }

        reconciled = reconcile_shot_events([pass_event], shot_timeline)

        self.assertEqual(reconciled, [pass_event])
        self.assertEqual(shot_timeline.sequences, [])
        self.assertTrue(all(
            frame["state"] == "possession" for frame in shot_timeline.frames
        ))

    def test_post_rim_pass_preempts_unresolved_shot(self):
        shot_timeline = ShotReboundTimeline(
            frames=[
                {"frame_index": frame, "state": "rebound_pending",
                 "sequence_id": 1}
                for frame in range(13)
            ],
            sequences=[{
                "sequence_id": 1,
                "shooter_id": 10,
                "release_frame": 2,
                "rim_frame": 6,
                "pending_end_frame": 12,
            }],
            events=[{
                "type": "shot_attempt",
                "frame_index": 4,
                "sequence_id": 1,
            }],
            candidates=[],
        )
        pass_event = {
            "type": "pass",
            "frame_index": 10,
            "release_frame": 8,
            "from_player_id": 20,
            "to_player_id": 30,
        }

        reconciled = reconcile_shot_events([pass_event], shot_timeline)

        self.assertEqual(reconciled, [pass_event])
        self.assertEqual(shot_timeline.sequences, [])

    def test_possession_timeline_exposes_release_flight_and_stable_catch(self):
        acquisitions = [10, 10, -1, -1, 20, 20, 20]
        states = [
            {"state": "confirmed", "confidence": 0.8,
             "ball_confidence": 0.9, "reason": "holder_reinforced"},
            {"state": "confirmed", "confidence": 0.8,
             "ball_confidence": 0.9, "reason": "holder_reinforced"},
            {"state": "loose", "confidence": 0.3,
             "ball_confidence": 0.8, "reason": "no_credible_candidate"},
            {"state": "loose", "confidence": 0.3,
             "ball_confidence": 0.8, "reason": "no_credible_candidate"},
            {"state": "confirmed", "confidence": 0.8,
             "ball_confidence": 0.9, "reason": "initial_holder_confirmed"},
            {"state": "confirmed", "confidence": 0.8,
             "ball_confidence": 0.9, "reason": "holder_reinforced"},
            {"state": "confirmed", "confidence": 0.8,
             "ball_confidence": 0.9, "reason": "holder_reinforced"},
        ]
        balls = [
            {1: {"bbox": [frame, 0, frame + 2, 2], "confidence": 0.9}}
            for frame in range(len(acquisitions))
        ]

        timeline = PossessionTimelineBuilder(
            minimum_catch_frames=3,
        ).build(
            acquisitions,
            holder_states=states,
            ball_tracks=balls,
        )

        self.assertEqual(
            [frame["state"] for frame in timeline.frames],
            [
                "controlled",
                "controlled",
                "released",
                "in_flight",
                "controlled",
                "controlled",
                "controlled",
            ],
        )
        self.assertEqual(len(timeline.transitions), 1)
        self.assertEqual(timeline.transitions[0]["status"], "confirmed")
        self.assertEqual(timeline.transitions[0]["release_frame"], 1)
        self.assertEqual(timeline.transitions[0]["catch_frame"], 4)
        self.assertEqual(timeline.transitions[0]["observed_loose_frames"], 2)

    def test_possession_timeline_rejects_receiver_without_stable_control(self):
        timeline = PossessionTimelineBuilder(
            minimum_catch_frames=3,
            catch_confirmation_frames=5,
        ).build([10, -1, 20, -1, -1])

        self.assertEqual(len(timeline.transitions), 1)
        self.assertEqual(timeline.transitions[0]["status"], "rejected")
        self.assertEqual(
            timeline.transitions[0]["reason"],
            "receiver_control_not_sustained",
        )

    def test_possession_timeline_does_not_transfer_across_scene_cut(self):
        timeline = PossessionTimelineBuilder(minimum_catch_frames=2).build(
            [10, 10, 20, 20],
            discontinuity_frames=[2],
        )

        self.assertEqual(timeline.transitions, [])

    def test_minimum_source_support_suppresses_initial_track_handoff(self):
        detector = PassInterceptionDetector(
            minimum_catch_frames=3,
            minimum_source_frames=5,
        )
        acquisitions = [10, 10, 10, 10, -1, 20, 20, 20]
        assignments = [{10: 1}] * 5 + [{20: 1}] * 3

        self.assertEqual(
            detector.detect_events(acquisitions, assignments),
            [],
        )

    def test_initial_source_requirement_does_not_reject_later_short_source(self):
        detector = PassInterceptionDetector(
            minimum_catch_frames=3,
            minimum_source_frames=3,
            minimum_initial_source_frames=6,
        )
        acquisitions = [99] * 6 + [-1, 10, 10, 10, -1, 20, 20, 20]
        assignments = (
            [{99: 2}] * 6
            + [{}]
            + [{10: 1}] * 3
            + [{}]
            + [{20: 1}] * 3
        )

        events = detector.detect_events(acquisitions, assignments)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1]["from_player_id"], 10)
        self.assertEqual(events[-1]["to_player_id"], 20)

    def test_long_flight_is_not_suppressed_by_close_endpoints(self):
        detector = PassInterceptionDetector(minimum_catch_frames=2)
        acquisitions = [10] + [-1] * 10 + [20, 20]
        assignments = [{10: 1}] + [{}] * 10 + [{20: 2}, {20: 2}]
        states = (
            [{"state": "confirmed", "ball_confidence": 0.9}]
            + [{"state": "loose", "ball_confidence": 0.8}] * 10
            + [{"state": "confirmed", "ball_confidence": 0.8}] * 2
        )
        players = [
            {
                10: {"bbox": [0, 0, 50, 100]},
                20: {"bbox": [30, 0, 80, 100]},
            }
            for _ in acquisitions
        ]

        events = detector.detect_events(
            acquisitions,
            assignments,
            holder_states=states,
            player_tracks=players,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "interception")

    def test_internal_linear_holder_bridge_can_confirm_pass_catch(self):
        detector = PassInterceptionDetector(
            minimum_catch_frames=3,
            catch_confirmation_frames=6,
        )
        acquisitions = [10, -1, 20, 20, 20]
        assignments = [{10: 1}, {}, {20: 1}, {20: 1}, {20: 1}]
        states = [
            {"holder_id": 10, "state": "confirmed",
             "reason": "holder_reinforced", "ball_confidence": 0.8},
            {"holder_id": None, "state": "loose",
             "reason": "ball_missing", "ball_confidence": None},
            {"holder_id": 20, "state": "confirmed",
             "reason": "initial_holder_confirmed", "ball_confidence": 0.7},
            {"holder_id": 20, "state": "confirmed",
             "reason": "same_holder_gap_bridged", "ball_confidence": None},
            {"holder_id": 20, "state": "confirmed",
             "reason": "holder_reinforced", "ball_confidence": 0.7},
        ]
        ball_tracks = [
            {1: {"bbox": [10, 40, 20, 50], "track_segment_id": 0}},
            {},
            {1: {"bbox": [110, 40, 120, 50], "track_segment_id": 1,
                 "interpolated": False}},
            {1: {"bbox": [120, 40, 130, 50], "track_segment_id": 1,
                 "interpolated": True}},
            {1: {"bbox": [130, 40, 140, 50], "track_segment_id": 1,
                 "interpolated": False}},
        ]
        players = [
            {
                10: {"bbox": [0, 0, 50, 100]},
                20: {"bbox": [100, 0, 150, 100]},
            }
            for _ in acquisitions
        ]

        self.assertEqual(
            detector.detect_events(
                acquisitions,
                assignments,
                holder_states=states,
                player_tracks=players,
            ),
            [],
        )
        events = detector.detect_events(
            acquisitions,
            assignments,
            holder_states=states,
            ball_tracks=ball_tracks,
            player_tracks=players,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "pass")
        self.assertEqual(events[0]["catch_frame"], 2)

    def test_event_team_hint_requires_strong_unknown_track_plurality(self):
        metadata = {"track_assignments": {
            "3": {
                "status": "unknown",
                "reason": "inconsistent_track_observations",
                "confident_observation_count": 24,
                "agreement": 0.75,
                "weight_share": 0.76,
                "team_vote_weights": {"1": 2.5, "2": 8.0},
            },
            "4": {
                "status": "unknown",
                "reason": "insufficient_confident_observations",
                "confident_observation_count": 4,
                "agreement": 1.0,
                "weight_share": 1.0,
                "team_vote_weights": {"1": 0.0, "2": 3.0},
            },
        }}

        self.assertEqual(build_event_team_hints(metadata), {3: 2})

    def test_video_2_regression_merges_four_spatially_credible_passes(self):
        def event(frame, source, receiver, *, event_type="pass"):
            return {
                "type": event_type,
                "frame_index": frame,
                "from_player_id": source,
                "to_player_id": receiver,
                "from_team_id": 2,
                "to_team_id": 2,
                "release_frame": frame - 2,
                "catch_frame": frame,
                "gap_frames": 1,
            }

        player_boxes = {
            8: [0, 0, 50, 100],
            3: [40, 0, 90, 100],
            4: [250, 0, 300, 100],
            5: [500, 0, 550, 100],
            11: [750, 0, 800, 100],
        }
        player_tracks = [
            {
                player_id: {"bbox": list(bbox)}
                for player_id, bbox in player_boxes.items()
            }
            for _ in range(174)
        ]
        primary = [event(107, 5, 11)]
        corroborating = [
            event(25, 8, 3),  # Close-range holder/track switch.
            event(43, 3, 4),
            event(64, 4, 5),
            event(107, 5, 11),  # Duplicate of semantic event.
            event(133, 11, 5),
            event(150, 5, 4, event_type="interception"),
        ]

        merged = merge_corroborated_pass_events(
            primary,
            corroborating,
            player_tracks,
        )

        self.assertEqual(
            [item["frame_index"] for item in merged],
            [43, 64, 107, 133],
        )
        self.assertEqual(
            [item["detection_source"] for item in merged],
            [
                "fused_ball_corroboration",
                "fused_ball_corroboration",
                "semantic_ball",
                "fused_ball_corroboration",
            ],
        )

    def test_event_rejects_reset_to_preexisting_competing_object(self):
        detector = PassInterceptionDetector(minimum_catch_frames=2)
        acquisitions = [10, 10, -1, 20, 20]
        assignments = [{10: 1}, {10: 1}, {}, {20: 1}, {20: 1}]
        ball_tracks = [
            {1: {
                "bbox": [0, 0, 5, 5],
                "candidates": [],
                "raw_candidates": [{"bbox": [80, 30, 85, 35]}],
                "track_segment_id": 0,
            }},
            {1: {
                "bbox": [2, 0, 7, 5],
                "candidates": [],
                "raw_candidates": [{"bbox": [80, 30, 85, 35]}],
                "track_segment_id": 0,
            }},
            {},
            {1: {"bbox": [80, 30, 85, 35], "track_segment_id": 1}},
            {1: {"bbox": [80, 30, 85, 35], "track_segment_id": 1}},
        ]
        players = [{20: {"bbox": [70, 0, 100, 100]}} for _ in acquisitions]

        self.assertEqual(
            detector.detect_events(
                acquisitions,
                assignments,
                ball_tracks=ball_tracks,
                player_tracks=players,
            ),
            [],
        )
        baseline_detector = PassInterceptionDetector(
            minimum_catch_frames=2,
            reject_preexisting_competing_takeovers=False,
        )
        self.assertEqual(
            len(baseline_detector.detect_events(
                acquisitions,
                assignments,
                ball_tracks=ball_tracks,
                player_tracks=players,
            )),
            1,
        )

    def test_event_can_use_continuous_ball_track_segment(self):
        detector = PassInterceptionDetector(minimum_catch_frames=2)
        acquisitions = [10, -1, 20, 20]
        assignments = [{10: 1}, {}, {20: 1}, {20: 1}]
        ball_tracks = [
            {1: {"bbox": [0, 0, 5, 5], "track_segment_id": 0}},
            {},
            {1: {"bbox": [80, 0, 85, 5], "track_segment_id": 0}},
            {1: {"bbox": [80, 0, 85, 5], "track_segment_id": 0}},
        ]

        events = detector.detect_events(
            acquisitions,
            assignments,
            ball_tracks=ball_tracks,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "pass")

    def test_event_rejects_transition_across_quarantined_takeover_chain(self):
        detector = PassInterceptionDetector(minimum_catch_frames=2)
        acquisitions = [10, -1, 20, 20]
        assignments = [{10: 1}, {}, {20: 1}, {20: 1}]
        ball_tracks = [
            {1: {"bbox": [0, 0, 5, 5], "track_segment_id": 0}},
            {1: {"candidate_rejections": [{
                "reason": "persistent_competing_takeover_chain",
                "candidate_chain_id": "takeover_chain_0",
            }]}},
            {1: {"bbox": [80, 0, 85, 5], "track_segment_id": 1}},
            {1: {"bbox": [80, 0, 85, 5], "track_segment_id": 1}},
        ]

        events = detector.detect_events(
            acquisitions,
            assignments,
            ball_tracks=ball_tracks,
            player_tracks=[{} for _ in acquisitions],
        )

        self.assertEqual(events, [])

    def test_scene_cut_resets_holder_history(self):
        detector = PassInterceptionDetector(minimum_catch_frames=2)
        acquisitions = [10, 10, 20, 20]
        assignments = [{10: 1}, {10: 1}, {20: 1}, {20: 1}]

        self.assertEqual(
            detector.detect_events(
                acquisitions,
                assignments,
                discontinuity_frames=[2],
            ),
            [],
        )

    def test_one_frame_source_holder_after_cut_does_not_emit_pass(self):
        detector = PassInterceptionDetector(
            minimum_catch_frames=2,
            catch_confirmation_frames=6,
        )
        acquisitions = [99, 99, 10, -1, 20, 20]
        assignments = [
            {99: 2},
            {99: 2},
            {10: 1},
            {},
            {20: 1},
            {20: 1},
        ]

        self.assertEqual(
            detector.detect_events(
                acquisitions,
                assignments,
                discontinuity_frames=[2],
            ),
            [],
        )

    def test_catch_confirmation_cannot_cross_scene_cut(self):
        detector = PassInterceptionDetector(
            minimum_catch_frames=3,
            catch_confirmation_frames=8,
        )
        acquisitions = [10, -1, 20, -1, 20, 20, 20]
        assignments = [{10: 1}, {}, {20: 1}, {}, {20: 1}, {20: 1}, {20: 1}]

        self.assertEqual(
            detector.detect_events(
                acquisitions,
                assignments,
                discontinuity_frames=[4],
            ),
            [],
        )

    def test_transient_cleanup_does_not_join_episodes_across_cut(self):
        detector = PassInterceptionDetector(transient_control_frames=8)
        acquisitions = [10, 20, 20, 30, 30]
        assignments = [
            {10: 1},
            {20: 2},
            {20: 2},
            {30: 2},
            {30: 2},
        ]

        self.assertEqual(
            detector.clean_transient_control_chains(
                acquisitions,
                assignments,
                discontinuity_frames=[3],
            ),
            acquisitions,
        )

    def test_rising_ball_through_background_player_is_not_a_pass(self):
        detector = PassInterceptionDetector(minimum_catch_frames=3)
        acquisitions = [10, -1, -1, 20, 20, 20, 20]
        assignments = [{10: 1}] + [{20: 1}] * 6
        players = [
            {
                10: {"bbox": [0, 100, 60, 220]},
                20: {"bbox": [20, 0, 80, 140]},
            }
            for _ in acquisitions
        ]
        ball_tracks = [
            {1: {
                "bbox": [30, 190 - 25 * frame, 40, 200 - 25 * frame],
                "confidence": 0.8,
                "interpolated": False,
            }}
            for frame in range(len(acquisitions))
        ]

        self.assertEqual(
            detector.detect_events(
                acquisitions,
                assignments,
                ball_tracks=ball_tracks,
                player_tracks=players,
            ),
            [],
        )

    def test_horizontal_ball_path_remains_a_pass(self):
        detector = PassInterceptionDetector(minimum_catch_frames=3)
        acquisitions = [10, -1, -1, 20, 20, 20, 20]
        assignments = [{10: 1}] + [{20: 1}] * 6
        players = [
            {
                10: {"bbox": [0, 80, 60, 220]},
                20: {"bbox": [100, 80, 160, 220]},
            }
            for _ in acquisitions
        ]
        ball_tracks = [
            {1: {
                "bbox": [20 + 20 * frame, 120, 30 + 20 * frame, 130],
                "confidence": 0.8,
                "interpolated": False,
            }}
            for frame in range(len(acquisitions))
        ]

        events = detector.detect_events(
            acquisitions,
            assignments,
            ball_tracks=ball_tracks,
            player_tracks=players,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "pass")

    def test_close_range_opponent_takeover_is_not_an_interception(self):
        detector = PassInterceptionDetector(minimum_catch_frames=2)
        acquisitions = [10, -1, -1, 20, 20]
        assignments = [{10: 1}, {}, {}, {20: 2}, {20: 2}]
        states = [
            {"state": "confirmed", "ball_confidence": 0.9},
            {"state": "loose", "ball_confidence": 0.8},
            {"state": "loose", "ball_confidence": 0.8},
            {"state": "confirmed", "ball_confidence": 0.8},
            {"state": "confirmed", "ball_confidence": 0.8},
        ]
        players = [
            {
                10: {"bbox": [0, 0, 50, 100]},
                20: {"bbox": [30, 0, 80, 100]},
            }
            for _ in acquisitions
        ]

        self.assertEqual(
            detector.detect_events(
                acquisitions,
                assignments,
                holder_states=states,
                player_tracks=players,
            ),
            [],
        )

    def test_spatially_separated_opponent_takeover_is_an_interception(self):
        detector = PassInterceptionDetector(minimum_catch_frames=2)
        acquisitions = [10, -1, 20, 20]
        assignments = [{10: 1}, {}, {20: 2}, {20: 2}]
        states = [
            {"state": "confirmed", "ball_confidence": 0.9},
            {"state": "loose", "ball_confidence": 0.8},
            {"state": "confirmed", "ball_confidence": 0.8},
            {"state": "confirmed", "ball_confidence": 0.8},
        ]
        players = [
            {
                10: {"bbox": [0, 0, 50, 100]},
                20: {"bbox": [200, 0, 250, 100]},
            }
            for _ in acquisitions
        ]

        events = detector.detect_events(
            acquisitions,
            assignments,
            holder_states=states,
            player_tracks=players,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "interception")

    def test_immediate_close_range_takeaway_remains_an_interception(self):
        detector = PassInterceptionDetector(minimum_catch_frames=2)
        acquisitions = [10, -1, 20, 20]
        assignments = [{10: 1}, {}, {20: 2}, {20: 2}]
        states = [
            {"state": "confirmed", "ball_confidence": 0.9},
            {"state": "loose", "ball_confidence": 0.8},
            {"state": "confirmed", "ball_confidence": 0.8},
            {"state": "confirmed", "ball_confidence": 0.8},
        ]
        players = [
            {
                10: {"bbox": [0, 0, 50, 100]},
                20: {"bbox": [30, 0, 80, 100]},
            }
            for _ in acquisitions
        ]

        events = detector.detect_events(
            acquisitions,
            assignments,
            holder_states=states,
            player_tracks=players,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "interception")

    def test_offline_possession_recovery_cannot_create_event_boundary(self):
        detector = PassInterceptionDetector(
            minimum_catch_frames=2,
            catch_confirmation_frames=5,
        )
        acquisitions = [10, -1, 20, 20, 20]
        assignments = [{10: 1}, {}, {20: 1}, {20: 1}, {20: 1}]
        states = [
            {"state": "confirmed", "reason": "holder_reinforced",
             "ball_confidence": 0.9},
            {"state": "loose", "reason": "ball_missing",
             "ball_confidence": None},
            {"state": "confirmed", "reason": "retrospective_holder_confirmation",
             "ball_confidence": 0.8},
            {"state": "confirmed", "reason": "initial_holder_confirmed",
             "ball_confidence": 0.8},
            {"state": "confirmed", "reason": "holder_reinforced",
             "ball_confidence": 0.8},
        ]

        events = detector.detect_events(
            acquisitions,
            assignments,
            holder_states=states,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["catch_frame"], 3)

        states[2]["reason"] = "same_holder_gap_bridged"
        events = detector.detect_events(
            acquisitions,
            assignments,
            holder_states=states,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["catch_frame"], 3)

    def test_event_waits_for_stable_receiver_after_provisional_catch(self):
        detector = PassInterceptionDetector(
            minimum_catch_frames=3,
            catch_confirmation_frames=8,
        )
        acquisitions = [10, -1, 20, -1, 20, 20, 20]
        assignments = [{10: 1}, {}, {20: 1}, {}, {20: 1}, {20: 1}, {20: 1}]

        events = detector.detect_events(acquisitions, assignments)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "pass")
        self.assertEqual(events[0]["release_frame"], 0)
        self.assertEqual(events[0]["transition_frame"], 2)
        self.assertEqual(events[0]["catch_frame"], 4)

    def test_event_rejects_receiver_that_never_establishes_control(self):
        detector = PassInterceptionDetector(
            minimum_catch_frames=3,
            catch_confirmation_frames=8,
        )
        acquisitions = [10, -1, 20, -1, -1]
        assignments = [{10: 1}, {}, {20: 1}, {}, {}]

        self.assertEqual(detector.detect_events(acquisitions, assignments), [])

    def test_catch_can_confirm_at_confirmation_window_boundary(self):
        detector = PassInterceptionDetector(
            minimum_catch_frames=3,
            catch_confirmation_frames=5,
        )
        acquisitions = [10, 20, -1, -1, 20, 20, 20]
        assignments = [{10: 1}] + [{20: 1}] * 6

        events = detector.detect_events(acquisitions, assignments)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["transition_frame"], 1)
        self.assertEqual(events[0]["catch_frame"], 4)

    def test_interception_rejects_weak_catch_observation(self):
        detector = PassInterceptionDetector(
            minimum_catch_frames=2,
            minimum_catch_ball_confidence=0.45,
        )
        acquisitions = [10, -1, 20, 20]
        assignments = [{10: 1}, {}, {20: 2}, {20: 2}]
        states = [
            {"state": "confirmed", "ball_confidence": 0.9},
            {"state": "loose", "ball_confidence": 0.8},
            {"state": "confirmed", "ball_confidence": 0.3},
            {"state": "confirmed", "ball_confidence": 0.9},
        ]

        self.assertEqual(
            detector.detect_events(
                acquisitions,
                assignments,
                holder_states=states,
            ),
            [],
        )

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

    def test_weak_observed_loose_ball_does_not_allow_interception(self):
        detector = PassInterceptionDetector(minimum_loose_ball_confidence=0.5)
        acquisitions = [10, -1, 20]
        assignments = [{10: 1}, {}, {20: 2}]
        weak_loose = [
            {"state": "confirmed", "ball_confidence": 0.9},
            {"state": "loose", "ball_confidence": 0.31},
            {"state": "confirmed", "ball_confidence": 0.8},
        ]

        self.assertEqual(
            detector.detect_interceptions(
                acquisitions,
                assignments,
                holder_states=weak_loose,
            ),
            [-1, -1, -1],
        )

    def test_brief_opponent_island_returning_to_source_is_removed(self):
        detector = PassInterceptionDetector(transient_control_frames=8)
        acquisitions = [10, 10, -1, 20, 20, -1, 10]
        assignments = [
            {10: 1},
            {10: 1},
            {},
            {20: 2},
            {20: 2},
            {},
            {10: 1},
        ]

        cleaned = detector.clean_transient_control_chains(
            acquisitions,
            assignments,
        )

        self.assertEqual(cleaned, [10, 10, -1, -1, -1, -1, 10])
        self.assertEqual(detector.detect_passes(cleaned, assignments), [-1] * 7)
        self.assertEqual(
            detector.detect_interceptions(cleaned, assignments),
            [-1] * 7,
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
