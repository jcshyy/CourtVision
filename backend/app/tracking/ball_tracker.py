import numpy as np
import supervision as sv
from ultralytics import YOLO

from backend.app.config import BALL_DETECTOR_PATH, WASB_BALL_DETECTOR_PATH
from backend.app.detection.wasb_ball_detector import WasbBallDetector
from backend.app.utils import load_cache, save_cache

BALL_DETECTION_IMAGE_SIZE = 640
BALL_DETECTION_BATCH_SIZE = 20
BALL_FULL_FRAME_CONFIDENCE = 0.25
BALL_ADAPTIVE_CONFIDENCE = 0.50
BALL_ADAPTIVE_TRIGGER_CONFIDENCE = 0.55
BALL_ADAPTIVE_MAX_CROPS_PER_FRAME = 4
BALL_TRACKING_CACHE_VERSION = "v6_adaptive_roi_full_pass"
BALL_ADAPTIVE_CACHE_VERSION = "v2_gated_predicted_hand_rim_conf050"
BALL_DETECTOR_BACKENDS = ("yolo", "wasb", "hybrid")
WASB_INTEGRATION_VERSION = "wasb_hrnet_step1_v1"
SOURCE_AWARE_HYBRID_VERSION = "wasb_hrnet_step1_source_aware_v2"
EBARD_BALL_CACHE_VERSION = "v7_ebard_shared_scene"
EBARD_HYBRID_CACHE_VERSION = "v7_ebard_wasb_shared_scene"
EBARD_ADAPTIVE_CACHE_VERSION = "v3_ebard_shared_scene"
EBARD_ADAPTIVE_HYBRID_CACHE_VERSION = "v3_ebard_wasb_shared_scene"
BALL_CLASS_ALIASES = {"ball", "basketball"}
HOOP_CLASS_ALIASES = {"hoop", "rim"}


class BallTracker:
    """Detects and filters basketball positions using the reference repo flow."""

    def __init__(
        self,
        model_path=BALL_DETECTOR_PATH,
        *,
        detector_backend="yolo",
        wasb_model_path=WASB_BALL_DETECTOR_PATH,
        semantic_model=None,
        semantic_detector_backend="current",
    ):
        if detector_backend not in BALL_DETECTOR_BACKENDS:
            raise ValueError(
                f"Unknown ball detector backend {detector_backend!r}; "
                f"expected one of {BALL_DETECTOR_BACKENDS}"
            )
        self.detector_backend = detector_backend
        self.semantic_detector_backend = semantic_detector_backend
        self.model = (
            (semantic_model or YOLO(model_path))
            if detector_backend in ("yolo", "hybrid")
            else None
        )
        self.wasb_detector = (
            WasbBallDetector(wasb_model_path)
            if detector_backend in ("wasb", "hybrid")
            else None
        )

    @property
    def cache_version(self):
        if self.semantic_detector_backend == "ebard":
            return (
                EBARD_HYBRID_CACHE_VERSION
                if self.detector_backend == "hybrid"
                else EBARD_BALL_CACHE_VERSION
                if self.detector_backend == "yolo"
                else f"{BALL_TRACKING_CACHE_VERSION}_{self.detector_backend}"
            )
        if self.detector_backend == "yolo":
            return BALL_TRACKING_CACHE_VERSION
        if self.detector_backend == "hybrid":
            return (
                f"{BALL_TRACKING_CACHE_VERSION}_hybrid_"
                f"{SOURCE_AWARE_HYBRID_VERSION}"
            )
        return (
            f"{BALL_TRACKING_CACHE_VERSION}_{self.detector_backend}_"
            f"{WASB_INTEGRATION_VERSION}"
        )

    @property
    def adaptive_cache_version(self):
        if self.semantic_detector_backend == "ebard":
            return (
                EBARD_ADAPTIVE_HYBRID_CACHE_VERSION
                if self.detector_backend == "hybrid"
                else EBARD_ADAPTIVE_CACHE_VERSION
                if self.detector_backend == "yolo"
                else f"{BALL_ADAPTIVE_CACHE_VERSION}_{self.detector_backend}"
            )
        if self.detector_backend == "yolo":
            return BALL_ADAPTIVE_CACHE_VERSION
        if self.detector_backend == "hybrid":
            return (
                f"{BALL_ADAPTIVE_CACHE_VERSION}_hybrid_"
                f"{SOURCE_AWARE_HYBRID_VERSION}"
            )
        return (
            f"{BALL_ADAPTIVE_CACHE_VERSION}_{self.detector_backend}_"
            f"{WASB_INTEGRATION_VERSION}"
        )

    def detect_frames(self, frames):
        if self.model is None:
            raise RuntimeError("YOLO detection is disabled for the WASB-only backend")
        detections = []

        for start in range(0, len(frames), BALL_DETECTION_BATCH_SIZE):
            detections.extend(
                self.model.predict(
                    frames[start : start + BALL_DETECTION_BATCH_SIZE],
                    conf=BALL_FULL_FRAME_CONFIDENCE,
                    imgsz=BALL_DETECTION_IMAGE_SIZE,
                    verbose=False,
                )
            )

        return detections

    def get_object_tracks(
        self,
        frames,
        read_from_cache=False,
        cache_path=None,
        player_tracks=None,
        detections=None,
        detections_provider=None,
    ):
        tracks = load_cache(cache_path, enabled=read_from_cache) if cache_path else None
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        candidate_frames = [[] for _ in frames]
        semantic_candidate_frames = [[] for _ in frames]
        rim_frames = [[] for _ in frames]
        if self.model is not None:
            if detections is None:
                detections = (
                    detections_provider()
                    if detections_provider is not None
                    else self.detect_frames(frames)
                )
            if len(detections) != len(frames):
                raise ValueError("Ball detections and video frames must align")
            for frame_index, detection in enumerate(detections):
                candidate_frames[frame_index] = [
                    {
                        **item,
                        "detection_source": (
                            f"{self.semantic_detector_backend}_full_frame"
                        ),
                    }
                    for item in _named_detections(detection, "Ball")
                ]
                semantic_candidate_frames[frame_index] = [
                    dict(item) for item in candidate_frames[frame_index]
                ]
                rim_frames[frame_index] = _named_detections(detection, "Hoop")
        if self.wasb_detector is not None:
            wasb_frames = self.wasb_detector.detect_frames(frames, step=1)
            candidate_frames = [
                _merge_ball_candidates(yolo_candidates, wasb_candidates)
                for yolo_candidates, wasb_candidates in zip(
                    candidate_frames,
                    wasb_frames,
                )
            ]

        tracks = _select_ball_track(candidate_frames, player_tracks)
        for frame_index, (frame, rims) in enumerate(zip(tracks, rim_frames)):
            frame[1]["rim_regions"] = [dict(rim) for rim in rims]
            frame[1]["adaptive_second_pass_completed"] = False
            if self.detector_backend == "hybrid":
                frame[1]["semantic_raw_candidates"] = [
                    dict(candidate)
                    for candidate in semantic_candidate_frames[frame_index]
                ]

        if cache_path:
            save_cache(cache_path, tracks)

        return tracks

    def enhance_tracks_with_adaptive_crops(
        self,
        frames,
        ball_tracks,
        player_tracks,
        *,
        read_from_cache=False,
        cache_path=None,
    ):
        """Add high-resolution ROI candidates to uncertain full-frame detections."""
        if not (len(frames) == len(ball_tracks) == len(player_tracks)):
            raise ValueError("Frames, ball tracks, and player tracks must align")
        cached = load_cache(cache_path, enabled=read_from_cache) if cache_path else None
        if cached is not None and len(cached) == len(frames):
            return cached
        if self.model is None:
            for frame in ball_tracks:
                info = frame.setdefault(1, {})
                info["adaptive_second_pass_completed"] = False
                info["adaptive_crop_count"] = 0
                info["adaptive_candidates_added"] = 0
            if cache_path:
                save_cache(cache_path, ball_tracks)
            return ball_tracks

        requests = _adaptive_crop_requests(frames, ball_tracks, player_tracks)
        crops = [
            frames[request["frame_index"]][
                request["bbox"][1] : request["bbox"][3],
                request["bbox"][0] : request["bbox"][2],
            ]
            for request in requests
        ]
        adaptive_candidates = [[] for _ in frames]
        for start in range(0, len(crops), BALL_DETECTION_BATCH_SIZE):
            batch_requests = requests[start : start + BALL_DETECTION_BATCH_SIZE]
            batch_results = self.model.predict(
                crops[start : start + BALL_DETECTION_BATCH_SIZE],
                conf=BALL_ADAPTIVE_CONFIDENCE,
                imgsz=BALL_DETECTION_IMAGE_SIZE,
                verbose=False,
            )
            for request, result in zip(batch_requests, batch_results):
                origin_x, origin_y = request["bbox"][:2]
                for detection in _named_detections(result, "Ball"):
                    bbox = detection["bbox"]
                    translated_bbox = [
                        bbox[0] + origin_x,
                        bbox[1] + origin_y,
                        bbox[2] + origin_x,
                        bbox[3] + origin_y,
                    ]
                    support_center = request.get("support_center")
                    support_radius = request.get("support_radius")
                    if (
                        support_center is not None
                        and support_radius is not None
                        and math_dist(_bbox_center(translated_bbox), support_center)
                        > support_radius
                    ):
                        continue
                    adaptive_candidates[request["frame_index"]].append({
                        "bbox": translated_bbox,
                        "confidence": detection["confidence"],
                        "detection_source": request["source"],
                        "roi_bbox": list(request["bbox"]),
                    })

        merged_frames = []
        crop_counts = [0] * len(frames)
        for request in requests:
            crop_counts[request["frame_index"]] += 1
        for frame_index, frame in enumerate(ball_tracks):
            full_candidates = frame.get(1, {}).get(
                "raw_candidates",
                frame.get(1, {}).get("candidates", []),
            )
            merged_frames.append(_merge_ball_candidates(
                full_candidates,
                adaptive_candidates[frame_index],
            ))

        enhanced = _select_ball_track(merged_frames, player_tracks)
        for frame_index, frame in enumerate(enhanced):
            info = frame[1]
            original_info = ball_tracks[frame_index].get(1, {})
            info["rim_regions"] = [
                dict(rim) for rim in original_info.get("rim_regions", [])
            ]
            info["adaptive_second_pass_completed"] = True
            info["adaptive_crop_count"] = crop_counts[frame_index]
            info["adaptive_candidates_added"] = len(adaptive_candidates[frame_index])
            if self.detector_backend == "hybrid":
                semantic_candidates = original_info.get(
                    "semantic_raw_candidates",
                    [
                        candidate
                        for candidate in original_info.get("raw_candidates", [])
                        if not _is_wasb_candidate(candidate)
                    ],
                )
                info["semantic_raw_candidates"] = _merge_ball_candidates(
                    semantic_candidates,
                    adaptive_candidates[frame_index],
                )

        if cache_path:
            save_cache(cache_path, enhanced)
        return enhanced

    @staticmethod
    def build_semantic_tracks(
        ball_positions,
        player_tracks=None,
        *,
        fused_tracks=None,
        discontinuity_frames=None,
        rescue_max_distance=35.0,
        minimum_wasb_confirmation_frames=3,
    ):
        """Build a YOLO-anchored track for possession and event decisions.

        The fused track remains the geometric/display result. WASB may replace
        a bounded YOLO interpolation when it agrees with that interpolation.
        A rescue stays non-confirmable unless multiple consecutive frames also
        carry consistent hand evidence for the same player.
        """
        if player_tracks is None:
            player_tracks = [{} for _ in ball_positions]
        if len(player_tracks) != len(ball_positions):
            raise ValueError("Player and semantic ball tracks must align")
        if fused_tracks is not None and len(fused_tracks) != len(ball_positions):
            raise ValueError("Fused and semantic ball tracks must align")

        semantic_positions = []
        for frame in ball_positions:
            info = frame.get(1, {})
            candidates = info.get("semantic_raw_candidates")
            if candidates is None:
                candidates = [
                    candidate
                    for candidate in info.get(
                        "raw_candidates",
                        info.get("candidates", []),
                    )
                    if not _is_wasb_candidate(candidate)
                ]
            semantic_positions.append({1: {
                "raw_candidates": [dict(candidate) for candidate in candidates],
                "candidates": [dict(candidate) for candidate in candidates],
                "rim_regions": [dict(rim) for rim in info.get("rim_regions", [])],
                "semantic_track": True,
            }})

        filtered = BallTracker.remove_wrong_detections(
            None,
            semantic_positions,
            player_tracks=player_tracks,
            discontinuity_frames=discontinuity_frames,
        )
        semantic_tracks = BallTracker.interpolate_positions(
            None,
            filtered,
            discontinuity_frames=discontinuity_frames,
        )
        # Filtering and interpolation rebuild frame dictionaries around the
        # selected ball. Keep scene context that event classifiers need; it
        # must never affect candidate selection itself.
        for frame_index, semantic_frame in enumerate(semantic_tracks):
            semantic_frame.setdefault(1, {})["rim_regions"] = [
                dict(rim)
                for rim in semantic_positions[frame_index]
                .get(1, {})
                .get("rim_regions", [])
            ]
        if fused_tracks is None:
            return semantic_tracks

        for frame_index, (semantic_frame, fused_frame) in enumerate(
            zip(semantic_tracks, fused_tracks)
        ):
            semantic_info = semantic_frame.get(1, {})
            fused_info = fused_frame.get(1, {})
            rescue_distance = (
                math_dist(
                    _bbox_center(semantic_info["bbox"]),
                    _bbox_center(fused_info["bbox"]),
                )
                if semantic_info.get("bbox") and fused_info.get("bbox")
                else None
            )
            if (
                (
                    semantic_info.get("bbox")
                    and not semantic_info.get("interpolated", False)
                )
                or fused_info.get("interpolated", False)
                or not _is_wasb_candidate(fused_info)
                or not fused_info.get("bbox")
                or (
                    rescue_distance is not None
                    and rescue_distance > rescue_max_distance
                )
            ):
                continue
            hand_pose = _nearest_hand_pose_evidence(
                fused_info["bbox"],
                player_tracks[frame_index],
            )
            if not semantic_info.get("bbox") and not hand_pose.get("supported", False):
                continue
            semantic_info.update({
                "bbox": list(fused_info["bbox"]),
                "confidence": None,
                "interpolated": True,
                "position_source": "wasb_guarded_rescue",
                "detection_source": fused_info.get(
                    "detection_source",
                    "wasb_temporal",
                ),
                "semantic_confirmable": False,
                "wasb_rescue_distance_px": rescue_distance,
                "wasb_rescue_confidence": fused_info.get("confidence"),
                "hand_pose_available": hand_pose.get("available", False),
                "hand_pose_supported": hand_pose.get("supported", False),
                "hand_pose_distance": hand_pose.get("normalized_distance"),
                "hand_pose_player_id": hand_pose.get("player_id"),
            })

        supported_streak = []
        supported_player_id = None
        for frame_index, semantic_frame in enumerate(semantic_tracks):
            info = semantic_frame.get(1, {})
            hand_player_id = info.get("hand_pose_player_id")
            hand_supported = bool(info.get("hand_pose_supported", False))
            if not info.get("bbox") or not hand_supported or hand_player_id is None:
                supported_streak = []
                supported_player_id = None
                continue
            transition_is_consistent = (
                not supported_streak
                or (
                    supported_streak[-1] == frame_index - 1
                    and supported_player_id == hand_player_id
                    and _hand_track_transition_is_consistent(
                        semantic_tracks[supported_streak[-1]].get(1, {}),
                        info,
                        player_tracks[frame_index].get(hand_player_id, {}),
                    )
                )
            )
            if not transition_is_consistent:
                supported_streak = []
            if not supported_streak:
                supported_player_id = hand_player_id
            supported_streak.append(frame_index)
            if len(supported_streak) < minimum_wasb_confirmation_frames:
                continue
            for supported_frame_index in supported_streak:
                supported_info = semantic_tracks[supported_frame_index].get(1, {})
                if supported_info.get("position_source") != "wasb_guarded_rescue":
                    continue
                wasb_confidence = supported_info.get("wasb_rescue_confidence")
                calibrated_confidence = (
                    max(0.45, min(0.75, float(wasb_confidence) * 0.75))
                    if wasb_confidence is not None
                    else 0.45
                )
                supported_info.update({
                    "confidence": calibrated_confidence,
                    "interpolated": False,
                    "position_source": "wasb_hand_confirmed",
                    "semantic_confirmable": True,
                    "wasb_confirmation_frames": len(supported_streak),
                })
        return semantic_tracks

    def remove_wrong_detections(
        self,
        ball_positions,
        player_tracks=None,
        discontinuity_frames=None,
    ):
        discontinuities = _normalized_discontinuities(
            discontinuity_frames,
            len(ball_positions),
        )
        candidate_frames = [
            frame.get(1, {}).get(
                "raw_candidates",
                frame.get(1, {}).get("candidates"),
            )
            for frame in ball_positions
        ]
        if any(candidates is not None for candidates in candidate_frames):
            candidate_frames = [
                [
                    _candidate_record(candidate, candidate_index)
                    for candidate_index, candidate in enumerate(candidates or [])
                    if _valid_detection(
                        candidate.get("bbox"),
                        candidate.get("confidence"),
                    )
                ]
                for candidates in candidate_frames
            ]
            raw_candidate_frames = [
                [dict(candidate) for candidate in candidates]
                for candidates in candidate_frames
            ]
            aligned_player_tracks = (
                player_tracks
                if player_tracks is not None
                else [{} for _ in candidate_frames]
            )
            pruned_candidate_frames = _prune_sustained_body_locked_candidates(
                candidate_frames,
                aligned_player_tracks,
            )
            body_locked_removed = [
                len(before) - len(after)
                for before, after in zip(candidate_frames, pruned_candidate_frames)
            ]
            rejection_provenance = _candidate_rejection_provenance(
                candidate_frames,
                pruned_candidate_frames,
                "sustained_upper_body_lock",
            )
            selected = _select_ball_track(
                pruned_candidate_frames,
                aligned_player_tracks,
                discontinuity_frames=discontinuities,
            )
            pose_input_candidate_frames = pruned_candidate_frames
            (
                pruned_candidate_frames,
                pose_locked_removed,
            ) = _remove_selected_pose_locked_candidates(
                selected,
                pruned_candidate_frames,
                aligned_player_tracks,
            )
            if any(pose_locked_removed):
                selected = _select_ball_track(
                    pruned_candidate_frames,
                    aligned_player_tracks,
                    discontinuity_frames=discontinuities,
                )
            _extend_candidate_rejection_provenance(
                rejection_provenance,
                _candidate_rejection_provenance(
                    pose_input_candidate_frames,
                    pruned_candidate_frames,
                    "selected_lower_body_pose_lock",
                ),
            )
            (
                quarantine_candidate_frames,
                takeover_chain_removed,
                takeover_rejections,
            ) = _quarantine_persistent_takeover_candidate_chains(
                selected,
                pruned_candidate_frames,
                aligned_player_tracks,
                discontinuity_frames=discontinuities,
            )
            _extend_candidate_rejection_provenance(
                rejection_provenance,
                takeover_rejections,
            )
            if any(takeover_chain_removed):
                pruned_candidate_frames = quarantine_candidate_frames
                selected = _select_ball_track(
                    pruned_candidate_frames,
                    aligned_player_tracks,
                    discontinuity_frames=discontinuities,
                )
            for frame_index, frame in enumerate(selected):
                info = frame.get(1)
                if info is None:
                    continue
                info["raw_candidates"] = [
                    dict(candidate)
                    for candidate in raw_candidate_frames[frame_index]
                ]
                info["raw_candidate_count"] = len(raw_candidate_frames[frame_index])
                info["body_locked_candidates_removed"] = body_locked_removed[
                    frame_index
                ]
                info["pose_locked_candidates_removed"] = pose_locked_removed[
                    frame_index
                ]
                info["takeover_chain_candidates_removed"] = (
                    takeover_chain_removed[frame_index]
                )
                info["candidate_rejections"] = rejection_provenance[frame_index]
            return _reject_uncertain_observations(
                selected,
                pruned_candidate_frames,
                aligned_player_tracks,
            )

        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for index in range(len(ball_positions)):
            current_box = ball_positions[index].get(1, {}).get("bbox", [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                last_good_frame_index = index
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get(
                "bbox", []
            )
            frame_gap = index - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            camera_motion = _median_player_motion(
                player_tracks,
                last_good_frame_index,
                index,
            )
            if (
                _camera_adjusted_top_left_distance(
                    last_good_box,
                    current_box,
                    camera_motion,
                )
                > adjusted_max_distance
            ):
                ball_positions[index] = {}
            else:
                last_good_frame_index = index

        return ball_positions

    def interpolate_positions(
        self,
        ball_positions,
        max_gap_frames=6,
        discontinuity_frames=None,
    ):
        """Fill only short gaps bounded by two observed ball detections.

        Long gaps and gaps at either edge remain unknown.  This prevents a
        stale or incorrect observation from being extended through an
        arbitrary portion of a clip and keeps synthetic positions explicit.
        """
        if max_gap_frames < 0:
            raise ValueError("Maximum interpolation gap must be non-negative")
        discontinuities = _normalized_discontinuities(
            discontinuity_frames,
            len(ball_positions),
        )

        result = []
        observed_indices = []
        for index, frame in enumerate(ball_positions):
            info = frame.get(1, {})
            bbox = info.get("bbox")
            if bbox:
                observed = dict(info)
                observed.update({
                    "bbox": list(bbox),
                    "interpolated": False,
                    "position_source": "observed",
                    "uncertainty_frames": 0,
                })
                result.append({1: observed})
                observed_indices.append(index)
            else:
                result.append({1: dict(info)} if info else {})

        for start_index, end_index in zip(observed_indices, observed_indices[1:]):
            gap_frames = end_index - start_index - 1
            if gap_frames <= 0 or gap_frames > max_gap_frames:
                continue
            if any(start_index < frame <= end_index for frame in discontinuities):
                continue
            if any(
                rejection.get("reason")
                == "persistent_competing_takeover_chain"
                for frame in result[start_index + 1 : end_index]
                for rejection in frame.get(1, {}).get("candidate_rejections", [])
            ):
                continue

            start_bbox = result[start_index][1]["bbox"]
            end_bbox = result[end_index][1]["bbox"]
            start_segment = result[start_index][1].get("track_segment_id")
            end_segment = result[end_index][1].get("track_segment_id")
            if (
                start_segment is not None
                and end_segment is not None
                and start_segment != end_segment
            ):
                continue
            for offset in range(1, gap_frames + 1):
                fraction = offset / (gap_frames + 1)
                bbox = [
                    start + fraction * (end - start)
                    for start, end in zip(start_bbox, end_bbox)
                ]
                interpolated = dict(result[start_index + offset].get(1, {}))
                interpolated.update({
                    "bbox": bbox,
                    "confidence": None,
                    "interpolated": True,
                    "position_source": "interpolated",
                    "interpolation_gap_frames": gap_frames,
                    "frames_since_observed": offset,
                    "frames_until_observed": gap_frames + 1 - offset,
                    "uncertainty_frames": min(offset, gap_frames + 1 - offset),
                    "track_segment_id": start_segment,
                })
                result[start_index + offset] = {1: interpolated}

        return result


def _named_detections(result, class_name):
    class_names = getattr(result, "names", {})
    aliases = (
        BALL_CLASS_ALIASES
        if class_name.lower() in BALL_CLASS_ALIASES
        else HOOP_CLASS_ALIASES
        if class_name.lower() in HOOP_CLASS_ALIASES
        else {class_name.lower()}
    )
    target_id = next(
        (
            class_id
            for class_id, name in class_names.items()
            if str(name).strip().lower() in aliases
        ),
        None,
    )
    if target_id is None:
        return []
    detections = sv.Detections.from_ultralytics(result)
    output = []
    for detection in detections:
        if int(detection[3]) != int(target_id):
            continue
        bbox = detection[0].tolist() if hasattr(detection[0], "tolist") else list(detection[0])
        output.append({
            "bbox": list(map(float, bbox)),
            "confidence": float(detection[2]),
        })
    return output


def _candidate_record(candidate, fallback_index):
    record = {
        "bbox": list(candidate["bbox"]),
        "confidence": float(candidate["confidence"]),
        "candidate_index": int(candidate.get("candidate_index", fallback_index)),
    }
    for field in ("detection_source", "roi_bbox"):
        if candidate.get(field) is not None:
            record[field] = (
                list(candidate[field]) if field == "roi_bbox" else candidate[field]
            )
    return record


def _adaptive_crop_requests(frames, ball_tracks, player_tracks):
    requests = []
    for frame_index, (frame, ball_frame, players) in enumerate(
        zip(frames, ball_tracks, player_tracks)
    ):
        info = ball_frame.get(1, {})
        confidence = info.get("confidence")
        if confidence is not None and confidence >= BALL_ADAPTIVE_TRIGGER_CONFIDENCE:
            continue
        frame_height, frame_width = frame.shape[:2]
        focus = _predicted_ball_center(ball_tracks, frame_index)
        if focus is None and info.get("bbox"):
            focus = _bbox_center(info["bbox"])

        proposed = []
        if focus is not None:
            proposed.append({
                "bbox": _square_crop_bbox(focus, 224, frame_width, frame_height),
                "source": "adaptive_predicted",
                "priority": 0,
                "support_center": tuple(focus),
                "support_radius": 80.0,
            })

        wrist_regions = []
        for player in players.values():
            player_bbox = player.get("bbox")
            pose = player.get("pose") or {}
            points = pose.get("keypoints_xy") or []
            confidences = pose.get("keypoint_confidences") or []
            if not player_bbox or len(points) <= 10 or len(confidences) <= 10:
                continue
            player_height = float(player_bbox[3]) - float(player_bbox[1])
            crop_size = max(96, min(224, int(round(0.55 * player_height))))
            for wrist_index in (9, 10):
                point = _confident_pose_point(points, confidences, wrist_index, 0.35)
                if point is None:
                    continue
                distance = (
                    math_dist(point, focus)
                    if focus is not None
                    else -player_height
                )
                wrist_regions.append((
                    distance,
                    {
                        "bbox": _square_crop_bbox(
                            point, crop_size, frame_width, frame_height
                        ),
                        "source": "adaptive_player_hand",
                        "priority": 1,
                        "support_center": tuple(map(float, point)),
                        "support_radius": 0.38 * crop_size,
                    },
                ))
        proposed.extend(region for _, region in sorted(wrist_regions, key=lambda item: item[0])[:2])

        rim_regions = []
        for rim in info.get("rim_regions", []):
            bbox = rim.get("bbox")
            if not bbox:
                continue
            center = _bbox_center(bbox)
            size = max(160, min(288, int(round(5 * max(
                float(bbox[2]) - float(bbox[0]),
                float(bbox[3]) - float(bbox[1]),
            )))))
            distance = math_dist(center, focus) if focus is not None else 0.0
            rim_regions.append((
                distance,
                {
                    "bbox": _square_crop_bbox(center, size, frame_width, frame_height),
                    "source": "adaptive_rim",
                    "priority": 2,
                    "support_center": tuple(center),
                    "support_radius": 0.42 * size,
                },
            ))
        proposed.extend(region for _, region in sorted(rim_regions, key=lambda item: item[0])[:2])

        accepted = []
        for request in sorted(proposed, key=lambda item: item["priority"]):
            if any(_bbox_iou(request["bbox"], previous["bbox"]) >= 0.70 for previous in accepted):
                continue
            accepted.append(request)
            if len(accepted) >= BALL_ADAPTIVE_MAX_CROPS_PER_FRAME:
                break
        for request in accepted:
            requests.append({
                "frame_index": frame_index,
                "bbox": request["bbox"],
                "source": request["source"],
                "support_center": request.get("support_center"),
                "support_radius": request.get("support_radius"),
            })
    return requests


def _predicted_ball_center(ball_tracks, frame_index, maximum_history=8):
    observations = []
    for previous_index in range(frame_index - 1, max(-1, frame_index - maximum_history - 1), -1):
        bbox = ball_tracks[previous_index].get(1, {}).get("bbox")
        if bbox:
            observations.append((previous_index, np.asarray(_bbox_center(bbox), dtype=float)))
            if len(observations) == 2:
                break
    if not observations:
        return None
    last_index, last_center = observations[0]
    if len(observations) == 1:
        return tuple(last_center.tolist())
    earlier_index, earlier_center = observations[1]
    velocity = (last_center - earlier_center) / max(1, last_index - earlier_index)
    prediction = last_center + velocity * (frame_index - last_index)
    return tuple(prediction.tolist())


def _square_crop_bbox(center, size, frame_width, frame_height):
    size = max(1, min(max(32, int(size)), frame_width, frame_height))
    center_x, center_y = map(float, center)
    x1 = int(round(center_x - size / 2))
    y1 = int(round(center_y - size / 2))
    x1 = max(0, min(x1, frame_width - size))
    y1 = max(0, min(y1, frame_height - size))
    return [x1, y1, x1 + size, y1 + size]


def _bbox_iou(first, second):
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def math_dist(first, second):
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))


def _is_wasb_candidate(candidate):
    return str(candidate.get("detection_source", "")).startswith("wasb")


def _merge_ball_candidates(full_candidates, adaptive_candidates, maximum_candidates=12):
    merged = []
    candidates = [
        dict(candidate)
        for candidate in [*(full_candidates or []), *(adaptive_candidates or [])]
        if _valid_detection(candidate.get("bbox"), candidate.get("confidence"))
    ]
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        candidate_center = _bbox_center(candidate["bbox"])
        duplicate = any(
            _bbox_iou(candidate["bbox"], kept["bbox"]) >= 0.40
            or math_dist(candidate_center, _bbox_center(kept["bbox"]))
            <= max(4.0, 0.5 * max(
                kept["bbox"][2] - kept["bbox"][0],
                kept["bbox"][3] - kept["bbox"][1],
            ))
            for kept in merged
        )
        if duplicate:
            continue
        merged.append(candidate)
        if len(merged) >= maximum_candidates:
            break
    return merged


def _select_ball_track(
    detection_frames,
    player_tracks=None,
    *,
    beam_size=40,
    max_observation_gap=8,
    discontinuity_frames=None,
):
    """Select a temporally coherent ball path from per-frame candidates.

    The beam keeps missing observations as an explicit option. Candidate
    confidence and player proximity provide local evidence, while
    camera-adjusted speed and acceleration discourage wrong-object jumps.
    Unsupported stationary detections are penalized because they are a common
    broadcast-overlay and crowd false-positive pattern; coherent airborne
    motion can still bridge between player-supported observations.
    """
    if player_tracks is None:
        player_tracks = [{} for _ in detection_frames]
    if len(player_tracks) != len(detection_frames):
        raise ValueError("Player tracks and ball candidate frames must align")
    if beam_size < 1 or max_observation_gap < 0:
        raise ValueError("Beam size must be positive and observation gap non-negative")
    discontinuities = _normalized_discontinuities(
        discontinuity_frames,
        len(detection_frames),
    )

    # score, last bbox, last frame, velocity, selected candidate indices
    beams = [(0.0, None, None, None, [])]
    for frame_index, detections in enumerate(detection_frames):
        next_beams = []
        for score, last_bbox, last_frame, velocity, path in beams:
            if frame_index in discontinuities:
                last_bbox, last_frame, velocity = None, None, None
            if (
                last_frame is not None
                and frame_index - last_frame > max_observation_gap
            ):
                last_bbox, last_frame, velocity = None, None, None

            # Unknown is preferable to forcing a weak or discontinuous target.
            next_beams.append(
                (score - 0.03, last_bbox, last_frame, velocity, path + [None])
            )

            for candidate_index, detection in enumerate(detections):
                bbox = detection.get("bbox")
                confidence = detection.get("confidence")
                if not _valid_detection(bbox, confidence):
                    continue

                player_distance = _minimum_bbox_distance(
                    bbox,
                    player_tracks[frame_index],
                )
                support = max(0.0, 1.0 - player_distance / 100.0)
                confidence_quality = max(
                    0.0,
                    min(1.0, (float(confidence) - 0.25) / 0.75),
                )
                detection_score = (
                    0.70 * support
                    + 0.80 * confidence_quality
                    - 0.55
                    - (0.40 if player_distance > 150.0 else 0.0)
                )

                transition_score = -0.08
                new_velocity = None
                if last_bbox is not None:
                    gap = frame_index - last_frame
                    camera_motion = np.asarray(
                        _median_player_motion(
                            player_tracks,
                            last_frame,
                            frame_index,
                        ),
                        dtype=float,
                    )
                    displacement = (
                        np.asarray(_bbox_center(bbox), dtype=float)
                        - np.asarray(_bbox_center(last_bbox), dtype=float)
                        - camera_motion
                    )
                    new_velocity = displacement / gap
                    speed = float(np.linalg.norm(new_velocity))
                    transition_score = 0.15 - min(speed / 120.0, 0.8)
                    if velocity is not None:
                        acceleration = float(
                            np.linalg.norm(new_velocity - velocity)
                        )
                        transition_score -= min(acceleration / 120.0, 0.8)
                    if player_distance > 100.0 and speed < 2.0:
                        transition_score -= 0.35
                    if speed > 90.0:
                        transition_score -= 1.5 + (speed - 90.0) / 100.0

                next_beams.append((
                    score + detection_score + transition_score,
                    bbox,
                    frame_index,
                    new_velocity,
                    path + [candidate_index],
                ))

        beams = sorted(next_beams, key=lambda item: item[0], reverse=True)[:beam_size]

    selected_path = beams[0][4] if beams else [None] * len(detection_frames)
    tracks = []
    track_segment_id = -1
    previous_selected_frame = None
    for frame_index, candidate_index in enumerate(selected_path):
        candidates = [
            _candidate_record(candidate, current_candidate_index)
            for current_candidate_index, candidate in enumerate(
                detection_frames[frame_index]
            )
            if _valid_detection(candidate.get("bbox"), candidate.get("confidence"))
        ]
        info = {
            "candidates": candidates,
            "raw_candidates": [dict(candidate) for candidate in candidates],
            "candidate_count": len(candidates),
            "raw_candidate_count": len(candidates),
            "sequence_selected": True,
        }
        if candidate_index is not None:
            tracking_discontinuity = (
                previous_selected_frame is not None
                and (
                    frame_index - previous_selected_frame > max_observation_gap
                    or any(
                        previous_selected_frame < cut <= frame_index
                        for cut in discontinuities
                    )
                )
            )
            if previous_selected_frame is None or tracking_discontinuity:
                track_segment_id += 1
            previous_selected_frame = frame_index
            chosen = detection_frames[frame_index][candidate_index]
            info.update({
                "bbox": list(chosen["bbox"]),
                "confidence": float(chosen["confidence"]),
                "detection_source": chosen.get("detection_source", "full_frame"),
                "selected_candidate_index": int(
                    chosen.get("candidate_index", candidate_index)
                ),
                "track_segment_id": track_segment_id,
                "tracking_discontinuity": tracking_discontinuity,
                "player_distance": _minimum_bbox_distance(
                    chosen["bbox"],
                    player_tracks[frame_index],
                ),
            })
            hand_pose = _nearest_hand_pose_evidence(
                chosen["bbox"],
                player_tracks[frame_index],
            )
            info.update({
                "hand_pose_available": hand_pose.get("available", False),
                "hand_pose_supported": hand_pose.get("supported", False),
                "hand_pose_distance": hand_pose.get("normalized_distance"),
                "hand_pose_player_id": hand_pose.get("player_id"),
            })
        tracks.append({1: info})
    return tracks


def _valid_detection(bbox, confidence):
    return bool(
        bbox
        and len(bbox) == 4
        and confidence is not None
        and np.isfinite(np.asarray([*bbox, confidence], dtype=float)).all()
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def _normalized_discontinuities(discontinuity_frames, frame_count):
    return {
        int(frame)
        for frame in (discontinuity_frames or [])
        if 0 < int(frame) < frame_count
    }


def _candidate_key(candidate):
    candidate_index = candidate.get("candidate_index")
    if candidate_index is not None:
        return "index", int(candidate_index)
    return (
        "observation",
        tuple(candidate.get("bbox") or ()),
        candidate.get("confidence"),
    )


def _candidate_rejection_provenance(before_frames, after_frames, reason):
    provenance = [[] for _ in before_frames]
    for frame_index, (before, after) in enumerate(zip(before_frames, after_frames)):
        kept = {_candidate_key(candidate) for candidate in after}
        for candidate in before:
            if _candidate_key(candidate) in kept:
                continue
            provenance[frame_index].append({
                "candidate_index": candidate.get("candidate_index"),
                "bbox": list(candidate.get("bbox") or []),
                "confidence": float(candidate.get("confidence", 0.0)),
                "reason": reason,
            })
    return provenance


def _extend_candidate_rejection_provenance(target, additions):
    for target_frame, addition_frame in zip(target, additions):
        known = {
            (_candidate_key(record), record.get("reason"))
            for record in target_frame
        }
        for record in addition_frame:
            key = (_candidate_key(record), record.get("reason"))
            if key not in known:
                target_frame.append(record)
                known.add(key)


def _quarantine_persistent_takeover_candidate_chains(
    selected_tracks,
    detection_frames,
    player_tracks,
    *,
    discontinuity_frames=None,
    minimum_chain_observations=6,
    minimum_coexisting_observations=2,
    minimum_takeover_observations=3,
    maximum_chain_gap_frames=18,
    maximum_relative_drift=0.18,
    maximum_median_relative_speed=0.012,
    minimum_nonconvergence_distance=0.45,
    maximum_prediction_error=0.45,
    minimum_velocity_discontinuity=0.10,
    maximum_confidence_margin=0.45,
    takeover_confirmation_window=12,
    source_history_gap=8,
):
    """Quarantine a persistent player-locked object that later takes over.

    The chain must first be independently visible beside a different selected
    ball, remain stable relative to one player, disappear into an observation
    gap, and then become the selected path without spatial convergence or
    camera-adjusted kinematic continuity. Hand proximity describes the lock
    but does not excuse a candidate that was already a separate object.
    """
    if not (
        len(selected_tracks) == len(detection_frames) == len(player_tracks)
    ):
        raise ValueError("Selected, candidate, and player frames must align")
    discontinuities = _normalized_discontinuities(
        discontinuity_frames,
        len(detection_frames),
    )

    chains = []
    for frame_index, candidates in enumerate(detection_frames):
        entries = []
        for candidate_index, candidate in enumerate(candidates):
            relation = _candidate_player_lock_relation(
                candidate.get("bbox"),
                player_tracks[frame_index],
            )
            if relation is None:
                continue
            entries.append({
                "frame": frame_index,
                "candidate": candidate_index,
                "candidate_index": int(
                    candidate.get("candidate_index", candidate_index)
                ),
                "bbox": list(candidate["bbox"]),
                "confidence": float(candidate["confidence"]),
                "center": np.asarray(_bbox_center(candidate["bbox"]), dtype=float),
                "player_id": relation["player_id"],
                "relative": np.asarray(relation["relative"], dtype=float),
                "scale": relation["player_height"],
                "hand_pose_supported": relation["hand_pose_supported"],
            })

        used_chains = set()
        for entry in entries:
            best = None
            for chain_index, chain in enumerate(chains):
                if chain_index in used_chains:
                    continue
                if chain["player_id"] != entry["player_id"]:
                    continue
                gap = frame_index - chain["last_frame"]
                if gap <= 0 or gap > maximum_chain_gap_frames:
                    continue
                if any(chain["last_frame"] < cut <= frame_index for cut in discontinuities):
                    continue
                drift = float(
                    np.linalg.norm(entry["relative"] - chain["last_relative"])
                )
                if drift > maximum_relative_drift:
                    continue
                match = (drift / gap, -len(chain["entries"]), chain_index)
                if best is None or match < best:
                    best = match
            if best is None:
                chains.append({
                    "chain_id": len(chains),
                    "player_id": entry["player_id"],
                    "last_frame": frame_index,
                    "last_relative": entry["relative"],
                    "entries": [entry],
                })
                used_chains.add(len(chains) - 1)
                continue
            chain_index = best[2]
            chain = chains[chain_index]
            chain["last_frame"] = frame_index
            chain["last_relative"] = entry["relative"]
            chain["entries"].append(entry)
            used_chains.add(chain_index)

    rejected = set()
    rejection_evidence = {}
    for chain in chains:
        entries = chain["entries"]
        if len(entries) < minimum_chain_observations:
            continue
        relative_speeds = [
            float(np.linalg.norm(current["relative"] - previous["relative"]))
            / (current["frame"] - previous["frame"])
            for previous, current in zip(entries, entries[1:])
            if current["frame"] > previous["frame"]
        ]
        if (
            not relative_speeds
            or float(np.median(relative_speeds)) > maximum_median_relative_speed
        ):
            continue

        selected_entry_keys = {
            (
                entry["frame"],
                entry["candidate_index"],
            )
            for entry in entries
            if _selected_entry_matches(entry, selected_tracks[entry["frame"]])
        }
        if len(selected_entry_keys) < minimum_takeover_observations:
            continue

        accepted_evidence = None
        for takeover in entries:
            takeover_key = (takeover["frame"], takeover["candidate_index"])
            if takeover_key not in selected_entry_keys:
                continue
            confirmed_takeovers = sum(
                (entry["frame"], entry["candidate_index"]) in selected_entry_keys
                and takeover["frame"] <= entry["frame"] <= (
                    takeover["frame"] + takeover_confirmation_window
                )
                for entry in entries
            )
            if confirmed_takeovers < minimum_takeover_observations:
                continue

            coexistence = []
            for entry in entries:
                if entry["frame"] >= takeover["frame"]:
                    break
                selected_info = selected_tracks[entry["frame"]].get(1, {})
                selected_bbox = selected_info.get("bbox")
                if not selected_bbox or _selected_entry_matches(
                    entry,
                    selected_tracks[entry["frame"]],
                ):
                    continue
                selected_confidence = float(selected_info.get("confidence", 0.0))
                if entry["confidence"] < selected_confidence - maximum_confidence_margin:
                    continue
                separation = (
                    _bbox_center_distance(entry["bbox"], selected_bbox)
                    / max(1.0, entry["scale"])
                )
                coexistence.append((entry, separation))
            if len(coexistence) < minimum_coexisting_observations:
                continue
            if min(separation for _, separation in coexistence) < (
                minimum_nonconvergence_distance
            ):
                continue

            source_frames = []
            chain_entries_by_frame = {}
            for entry in entries:
                chain_entries_by_frame.setdefault(entry["frame"], []).append(entry)
            for source_frame in range(takeover["frame"] - 1, -1, -1):
                if source_frame in discontinuities:
                    break
                source_info = selected_tracks[source_frame].get(1, {})
                if not source_info.get("bbox"):
                    continue
                if any(
                    _selected_entry_matches(entry, selected_tracks[source_frame])
                    for entry in chain_entries_by_frame.get(source_frame, [])
                ):
                    continue
                source_frames.append(source_frame)
                if len(source_frames) == 2:
                    break
            if len(source_frames) < 2:
                continue
            latest_source, earlier_source = source_frames
            if takeover["frame"] - latest_source <= 1:
                continue
            if latest_source - earlier_source > source_history_gap:
                continue
            prediction_error, velocity_discontinuity = (
                _camera_adjusted_takeover_discontinuity(
                    selected_tracks,
                    player_tracks,
                    earlier_source,
                    latest_source,
                    takeover,
                )
            )
            if (
                prediction_error <= maximum_prediction_error
                or velocity_discontinuity <= minimum_velocity_discontinuity
            ):
                continue
            accepted_evidence = {
                "coexisting_observations": len(coexistence),
                "takeover_observations": confirmed_takeovers,
                "minimum_separation_player_heights": round(
                    min(separation for _, separation in coexistence),
                    4,
                ),
                "prediction_error_player_heights": round(prediction_error, 4),
                "velocity_discontinuity_player_heights_per_frame": round(
                    velocity_discontinuity,
                    4,
                ),
                "median_relative_speed_player_heights_per_frame": round(
                    float(np.median(relative_speeds)),
                    4,
                ),
                "observation_gap_frames": takeover["frame"] - latest_source - 1,
                "hand_pose_supported_observations": sum(
                    entry["hand_pose_supported"] for entry in entries
                ),
            }
            break
        if accepted_evidence is None:
            continue
        chain_label = f"takeover_chain_{chain['chain_id']}"
        for entry in entries:
            key = (entry["frame"], entry["candidate"])
            rejected.add(key)
            rejection_evidence[key] = {
                "candidate_index": entry["candidate_index"],
                "bbox": list(entry["bbox"]),
                "confidence": entry["confidence"],
                "reason": "persistent_competing_takeover_chain",
                "candidate_chain_id": chain_label,
                "chain_evidence": dict(accepted_evidence),
            }

    removed_per_frame = [0] * len(detection_frames)
    provenance = [[] for _ in detection_frames]
    filtered = []
    for frame_index, candidates in enumerate(detection_frames):
        kept = []
        for candidate_index, candidate in enumerate(candidates):
            key = (frame_index, candidate_index)
            if key not in rejected:
                kept.append(candidate)
                continue
            removed_per_frame[frame_index] += 1
            provenance[frame_index].append(rejection_evidence[key])
        filtered.append(kept)
    return filtered, removed_per_frame, provenance


def _candidate_player_lock_relation(ball_bbox, players, maximum_outside_ratio=0.05):
    if not ball_bbox or not players:
        return None
    center_x, center_y = _bbox_center(ball_bbox)
    matches = []
    for player_id, player in players.items():
        bbox = player.get("bbox")
        if not bbox:
            continue
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        if width <= 0 or height <= 0:
            continue
        relative_x = (center_x - float(bbox[0])) / width
        relative_y = (center_y - float(bbox[1])) / height
        if not (
            -maximum_outside_ratio <= relative_x <= 1.0 + maximum_outside_ratio
            and -maximum_outside_ratio <= relative_y <= 1.0 + maximum_outside_ratio
        ):
            continue
        hand_pose = _nearest_hand_pose_evidence(
            ball_bbox,
            players,
            player_id=player_id,
        )
        outside = max(
            0.0,
            -relative_x,
            relative_x - 1.0,
            -relative_y,
            relative_y - 1.0,
        )
        matches.append((
            outside,
            width * height,
            int(player_id),
            relative_x,
            relative_y,
            height,
            bool(hand_pose.get("supported", False)),
        ))
    if not matches:
        return None
    _, _, player_id, relative_x, relative_y, height, hand_supported = min(matches)
    return {
        "player_id": player_id,
        "relative": (relative_x, relative_y),
        "player_height": height,
        "hand_pose_supported": hand_supported,
    }


def _selected_entry_matches(entry, selected_frame):
    info = selected_frame.get(1, {})
    selected_index = info.get("selected_candidate_index")
    if selected_index is not None:
        return int(selected_index) == entry["candidate_index"]
    return info.get("bbox") == entry["bbox"]


def _camera_adjusted_takeover_discontinuity(
    selected_tracks,
    player_tracks,
    earlier_frame,
    latest_frame,
    takeover,
):
    earlier_center = np.asarray(
        _bbox_center(selected_tracks[earlier_frame][1]["bbox"]),
        dtype=float,
    )
    latest_center = np.asarray(
        _bbox_center(selected_tracks[latest_frame][1]["bbox"]),
        dtype=float,
    )
    takeover_center = takeover["center"]
    history_gap = latest_frame - earlier_frame
    takeover_gap = takeover["frame"] - latest_frame
    history_camera_motion = np.asarray(
        _median_player_motion(player_tracks, earlier_frame, latest_frame),
        dtype=float,
    )
    takeover_camera_motion = np.asarray(
        _median_player_motion(player_tracks, latest_frame, takeover["frame"]),
        dtype=float,
    )
    history_velocity = (
        latest_center - earlier_center - history_camera_motion
    ) / history_gap
    takeover_velocity = (
        takeover_center - latest_center - takeover_camera_motion
    ) / takeover_gap
    predicted_center = (
        latest_center
        + takeover_camera_motion
        + history_velocity * takeover_gap
    )
    scale = max(1.0, takeover["scale"])
    return (
        float(np.linalg.norm(takeover_center - predicted_center)) / scale,
        float(np.linalg.norm(takeover_velocity - history_velocity)) / scale,
    )


def _reject_uncertain_observations(
    tracks,
    detection_frames,
    player_tracks,
    *,
    minimum_body_lock_frames=3,
    minimum_hand_support_frames=2,
):
    """Remove body-locked and locally ambiguous selected observations."""
    rejected = {}
    central_run = []
    central_player_id = None
    central_hand_support_count = 0

    for frame_index, frame in enumerate(tracks):
        info = frame.get(1, {})
        bbox = info.get("bbox")
        player_id = (
            _central_upper_player_id(bbox, player_tracks[frame_index])
            if bbox
            else None
        )
        hand_supported = False
        if player_id is not None:
            hand_pose = _nearest_hand_pose_evidence(
                bbox,
                player_tracks[frame_index],
                player_id=player_id,
            )
            info.update({
                "hand_pose_available": hand_pose.get("available", False),
                "hand_pose_supported": hand_pose.get("supported", False),
                "hand_pose_distance": hand_pose.get("normalized_distance"),
                "hand_pose_player_id": hand_pose.get("player_id"),
            })
            hand_supported = hand_pose.get("supported", False)
        if (
            player_id is not None
            and player_id == central_player_id
            and central_run
            and frame_index == central_run[-1] + 1
        ):
            central_run.append(frame_index)
            central_hand_support_count += int(hand_supported)
        else:
            if (
                len(central_run) >= minimum_body_lock_frames
                and central_hand_support_count < minimum_hand_support_frames
            ):
                for rejected_frame in central_run:
                    rejected.setdefault(
                        rejected_frame,
                        "sustained_central_upper_body_lock",
                    )
            central_run = [frame_index] if player_id is not None else []
            central_player_id = player_id
            central_hand_support_count = (
                int(hand_supported) if player_id is not None else 0
            )

        if not bbox or info.get("player_distance", 80.0) <= 50.0:
            continue
        selected_confidence = float(info.get("confidence", 0.0))
        for candidate in detection_frames[frame_index]:
            if candidate.get("bbox") == bbox:
                continue
            if (
                float(candidate.get("confidence", 0.0))
                >= selected_confidence - 0.15
                and _minimum_bbox_distance(
                    candidate.get("bbox"),
                    player_tracks[frame_index],
                )
                <= 50.0
            ):
                rejected.setdefault(
                    frame_index,
                    "unsupported_selection_with_supported_competitor",
                )
                break

    if (
        len(central_run) >= minimum_body_lock_frames
        and central_hand_support_count < minimum_hand_support_frames
    ):
        for rejected_frame in central_run:
            rejected.setdefault(
                rejected_frame,
                "sustained_central_upper_body_lock",
            )
    for frame_index, reason in rejected.items():
        info = tracks[frame_index].get(1, {})
        if not info or not info.get("bbox"):
            continue
        rejected_observation = {
            "bbox": list(info["bbox"]),
            "confidence": info.get("confidence"),
            "selected_candidate_index": info.get("selected_candidate_index"),
            "track_segment_id": info.get("track_segment_id"),
            "reason": reason,
        }
        info["observation_rejected"] = True
        info["observation_rejection"] = rejected_observation
        info["position_source"] = "rejected"
        info.pop("bbox", None)
        info.pop("confidence", None)
        info.pop("selected_candidate_index", None)
    return tracks


def _remove_selected_pose_locked_candidates(
    selected_tracks,
    detection_frames,
    player_tracks,
    *,
    minimum_observations=5,
    maximum_gap_frames=2,
    maximum_relative_motion=0.10,
    minimum_competing_frames=2,
):
    """Blacklist sustained lower-body selections unsupported by hand pose.

    A real dribble moves substantially relative to the player's body. Jersey,
    shoe, and floor-logo false positives instead remain nearly fixed within a
    player box. The chain is only removed when independent competing ball
    candidates exist, after which the temporal lattice gets a second chance
    to select them rather than turning the whole interval into interpolation.
    """
    if len(selected_tracks) != len(detection_frames) or len(selected_tracks) != len(
        player_tracks
    ):
        raise ValueError("Selected, candidate, and player frames must align")

    chains = []
    current = []
    for frame_index, frame in enumerate(selected_tracks):
        bbox = frame.get(1, {}).get("bbox")
        relation = _lower_body_pose_lock_relation(
            bbox,
            player_tracks[frame_index],
        )
        entry = None
        if relation is not None:
            entry = {
                "frame": frame_index,
                "bbox": bbox,
                "relative": np.asarray(
                    [relation["relative_x"], relation["relative_y"]],
                    dtype=float,
                ),
                "has_competitor": _has_free_competing_candidate(
                    bbox,
                    detection_frames[frame_index],
                    player_tracks[frame_index],
                ),
            }
        if entry is not None and current:
            gap = frame_index - current[-1]["frame"]
            relative_motion = float(
                np.linalg.norm(entry["relative"] - current[-1]["relative"])
            ) / gap
            if gap <= maximum_gap_frames and relative_motion <= maximum_relative_motion:
                current.append(entry)
                continue
        if current:
            chains.append(current)
        current = [entry] if entry is not None else []
    if current:
        chains.append(current)

    rejected = set()
    for chain in chains:
        if len(chain) < minimum_observations:
            continue
        if sum(entry["has_competitor"] for entry in chain) < minimum_competing_frames:
            continue
        rejected.update((entry["frame"], tuple(entry["bbox"])) for entry in chain)

    removed_per_frame = [0] * len(detection_frames)
    filtered = []
    for frame_index, candidates in enumerate(detection_frames):
        kept = []
        for candidate in candidates:
            key = (frame_index, tuple(candidate.get("bbox") or ()))
            if key in rejected:
                removed_per_frame[frame_index] += 1
            else:
                kept.append(candidate)
        filtered.append(kept)
    return filtered, removed_per_frame


def _lower_body_pose_lock_relation(ball_bbox, players):
    if not ball_bbox:
        return None
    center_x, center_y = _bbox_center(ball_bbox)
    containing = []
    for player_id, player in players.items():
        bbox = player.get("bbox")
        if not bbox or not (
            bbox[0] <= center_x <= bbox[2]
            and bbox[1] <= center_y <= bbox[3]
        ):
            continue
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        if width <= 0 or height <= 0:
            continue
        relative_x = (center_x - float(bbox[0])) / width
        relative_y = (center_y - float(bbox[1])) / height
        if not (0.10 <= relative_x <= 0.90 and 0.50 <= relative_y <= 0.98):
            continue
        hand_pose = _nearest_hand_pose_evidence(
            ball_bbox,
            players,
            player_id=player_id,
        )
        if not hand_pose.get("available") or hand_pose.get("supported"):
            continue
        containing.append((
            width * height,
            relative_x,
            relative_y,
            hand_pose,
        ))
    if not containing:
        return None
    _, relative_x, relative_y, hand_pose = min(containing)
    return {
        "relative_x": relative_x,
        "relative_y": relative_y,
        "hand_pose_distance": hand_pose.get("normalized_distance"),
    }


def _has_free_competing_candidate(selected_bbox, candidates, players):
    selected_confidence = next(
        (
            float(candidate.get("confidence", 0.0))
            for candidate in candidates
            if candidate.get("bbox") == selected_bbox
        ),
        0.0,
    )
    return any(
        candidate.get("bbox") != selected_bbox
        and _valid_detection(candidate.get("bbox"), candidate.get("confidence"))
        and float(candidate["confidence"]) >= selected_confidence - 0.45
        and _minimum_bbox_distance(candidate["bbox"], players) > 5.0
        for candidate in candidates
    )


def _prune_sustained_body_locked_candidates(
    detection_frames,
    player_tracks,
    *,
    minimum_observations=8,
    minimum_span_frames=20,
    maximum_gap_frames=5,
    maximum_normalized_speed=0.10,
):
    """Remove slow candidate chains locked to a player's head/upper torso.

    Candidate chains are linked by player-scale-normalized image motion, not
    player track identity. This keeps the rejection stable when overlapping
    player detections exchange IDs. Brief overhead catches and shot gathers
    remain eligible because they do not form a long, slow body-relative run.
    """
    if player_tracks is None:
        return [list(detections) for detections in detection_frames]
    if len(detection_frames) != len(player_tracks):
        raise ValueError("Player tracks and ball candidate frames must align")

    chains = []
    for frame_index, detections in enumerate(detection_frames):
        entries = []
        for candidate_index, candidate in enumerate(detections):
            bbox = candidate.get("bbox")
            relation = _upper_body_relation(bbox, player_tracks[frame_index])
            if relation is None or relation.get("hand_pose_supported"):
                continue
            entries.append({
                "frame": frame_index,
                "candidate": candidate_index,
                "center": np.asarray(_bbox_center(bbox), dtype=float),
                "scale": relation["player_height"],
            })

        used_chains = set()
        for entry in entries:
            best = None
            for chain_index, chain in enumerate(chains):
                if chain_index in used_chains:
                    continue
                gap = frame_index - chain["last_frame"]
                if gap <= 0 or gap > maximum_gap_frames:
                    continue
                scale = max(1.0, (entry["scale"] + chain["last_scale"]) / 2)
                normalized_speed = (
                    float(np.linalg.norm(entry["center"] - chain["last_center"]))
                    / gap
                    / scale
                )
                if normalized_speed > maximum_normalized_speed:
                    continue
                candidate_match = (normalized_speed, -len(chain["entries"]), chain_index)
                if best is None or candidate_match < best:
                    best = candidate_match

            if best is None:
                chains.append({
                    "last_frame": frame_index,
                    "last_center": entry["center"],
                    "last_scale": entry["scale"],
                    "entries": [entry],
                })
                used_chains.add(len(chains) - 1)
                continue

            chain_index = best[2]
            chain = chains[chain_index]
            chain["last_frame"] = frame_index
            chain["last_center"] = entry["center"]
            chain["last_scale"] = entry["scale"]
            chain["entries"].append(entry)
            used_chains.add(chain_index)

    rejected = set()
    for chain in chains:
        entries = chain["entries"]
        span = entries[-1]["frame"] - entries[0]["frame"] + 1
        if len(entries) < minimum_observations or span < minimum_span_frames:
            continue
        rejected.update(
            (entry["frame"], entry["candidate"])
            for entry in entries
        )

    return [
        [
            candidate
            for candidate_index, candidate in enumerate(detections)
            if (frame_index, candidate_index) not in rejected
        ]
        for frame_index, detections in enumerate(detection_frames)
    ]


def _upper_body_relation(ball_bbox, players):
    if not ball_bbox:
        return None
    center_x, center_y = _bbox_center(ball_bbox)
    containing = []
    for player_id, player in players.items():
        bbox = player.get("bbox")
        if not bbox or not (
            bbox[0] <= center_x <= bbox[2]
            and bbox[1] <= center_y <= bbox[3]
        ):
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= 0 or height <= 0:
            continue
        relative_x = (center_x - bbox[0]) / width
        relative_y = (center_y - bbox[1]) / height
        if 0.10 <= relative_x <= 0.90 and 0.0 <= relative_y <= 0.22:
            hand_pose = _nearest_hand_pose_evidence(
                ball_bbox,
                players,
                player_id=player_id,
            )
            containing.append((
                width * height,
                height,
                relative_x,
                relative_y,
                hand_pose,
            ))
    if not containing:
        return None
    _, height, relative_x, relative_y, hand_pose = min(
        containing,
        key=lambda item: item[0],
    )
    return {
        "player_height": height,
        "relative_x": relative_x,
        "relative_y": relative_y,
        "hand_pose_available": hand_pose.get("available", False),
        "hand_pose_supported": hand_pose.get("supported", False),
        "hand_pose_distance": hand_pose.get("normalized_distance"),
    }


def _top_left_distance(previous_box, current_box):
    return (
        (previous_box[0] - current_box[0]) ** 2
        + (previous_box[1] - current_box[1]) ** 2
    ) ** 0.5


def _select_ball_detection(detections, players, previous_bbox):
    if not detections:
        return None
    highest_confidence = max(detections, key=lambda item: item["confidence"])
    if _minimum_bbox_distance(highest_confidence["bbox"], players) <= 80:
        return highest_confidence

    minimum_alternative_confidence = max(
        0.5,
        highest_confidence["confidence"] - 0.35,
    )
    supported = [
        detection
        for detection in detections
        if detection["confidence"] >= minimum_alternative_confidence
        and _minimum_bbox_distance(detection["bbox"], players) <= 50
    ]
    if not supported:
        return highest_confidence

    def fallback_score(detection):
        continuity = 0.5
        if previous_bbox is not None:
            continuity = max(
                0.0,
                1.0 - _bbox_center_distance(previous_bbox, detection["bbox"]) / 150.0,
            )
        return 0.75 * detection["confidence"] + 0.25 * continuity

    return max(supported, key=fallback_score)


def _minimum_bbox_distance(ball_bbox, players):
    if not players:
        return 80.0
    center_x = (ball_bbox[0] + ball_bbox[2]) / 2
    center_y = (ball_bbox[1] + ball_bbox[3]) / 2
    distances = []
    for player in players.values():
        bbox = player.get("bbox")
        if not bbox:
            continue
        closest_x = min(max(center_x, bbox[0]), bbox[2])
        closest_y = min(max(center_y, bbox[1]), bbox[3])
        distances.append(
            ((center_x - closest_x) ** 2 + (center_y - closest_y) ** 2) ** 0.5
        )
    return min(distances, default=80.0)


def _central_upper_player_id(ball_bbox, players):
    center_x, center_y = _bbox_center(ball_bbox)
    containing = []
    for player_id, player in players.items():
        bbox = player.get("bbox")
        if not bbox or not (
            bbox[0] <= center_x <= bbox[2]
            and bbox[1] <= center_y <= bbox[3]
        ):
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= 0 or height <= 0:
            continue
        relative_x = (center_x - bbox[0]) / width
        relative_y = (center_y - bbox[1]) / height
        if 0.15 <= relative_x <= 0.85 and 0.10 <= relative_y <= 0.40:
            containing.append((width * height, player_id))
    return min(containing)[1] if containing else None


def _hand_track_transition_is_consistent(
    previous_ball,
    current_ball,
    player,
    *,
    maximum_player_height_fraction=0.35,
):
    previous_bbox = previous_ball.get("bbox")
    current_bbox = current_ball.get("bbox")
    player_bbox = player.get("bbox")
    if not previous_bbox or not current_bbox or not player_bbox:
        return False
    player_height = float(player_bbox[3]) - float(player_bbox[1])
    if player_height <= 0:
        return False
    displacement = math_dist(
        _bbox_center(previous_bbox),
        _bbox_center(current_bbox),
    )
    return displacement / player_height <= maximum_player_height_fraction


def _nearest_hand_pose_evidence(
    ball_bbox,
    players,
    *,
    player_id=None,
    minimum_keypoint_confidence=0.35,
    maximum_supported_distance=0.05,
):
    """Measure ball proximity to confident wrists and elbow-to-wrist segments.

    Distances are normalized by player height so the signal scales across
    resolutions and camera zoom. Missing or weak poses return unavailable and
    therefore remain neutral in candidate selection.
    """
    if not ball_bbox or not players:
        return {"available": False, "supported": False}
    ball_center = np.asarray(_bbox_center(ball_bbox), dtype=float)
    evidence = []
    player_items = (
        [(player_id, players.get(player_id, {}))]
        if player_id is not None
        else players.items()
    )
    for current_player_id, player in player_items:
        player_bbox = player.get("bbox")
        pose = player.get("pose")
        if not player_bbox or not pose:
            continue
        player_height = float(player_bbox[3]) - float(player_bbox[1])
        points = pose.get("keypoints_xy") or []
        confidences = pose.get("keypoint_confidences") or []
        if player_height <= 0 or len(points) <= 10 or len(confidences) <= 10:
            continue

        distances = []
        for elbow_index, wrist_index in ((7, 9), (8, 10)):
            wrist = _confident_pose_point(
                points,
                confidences,
                wrist_index,
                minimum_keypoint_confidence,
            )
            if wrist is None:
                continue
            distances.append(float(np.linalg.norm(ball_center - wrist)))
            elbow = _confident_pose_point(
                points,
                confidences,
                elbow_index,
                minimum_keypoint_confidence,
            )
            if elbow is not None:
                distances.append(_point_to_segment_distance(ball_center, elbow, wrist))
        if not distances:
            continue
        normalized_distance = min(distances) / player_height
        evidence.append((normalized_distance, current_player_id))

    if not evidence:
        return {"available": False, "supported": False}
    normalized_distance, matched_player_id = min(evidence)
    return {
        "available": True,
        "supported": normalized_distance <= maximum_supported_distance,
        "normalized_distance": float(normalized_distance),
        "player_id": matched_player_id,
    }


def _confident_pose_point(points, confidences, index, threshold):
    confidence = float(confidences[index])
    point = np.asarray(points[index], dtype=float)
    if (
        confidence < threshold
        or point.shape != (2,)
        or not np.isfinite(point).all()
        or (point <= 0).any()
    ):
        return None
    return point


def _point_to_segment_distance(point, start, end):
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, segment) / length_squared)
    projection = start + min(1.0, max(0.0, fraction)) * segment
    return float(np.linalg.norm(point - projection))


def _bbox_center_distance(first_bbox, second_bbox):
    first_center = _bbox_center(first_bbox)
    second_center = _bbox_center(second_bbox)
    return (
        (first_center[0] - second_center[0]) ** 2
        + (first_center[1] - second_center[1]) ** 2
    ) ** 0.5


def _bbox_center(bbox):
    return (
        (bbox[0] + bbox[2]) / 2,
        (bbox[1] + bbox[3]) / 2,
    )


def _camera_adjusted_top_left_distance(previous_box, current_box, camera_motion):
    dx = current_box[0] - previous_box[0] - camera_motion[0]
    dy = current_box[1] - previous_box[1] - camera_motion[1]
    return (dx**2 + dy**2) ** 0.5


def _median_player_motion(player_tracks, previous_index, current_index):
    if player_tracks is None:
        return 0.0, 0.0
    previous_players = player_tracks[previous_index]
    current_players = player_tracks[current_index]
    displacements = []
    for player_id in previous_players.keys() & current_players.keys():
        previous_bbox = previous_players[player_id].get("bbox")
        current_bbox = current_players[player_id].get("bbox")
        if not previous_bbox or not current_bbox:
            continue
        displacements.append(
            (
                current_bbox[0] - previous_bbox[0],
                current_bbox[1] - previous_bbox[1],
            )
        )
    if len(displacements) < 3:
        return 0.0, 0.0
    return tuple(np.median(np.asarray(displacements), axis=0).tolist())
