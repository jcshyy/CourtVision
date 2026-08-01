import numpy as np
import supervision as sv
from ultralytics import YOLO

from backend.app.config import BALL_DETECTOR_PATH
from backend.app.utils import load_cache, save_cache

BALL_TRACKING_CACHE_VERSION = "v5_temporal_candidate_lattice"


class BallTracker:
    """Detects and filters basketball positions using the reference repo flow."""

    def __init__(self, model_path=BALL_DETECTOR_PATH):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for start in range(0, len(frames), batch_size):
            detections.extend(
                self.model.predict(
                    frames[start : start + batch_size],
                    conf=0.25,
                )
            )

        return detections

    def get_object_tracks(
        self,
        frames,
        read_from_cache=False,
        cache_path=None,
        player_tracks=None,
    ):
        tracks = load_cache(cache_path, enabled=read_from_cache) if cache_path else None
        if tracks is not None and len(tracks) == len(frames):
            return tracks

        detections = self.detect_frames(frames)
        candidate_frames = []
        for detection in detections:
            class_names_inv = {value: key for key, value in detection.names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            ball_detections = []

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                confidence = frame_detection[2]

                if class_id == class_names_inv["Ball"]:
                    ball_detections.append(
                        {"bbox": bbox, "confidence": float(confidence)}
                    )

            candidate_frames.append(ball_detections)

        tracks = _select_ball_track(candidate_frames, player_tracks)

        if cache_path:
            save_cache(cache_path, tracks)

        return tracks

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
                    {
                        "bbox": list(candidate["bbox"]),
                        "confidence": float(candidate["confidence"]),
                        "candidate_index": int(
                            candidate.get("candidate_index", candidate_index)
                        ),
                    }
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
            {
                "bbox": list(candidate["bbox"]),
                "confidence": float(candidate["confidence"]),
                "candidate_index": int(
                    candidate.get("candidate_index", current_candidate_index)
                ),
            }
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
