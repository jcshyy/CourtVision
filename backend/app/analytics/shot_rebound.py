from __future__ import annotations

from dataclasses import dataclass
import math
import statistics

from backend.app.analytics.possession_timeline import PossessionTimeline


@dataclass
class ShotReboundTimeline:
    """Inspectable shot/rebound state layered over possession."""

    frames: list[dict]
    sequences: list[dict]
    events: list[dict]
    candidates: list[dict]

    def to_dict(self):
        return {
            "frames": [dict(frame) for frame in self.frames],
            "sequences": [dict(sequence) for sequence in self.sequences],
            "events": [dict(event) for event in self.events],
            "candidates": [dict(candidate) for candidate in self.candidates],
        }


class ShotReboundDetector:
    """Recognize shot attempts from release, arc, and rim-approach evidence.

    The production contract intentionally stops at ``shot_attempt``. Broadcast
    ball tracks are not reliable enough at the rim to publish make/miss claims.
    Post-shot control is used only to bound and de-duplicate attempt windows;
    it is not exported as an outcome or rebound event.
    """

    def __init__(
        self,
        *,
        minimum_flight_observations=3,
        minimum_rise_player_heights=0.07,
        maximum_rim_distance_player_heights=2.0,
        minimum_approach_player_heights=0.08,
        minimum_trajectory_strength=0.3,
        maximum_pending_frames=90,
        maximum_launch_lookback_frames=20,
        minimum_post_shot_control_frames=3,
        team_lookup_frames=30,
        event_team_hints=None,
    ):
        self.minimum_flight_observations = max(
            2, int(minimum_flight_observations)
        )
        self.minimum_rise_player_heights = max(
            0.0, float(minimum_rise_player_heights)
        )
        self.maximum_rim_distance_player_heights = max(
            0.1, float(maximum_rim_distance_player_heights)
        )
        self.minimum_approach_player_heights = max(
            0.0, float(minimum_approach_player_heights)
        )
        self.minimum_trajectory_strength = max(
            0.0, float(minimum_trajectory_strength)
        )
        self.maximum_pending_frames = max(1, int(maximum_pending_frames))
        self.maximum_launch_lookback_frames = max(
            3, int(maximum_launch_lookback_frames)
        )
        self.minimum_post_shot_control_frames = max(
            2, int(minimum_post_shot_control_frames)
        )
        self.team_lookup_frames = max(1, int(team_lookup_frames))
        self.event_team_hints = {
            int(player_id): int(team_id)
            for player_id, team_id in (event_team_hints or {}).items()
            if int(team_id) in (1, 2)
        }

    def detect(
        self,
        possession_timeline,
        player_assignment,
        ball_tracks,
        player_tracks,
        *,
        discontinuity_frames=None,
    ):
        if not isinstance(possession_timeline, PossessionTimeline):
            raise TypeError("possession_timeline must be a PossessionTimeline")
        frame_count = len(possession_timeline.frames)
        if len(ball_tracks) != frame_count:
            raise ValueError("Ball tracks and possession timeline must align")
        discontinuities = _normalize_discontinuities(
            discontinuity_frames, frame_count
        )
        states = [
            {"frame_index": frame, "state": "possession"}
            for frame in range(frame_count)
        ]
        sequences = []
        events = []
        candidate_diagnostics = []

        base_candidates = [
            dict(transition) for transition in possession_timeline.transitions
        ]
        base_candidates.extend(_terminal_release_candidates(
            possession_timeline,
            frame_count,
        ))
        candidates = []
        for transition in base_candidates:
            holder_release = int(transition["release_frame"])
            transition_catch = (
                int(transition["catch_frame"])
                if transition.get("status") == "confirmed"
                and transition.get("catch_frame") is not None
                else None
            )
            transition_end = (
                transition_catch
                if transition_catch is not None
                else min(
                    frame_count - 1,
                    int(transition.get("transition_frame", frame_count - 1)),
                    holder_release + self.maximum_pending_frames,
                )
            )
            windows = _rim_candidate_windows(
                ball_tracks,
                player_tracks,
                int(transition["from_player_id"]),
                holder_release,
                transition_end,
                self.maximum_launch_lookback_frames,
                self.maximum_rim_distance_player_heights,
            )
            if not windows:
                candidates.append(dict(transition))
                continue
            for evidence_start, evidence_end, rim_seed in windows:
                candidate = dict(transition)
                candidate["evidence_start_frame"] = evidence_start
                candidate["evidence_end_frame"] = evidence_end
                candidate["rim_seed_frame"] = rim_seed
                candidates.append(candidate)
        confirmed_rim_frames = []
        confirmed_release_frames = []
        for transition in candidates:
            holder_release = int(transition["release_frame"])
            catch = (
                int(transition["catch_frame"])
                if transition.get("status") == "confirmed"
                and transition.get("catch_frame") is not None
                else None
            )
            evidence_end = transition.get("evidence_end_frame") or (
                catch
                if catch is not None
                else min(
                    frame_count - 1,
                    int(transition.get("transition_frame", frame_count - 1)),
                    holder_release + self.maximum_pending_frames,
                )
            )
            evidence_start = int(
                transition.get("evidence_start_frame", holder_release)
            )
            evidence_end = int(evidence_end)
            later_cuts = [
                cut for cut in discontinuities
                if evidence_start < cut <= evidence_end
            ]
            if later_cuts:
                evidence_end = min(later_cuts) - 1
            if evidence_end <= evidence_start:
                continue
            shooter_id = int(transition["from_player_id"])
            evidence = self._shot_evidence(
                ball_tracks,
                player_tracks,
                shooter_id,
                evidence_start,
                evidence_end,
            )
            candidate_diagnostics.append({
                "holder_release_frame": holder_release,
                "evidence_start_frame": evidence_start,
                "evidence_end_frame": evidence_end,
                "rim_seed_frame": transition.get("rim_seed_frame"),
                "source_player_id": shooter_id,
                "transition_status": transition.get("status"),
                "receiver_player_id": transition.get("to_player_id"),
                "evidence": evidence,
            })
            if not evidence.get("confirmed"):
                continue
            release = int(evidence["inferred_release_frame"])
            rim_frame = int(evidence["rim_approach_frame"])
            if (
                any(abs(rim_frame - prior) <= 8 for prior in confirmed_rim_frames)
                or any(
                    abs(release - prior) <= 5
                    for prior in confirmed_release_frames
                )
            ):
                continue
            if catch is not None and catch <= release:
                continue
            confirmed_rim_frames.append(rim_frame)
            confirmed_release_frames.append(release)
            pending_limit = _pending_end_frame(
                rim_frame,
                frame_count,
                discontinuities,
                self.maximum_pending_frames,
            )
            post_shot_control = _find_stable_control(
                possession_timeline.acquisitions,
                rim_frame + 1,
                pending_limit,
                self.minimum_post_shot_control_frames,
            )
            post_control_frame = (
                int(post_shot_control["catch_frame"])
                if post_shot_control is not None
                else None
            )
            post_control_holder_id = (
                int(post_shot_control["holder_id"])
                if post_shot_control is not None
                else None
            )
            pending_end = post_control_frame or pending_limit

            shooter_team, shooter_team_source = self._holder_team(
                player_assignment, shooter_id, release
            )

            sequence_id = len(sequences) + 1
            if (
                sequences
                and release <= int(sequences[-1]["pending_end_frame"])
                and (
                    sequences[-1].get("post_shot_control_frame") is not None
                    or (
                        shooter_id == sequences[-1].get("shooter_id")
                        and release - int(sequences[-1]["release_frame"])
                        <= max(8, self.maximum_launch_lookback_frames // 2)
                    )
                )
            ):
                # This trajectory is inside a previously bounded attempt
                # window, so it is another view of the same ball path.
                confirmed_rim_frames.pop()
                confirmed_release_frames.pop()
                continue
            sequence = {
                "sequence_id": sequence_id,
                "status": "confirmed_attempt",
                "release_frame": release,
                "holder_release_frame": holder_release,
                "rim_frame": rim_frame,
                "pending_end_frame": pending_end,
                "shooter_id": shooter_id,
                "shooter_team_id": shooter_team,
                "post_shot_control_frame": post_control_frame,
                "post_shot_control_holder_id": post_control_holder_id,
                "evidence": evidence,
            }
            sequences.append(sequence)

            shot_event = {
                "type": "shot_attempt",
                # Localize the candidate between launch and rim confirmation.
                # Both boundary frames remain available as explicit evidence.
                "frame_index": round(
                    (release + int(evidence["rim_approach_frame"])) / 2
                ),
                "from_player_id": shooter_id,
                "to_player_id": None,
                "from_team_id": shooter_team,
                "to_team_id": shooter_team,
                "release_frame": release,
                "sequence_id": sequence_id,
                "trajectory_evidence": evidence,
                "team_resolution": {"from": shooter_team_source},
            }
            events.append(shot_event)

            states[release] = {
                "frame_index": release,
                "state": "shot_attempt",
                "sequence_id": sequence_id,
                "shooter_id": shooter_id,
            }
            for frame in range(release + 1, rim_frame + 1):
                states[frame] = {
                    "frame_index": frame,
                    "state": "shot_in_flight",
                    "sequence_id": sequence_id,
                    "shooter_id": shooter_id,
                }

        return ShotReboundTimeline(
            states,
            sequences,
            events,
            candidate_diagnostics,
        )

    def _shot_evidence(
        self,
        ball_tracks,
        player_tracks,
        shooter_id,
        release,
        catch,
    ):
        observations = []
        for frame in range(release, catch + 1):
            ball = ball_tracks[frame].get(1, {})
            bbox = ball.get("bbox")
            if not bbox:
                continue
            center = _bbox_center(bbox)
            observations.append({
                "frame": frame,
                "center": center,
                "interpolated": bool(ball.get("interpolated", False)),
                "rims": _rim_centers(ball.get("rim_regions", [])),
            })
        observed = [item for item in observations if not item["interpolated"]]
        if len(observed) < self.minimum_flight_observations:
            return {
                "confirmed": False,
                "reason": "insufficient_observed_flight",
                "observed_ball_frames": len(observed),
            }

        player_height = _nearby_player_height(
            player_tracks, shooter_id, release
        )
        if not player_height:
            ball_heights = [
                float(ball_tracks[item["frame"]][1]["bbox"][3])
                - float(ball_tracks[item["frame"]][1]["bbox"][1])
                for item in observed
            ]
            player_height = max(40.0, statistics.median(ball_heights) * 8.0)

        rim_matches = []
        for item in observations:
            for rim_center, rim_bbox in item["rims"]:
                distance = math.dist(item["center"], rim_center)
                rim_matches.append((distance, item, rim_center, rim_bbox))
        if not rim_matches:
            return {
                "confirmed": False,
                "reason": "no_rim_evidence",
                "observed_ball_frames": len(observed),
            }
        rim_distance, closest, rim_center, rim_bbox = min(
            rim_matches, key=lambda match: match[0]
        )
        observed_pre_rim = [
            item
            for item in observed
            if max(release, int(closest["frame"]) - self.maximum_launch_lookback_frames)
            <= int(item["frame"])
            <= int(closest["frame"])
        ]
        pre_rim = observed_pre_rim
        if len(pre_rim) < self.minimum_flight_observations:
            # A basketball detector often loses the ball behind hands and the
            # rim.  Interpolation is useful for the *shape* of a candidate,
            # but only after two real observations anchor that trajectory.
            pre_rim = [
                item
                for item in observations
                if max(
                    release,
                    int(closest["frame"])
                    - self.maximum_launch_lookback_frames,
                )
                <= int(item["frame"])
                <= int(closest["frame"])
            ]
        if (
            len(observed_pre_rim) < 2
            or len(pre_rim) < self.minimum_flight_observations
        ):
            return {
                "confirmed": False,
                "reason": "insufficient_pre_rim_flight",
                "observed_ball_frames": len(observed),
                "observed_pre_rim_frames": len(observed_pre_rim),
                "total_pre_rim_frames": len(pre_rim),
            }
        launch = _infer_launch_observation(pre_rim)
        post_launch = [
            item for item in observed
            if int(launch["frame"]) <= int(item["frame"])
            <= int(closest["frame"])
        ]
        apex = min(post_launch, key=lambda item: item["center"][1])
        start = launch["center"]
        rise = start[1] - apex["center"][1]
        release_rim = min(
            (
                (math.dist(start, center), center)
                for item in observations
                for center, _ in item["rims"]
            ),
            default=(None, None),
            key=lambda match: match[0] if match[0] is not None else math.inf,
        )
        release_rim_distance = release_rim[0]
        approach = (
            release_rim_distance - rim_distance
            if release_rim_distance is not None
            else 0.0
        )
        rise_normalized = rise / player_height
        rim_distance_normalized = rim_distance / player_height
        approach_normalized = approach / player_height
        closest_after_release = int(closest["frame"]) > int(launch["frame"])
        rose_before_rim = int(apex["frame"]) <= int(closest["frame"])
        confirmed = (
            closest_after_release
            and rose_before_rim
            and rise_normalized >= self.minimum_rise_player_heights
            and rim_distance_normalized
            <= self.maximum_rim_distance_player_heights
            and approach_normalized >= self.minimum_approach_player_heights
            and rise_normalized + approach_normalized
            >= self.minimum_trajectory_strength
        )
        rejection = None
        if not closest_after_release:
            rejection = "rim_contact_precedes_release"
        elif not rose_before_rim:
            rejection = "apex_after_rim_approach"
        elif rise_normalized < self.minimum_rise_player_heights:
            rejection = "insufficient_vertical_rise"
        elif rim_distance_normalized > self.maximum_rim_distance_player_heights:
            rejection = "ball_did_not_approach_rim"
        elif approach_normalized < self.minimum_approach_player_heights:
            rejection = "trajectory_not_hoop_directed"
        elif (
            rise_normalized + approach_normalized
            < self.minimum_trajectory_strength
        ):
            rejection = "insufficient_combined_trajectory_strength"

        return {
            "confirmed": confirmed,
            "reason": "release_arc_rim_approach" if confirmed else rejection,
            "observed_ball_frames": len(observed),
            "interpolated_ball_frames": len(observations) - len(observed),
            "observed_pre_rim_frames": len(observed_pre_rim),
            "player_height_pixels": round(player_height, 3),
            "rise_player_heights": round(rise_normalized, 4),
            "rim_distance_player_heights": round(rim_distance_normalized, 4),
            "rim_approach_player_heights": round(approach_normalized, 4),
            "trajectory_strength": round(
                rise_normalized + approach_normalized, 4
            ),
            "apex_frame": int(apex["frame"]),
            "inferred_release_frame": int(launch["frame"]),
            "holder_release_frame": int(release),
            "rim_approach_frame": int(closest["frame"]),
            "rim_bbox": [round(float(value), 2) for value in rim_bbox],
        }

    def _holder_team(self, assignments, holder_id, frame):
        start = max(0, frame - self.team_lookup_frames + 1)
        for lookup_frame in range(min(frame, len(assignments) - 1), start - 1, -1):
            team_id = assignments[lookup_frame].get(holder_id)
            if team_id in (1, 2):
                return int(team_id), "frame_assignment"
        hinted_team = self.event_team_hints.get(holder_id)
        if hinted_team in (1, 2):
            return hinted_team, "high_consensus_event_hint"
        return None, "unknown"


def reconcile_shot_events(possession_events, shot_timeline):
    """Replace pass/turnover interpretations covered by a shot sequence."""
    if not isinstance(shot_timeline, ShotReboundTimeline):
        raise TypeError("shot_timeline must be a ShotReboundTimeline")
    preempted_ids = {
        int(sequence["sequence_id"])
        for sequence in shot_timeline.sequences
        if any(
            _stable_catch_precedes_rim(event, sequence)
            or _post_rim_possession_event_preempts(event, sequence)
            for event in possession_events
        )
    }
    if preempted_ids:
        shot_timeline.sequences = [
            sequence
            for sequence in shot_timeline.sequences
            if int(sequence["sequence_id"]) not in preempted_ids
        ]
        shot_timeline.events = [
            event
            for event in shot_timeline.events
            if int(event.get("sequence_id", -1)) not in preempted_ids
        ]
        for frame_index, frame in enumerate(shot_timeline.frames):
            if int(frame.get("sequence_id", -1)) in preempted_ids:
                frame.clear()
                frame.update({
                    "frame_index": frame_index,
                    "state": "possession",
                })

    retained = []
    for event in possession_events:
        covered = any(
            _event_is_covered_by_shot(event, sequence)
            for sequence in shot_timeline.sequences
        )
        if not covered:
            retained.append(event)
    return sorted(
        retained + [dict(event) for event in shot_timeline.events],
        key=lambda event: (int(event["frame_index"]), event["type"]),
    )


def _stable_catch_precedes_rim(event, sequence):
    if event.get("type") not in ("pass", "interception"):
        return False
    if event.get("from_player_id") != sequence.get("shooter_id"):
        return False
    rim_frame = int(sequence.get("evidence", {}).get(
        "rim_approach_frame", sequence["pending_end_frame"]
    ))
    event_frame = int(event.get("catch_frame", event.get("frame_index", -1)))
    event_release = int(event.get("release_frame", event_frame))
    holder_release = int(sequence.get(
        "holder_release_frame", sequence["release_frame"]
    ))
    return (
        holder_release - 3 <= event_release <= rim_frame
        and event_frame <= rim_frame
    )


def _event_is_covered_by_shot(event, sequence):
    if event.get("type") not in ("pass", "interception"):
        return False
    release = int(sequence["release_frame"])
    pending_end = int(sequence["pending_end_frame"])
    event_release = int(event.get("release_frame", event.get("frame_index", -1)))
    event_frame = int(event.get("frame_index", -1))
    same_source = event.get("from_player_id") == sequence.get("shooter_id")
    return (
        same_source
        and release - 2 <= event_release <= pending_end
        and release < event_frame <= pending_end + 2
    )


def _post_rim_possession_event_preempts(event, sequence):
    """Reject a shot if normal possession resumes before its resolution.

    A true shot may initially look like a pass from the shooter to the
    rebounder; that transition starts at the shot release and is reconciled
    later.  A pass or interception whose own release starts *after* the rim
    frame is different evidence: play has already resumed while the proposed
    shot still claims to be waiting for a rebound.
    """
    if event.get("type") not in ("pass", "interception"):
        return False
    rim_frame = int(sequence.get("rim_frame", sequence["release_frame"]))
    pending_end = int(sequence["pending_end_frame"])
    event_frame = int(event.get("frame_index", -1))
    event_release = int(event.get("release_frame", event_frame))
    return (
        rim_frame < event_release
        and event_release <= event_frame
        and event_frame <= pending_end + 2
    )


def _rim_centers(rim_regions):
    centers = []
    for rim in rim_regions or []:
        bbox = rim.get("bbox") if isinstance(rim, dict) else None
        if bbox and len(bbox) >= 4:
            centers.append((_bbox_center(bbox), list(bbox[:4])))
    return centers


def _rim_candidate_windows(
    ball_tracks,
    player_tracks,
    shooter_id,
    start,
    end,
    lookback_frames,
    maximum_distance_player_heights,
):
    """Split a long possession transition around distinct rim approaches.

    BARD clips can contain a block, a retained offensive possession, and a
    second or third shot without the possession tracker producing a clean
    holder transition between them.  Selecting only the globally closest rim
    observation collapses those attempts into one.  Local rim seeds preserve
    each trajectory while the regular rise/approach gates still reject a ball
    merely carried near the paint.
    """
    approaches = []
    first = max(0, int(start))
    last = min(len(ball_tracks) - 1, int(end))
    for frame in range(first, last + 1):
        ball = ball_tracks[frame].get(1, {})
        if not ball.get("bbox") or ball.get("interpolated", False):
            continue
        rims = _rim_centers(ball.get("rim_regions", []))
        if not rims:
            continue
        center = _bbox_center(ball["bbox"])
        distance = min(math.dist(center, rim_center) for rim_center, _ in rims)
        scale = _nearby_player_height(player_tracks, shooter_id, frame)
        if not scale and player_tracks is not None and frame < len(player_tracks):
            heights = [
                float(track["bbox"][3]) - float(track["bbox"][1])
                for track in player_tracks[frame].values()
                if track.get("bbox")
                and float(track["bbox"][3]) > float(track["bbox"][1])
            ]
            scale = statistics.median(heights) if heights else None
        scale = max(40.0, float(scale or 80.0))
        normalized = distance / scale
        if normalized <= maximum_distance_player_heights:
            approaches.append((normalized, frame))
    if not approaches:
        return []
    # Only split a transition when at least one observation strongly anchors
    # the ball to the rim.  Without this guard, an ordinary lob/pass whose
    # closest point is merely inside the broad recovery gate becomes a local
    # false shot.  The unsplit transition still goes through the normal
    # trajectory evaluator, which preserves recall for noisy blocked shots.
    if min(distance for distance, _ in approaches) > 0.6:
        return []

    separation = max(12, int(round(lookback_frames * 0.7)))
    selected = []
    for _, frame in sorted(approaches):
        if all(abs(frame - prior) > separation for prior in selected):
            selected.append(frame)
    selected.sort()
    lookahead = max(4, int(round(lookback_frames / 6)))
    return [
        (
            max(first, frame - int(lookback_frames)),
            min(last, frame + lookahead),
            frame,
        )
        for frame in selected
    ]


def _infer_launch_observation(observations, maximum_gap_frames=8, tolerance=5.0):
    """Find the start of the strongest observed ascent toward the rim.

    A simple lowest-point search is brittle when the ball disappears during a
    shot and is reacquired later on its descent. Sustained ascent runs preserve
    the pre-occlusion launch boundary while still allowing small detector
    jitter and short missing spans.
    """
    if len(observations) < 2:
        return observations[0]
    best_start = observations[0]
    best_rise = 0.0
    run_start = observations[0]
    run_min_y = float(observations[0]["center"][1])
    previous = observations[0]
    for current in observations[1:]:
        gap = int(current["frame"]) - int(previous["frame"])
        current_y = float(current["center"][1])
        previous_y = float(previous["center"][1])
        if gap <= maximum_gap_frames and current_y <= previous_y + tolerance:
            run_min_y = min(run_min_y, current_y)
        else:
            rise = float(run_start["center"][1]) - run_min_y
            if rise > best_rise:
                best_start, best_rise = run_start, rise
            run_start = current
            run_min_y = current_y
        previous = current
    rise = float(run_start["center"][1]) - run_min_y
    if rise > best_rise:
        best_start = run_start
    return best_start


def _find_stable_control(acquisitions, start, end, minimum_frames):
    """Return a new stable post-shot control run that begins after ``start``.

    Starting the scan in the middle of an existing control run used to turn a
    player who was already associated with the ball at the rim into a false
    new control boundary. The holder must change from the value immediately
    before the post-rim window (or follow a genuinely loose frame) before its
    stability support is counted.
    """
    run_holder = None
    run_start = None
    run_frames = 0
    scan_start = max(0, start)
    previous_holder = (
        acquisitions[scan_start - 1] if scan_start > 0 else None
    )
    eligible_run = previous_holder in (None, -1)
    for frame in range(scan_start, min(len(acquisitions) - 1, end) + 1):
        holder = acquisitions[frame]
        if holder not in (None, -1) and holder == run_holder:
            run_frames += 1
        elif holder not in (None, -1):
            run_holder = int(holder)
            run_start = frame
            run_frames = 1
            eligible_run = previous_holder in (None, -1) or holder != previous_holder
        else:
            run_holder = None
            run_start = None
            run_frames = 0
            eligible_run = True
        previous_holder = holder
        if eligible_run and run_frames >= minimum_frames:
            return {
                "holder_id": int(run_holder),
                "catch_frame": int(run_start),
                "confirmation_frame": int(frame),
                "support_frames": int(run_frames),
            }
    return None


def _bbox_center(bbox):
    return (
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    )


def _nearby_player_height(player_tracks, player_id, frame, radius=4):
    if player_tracks is None:
        return None
    for offset in range(radius + 1):
        candidate_frames = (frame,) if offset == 0 else (frame - offset, frame + offset)
        for candidate in candidate_frames:
            if not 0 <= candidate < len(player_tracks):
                continue
            bbox = player_tracks[candidate].get(player_id, {}).get("bbox")
            if bbox:
                height = float(bbox[3]) - float(bbox[1])
                if height > 0:
                    return height
    return None


def _normalize_discontinuities(discontinuity_frames, frame_count):
    return {
        int(frame)
        for frame in (discontinuity_frames or [])
        if 0 < int(frame) < frame_count
    }


def _terminal_release_candidates(possession_timeline, frame_count):
    if not possession_timeline.segments or frame_count < 2:
        return []
    last = possession_timeline.segments[-1]
    release = int(last["last_frame"])
    if release >= frame_count - 2:
        return []
    return [{
        "status": "terminal",
        "from_player_id": int(last["holder_id"]),
        "to_player_id": None,
        "release_frame": release,
        "catch_frame": None,
        "transition_frame": frame_count - 1,
        "source_support_frames": int(last.get("support_frames") or 0),
    }]


def _pending_end_frame(
    release,
    frame_count,
    discontinuities,
    maximum_pending_frames,
):
    end = min(frame_count - 1, release + maximum_pending_frames)
    later_cuts = [frame for frame in discontinuities if release < frame <= end]
    if later_cuts:
        end = min(later_cuts) - 1
    return max(release, end)
