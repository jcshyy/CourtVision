from __future__ import annotations

from dataclasses import dataclass, field
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
    arbitration: list[dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "frames": [dict(frame) for frame in self.frames],
            "sequences": [dict(sequence) for sequence in self.sequences],
            "events": [dict(event) for event in self.events],
            "candidates": [dict(candidate) for candidate in self.candidates],
            "arbitration": [dict(decision) for decision in self.arbitration],
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
        # Rim-anchored flights must not depend on a clean holder transition:
        # a false catch can otherwise truncate a real layup or jump-shot arc.
        rim_candidates = []
        boundaries = [0, *sorted(discontinuities), frame_count]
        for first, stop in zip(boundaries, boundaries[1:]):
            for start, end, seed in _rim_candidate_windows(
                ball_tracks, player_tracks, None, first, stop - 1,
                self.maximum_launch_lookback_frames,
                self.maximum_rim_distance_player_heights,
            ):
                rim_candidates.append({
                    "from_player_id": None, "release_frame": start,
                    "transition_frame": end, "status": "rim_seeded",
                    "evidence_start_frame": start, "evidence_end_frame": end,
                    "rim_seed_frame": seed,
                })
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
        # Prefer holder-grounded release boundaries when available. Global rim
        # windows are recovery hypotheses, not grounds to move a known release
        # backward across an earlier pass into the shooter.
        candidates.extend(rim_candidates)
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
            shooter_id = transition.get("from_player_id")
            if shooter_id is not None:
                shooter_id = int(shooter_id)
            evidence = self._shot_evidence(
                ball_tracks,
                player_tracks,
                shooter_id,
                evidence_start,
                evidence_end,
            )
            diagnostic = {
                "holder_release_frame": holder_release,
                "evidence_start_frame": evidence_start,
                "evidence_end_frame": evidence_end,
                "rim_seed_frame": transition.get("rim_seed_frame"),
                "source_player_id": shooter_id,
                "transition_status": transition.get("status"),
                "receiver_player_id": transition.get("to_player_id"),
                "evidence": evidence,
            }
            candidate_diagnostics.append(diagnostic)
            if not evidence.get("confirmed"):
                continue
            release = int(evidence["inferred_release_frame"])
            rim_frame = int(evidence["rim_approach_frame"])
            duplicate = next((sequence for sequence in sequences
                              if _same_shot_flight(release, rim_frame, sequence)), None)
            if duplicate is not None:
                diagnostic["decision"] = "duplicate_flight"
                diagnostic["sequence_id"] = duplicate["sequence_id"]
                continue
            if catch is not None and catch <= release:
                continue
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
            pending_end = post_control_frame if post_control_frame is not None else pending_limit

            shooter_id = _release_holder(
                possession_timeline, release, shooter_id,
                max(3, self.minimum_post_shot_control_frames * 3),
                minimum_frame=max((cut for cut in discontinuities if cut <= release), default=0),
            )
            hand_player = (evidence.get("shooting_hand_support") or {}).get("player_id")
            if hand_player is not None:
                shooter_id = hand_player if shooter_id in (None, hand_player) else None

            shooter_team, shooter_team_source = self._holder_team(
                player_assignment, shooter_id, release
            )

            sequence_id = len(sequences) + 1
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
            diagnostic["decision"] = "accepted"
            diagnostic["sequence_id"] = sequence_id

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

        return ShotReboundTimeline(
            _shot_state_frames(frame_count, sequences),
            sorted(sequences, key=lambda sequence: sequence["release_frame"]),
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
        if not player_height and player_tracks is not None:
            heights = [track["bbox"][3] - track["bbox"][1]
                       for track in player_tracks[release].values()
                       if track.get("bbox") and track["bbox"][3] > track["bbox"][1]]
            player_height = statistics.median(heights) if heights else None
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
        if not post_launch:
            return {"confirmed": False, "reason": "no_observed_post_launch_ball",
                    "observed_ball_frames": len(observed)}
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
        # Ball motion in image coordinates alone is ambiguous during a camera
        # pan or a pass below the hoop. Even a nearby 2D ball must reach rim
        # height, or have an elevated release consistent with a blocked shot;
        # being underneath the basket alone is not shot evidence.
        relative_heights = [
            (item["center"][1] - min(item["rims"],
             key=lambda rim: math.dist(item["center"], rim[0]))[0][1]) / player_height
            for item in post_launch if item["rims"]
        ]
        rim_height_reached = bool(relative_heights) and min(relative_heights) <= 0.15
        launch_box = None
        if player_tracks is not None:
            launch_box = player_tracks[int(launch["frame"])].get(shooter_id, {}).get("bbox")
        elevated_release = bool(launch_box) and (
            launch_box[0] - player_height * 0.2 <= start[0] <= launch_box[2] + player_height * 0.2
            and launch_box[1] - player_height * 0.25 <= start[1] <= launch_box[1] + player_height * 0.35
        )
        shooting_hands = _shooting_hand_support(post_launch, player_tracks)
        shot_height_evidence = (rim_height_reached
                                or ((elevated_release or shooting_hands is not None)
                                    and rise_normalized >= 0.2))
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
            and shot_height_evidence
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
        elif not shot_height_evidence:
            rejection = "below_rim_transfer_without_shooting_release"

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
            "inferred_release_frame": int(shooting_hands["first_frame"] if shooting_hands else launch["frame"]),
            "holder_release_frame": int(release),
            "rim_approach_frame": int(closest["frame"]),
            "rim_bbox": [round(float(value), 2) for value in rim_bbox],
            "rim_height_reached": rim_height_reached,
            "elevated_release": elevated_release,
            "minimum_ball_height_relative_to_rim": round(min(relative_heights), 4) if relative_heights else None,
            "shooting_hand_support": shooting_hands,
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
    """Arbitrate provisional possession events against complete shot flights.

    A catch BEFORE the rim may refute a shot (a lob). New play AFTER the rim
    cannot erase an attempt. Identity changes do not create a second ball.
    Decisions remain inspectable separately from the public event contract.
    """
    if not isinstance(shot_timeline, ShotReboundTimeline):
        raise TypeError("shot_timeline must be a ShotReboundTimeline")
    preempted_ids = {
        int(sequence["sequence_id"])
        for sequence in shot_timeline.sequences
        if any(
            _stable_catch_precedes_rim(event, sequence)
            for event in possession_events
        )
    }
    if preempted_ids:
        shot_timeline.arbitration.extend({
            "sequence_id": sequence_id, "decision": "rejected_shot",
            "reason": "stable_catch_before_rim",
        } for sequence_id in sorted(preempted_ids))
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
        shot_timeline.frames = _shot_state_frames(
            len(shot_timeline.frames), shot_timeline.sequences,
        )

    retained = []
    for event in possession_events:
        covered = next((sequence for sequence in shot_timeline.sequences
                        if _event_is_covered_by_shot(event, sequence)), None)
        if covered is None:
            retained.append(event)
        else:
            shot_timeline.arbitration.append({
                "event": dict(event), "decision": "suppressed_possession_event",
                "reason": "shot_flight_or_rebound_acquisition",
                "sequence_id": covered["sequence_id"],
            })
    return sorted(
        retained + [dict(event) for event in shot_timeline.events],
        key=lambda event: (int(event["frame_index"]), event["type"]),
    )


def _stable_catch_precedes_rim(event, sequence):
    if event.get("type") not in ("pass", "interception"):
        return False
    rim_frame = int(sequence.get("evidence", {}).get(
        "rim_approach_frame", sequence["pending_end_frame"]
    ))
    event_frame = int(event.get("catch_frame", event.get("frame_index", -1)))
    event_release = int(event.get("release_frame", event_frame))
    evidence = sequence.get("evidence", {})
    # A high pass can approach rim *height* without approaching the basket.
    # Stable control ending that weak flight outweighs image-space arc shape,
    # even if the two possession streams disagree about release/actor IDs.
    # This exception never treats a later outlet as a veto of a true rim flight.
    if (evidence.get("rim_distance_player_heights", 0) > 0.6
            and not evidence.get("shooting_hand_support")
            and (event.get("possession_evidence", {}).get("receiver_support_frames") or 0) >= 3
            and event_release < rim_frame
            and int(sequence["release_frame"]) < event_frame <= rim_frame + 2):
        return True
    same_source = event.get("from_player_id") is not None and event.get("from_player_id") == sequence.get("shooter_id")
    if not same_source and not (
        abs(event_release - int(sequence["release_frame"])) <= 3
        and (event.get("possession_evidence", {}).get("receiver_support_frames") or 0) >= 3
    ):
        return False
    holder_release = int(sequence.get(
        "holder_release_frame", sequence["release_frame"]
    ))
    return (
        holder_release - 3 <= event_release <= rim_frame
        and event_release <= int(sequence["release_frame"]) + 2
        and event_frame < rim_frame
    )


def _event_is_covered_by_shot(event, sequence):
    if event.get("type") not in ("pass", "interception"):
        return False
    release = int(sequence["release_frame"])
    pending_end = int(sequence["pending_end_frame"])
    rim_frame = int(sequence.get("rim_frame", sequence.get("evidence", {}).get(
        "rim_approach_frame", pending_end)))
    event_release = int(event.get("release_frame", event.get("frame_index", -1)))
    event_frame = int(event.get("frame_index", -1))
    holder_release = int(sequence.get("holder_release_frame", release))
    control = sequence.get("post_shot_control_frame")
    # An outlet/inbound starts a new flight after established post-rim control.
    if event_release > rim_frame and (control is None or event_release >= int(control)):
        return False
    return (
        min(holder_release, release) - 2 <= event_release <= pending_end
        and release < event_frame <= pending_end + 2
    )


def _same_shot_flight(release, rim_frame, sequence):
    """Cluster overlapping ball flights, not shooter IDs or pending windows."""
    previous_release = int(sequence["release_frame"])
    previous_rim = int(sequence["rim_frame"])
    control = sequence.get("post_shot_control_frame")
    if control is not None and previous_rim < int(control) <= release:
        return False
    overlap = min(rim_frame, previous_rim) - max(release, previous_release)
    # One ball cannot be in two shot flights at once. Allow a two-frame seam
    # from local rim-window splitting, but the established-control check above
    # protects a genuinely new release after a rebound.
    return overlap >= -2


def _shot_state_frames(frame_count, sequences):
    frames = [{"frame_index": index, "state": "possession"} for index in range(frame_count)]
    for sequence in sorted(sequences, key=lambda item: item["release_frame"]):
        release, rim = int(sequence["release_frame"]), int(sequence["rim_frame"])
        control = sequence.get("post_shot_control_frame")
        for index in range(release, int(sequence["pending_end_frame"]) + 1):
            state = ("shot_attempt" if index == release else "shot_in_flight" if index <= rim
                     else "post_shot_control" if index == control else "post_shot_unresolved")
            frames[index] = {"frame_index": index, "state": state,
                             "sequence_id": sequence["sequence_id"], "shooter_id": sequence["shooter_id"],
                             "holder_id": sequence.get("post_shot_control_holder_id") if index == control else None}
    return frames


def _release_holder(timeline, release, fallback, lookback, *, minimum_frame=0):
    """Use recent stable control, never a stale owner from an earlier play."""
    eligible = [segment for segment in timeline.segments
                if segment["start_frame"] <= release
                and segment["last_frame"] >= max(minimum_frame, release - lookback)
                and sum(holder == segment["holder_id"] for holder in timeline.acquisitions[
                    max(minimum_frame, segment["start_frame"], release - lookback):release + 1]) >= 2]
    if not eligible:
        return None
    # Most recent control wins; ties favor the candidate's source identity.
    selected = max(eligible, key=lambda segment: (
        min(release, segment["last_frame"]),
        segment["holder_id"] == fallback,
        segment["support_frames"],
    ))
    return int(selected["holder_id"])


def _shooting_hand_support(observations, player_tracks):
    """Two observed ball contacts with a high wrist support blocked attempts."""
    if player_tracks is None:
        return None
    supported = {}
    for observation in observations:
        frame = int(observation["frame"])
        for player_id, player in player_tracks[frame].items():
            box = player.get("bbox")
            pose = player.get("pose", {})
            points = pose.get("keypoints_xy", [])
            confidence = pose.get("keypoint_confidences", [])
            if box is None or len(points) < 11 or len(confidence) < 11:
                continue
            height = box[3] - box[1]
            if height <= 0:
                continue
            for wrist in (9, 10):
                if (confidence[wrist] >= 0.4 and points[wrist][1] <= box[1] + 0.2 * height
                        and math.dist(points[wrist], observation["center"]) <= height * 0.2):
                    supported.setdefault(player_id, []).append(frame)
                    break
    eligible = [(player_id, frames) for player_id, frames in supported.items() if len(frames) >= 2]
    if not eligible:
        return None
    player_id, frames = max(eligible, key=lambda item: len(item[1]))
    return {"player_id": int(player_id), "first_frame": min(frames), "last_frame": max(frames),
            "observed_frames": len(frames)}


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

    separation = max(12, int(round(lookback_frames * 0.25)))
    selected = []
    for distance, frame in sorted(approaches):
        if distance > 0.6:
            continue
        if all(abs(frame - prior) > separation for prior in selected):
            selected.append(frame)
    selected.sort()
    lookahead = max(4, int(round(lookback_frames / 6)))
    return [
        (
            max(first, frame - int(lookback_frames), selected[index - 1] + 1 if index else first),
            min(last, frame + lookahead),
            frame,
        )
        for index, frame in enumerate(selected)
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
            if abs(current_y - float(run_start["center"][1])) < 1e-6 and run_min_y >= current_y:
                # A stationary gather precedes launch; use its last frame.
                run_start = current
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
    candidates = []
    segments = possession_timeline.segments
    for index, last in enumerate(segments):
        following = segments[index + 1] if index + 1 < len(segments) else None
        if following is not None and following.get("segment", 0) == last.get("segment", 0):
            continue
        end = int(following.get("segment_start", following["start_frame"])) - 1 if following else frame_count - 1
        release = int(last["last_frame"])
        if release >= end - 1:
            continue
        candidates.append({
            "status": "terminal",
            "from_player_id": int(last["holder_id"]),
            "to_player_id": None,
            "release_frame": release,
            "catch_frame": None,
            "transition_frame": end,
            "source_support_frames": int(last.get("support_frames") or 0),
        })
    return candidates


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
