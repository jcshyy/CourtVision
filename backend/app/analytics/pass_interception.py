import math
import statistics

from backend.app.analytics.possession_timeline import (
    PossessionTimeline,
    PossessionTimelineBuilder,
)


class PassInterceptionDetector:
    """Detects passes and interceptions using the reference repo's holder changes."""

    def __init__(
        self,
        max_holder_gap_frames=None,
        team_lookup_frames=30,
        transient_control_frames=12,
        minimum_loose_ball_confidence=0.5,
        minimum_catch_ball_confidence=0.45,
        minimum_catch_frames=3,
        minimum_source_frames=1,
        minimum_initial_source_frames=1,
        catch_confirmation_frames=30,
        reject_preexisting_competing_takeovers=True,
        minimum_interception_player_separation=0.75,
        event_team_hints=None,
    ):
        self.max_holder_gap_frames = max_holder_gap_frames
        self.team_lookup_frames = max(1, int(team_lookup_frames))
        self.transient_control_frames = max(1, int(transient_control_frames))
        self.minimum_loose_ball_confidence = float(
            minimum_loose_ball_confidence
        )
        self.minimum_catch_ball_confidence = float(
            minimum_catch_ball_confidence
        )
        self.minimum_catch_frames = max(1, int(minimum_catch_frames))
        self.minimum_source_frames = max(1, int(minimum_source_frames))
        self.minimum_initial_source_frames = max(
            self.minimum_source_frames,
            int(minimum_initial_source_frames),
        )
        self.catch_confirmation_frames = max(
            self.minimum_catch_frames,
            int(catch_confirmation_frames),
        )
        self.reject_preexisting_competing_takeovers = bool(
            reject_preexisting_competing_takeovers
        )
        self.minimum_interception_player_separation = max(
            0.0,
            float(minimum_interception_player_separation),
        )
        self.event_team_hints = {
            int(player_id): int(team_id)
            for player_id, team_id in (event_team_hints or {}).items()
            if int(team_id) in (1, 2)
        }
        self.last_possession_timeline = None

    def clean_transient_control_chains(
        self,
        ball_acquisition,
        player_assignment,
        holder_states=None,
        discontinuity_frames=None,
    ):
        """Remove brief holder islands before interpreting player changes.

        Unknown frames do not establish control. A short middle episode is
        treated as contested tracking noise when possession returns to the
        same player/team, or as the transient first touch when the next stable
        holder is on the middle player's team.
        """
        cleaned = list(ball_acquisition)
        discontinuities = _normalize_discontinuities(
            discontinuity_frames,
            len(cleaned),
        )
        while True:
            episodes = _holder_episodes(cleaned, discontinuities)
            changed = False
            for previous, middle, following in zip(
                episodes,
                episodes[1:],
                episodes[2:],
            ):
                if not (
                    previous["segment"]
                    == middle["segment"]
                    == following["segment"]
                ):
                    continue
                if (
                    following["start_frame"] - middle["start_frame"]
                    > self.transient_control_frames
                ):
                    continue

                previous_team = self._holder_team(
                    player_assignment,
                    previous["holder_id"],
                    previous["last_frame"],
                    minimum_frame=previous["segment_start"],
                )
                middle_team = self._holder_team(
                    player_assignment,
                    middle["holder_id"],
                    middle["last_frame"],
                    minimum_frame=middle["segment_start"],
                )
                following_team = self._holder_team(
                    player_assignment,
                    following["holder_id"],
                    following["start_frame"],
                    minimum_frame=following["segment_start"],
                )
                returned_to_source = (
                    previous["holder_id"] == following["holder_id"]
                    or (
                        previous_team in (1, 2)
                        and previous_team == following_team
                        and middle_team != previous_team
                    )
                )
                transient_first_touch = (
                    previous_team in (1, 2)
                    and middle_team in (1, 2)
                    and previous_team != middle_team
                    and middle_team == following_team
                )
                if not (returned_to_source or transient_first_touch):
                    continue

                for frame in middle["frames"]:
                    cleaned[frame] = -1
                changed = True
                break
            if not changed:
                break
        return cleaned

    def detect_passes(
        self,
        ball_acquisition,
        player_assignment,
        discontinuity_frames=None,
    ):
        passes = [-1] * len(ball_acquisition)
        previous_holder = -1
        previous_frame = -1
        discontinuities = _normalize_discontinuities(
            discontinuity_frames,
            len(ball_acquisition),
        )
        segment_start = 0

        for frame in range(1, len(ball_acquisition)):
            if frame in discontinuities:
                previous_holder = -1
                previous_frame = -1
                segment_start = frame
            elif ball_acquisition[frame - 1] != -1:
                previous_holder = ball_acquisition[frame - 1]
                previous_frame = frame - 1

            current_holder = ball_acquisition[frame]

            if (
                previous_holder != -1
                and current_holder != -1
                and previous_holder != current_holder
                and self._gap_is_valid(frame, previous_frame)
            ):
                previous_team = self._holder_team(
                    player_assignment, previous_holder, previous_frame,
                    minimum_frame=segment_start,
                )
                current_team = self._holder_team(
                    player_assignment, current_holder, frame,
                    minimum_frame=segment_start,
                )

                if previous_team == current_team and previous_team != -1:
                    passes[frame] = previous_team

        return passes

    def detect_interceptions(
        self,
        ball_acquisition,
        player_assignment,
        holder_states=None,
        discontinuity_frames=None,
    ):
        interceptions = [-1] * len(ball_acquisition)
        previous_holder = -1
        previous_frame = -1
        discontinuities = _normalize_discontinuities(
            discontinuity_frames,
            len(ball_acquisition),
        )
        segment_start = 0

        for frame in range(1, len(ball_acquisition)):
            if frame in discontinuities:
                previous_holder = -1
                previous_frame = -1
                segment_start = frame
            elif ball_acquisition[frame - 1] != -1:
                previous_holder = ball_acquisition[frame - 1]
                previous_frame = frame - 1

            current_holder = ball_acquisition[frame]

            if (
                previous_holder != -1
                and current_holder != -1
                and previous_holder != current_holder
                and self._gap_is_valid(frame, previous_frame)
                and self._interception_evidence_is_valid(
                    holder_states,
                    previous_frame,
                    frame,
                )
            ):
                previous_team = self._holder_team(
                    player_assignment, previous_holder, previous_frame,
                    minimum_frame=segment_start,
                )
                current_team = self._holder_team(
                    player_assignment, current_holder, frame,
                    minimum_frame=segment_start,
                )

                if previous_team != current_team and previous_team != -1 and current_team != -1:
                    interceptions[frame] = current_team

        return interceptions

    def _gap_is_valid(self, current_frame, previous_frame):
        return (
            self.max_holder_gap_frames is None
            or current_frame - previous_frame - 1 <= self.max_holder_gap_frames
        )

    def _holder_team(
        self,
        player_assignment,
        holder_id,
        frame,
        *,
        minimum_frame=0,
    ):
        return self._holder_team_resolution(
            player_assignment,
            holder_id,
            frame,
            minimum_frame=minimum_frame,
        )[0]

    def _holder_team_resolution(
        self,
        player_assignment,
        holder_id,
        frame,
        *,
        minimum_frame=0,
    ):
        start = max(minimum_frame, frame - self.team_lookup_frames + 1)
        for lookup_frame in range(frame, start - 1, -1):
            team_id = player_assignment[lookup_frame].get(holder_id)
            if team_id in (1, 2):
                return int(team_id), "frame_assignment"
        hinted_team = self.event_team_hints.get(int(holder_id))
        if hinted_team in (1, 2):
            return hinted_team, "high_consensus_event_hint"
        return -1, "unknown"

    def _interception_evidence_is_valid(
        self,
        holder_states,
        previous_frame,
        current_frame,
    ):
        if holder_states is None:
            return True
        return any(
            state.get("state") == "loose"
            and state.get("ball_confidence") is not None
            and float(state["ball_confidence"])
            >= self.minimum_loose_ball_confidence
            for state in holder_states[previous_frame + 1 : current_frame]
        )

    def detect_events(
        self,
        ball_acquisition,
        player_assignment,
        holder_states=None,
        ball_tracks=None,
        player_tracks=None,
        discontinuity_frames=None,
        possession_timeline=None,
    ):
        """Interpret confirmed possession segments as passes or interceptions."""
        if possession_timeline is None:
            possession_timeline = self.build_possession_timeline(
                ball_acquisition,
                holder_states=holder_states,
                ball_tracks=ball_tracks,
                player_tracks=player_tracks,
                discontinuity_frames=discontinuity_frames,
            )
        elif not isinstance(possession_timeline, PossessionTimeline):
            raise TypeError("possession_timeline must be a PossessionTimeline")
        self.last_possession_timeline = possession_timeline
        ball_acquisition = possession_timeline.acquisitions
        events = []
        discontinuities = _normalize_discontinuities(
            discontinuity_frames,
            len(ball_acquisition),
        )
        for transition in possession_timeline.transitions:
            if transition.get("status") != "confirmed":
                transition["event_status"] = "rejected"
                transition["event_rejection_reason"] = transition.get("reason")
                continue
            transition["event_status"] = "candidate"
            transition.pop("event_rejection_reason", None)
            previous_holder = int(transition["from_player_id"])
            current_holder = int(transition["to_player_id"])
            previous_frame = int(transition["release_frame"])
            holder_tail_frame = int(
                transition.get("holder_tail_frame", previous_frame)
            )
            frame = int(transition["transition_frame"])
            catch_frame = int(transition["catch_frame"])
            segment_start = int(transition.get("segment_start", 0))
            source_start = int(transition.get("source_start_frame", previous_frame))
            if not self._gap_is_valid(catch_frame, previous_frame):
                transition["event_status"] = "rejected"
                transition["event_rejection_reason"] = "holder_gap_too_long"
                continue
            required_source_frames = self.minimum_source_frames
            if source_start == segment_start:
                required_source_frames = max(
                    required_source_frames,
                    self.minimum_initial_source_frames,
                )
            if segment_start > 0 and source_start == segment_start:
                required_source_frames = max(
                    required_source_frames,
                    self.minimum_catch_frames,
                )
            if int(transition.get("source_support_frames") or 0) < required_source_frames:
                transition["event_status"] = "rejected"
                transition["event_rejection_reason"] = "source_control_too_short"
                continue
            if (
                self.reject_preexisting_competing_takeovers
                and not _ball_track_transition_is_credible(
                    ball_tracks,
                    player_tracks,
                    current_holder,
                    previous_frame,
                    catch_frame,
                )
            ):
                transition["event_status"] = "rejected"
                transition["event_rejection_reason"] = "ball_transition_not_credible"
                continue

            previous_team, previous_team_source = self._holder_team_resolution(
                player_assignment,
                previous_holder,
                previous_frame,
                minimum_frame=segment_start,
            )
            current_team, current_team_source = self._holder_team_resolution(
                player_assignment,
                current_holder,
                catch_frame,
                minimum_frame=segment_start,
            )
            if previous_team not in (1, 2) or current_team not in (1, 2):
                transition["event_status"] = "rejected"
                transition["event_rejection_reason"] = "team_assignment_unknown"
                continue

            event_type = None
            if previous_team == current_team:
                if _is_rising_flythrough(
                    ball_tracks,
                    player_tracks,
                    previous_holder,
                    previous_frame,
                    catch_frame,
                    self.minimum_catch_frames,
                ):
                    transition["event_status"] = "rejected"
                    transition["event_rejection_reason"] = "rising_flythrough"
                else:
                    event_type = "pass"
            else:
                loose_evidence = self._interception_evidence_is_valid(
                    holder_states,
                    previous_frame,
                    frame,
                )
                catch_evidence = self._catch_evidence_is_valid(
                    holder_states,
                    catch_frame,
                )
                close_range = _is_close_range_contested_turnover(
                    player_tracks,
                    previous_holder,
                    current_holder,
                    previous_frame,
                    catch_frame,
                    self.minimum_interception_player_separation,
                    self.minimum_catch_frames,
                )
                if not loose_evidence:
                    transition["event_status"] = "rejected"
                    transition["event_rejection_reason"] = "no_observed_loose_ball"
                elif not catch_evidence:
                    transition["event_status"] = "rejected"
                    transition["event_rejection_reason"] = "weak_catch_evidence"
                elif close_range:
                    transition["event_status"] = "rejected"
                    transition["event_rejection_reason"] = (
                        "brief_close_range_contested_takeover"
                    )
                else:
                    event_type = "interception"
            if event_type is None:
                continue
            transition["event_status"] = "emitted"
            transition["event_type"] = event_type

            events.append({
                "type": event_type,
                "frame_index": catch_frame,
                "from_player_id": previous_holder,
                "to_player_id": current_holder,
                "from_team_id": previous_team,
                "to_team_id": current_team,
                "release_frame": previous_frame,
                "holder_tail_frame": holder_tail_frame,
                "release_localization_reason": transition.get(
                    "release_localization_reason"
                ),
                "catch_frame": catch_frame,
                "gap_frames": catch_frame - previous_frame - 1,
                "transition_frame": frame,
                "confirmation_frames": catch_frame - frame + 1,
                "possession_evidence": {
                    "source_support_frames": transition.get(
                        "source_support_frames",
                    ),
                    "receiver_support_frames": transition.get(
                        "receiver_support_frames",
                    ),
                    "observed_loose_frames": transition.get(
                        "observed_loose_frames",
                    ),
                    "observed_flight_frames": transition.get(
                        "observed_flight_frames",
                    ),
                    "interpolated_flight_frames": transition.get(
                        "interpolated_flight_frames",
                    ),
                },
                "team_resolution": {
                    "from": previous_team_source,
                    "to": current_team_source,
                },
            })

        return events

    def build_possession_timeline(
        self,
        ball_acquisition,
        *,
        holder_states=None,
        ball_tracks=None,
        player_tracks=None,
        discontinuity_frames=None,
    ):
        """Build the causal possession timeline used by event detection."""
        causal_acquisition = list(ball_acquisition)
        if holder_states is not None:
            # Retrospective possession recovery is useful for visualization,
            # but it is hindsight rather than release/catch evidence.
            causal_acquisition = _causal_event_acquisitions(
                causal_acquisition,
                holder_states,
                ball_tracks=ball_tracks,
                player_tracks=player_tracks,
            )
        builder = PossessionTimelineBuilder(
            minimum_catch_frames=self.minimum_catch_frames,
            catch_confirmation_frames=self.catch_confirmation_frames,
        )
        timeline = builder.build(
            causal_acquisition,
            holder_states=holder_states,
            ball_tracks=ball_tracks,
            discontinuity_frames=discontinuity_frames,
        )
        self.last_possession_timeline = timeline
        return timeline

    def _stable_catch_frame(
        self,
        ball_acquisition,
        frame,
        holder_id,
        *,
        discontinuities=None,
    ):
        end = min(
            len(ball_acquisition),
            frame + self.catch_confirmation_frames + 1,
        )
        run_start = None
        run_frames = 0
        for candidate_frame in range(frame, end):
            if candidate_frame != frame and candidate_frame in (discontinuities or set()):
                return None
            candidate_holder = ball_acquisition[candidate_frame]
            if candidate_holder == holder_id:
                if run_start is None:
                    run_start = candidate_frame
                run_frames += 1
                if run_frames >= self.minimum_catch_frames:
                    return run_start
            elif candidate_holder == -1:
                run_start = None
                run_frames = 0
            else:
                return None
        return None

    def _catch_evidence_is_valid(self, holder_states, catch_frame):
        if holder_states is None:
            return True
        if not 0 <= catch_frame < len(holder_states):
            return False
        state = holder_states[catch_frame]
        ball_confidence = state.get("ball_confidence")
        return (
            state.get("state") == "confirmed"
            and ball_confidence is not None
            and float(ball_confidence) >= self.minimum_catch_ball_confidence
        )


def _causal_event_acquisitions(
    ball_acquisition,
    holder_states,
    *,
    ball_tracks=None,
    player_tracks=None,
):
    """Remove hindsight-only possession while retaining proven inner bridges."""
    causal = list(ball_acquisition)
    for frame, holder_id in enumerate(causal):
        if frame >= len(holder_states):
            continue
        reason = holder_states[frame].get("reason")
        if reason == "retrospective_holder_confirmation":
            causal[frame] = -1
        elif (
            reason == "same_holder_gap_bridged"
            and not _trusted_internal_holder_bridge(
                frame,
                holder_id,
                ball_acquisition,
                holder_states,
                ball_tracks,
                player_tracks,
            )
        ):
            causal[frame] = -1
    return causal


def _trusted_internal_holder_bridge(
    frame,
    holder_id,
    ball_acquisition,
    holder_states,
    ball_tracks,
    player_tracks,
    *,
    maximum_gap_frames=2,
    maximum_midpoint_error=8.0,
    maximum_height_error_fraction=0.08,
    maximum_player_distance_fraction=0.25,
):
    """Trust a short trajectory-supported gap inside one proven possession.

    The offline holder model may bridge several frames after seeing the same
    holder on both sides. Event detection accepts at most two of those frames,
    and only when real detections anchor one ball-track segment, every middle
    point is interpolation, the path is linear, and it remains close to the
    same player. This can extend established control but cannot create a holder.
    """
    if (
        holder_id in (-1, None)
        or ball_tracks is None
        or not 0 < frame < len(ball_acquisition) - 1
        or len(ball_tracks) != len(ball_acquisition)
        or holder_states[frame].get("reason") != "same_holder_gap_bridged"
    ):
        return False
    hindsight_reasons = {
        "retrospective_holder_confirmation",
        "same_holder_gap_bridged",
    }

    left = frame - 1
    while left >= 0 and holder_states[left].get("reason") == "same_holder_gap_bridged":
        left -= 1
    right = frame + 1
    while right < len(holder_states) and holder_states[right].get("reason") == "same_holder_gap_bridged":
        right += 1
    gap_frames = right - left - 1
    if (
        left < 0
        or right >= len(ball_acquisition)
        or not 0 < gap_frames <= int(maximum_gap_frames)
        or ball_acquisition[left] != holder_id
        or ball_acquisition[right] != holder_id
        or holder_states[left].get("reason") in hindsight_reasons
        or holder_states[right].get("reason") in hindsight_reasons
    ):
        return False

    before = ball_tracks[left].get(1, {})
    after = ball_tracks[right].get(1, {})
    middle = [ball_tracks[index].get(1, {}) for index in range(left + 1, right)]
    if (
        before.get("interpolated", False)
        or after.get("interpolated", False)
        or not before.get("bbox")
        or not after.get("bbox")
        or any(not item.get("interpolated", False) or not item.get("bbox") for item in middle)
    ):
        return False
    sequence = [before, *middle, after]
    segments = [item.get("track_segment_id") for item in sequence]
    if any(segment is None for segment in segments) or len(set(segments)) != 1:
        return False
    if any(
        rejection.get("reason") == "persistent_competing_takeover_chain"
        for info in sequence
        for rejection in info.get("candidate_rejections", [])
    ):
        return False

    before_center = _bbox_center(before["bbox"])
    after_center = _bbox_center(after["bbox"])
    for offset, info in enumerate(middle, start=1):
        index = left + offset
        fraction = offset / (gap_frames + 1)
        expected = (
            before_center[0] + (after_center[0] - before_center[0]) * fraction,
            before_center[1] + (after_center[1] - before_center[1]) * fraction,
        )
        center = _bbox_center(info["bbox"])
        player_bbox = None
        if player_tracks is not None and index < len(player_tracks):
            player_bbox = player_tracks[index].get(holder_id, {}).get("bbox")
        if not player_bbox:
            return False
        player_height = max(0.0, float(player_bbox[3]) - float(player_bbox[1]))
        allowed_error = max(
            float(maximum_midpoint_error),
            player_height * float(maximum_height_error_fraction),
        )
        trajectory_error = math.dist(center, expected)
        player_distance = _point_to_bbox_distance(center, player_bbox)
        if (
            trajectory_error > allowed_error
            or player_distance > player_height * float(maximum_player_distance_fraction)
        ):
            return False
    return True


def _point_to_bbox_distance(point, bbox):
    x, y = point
    x1, y1, x2, y2 = (float(value) for value in bbox)
    closest_x = min(max(float(x), x1), x2)
    closest_y = min(max(float(y), y1), y2)
    return math.dist((float(x), float(y)), (closest_x, closest_y))


def summarize_events(events):
    return {
        "passes": sum(event["type"] == "pass" for event in events),
        "interceptions": sum(event["type"] == "interception" for event in events),
    }


def build_event_team_hints(
    assignment_metadata,
    *,
    minimum_confident_observations=12,
    minimum_agreement=0.75,
    minimum_weight_share=0.75,
):
    """Recover conservative, event-only team hints for borderline tracks.

    Automatic team assignment intentionally leaves inconsistent tracks unknown.
    A pass boundary can still use a strong plurality when the dominant color
    vote agrees on at least three quarters of the evidence.  The hint is not
    written back to the possession timeline or rendered as a normal team label.
    """
    hints = {}
    track_assignments = (assignment_metadata or {}).get("track_assignments", {})
    for player_id, decision in track_assignments.items():
        if (
            decision.get("status") != "unknown"
            or decision.get("reason") != "inconsistent_track_observations"
            or int(decision.get("confident_observation_count") or 0)
            < int(minimum_confident_observations)
            or float(decision.get("agreement") or 0.0)
            < float(minimum_agreement)
            or float(decision.get("weight_share") or 0.0)
            < float(minimum_weight_share)
        ):
            continue
        weighted_votes = {
            int(team_id): float(weight)
            for team_id, weight in decision.get("team_vote_weights", {}).items()
            if int(team_id) in (1, 2)
        }
        if not weighted_votes:
            continue
        winning_team = max(weighted_votes, key=weighted_votes.get)
        total_weight = sum(weighted_votes.values())
        if (
            total_weight <= 0
            or weighted_votes[winning_team] / total_weight
            < float(minimum_weight_share)
        ):
            continue
        hints[int(player_id)] = int(winning_team)
    return hints


def merge_corroborated_pass_events(
    primary_events,
    corroborating_events,
    player_tracks,
    *,
    duplicate_window_frames=8,
    minimum_player_separation=1.25,
    minimum_gap_frames=1,
):
    """Add spatially credible fused-track passes to semantic-track events.

    The fused E-BARD/WASB path has better flight continuity but more false
    takeovers.  It may therefore supplement passes only: interceptions remain
    semantic-only, close-range identity switches are rejected, and nearby
    primary events win during deduplication.
    """
    merged = []
    duplicate_window_frames = max(0, int(duplicate_window_frames))
    minimum_gap_frames = max(0, int(minimum_gap_frames))
    for event in primary_events:
        if event.get("type") == "pass" and any(
            _same_pass_flight(existing, event, player_tracks, duplicate_window_frames)
            for existing in merged
        ):
            continue
        merged.append({"detection_source": "semantic_ball", **event})
    for event in corroborating_events:
        if event.get("type") != "pass":
            continue
        if int(event.get("gap_frames") or 0) < minimum_gap_frames:
            continue
        separation = _player_transition_separation(
            player_tracks,
            event.get("from_player_id"),
            event.get("to_player_id"),
            event.get("release_frame"),
            event.get("catch_frame"),
        )
        if separation is None or separation < float(minimum_player_separation):
            continue
        if any(_same_pass_flight(existing, event, player_tracks, duplicate_window_frames)
               for existing in merged):
            continue
        merged.append({
            "detection_source": "fused_ball_corroboration",
            **event,
            "player_separation_heights": round(float(separation), 4),
        })
    return sorted(merged, key=lambda event: int(event["frame_index"]))


def _same_pass_flight(left, right, player_tracks, duplicate_window_frames):
    """Two observations of one transfer, not merely events close in time.

    Endpoint identity (or overlapping track boxes for an ID switch) and flight
    overlap are both required. This preserves fast A->B->C passes while merging
    delayed catches from the semantic and fused ball paths.
    """
    if left.get("type") != "pass" or right.get("type") != "pass":
        return False
    for team_field in ("from_team_id", "to_team_id"):
        if (left.get(team_field) is not None and right.get(team_field) is not None
                and left[team_field] != right[team_field]):
            return False
    for actor, boundary in (("from_player_id", "release_frame"), ("to_player_id", "catch_frame")):
        if left.get(actor) is None or right.get(actor) is None:
            return False
        if left.get(actor) is not None and left.get(actor) == right.get(actor):
            continue
        if player_tracks is None:
            return False
        frame = min(int(left.get(boundary, left["frame_index"])),
                    int(right.get(boundary, right["frame_index"])))
        first = _nearby_player_bbox(player_tracks, left.get(actor), frame)
        second = _nearby_player_bbox(player_tracks, right.get(actor), frame)
        if not first or not second:
            return False
        intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
            0, min(first[3], second[3]) - max(first[1], second[1]))
        union = ((first[2] - first[0]) * (first[3] - first[1])
                 + (second[2] - second[0]) * (second[3] - second[1]) - intersection)
        if union <= 0 or intersection / union < 0.6:
            return False
    if left.get("release_frame") is None or right.get("release_frame") is None:
        return abs(int(left["frame_index"]) - int(right["frame_index"])) <= duplicate_window_frames
    start = max(int(left["release_frame"]), int(right["release_frame"]))
    end = min(int(left.get("catch_frame", left["frame_index"])),
              int(right.get("catch_frame", right["frame_index"])))
    shorter = min(int(left.get("catch_frame", left["frame_index"])) - int(left["release_frame"]),
                  int(right.get("catch_frame", right["frame_index"])) - int(right["release_frame"]))
    return shorter > 0 and end - start >= max(1, shorter * 0.5)


def _player_transition_separation(
    player_tracks,
    source_holder,
    receiver_id,
    release_frame,
    catch_frame,
):
    if player_tracks is None:
        return None
    source_bbox = _nearby_player_bbox(
        player_tracks,
        source_holder,
        int(release_frame),
    )
    receiver_bbox = _nearby_player_bbox(
        player_tracks,
        receiver_id,
        int(catch_frame),
    )
    if source_bbox is None or receiver_bbox is None:
        return None
    source_height = float(source_bbox[3]) - float(source_bbox[1])
    receiver_height = float(receiver_bbox[3]) - float(receiver_bbox[1])
    scale = (source_height + receiver_height) / 2.0
    if scale <= 0:
        return None
    source_center = _bbox_center(source_bbox)
    receiver_center = _bbox_center(receiver_bbox)
    return (
        (receiver_center[0] - source_center[0]) ** 2
        + (receiver_center[1] - source_center[1]) ** 2
    ) ** 0.5 / scale


def events_from_arrays(
    passes,
    interceptions,
    ball_acquisition,
    player_assignment=None,
):
    events = []

    for frame_index, team_id in enumerate(passes):
        if team_id != -1:
            events.append(
                _array_event(
                    "pass",
                    frame_index,
                    team_id,
                    ball_acquisition,
                    player_assignment,
                )
            )

    for frame_index, team_id in enumerate(interceptions):
        if team_id != -1:
            events.append(
                _array_event(
                    "interception",
                    frame_index,
                    team_id,
                    ball_acquisition,
                    player_assignment,
                )
            )

    return sorted(events, key=lambda event: event["frame_index"])


def _array_event(
    event_type,
    frame_index,
    team_id,
    ball_acquisition,
    player_assignment=None,
):
    previous_holder, previous_frame = _previous_holder_info(
        ball_acquisition,
        frame_index,
    )
    previous_team = None
    if (
        player_assignment is not None
        and previous_frame is not None
        and previous_frame < len(player_assignment)
    ):
        previous_team = player_assignment[previous_frame].get(previous_holder)
    return {
        "type": event_type,
        "frame_index": frame_index,
        "from_player_id": previous_holder,
        "to_player_id": ball_acquisition[frame_index],
        "from_team_id": team_id if event_type == "pass" else previous_team,
        "to_team_id": team_id,
        "release_frame": previous_frame,
        "catch_frame": frame_index,
        "gap_frames": (
            frame_index - previous_frame - 1
            if previous_frame is not None
            else None
        ),
    }


def _previous_holder(ball_acquisition, frame_index):
    return _previous_holder_info(ball_acquisition, frame_index)[0]


def _previous_holder_info(ball_acquisition, frame_index):
    for index in range(frame_index - 1, -1, -1):
        player_id = ball_acquisition[index]
        if player_id is not None and player_id != -1:
            return player_id, index

    return None, None


PassAndInterceptionDetector = PassInterceptionDetector


def _ball_track_transition_is_credible(
    ball_tracks,
    player_tracks,
    receiver_id,
    release_frame,
    catch_frame,
    *,
    lookback_frames=8,
    minimum_competing_observations=2,
    maximum_relative_distance=0.12,
):
    """Reject a reset onto an object that demonstrably predated the pass.

    A long observation gap is not enough to reject an event by itself. But if
    the receiver-side object was already visible at a stable player-relative
    location while a different ball candidate was selected, it cannot be the
    same basketball that subsequently crossed the gap.
    """
    if ball_tracks is None or player_tracks is None:
        return True
    if not (
        0 <= release_frame < len(ball_tracks)
        and 0 <= catch_frame < len(ball_tracks)
    ):
        return False
    release_segment = ball_tracks[release_frame].get(1, {}).get("track_segment_id")
    catch_segment = ball_tracks[catch_frame].get(1, {}).get("track_segment_id")
    if (
        release_segment is None
        or catch_segment is None
        or release_segment == catch_segment
    ):
        return True
    if any(
        rejection.get("reason") == "persistent_competing_takeover_chain"
        for frame in ball_tracks[release_frame + 1 : catch_frame + 1]
        for rejection in frame.get(1, {}).get("candidate_rejections", [])
    ):
        return False

    catch_ball_bbox = ball_tracks[catch_frame].get(1, {}).get("bbox")
    catch_player_bbox = player_tracks[catch_frame].get(receiver_id, {}).get("bbox")
    target_relative = _relative_bbox_center(catch_ball_bbox, catch_player_bbox)
    if target_relative is None:
        return True

    competing_observations = 0
    start = max(0, release_frame - max(0, int(lookback_frames)) + 1)
    for frame in range(start, release_frame + 1):
        info = ball_tracks[frame].get(1, {})
        selected_bbox = info.get("bbox")
        receiver_bbox = player_tracks[frame].get(receiver_id, {}).get("bbox")
        if not selected_bbox or not receiver_bbox:
            continue
        for candidate in info.get(
            "raw_candidates",
            info.get("candidates", []),
        ):
            candidate_bbox = candidate.get("bbox")
            if not candidate_bbox or candidate_bbox == selected_bbox:
                continue
            relative = _relative_bbox_center(candidate_bbox, receiver_bbox)
            if relative is None:
                continue
            distance = (
                (relative[0] - target_relative[0]) ** 2
                + (relative[1] - target_relative[1]) ** 2
            ) ** 0.5
            if distance <= maximum_relative_distance:
                competing_observations += 1
                break
        if competing_observations >= minimum_competing_observations:
            return False
    return True


def _relative_bbox_center(ball_bbox, player_bbox):
    if not ball_bbox or not player_bbox:
        return None
    width = float(player_bbox[2]) - float(player_bbox[0])
    height = float(player_bbox[3]) - float(player_bbox[1])
    if width <= 0 or height <= 0:
        return None
    center_x, center_y = _bbox_center(ball_bbox)
    return (
        (center_x - float(player_bbox[0])) / width,
        (center_y - float(player_bbox[1])) / height,
    )


def _is_close_range_contested_turnover(
    player_tracks,
    source_holder,
    receiver_id,
    release_frame,
    catch_frame,
    minimum_separation,
    minimum_loose_frames,
    maximum_loose_frames=None,
):
    """Separate strips/contested recoveries from pass interceptions.

    An interception implies that the defender took a pass. When the source
    and new holder occupy the same small interaction area, the observable
    event is instead a steal, strip, or contested recovery. The current event
    schema has no safe subtype for those, so suppress the incorrect label.
    """
    loose_frames = catch_frame - release_frame - 1
    if maximum_loose_frames is None:
        maximum_loose_frames = max(6, int(minimum_loose_frames) * 3)
    if (
        player_tracks is None
        or minimum_separation <= 0
        or loose_frames < minimum_loose_frames
        or loose_frames > maximum_loose_frames
    ):
        return False
    if not (
        0 <= release_frame < len(player_tracks)
        and 0 <= catch_frame < len(player_tracks)
    ):
        return False
    source_bbox = _nearby_player_bbox(
        player_tracks,
        source_holder,
        release_frame,
    )
    receiver_bbox = _nearby_player_bbox(
        player_tracks,
        receiver_id,
        catch_frame,
    )
    if source_bbox is None or receiver_bbox is None:
        return False
    source_height = float(source_bbox[3]) - float(source_bbox[1])
    receiver_height = float(receiver_bbox[3]) - float(receiver_bbox[1])
    scale = (source_height + receiver_height) / 2.0
    if scale <= 0:
        return False
    source_center = _bbox_center(source_bbox)
    receiver_center = _bbox_center(receiver_bbox)
    separation = (
        (receiver_center[0] - source_center[0]) ** 2
        + (receiver_center[1] - source_center[1]) ** 2
    ) ** 0.5 / scale
    return separation < minimum_separation


def _is_rising_flythrough(
    ball_tracks,
    player_tracks,
    source_holder,
    release_frame,
    catch_frame,
    confirmation_frames,
):
    """Identify an airborne ball merely crossing a background player box."""
    if ball_tracks is None or player_tracks is None:
        return False
    if len(ball_tracks) != len(player_tracks):
        return False

    source_bbox = _nearby_player_bbox(
        player_tracks,
        source_holder,
        release_frame,
    )
    if source_bbox is None:
        return False
    source_height = float(source_bbox[3] - source_bbox[1])
    if source_height <= 0:
        return False

    end_frame = min(
        len(ball_tracks) - 1,
        catch_frame + max(confirmation_frames, 3),
    )
    observations = []
    for frame in range(max(0, release_frame), end_frame + 1):
        info = ball_tracks[frame].get(1, {})
        bbox = info.get("bbox")
        if not bbox or info.get("interpolated", False):
            continue
        observations.append((frame, _bbox_center(bbox)))
    if len(observations) < 5:
        return False

    displacements = []
    for (start_frame, start), (end_frame, end) in zip(
        observations,
        observations[1:],
    ):
        gap = end_frame - start_frame
        if gap <= 0:
            continue
        camera_dx, camera_dy = _median_player_motion(
            player_tracks,
            start_frame,
            end_frame,
        )
        displacements.append((
            (end[0] - start[0] - camera_dx) / gap,
            (end[1] - start[1] - camera_dy) / gap,
        ))
    if len(displacements) < 4:
        return False

    net_dx = sum(dx for dx, _ in displacements)
    net_dy = sum(dy for _, dy in displacements)
    upward_fraction = sum(dy < 0 for _, dy in displacements) / len(displacements)
    recent_vertical_speed = statistics.median(
        dy for _, dy in displacements[-3:]
    )
    return (
        net_dy <= -0.75 * source_height
        and abs(net_dy) >= 1.5 * abs(net_dx)
        and upward_fraction >= 0.70
        and recent_vertical_speed <= -0.01 * source_height
    )


def _nearby_player_bbox(player_tracks, player_id, frame, radius=5):
    for offset in range(radius + 1):
        indices = [frame - offset]
        if offset:
            indices.append(frame + offset)
        for index in indices:
            if not 0 <= index < len(player_tracks):
                continue
            bbox = player_tracks[index].get(player_id, {}).get("bbox")
            if bbox:
                return bbox
    return None


def _median_player_motion(player_tracks, start_frame, end_frame):
    start_players = player_tracks[start_frame]
    end_players = player_tracks[end_frame]
    deltas = []
    for player_id in start_players.keys() & end_players.keys():
        start_bbox = start_players[player_id].get("bbox")
        end_bbox = end_players[player_id].get("bbox")
        if not start_bbox or not end_bbox:
            continue
        start = _bbox_center(start_bbox)
        end = _bbox_center(end_bbox)
        deltas.append((end[0] - start[0], end[1] - start[1]))
    if not deltas:
        return 0.0, 0.0
    return (
        statistics.median(dx for dx, _ in deltas),
        statistics.median(dy for _, dy in deltas),
    )


def _bbox_center(bbox):
    return (
        (float(bbox[0]) + float(bbox[2])) / 2,
        (float(bbox[1]) + float(bbox[3])) / 2,
    )


def _normalize_discontinuities(discontinuity_frames, frame_count):
    return {
        int(frame)
        for frame in (discontinuity_frames or [])
        if 0 < int(frame) < frame_count
    }


def _holder_episodes(ball_acquisition, discontinuities=None):
    """Return known-holder episodes while leaving unknown frames as gaps."""
    episodes = []
    current = None
    segment = 0
    segment_start = 0
    discontinuities = discontinuities or set()
    for frame, holder_id in enumerate(ball_acquisition):
        if frame in discontinuities:
            current = None
            segment += 1
            segment_start = frame
        if holder_id is None or holder_id == -1:
            continue
        if current is None or current["holder_id"] != holder_id:
            current = {
                "holder_id": holder_id,
                "start_frame": frame,
                "last_frame": frame,
                "frames": [frame],
                "segment": segment,
                "segment_start": segment_start,
            }
            episodes.append(current)
        else:
            current["last_frame"] = frame
            current["frames"].append(frame)
    return episodes
