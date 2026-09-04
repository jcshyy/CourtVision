from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PossessionTimeline:
    """Inspectable temporal possession state used by event interpretation."""

    acquisitions: list[int]
    frames: list[dict]
    segments: list[dict]
    transitions: list[dict]

    def to_dict(self):
        return {
            "frames": [dict(frame) for frame in self.frames],
            "segments": [dict(segment) for segment in self.segments],
            "transitions": [dict(transition) for transition in self.transitions],
        }


class PossessionTimelineBuilder:
    """Turn causal holder evidence into stable possession segments.

    The holder model answers the local question "who appears to control the
    ball in this frame?"  This layer answers the temporal question needed by
    event detection: which control runs are established, where a release
    occurs, and whether a later receiver establishes a stable catch.
    """

    def __init__(
        self,
        *,
        minimum_catch_frames=3,
        catch_confirmation_frames=30,
    ):
        self.minimum_catch_frames = max(1, int(minimum_catch_frames))
        self.catch_confirmation_frames = max(
            self.minimum_catch_frames,
            int(catch_confirmation_frames),
        )

    def build(
        self,
        acquisitions,
        *,
        holder_states=None,
        ball_tracks=None,
        discontinuity_frames=None,
    ):
        causal = [
            -1 if holder_id in (None, -1) else int(holder_id)
            for holder_id in acquisitions
        ]
        frame_count = len(causal)
        if holder_states is not None and len(holder_states) != frame_count:
            raise ValueError("Holder states and acquisitions must have the same frame count")
        if ball_tracks is not None and len(ball_tracks) != frame_count:
            raise ValueError("Ball tracks and acquisitions must have the same frame count")

        discontinuities = _normalize_discontinuities(
            discontinuity_frames,
            frame_count,
        )
        segments = _holder_segments(causal, discontinuities)
        frames = _frame_records(causal, holder_states, ball_tracks, discontinuities)
        transitions = self._transitions(
            causal,
            segments,
            holder_states,
            ball_tracks,
            discontinuities,
        )
        _mark_release_and_flight_frames(frames, transitions, ball_tracks)
        return PossessionTimeline(causal, frames, segments, transitions)

    def _transitions(
        self,
        acquisitions,
        segments,
        holder_states,
        ball_tracks,
        discontinuities,
    ):
        transitions = []
        for source, receiver in zip(segments, segments[1:]):
            if source["segment"] != receiver["segment"]:
                continue
            transition_frame = int(receiver["start_frame"])
            catch_frame = _stable_catch_frame(
                acquisitions,
                transition_frame,
                receiver["holder_id"],
                minimum_frames=self.minimum_catch_frames,
                confirmation_frames=self.catch_confirmation_frames,
                discontinuities=discontinuities,
            )
            holder_tail_frame = int(source["last_frame"])
            release_frame, release_reason = _authoritative_release_frame(
                acquisitions,
                holder_states,
                source,
            )
            gap_end = catch_frame if catch_frame is not None else transition_frame
            gap_states = (
                holder_states[release_frame + 1 : gap_end]
                if holder_states is not None
                else []
            )
            gap_tracks = (
                ball_tracks[release_frame + 1 : gap_end]
                if ball_tracks is not None
                else []
            )
            status = "confirmed" if catch_frame is not None else "rejected"
            reason = (
                "stable_receiver_control"
                if catch_frame is not None
                else "receiver_control_not_sustained"
            )
            receiver_support = (
                _controlled_run_length(
                    acquisitions,
                    catch_frame,
                    receiver["holder_id"],
                    discontinuities,
                )
                if catch_frame is not None
                else 0
            )
            transitions.append({
                "status": status,
                "reason": reason,
                "segment": int(source["segment"]),
                "segment_start": int(source["segment_start"]),
                "source_start_frame": int(source["start_frame"]),
                "from_player_id": int(source["holder_id"]),
                "to_player_id": int(receiver["holder_id"]),
                "release_frame": release_frame,
                "holder_tail_frame": holder_tail_frame,
                "release_localization_reason": release_reason,
                "transition_frame": transition_frame,
                "catch_frame": catch_frame,
                "gap_frames": (
                    catch_frame - release_frame - 1
                    if catch_frame is not None
                    else None
                ),
                "source_support_frames": int(source["support_frames"]),
                "receiver_support_frames": int(receiver_support),
                "observed_loose_frames": sum(
                    state.get("state") == "loose"
                    and state.get("ball_confidence") is not None
                    for state in gap_states
                ),
                "observed_flight_frames": sum(
                    _is_observed_ball(frame_ball) for frame_ball in gap_tracks
                ),
                "interpolated_flight_frames": sum(
                    bool(frame_ball.get(1, {}).get("interpolated", False))
                    for frame_ball in gap_tracks
                ),
            })
        return transitions


def _authoritative_release_frame(acquisitions, holder_states, source):
    """Exclude provisional holder carry from a transfer's release time.

    A holder can be retained for visualization during a missing/interpolated
    ball or the first frame of a pending switch. Those frames are useful for
    continuity but are not direct evidence that the player still controls the
    ball. Prefer the latest authoritative source-control observation; retain
    the original segment tail separately for inspection.
    """
    fallback = int(source["last_frame"])
    if holder_states is None:
        return fallback, "segment_tail_without_holder_diagnostics"
    provisional_reasons = {
        "brief_ball_gap",
        "brief_interpolated_gap",
        "retrospective_holder_confirmation",
        "same_holder_gap_bridged",
        "switch_pending",
    }
    holder_id = int(source["holder_id"])
    for frame_index in range(fallback, int(source["start_frame"]) - 1, -1):
        if acquisitions[frame_index] != holder_id:
            continue
        reason = holder_states[frame_index].get("reason")
        if reason not in provisional_reasons:
            return frame_index, (
                "authoritative_holder_observation"
                if frame_index != fallback
                else "authoritative_segment_tail"
            )
    return fallback, "no_authoritative_holder_observation"


def _frame_records(acquisitions, holder_states, ball_tracks, discontinuities):
    frames = []
    run_holder = None
    run_support = 0
    for frame_index, holder_id in enumerate(acquisitions):
        if frame_index in discontinuities:
            run_holder = None
            run_support = 0
        holder = None if holder_id == -1 else int(holder_id)
        if holder is not None and holder == run_holder:
            run_support += 1
        elif holder is not None:
            run_holder = holder
            run_support = 1
        else:
            run_holder = None
            run_support = 0

        evidence = (
            holder_states[frame_index]
            if holder_states is not None
            else {}
        )
        ball = (
            ball_tracks[frame_index].get(1, {})
            if ball_tracks is not None
            else {}
        )
        evidence_state = evidence.get("state")
        if holder is not None:
            state = "controlled"
        elif evidence_state == "candidate":
            state = "candidate_control"
        elif evidence_state == "loose":
            state = "loose"
        else:
            state = "unknown"
        frames.append({
            "frame_index": frame_index,
            "state": state,
            "holder_id": holder,
            "candidate_id": evidence.get("candidate_id"),
            "confidence": round(float(evidence.get("confidence") or 0.0), 4),
            "support_frames": run_support,
            "reason": evidence.get("reason", "acquisition_timeline"),
            "ball_observed": _is_observed_ball({1: ball}) if ball else False,
            "ball_confidence": evidence.get(
                "ball_confidence",
                ball.get("confidence"),
            ),
            "ball_source": ball.get(
                "position_source",
                ball.get("detection_source"),
            ),
        })
    return frames


def _holder_segments(acquisitions, discontinuities):
    segments = []
    current = None
    segment_id = 0
    segment_start = 0
    for frame_index, holder_id in enumerate(acquisitions):
        if frame_index in discontinuities:
            current = None
            segment_id += 1
            segment_start = frame_index
        if holder_id == -1:
            continue
        if current is None or current["holder_id"] != holder_id:
            current = {
                "holder_id": int(holder_id),
                "start_frame": frame_index,
                "last_frame": frame_index,
                "support_frames": 1,
                "segment": segment_id,
                "segment_start": segment_start,
            }
            segments.append(current)
        else:
            current["last_frame"] = frame_index
            current["support_frames"] += 1
    return segments


def _stable_catch_frame(
    acquisitions,
    transition_frame,
    holder_id,
    *,
    minimum_frames,
    confirmation_frames,
    discontinuities,
):
    end = min(
        len(acquisitions),
        transition_frame + confirmation_frames + 1,
    )
    run_start = None
    run_frames = 0
    for frame_index in range(transition_frame, end):
        if frame_index != transition_frame and frame_index in discontinuities:
            return None
        candidate = acquisitions[frame_index]
        if candidate == holder_id:
            if run_start is None:
                run_start = frame_index
            run_frames += 1
            if run_frames >= minimum_frames:
                return run_start
        elif candidate == -1:
            run_start = None
            run_frames = 0
        else:
            return None
    return None


def _controlled_run_length(acquisitions, start_frame, holder_id, discontinuities):
    support = 0
    for frame_index in range(start_frame, len(acquisitions)):
        if frame_index != start_frame and frame_index in discontinuities:
            break
        if acquisitions[frame_index] != holder_id:
            break
        support += 1
    return support


def _mark_release_and_flight_frames(frames, transitions, ball_tracks):
    for transition in transitions:
        if transition["status"] != "confirmed":
            continue
        release = transition["release_frame"]
        catch = transition["catch_frame"]
        for frame_index in range(release + 1, catch):
            frame = frames[frame_index]
            if frame.get("holder_id") is not None:
                continue
            if frame_index == release + 1:
                frame["state"] = "released"
            elif ball_tracks is not None and ball_tracks[frame_index].get(1, {}).get("bbox"):
                frame["state"] = "in_flight"
            else:
                frame["state"] = "loose"
            frame["source_holder_id"] = transition["from_player_id"]
            frame["receiver_id"] = transition["to_player_id"]


def _is_observed_ball(frame_ball):
    ball = frame_ball.get(1, {})
    return bool(ball.get("bbox")) and not bool(ball.get("interpolated", False))


def _normalize_discontinuities(discontinuity_frames, frame_count):
    return {
        int(frame)
        for frame in (discontinuity_frames or [])
        if 0 < int(frame) < frame_count
    }
