"""Deterministic, provider-independent Trusted Evidence Package v2.

Only allowlisted observations enter this package. Missing measurements are absent,
not zero. No detector currently satisfies a verification gate: events stay candidates.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import pairwise

VERSION = "2.0"
MIN_SUPPORT = 3
MIN_OBSERVATION_SECONDS = 0.5
MAX_SPEED_MPS = 12.0
SUSTAINED_SECONDS = 0.5


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value):
    return type(value) is int and value >= 0


def team(value):
    return value if type(value) is int and value in (1, 2) else None


def track(value):
    # Raw strings could contain names or prompt injection. Pipeline IDs are integers.
    return f"track-{value}" if integer(value) else None


def ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def stable_id(prefix, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return prefix + "-" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def obj(required, optional=None):
    return {
        "type": "object",
        "properties": {**required, **(optional or {})},
        "required": list(required),
        "additionalProperties": False,
    }


def array(items):
    return {"type": "array", "items": items}


N = {"type": "number", "minimum": 0}
I = {"type": "integer", "minimum": 0}
S = {"type": "string"}
R = {"type": ["number", "null"], "minimum": 0, "maximum": 1}
T = {"enum": [None, 1, 2]}
NS = {"type": ["string", "null"]}
COVERAGE = obj(
    {
        "id": S,
        "durationSeconds": N,
        "frameCount": I,
        "sampledFrameCount": I,
        "usableFrameCount": I,
        "usableFrameRatio": R,
        "knownPossessionFrames": I,
        "unknownPossessionFrames": I,
        "calibratedFrameCount": I,
        "calibrationCoverage": R,
        "knownTeamObservations": I,
        "unknownTeamObservations": I,
        "knownTeamRatio": R,
        "omittedEventCount": I,
    }
)
SEGMENT = obj(
    {
        "id": S,
        "trackId": S,
        "teamId": T,
        "startFrame": I,
        "endFrame": I,
        "timeSeconds": N,
        "endSeconds": N,
        "supportFrames": I,
    }
)
TRACK = obj(
    {
        "id": S,
        "trackId": S,
        "teamId": T,
        "observedFrames": I,
        "observationSeconds": N,
        "continuityRatio": R,
        "continuousRuns": I,
        "calibratedFrames": I,
        "measurementQuality": S,
        "uncertaintyReasons": array(S),
    },
    {
        "distanceMeters": N,
        "measuredSeconds": N,
        "averageSpeedMps": N,
        "p95SpeedMps": N,
        "sustainedPeakSpeedMps": N,
        "peakTimeSeconds": N,
    },
)
EVENT = obj(
    {
        "id": S,
        "type": {"enum": ["pass", "interception", "shot_attempt"]},
        "status": {"enum": ["candidate"]},
        "timeSeconds": N,
        "frameIndex": I,
        "replayStartSeconds": N,
        "replayEndSeconds": N,
        "fromTrackId": NS,
        "toTrackId": NS,
        "fromTeamId": T,
        "toTeamId": T,
        "supportFrames": I,
        "uncertaintyReasons": array(S),
    },
    {"releaseFrame": I, "catchFrame": I},
)
SCHEMA = obj(
    {
        "schemaVersion": {"enum": [VERSION]},
        "coverage": COVERAGE,
        "possession": obj(
            {
                "id": S,
                "denominator": {"enum": ["sampled_frames_with_possession_diagnostics"]},
                "eligibleFrames": I,
                "team1Frames": I,
                "team2Frames": I,
                "unknownFrames": I,
                "team1Share": R,
                "team2Share": R,
                "unknownShare": R,
                "segments": array(SEGMENT),
            }
        ),
        "tracks": array(TRACK),
        "events": array(EVENT),
        "sequences": array(
            obj(
                {
                    "id": S,
                    "evidenceIds": array(S),
                    "relation": {"enum": ["temporal_adjacency_only"]},
                }
            )
        ),
        "provenance": obj(
            {
                "fps": N,
                "algorithms": obj(
                    {"evidence": S, "possession": S, "movement": S, "events": S},
                    {"review": S},
                ),
                "units": obj({"time": S, "distance": S, "speed": S}),
                "thresholds": obj(
                    {
                        "minimumSupportFrames": I,
                        "maximumEvents": I,
                        "minimumObservationSeconds": N,
                        "maximumSpeedMps": N,
                        "sustainedWindowSeconds": N,
                    }
                ),
                "calibrationStatus": S,
                "verificationGate": S,
            }
        ),
        "limitations": array(S),
        "summaryOptions": array(S),
        "supportedClaims": array(obj({"claim": S, "evidenceIds": array(S)})),
    }
)


def validate_structure(value, schema):
    """Strict dependency-free validator for the JSON Schema subset above."""
    if "enum" in schema:
        if not any(type(value) is type(v) and value == v for v in schema["enum"]):
            raise ValueError("Invalid enum")
        return
    kinds = schema.get("type")
    kinds = kinds if isinstance(kinds, list) else [kinds]
    matches = {
        "object": type(value) is dict,
        "array": type(value) is list,
        "string": type(value) is str,
        "number": number(value),
        "integer": integer(value),
        "null": value is None,
    }
    if not any(matches.get(kind, False) for kind in kinds):
        raise ValueError("Invalid evidence field type")
    if number(value) and (
        value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf)
    ):
        raise ValueError("Evidence out of range")
    if isinstance(value, dict):
        if set(value) - set(schema["properties"]) or set(schema["required"]) - set(
            value
        ):
            raise ValueError("Invalid evidence fields")
        for key, child in value.items():
            validate_structure(child, schema["properties"][key])
    if isinstance(value, list):
        for child in value:
            validate_structure(child, schema["items"])


def evidence_facts(package):
    return [
        package["coverage"],
        package["possession"],
        *package["possession"]["segments"],
        *package["tracks"],
        *package["events"],
        *package["sequences"],
    ]


def validate_trusted_evidence_v2(package):
    validate_structure(package, SCHEMA)
    coverage, possession = package["coverage"], package["possession"]
    duration, count = coverage["durationSeconds"], coverage["frameCount"]
    if duration <= 0 or package["provenance"]["fps"] <= 0:
        raise ValueError("Invalid clip dimensions")
    facts = evidence_facts(package)
    ids = {fact["id"] for fact in facts}
    if len(ids) != len(facts):
        raise ValueError("Duplicate evidence IDs")
    tracks = {p["trackId"]: p for p in package["tracks"]}
    for fact in facts:
        for key, value in fact.items():
            if (
                key
                in {
                    "timeSeconds",
                    "endSeconds",
                    "peakTimeSeconds",
                    "replayStartSeconds",
                    "replayEndSeconds",
                }
                and not 0 <= value <= duration
            ):
                raise ValueError("Timestamp outside clip")
            if (
                key
                in {
                    "frameIndex",
                    "releaseFrame",
                    "catchFrame",
                    "startFrame",
                    "endFrame",
                }
                and value >= count
            ):
                raise ValueError("Frame outside clip")
            if (
                key in {"trackId", "fromTrackId", "toTrackId"}
                and value is not None
                and value not in tracks
            ):
                raise ValueError("Missing track reference")
        if "evidenceIds" in fact and (
            not fact["evidenceIds"]
            or any(ref not in ids or ref == fact["id"] for ref in fact["evidenceIds"])
        ):
            raise ValueError("Missing evidence reference")
    eligible = possession["eligibleFrames"]
    if (
        eligible
        != possession["team1Frames"]
        + possession["team2Frames"]
        + possession["unknownFrames"]
    ):
        raise ValueError("Possession denominator mismatch")
    for key in ("team1", "team2", "unknown"):
        if possession[key + "Share"] != ratio(possession[key + "Frames"], eligible):
            raise ValueError("Possession share mismatch")
    if not eligible <= coverage["sampledFrameCount"] <= count:
        raise ValueError("Coverage exceeds clip")
    if coverage["knownPossessionFrames"] + coverage["unknownPossessionFrames"] != count:
        raise ValueError("Possession coverage mismatch")
    for p in package["tracks"]:
        for key in ("averageSpeedMps", "p95SpeedMps", "sustainedPeakSpeedMps"):
            if key in p and p[key] > MAX_SPEED_MPS:
                raise ValueError("Implausible speed")
        if "distanceMeters" in p and (
            p["observationSeconds"] < MIN_OBSERVATION_SECONDS
            or p.get("measuredSeconds", 0) < MIN_OBSERVATION_SECONDS
        ):
            raise ValueError("Insufficient movement support")
    for segment in possession["segments"]:
        if (
            segment["supportFrames"] < MIN_SUPPORT
            or segment["startFrame"] > segment["endFrame"]
        ):
            raise ValueError("Invalid possession support")
    for event in package["events"]:
        if event["fromTeamId"] is not None and event["toTeamId"] is not None:
            same = event["fromTeamId"] == event["toTeamId"]
            if (event["type"] == "pass" and not same) or (
                event["type"] == "interception" and same
            ):
                raise ValueError("Contradictory event teams")
        if (
            abs(
                event["frameIndex"] / package["provenance"]["fps"]
                - event["timeSeconds"]
            )
            > 0.002
        ):
            raise ValueError("Event timestamp mismatch")
        if (
            not event["replayStartSeconds"]
            <= event["timeSeconds"]
            <= event["replayEndSeconds"]
        ):
            raise ValueError("Invalid replay window")
        if event.get("releaseFrame", 0) > event.get("catchFrame", count):
            raise ValueError("Contradictory event boundaries")
        if (
            event["supportFrames"] < MIN_SUPPORT
            and "insufficient_support" not in event["uncertaintyReasons"]
        ):
            raise ValueError("Missing support limitation")
    for i, left in enumerate(package["events"]):
        for right in package["events"][i + 1 :]:
            if (
                left["fromTrackId"]
                and left.get("releaseFrame") is not None
                and (left["fromTrackId"], left["releaseFrame"])
                == (right["fromTrackId"], right.get("releaseFrame"))
                and (left["type"], left["toTrackId"])
                != (right["type"], right["toTrackId"])
            ):
                raise ValueError("Conflicting release events")
    event_map = {event["id"]: event for event in package["events"]}
    for sequence in package["sequences"]:
        linked = [event_map[ref] for ref in sequence["evidenceIds"] if ref in event_map]
        if len(linked) > 1:
            segment = _pass_shot_segment(linked[0], linked[1], possession["segments"])
            if len(linked) != 2 or segment is None or sequence["evidenceIds"] != [linked[0]["id"], segment["id"], linked[1]["id"]]:
                raise ValueError("Unsupported play sequence")
    for claim in package["supportedClaims"]:
        if not claim["evidenceIds"] or any(
            ref not in ids for ref in claim["evidenceIds"]
        ):
            raise ValueError("Invalid supported claim reference")
    for key, numerator in (
        ("usableFrameRatio", "usableFrameCount"),
        ("calibrationCoverage", "calibratedFrameCount"),
    ):
        if coverage[numerator] > count or coverage[key] != ratio(
            coverage[numerator], count
        ):
            raise ValueError("Invalid coverage ratio")
    if coverage["knownTeamRatio"] != ratio(
        coverage["knownTeamObservations"],
        coverage["knownTeamObservations"] + coverage["unknownTeamObservations"],
    ):
        raise ValueError("Invalid team coverage")
    return package


def build_trusted_evidence_v2(analysis):
    source = analysis.get("source", {})
    duration, fps = source.get("durationSeconds"), source.get("fps")
    if not number(duration) or duration <= 0 or not number(fps) or fps <= 0:
        raise ValueError("Positive duration and FPS required")
    count = source.get("frameCount", round(duration * fps))
    if (
        not integer(count)
        or count < 1
        or abs(count / fps - duration) > max(0.002, 1 / fps)
    ):
        raise ValueError("Inconsistent source frame count")
    frames = analysis.get("evidenceFrames", analysis.get("frames", []))
    if not isinstance(frames, list):
        raise ValueError("Frames must be a list")  # noqa: TRY004 - uniform validation API
    indexed = {}
    for frame in frames:
        index = frame.get("frameIndex") if isinstance(frame, dict) else None
        if not integer(index) or index >= count or index in indexed:
            raise ValueError("Invalid or duplicate frame index")
        indexed[index] = frame
    diagnostics = analysis.get("diagnostics", {})
    timeline = diagnostics.get("possessionTimeline", {})
    timeline = timeline.get("semantic", timeline)
    holder_frames = {
        f["frame_index"]: f
        for f in timeline.get("frames", [])
        if integer(f.get("frame_index"))
    }
    cuts = set(analysis.get("measurements", {}).get("discontinuityFrames", []))
    observations = defaultdict(list)
    known_teams = unknown_teams = calibrated = usable = 0
    possession_counts = Counter()
    runs, current = [], None
    for index, frame in sorted(indexed.items()):
        players = frame.get("players", [])
        seen = set()
        for p in players:
            pid = track(p.get("id"))
            if pid is None or pid in seen:
                raise ValueError("Invalid or duplicate opaque track ID")
            seen.add(pid)
            observations[pid].append((index, p, frame.get("courtCalibrated") is True))
            known_teams += team(p.get("teamId")) is not None
            unknown_teams += team(p.get("teamId")) is None
        usable += bool(players)
        calibrated += frame.get("courtCalibrated") is True
        diagnostic = holder_frames.get(index)
        # Explicit diagnostics include unknown/loose frames in the denominator.
        eligible = diagnostic is not None
        provisional = {
            "brief_ball_gap",
            "brief_interpolated_gap",
            "retrospective_holder_confirmation",
            "same_holder_gap_bridged",
            "switch_pending",
        }
        holder = (
            track(diagnostic.get("holder_id"))
            if eligible
            and diagnostic.get("state") == "controlled"
            and diagnostic.get("reason") not in provisional
            else None
        )
        player = next((p for p in players if track(p.get("id")) == holder), None)
        holder_team = team(player.get("teamId")) if player else None
        if eligible:
            possession_counts[holder_team] += 1
        if holder and player and eligible:
            if (
                current
                and current["trackId"] == holder
                and current["teamId"] == holder_team
                and current["endFrame"] + 1 == index
                and index not in cuts
            ):
                current["endFrame"] = index
                current["supportFrames"] += 1
            else:
                current = {
                    "trackId": holder,
                    "teamId": holder_team,
                    "startFrame": index,
                    "endFrame": index,
                    "supportFrames": 1,
                }
                runs.append(current)
        else:
            current = None
    segments = []
    for run in runs:
        if run["supportFrames"] >= MIN_SUPPORT:
            segments.append(
                {
                    "id": stable_id("possession", run),
                    **run,
                    "timeSeconds": run["startFrame"] / fps,
                    "endSeconds": min(duration, (run["endFrame"] + 1) / fps),
                }
            )
    tracks = []
    for pid, rows in sorted(observations.items()):
        assigned = {team(p.get("teamId")) for _, p, _ in rows}
        run_count = sum(
            i == 0 or index != rows[i - 1][0] + 1 or index in cuts
            for i, (index, _, _) in enumerate(rows)
        )
        p = {
            "id": pid,
            "trackId": pid,
            "teamId": next(iter(assigned)) if len(assigned) == 1 else None,
            "observedFrames": len(rows),
            "observationSeconds": len(rows) / fps,
            "continuityRatio": ratio(len(rows), rows[-1][0] - rows[0][0] + 1),
            "continuousRuns": run_count,
            "calibratedFrames": sum(c for _, _, c in rows),
            "measurementQuality": "unavailable",
            "uncertaintyReasons": [],
        }
        if run_count > 1:
            p["uncertaintyReasons"].append("tracking_discontinuity")
        if len(assigned) > 1:
            p["uncertaintyReasons"].append("team_assignment_inconsistent")
        samples, window, peaks = [], [], []
        size = max(1, math.ceil(fps * SUSTAINED_SECONDS))
        for i, (index, player, calibration) in enumerate(rows):
            distance = player.get("distanceMeters")
            valid = (
                i > 0
                and index == rows[i - 1][0] + 1
                and index not in cuts
                and calibration
                and rows[i - 1][2]
                and number(distance)
                and 0 <= distance * fps <= MAX_SPEED_MPS
            )
            if not valid:
                window = []
                if number(distance) and distance * fps > MAX_SPEED_MPS:
                    p["uncertaintyReasons"].append("speed_outlier_filtered")
                continue
            samples.append(distance)
            window.append(distance * fps)
            if len(window) >= size:
                peaks.append((sum(window[-size:]) / size, index / fps))
        if len(samples) / fps >= MIN_OBSERVATION_SECONDS:
            speeds = sorted(d * fps for d in samples)
            p.update(
                distanceMeters=sum(samples),
                measuredSeconds=len(samples) / fps,
                averageSpeedMps=sum(speeds) / len(speeds),
                p95SpeedMps=speeds[math.ceil(len(speeds) * 0.95) - 1],
                measurementQuality="calibrated_observed_intervals",
            )
            if peaks:
                peak = max(peaks, key=lambda x: x[0])
                p.update(sustainedPeakSpeedMps=peak[0], peakTimeSeconds=peak[1])
        else:
            p["uncertaintyReasons"].append("insufficient_calibrated_measurements")
        p["uncertaintyReasons"] = sorted(set(p["uncertaintyReasons"]))
        tracks.append(p)
    known_tracks = {p["trackId"] for p in tracks}
    track_teams = {p["trackId"]: p["teamId"] for p in tracks}
    events, omitted = [], 0
    raw_events = analysis.get("events", [])
    if not isinstance(raw_events, list):
        raise ValueError("Events must be a list")  # noqa: TRY004 - uniform validation API
    for raw in raw_events:
        t = raw.get("timeSeconds")
        if (
            raw.get("type") not in {"pass", "interception", "shot_attempt"}
            or not number(t)
            or not 0 <= t < duration
        ):
            omitted += 1
            continue
        frame = raw.get("frameIndex", round(t * fps))
        if not integer(frame) or frame >= count or abs(frame / fps - t) > 0.002:
            omitted += 1
            continue
        detail = raw.get("evidence", {})
        reasons = ["verification_gate_not_available"]
        event = {
            "type": raw["type"],
            "status": "candidate",
            "timeSeconds": t,
            "frameIndex": frame,
            "replayStartSeconds": max(0, t - 2),
            "replayEndSeconds": min(duration, t + 2),
            "fromTeamId": team(raw.get("fromTeamId")),
            "toTeamId": team(raw.get("toTeamId")),
        }
        for side in ("from", "to"):
            pid = track(detail.get(side + "_player_id"))
            event[side + "TrackId"] = pid if pid in known_tracks else None
            if pid not in known_tracks:
                reasons.append("missing_track_reference")
        if any(
            event[side + "TrackId"] in track_teams
            and track_teams[event[side + "TrackId"]] is not None
            and event[side + "TeamId"] is not None
            and track_teams[event[side + "TrackId"]] != event[side + "TeamId"]
            for side in ("from", "to")
        ):
            omitted += 1
            continue
        for source_key, target in (
            ("release_frame", "releaseFrame"),
            ("catch_frame", "catchFrame"),
        ):
            value = detail.get(source_key)
            if integer(value) and value < count:
                event[target] = value
        if event.get("releaseFrame", 0) > event.get("catchFrame", count):
            omitted += 1
            continue
        support = detail.get("possession_evidence", {})
        supports = [
            support.get("source_support_frames"),
            support.get("receiver_support_frames"),
        ]
        event["supportFrames"] = (
            min(supports) if all(integer(s) and s <= count for s in supports) else 0
        )
        if event["supportFrames"] < MIN_SUPPORT:
            reasons.append("insufficient_support")
        if event["fromTeamId"] is None or event["toTeamId"] is None:
            reasons.append("unknown_team_assignment")
        if (
            event["fromTeamId"]
            and event["toTeamId"]
            and (
                (event["type"] == "pass" and event["fromTeamId"] != event["toTeamId"])
                or (
                    event["type"] == "interception"
                    and event["fromTeamId"] == event["toTeamId"]
                )
            )
        ):
            omitted += 1
            continue
        event["uncertaintyReasons"] = sorted(set(reasons))
        event["id"] = stable_id("event", event)
        events.append(event)
    events = list({e["id"]: e for e in events}.values())
    # Conflicting descriptions of the same release cannot both be trusted.
    conflicts = set()
    for i, left in enumerate(events):
        for right in events[i + 1 :]:
            if (
                left.get("fromTrackId")
                and left.get("releaseFrame") is not None
                and (left["fromTrackId"], left["releaseFrame"])
                == (right["fromTrackId"], right.get("releaseFrame"))
                and (left["type"], left["toTrackId"])
                != (right["type"], right["toTrackId"])
            ):
                conflicts.update((left["id"], right["id"]))
    omitted += len(conflicts)
    events = sorted(
        (e for e in events if e["id"] not in conflicts),
        key=lambda e: (e["timeSeconds"], e["id"]),
    )
    events = events[:200]
    omitted = len(raw_events) - len(events)
    sequences = []
    for event in events:
        linked = [
            s["id"]
            for s in segments
            if (
                s["trackId"] == event["fromTrackId"]
                and s["startFrame"] <= event.get("releaseFrame", -1) <= s["endFrame"]
            )
            or (
                s["trackId"] == event["toTrackId"]
                and s["startFrame"] <= event.get("catchFrame", -1) <= s["endFrame"]
            )
        ]
        if linked:
            refs = linked + [event["id"]]
            sequences.append(
                {
                    "id": stable_id("sequence", refs),
                    "evidenceIds": refs,
                    "relation": "temporal_adjacency_only",
                }
            )
    eligible = sum(possession_counts.values())
    # Short or provisional control runs remain unknown, even with a holder label.
    possession_counts[1] = sum(s["supportFrames"] for s in segments if s["teamId"] == 1)
    possession_counts[2] = sum(s["supportFrames"] for s in segments if s["teamId"] == 2)
    known = possession_counts[1] + possession_counts[2]
    possession_counts[None] = eligible - known
    package = {
        "schemaVersion": VERSION,
        "coverage": {
            "id": "coverage",
            "durationSeconds": duration,
            "frameCount": count,
            "sampledFrameCount": len(indexed),
            "usableFrameCount": usable,
            "usableFrameRatio": ratio(usable, count),
            "knownPossessionFrames": known,
            "unknownPossessionFrames": count - known,
            "calibratedFrameCount": calibrated,
            "calibrationCoverage": ratio(calibrated, count),
            "knownTeamObservations": known_teams,
            "unknownTeamObservations": unknown_teams,
            "knownTeamRatio": ratio(known_teams, known_teams + unknown_teams),
            "omittedEventCount": omitted,
        },
        "possession": {
            "id": "possession",
            "denominator": "sampled_frames_with_possession_diagnostics",
            "eligibleFrames": eligible,
            "team1Frames": possession_counts[1],
            "team2Frames": possession_counts[2],
            "unknownFrames": possession_counts[None],
            "team1Share": ratio(possession_counts[1], eligible),
            "team2Share": ratio(possession_counts[2], eligible),
            "unknownShare": ratio(possession_counts[None], eligible),
            "segments": segments,
        },
        "tracks": tracks,
        "events": events,
        "sequences": sequences,
        "provenance": {
            "fps": fps,
            "algorithms": {
                "evidence": VERSION,
                "possession": "semantic_contiguous_v2",
                "movement": "pipeline_distance_filtered_v2",
                "events": "candidate_allowlist_v2",
            },
            "units": {
                "time": "seconds",
                "distance": "meters",
                "speed": "meters_per_second",
            },
            "thresholds": {
                "minimumSupportFrames": MIN_SUPPORT,
                "maximumEvents": 200,
                "minimumObservationSeconds": MIN_OBSERVATION_SECONDS,
                "maximumSpeedMps": MAX_SPEED_MPS,
                "sustainedWindowSeconds": SUSTAINED_SECONDS,
            },
            "calibrationStatus": "full"
            if calibrated == count
            else "partial"
            if calibrated
            else "unavailable",
            "verificationGate": "none_available_candidates_only",
        },
        "limitations": [
            "Track IDs are opaque observations, not player identities.",
            "Events are candidates; scores, outcomes, schemes and causes are unsupported.",
            "Movement covers accepted calibrated intervals only; court zones are omitted without validated orientation.",
            "Possession shares use eligible diagnostic frames, including unknowns; missing frames remain unknown coverage.",
        ],
    }
    if omitted:
        package["limitations"].append(
            f"{omitted} raw event entries were omitted by validation, deduplication, contradiction checks or the 200-event limit."
        )
    if conflicts:
        package["limitations"].append(
            "Conflicting descriptions of the same release were excluded."
        )
    _add_replay_review(package)
    return validate_trusted_evidence_v2(package)


MAX_SEQUENCE_GAP_SECONDS = 5.0


def _pass_shot_segment(first, second, segments):
    """Link candidates only across continuous control by the receiving track."""
    if (first["type"] != "pass" or second["type"] != "shot_attempt"
            or first["supportFrames"] < MIN_SUPPORT
            or not first["toTrackId"] or first["fromTrackId"] == first["toTrackId"]
            or first["toTrackId"] != second["fromTrackId"]
            or first["toTeamId"] is None or first["toTeamId"] != second["fromTeamId"]
            or not 0 < second["timeSeconds"] - first["timeSeconds"] <= MAX_SEQUENCE_GAP_SECONDS):
        return None
    catch, release = first.get("catchFrame"), second.get("releaseFrame")
    if catch is None or release is None or catch > release:
        return None
    return next((segment for segment in segments
        if segment["trackId"] == first["toTrackId"]
        and segment["teamId"] == first["toTeamId"]
        and segment["startFrame"] <= catch <= release <= segment["endFrame"]
        and segment["supportFrames"] == segment["endFrame"] - segment["startFrame"] + 1), None)


def _actor(track_id, team_id):
    if track_id is None:
        return f"Team {team_id}" if team_id is not None else None
    label = "T" + track_id.removeprefix("track-")
    return f"Team {team_id} track {label}" if team_id is not None else f"track {label} (team unknown)"


def _event_review(event):
    label = {"shot_attempt": "shot-attempt", "pass": "pass", "interception": "interception"}[event["type"]]
    actor = _actor(event["fromTrackId"], event["fromTeamId"])
    destination = _actor(event["toTrackId"], event["toTeamId"])
    context = f", associated with {actor}" if actor else ""
    if event["type"] in {"pass", "interception"} and actor and destination and event["fromTrackId"] != event["toTrackId"]:
        context = f", from {actor} to {destination}"
    headline = f"Review the {label} candidate near {event['timeSeconds']:.1f}s{context}."
    action = "buildup and release" if event["type"] == "shot_attempt" else "release and possible catch"
    detail = f"Start the replay at {event['replayStartSeconds']:.1f}s to inspect the {action}."
    if event["type"] == "shot_attempt":
        detail += " The shot outcome is not established."
    return headline, headline + " " + detail


def _add_replay_review(package):
    """Author replay-first explanations deterministically, before model selection."""
    package["provenance"]["algorithms"]["review"] = "replay_review_v1_gap_5s"
    events = package["events"]
    segments = package["possession"]["segments"]
    claims = []
    linked_events = set()
    # Adjacent candidates avoid silently skipping a conflicting intervening action.
    for first, second in pairwise(events):
        segment = _pass_shot_segment(first, second, segments)
        if segment is None:
            continue
        refs = [first["id"], segment["id"], second["id"]]
        sequence = {"id": stable_id("sequence", refs), "evidenceIds": refs,
                    "relation": "temporal_adjacency_only"}
        package["sequences"].append(sequence)
        receiver = _actor(first["toTrackId"], first["toTeamId"])
        headline = (f"A pass candidate near {first['timeSeconds']:.1f}s is followed by a "
                    f"shot-attempt candidate near {second['timeSeconds']:.1f}s involving {receiver}.")
        detail = (f"Replay from {first['replayStartSeconds']:.1f}s to inspect the possible catch and later release. "
                  "The actions remain candidates; the shot outcome is not established.")
        claims.append({"claim": headline + " " + detail, "evidenceIds": [sequence["id"]]})
        linked_events.update((first["id"], second["id"]))
    for event in events:
        if event["id"] in linked_events:
            continue
        _, claim = _event_review(event)
        claims.append({"claim": claim, "evidenceIds": [event["id"]]})
    fallback = "The available evidence does not support a clear play-by-play summary. Review the clip directly."
    if not claims:
        claims = [{"claim": fallback, "evidenceIds": ["coverage"]}]
    # The overall sentence introduces the review; key moments carry the detail.
    if linked_events:
        overview = "The review includes a pass candidate followed by a shot-attempt candidate involving the same receiving track. Replay is needed to assess both actions."
    elif len(events) == 1:
        label = events[0]["type"].replace("_", "-")
        article = "an" if label == "interception" else "a"
        overview = f"The main review moment is {article} {label} candidate. "
        overview += "Its outcome is not established." if events[0]["type"] == "shot_attempt" else "Replay is needed to assess the possible transfer."
    elif events:
        overview = "Use the replay cues to inspect the detected action candidates. Their outcomes remain unverified."
    else:
        overview = fallback
    package["summaryOptions"] = [overview]
    package["supportedClaims"] = claims
    package["limitations"] = [
        "These are possible actions to review, not verified plays or shot outcomes.",
        "Track labels help locate the action; they are not player names or jersey numbers.",
    ] + (["Ball control is uncertain for part of the clip, so possession conclusions are limited."]
         if package["coverage"]["unknownPossessionFrames"] else []) + ([
        "Some court positions are uncertain; movement estimates cover only accepted measurements."
    ] if package["provenance"]["calibrationStatus"] != "full" else []) + ([
        "Some event entries were excluded because they were unsupported, duplicated, conflicting, or beyond the review limit."
    ] if package["coverage"]["omittedEventCount"] else [])
