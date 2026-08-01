import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = (
    ROOT / "benchmarks" / "courtvision_nba_holdout_v1" / "calibration"
)
BALL_VISIBILITY = {"visible", "occluded", "out_of_frame", "uncertain"}
POSSESSION_STATES = {
    "controlled", "loose", "in_flight", "shot", "dead", "unknown",
}
CONFIDENCE = {"low", "medium", "high"}
REVIEW_STATUS = {"draft", "verified", "rejected"}
EVENT_TYPES = {
    "pass",
    "inbound_pass",
    "shot",
    "rebound",
    "offensive_rebound",
    "defensive_rebound",
    "steal",
    "interception",
    "turnover",
    "deflection",
    "dead_ball",
    "camera_cut",
}


def _records(path):
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            yield line_number, json.loads(line)


def _scene_index(manifest):
    clips = {}
    scenes = {}
    for clip in manifest.get("clips", []):
        clip_id = clip.get("id")
        if clip_id in clips:
            raise ValueError(f"duplicate clip id {clip_id}")
        clips[clip_id] = clip
        expected_start = 0
        for scene in clip.get("scenes", []):
            scene_id = scene.get("id")
            if scene_id in scenes:
                raise ValueError(f"duplicate scene id {scene_id}")
            start = scene.get("start_frame")
            end = scene.get("end_frame")
            if start != expected_start or not isinstance(end, int) or end < start:
                raise ValueError(f"{clip_id}: scenes are not contiguous at {scene_id}")
            if end >= clip["frame_count"]:
                raise ValueError(f"{scene_id}: scene exceeds clip")
            scenes[scene_id] = (clip, scene)
            expected_start = end + 1
        if expected_start != clip["frame_count"]:
            raise ValueError(f"{clip_id}: scenes do not cover the clip")
    return clips, scenes


def _validate_frame_record(line, record, clips, scenes, seen):
    prefix = f"frames:{line}"
    clip = clips.get(record.get("video_id"))
    if clip is None:
        raise ValueError(f"{prefix}: unknown video")
    frame_index = record.get("frame_index")
    key = (clip["id"], frame_index)
    if key in seen:
        raise ValueError(f"{prefix}: duplicate frame {key}")
    seen.add(key)
    if not isinstance(frame_index, int) or not 0 <= frame_index < clip["frame_count"]:
        raise ValueError(f"{prefix}: invalid frame_index")
    scene_entry = scenes.get(record.get("scene_id"))
    if scene_entry is None or scene_entry[0]["id"] != clip["id"]:
        raise ValueError(f"{prefix}: invalid scene")
    scene = scene_entry[1]
    if not scene["start_frame"] <= frame_index <= scene["end_frame"]:
        raise ValueError(f"{prefix}: frame outside scene")

    ball = record.get("ball", {})
    visibility = ball.get("visibility")
    center = ball.get("center_px")
    if visibility not in BALL_VISIBILITY:
        raise ValueError(f"{prefix}: invalid ball visibility")
    if ball.get("confidence") not in CONFIDENCE:
        raise ValueError(f"{prefix}: invalid ball confidence")
    if visibility == "visible":
        if (
            not isinstance(center, list)
            or len(center) != 2
            or not all(isinstance(value, (int, float)) for value in center)
            or not 0 <= center[0] < clip["width"]
            or not 0 <= center[1] < clip["height"]
        ):
            raise ValueError(f"{prefix}: invalid visible ball center")
    elif center is not None:
        raise ValueError(f"{prefix}: non-visible ball cannot have center")

    possession = record.get("possession", {})
    state = possession.get("state")
    team_id = possession.get("team_id")
    holder = possession.get("holder")
    if state not in POSSESSION_STATES:
        raise ValueError(f"{prefix}: invalid possession state")
    scene_teams = set(scene.get("teams", {}))
    if state == "controlled":
        if team_id not in scene_teams or not isinstance(holder, str) or not holder:
            raise ValueError(
                f"{prefix}: controlled possession needs a scene-local team and holder"
            )
    elif team_id is not None or holder is not None:
        raise ValueError(f"{prefix}: non-controlled possession cannot claim team/holder")
    if record.get("review_status") not in REVIEW_STATUS - {"rejected"}:
        raise ValueError(f"{prefix}: invalid review status")


def _validate_event_record(line, event, clips, scenes):
    prefix = f"events:{line}"
    clip = clips.get(event.get("video_id"))
    if clip is None:
        raise ValueError(f"{prefix}: unknown video")
    scene_entry = scenes.get(event.get("scene_id"))
    if scene_entry is None or scene_entry[0]["id"] != clip["id"]:
        raise ValueError(f"{prefix}: invalid scene")
    scene = scene_entry[1]
    event_type = event.get("event_type")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"{prefix}: invalid event type")
    start = event.get("start_frame")
    end = event.get("end_frame")
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or start > end
        or start < scene["start_frame"]
        or end > scene["end_frame"]
    ):
        raise ValueError(f"{prefix}: invalid event range")
    for field in ("release_frame", "catch_frame"):
        value = event.get(field)
        if value is not None and (
            not isinstance(value, int)
            or value < start
            or value > end
        ):
            raise ValueError(f"{prefix}: invalid {field}")
    teams = set(scene.get("teams", {}))
    for field in ("from_team_id", "to_team_id"):
        if event.get(field) not in teams | {None}:
            raise ValueError(f"{prefix}: {field} is not scene-local")
    if event.get("confidence") not in CONFIDENCE:
        raise ValueError(f"{prefix}: invalid confidence")
    if event.get("review_status") not in REVIEW_STATUS:
        raise ValueError(f"{prefix}: invalid review status")

    release = event.get("release_frame")
    catch = event.get("catch_frame")
    if event_type in {"pass", "inbound_pass"}:
        if release is None or catch is None or release > catch:
            raise ValueError(f"{prefix}: pass needs ordered release/catch")
    elif event_type == "shot":
        if release is None or catch is not None or event.get("to_team_id") is not None:
            raise ValueError(f"{prefix}: invalid shot endpoints")
    elif event_type in {
        "rebound", "offensive_rebound", "defensive_rebound",
    }:
        if release is not None or catch is None or event.get("from_team_id") is not None:
            raise ValueError(f"{prefix}: invalid rebound endpoints")
    elif event_type == "camera_cut":
        if start != end or release is not None or catch is not None:
            raise ValueError(f"{prefix}: camera cut must be a point event")


def validate(calibration_dir=DEFAULT_CALIBRATION):
    calibration_dir = Path(calibration_dir)
    manifest = json.loads(
        (calibration_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("courtvision_predictions_used") is not False:
        raise ValueError("manifest must explicitly state that predictions were not used")
    clips, scenes = _scene_index(manifest)
    seen = set()
    counts = Counter()
    for line, record in _records(calibration_dir / "frames.jsonl"):
        _validate_frame_record(line, record, clips, scenes, seen)
        counts[f"frame_{record['review_status']}"] += 1
        counts[f"ball_{record['ball']['visibility']}"] += 1
    for clip in clips.values():
        expected = {(clip["id"], frame) for frame in range(0, clip["frame_count"], 15)}
        actual = {key for key in seen if key[0] == clip["id"]}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"{clip['id']}: sample mismatch missing={missing[:5]} extra={extra[:5]}"
            )
    for line, event in _records(calibration_dir / "events.jsonl"):
        _validate_event_record(line, event, clips, scenes)
        counts[f"event_{event['review_status']}"] += 1
        counts[f"event_type_{event['event_type']}"] += 1
    return dict(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the sealed NBA holdout calibration annotations."
    )
    parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION
    )
    summary = validate(parser.parse_args().calibration_dir.resolve())
    print(f"NBA holdout calibration annotations valid: {summary}")
