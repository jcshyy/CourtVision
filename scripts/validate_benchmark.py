import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "courtvision_v1"
BALL_VISIBILITY = {"visible", "occluded", "out_of_frame", "uncertain"}
POSSESSION_STATES = {"controlled", "loose", "in_flight", "shot", "dead", "unknown"}
TEAMS = {None, "team_a", "team_b"}
EVENT_TYPES = {
    "pass", "steal", "interception", "offensive_rebound", "defensive_rebound",
    "shot", "deflection", "dead_ball", "unknown_change",
}


def _records(path):
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            yield line_number, json.loads(line)


def validate(benchmark_dir):
    dataset = json.loads((benchmark_dir / "dataset.json").read_text(encoding="utf-8"))
    videos = {video["id"]: video for video in dataset["videos"]}
    errors = []
    counts = Counter()
    seen = set()
    for line, record in _records(benchmark_dir / "annotations.jsonl"):
        key = (record.get("video_id"), record.get("frame_index"))
        if key in seen:
            errors.append(f"annotations:{line}: duplicate {key}")
        seen.add(key)
        video = videos.get(record.get("video_id"))
        if video is None:
            errors.append(f"annotations:{line}: unknown video")
            continue
        frame = record.get("frame_index")
        if not isinstance(frame, int) or not 0 <= frame < video["frame_count"]:
            errors.append(f"annotations:{line}: invalid frame_index")
        image = benchmark_dir / record.get("image_path", "")
        if not image.is_file():
            errors.append(f"annotations:{line}: missing image {image}")
        status = record.get("review_status")
        if status not in {"pending", "verified"}:
            errors.append(f"annotations:{line}: invalid review_status")
        visibility = record.get("ball", {}).get("visibility")
        center = record.get("ball", {}).get("center_px")
        if visibility not in BALL_VISIBILITY:
            errors.append(f"annotations:{line}: invalid ball visibility")
        if center is not None:
            if (
                not isinstance(center, list) or len(center) != 2
                or not all(isinstance(value, (int, float)) for value in center)
                or not 0 <= center[0] < video["width"]
                or not 0 <= center[1] < video["height"]
            ):
                errors.append(f"annotations:{line}: invalid ball center")
        if status == "verified" and visibility == "visible" and center is None:
            errors.append(f"annotations:{line}: verified visible ball needs center")
        possession = record.get("possession", {})
        state, team = possession.get("state"), possession.get("team")
        if state not in POSSESSION_STATES or team not in TEAMS:
            errors.append(f"annotations:{line}: invalid possession label")
        if state == "controlled" and team is None:
            errors.append(f"annotations:{line}: controlled possession needs team")
        if state != "controlled" and team is not None:
            errors.append(f"annotations:{line}: non-controlled state cannot name team")
        counts[status] += 1
    for line, event in _records(benchmark_dir / "events.jsonl"):
        video = videos.get(event.get("video_id"))
        if video is None:
            errors.append(f"events:{line}: unknown video")
            continue
        if event.get("event_type") not in EVENT_TYPES:
            errors.append(f"events:{line}: invalid event type")
        if event.get("review_status") not in {"pending", "verified", "rejected"}:
            errors.append(f"events:{line}: invalid review_status")
        if event.get("start_frame", -1) > event.get("end_frame", -1):
            errors.append(f"events:{line}: start follows end")
        for field in ("release_frame", "catch_frame"):
            value = event.get(field)
            if value is not None and (
                not isinstance(value, int) or not 0 <= value < video["frame_count"]
            ):
                errors.append(f"events:{line}: invalid {field}")
        for field in ("from_team", "to_team"):
            if event.get(field) not in TEAMS:
                errors.append(f"events:{line}: invalid {field}")
        if event.get("review_status") == "verified":
            event_type = event.get("event_type")
            release = event.get("release_frame")
            catch = event.get("catch_frame")
            if event_type in {"pass", "steal", "interception"}:
                if release is None or catch is None:
                    errors.append(
                        f"events:{line}: verified {event_type} needs release and catch"
                    )
                elif release > catch:
                    errors.append(f"events:{line}: release follows catch")
            elif event_type == "shot":
                if release is None:
                    errors.append(f"events:{line}: verified shot needs release")
                if catch is not None:
                    errors.append(f"events:{line}: verified shot cannot have catch")
                if event.get("to_team") is not None:
                    errors.append(f"events:{line}: verified shot cannot have to_team")
            elif event_type in {"offensive_rebound", "defensive_rebound"}:
                if release is not None:
                    errors.append(f"events:{line}: verified rebound cannot have release")
                if catch is None:
                    errors.append(f"events:{line}: verified rebound needs catch")
                if event.get("from_team") is not None:
                    errors.append(f"events:{line}: verified rebound cannot have from_team")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Benchmark valid: {dict(counts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate benchmark labels.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    validate(parser.parse_args().benchmark_dir.resolve())
