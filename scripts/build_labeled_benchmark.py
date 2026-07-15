import argparse
import hashlib
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "courtvision_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Build the CourtVision review set.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {path}")
    result = {
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return result


def _load_jsonl(path):
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[(record["video_id"], record["frame_index"])] = record
    return records


def _write_jsonl(path, records):
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def _sample_frames(frame_count, interval, focus_frames, radius):
    frames = set(range(0, frame_count, interval))
    frames.add(frame_count - 1)
    for focus in focus_frames:
        frames.update(range(max(0, focus - radius), min(frame_count, focus + radius + 1)))
    return sorted(frames)


def _extract(video_path, frame_indices, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {video_path}")
    for frame_index in frame_indices:
        output_path = output_dir / f"{frame_index:06d}.jpg"
        if output_path.exists():
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or not cv2.imwrite(str(output_path), frame):
            capture.release()
            raise ValueError(f"Could not extract frame {frame_index} from {video_path}")
    capture.release()


def _suggestions(video, frame_indices):
    report_name = video.get("suggestion_report")
    if not report_name:
        return []
    report_path = ROOT / report_name
    if not report_path.exists():
        return []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    states = report.get("possession", {}).get("holder_states", [])
    events_by_frame = {}
    for event in report.get("events", []):
        events_by_frame.setdefault(event.get("frame_index"), []).append(event)
    output = []
    for frame_index in frame_indices:
        state = states[frame_index] if frame_index < len(states) else None
        output.append({
            "video_id": video["id"],
            "frame_index": frame_index,
            "source_report": report_name,
            "holder_state_suggestion": state,
            "event_suggestions": events_by_frame.get(frame_index, []),
            "ground_truth": False,
        })
    return output


def build(benchmark_dir):
    manifest = json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))
    existing = _load_jsonl(benchmark_dir / "annotations.jsonl")
    interval = manifest["sampling"]["regular_interval_frames"]
    radius = manifest["sampling"]["event_context_radius_frames"]
    dataset_videos = []
    records = []
    suggestions = []
    for split_index, video in enumerate(manifest["videos"]):
        video_path = ROOT / video["path"]
        metadata = _metadata(video_path)
        indices = _sample_frames(
            metadata["frame_count"], interval, video.get("focus_frames", []), radius
        )
        _extract(video_path, indices, benchmark_dir / "frames" / video["id"])
        suggestions.extend(_suggestions(video, indices))
        dataset_videos.append({
            **video,
            **metadata,
            "sha256": _sha256(video_path),
            "sample_count": len(indices),
        })
        split = "validation" if split_index in (1, 4) else "test"
        for frame_index in indices:
            key = (video["id"], frame_index)
            record = existing.get(key, {
                "video_id": video["id"],
                "frame_index": frame_index,
                "timestamp_seconds": round(frame_index / metadata["fps"], 6),
                "image_path": f"frames/{video['id']}/{frame_index:06d}.jpg",
                "split": split,
                "review_status": "pending",
                "ball": {"visibility": "uncertain", "center_px": None},
                "possession": {"state": "unknown", "team": None},
                "notes": "",
            })
            records.append(record)
    dataset = {
        "schema_version": manifest["schema_version"],
        "benchmark_id": manifest["benchmark_id"],
        "videos": dataset_videos,
        "annotation_count": len(records),
        "verified_count": sum(r["review_status"] == "verified" for r in records),
        "suggestion_count": len(suggestions),
    }
    (benchmark_dir / "dataset.json").write_text(
        json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
    )
    _write_jsonl(benchmark_dir / "annotations.jsonl", records)
    _write_jsonl(benchmark_dir / "suggestions.jsonl", suggestions)
    print(f"Built {len(records)} records in {benchmark_dir}")


if __name__ == "__main__":
    build(parse_args().benchmark_dir.resolve())
