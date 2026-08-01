import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = (
    ROOT / "benchmarks" / "courtvision_nba_holdout_v1" / "calibration_batch_02"
)


def _scene_for_frame(clip, frame_index):
    matches = [
        scene
        for scene in clip["scenes"]
        if scene["start_frame"] <= frame_index <= scene["end_frame"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{clip['id']} frame {frame_index}: expected exactly one scene"
        )
    return matches[0]


def _possession_for_frame(clip_id, clip_spec, frame_index):
    matches = [
        segment
        for segment in clip_spec["possession_segments"]
        if segment["start_frame"] <= frame_index <= segment["end_frame"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{clip_id} frame {frame_index}: expected exactly one possession segment"
        )
    segment = matches[0]
    return {
        "state": segment["state"],
        "team_id": segment.get("team_id"),
        "holder": segment.get("holder"),
    }, segment.get("notes")


def build(calibration_dir=DEFAULT_CALIBRATION):
    calibration_dir = Path(calibration_dir)
    manifest = json.loads(
        (calibration_dir / "manifest.json").read_text(encoding="utf-8")
    )
    spec = json.loads(
        (calibration_dir / "annotation_spec.json").read_text(encoding="utf-8")
    )
    interval = manifest.get("sampling_interval_frames", 15)
    if interval != 15:
        raise ValueError("annotation validator currently requires a 15-frame interval")

    clip_specs = spec.get("clips", {})
    manifest_ids = {clip["id"] for clip in manifest["clips"]}
    if set(clip_specs) != manifest_ids:
        raise ValueError("annotation spec clip ids must exactly match the manifest")

    frame_records = []
    for clip in manifest["clips"]:
        clip_id = clip["id"]
        clip_spec = clip_specs[clip_id]
        overrides = clip_spec.get("ball_overrides", {})
        sampled = set(range(0, clip["frame_count"], interval))
        override_frames = {int(frame) for frame in overrides}
        if not override_frames <= sampled:
            raise ValueError(f"{clip_id}: ball override is not on a sampled frame")

        for frame_index in sorted(sampled):
            scene = _scene_for_frame(clip, frame_index)
            possession, segment_notes = _possession_for_frame(
                clip_id, clip_spec, frame_index
            )
            override = overrides.get(str(frame_index))
            if override is None:
                ball = {
                    "visibility": "uncertain",
                    "center_px": None,
                    "confidence": "low",
                }
                notes = segment_notes or (
                    "Raw-video draft; center withheld because the ball is not "
                    "visually separable at this sample."
                )
            else:
                ball = {
                    "visibility": override["visibility"],
                    "center_px": override.get("center_px"),
                    "confidence": override["confidence"],
                }
                notes = override.get("notes") or segment_notes or (
                    "Ball position labeled from the raw full-resolution frame."
                )
            frame_records.append({
                "video_id": clip_id,
                "frame_index": frame_index,
                "scene_id": scene["id"],
                "ball": ball,
                "possession": possession,
                "review_status": "draft",
                "notes": notes,
            })

    event_records = spec.get("events", [])
    (calibration_dir / "frames.jsonl").write_text(
        "".join(
            json.dumps(record, separators=(",", ":")) + "\n"
            for record in frame_records
        ),
        encoding="utf-8",
    )
    (calibration_dir / "events.jsonl").write_text(
        "".join(
            json.dumps(record, separators=(",", ":")) + "\n"
            for record in event_records
        ),
        encoding="utf-8",
    )
    return {"frames": len(frame_records), "events": len(event_records)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expand a raw-video annotation spec into benchmark JSONL files."
    )
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION)
    print(build(parser.parse_args().calibration_dir.resolve()))
