import argparse
import ast
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "courtvision_nba_holdout_v1"
DEFAULT_OUTPUT = ROOT / "holdout_videos" / "nba_sealed_v1"


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
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    result = {
        "frame_count": frame_count,
        "fps": fps,
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": round(frame_count / fps, 6) if fps else 0.0,
    }
    capture.release()
    return result


def _source_path(selection, clip):
    source = selection["sources"][clip["source"]]
    return ROOT / source["local_root"] / clip["source_path"]


def _load_bard_events(selection):
    bard_root = ROOT / selection["sources"]["bard"]["local_root"]
    result = {}
    for year in ("2024", "2025"):
        path = bard_root / "validation" / year / "benchmark.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                name = Path(row["files"]).name
                result[name] = ast.literal_eval(row["actions_name"])
    return result


def _copy_full(source_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != output_path.resolve():
        shutil.copy2(source_path, output_path)


def _extract_window(source_path, output_path, start_seconds, duration_seconds):
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {source_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata for {source_path}")
    start_frame = round(start_seconds * fps)
    requested_frames = round(duration_seconds * fps)
    if start_frame < 0 or start_frame + requested_frames > frame_count:
        capture.release()
        raise ValueError(
            f"Requested window {start_seconds}+{duration_seconds}s exceeds "
            f"{frame_count / fps:.3f}s source {source_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not create {output_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    while written < requested_frames:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    capture.release()
    if written != requested_frames:
        output_path.unlink(missing_ok=True)
        raise ValueError(
            f"Expected {requested_frames} frames but wrote {written} from {source_path}"
        )


def validate_selection(selection):
    errors = []
    clips = selection.get("clips", [])
    policy = selection.get("selection_policy", {})
    ids = [clip.get("id") for clip in clips]
    source_keys = [
        (clip.get("source"), clip.get("source_path")) for clip in clips
    ]
    nba = [clip for clip in clips if clip.get("cohort") == "nba"]
    non_nba = [clip for clip in clips if clip.get("cohort") != "nba"]
    game_ids = [clip.get("game_id") for clip in nba]
    seen_games = set(selection.get("known_seen_source_games", []))
    if len(clips) != policy.get("target_clip_count"):
        errors.append("clip count does not match selection policy")
    if len(nba) != policy.get("nba_clip_count"):
        errors.append("NBA clip count does not match selection policy")
    if len(non_nba) != policy.get("non_nba_clip_count"):
        errors.append("non-NBA clip count does not match selection policy")
    if len(ids) != len(set(ids)):
        errors.append("clip IDs are not unique")
    if len(source_keys) != len(set(source_keys)):
        errors.append("source paths are not unique")
    if len(game_ids) != len(set(game_ids)):
        errors.append("NBA source games are not unique")
    overlap = sorted(set(game_ids) & seen_games)
    if overlap:
        errors.append(f"selected NBA games were already seen: {overlap}")
    expected_cohorts = {
        "nba": 24,
        "ncaa_mens": 1,
        "ncaa_womens": 1,
        "nba_g_league": 1,
        "fiba": 1,
        "womens_pro": 1,
        "fixed_camera": 1,
    }
    actual_cohorts = Counter(clip.get("cohort") for clip in clips)
    if actual_cohorts != Counter(expected_cohorts):
        errors.append(
            f"cohort composition mismatch: expected {expected_cohorts}, "
            f"got {dict(actual_cohorts)}"
        )
    for clip in clips:
        if clip.get("copy_mode") not in {"full", "window"}:
            errors.append(f"{clip.get('id')}: invalid copy_mode")
        if clip.get("copy_mode") == "window":
            if clip.get("start_seconds", -1) < 0:
                errors.append(f"{clip.get('id')}: invalid start_seconds")
            if clip.get("duration_seconds", 0) <= 0:
                errors.append(f"{clip.get('id')}: invalid duration_seconds")
    if errors:
        raise ValueError("\n".join(errors))


def build(benchmark_dir=DEFAULT_BENCHMARK, output_dir=DEFAULT_OUTPUT):
    selection_path = benchmark_dir / "source_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    bard_events = _load_bard_events(selection)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_clips = []
    checksums = []
    pending = []
    for clip in selection["clips"]:
        source_path = _source_path(selection, clip)
        output_path = output_dir / f"{clip['id']}.mp4"
        record = {
            "id": clip["id"],
            "cohort": clip["cohort"],
            "video_path": output_path.relative_to(ROOT).as_posix(),
            "source": clip["source"],
            "coverage_tags": clip["coverage_tags"],
            "selection_status": "pending_source",
        }
        if not source_path.is_file():
            pending.append({
                "id": clip["id"],
                "source": clip["source"],
                "source_path": clip["source_path"],
            })
            dataset_clips.append(record)
            continue
        if clip["copy_mode"] == "full":
            _copy_full(source_path, output_path)
        else:
            _extract_window(
                source_path,
                output_path,
                clip["start_seconds"],
                clip["duration_seconds"],
            )
        metadata = _metadata(output_path)
        digest = _sha256(output_path)
        source_digest = _sha256(source_path)
        record.update({
            **metadata,
            "sha256": digest,
            "source_sha256": source_digest,
            "source_size_bytes": source_path.stat().st_size,
            "selection_status": "ready_unannotated",
        })
        if clip["source"] == "bard":
            record["source_event_labels"] = bard_events.get(source_path.name, [])
        if clip["copy_mode"] == "window":
            record["source_window"] = {
                "start_seconds": clip["start_seconds"],
                "duration_seconds": clip["duration_seconds"],
            }
        dataset_clips.append(record)
        checksums.append(f"{digest}  {output_path.name}")
    dataset = {
        "schema_version": selection["schema_version"],
        "benchmark_id": selection["benchmark_id"],
        "selection_frozen_utc": selection["selection_frozen_utc"],
        "selection_sha256": _sha256(selection_path),
        "clip_count": len(dataset_clips),
        "ready_clip_count": sum(
            clip["selection_status"] == "ready_unannotated"
            for clip in dataset_clips
        ),
        "pending_clip_count": len(pending),
        "annotation_status": "not_started",
        "courtvision_evaluated": False,
        "clips": dataset_clips,
        "pending_sources": pending,
    }
    (benchmark_dir / "dataset.json").write_text(
        json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
    )
    (benchmark_dir / "checksums.sha256").write_text(
        "\n".join(checksums) + ("\n" if checksums else ""),
        encoding="utf-8",
    )
    print(
        f"Holdout materialized: {dataset['ready_clip_count']}/"
        f"{dataset['clip_count']} clips; {dataset['pending_clip_count']} pending"
    )
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize the sealed NBA-weighted CourtVision holdout."
    )
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.benchmark_dir.resolve(), args.output_dir.resolve())
