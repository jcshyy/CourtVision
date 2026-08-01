import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from scripts.build_nba_holdout import (
        DEFAULT_BENCHMARK,
        ROOT,
        _metadata,
        _sha256,
        validate_selection,
    )
except ModuleNotFoundError:
    from build_nba_holdout import (
        DEFAULT_BENCHMARK,
        ROOT,
        _metadata,
        _sha256,
        validate_selection,
    )


def validate(benchmark_dir=DEFAULT_BENCHMARK, require_complete=False):
    selection_path = benchmark_dir / "source_selection.json"
    dataset_path = benchmark_dir / "dataset.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection)
    if not dataset_path.exists():
        raise ValueError("dataset.json is missing; run build_nba_holdout.py")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    errors = []
    if dataset.get("selection_sha256") != _sha256(selection_path):
        errors.append("source selection changed after dataset materialization")
    if dataset.get("clip_count") != len(selection["clips"]):
        errors.append("dataset clip count does not match source selection")
    records = {record["id"]: record for record in dataset.get("clips", [])}
    if len(records) != len(dataset.get("clips", [])):
        errors.append("dataset contains duplicate clip IDs")
    ready = 0
    for selected in selection["clips"]:
        record = records.get(selected["id"])
        if record is None:
            errors.append(f"{selected['id']}: missing dataset record")
            continue
        if record.get("selection_status") == "pending_source":
            continue
        path = ROOT / record.get("video_path", "")
        if not path.is_file():
            errors.append(f"{selected['id']}: missing materialized video")
            continue
        if path.parent.name != "nba_sealed_v1":
            errors.append(f"{selected['id']}: video is outside sealed output directory")
        digest = _sha256(path)
        if digest != record.get("sha256"):
            errors.append(f"{selected['id']}: checksum mismatch")
        metadata = _metadata(path)
        for field in ("frame_count", "width", "height"):
            if metadata[field] != record.get(field):
                errors.append(f"{selected['id']}: {field} mismatch")
        if abs(metadata["fps"] - record.get("fps", 0.0)) > 0.001:
            errors.append(f"{selected['id']}: fps mismatch")
        if selected["copy_mode"] == "window":
            expected = selected["duration_seconds"]
            if abs(metadata["duration_seconds"] - expected) > 1 / metadata["fps"]:
                errors.append(f"{selected['id']}: window duration mismatch")
        ready += 1
    stray = sorted(
        path.name
        for path in (ROOT / "holdout_videos" / "nba_sealed_v1").glob("*")
        if path.is_file() and path.suffix.lower() != ".mp4"
    )
    if stray:
        errors.append(f"non-video files found in sealed clip directory: {stray}")
    analyzed = sorted(
        path.name
        for path in (ROOT / "holdout_videos" / "nba_sealed_v1").glob("*analy*")
    )
    if analyzed:
        errors.append(f"analyzer outputs found in sealed clip directory: {analyzed}")
    if dataset.get("courtvision_evaluated") is not False:
        errors.append("dataset must remain marked as not evaluated before annotation")
    if dataset.get("annotation_status") != "not_started":
        errors.append("unexpected annotation status")
    pending = len(selection["clips"]) - ready
    if require_complete and pending:
        errors.append(f"{pending} source clips are still pending")
    if errors:
        raise ValueError("\n".join(errors))
    cohorts = Counter(record["cohort"] for record in dataset["clips"])
    print(
        f"NBA holdout valid: {ready}/{len(selection['clips'])} ready, "
        f"{pending} pending, cohorts={dict(cohorts)}"
    )
    return {"ready": ready, "pending": pending, "cohorts": dict(cohorts)}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the sealed NBA holdout.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate(args.benchmark_dir.resolve(), args.require_complete)
