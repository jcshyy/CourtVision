"""Pre-register a deterministic validation sample, then extract exact source clips.

Never execute code from downloaded annotations or the upstream extraction script.
Selection depends only on official validation IDs, not labels or predictions.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_multisports_events import AnnotationUnpickler, convert
from scripts.download_multisports import REVISION, FILES

SEED = "courtvision-multisports-v1:"


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def freeze_sources():
    paths = [ROOT / "main.py", *sorted((ROOT / "backend/app").rglob("*.py"))]
    paths += [ROOT / "backend/models" / name for name in
              ("ebard_yolov8n.pt", "wasb_basketball_torchscript.pt", "yolo11n-pose.pt", "court_keypoint_detector.pt")]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "holdout_sources/multisports/data/trainval")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/multisports-independent-v1")
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--range-downloads", action="store_true", help="Use the verified-length, hashed clips retrieved by the official HTTP-range downloader")
    args = parser.parse_args()
    gt_path = args.data / "multisports_GT.pkl"
    if sha(gt_path) != FILES["multisports_GT.pkl"]:
        raise ValueError("Official annotation checksum mismatch")
    with gt_path.open("rb") as source:
        data = AnnotationUnpickler(source).load()
    ids = [v for v in data["test_videos"][0] if v.startswith("basketball/")]
    ordered = sorted(ids, key=lambda v: hashlib.sha256((SEED + v).encode()).hexdigest())
    if not 0 < args.sample_size <= len(ids):
        raise ValueError("Sample size is outside official split")
    selected = ordered[:args.sample_size]
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["selected_video_ids"] != selected:
            raise ValueError("Refusing to change a pre-registered sample")
        if protocol["frozen_sources"] != freeze_sources():
            raise ValueError("Production source/model changed since registration")
    else:
        protocol = {
            "registered_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "MCG-NJU/SportsAction", "revision": REVISION,
            "annotation_sha256": FILES["multisports_GT.pkl"],
            "archive_sha256": FILES["basketball.tar"],
            "selection": "Lowest SHA256(seed + official validation video ID), no label filtering",
            "seed": SEED, "official_basketball_validation_video_count": len(ids),
            "official_basketball_validation_frame_count": sum(data["nframes"][v] for v in ids),
            "selected_video_ids": selected,
            "sample_frame_count": sum(data["nframes"][v] for v in selected),
            "primary_tolerance_seconds": 0.25, "sensitivity_tolerance_seconds": 0.5,
            "pipeline_flags": ["--analysis-only", "--allow-uncertain-teams",
                               "--scene-detector-backend", "ebard", "--ball-detector-backend", "hybrid"],
            "resize_or_resample": False, "ground_truth_used_for_team_assignment": False,
            "frozen_sources": freeze_sources(),
            "scope": "Independent-dataset pilot, not full split or guaranteed pretraining-disjoint evaluation",
        }
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Registered {len(selected)}/{len(ids)} videos; {protocol['sample_frame_count']} frames", flush=True)
    if args.register_only:
        return
    clips_root = args.output / "clips"
    wanted = {v + ".mp4": v for v in selected}
    found = {}
    if args.range_downloads:
        downloaded = json.loads((args.output / "range_downloads.json").read_text())
        if downloaded["revision"] != REVISION:
            raise ValueError("Range download revision mismatch")
        for record in downloaded["clips"]:
            destination = Path(record["path"]).resolve()
            if not destination.is_relative_to(clips_root.resolve()) or sha(destination) != record["sha256"]:
                raise ValueError("Range-downloaded clip path/hash mismatch")
            found[record["video_id"]] = destination
    else:
        archive = args.data / "basketball.tar"
        if sha(archive) != FILES["basketball.tar"]:
            raise ValueError("Official basketball archive checksum mismatch")
        with tarfile.open(archive, "r|") as source:
            for member in source:
                name = str(PurePosixPath(member.name))
                if name.startswith("./"):
                    name = name[2:]
                if name not in wanted:
                    continue
                if not member.isfile() or ".." in PurePosixPath(name).parts:
                    raise ValueError("Unsafe selected archive member")
                destination = (clips_root / name).resolve()
                if not destination.is_relative_to(clips_root.resolve()):
                    raise ValueError("Unsafe extraction path")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    with source.extractfile(member) as incoming, destination.open("xb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                if destination.stat().st_size != member.size:
                    raise ValueError(f"Incomplete extracted clip: {name}")
                found[wanted[name]] = destination
                print(f"Extracted {name}", flush=True)
    if set(found) != set(selected):
        raise ValueError(f"Missing selected clips: {sorted(set(selected) - set(found))}")
    import cv2
    fps_by_video, videos = {}, []
    for video_id in selected:
        capture = cv2.VideoCapture(str(found[video_id]))
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        capture.release()
        if count != data["nframes"][video_id] or size != tuple(data["resolution"][video_id]) or fps <= 0:
            raise ValueError(f"Video/annotation metadata mismatch: {video_id} {count} {size} {fps}")
        fps_by_video[video_id] = fps
        videos.append({"video_id": video_id, "path": str(found[video_id]), "fps": fps,
                       "frame_count": count, "height": size[0], "width": size[1], "sha256": sha(found[video_id])})
    subset = dict(data)
    subset["test_videos"] = [selected]
    annotations = convert(subset, fps_by_video)
    annotations["name"] = "MultiSports deterministic independent basketball pilot v1"
    annotations["source_sha256"] = protocol["annotation_sha256"]
    (args.output / "annotations.json").write_text(json.dumps(annotations, indent=2) + "\n")
    (args.output / "videos.json").write_text(json.dumps(videos, indent=2) + "\n")
    print(f"Prepared {len(videos)} source clips, {len(annotations['events'])} scored event annotations", flush=True)


if __name__ == "__main__":
    main()
