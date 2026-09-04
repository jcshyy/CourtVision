"""Compare explicit archived E-BARD manifests with a fresh event replay.

The archived comparator is not a rerun of pre-change code. Both runs reuse the
same detector-cache identities; the report distinguishes this provenance.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_event_sequences import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_files = ["main.py", "backend/app/analytics/shot_rebound.py",
                    "backend/app/analytics/event_lifecycle.py",
                    "backend/app/analytics/pass_interception.py",
                    "backend/app/analytics/possession_timeline.py",
                    "scripts/evaluate_event_sequences.py"]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in source_files}
    archived = args.output_dir / "archived_baseline"
    archived.mkdir(exist_ok=True)
    shot_annotations = json.loads((ROOT / "benchmarks/event_sequences_v2/local_shot_windows.json").read_text())
    dataset = json.loads((ROOT / "benchmarks/courtvision_v1/dataset.json").read_text())
    pass_annotations = {
        "name": "Existing courtvision_v1 verified possession events",
        "scored_types": ["pass", "interception"],
        "note": "Existing candidate-reviewed annotations; not exhaustive independent ground truth",
        "videos": [{"video_id": v["id"], "fps": v["fps"], "frame_count": v["frame_count"], "split": "calibration"}
                   for v in dataset["videos"]],
        "events": [],
    }
    for line in (ROOT / "benchmarks/courtvision_v1/events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("review_status") != "verified":
            continue
        boundary = event.get("release_frame") if event["event_type"] == "pass" else event.get("catch_frame")
        if boundary is None:
            continue
        pass_annotations["events"].append({"video_id": event["video_id"], "type": event["event_type"],
                                           "start_frame": boundary, "end_frame": boundary})
    for video in shot_annotations["videos"] + pass_annotations["videos"]:
        video_id = video["video_id"]
        source = ROOT / "output_videos" / f"{video_id}_ebard_wasb_analysis.json"
        if source.exists():
            shutil.copyfile(source, archived / f"{video_id}_analysis.json")
    for name, annotations in (("shot_windows", shot_annotations), ("possession_events", pass_annotations)):
        (args.output_dir / f"{name}_annotations.json").write_text(json.dumps(annotations, indent=2) + "\n")
        for label, directory in (("archived", archived), ("candidate", args.candidate_dir)):
            report = evaluate(annotations, directory, tolerance_seconds=0.25)
            report["provenance"] = "archived August 21 E-BARD/WASB manifest" if label == "archived" else "fresh event replay on fixed detector caches"
            report["prediction_sha256"] = {
                v["video_id"]: hashlib.sha256((directory / f'{v["video_id"]}_analysis.json').read_bytes()).hexdigest()
                for v in annotations["videos"]
                if (directory / f'{v["video_id"]}_analysis.json').exists()
            }
            if label == "candidate":
                report["candidate_source_sha256"] = source_hashes
            (args.output_dir / f"{name}_{label}_report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(name, label, report["micro"], "coverage", report["coverage"])


if __name__ == "__main__":
    main()
