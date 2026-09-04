"""Run the pre-registered independent pilot without adjusting production logic."""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_multisports_validation import freeze_sources, sha
from scripts.evaluate_event_sequences import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "runs/multisports-independent-v1")
    parser.add_argument("--limit", type=int, help="Run only the first N registered clips; report missing coverage")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    directory = args.directory.resolve()
    protocol = json.loads((directory / "protocol.json").read_text())
    if freeze_sources() != protocol["frozen_sources"]:
        raise ValueError("Production source or weights changed after registration")
    videos = json.loads((directory / "videos.json").read_text())
    if [v["video_id"] for v in videos] != protocol["selected_video_ids"]:
        raise ValueError("Prepared videos do not match registered selection")
    annotations = json.loads((directory / "annotations.json").read_text())
    analyses, logs = directory / "analyses", directory / "logs"
    stub_directory = Path(protocol.get("stub_directory", directory / "stubs")).resolve()
    if not stub_directory.is_relative_to(ROOT / "runs"):
        raise ValueError("Validation stub directory must remain under runs")
    logs.mkdir(exist_ok=True)
    status_path = directory / "run_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        "started_utc": datetime.now(timezone.utc).isoformat(), "runs": {},
    }

    def save_reports():
        for label, tolerance in (("primary", protocol["primary_tolerance_seconds"]),
                                 ("sensitivity", protocol["sensitivity_tolerance_seconds"])):
            report = evaluate(annotations, analyses, tolerance_seconds=tolerance, split="validation")
            report["protocol_sha256"] = sha(directory / "protocol.json")
            report["official_split_video_count"] = protocol["official_basketball_validation_video_count"]
            report["scope"] = protocol["scope"]
            (directory / f"{label}_report.json").write_text(json.dumps(report, indent=2) + "\n")
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    pending = []
    for video in videos[:args.limit]:
        video_id = video["video_id"]
        destination = analyses / (video_id + "_analysis.json")
        previous = status["runs"].get(video_id, {})
        if destination.exists() and previous.get("returncode") == 0:
            if sha(destination) != previous.get("analysis_sha256"):
                raise ValueError("Completed analysis changed since previous run")
            print(f"Already complete: {video_id}", flush=True)
            continue
        if destination.exists():
            raise ValueError(f"Unverified pre-existing analysis: {destination}")
        if sha(video["path"]) != video["sha256"]:
            raise ValueError("Source clip checksum changed")
        pending.append(video)

    def run_video(video):
        video_id = video["video_id"]
        destination = analyses / (video_id + "_analysis.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        entrypoint = ROOT / "main.py"
        for amendment in protocol.get("amendments", []):
            if amendment.get("kind") == "decode allocation safety-ceiling exception" \
                    and amendment.get("trigger_video_id") == video_id:
                entrypoint = ROOT / amendment["wrapper"]
                if sha(entrypoint) != amendment["wrapper_sha256"]:
                    raise ValueError("Expanded-decode wrapper changed after registration")
        command = [sys.executable, str(entrypoint), video["path"],
                   *protocol["pipeline_flags"], "--stub-path", str(stub_directory),
                   "--output-analysis", str(destination)]
        log_path = logs / (Path(video_id).name + ".log")
        started = time.monotonic()
        print(f"Running {video_id}: {video['frame_count']} frames at {video['fps']} FPS", flush=True)
        environment = dict(os.environ, PYTHONUNBUFFERED="1")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record = {"returncode": result.returncode, "elapsed_seconds": round(time.monotonic() - started, 3),
                  "log": str(log_path), "frame_count": video["frame_count"],
                  "source_snapshot": protocol.get("active_source_snapshot", "registered")}
        if result.returncode == 0:
            analysis = json.loads(destination.read_text())
            if analysis["source"]["frameCount"] != video["frame_count"]:
                raise ValueError("Pipeline frame count differs from annotated source")
            record["analysis_sha256"] = sha(destination)
            record["event_counts"] = {kind: sum(e["type"] == kind for e in analysis["events"])
                                      for kind in ("pass", "interception", "shot_attempt")}
        return video_id, record

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_video, video): video["video_id"] for video in pending}
        status["active_video_ids"] = list(futures.values())
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        for future in as_completed(futures):
            video_id, record = future.result()
            status["runs"][video_id] = record
            status["active_video_ids"].remove(video_id)
            save_reports()
            print(f"Completed {video_id}: {record}", flush=True)
            if freeze_sources() != protocol["frozen_sources"]:
                raise ValueError("Production source or weights changed during validation")
    save_reports()
    report = json.loads((directory / "primary_report.json").read_text())
    print(json.dumps({"coverage": report["coverage"], "micro": report["micro"],
                      "by_class": report["by_class"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
