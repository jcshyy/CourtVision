"""Record validation infrastructure amendments before retrying failed clips."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_multisports_validation import freeze_sources, sha

RUN = ROOT / "runs/multisports-independent-v1"
EMPTY_KEYPOINT_CLIP = "basketball/v_SHFVKZ6HJc8_c603"
LARGE_CLIP = "basketball/v_4LXTUim5anY_c012"
RECOVERED_CLIP = "basketball/v_It_vvQR6RPM_c002"


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main():
    protocol_path = RUN / "protocol.json"
    status_path = RUN / "run_status.json"
    protocol = json.loads(protocol_path.read_text())
    status = json.loads(status_path.read_text())
    if protocol.get("amendments"):
        if protocol["frozen_sources"] != freeze_sources():
            raise ValueError("Sources changed after the recorded amendment")
        wrapper = ROOT / "scripts/run_pipeline_expanded_decode.py"
        allocation = next(item for item in protocol["amendments"]
                          if item["kind"] == "decode allocation safety-ceiling exception")
        current_wrapper_sha = sha(wrapper)
        if allocation["wrapper_sha256"] != current_wrapper_sha:
            previous_sha = allocation["wrapper_sha256"]
            allocation["wrapper_sha256"] = current_wrapper_sha
            protocol["amendments"].append({
                "registered_utc": datetime.now(timezone.utc).isoformat(),
                "kind": "expanded-decode wrapper startup-path fix",
                "trigger_video_id": LARGE_CLIP,
                "change": "Add the repository root to sys.path before importing the unchanged production entrypoint.",
                "old_wrapper_sha256": previous_sha,
                "new_wrapper_sha256": current_wrapper_sha,
                "registered_after_startup_failure_before_video_decode_or_inference": True,
                "detector_event_threshold_or_scoring_change": False,
            })
            protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
            print("Recorded pre-inference wrapper startup-path amendment")
            return
        print("Technical amendments already recorded")
        return

    original = protocol["frozen_sources"]
    current = freeze_sources()
    changed = {name for name in set(original) | set(current) if original.get(name) != current.get(name)}
    expected = {"backend/app/analytics/event_lifecycle.py"}
    if changed != expected:
        raise ValueError(f"Unexpected source changes since registration: {sorted(changed)}")

    wrapper = ROOT / "scripts/run_pipeline_expanded_decode.py"
    timestamp = datetime.now(timezone.utc).isoformat()
    protocol["original_frozen_sources"] = original
    protocol["frozen_sources"] = current
    protocol["source_snapshots"] = {
        "registered": canonical_hash(original),
        "technical_amendment_1": canonical_hash(current),
    }
    protocol["active_source_snapshot"] = "technical_amendment_1"
    protocol["amendments"] = [
        {
            "registered_utc": timestamp,
            "kind": "crash-only robustness guard",
            "trigger_video_id": EMPTY_KEYPOINT_CLIP,
            "change": "Treat an empty court-keypoint model result as unavailable geometry instead of indexing it.",
            "changed_file": "backend/app/analytics/event_lifecycle.py",
            "old_sha256": original["backend/app/analytics/event_lifecycle.py"],
            "new_sha256": current["backend/app/analytics/event_lifecycle.py"],
            "registered_after_failure_before_retry": True,
            "detector_event_threshold_or_scoring_change": False,
        },
        {
            "registered_utc": timestamp,
            "kind": "decode allocation safety-ceiling exception",
            "trigger_video_id": LARGE_CLIP,
            "change": "Raise max_decoded_bytes from 2 GiB to 3 GiB for this full 875-frame 1280x720 clip only.",
            "wrapper": str(wrapper.relative_to(ROOT)).replace("\\", "/"),
            "wrapper_sha256": sha(wrapper),
            "registered_after_guard_failure_before_inference": True,
            "resize_resample_detector_threshold_or_scoring_change": False,
        },
    ]

    destination = RUN / "analyses" / f"{RECOVERED_CLIP}_analysis.json"
    analysis = json.loads(destination.read_text())
    videos = {item["video_id"]: item for item in json.loads((RUN / "videos.json").read_text())}
    if analysis["source"]["frameCount"] != videos[RECOVERED_CLIP]["frame_count"]:
        raise ValueError("Interrupted-run analysis has the wrong frame count")
    recovered = {
        "returncode": 0,
        "elapsed_seconds": None,
        "log": str(RUN / "logs" / (Path(RECOVERED_CLIP).name + ".log")),
        "frame_count": videos[RECOVERED_CLIP]["frame_count"],
        "analysis_sha256": sha(destination),
        "event_counts": {kind: sum(event["type"] == kind for event in analysis["events"])
                         for kind in ("pass", "interception", "shot_attempt")},
        "source_snapshot": "registered",
        "recovered_after_runner_integrity_stop": True,
    }
    status["runs"][RECOVERED_CLIP] = recovered
    for record in status["runs"].values():
        record.setdefault("source_snapshot", "registered")
    status["active_video_ids"] = []
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"Recorded two technical amendments and recovered {RECOVERED_CLIP}")


if __name__ == "__main__":
    main()
