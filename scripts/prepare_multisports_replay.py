"""Freeze a post-change MultiSports replay that reuses detector caches."""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_multisports_validation import freeze_sources, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/multisports-independent-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/multisports-independent-v2")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Replay directory already exists: {output}")
    output.mkdir(parents=True)
    for name in ("annotations.json", "videos.json"):
        shutil.copyfile(source / name, output / name)

    parent = json.loads((source / "protocol.json").read_text())
    wrapper = ROOT / "scripts/run_pipeline_expanded_decode.py"
    protocol = {
        key: parent[key]
        for key in (
            "dataset", "revision", "annotation_sha256", "archive_sha256",
            "selection", "seed", "official_basketball_validation_video_count",
            "official_basketball_validation_frame_count", "selected_video_ids",
            "sample_frame_count", "primary_tolerance_seconds",
            "sensitivity_tolerance_seconds", "pipeline_flags",
            "resize_or_resample", "ground_truth_used_for_team_assignment", "scope",
        )
    }
    protocol.update({
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "One post-change replay after development-only selection; no MultiSports tuning",
        "parent_protocol_sha256": sha(source / "protocol.json"),
        "frozen_sources": freeze_sources(),
        "active_source_snapshot": "possession_continuity_v2",
        "stub_directory": str((source / "stubs").resolve()),
        "amendments": [{
            "kind": "decode allocation safety-ceiling exception",
            "trigger_video_id": "basketball/v_4LXTUim5anY_c012",
            "wrapper": str(wrapper.relative_to(ROOT)).replace("\\", "/"),
            "wrapper_sha256": sha(wrapper),
            "resize_resample_detector_threshold_or_scoring_change": False,
        }],
    })
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Frozen replay at {output}")


if __name__ == "__main__":
    main()
