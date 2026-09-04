"""Summarize detector/possession support inside the frozen MultiSports events."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/multisports-independent-v1"


def ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else 0.0


def main():
    annotations = json.loads((RUN / "annotations.json").read_text())
    report = json.loads((RUN / "primary_report.json").read_text())
    per_video = {item["video_id"]: item for item in report["per_video"]}
    grouped = defaultdict(list)
    for event in annotations["events"]:
        grouped[event["video_id"]].append(event)

    totals = defaultdict(lambda: {"frames": 0, "ball_observed": 0, "controlled": 0,
                                  "event_count": 0})
    whole = {"frames": 0, "ball_observed": 0, "controlled": 0}
    for video in annotations["videos"]:
        video_id = video["video_id"]
        analysis_path = RUN / "analyses" / f"{video_id}_analysis.json"
        analysis = json.loads(analysis_path.read_text())
        frames = analysis["diagnostics"]["possessionTimeline"]["fused"]["frames"]
        whole["frames"] += len(frames)
        whole["ball_observed"] += sum(bool(frame.get("ball_observed")) for frame in frames)
        whole["controlled"] += sum(frame.get("state") == "controlled" for frame in frames)
        matched = {item["truth_index"] for item in per_video[video_id]["matches"]}
        for index, event in enumerate(grouped[video_id]):
            bucket = f"{event['type']}:{'matched' if index in matched else 'missed'}"
            selected = frames[event["start_frame"]:event["end_frame"] + 1]
            totals[bucket]["event_count"] += 1
            totals[bucket]["frames"] += len(selected)
            totals[bucket]["ball_observed"] += sum(bool(frame.get("ball_observed")) for frame in selected)
            totals[bucket]["controlled"] += sum(frame.get("state") == "controlled" for frame in selected)

    output = {
        "whole_sample": {**whole,
                         "ball_observed_fraction": ratio(whole["ball_observed"], whole["frames"]),
                         "controlled_fraction": ratio(whole["controlled"], whole["frames"])},
        "ground_truth_intervals": {},
    }
    for key, counts in sorted(totals.items()):
        output["ground_truth_intervals"][key] = {
            **counts,
            "ball_observed_fraction": ratio(counts["ball_observed"], counts["frames"]),
            "controlled_fraction": ratio(counts["controlled"], counts["frames"]),
        }
    destination = RUN / "diagnostic_summary.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
