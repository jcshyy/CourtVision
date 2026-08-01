import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_benchmark import (
    _json_scalar,
    _pipeline_predictions,
    _prf,
    match_events,
)


DEFAULT_CALIBRATION = (
    ROOT
    / "benchmarks"
    / "courtvision_nba_holdout_v1"
    / "calibration_batch_02"
)
SUPPORTED_TRUTH_TYPES = {
    "pass": "pass",
    "steal": "interception",
    "interception": "interception",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score cached pass/turnover predictions on reviewed NBA clips."
    )
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--event-tolerance", type=int, default=15)
    parser.add_argument("--without-hand-evidence", action="store_true")
    parser.add_argument("--baseline-association", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def _jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _dataset_videos(calibration_dir):
    manifest = json.loads((calibration_dir / "manifest.json").read_text(encoding="utf-8"))
    wanted = {clip["id"] for clip in manifest["clips"]}
    dataset_path = calibration_dir.parent / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    videos = {clip["id"]: clip for clip in dataset["clips"] if clip["id"] in wanted}
    missing = wanted - videos.keys()
    if missing:
        raise ValueError(f"Calibration clips missing from dataset: {sorted(missing)}")
    return videos


def _recompute_events(
    pipeline,
    video,
    *,
    without_hand_evidence=False,
    reject_preexisting_competing_takeovers=True,
):
    from backend.app.analytics import BallAcquisitionDetector, PassInterceptionDetector

    ball_tracks = copy.deepcopy(pipeline["ball_tracks"])
    if without_hand_evidence:
        for frame in ball_tracks:
            info = frame.get(1, {})
            info.pop("hand_pose_supported", None)
            info.pop("hand_pose_player_id", None)

    holder_states = BallAcquisitionDetector(fps=video["fps"]).detect_holder_states(
        pipeline["player_tracks"],
        ball_tracks,
    )
    acquisitions = [
        state["holder_id"] if state["holder_id"] is not None else -1
        for state in holder_states
    ]
    detector = PassInterceptionDetector(
        max_holder_gap_frames=max(1, round(video["fps"] * 0.9)),
        minimum_catch_frames=max(2, round(video["fps"] * 0.1)),
        catch_confirmation_frames=max(3, round(video["fps"])),
        reject_preexisting_competing_takeovers=(
            reject_preexisting_competing_takeovers
        ),
    )
    acquisitions = detector.clean_transient_control_chains(
        acquisitions,
        pipeline["assignments"],
        holder_states=holder_states,
    )
    return detector.detect_events(
        acquisitions,
        pipeline["assignments"],
        holder_states=holder_states,
        ball_tracks=ball_tracks,
        player_tracks=pipeline["player_tracks"],
    )


def _canonical_truth(events, available_video_ids):
    result = []
    for event in events:
        canonical_type = SUPPORTED_TRUTH_TYPES.get(event.get("event_type"))
        if (
            event.get("review_status") != "verified"
            or event.get("video_id") not in available_video_ids
            or canonical_type is None
        ):
            continue
        normalized = dict(event)
        normalized["event_type"] = canonical_type
        result.append(normalized)
    return result


def _score_events(truth, predictions, tolerance):
    matching = match_events(truth, predictions, tolerance)
    return {
        **_prf(
            len(matching["matches"]),
            len(matching["unmatched_predictions"]),
            len(matching["unmatched_truth"]),
        ),
        "truth_count": len(truth),
        "prediction_count": len(predictions),
        "matches": [
            {
                "video_id": truth[truth_index]["video_id"],
                "event_type": truth[truth_index]["event_type"],
                "truth_frame": (
                    truth[truth_index].get("catch_frame")
                    if truth[truth_index].get("catch_frame") is not None
                    else truth[truth_index].get("release_frame")
                ),
                "prediction_frame": predictions[prediction_index]["frame_index"],
                "absolute_frame_error": delta,
            }
            for truth_index, prediction_index, delta in matching["matches"]
        ],
        "unmatched_truth": [truth[index] for index in matching["unmatched_truth"]],
        "unmatched_predictions": [
            predictions[index] for index in matching["unmatched_predictions"]
        ],
    }


def score(
    calibration_dir,
    tolerance=15,
    without_hand_evidence=False,
    baseline_association=False,
    require_all=False,
):
    videos = _dataset_videos(calibration_dir)
    pipelines = {}
    skipped = {}
    for video_id, video in videos.items():
        try:
            pipelines[video_id] = _pipeline_predictions(video)
        except FileNotFoundError as error:
            skipped[video_id] = str(error)
    if require_all and skipped:
        raise FileNotFoundError(f"Missing calibration caches: {sorted(skipped)}")

    predictions_by_video = {}
    for video_id, pipeline in pipelines.items():
        predictions = (
            _recompute_events(
                pipeline,
                videos[video_id],
                without_hand_evidence=(
                    without_hand_evidence or baseline_association
                ),
                reject_preexisting_competing_takeovers=(
                    not baseline_association
                ),
            )
            if without_hand_evidence or baseline_association
            else pipeline["events"]
        )
        for event in predictions:
            event["video_id"] = video_id
        predictions_by_video[video_id] = predictions

    all_events = _jsonl(calibration_dir / "events.jsonl")
    truth = _canonical_truth(all_events, pipelines.keys())
    predictions = [
        event
        for video_id in pipelines
        for event in predictions_by_video[video_id]
    ]
    per_video = {}
    for video_id in pipelines:
        per_video[video_id] = _score_events(
            [event for event in truth if event["video_id"] == video_id],
            predictions_by_video[video_id],
            tolerance,
        )
    return {
        "calibration": str(calibration_dir),
        "mode": (
            "baseline_association"
            if baseline_association
            else "without_hand_evidence"
            if without_hand_evidence
            else "current"
        ),
        "event_tolerance_frames": tolerance,
        "scored_video_ids": sorted(pipelines),
        "skipped_video_ids": skipped,
        "supported_ground_truth_types": sorted(SUPPORTED_TRUTH_TYPES),
        "unsupported_events_excluded": sum(
            event.get("review_status") == "verified"
            and event.get("event_type") not in SUPPORTED_TRUTH_TYPES
            for event in all_events
            if event.get("video_id") in pipelines
        ),
        "aggregate": _score_events(truth, predictions, tolerance),
        "per_video": per_video,
    }


def _pct(value):
    return f"{100 * value:.1f}%"


def _markdown(report):
    aggregate = report["aggregate"]
    lines = [
        "# NBA calibration event report",
        "",
        f"Mode: `{report['mode']}`. Scored clips: "
        f"{', '.join(report['scored_video_ids']) or 'none'}.",
        "",
        "Only verified passes and steals/interceptions are scored. Shots, rebounds, "
        "deflections, dead balls, camera cuts, and the remaining draft event are "
        "reported as outside the current detector's supported output types.",
        "",
        "| Scope | Truth | Predicted | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Aggregate | {aggregate['truth_count']} | {aggregate['prediction_count']} | "
        f"{_pct(aggregate['precision'])} | {_pct(aggregate['recall'])} | "
        f"{_pct(aggregate['f1'])} |",
    ]
    for video_id, metrics in report["per_video"].items():
        lines.append(
            f"| {video_id} | {metrics['truth_count']} | {metrics['prediction_count']} | "
            f"{_pct(metrics['precision'])} | {_pct(metrics['recall'])} | "
            f"{_pct(metrics['f1'])} |"
        )
    if report["skipped_video_ids"]:
        lines.extend([
            "",
            "Caches not available yet: "
            + ", ".join(sorted(report["skipped_video_ids"])),
        ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    calibration_dir = args.calibration_dir.resolve()
    report = score(
        calibration_dir,
        tolerance=args.event_tolerance,
        without_hand_evidence=args.without_hand_evidence,
        baseline_association=args.baseline_association,
        require_all=args.require_all,
    )
    markdown = _markdown(report)
    print(markdown, end="")
    if args.output_json:
        args.output_json.write_text(
            json.dumps(report, indent=2, default=_json_scalar) + "\n",
            encoding="utf-8",
        )
    if args.output_markdown:
        args.output_markdown.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
