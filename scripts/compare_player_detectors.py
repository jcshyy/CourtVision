"""Compare current and E-BARD player detections on the same video frames.

This is a detector-agreement diagnostic, not an accuracy benchmark. Accuracy
requires independently annotated player and referee boxes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.tracking import PlayerTracker
from backend.app.tracking.player_tracker import resolve_player_class_ids
from backend.app.utils import probe_video, read_video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the current and E-BARD player detectors on one clip."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path; defaults under runs/player_detector_comparison.",
    )
    return parser.parse_args()


def box_iou(first, second):
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def result_entities(result):
    player_id, referee_id = resolve_player_class_ids(result.names)
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    return {
        "players": [
            {"bbox": box.tolist(), "confidence": float(confidence)}
            for box, confidence, class_id in zip(boxes, confidences, class_ids)
            if class_id == player_id
        ],
        "referees": [
            {"bbox": box.tolist(), "confidence": float(confidence)}
            for box, confidence, class_id in zip(boxes, confidences, class_ids)
            if referee_id is not None and class_id == referee_id
        ],
    }


def greedy_matches(first, second, threshold=0.5):
    candidates = sorted(
        (
            (box_iou(left["bbox"], right["bbox"]), left_index, right_index)
            for left_index, left in enumerate(first)
            for right_index, right in enumerate(second)
        ),
        reverse=True,
    )
    used_first = set()
    used_second = set()
    matches = []
    for iou, first_index, second_index in candidates:
        if iou < threshold:
            break
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        matches.append(iou)
    return matches


def detector_summary(entities, elapsed_seconds):
    players = [item for frame in entities for item in frame["players"]]
    referees = [item for frame in entities for item in frame["referees"]]
    frame_count = len(entities)
    return {
        "elapsed_seconds": round(elapsed_seconds, 3),
        "inference_fps": round(frame_count / elapsed_seconds, 3)
        if elapsed_seconds > 0
        else None,
        "player_detections": len(players),
        "referee_detections": len(referees),
        "frames_with_players": sum(bool(frame["players"]) for frame in entities),
        "mean_players_per_frame": round(len(players) / frame_count, 3)
        if frame_count
        else 0.0,
        "mean_player_confidence": round(
            float(np.mean([item["confidence"] for item in players])), 4
        )
        if players
        else None,
    }


def compare(video, *, start_seconds, duration_seconds, target_fps, max_width):
    metadata = probe_video(video)
    frames = read_video(
        video,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        target_fps=min(target_fps, metadata["fps"]),
        max_width=max_width,
    )
    if not frames:
        raise ValueError("The selected video interval produced no frames")

    detector_results = {}
    for backend in ("current", "ebard"):
        tracker = PlayerTracker(detector_backend=backend)
        started = time.perf_counter()
        results = tracker.detect_frames(frames)
        elapsed = time.perf_counter() - started
        entities = [result_entities(result) for result in results]
        detector_results[backend] = {
            "summary": detector_summary(entities, elapsed),
            "entities": entities,
            "confidence_threshold": tracker.confidence,
            "model_path": str(tracker.model_path),
        }

    frame_agreement = []
    matched_ious = []
    for frame_index, (current, ebard) in enumerate(
        zip(
            detector_results["current"]["entities"],
            detector_results["ebard"]["entities"],
        )
    ):
        matches = greedy_matches(current["players"], ebard["players"])
        matched_ious.extend(matches)
        frame_agreement.append(
            {
                "frame_index": frame_index,
                "current_players": len(current["players"]),
                "ebard_players": len(ebard["players"]),
                "matched_players_iou_50": len(matches),
                "current_referees": len(current["referees"]),
                "ebard_referees": len(ebard["referees"]),
            }
        )

    current_total = detector_results["current"]["summary"]["player_detections"]
    ebard_total = detector_results["ebard"]["summary"]["player_detections"]
    matched_total = len(matched_ious)
    most_different = sorted(
        frame_agreement,
        key=lambda row: abs(row["current_players"] - row["ebard_players"]),
        reverse=True,
    )[:20]
    return {
        "schema_version": 1,
        "comparison_type": "unlabeled_detector_agreement",
        "warning": (
            "Detection count and cross-model agreement are not accuracy metrics; "
            "independent player/referee ground truth is required to select a winner."
        ),
        "video": str(Path(video).resolve()),
        "selection": {
            "start_seconds": start_seconds,
            "duration_seconds": duration_seconds,
            "target_fps": target_fps,
            "max_width": max_width,
            "frame_count": len(frames),
        },
        "detectors": {
            backend: {
                key: value
                for key, value in result.items()
                if key != "entities"
            }
            for backend, result in detector_results.items()
        },
        "agreement": {
            "matched_players_iou_50": matched_total,
            "current_unmatched_players": current_total - matched_total,
            "ebard_unmatched_players": ebard_total - matched_total,
            "mean_matched_iou": round(float(np.mean(matched_ious)), 4)
            if matched_ious
            else None,
            "most_different_frames": most_different,
        },
    }


def main():
    args = parse_args()
    output = args.output or (
        ROOT
        / "runs"
        / "player_detector_comparison"
        / f"{args.video.stem}.json"
    )
    report = compare(
        args.video,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        target_fps=args.target_fps,
        max_width=args.max_width,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["detectors"], indent=2))
    print(json.dumps(report["agreement"], indent=2))
    print(f"Saved comparison to {output}")


if __name__ == "__main__":
    main()
