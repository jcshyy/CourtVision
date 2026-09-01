import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import supervision as sv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.detection import PlayerPoseDetector, attach_player_poses
from backend.app.tracking import BallTracker, PlayerTracker
from scripts.score_nba_detection_benchmark import score_predictions, xywh_to_xyxy
from scripts.validate_nba_detection_benchmark import DEFAULT_BENCHMARK, load_benchmark, validate


FRAME_PATTERN = re.compile(r"^(?P<sequence>.+)-(?P<frame>\d+)\.png$")


def sequence_order(image):
    original_name = image.get("extra", {}).get("name", image["file_name"])
    match = FRAME_PATTERN.match(original_name)
    if match is None:
        return original_name, image["id"]
    return match.group("sequence"), int(match.group("frame"))


def run(benchmark_dir, output_path, split="test"):
    manifest, test_annotation_path, test_coco = load_benchmark(benchmark_dir)
    if split == "test":
        validation = validate(benchmark_dir)
        annotation_path, coco = test_annotation_path, test_coco
    else:
        annotation_path = benchmark_dir / "data" / split / "_annotations.coco.json"
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        validation = {"images": len(coco["images"]), "split": split}
    category_names = {entry["id"]: entry["name"] for entry in coco["categories"]}
    target_ids = {
        category_id
        for category_id, name in category_names.items()
        if name in set(manifest["target_categories"])
    }
    truths_by_image = {image["id"]: [] for image in coco["images"]}
    for annotation in coco["annotations"]:
        if annotation["category_id"] in target_ids:
            truths_by_image[annotation["image_id"]].append({
                "id": annotation["id"],
                "bbox": xywh_to_xyxy(annotation["bbox"]),
                "area": float(annotation["area"]),
            })

    sequences = defaultdict(list)
    for image in coco["images"]:
        sequence, frame_index = sequence_order(image)
        sequences[sequence].append((frame_index, image))

    player_tracker = PlayerTracker()
    pose_detector = PlayerPoseDetector()
    ball_tracker = BallTracker()
    predictions_by_image = {image["id"]: [] for image in coco["images"]}
    crop_count = 0
    adaptive_candidate_count = 0
    for sequence in sorted(sequences):
        ordered = [image for _, image in sorted(sequences[sequence])]
        frames = [
            cv2.imread(str(annotation_path.parent / image["file_name"]))
            for image in ordered
        ]
        if any(frame is None for frame in frames):
            raise ValueError(f"Failed to load an image in sequence {sequence}")
        player_tracker.tracker = sv.ByteTrack()
        player_tracks = player_tracker.get_object_tracks(frames)
        full_tracks = ball_tracker.get_object_tracks(frames, player_tracks=player_tracks)
        poses = pose_detector.get_player_poses(
            frames,
            player_tracks,
            ball_tracks=full_tracks,
        )
        enriched_players = attach_player_poses(player_tracks, poses)
        enhanced_tracks = ball_tracker.enhance_tracks_with_adaptive_crops(
            frames,
            full_tracks,
            enriched_players,
        )
        for image, track in zip(ordered, enhanced_tracks):
            info = track.get(1, {})
            crop_count += info.get("adaptive_crop_count", 0)
            adaptive_candidate_count += info.get("adaptive_candidates_added", 0)
            predictions_by_image[image["id"]] = [
                {
                    "image_id": image["id"],
                    "bbox": list(candidate["bbox"]),
                    "confidence": float(candidate["confidence"]),
                    "detection_source": candidate.get("detection_source", "full_frame"),
                }
                for candidate in info.get("raw_candidates", [])
            ]

    mixed_predictions = {
        image_id: [
            prediction
            for prediction in predictions
            if prediction["detection_source"] == "full_frame"
            or prediction["confidence"] >= 0.50
        ]
        for image_id, predictions in predictions_by_image.items()
    }
    report = {
        "benchmark_id": manifest["benchmark_id"],
        "mode": "full_frame_plus_adaptive_predicted_hand_rim_crops",
        "dataset": validation,
        "sequence_count": len(sequences),
        "adaptive_crop_count": crop_count,
        "adaptive_candidate_count": adaptive_candidate_count,
        "metrics_conf_025": score_predictions(predictions_by_image, truths_by_image, 0.25),
        "metrics_conf_050": score_predictions(predictions_by_image, truths_by_image, 0.50),
        "metrics_full_025_adaptive_050": score_predictions(
            mixed_predictions,
            truths_by_image,
            0.25,
        ),
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score adaptive ball ROI inference on NBA sequences.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    args = parser.parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    output = args.output or benchmark_dir / (
        "adaptive_report.json" if args.split == "test" else "adaptive_valid_report.json"
    )
    print(json.dumps(run(benchmark_dir, output.resolve(), args.split), indent=2))
