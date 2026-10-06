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
from backend.app.tracking.ball_tracker import BALL_DETECTOR_BACKENDS
from backend.app.tracking.player_tracker import PLAYER_DETECTOR_BACKENDS
from scripts.score_nba_detection_benchmark import score_predictions, xywh_to_xyxy
from scripts.validate_nba_detection_benchmark import DEFAULT_BENCHMARK, load_benchmark, validate


FRAME_PATTERN = re.compile(r"^(?P<sequence>.+)-(?P<frame>\d+)\.png$")


def is_adaptive(prediction):
    return str(prediction.get("detection_source", "")).startswith("adaptive")


def candidate_predictions(scale, image_id, candidates):
    """Map candidates from inference coordinates back to annotation coordinates."""
    scale_x, scale_y = scale
    return [
        {
            "image_id": image_id,
            "bbox": [
                candidate["bbox"][0] * scale_x,
                candidate["bbox"][1] * scale_y,
                candidate["bbox"][2] * scale_x,
                candidate["bbox"][3] * scale_y,
            ],
            "confidence": float(candidate["confidence"]),
            "detection_source": candidate.get("detection_source", "full_frame"),
        }
        for candidate in candidates
    ]


def score_center_hits(predictions_by_image, truths_by_image, confidence=0.25):
    """Score point-like detections: a hit is a predicted center inside an unmatched truth box.

    IoU@0.5 cannot credit WASB, whose heatmap peaks become fixed-size squares
    that are smaller than the stretched ground-truth boxes.
    """
    true_positives = false_positives = total_truth = 0
    for image_id, truths in truths_by_image.items():
        total_truth += len(truths)
        unmatched = list(truths)
        predictions = sorted(
            (p for p in predictions_by_image.get(image_id, []) if p["confidence"] >= confidence),
            key=lambda p: p["confidence"],
            reverse=True,
        )
        for prediction in predictions:
            center_x = (prediction["bbox"][0] + prediction["bbox"][2]) / 2
            center_y = (prediction["bbox"][1] + prediction["bbox"][3]) / 2
            hit = next(
                (
                    truth for truth in unmatched
                    if truth["bbox"][0] <= center_x <= truth["bbox"][2]
                    and truth["bbox"][1] <= center_y <= truth["bbox"][3]
                ),
                None,
            )
            if hit is None:
                false_positives += 1
            else:
                unmatched.remove(hit)
                true_positives += 1
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / total_truth if total_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confidence": confidence,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": total_truth - true_positives,
    }


def default_output_name(split, scene_backend, ball_backend, restore_size=None):
    stem = "adaptive_report" if split == "test" else "adaptive_valid_report"
    suffix = f"_{restore_size[0]}x{restore_size[1]}" if restore_size else ""
    if (scene_backend, ball_backend) == ("current", "yolo") and not restore_size:
        return f"{stem}.json"
    return f"{stem}_{scene_backend}_{ball_backend}{suffix}.json"


def sequence_order(image):
    original_name = image.get("extra", {}).get("name", image["file_name"])
    match = FRAME_PATTERN.match(original_name)
    if match is None:
        return original_name, image["id"]
    return match.group("sequence"), int(match.group("frame"))


def run(
    benchmark_dir,
    output_path,
    split="test",
    scene_backend="current",
    ball_backend="yolo",
    restore_size=None,
):
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

    # Mirror main.py: E-BARD shares one scene pass with the ball tracker.
    player_tracker = PlayerTracker(detector_backend=scene_backend)
    pose_detector = PlayerPoseDetector()
    share_scene_inference = scene_backend == "ebard" and ball_backend in ("yolo", "hybrid")
    ball_tracker = BallTracker(
        detector_backend=ball_backend,
        semantic_model=player_tracker.model if share_scene_inference else None,
        semantic_detector_backend=scene_backend if share_scene_inference else "current",
    )
    image_ids = [image["id"] for image in coco["images"]]
    predictions_by_image = {image_id: [] for image_id in image_ids}
    full_frame_by_image = {image_id: [] for image_id in image_ids}
    semantic_by_image = {image_id: [] for image_id in image_ids}
    wasb_by_image = {image_id: [] for image_id in image_ids}
    selected_by_image = {image_id: [] for image_id in image_ids}
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
        scale = (1.0, 1.0)
        if restore_size:
            # Roboflow stretched 16:9 broadcast frames to 640x640; undo that
            # before inference and map boxes back to annotation coordinates.
            source_height, source_width = frames[0].shape[:2]
            scale = (source_width / restore_size[0], source_height / restore_size[1])
            frames = [cv2.resize(frame, restore_size, interpolation=cv2.INTER_LINEAR) for frame in frames]
        player_tracker.tracker = sv.ByteTrack()
        scene_detections = (
            player_tracker.detect_frames(frames) if share_scene_inference else None
        )
        player_tracks = player_tracker.get_object_tracks(
            frames,
            detections_provider=(lambda: scene_detections) if share_scene_inference else None,
        )
        full_tracks = ball_tracker.get_object_tracks(
            frames,
            player_tracks=player_tracks,
            detections=scene_detections,
        )
        if ball_tracker.wasb_detector is not None:
            # Re-run WASB alone so its contribution is not hidden by candidate merging.
            for image, candidates in zip(ordered, ball_tracker.wasb_detector.detect_frames(frames, step=1)):
                wasb_by_image[image["id"]] = candidate_predictions(scale, image["id"], candidates)
        for image, track in zip(ordered, full_tracks):
            info = track.get(1, {})
            full_frame_by_image[image["id"]] = candidate_predictions(
                scale, image["id"], info.get("raw_candidates", [])
            )
            semantic_by_image[image["id"]] = candidate_predictions(
                scale,
                image["id"],
                info.get("semantic_raw_candidates", info.get("raw_candidates", []))
                if ball_tracker.model is not None
                else [],
            )
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
        filtered_tracks = ball_tracker.remove_wrong_detections(
            enhanced_tracks,
            player_tracks=enriched_players,
        )
        for image, track, filtered in zip(ordered, enhanced_tracks, filtered_tracks):
            info = track.get(1, {})
            crop_count += info.get("adaptive_crop_count", 0)
            adaptive_candidate_count += info.get("adaptive_candidates_added", 0)
            predictions_by_image[image["id"]] = candidate_predictions(
                scale, image["id"], info.get("raw_candidates", [])
            )
            selected = filtered.get(1, {})
            if selected.get("bbox") is not None:
                selected_by_image[image["id"]] = candidate_predictions(scale, image["id"], [selected])

    mixed_predictions = {
        image_id: [
            prediction
            for prediction in predictions
            if not is_adaptive(prediction) or prediction["confidence"] >= 0.50
        ]
        for image_id, predictions in predictions_by_image.items()
    }
    report = {
        "benchmark_id": manifest["benchmark_id"],
        "mode": "full_frame_plus_adaptive_predicted_hand_rim_crops",
        "scene_detector_backend": scene_backend,
        "ball_detector_backend": ball_backend,
        "shared_scene_inference": share_scene_inference,
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
        # Candidate pools before adaptive crops, split by source.
        "metrics_full_frame_pool_025": score_predictions(full_frame_by_image, truths_by_image, 0.25),
        "metrics_semantic_only_025": score_predictions(semantic_by_image, truths_by_image, 0.25),
        "metrics_wasb_only_025": (
            score_predictions(wasb_by_image, truths_by_image, 0.25)
            if ball_tracker.wasb_detector is not None
            else None
        ),
        # The single temporally selected ball per frame that production consumes.
        "metrics_selected_track": score_predictions(selected_by_image, truths_by_image, 0.0),
        "restore_size": list(restore_size) if restore_size else None,
        "center_hit_metrics": {
            "all_candidates_025": score_center_hits(predictions_by_image, truths_by_image, 0.25),
            "full_frame_pool_025": score_center_hits(full_frame_by_image, truths_by_image, 0.25),
            "semantic_only_025": score_center_hits(semantic_by_image, truths_by_image, 0.25),
            "wasb_only_025": (
                score_center_hits(wasb_by_image, truths_by_image, 0.25)
                if ball_tracker.wasb_detector is not None
                else None
            ),
            "selected_track": score_center_hits(selected_by_image, truths_by_image, 0.0),
        },
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score adaptive ball ROI inference on NBA sequences.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument(
        "--scene-detector-backend",
        choices=PLAYER_DETECTOR_BACKENDS,
        default="current",
        help="Use 'ebard' with --ball-detector-backend hybrid to score the production default.",
    )
    parser.add_argument("--ball-detector-backend", choices=BALL_DETECTOR_BACKENDS, default="yolo")
    parser.add_argument(
        "--restore-size",
        type=lambda value: tuple(int(part) for part in value.lower().split("x")),
        help="Resize stretched 640x640 frames to WIDTHxHEIGHT (e.g. 1280x720) before inference.",
    )
    args = parser.parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    output = args.output or benchmark_dir / default_output_name(
        args.split, args.scene_detector_backend, args.ball_detector_backend, args.restore_size
    )
    report = run(
        benchmark_dir,
        output.resolve(),
        args.split,
        args.scene_detector_backend,
        args.ball_detector_backend,
        args.restore_size,
    )
    print(json.dumps(report, indent=2))
