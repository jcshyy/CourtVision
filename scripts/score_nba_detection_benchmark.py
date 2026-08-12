import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_nba_detection_benchmark import DEFAULT_BENCHMARK, load_benchmark, validate


DEFAULT_MODEL = ROOT / "backend" / "models" / "ball_detector_model.pt"
IOU_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.5, 0.96, 0.05))


def xywh_to_xyxy(box):
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def box_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_predictions(predictions_by_image, truths_by_image, iou_threshold, confidence=0.0):
    ranked = sorted(
        (
            prediction
            for predictions in predictions_by_image.values()
            for prediction in predictions
            if prediction["confidence"] >= confidence
        ),
        key=lambda prediction: prediction["confidence"],
        reverse=True,
    )
    matched_truth = defaultdict(set)
    matches = []
    false_positives = []
    for prediction in ranked:
        image_id = prediction["image_id"]
        candidates = [
            (box_iou(prediction["bbox"], truth["bbox"]), truth_index)
            for truth_index, truth in enumerate(truths_by_image.get(image_id, []))
            if truth_index not in matched_truth[image_id]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            matched_truth[image_id].add(best_index)
            matches.append((prediction, truths_by_image[image_id][best_index], best_iou))
        else:
            false_positives.append(prediction)
    total_truth = sum(len(truths) for truths in truths_by_image.values())
    return {
        "ranked": ranked,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": total_truth - len(matches),
        "total_truth": total_truth,
    }


def average_precision(predictions_by_image, truths_by_image, iou_threshold):
    matched = match_predictions(predictions_by_image, truths_by_image, iou_threshold)
    if not matched["total_truth"]:
        return 0.0
    match_keys = {id(prediction) for prediction, _, _ in matched["matches"]}
    true_positive = np.asarray(
        [1.0 if id(prediction) in match_keys else 0.0 for prediction in matched["ranked"]]
    )
    if true_positive.size == 0:
        return 0.0
    false_positive = 1.0 - true_positive
    recall = np.cumsum(true_positive) / matched["total_truth"]
    precision = np.cumsum(true_positive) / np.maximum(
        np.cumsum(true_positive) + np.cumsum(false_positive), 1e-12
    )
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changing = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changing + 1] - recall[changing]) * precision[changing + 1]))


def size_bucket(area):
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def build_error_analysis(coco, predictions_by_image, truths_by_image, confidence=0.25):
    images = {image["id"]: image for image in coco["images"]}
    fixed = match_predictions(predictions_by_image, truths_by_image, 0.5, confidence)
    matched_truth_ids = {truth["id"] for _, truth, _ in fixed["matches"]}
    false_positive_ids = {id(prediction) for prediction in fixed["false_positives"]}
    miss_reasons = defaultdict(int)
    false_positive_reasons = defaultdict(int)
    cases = []

    for image_id, image in images.items():
        truths = truths_by_image.get(image_id, [])
        predictions = predictions_by_image.get(image_id, [])
        missed = []
        for truth in truths:
            if truth["id"] in matched_truth_ids:
                continue
            overlaps = sorted(
                ((box_iou(prediction["bbox"], truth["bbox"]), prediction) for prediction in predictions),
                key=lambda item: item[0],
                reverse=True,
            )
            best_iou, best_prediction = overlaps[0] if overlaps else (0.0, None)
            if best_iou >= 0.5:
                reason = "low_confidence"
            elif best_iou >= 0.1:
                reason = "poor_localization"
            else:
                reason = "no_candidate"
            miss_reasons[reason] += 1
            missed.append({
                **truth,
                "reason": reason,
                "best_candidate_iou": best_iou,
                "best_candidate_confidence": (
                    best_prediction["confidence"] if best_prediction is not None else None
                ),
            })

        false_positives = []
        for prediction in predictions:
            if id(prediction) not in false_positive_ids:
                continue
            best_truth_iou = max(
                (box_iou(prediction["bbox"], truth["bbox"]) for truth in truths),
                default=0.0,
            )
            reason = "localization_or_duplicate" if best_truth_iou >= 0.1 else "background"
            false_positive_reasons[reason] += 1
            false_positives.append({
                **prediction,
                "reason": reason,
                "best_truth_iou": best_truth_iou,
            })

        if missed or false_positives:
            cases.append({
                "image_id": image_id,
                "file_name": image["file_name"],
                "missed_truths": missed,
                "false_positives": false_positives,
                "truths": truths,
                "predictions_at_threshold": [
                    prediction for prediction in predictions
                    if prediction["confidence"] >= confidence
                ],
            })

    return {
        "confidence": confidence,
        "iou_threshold": 0.5,
        "miss_reason_counts": dict(sorted(miss_reasons.items())),
        "false_positive_reason_counts": dict(sorted(false_positive_reasons.items())),
        "case_count": len(cases),
        "cases": cases,
    }


def score_predictions(predictions_by_image, truths_by_image, confidence=0.25):
    ap_by_iou = {
        f"{threshold:.2f}": average_precision(predictions_by_image, truths_by_image, threshold)
        for threshold in IOU_THRESHOLDS
    }
    fixed = match_predictions(predictions_by_image, truths_by_image, 0.5, confidence)
    true_positives = len(fixed["matches"])
    false_positives = len(fixed["false_positives"])
    false_negatives = fixed["false_negatives"]
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / fixed["total_truth"] if fixed["total_truth"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    center_errors = []
    matched_truth_ids = set()
    for prediction, truth, _ in fixed["matches"]:
        prediction_center = ((prediction["bbox"][0] + prediction["bbox"][2]) / 2, (prediction["bbox"][1] + prediction["bbox"][3]) / 2)
        truth_center = ((truth["bbox"][0] + truth["bbox"][2]) / 2, (truth["bbox"][1] + truth["bbox"][3]) / 2)
        center_errors.append(math.dist(prediction_center, truth_center))
        matched_truth_ids.add(truth["id"])

    bucket_totals = defaultdict(int)
    bucket_matches = defaultdict(int)
    for truths in truths_by_image.values():
        for truth in truths:
            bucket = size_bucket(truth["area"])
            bucket_totals[bucket] += 1
            if truth["id"] in matched_truth_ids:
                bucket_matches[bucket] += 1

    positive_images = [image_id for image_id, truths in truths_by_image.items() if truths]
    negative_images = [image_id for image_id, truths in truths_by_image.items() if not truths]
    top_candidate_hits = 0
    for image_id in positive_images:
        candidates = [p for p in predictions_by_image.get(image_id, []) if p["confidence"] >= confidence]
        if candidates:
            top = max(candidates, key=lambda prediction: prediction["confidence"])
            if max((box_iou(top["bbox"], truth["bbox"]) for truth in truths_by_image[image_id]), default=0.0) >= 0.5:
                top_candidate_hits += 1
    negative_false_positives = sum(
        len([p for p in predictions_by_image.get(image_id, []) if p["confidence"] >= confidence])
        for image_id in negative_images
    )

    return {
        "ap50": ap_by_iou["0.50"],
        "map50_95": float(np.mean(list(ap_by_iou.values()))),
        "ap_by_iou": ap_by_iou,
        "fixed_confidence": confidence,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "mean_center_error_px": float(np.mean(center_errors)) if center_errors else None,
        "median_center_error_px": float(np.median(center_errors)) if center_errors else None,
        "recall_by_size": {
            bucket: bucket_matches[bucket] / total if total else None
            for bucket, total in sorted(bucket_totals.items())
        },
        "truth_count_by_size": dict(sorted(bucket_totals.items())),
        "positive_images": len(positive_images),
        "negative_images": len(negative_images),
        "negative_frame_false_positives": negative_false_positives,
        "top_candidate_accuracy": top_candidate_hits / len(positive_images) if positive_images else None,
    }


def sweep_confidence_thresholds(predictions_by_image, truths_by_image):
    rows = []
    for confidence in (
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    ):
        matched = match_predictions(predictions_by_image, truths_by_image, 0.5, confidence)
        true_positives = len(matched["matches"])
        false_positives = len(matched["false_positives"])
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
        recall = true_positives / matched["total_truth"] if matched["total_truth"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "confidence": confidence,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": matched["false_negatives"],
        })
    return rows


def markdown_report(report):
    metrics = report["metrics"]
    overlap = report["benchmark_integrity"]["training_overlap_status"]
    lines = [
        "# NBA Ball Detection Baseline",
        "",
        f"- Model: `{report['model']['path']}`",
        f"- Input size: {report['model']['imgsz']}",
        f"- Test images: {report['dataset']['images']}",
        f"- Ball boxes: {report['dataset']['target_annotations']}",
        f"- Training overlap status: **{overlap}**",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| AP50 | {metrics['ap50']:.3f} |",
        f"| mAP50:95 | {metrics['map50_95']:.3f} |",
        f"| Precision @ {metrics['fixed_confidence']:.2f} | {metrics['precision']:.3f} |",
        f"| Recall @ {metrics['fixed_confidence']:.2f} | {metrics['recall']:.3f} |",
        f"| F1 @ {metrics['fixed_confidence']:.2f} | {metrics['f1']:.3f} |",
        f"| Top-candidate accuracy | {metrics['top_candidate_accuracy']:.3f} |",
        f"| Mean center error | {metrics['mean_center_error_px']:.2f} px |" if metrics["mean_center_error_px"] is not None else "| Mean center error | n/a |",
        f"| Negative-frame false positives | {metrics['negative_frame_false_positives']} |",
        "",
        "## Recall by object size",
        "",
    ]
    for bucket, value in metrics["recall_by_size"].items():
        lines.append(f"- {bucket}: {value:.3f} ({metrics['truth_count_by_size'][bucket]} boxes)")
    lines.extend([
        "",
        "> This is a correlated-frame regression set from a small number of game segments. It is not a broad generalization estimate. Exact overlap with the checkpoint's original training images has not yet been audited.",
        "",
    ])
    return "\n".join(lines)


def run(
    benchmark_dir,
    model_path,
    output_json,
    output_markdown,
    confidence,
    imgsz,
    batch_size,
    output_errors=None,
    split="test",
):
    manifest, test_annotation_path, test_coco = load_benchmark(benchmark_dir)
    if split == "test":
        validation = validate(benchmark_dir)
        annotation_path, coco = test_annotation_path, test_coco
    else:
        annotation_path = benchmark_dir / "data" / split / "_annotations.coco.json"
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        target_names = set(manifest["target_categories"])
        category_names = {entry["id"]: entry["name"] for entry in coco["categories"]}
        validation = {
            "images": len(coco["images"]),
            "target_annotations": sum(
                category_names[annotation["category_id"]] in target_names
                for annotation in coco["annotations"]
            ),
            "split": split,
        }
    categories = {entry["id"]: entry["name"] for entry in coco["categories"]}
    target_ids = {category_id for category_id, name in categories.items() if name in manifest["target_categories"]}
    truths_by_image = {image["id"]: [] for image in coco["images"]}
    for annotation in coco["annotations"]:
        if annotation["category_id"] in target_ids:
            truths_by_image[annotation["image_id"]].append({
                "id": annotation["id"],
                "bbox": xywh_to_xyxy(annotation["bbox"]),
                "area": float(annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3])),
            })

    model = YOLO(str(model_path))
    ball_class_ids = [class_id for class_id, name in model.names.items() if name.lower() == "ball"]
    if not ball_class_ids:
        raise ValueError(f"Model has no Ball class: {model.names}")
    image_paths = [annotation_path.parent / image["file_name"] for image in coco["images"]]
    predictions_by_image = {image["id"]: [] for image in coco["images"]}
    results = model.predict(
        [str(path) for path in image_paths],
        conf=0.001,
        classes=ball_class_ids,
        imgsz=imgsz,
        batch=batch_size,
        max_det=50,
        verbose=False,
    )
    for image, result in zip(coco["images"], results):
        for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            predictions_by_image[image["id"]].append({
                "image_id": image["id"],
                "bbox": list(map(float, box)),
                "confidence": float(score),
            })

    try:
        reported_model_path = str(model_path.relative_to(ROOT))
    except ValueError:
        reported_model_path = str(model_path)
    confidence_sweep = sweep_confidence_thresholds(predictions_by_image, truths_by_image)
    report = {
        "benchmark_id": manifest["benchmark_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": validation,
        "benchmark_integrity": manifest["integrity"],
        "model": {"path": reported_model_path, "class_ids": ball_class_ids, "imgsz": imgsz},
        "metrics": score_predictions(predictions_by_image, truths_by_image, confidence),
        "confidence_sweep": confidence_sweep,
        "best_f1_threshold": max(confidence_sweep, key=lambda row: row["f1"]),
    }
    error_analysis = build_error_analysis(coco, predictions_by_image, truths_by_image, confidence)
    error_analysis["split"] = split
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(markdown_report(report), encoding="utf-8")
    if output_errors is not None:
        output_errors.write_text(json.dumps(error_analysis, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the production ball detector on NBA frames.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--output-errors", type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    args = parser.parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    report_stem = "baseline_report" if args.split == "test" else f"{args.split}_report"
    output_json = args.output_json or benchmark_dir / f"{report_stem}.json"
    output_markdown = args.output_markdown or benchmark_dir / f"{report_stem}.md"
    output_errors = args.output_errors or benchmark_dir / f"error_analysis_{args.split}.json"
    report = run(
        benchmark_dir,
        args.model.resolve(),
        output_json.resolve(),
        output_markdown.resolve(),
        args.confidence,
        args.imgsz,
        args.batch_size,
        output_errors.resolve(),
        args.split,
    )
    print(json.dumps(report["metrics"], indent=2))
