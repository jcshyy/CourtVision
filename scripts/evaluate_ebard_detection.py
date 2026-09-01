"""Benchmark CourtVision and E-BARD detectors on E-BARD ground truth.

The evaluator normalizes each checkpoint's class vocabulary before scoring, so
CourtVision's seven-class model can be compared fairly with E-BARD's four-class
model. Average precision uses COCO-style 101-point interpolation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import EBARD_YOLO_DETECTOR_PATH, PLAYER_DETECTOR_PATH


DEFAULT_DATASET = ROOT / "benchmarks" / "ebard_detection" / "data"
DEFAULT_OUTPUT = ROOT / "runs" / "ebard_detection" / "detector_report.json"
CLASS_NAMES = ("basketball", "hoop", "player", "referee")
CLASS_ALIASES = {
    "basketball": {"ball", "basketball"},
    "hoop": {"hoop", "rim"},
    "player": {"player", "players"},
    "referee": {"ref", "refs", "referee", "referees"},
}
MODEL_SPECS = {
    "current": {
        "path": PLAYER_DETECTOR_PATH,
        "operating_confidence": 0.50,
    },
    "ebard": {
        "path": EBARD_YOLO_DETECTOR_PATH,
        "operating_confidence": 0.25,
    },
}
IOU_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.50, 0.96, 0.05))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CourtVision and E-BARD on annotated NBA frames."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help=(
            "Physical YOLO folder to evaluate. E-BARD's published eval uses test; "
            "its data.yaml names that folder val."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument(
        "--device",
        default="auto",
        help="Ultralytics device, such as cpu, 0, or auto.",
    )
    parser.add_argument(
        "--failure-images",
        type=int,
        default=12,
        help="Number of highest-error side-by-side frames to render.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N frames for a quick smoke test.",
    )
    return parser.parse_args()


def canonical_class_map(model_names):
    items = model_names.items() if hasattr(model_names, "items") else enumerate(model_names)
    normalized_aliases = {
        alias: canonical
        for canonical, aliases in CLASS_ALIASES.items()
        for alias in aliases
    }
    return {
        int(class_id): normalized_aliases[name.strip().lower()]
        for class_id, name in items
        if name.strip().lower() in normalized_aliases
    }


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


def yolo_box_to_xyxy(values, width, height):
    center_x, center_y, box_width, box_height = map(float, values)
    x1 = (center_x - box_width / 2) * width
    y1 = (center_y - box_height / 2) * height
    x2 = (center_x + box_width / 2) * width
    y2 = (center_y + box_height / 2) * height
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def load_ground_truth(dataset_root, split="test", limit=None):
    split_root = Path(dataset_root) / "yolo" / split
    image_root = split_root / "images"
    label_root = split_root / "labels"
    image_paths = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in image_root.glob(pattern)
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(
            f"No E-BARD images found at {image_root}. "
            "Run scripts/prepare_ebard_detection_dataset.py first."
        )

    samples = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Unable to read benchmark image: {image_path}")
        height, width = image.shape[:2]
        label_path = label_root / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing ground-truth label: {label_path}")
        entities = []
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"Malformed label {label_path}:{line_number}")
            class_id = int(fields[0])
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(
                    f"Unknown E-BARD class {class_id} at {label_path}:{line_number}"
                )
            entities.append(
                {
                    "class": CLASS_NAMES[class_id],
                    "bbox": yolo_box_to_xyxy(fields[1:], width, height),
                }
            )
        samples.append(
            {
                "image_path": image_path,
                "width": width,
                "height": height,
                "ground_truth": entities,
            }
        )
    return samples


def result_predictions(result, class_map):
    if result.boxes is None:
        return []
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    return [
        {
            "class": class_map[class_id],
            "bbox": [float(value) for value in box],
            "confidence": float(confidence),
        }
        for box, confidence, class_id in zip(boxes, confidences, class_ids)
        if class_id in class_map
    ]


def _match_frame_class(ground_truth, predictions, iou_threshold):
    predictions = sorted(predictions, key=lambda item: item[1]["confidence"], reverse=True)
    unmatched_ground_truth = set(range(len(ground_truth)))
    matches = []
    unmatched_predictions = []
    for original_index, prediction in predictions:
        candidates = [
            (box_iou(prediction["bbox"], ground_truth[index]["bbox"]), index)
            for index in unmatched_ground_truth
        ]
        best_iou, best_index = max(candidates, default=(0.0, None))
        if best_index is not None and best_iou >= iou_threshold:
            unmatched_ground_truth.remove(best_index)
            matches.append((best_index, original_index, best_iou))
        else:
            unmatched_predictions.append(original_index)
    return matches, sorted(unmatched_ground_truth), unmatched_predictions


def frame_matches(sample, predictions, confidence_threshold, iou_threshold=0.5):
    outcomes = {}
    for class_name in CLASS_NAMES:
        ground_truth = [
            item for item in sample["ground_truth"] if item["class"] == class_name
        ]
        class_predictions = [
            (index, item)
            for index, item in enumerate(predictions)
            if item["class"] == class_name
            and item["confidence"] >= confidence_threshold
        ]
        matches, missed, false_positives = _match_frame_class(
            ground_truth, class_predictions, iou_threshold
        )
        outcomes[class_name] = {
            "ground_truth": ground_truth,
            "matches": matches,
            "missed_ground_truth": missed,
            "false_positive_predictions": false_positives,
        }
    return outcomes


def detection_metrics(samples, predictions_by_frame, confidence_threshold, iou_threshold=0.5):
    counts = {
        class_name: {"tp": 0, "fp": 0, "fn": 0}
        for class_name in CLASS_NAMES
    }
    for sample, predictions in zip(samples, predictions_by_frame):
        outcomes = frame_matches(
            sample, predictions, confidence_threshold, iou_threshold
        )
        for class_name, outcome in outcomes.items():
            counts[class_name]["tp"] += len(outcome["matches"])
            counts[class_name]["fp"] += len(
                outcome["false_positive_predictions"]
            )
            counts[class_name]["fn"] += len(outcome["missed_ground_truth"])

    per_class = {}
    for class_name, class_counts in counts.items():
        tp, fp, fn = (class_counts[key] for key in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[class_name] = {
            **class_counts,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": tp + fn,
        }
    total = {
        key: sum(item[key] for item in counts.values()) for key in ("tp", "fp", "fn")
    }
    micro_precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
    micro_recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "per_class": per_class,
        "macro_f1": round(float(np.mean([item["f1"] for item in per_class.values()])), 6),
        "micro": {
            **total,
            "precision": round(micro_precision, 6),
            "recall": round(micro_recall, 6),
            "f1": round(micro_f1, 6),
        },
    }


def interpolated_average_precision(recalls, precisions):
    if not recalls:
        return 0.0
    recalls_array = np.asarray(recalls)
    precisions_array = np.asarray(precisions)
    return float(
        np.mean(
            [
                np.max(precisions_array[recalls_array >= recall], initial=0.0)
                for recall in np.linspace(0.0, 1.0, 101)
            ]
        )
    )


def class_average_precision(samples, predictions_by_frame, class_name, iou_threshold):
    ground_truth_by_frame = {
        frame_index: [
            item for item in sample["ground_truth"] if item["class"] == class_name
        ]
        for frame_index, sample in enumerate(samples)
    }
    ground_truth_count = sum(len(items) for items in ground_truth_by_frame.values())
    detections = sorted(
        (
            (prediction["confidence"], frame_index, prediction)
            for frame_index, predictions in enumerate(predictions_by_frame)
            for prediction in predictions
            if prediction["class"] == class_name
        ),
        reverse=True,
        key=lambda item: item[0],
    )
    matched_by_frame = {frame_index: set() for frame_index in ground_truth_by_frame}
    true_positives = []
    false_positives = []
    for _, frame_index, prediction in detections:
        available = [
            index
            for index in range(len(ground_truth_by_frame[frame_index]))
            if index not in matched_by_frame[frame_index]
        ]
        candidates = [
            (
                box_iou(
                    prediction["bbox"],
                    ground_truth_by_frame[frame_index][index]["bbox"],
                ),
                index,
            )
            for index in available
        ]
        best_iou, best_index = max(candidates, default=(0.0, None))
        is_match = best_index is not None and best_iou >= iou_threshold
        if is_match:
            matched_by_frame[frame_index].add(best_index)
        true_positives.append(1 if is_match else 0)
        false_positives.append(0 if is_match else 1)
    if ground_truth_count == 0:
        return 0.0
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recalls = (cumulative_tp / ground_truth_count).tolist()
    precisions = (cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)).tolist()
    return interpolated_average_precision(recalls, precisions)


def average_precision_metrics(samples, predictions_by_frame):
    per_class = {}
    for class_name in CLASS_NAMES:
        by_iou = {
            f"{threshold:.2f}": class_average_precision(
                samples, predictions_by_frame, class_name, threshold
            )
            for threshold in IOU_THRESHOLDS
        }
        per_class[class_name] = {
            "ap50": round(by_iou["0.50"], 6),
            "map50_95": round(float(np.mean(list(by_iou.values()))), 6),
            "ap_by_iou": {key: round(value, 6) for key, value in by_iou.items()},
        }
    return {
        "method": "COCO-style 101-point interpolation",
        "per_class": per_class,
        "map50": round(float(np.mean([item["ap50"] for item in per_class.values()])), 6),
        "map50_95": round(
            float(np.mean([item["map50_95"] for item in per_class.values()])), 6
        ),
    }


def threshold_sweep(samples, predictions_by_frame, operating_confidence):
    thresholds = sorted(
        set(round(float(value), 3) for value in np.arange(0.05, 0.96, 0.05))
        | {round(float(operating_confidence), 3)}
    )
    rows = [
        detection_metrics(samples, predictions_by_frame, threshold)
        for threshold in thresholds
    ]
    best = max(rows, key=lambda item: (item["macro_f1"], item["micro"]["f1"]))
    return {
        "best_macro_f1": {
            "confidence_threshold": best["confidence_threshold"],
            "macro_f1": best["macro_f1"],
            "micro_f1": best["micro"]["f1"],
        },
        "rows": [
            {
                "confidence_threshold": row["confidence_threshold"],
                "macro_f1": row["macro_f1"],
                "micro_f1": row["micro"]["f1"],
            }
            for row in rows
        ],
    }


def benchmark_detector(name, spec, samples, *, min_confidence, nms_iou, imgsz, batch_size, device):
    model_path = Path(spec["path"])
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {name} detector: {model_path}")
    model = YOLO(str(model_path))
    class_map = canonical_class_map(model.names)
    missing = sorted(set(CLASS_NAMES) - set(class_map.values()))
    if missing:
        raise ValueError(f"{name} detector is missing benchmark classes: {missing}")

    image_paths = [str(sample["image_path"]) for sample in samples]
    model.predict(
        image_paths[:1],
        conf=min_confidence,
        iou=nms_iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    predictions_by_frame = []
    started = time.perf_counter()
    for start in range(0, len(image_paths), batch_size):
        results = model.predict(
            image_paths[start : start + batch_size],
            conf=min_confidence,
            iou=nms_iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        predictions_by_frame.extend(
            result_predictions(result, class_map) for result in results
        )
    elapsed_seconds = time.perf_counter() - started
    operating_confidence = float(spec["operating_confidence"])
    report = {
        "model_path": str(model_path.resolve()),
        "parameter_count": sum(parameter.numel() for parameter in model.model.parameters()),
        "class_map": {str(key): value for key, value in class_map.items()},
        "inference": {
            "device": str(device),
            "imgsz": imgsz,
            "batch_size": batch_size,
            "minimum_prediction_confidence": min_confidence,
            "nms_iou": nms_iou,
            "warmup_excluded": True,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "frames_per_second": round(len(samples) / elapsed_seconds, 3),
        },
        "operating_point": detection_metrics(
            samples, predictions_by_frame, operating_confidence
        ),
        "average_precision": average_precision_metrics(samples, predictions_by_frame),
        "threshold_sweep": threshold_sweep(
            samples, predictions_by_frame, operating_confidence
        ),
    }
    return report, predictions_by_frame


def _draw_panel(sample, predictions, title, confidence_threshold):
    image = cv2.imread(str(sample["image_path"]))
    outcomes = frame_matches(sample, predictions, confidence_threshold)
    matched_prediction_indices = {
        prediction_index
        for outcome in outcomes.values()
        for _, prediction_index, _ in outcome["matches"]
    }
    missed = {
        (class_name, index)
        for class_name, outcome in outcomes.items()
        for index in outcome["missed_ground_truth"]
    }
    for class_name, outcome in outcomes.items():
        for index, item in enumerate(outcome["ground_truth"]):
            color = (0, 0, 255) if (class_name, index) in missed else (0, 180, 0)
            x1, y1, x2, y2 = map(int, item["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                image,
                f"GT {class_name}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    for index, prediction in enumerate(predictions):
        if prediction["confidence"] < confidence_threshold:
            continue
        color = (255, 170, 0) if index in matched_prediction_indices else (0, 140, 255)
        x1, y1, x2, y2 = map(int, prediction["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{prediction['class']} {prediction['confidence']:.2f}",
            (x1, min(image.shape[0] - 8, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(
        image,
        title,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if image.shape[1] > 960:
        scale = 960 / image.shape[1]
        image = cv2.resize(
            image,
            (960, round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image


def render_failure_cases(samples, predictions, output_root, limit):
    if limit <= 0:
        return []
    scores = []
    for frame_index, sample in enumerate(samples):
        error_count = 0
        for name, spec in MODEL_SPECS.items():
            outcomes = frame_matches(
                sample,
                predictions[name][frame_index],
                spec["operating_confidence"],
            )
            error_count += sum(
                len(outcome["missed_ground_truth"])
                + len(outcome["false_positive_predictions"])
                for outcome in outcomes.values()
            )
        scores.append((error_count, frame_index))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    written = []
    for rank, (error_count, frame_index) in enumerate(
        sorted(scores, reverse=True)[:limit], start=1
    ):
        sample = samples[frame_index]
        panels = [
            _draw_panel(
                sample,
                predictions[name][frame_index],
                f"{name} | conf {MODEL_SPECS[name]['operating_confidence']:.2f}",
                MODEL_SPECS[name]["operating_confidence"],
            )
            for name in ("current", "ebard")
        ]
        if panels[0].shape[0] != panels[1].shape[0]:
            target_height = min(panel.shape[0] for panel in panels)
            panels = [
                cv2.resize(
                    panel,
                    (round(panel.shape[1] * target_height / panel.shape[0]), target_height),
                    interpolation=cv2.INTER_AREA,
                )
                for panel in panels
            ]
        combined = np.concatenate(panels, axis=1)
        filename = f"{rank:02d}_errors-{error_count}_{sample['image_path'].stem}.jpg"
        path = output_root / filename
        cv2.imwrite(str(path), combined)
        written.append(str(path.resolve()))
    return written


def dataset_summary(samples, dataset, split):
    counts = Counter(
        item["class"] for sample in samples for item in sample["ground_truth"]
    )
    return {
        "root": str(Path(dataset).resolve()),
        "physical_split": split,
        "upstream_data_yaml_alias": "val" if split == "test" else None,
        "frame_count": len(samples),
        "instance_count": sum(counts.values()),
        "instances_per_class": {name: counts[name] for name in CLASS_NAMES},
    }


def main():
    args = parse_args()
    device = 0 if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    samples = load_ground_truth(args.dataset, args.split, args.limit)
    detector_reports = {}
    predictions = {}
    for name, spec in MODEL_SPECS.items():
        print(f"Evaluating {name} on {len(samples)} E-BARD {args.split} frames...")
        detector_reports[name], predictions[name] = benchmark_detector(
            name,
            spec,
            samples,
            min_confidence=args.min_confidence,
            nms_iou=args.nms_iou,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            device=device,
        )
    failure_root = args.output.parent / "failure_cases"
    failure_cases = render_failure_cases(
        samples, predictions, failure_root, args.failure_images
    )
    report = {
        "schema_version": 1,
        "benchmark": "E-BARD object detection",
        "dataset": dataset_summary(samples, args.dataset, args.split),
        "metric_definition": {
            "operating_point": "greedy one-to-one matching at IoU >= 0.50",
            "average_precision": "COCO-style 101-point interpolated AP",
            "map50_95_iou_thresholds": list(IOU_THRESHOLDS),
        },
        "detectors": detector_reports,
        "failure_case_images": failure_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, detector in detector_reports.items():
        summary = detector["average_precision"]
        timing = detector["inference"]
        print(
            f"{name}: mAP50={summary['map50']:.4f}, "
            f"mAP50-95={summary['map50_95']:.4f}, "
            f"FPS={timing['frames_per_second']:.2f}"
        )
    print(f"Saved benchmark to {args.output}")


if __name__ == "__main__":
    main()
