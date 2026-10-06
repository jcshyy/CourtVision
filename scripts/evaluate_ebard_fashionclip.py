"""Evaluate CourtVision's FashionCLIP on E-BARD jersey-color crops."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = ROOT / "benchmarks" / "ebard_team_attribution" / "data"
DEFAULT_OUTPUT = (
    ROOT / "runs" / "ebard_team_attribution" / "fashionclip_test_report.json"
)
MODEL_NAME = "patrickjohncyh/fashion-clip"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce E-BARD FashionCLIP team-color evaluation."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def load_rows(dataset, split, limit=None):
    dataset = Path(dataset).resolve()
    labels_path = dataset / f"{split}_labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Missing {labels_path}; run scripts/prepare_ebard_team_dataset.py"
        )
    rows = []
    with labels_path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            relative = str(row["image_path"]).replace("\\", "/").lstrip("./")
            image_path = (dataset / relative).resolve()
            if dataset not in image_path.parents:
                raise ValueError(f"Dataset image escaped its root: {relative}")
            prefix = image_path.name.split("_")[0]
            colors_path = dataset / "colors" / f"{prefix}_color.json"
            colors = sorted(set(json.loads(colors_path.read_text(encoding="utf-8")).values()))
            rows.append(
                {
                    "image_path": image_path,
                    "label": str(row["label"]).lower(),
                    "prefix": prefix,
                    "colors": colors,
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def classification_metrics(truth, predictions):
    labels = sorted(set(truth) | set(predictions))
    confusion = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }
    for actual, predicted in zip(truth, predictions):
        confusion[actual][predicted] += 1

    per_class = {}
    for label in labels:
        true_positive = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[actual][label] for actual in labels)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    total = len(truth)
    accuracy = sum(actual == predicted for actual, predicted in zip(truth, predictions)) / total
    scored_labels = [label for label in labels if per_class[label]["support"]]
    macro_f1 = sum(per_class[label]["f1"] for label in scored_labels) / len(scored_labels)
    weighted_f1 = sum(
        per_class[label]["f1"] * per_class[label]["support"]
        for label in scored_labels
    ) / total
    return {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def feature_tensor(output):
    """Normalize Transformers 4 tensor and Transformers 5 model-output APIs."""
    if hasattr(output, "norm"):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        raise TypeError(f"Unsupported CLIP feature output: {type(output).__name__}")
    return pooled


def evaluate(dataset, split="test", *, batch_size=32, limit=None, allow_download=False):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    rows = load_rows(dataset, split, limit)
    if not rows:
        raise ValueError("The selected E-BARD split contains no rows")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(
        MODEL_NAME,
        local_files_only=not allow_download,
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME,
        local_files_only=not allow_download,
    )
    model.eval()

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["prefix"]].append(row)

    strategies = {
        "ebard_prompt": {
            "predictions": [],
            "truth": [],
            "prompt": "A photo of a {color} jersey",
            "include_referee": False,
        },
        "courtvision_prompt_with_referee": {
            "predictions": [],
            "truth": [],
            "prompt": "a basketball player wearing a primarily {color} team uniform",
            "include_referee": True,
        },
    }
    processed = 0
    for game_rows in grouped.values():
        colors = game_rows[0]["colors"]
        text_features = {}
        text_labels = {}
        for name, strategy in strategies.items():
            labels = list(colors)
            prompts = [strategy["prompt"].format(color=color) for color in colors]
            if strategy["include_referee"]:
                labels.append("referee")
                prompts.append(
                    "a basketball referee wearing a black and gray referee uniform"
                )
            inputs = processor(text=prompts, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = feature_tensor(model.get_text_features(**inputs))
                features = features / features.norm(dim=-1, keepdim=True)
            text_features[name] = features
            text_labels[name] = labels

        for start in range(0, len(game_rows), batch_size):
            batch = game_rows[start : start + batch_size]
            images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                image_features = feature_tensor(model.get_image_features(**inputs))
                image_features = image_features / image_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            for name, strategy in strategies.items():
                indices = (image_features @ text_features[name].T).argmax(dim=1)
                strategy["predictions"].extend(
                    text_labels[name][int(index)] for index in indices.cpu()
                )
                strategy["truth"].extend(row["label"] for row in batch)
            processed += len(batch)
            if processed % 100 < len(batch):
                print(f"Evaluated {processed}/{len(rows)} E-BARD crops")

    return {
        "schema_version": 1,
        "dataset": str(Path(dataset).resolve()),
        "split": split,
        "sample_count": len(rows),
        "model": MODEL_NAME,
        "device": device,
        "class_distribution": dict(sorted(Counter(row["label"] for row in rows).items())),
        "strategies": {
            name: {
                "prompt": strategy["prompt"],
                "includes_referee_prompt": strategy["include_referee"],
                **classification_metrics(strategy["truth"], strategy["predictions"]),
            }
            for name, strategy in strategies.items()
        },
    }


def main():
    args = parse_args()
    report = evaluate(
        args.dataset,
        args.split,
        batch_size=args.batch_size,
        limit=args.limit,
        allow_download=args.allow_download,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, result in report["strategies"].items():
        print(
            f"{name}: accuracy={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}, "
            f"weighted_f1={result['weighted_f1']:.4f}"
        )
    print(f"Saved E-BARD FashionCLIP report to {args.output}")


if __name__ == "__main__":
    main()
