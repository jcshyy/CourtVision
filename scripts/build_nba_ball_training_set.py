import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "courtvision_nba_detection_v1"
TRAINING_SPLITS = ("train", "valid")
TARGET_CATEGORIES = {"ball"}


def link_or_copy(source, destination):
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(benchmark_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    seen_file_names = set()
    training_hashes = set()
    for split in TRAINING_SPLITS:
        source_dir = benchmark_dir / "data" / split
        coco = json.loads((source_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
        categories = {entry["id"]: entry["name"] for entry in coco["categories"]}
        target_ids = {category_id for category_id, name in categories.items() if name in TARGET_CATEGORIES}
        images = {image["id"]: image for image in coco["images"]}
        annotations = {image_id: [] for image_id in images}
        for annotation in coco["annotations"]:
            if annotation["category_id"] in target_ids:
                annotations[annotation["image_id"]].append(annotation)

        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        box_count = 0
        positive_images = 0
        for image_id, image in images.items():
            file_name = image["file_name"]
            if file_name in seen_file_names:
                raise ValueError(f"Duplicate filename across training splits: {file_name}")
            seen_file_names.add(file_name)
            link_or_copy(source_dir / file_name, image_dir / file_name)
            training_hashes.add(file_sha256(source_dir / file_name))
            lines = []
            for annotation in annotations[image_id]:
                x, y, width, height = map(float, annotation["bbox"])
                center_x = (x + width / 2) / image["width"]
                center_y = (y + height / 2) / image["height"]
                lines.append(
                    f"0 {center_x:.8f} {center_y:.8f} "
                    f"{width / image['width']:.8f} {height / image['height']:.8f}"
                )
            (label_dir / f"{Path(file_name).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            box_count += len(lines)
            positive_images += bool(lines)
        summary[split] = {
            "images": len(images),
            "ball_boxes": box_count,
            "positive_images": positive_images,
            "negative_images": len(images) - positive_images,
        }

    test_names = {
        image["file_name"]
        for image in json.loads(
            (benchmark_dir / "data" / "test" / "_annotations.coco.json").read_text(encoding="utf-8")
        )["images"]
    }
    overlap = seen_file_names & test_names
    if overlap:
        raise ValueError(f"Test filenames leaked into training export: {sorted(overlap)[:3]}")
    test_hashes = {
        file_sha256(benchmark_dir / "data" / "test" / file_name)
        for file_name in test_names
    }
    hash_overlap = training_hashes & test_hashes
    if hash_overlap:
        raise ValueError(f"Exact test image content leaked into training export: {len(hash_overlap)} hashes")
    summary["leakage_audit"] = {
        "test_filename_overlap": 0,
        "test_exact_sha256_overlap": 0,
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build one-class YOLO data without test leakage.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    output_dir = (args.output_dir or benchmark_dir / "training_yolo").resolve()
    print(json.dumps(build(benchmark_dir, output_dir), indent=2))
