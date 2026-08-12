import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "courtvision_nba_detection_v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark(benchmark_dir):
    manifest_path = benchmark_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotation_path = benchmark_dir / manifest["annotation_path"]
    if not annotation_path.is_file():
        raise ValueError(
            f"Missing annotations: {annotation_path}. Extract the COCO download under "
            f"{benchmark_dir / 'data'}."
        )
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    return manifest, annotation_path, annotations


def validate(benchmark_dir):
    manifest, annotation_path, coco = load_benchmark(benchmark_dir)
    errors = []
    expected = manifest["expected"]
    actual_hash = sha256(annotation_path)
    if actual_hash != manifest["annotation_sha256"]:
        errors.append(
            f"annotation SHA256 mismatch: expected {manifest['annotation_sha256']}, "
            f"found {actual_hash}"
        )

    categories = {entry["id"]: entry["name"] for entry in coco.get("categories", [])}
    target_names = set(manifest["target_categories"])
    target_ids = {category_id for category_id, name in categories.items() if name in target_names}
    missing_categories = target_names - set(categories.values())
    if missing_categories:
        errors.append(f"missing target categories: {sorted(missing_categories)}")

    images = coco.get("images", [])
    image_by_id = {image["id"]: image for image in images}
    if len(images) != expected["images"]:
        errors.append(f"expected {expected['images']} images, found {len(images)}")
    if len(image_by_id) != len(images):
        errors.append("duplicate image IDs")

    target_annotations = [
        annotation
        for annotation in coco.get("annotations", [])
        if annotation.get("category_id") in target_ids
    ]
    if len(target_annotations) != expected["target_annotations"]:
        errors.append(
            f"expected {expected['target_annotations']} target annotations, "
            f"found {len(target_annotations)}"
        )

    image_root = annotation_path.parent
    for image in images:
        if image.get("width") != expected["image_width"] or image.get("height") != expected["image_height"]:
            errors.append(
                f"image {image.get('id')} has unexpected dimensions "
                f"{image.get('width')}x{image.get('height')}"
            )
        image_path = image_root / image.get("file_name", "")
        if not image_path.is_file():
            errors.append(f"missing image: {image_path}")

    category_counts = Counter()
    for annotation in target_annotations:
        image = image_by_id.get(annotation.get("image_id"))
        if image is None:
            errors.append(f"annotation {annotation.get('id')} has unknown image_id")
            continue
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"annotation {annotation.get('id')} has invalid bbox")
            continue
        x, y, width, height = bbox
        if width <= 0 or height <= 0 or x < 0 or y < 0:
            errors.append(f"annotation {annotation.get('id')} has non-positive or negative bbox")
        elif x + width > image["width"] + 1e-6 or y + height > image["height"] + 1e-6:
            errors.append(f"annotation {annotation.get('id')} bbox exceeds image bounds")
        category_counts[categories[annotation["category_id"]]] += 1

    if errors:
        raise ValueError("\n".join(errors))
    return {
        "images": len(images),
        "target_annotations": len(target_annotations),
        "category_counts": dict(sorted(category_counts.items())),
        "annotation_sha256": actual_hash,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the NBA ball detection benchmark.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    result = validate(parser.parse_args().benchmark_dir.resolve())
    print(f"Benchmark valid: {json.dumps(result, sort_keys=True)}")
