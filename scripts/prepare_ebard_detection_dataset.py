"""Download, verify, and extract E-BARD's YOLO detection benchmark."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "ebard_detection" / "data"
DATASET_URL = (
    "https://huggingface.co/datasets/GabrieleGiudici/"
    "E-BARD-detection/resolve/main/all.zip?download=true"
)
DATASET_SHA256 = "4b0a5ef8fd25565714e6b36a7020bc68b1cc2765afdc82a3b3d7a099e5c2ab81"
EXPECTED_SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Install the published E-BARD YOLO detection benchmark."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Use an existing all.zip instead of downloading it.",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_is_ready(output, split):
    split_root = Path(output) / "yolo" / split
    images = split_root / "images"
    labels = split_root / "labels"
    if not images.is_dir() or not labels.is_dir():
        return False
    image_stems = {path.stem for path in images.glob("*.jpg")}
    label_stems = {path.stem for path in labels.glob("*.txt")}
    return bool(image_stems) and image_stems == label_stems


def dataset_is_ready(output):
    output = Path(output)
    return (output / "yolo" / "data.yaml").is_file() and all(
        split_is_ready(output, split) for split in EXPECTED_SPLITS
    )


def _safe_yolo_members(bundle):
    members = []
    for member in bundle.infolist():
        path = PurePosixPath(member.filename)
        if not path.parts or path.parts[0] != "yolo":
            continue
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe path in E-BARD archive: {member.filename}")
        members.append(member)
    if not members:
        raise RuntimeError("E-BARD archive does not contain the YOLO benchmark")
    return members


def install_dataset(output=DEFAULT_OUTPUT, *, archive=None):
    output = Path(output)
    if dataset_is_ready(output):
        return output
    if output.exists():
        raise RuntimeError(
            f"Dataset target exists but is incomplete: {output}. "
            "Move it aside and rerun the installer."
        )

    with tempfile.TemporaryDirectory(prefix="courtvision-ebard-detection-") as directory:
        temporary_root = Path(directory)
        archive_path = Path(archive) if archive else temporary_root / "all.zip"
        if archive is None:
            urllib.request.urlretrieve(DATASET_URL, archive_path)
        checksum = sha256(archive_path)
        if checksum != DATASET_SHA256:
            raise RuntimeError(
                "E-BARD detection archive failed verification: "
                f"expected {DATASET_SHA256}, received {checksum}"
            )
        extracted = temporary_root / "data"
        with zipfile.ZipFile(archive_path) as bundle:
            bundle.extractall(extracted, members=_safe_yolo_members(bundle))
        if not dataset_is_ready(extracted):
            raise RuntimeError("E-BARD archive does not have the expected YOLO layout")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(output))
    return output


def main():
    args = parse_args()
    try:
        output = install_dataset(args.output, archive=args.archive)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error)) from error
    print(f"E-BARD detection data ready: {output}")
    print(f"Archive SHA-256: {DATASET_SHA256}")
    print("Extracted YOLO data only; Qwen task files remain in the source archive.")


if __name__ == "__main__":
    main()
