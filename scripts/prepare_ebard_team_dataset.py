"""Download, verify, and extract the E-BARD team-attribution dataset."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "ebard_team_attribution" / "data"
DATASET_URL = (
    "https://huggingface.co/datasets/GabrieleGiudici/"
    "E-BARD-TeamAttribution/resolve/main/all.zip?download=true"
)
DATASET_SHA256 = "f25c2bb8d8527992e68fc952d840f6c379b4ce24412d0dd764bdcc9c95134be7"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Install the published E-BARD team-attribution dataset."
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


def dataset_is_ready(output):
    output = Path(output)
    return all(
        (output / filename).is_file()
        for filename in ("train_labels.csv", "valid_labels.csv", "test_labels.csv")
    ) and (output / "colors").is_dir()


def install_dataset(output=DEFAULT_OUTPUT, *, archive=None):
    output = Path(output)
    if dataset_is_ready(output):
        return output
    if output.exists():
        raise RuntimeError(
            f"Dataset target exists but is incomplete: {output}. "
            "Move it aside and rerun the installer."
        )

    with tempfile.TemporaryDirectory(prefix="courtvision-ebard-") as directory:
        temporary_root = Path(directory)
        archive_path = Path(archive) if archive else temporary_root / "all.zip"
        if archive is None:
            urllib.request.urlretrieve(DATASET_URL, archive_path)
        checksum = sha256(archive_path)
        if checksum != DATASET_SHA256:
            raise RuntimeError(
                "E-BARD dataset archive failed verification: "
                f"expected {DATASET_SHA256}, received {checksum}"
            )
        extracted = temporary_root / "data"
        with zipfile.ZipFile(archive_path) as bundle:
            bundle.extractall(extracted)
        if not dataset_is_ready(extracted):
            raise RuntimeError("E-BARD archive does not have the expected layout")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(output))
    return output


def main():
    args = parse_args()
    try:
        output = install_dataset(args.output, archive=args.archive)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error)) from error
    print(f"E-BARD team-attribution data ready: {output}")
    print(f"Archive SHA-256: {DATASET_SHA256}")


if __name__ == "__main__":
    main()
