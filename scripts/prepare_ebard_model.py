"""Download and verify the pinned E-BARD YOLOv8n detector checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "backend" / "models" / "ebard_yolov8n.pt"
EBARD_REVISION = "3f4789c4431aa73269f60107a4ba0a5f86b7af8b"
EBARD_FILENAME = "BODD_yolov8n_0001.pt"
EBARD_URL = (
    "https://huggingface.co/GabrieleGiudici/E-BARD-detection-models/resolve/"
    f"{EBARD_REVISION}/{EBARD_FILENAME}?download=true"
)
EBARD_SHA256 = "dfe3534d51bb21024d1a400c37f0c1fbf0c8b96ea9a56a5f3cb5454813bfd641"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Install the pinned E-BARD basketball YOLOv8n checkpoint."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing file even when its checksum does not match.",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_model(output=DEFAULT_OUTPUT, *, force=False):
    output = Path(output)
    if output.is_file():
        checksum = sha256(output)
        if checksum == EBARD_SHA256:
            return output
        if not force:
            raise RuntimeError(
                f"Existing file has unexpected SHA-256 {checksum}: {output}. "
                "Use --force to replace it."
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        urllib.request.urlretrieve(EBARD_URL, partial)
        checksum = sha256(partial)
        if checksum != EBARD_SHA256:
            raise RuntimeError(
                "Downloaded E-BARD checkpoint failed verification: "
                f"expected {EBARD_SHA256}, received {checksum}"
            )
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return output


def main():
    args = parse_args()
    try:
        output = install_model(args.output, force=args.force)
    except (OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(f"E-BARD YOLOv8n ready: {output}")
    print(f"SHA-256: {EBARD_SHA256}")


if __name__ == "__main__":
    sys.exit(main())
