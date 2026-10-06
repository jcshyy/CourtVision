"""Download pinned official MultiSports basketball data using local HF login."""
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

REVISION = "01600ce7eabbf42a5ee7c82b82f49a11597b3a5f"
FILES = {
    "multisports_GT.pkl": "e6579fb0986713c0a681dd0abef2475eedb416915706b69c3e8580f501a46942",
    "generate_rgb.py": None,
    "basketball.tar": "f1953377e83b13beff62571e4fc980b21e240a9811c4ea82bd69b8de150ea8a9",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("holdout_sources/multisports"))
    args = parser.parse_args()
    for name, expected in FILES.items():
        print(f"Downloading {name} at {REVISION}", flush=True)
        path = Path(hf_hub_download(
            "MCG-NJU/SportsAction", f"data/trainval/{name}", repo_type="dataset",
            revision=REVISION, local_dir=args.directory,
        ))
        if expected:
            with path.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum mismatch: {name}")
            print(f"Verified {name}: {actual}", flush=True)
        print(f"Available: {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
