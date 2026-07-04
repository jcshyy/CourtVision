import hashlib
import re
from pathlib import Path


def safe_video_stem(input_video: str | Path) -> str:
    video_path = Path(input_video)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("._")
    return safe_stem or "video"


def video_cache_dir(
    cache_root: str | Path,
    input_video: str | Path,
    processing_identity: str | None = None,
) -> Path:
    """Return a stable cache directory unique to the input video's contents."""
    video_path = Path(input_video)
    digest = hashlib.sha256()
    with video_path.open("rb") as video_file:
        for chunk in iter(lambda: video_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if processing_identity:
        digest.update(b"\0processing\0")
        digest.update(processing_identity.encode("utf-8"))

    return Path(cache_root) / f"{safe_video_stem(video_path)}-{digest.hexdigest()}"


def cache_path(cache_dir: str | Path, name: str) -> Path:
    return Path(cache_dir) / name


def default_job_output_path(
    output_root: str | Path,
    input_video: str | Path,
) -> Path:
    """Return the stable default output path for an input video."""
    return Path(output_root) / f"{safe_video_stem(input_video)}.avi"
