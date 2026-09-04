"""Run the production CLI with a larger decoded-video safety ceiling.

This wrapper changes only the allocation guard used while reading a video. It
does not resize, resample, or alter detector, event, or scoring configuration.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as pipeline
from backend.app.utils import read_video as production_read_video


def expanded_read_video(*args, **kwargs):
    kwargs["max_decoded_bytes"] = 3 * 1024**3
    return production_read_video(*args, **kwargs)


pipeline.read_video = expanded_read_video


if __name__ == "__main__":
    pipeline.main()
