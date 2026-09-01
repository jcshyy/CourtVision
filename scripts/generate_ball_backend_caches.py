"""Generate isolated ball-track caches for an experimental detector backend."""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.tracking import BallTracker
from backend.app.tracking.player_tracker import PLAYER_TRACKING_ALGORITHM_VERSION
from backend.app.utils import read_video
from scripts.score_benchmark import DEFAULT_BENCHMARK, _cache_dir, _first_existing


def generate(benchmark_dir, backend, overwrite=False, video_ids=None):
    dataset = json.loads((benchmark_dir / "dataset.json").read_text(encoding="utf-8"))
    tracker = BallTracker(detector_backend=backend)
    results = []
    for video in dataset["videos"]:
        if video_ids and video["id"] not in video_ids:
            continue
        cache_dir = _cache_dir(video)
        output_path = cache_dir / f"ball_track_stubs_{tracker.cache_version}.pkl"
        if output_path.is_file() and not overwrite:
            results.append({
                "video_id": video["id"],
                "cache_path": str(output_path),
                "status": "existing",
            })
            print(f"{video['id']}: using {output_path.name}", flush=True)
            continue
        player_path = _first_existing(
            cache_dir,
            f"player_track_{PLAYER_TRACKING_ALGORITHM_VERSION}.pkl",
            "player_track_stubs.pkl",
        )
        with player_path.open("rb") as source:
            player_tracks = pickle.load(source)
        started = time.perf_counter()
        frames = read_video(ROOT / video["path"])
        if len(frames) != video["frame_count"]:
            raise ValueError(
                f"{video['id']} decoded {len(frames)} frames; "
                f"expected {video['frame_count']}"
            )
        tracks = tracker.get_object_tracks(frames, player_tracks=player_tracks)
        with output_path.open("wb") as destination:
            pickle.dump(tracks, destination)
        elapsed = time.perf_counter() - started
        result = {
            "video_id": video["id"],
            "frames": len(frames),
            "seconds": elapsed,
            "cache_path": str(output_path),
            "status": "generated",
        }
        results.append(result)
        print(
            f"{video['id']}: {len(frames)} frames in {elapsed:.1f}s -> "
            f"{output_path.name}",
            flush=True,
        )
    return {"backend": backend, "cache_version": tracker.cache_version, "videos": results}


def main():
    parser = argparse.ArgumentParser(
        description="Generate versioned ball caches for YOLO, WASB, or hybrid inference."
    )
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--backend", choices=("yolo", "wasb", "hybrid"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-id", action="append", dest="video_ids")
    args = parser.parse_args()
    print(json.dumps(
        generate(
            args.benchmark_dir.resolve(),
            args.backend,
            args.overwrite,
            set(args.video_ids or ()),
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
