"""Compose existing YOLO and WASB candidates without repeating inference."""

import json
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.tracking.ball_tracker import (
    BALL_TRACKING_CACHE_VERSION,
    SOURCE_AWARE_HYBRID_VERSION,
    WASB_INTEGRATION_VERSION,
    _merge_ball_candidates,
    _select_ball_track,
)
from backend.app.tracking.player_tracker import PLAYER_TRACKING_ALGORITHM_VERSION
from scripts.score_benchmark import DEFAULT_BENCHMARK, _cache_dir, _first_existing


YOLO_SOURCE_VERSION = "v5_temporal_candidate_lattice"
WASB_SOURCE_VERSION = (
    f"{BALL_TRACKING_CACHE_VERSION}_wasb_{WASB_INTEGRATION_VERSION}"
)
HYBRID_CACHE_VERSION = (
    f"{BALL_TRACKING_CACHE_VERSION}_hybrid_{SOURCE_AWARE_HYBRID_VERSION}"
)


def _load(path):
    with path.open("rb") as source:
        return pickle.load(source)


def compose(benchmark_dir=DEFAULT_BENCHMARK):
    dataset = json.loads((benchmark_dir / "dataset.json").read_text(encoding="utf-8"))
    outputs = []
    for video in dataset["videos"]:
        cache_dir = _cache_dir(video)
        yolo_path = cache_dir / f"ball_track_stubs_{YOLO_SOURCE_VERSION}.pkl"
        wasb_path = cache_dir / f"ball_track_stubs_{WASB_SOURCE_VERSION}.pkl"
        if not yolo_path.is_file():
            raise FileNotFoundError(yolo_path)
        if not wasb_path.is_file():
            raise FileNotFoundError(wasb_path)
        player_path = _first_existing(
            cache_dir,
            f"player_track_{PLAYER_TRACKING_ALGORITHM_VERSION}.pkl",
            "player_track_stubs.pkl",
        )
        yolo_tracks = _load(yolo_path)
        wasb_tracks = _load(wasb_path)
        player_tracks = _load(player_path)
        if not (len(yolo_tracks) == len(wasb_tracks) == len(player_tracks)):
            raise ValueError(f"Unaligned caches for {video['id']}")
        candidate_frames = []
        semantic_candidate_frames = []
        for yolo_frame, wasb_frame in zip(yolo_tracks, wasb_tracks):
            yolo_info = yolo_frame.get(1, {})
            wasb_info = wasb_frame.get(1, {})
            yolo_candidates = [
                {**candidate, "detection_source": "full_frame"}
                for candidate in yolo_info.get(
                    "raw_candidates",
                    yolo_info.get("candidates", []),
                )
            ]
            wasb_candidates = wasb_info.get(
                "raw_candidates",
                wasb_info.get("candidates", []),
            )
            semantic_candidate_frames.append([
                dict(candidate) for candidate in yolo_candidates
            ])
            candidate_frames.append(
                _merge_ball_candidates(yolo_candidates, wasb_candidates)
            )
        hybrid_tracks = _select_ball_track(candidate_frames, player_tracks)
        for frame_index, (hybrid_frame, yolo_frame) in enumerate(
            zip(hybrid_tracks, yolo_tracks)
        ):
            hybrid_frame[1]["rim_regions"] = list(
                yolo_frame.get(1, {}).get("rim_regions", [])
            )
            hybrid_frame[1]["adaptive_second_pass_completed"] = False
            hybrid_frame[1]["semantic_raw_candidates"] = [
                dict(candidate)
                for candidate in semantic_candidate_frames[frame_index]
            ]
        output_path = cache_dir / f"ball_track_stubs_{HYBRID_CACHE_VERSION}.pkl"
        with output_path.open("wb") as destination:
            pickle.dump(hybrid_tracks, destination)
        outputs.append(str(output_path))
        print(f"{video['id']}: {output_path.name}", flush=True)
    return outputs


if __name__ == "__main__":
    compose()
