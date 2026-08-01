import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.team_assignment.team_assigner import (
    TeamAssigner,
    _color_distance,
    _confident_nearest_team,
    _discover_team_colors_result,
    _jersey_observation,
    _median_color,
    _nearest_team,
    _team_distances,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit per-track jersey evidence against cached assignments."
    )
    parser.add_argument("video")
    parser.add_argument("track_cache")
    parser.add_argument("assignment_cache")
    parser.add_argument("output_json")
    return parser.parse_args()


def _load(path):
    with Path(path).open("rb") as file:
        return pickle.load(file)


def _read_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def _cached_track_team(assignments, track_id):
    values = [frame[track_id] for frame in assignments if track_id in frame]
    return Counter(values).most_common(1)[0][0] if values else None


def audit(video, track_cache, assignment_cache):
    frames = _read_video(video)
    tracks = _load(track_cache)
    assignments = _load(assignment_cache)
    if len(frames) != len(tracks):
        raise ValueError(
            f"Frame/cache length mismatch: video={len(frames)}, tracks={len(tracks)}"
        )

    discovery = _discover_team_colors_result(frames, tracks)
    if discovery["status"] != "confident":
        return {"status": discovery["status"], "discovery": discovery}

    prototypes = {1: discovery["prototypes"][0], 2: discovery["prototypes"][1]}
    separation = _color_distance(prototypes[1], prototypes[2])
    observations_by_track = defaultdict(list)
    for frame_index, (frame, frame_tracks) in enumerate(zip(frames, tracks)):
        for track_id, track in frame_tracks.items():
            observation = _jersey_observation(frame, track["bbox"])
            feature = observation["feature"]
            distances = _team_distances(feature, prototypes)
            numeric = [value for value in distances.values() if value is not None]
            nearest_distance = min(numeric) if numeric else None
            margin = abs(numeric[0] - numeric[1]) if len(numeric) == 2 else None
            observations_by_track[track_id].append(
                {
                    "frame": frame_index,
                    "accepted": observation["accepted"],
                    "rejection_reason": observation["rejection_reason"],
                    "feature": feature,
                    "nearest_team": _nearest_team(feature, prototypes),
                    "confident_team": _confident_nearest_team(feature, prototypes),
                    "prototype_distances": distances,
                    "nearest_distance_ratio": (
                        nearest_distance / separation
                        if nearest_distance is not None and separation > 0
                        else None
                    ),
                    "margin_ratio": (
                        margin / separation
                        if margin is not None and separation > 0
                        else None
                    ),
                    "torso_area": observation.get("torso_area"),
                    "blur_variance": observation.get("blur_variance"),
                    "visible_fraction": observation.get("visible_fraction"),
                    "filtered_fraction": observation.get("filtered_fraction"),
                    "used_visible_fallback": observation.get("used_visible_fallback"),
                }
            )

    track_summaries = []
    for track_id, observations in sorted(observations_by_track.items()):
        accepted = [item for item in observations if item["accepted"]]
        confident = [item for item in accepted if item["confident_team"] is not None]
        features = [item["feature"] for item in confident]
        representative = _median_color(features) if features else None
        team_counts = Counter(item["confident_team"] for item in confident)
        rejection_counts = Counter(
            item["rejection_reason"] for item in observations if not item["accepted"]
        )
        track_summaries.append(
            {
                "track_id": track_id,
                "visible_frame_range": [
                    observations[0]["frame"],
                    observations[-1]["frame"],
                ],
                "visible_frame_count": len(observations),
                "accepted_observation_count": len(accepted),
                "confident_observation_count": len(confident),
                "rejection_counts": dict(rejection_counts),
                "team_vote_counts": dict(team_counts),
                "vote_agreement": (
                    max(team_counts.values()) / sum(team_counts.values())
                    if team_counts
                    else None
                ),
                "median_nearest_distance_ratio": _median(
                    item["nearest_distance_ratio"] for item in confident
                ),
                "median_margin_ratio": _median(
                    item["margin_ratio"] for item in confident
                ),
                "representative_feature": representative,
                "representative_team": _nearest_team(representative, prototypes),
                "cached_team": _cached_track_team(assignments, track_id),
                "observations": observations,
            }
        )

    return {
        "status": "ok",
        "video": str(video),
        "frame_count": len(frames),
        "discovery": discovery,
        "team_prototypes": prototypes,
        "track_summaries": track_summaries,
    }


def _median(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def main():
    args = parse_args()
    result = audit(args.video, args.track_cache, args.assignment_cache)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, default=_json_scalar) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path.resolve()}")


def _json_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
