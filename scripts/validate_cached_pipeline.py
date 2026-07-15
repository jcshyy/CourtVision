import argparse
import copy
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.analytics import (
    BallAcquisitionDetector,
    PassInterceptionDetector,
    SpeedAndDistanceCalculator,
    TacticalViewConverter,
    events_from_arrays,
)
from backend.app.analytics.ball_acquisition import summarize_acquisition_segments
from backend.app.team_assignment import TeamAssigner
from backend.app.tracking.ball_tracker import BallTracker, BALL_TRACKING_CACHE_VERSION
from backend.app.tracking.player_tracker import PLAYER_TRACKING_ALGORITHM_VERSION
from backend.app.utils import probe_video, read_video
from backend.app.visualization.team_ball_control_drawer import TeamBallControlDrawer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate cached CourtVision analytics and emit a JSON report."
    )
    parser.add_argument("video")
    parser.add_argument("cache_dir")
    parser.add_argument("output_json")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    return parser.parse_args()


def _load(path):
    with Path(path).open("rb") as file:
        return pickle.load(file)


def _speed_summary(distances, speeds, tactical_positions):
    player_totals = Counter()
    speed_records = []
    for frame_index, frame_distances in enumerate(distances):
        for player_id, distance in frame_distances.items():
            player_totals[int(player_id)] += float(distance)
        for player_id, speed in speeds[frame_index].items():
            speed_records.append(
                {
                    "frame": frame_index,
                    "player_id": int(player_id),
                    "speed_kmh": float(speed),
                    "frame_distance_meters": float(
                        frame_distances.get(player_id, 0.0)
                    ),
                    "tactical_position": tactical_positions[frame_index].get(
                        player_id
                    ),
                    "previous_tactical_position": (
                        tactical_positions[frame_index - 1].get(player_id)
                        if frame_index > 0
                        else None
                    ),
                }
            )
    speed_records.sort(key=lambda item: item["speed_kmh"], reverse=True)
    return {
        "player_total_distance_meters": {
            str(player_id): round(distance, 3)
            for player_id, distance in sorted(player_totals.items())
        },
        "maximum_speed_records": speed_records[:20],
        "speed_over_40_kmh_count": sum(
            item["speed_kmh"] > 40 for item in speed_records
        ),
    }


def validate(
    video_path,
    cache_dir,
    *,
    start_seconds=0.0,
    duration_seconds=None,
    target_fps=None,
    max_width=None,
):
    video_path = Path(video_path)
    cache_dir = Path(cache_dir)
    metadata = probe_video(video_path)
    fps = min(target_fps or metadata["fps"], metadata["fps"])
    frames = read_video(
        video_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        target_fps=fps,
        max_width=max_width,
    )

    player_track_name = (
        f"player_track_{PLAYER_TRACKING_ALGORITHM_VERSION}.pkl"
    )
    assigner = TeamAssigner()
    player_tracks = _load(cache_dir / player_track_name)
    versioned_ball_cache = cache_dir / f"ball_track_stubs_{BALL_TRACKING_CACHE_VERSION}.pkl"
    ball_tracks = _load(
        versioned_ball_cache
        if versioned_ball_cache.exists()
        else cache_dir / "ball_track_stubs.pkl"
    )
    court_keypoints = _load(cache_dir / "court_key_points_stub.pkl")
    assignments = _load(cache_dir / assigner.cache_filename)
    assignment_metadata_path = cache_dir / assigner.metadata_filename
    assignment_metadata = json.loads(
        assignment_metadata_path.read_text(encoding="utf-8")
    )

    ball_tracks = BallTracker.remove_wrong_detections(
        None,
        copy.deepcopy(ball_tracks),
        player_tracks=player_tracks,
    )
    ball_tracks = BallTracker.interpolate_positions(None, ball_tracks)
    acquisition_detector = BallAcquisitionDetector(fps=fps)
    acquisition_candidates = acquisition_detector.detect_candidates(
        player_tracks,
        ball_tracks,
    )
    holder_states = acquisition_detector.detect_holder_states(
        player_tracks,
        ball_tracks,
    )
    acquisitions = [
        state["holder_id"] if state["holder_id"] is not None else -1
        for state in holder_states
    ]
    event_detector = PassInterceptionDetector(
        max_holder_gap_frames=max(1, round(fps)),
    )
    acquisitions = event_detector.clean_transient_control_chains(
        acquisitions,
        assignments,
        holder_states=holder_states,
    )
    passes = event_detector.detect_passes(acquisitions, assignments)
    interceptions = event_detector.detect_interceptions(
        acquisitions,
        assignments,
        holder_states=holder_states,
    )
    events = events_from_arrays(
        passes,
        interceptions,
        acquisitions,
        assignments,
    )

    team_control_drawer = TeamBallControlDrawer()
    team_control = team_control_drawer.get_team_ball_control(
        assignments,
        acquisitions,
    )
    team_1_share, team_2_share = team_control_drawer.get_control_percentages(
        team_control
    )

    tactical_converter = TacticalViewConverter("images/basketball_court.png")
    validated_keypoints = tactical_converter.validate_keypoints(court_keypoints)
    tactical_positions = tactical_converter.transform_players_to_tactical_view(
        validated_keypoints,
        player_tracks,
    )
    speed_calculator = SpeedAndDistanceCalculator(
        tactical_converter.width,
        tactical_converter.height,
        tactical_converter.actual_width_in_meters,
        tactical_converter.actual_height_in_meters,
    )
    discontinuity_frames = tactical_converter.last_diagnostics.get(
        "temporal_discontinuity",
        [],
    )
    tactical_positions = speed_calculator.smooth_positions(
        tactical_positions,
        discontinuity_frames=discontinuity_frames,
        window_radius=max(1, round(fps * 0.1)),
    )
    distances = speed_calculator.calculate_distance(
        tactical_positions,
        discontinuity_frames=discontinuity_frames,
    )
    speeds = speed_calculator.calculate_speed(
        distances,
        fps=fps,
        tactical_player_positions=tactical_positions,
    )

    assignment_counts = Counter(
        int(team_id) for frame in assignments for team_id in frame.values()
    )
    team_control_counts = Counter(int(team_id) for team_id in team_control)
    return {
        "video": str(video_path),
        "cache_dir": str(cache_dir),
        "source": metadata,
        "frame_counts": {
            "decoded": len(frames),
            "player_tracks": len(player_tracks),
            "assignments": len(assignments),
            "acquisitions": len(acquisitions),
            "tactical_positions": len(tactical_positions),
        },
        "team_assignment": {
            "metadata": assignment_metadata,
            "frame_label_counts": dict(assignment_counts),
        },
        "possession": {
            "state_counts": dict(Counter(state["state"] for state in holder_states)),
            "holder_states": holder_states,
            "candidate_segments": summarize_acquisition_segments(
                acquisition_candidates
            ),
            "segments": summarize_acquisition_segments(acquisitions),
            "known_holder_frame_count": sum(
                player_id not in (-1, None) for player_id in acquisitions
            ),
            "team_control_frame_counts": dict(team_control_counts),
            "known_team_denominator": team_control_counts[1]
            + team_control_counts[2],
            "team_1_percentage": round(team_1_share * 100, 3),
            "team_2_percentage": round(team_2_share * 100, 3),
        },
        "events": events,
        "homography": {
            key: {
                "count": len(frame_indices),
                "frames": frame_indices,
            }
            for key, frame_indices in tactical_converter.last_diagnostics.items()
        },
        "speed_distance": _speed_summary(
            distances,
            speeds,
            tactical_positions,
        ),
    }


def main():
    args = parse_args()
    report = validate(
        args.video,
        args.cache_dir,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        target_fps=args.target_fps,
        max_width=args.max_width,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, default=_json_scalar) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path.resolve()}")


def _json_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
