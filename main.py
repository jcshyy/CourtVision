import argparse
import json
from pathlib import Path

from backend.app.cache_paths import cache_path, default_job_output_path, video_cache_dir
from backend.app.analytics import (
    BallAcquisitionDetector,
    PassInterceptionDetector,
    SpeedAndDistanceCalculator,
    TacticalViewConverter,
)
from backend.app.config import OUTPUT_DIR, STUBS_DIR
from backend.app.detection import (
    CourtKeypointDetector,
    PlayerPoseDetector,
    attach_player_poses,
)
from backend.app.team_assignment import NeedsTeamColorsError, TeamAssigner
from backend.app.tracking import BallTracker, PlayerTracker
from backend.app.tracking.player_tracker import PLAYER_TRACKING_ALGORITHM_VERSION
from backend.app.utils import (
    detect_scene_discontinuities,
    probe_video,
    read_video,
    save_video,
)
from backend.app.visualization import (
    BallTracksDrawer,
    CourtKeypointDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    PlayerTracksDrawer,
    SpeedAndDistanceDrawer,
    TeamBallControlDrawer,
)

TEAM_DISPLAY_COLORS = {
    1: (255, 80, 0),   # Bright blue in OpenCV BGR order.
    2: (0, 215, 255),  # Bright yellow in OpenCV BGR order.
}


def events_to_overlay_arrays(events, frame_count):
    """Convert validated transition events into the legacy per-frame drawer shape."""
    passes = [-1] * frame_count
    interceptions = [-1] * frame_count
    for event in events:
        frame_index = int(event.get("frame_index", -1))
        team_id = event.get("to_team_id")
        if not 0 <= frame_index < frame_count or team_id not in (1, 2):
            continue
        if event.get("type") == "pass":
            passes[frame_index] = team_id
        elif event.get("type") == "interception":
            interceptions[frame_index] = team_id
    return passes, interceptions


def parse_args():
    parser = argparse.ArgumentParser(description="Basketball Video Analysis")
    parser.add_argument(
        "input_video",
        nargs="?",
        default="input_videos/video_1.mp4",
        help="Path to input video file.",
    )
    parser.add_argument(
        "--output-video",
        "--output_video",
        default=None,
        help="Path to output video file.",
    )
    parser.add_argument(
        "--output-analysis",
        default=None,
        help="Optional path for a web-review analysis manifest.",
    )
    parser.add_argument(
        "--stub-path",
        "--stub_path",
        default=str(STUBS_DIR),
        help="Path to stub directory.",
    )
    parser.add_argument(
        "--team-1-color",
        default=None,
        help="Team 1 primary jersey color in #RRGGBB format.",
    )
    parser.add_argument(
        "--team-2-color",
        default=None,
        help="Team 2 primary jersey color in #RRGGBB format.",
    )
    parser.add_argument(
        "--allow-uncertain-teams",
        action="store_true",
        help=(
            "Continue when team assignment is uncertain, leaving unresolved players "
            "unknown. Team-level event totals may be inaccurate."
        ),
    )
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    parser.add_argument(
        "--ball-detector-backend",
        choices=("yolo", "wasb", "hybrid"),
        default="yolo",
        help="Ball candidate source. The production default remains yolo.",
    )
    args = parser.parse_args()
    if (args.team_1_color is None) != (args.team_2_color is None):
        parser.error("--team-1-color and --team-2-color must be provided together")
    if args.start_seconds < 0:
        parser.error("--start-seconds must be non-negative")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if args.target_fps is not None and args.target_fps <= 0:
        parser.error("--target-fps must be positive")
    if args.max_width is not None and args.max_width <= 0:
        parser.error("--max-width must be positive")
    return args


def default_output_path(input_video):
    return default_job_output_path(OUTPUT_DIR, input_video)


def write_analysis_manifest(
    output_path,
    *,
    fps,
    frame_count,
    court_width,
    court_height,
    tactical_player_positions,
    player_assignment,
    ball_acquisition,
    events,
    tactical_diagnostics,
    assignment_metadata,
):
    """Write the evidence used by the web review surface.

    The manifest intentionally labels detections as beta candidates. It exposes
    unknowns instead of promoting pipeline output to verified game statistics.
    """
    manifest_path = Path(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for frame_index, tactical_positions in enumerate(tactical_player_positions):
        assignments = (
            player_assignment[frame_index]
            if frame_index < len(player_assignment)
            else {}
        )
        holder_id = (
            ball_acquisition[frame_index]
            if frame_index < len(ball_acquisition)
            and ball_acquisition[frame_index] != -1
            else None
        )
        possession_team_id = _json_safe(assignments.get(holder_id))
        if possession_team_id not in (1, 2):
            possession_team_id = None
        players = []
        for player_id, position in tactical_positions.items():
            team_id = _json_safe(assignments.get(player_id))
            if team_id not in (1, 2):
                team_id = None
            players.append(
                {
                    "id": int(player_id),
                    "x": round(float(position[0]), 2),
                    "y": round(float(position[1]), 2),
                    "teamId": team_id,
                    "isHolder": bool(player_id == holder_id),
                }
            )
        frames.append(
            {
                "frameIndex": frame_index,
                "timeSeconds": round(frame_index / fps, 3),
                "possessionTeamId": possession_team_id,
                "players": players,
            }
        )

    review_events = []
    for event_index, event in enumerate(events):
        frame_index = int(event["frame_index"])
        review_events.append(
            {
                "id": f"event-{event_index + 1}",
                "type": event["type"],
                "frameIndex": frame_index,
                "timeSeconds": round(frame_index / fps, 3),
                "fromTeamId": _json_safe(event.get("from_team_id")),
                "toTeamId": _json_safe(event.get("to_team_id")),
                "status": (
                    "candidate"
                    if event.get("to_team_id") in (1, 2)
                    else "unknown"
                ),
                "evidence": {
                    key: _json_safe(value)
                    for key, value in event.items()
                    if key not in {"type", "frame_index", "from_team_id", "to_team_id"}
                },
            }
        )

    payload = {
        "schemaVersion": 1,
        "beta": True,
        "disclaimer": (
            "Experimental analysis. Review every result against the source play; "
            "do not treat it as an authoritative coaching decision."
        ),
        "source": {
            "fps": fps,
            "frameCount": frame_count,
            "durationSeconds": round(frame_count / fps, 3),
        },
        "court": {"width": court_width, "height": court_height},
        "events": review_events,
        "frames": frames,
        "diagnostics": {
            "tacticalView": _json_safe(tactical_diagnostics),
            "teamAssignment": _json_safe(assignment_metadata),
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def filter_ball_tracks_with_pose(
    video_frames,
    player_tracks,
    raw_ball_tracks,
    player_pose_detector,
    ball_tracker,
    *,
    pose_cache_path=None,
    adaptive_cache_path=None,
    discontinuity_frames=None,
    include_semantic_track=False,
):
    """Attach pose evidence before selecting and interpolating the ball path."""
    player_poses = player_pose_detector.get_player_poses(
        video_frames,
        player_tracks,
        ball_tracks=raw_ball_tracks,
        read_from_cache=True,
        cache_path=pose_cache_path,
    )
    enriched_player_tracks = attach_player_poses(player_tracks, player_poses)
    enhance = getattr(ball_tracker, "enhance_tracks_with_adaptive_crops", None)
    if callable(enhance):
        raw_ball_tracks = enhance(
            video_frames,
            raw_ball_tracks,
            enriched_player_tracks,
            read_from_cache=True,
            cache_path=adaptive_cache_path,
        )
    filtered_ball_tracks = ball_tracker.remove_wrong_detections(
        raw_ball_tracks,
        player_tracks=enriched_player_tracks,
        discontinuity_frames=discontinuity_frames,
    )
    ball_tracks = ball_tracker.interpolate_positions(
        filtered_ball_tracks,
        discontinuity_frames=discontinuity_frames,
    )
    if not include_semantic_track:
        return enriched_player_tracks, ball_tracks
    if getattr(ball_tracker, "detector_backend", "yolo") == "hybrid":
        semantic_ball_tracks = ball_tracker.build_semantic_tracks(
            raw_ball_tracks,
            enriched_player_tracks,
            fused_tracks=ball_tracks,
            discontinuity_frames=discontinuity_frames,
        )
    else:
        semantic_ball_tracks = ball_tracks
    return enriched_player_tracks, ball_tracks, semantic_ball_tracks


def main():
    args = parse_args()
    video_path = args.input_video
    output_path = args.output_video or default_output_path(video_path)
    try:
        team_assigner = TeamAssigner(
            team_1_color=args.team_1_color,
            team_2_color=args.team_2_color,
            tracking_algorithm_version=PLAYER_TRACKING_ALGORITHM_VERSION,
            allow_uncertain_teams=args.allow_uncertain_teams,
        )
    except ValueError as error:
        raise SystemExit(f"Invalid team-color configuration: {error}") from error

    video_metadata = probe_video(video_path)
    output_fps = min(args.target_fps or video_metadata["fps"], video_metadata["fps"])
    processing_options = {
        "start_seconds": args.start_seconds,
        "duration_seconds": args.duration_seconds,
        "target_fps": output_fps,
        "max_width": args.max_width,
    }
    processing_identity = json.dumps(processing_options, sort_keys=True)
    try:
        video_frames = read_video(
            video_path,
            start_seconds=args.start_seconds,
            duration_seconds=args.duration_seconds,
            target_fps=output_fps,
            max_width=args.max_width,
            max_decoded_bytes=2 * 1024**3,
        )
    except MemoryError as error:
        raise SystemExit(f"Video selection is too large: {error}") from error
    print(f"Loaded {len(video_frames)} frames from {video_path}")
    scene_discontinuity_frames = detect_scene_discontinuities(video_frames)
    print(
        "Detected persistent scene discontinuities: "
        f"{len(scene_discontinuity_frames)}"
    )
    cache_dir = video_cache_dir(args.stub_path, video_path, processing_identity)
    print(f"Using per-video cache directory: {cache_dir.resolve()}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    assignment_metadata_path = cache_dir / team_assigner.metadata_filename

    player_tracker = PlayerTracker()
    player_pose_detector = PlayerPoseDetector()
    ball_tracker = BallTracker(detector_backend=args.ball_detector_backend)
    court_keypoint_detector = CourtKeypointDetector()

    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_cache=True,
        cache_path=cache_path(cache_dir, player_tracker.cache_filename),
    )
    raw_ball_tracks = ball_tracker.get_object_tracks(
        video_frames,
        read_from_cache=True,
        cache_path=cache_path(
            cache_dir, f"ball_track_stubs_{ball_tracker.cache_version}.pkl"
        ),
        player_tracks=player_tracks,
    )
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(
        video_frames,
        read_from_cache=True,
        cache_path=cache_path(cache_dir, "court_key_points_stub.pkl"),
    )

    player_tracks, ball_tracks, semantic_ball_tracks = filter_ball_tracks_with_pose(
        video_frames,
        player_tracks,
        raw_ball_tracks,
        player_pose_detector,
        ball_tracker,
        pose_cache_path=cache_path(
            cache_dir,
            player_pose_detector.cache_filename,
        ),
        adaptive_cache_path=cache_path(
            cache_dir,
            f"ball_adaptive_stubs_{ball_tracker.adaptive_cache_version}.pkl",
        ),
        discontinuity_frames=scene_discontinuity_frames,
        include_semantic_track=True,
    )

    try:
        player_assignment = team_assigner.get_player_teams_across_frames(
            video_frames,
            player_tracks,
            read_from_cache=True,
            cache_path=cache_path(cache_dir, team_assigner.cache_filename),
        )
    except NeedsTeamColorsError as error:
        assignment_metadata_path.write_text(
            json.dumps(team_assigner.assignment_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(error.result, sort_keys=True))
        raise SystemExit(2) from error
    assignment_metadata_path.write_text(
        json.dumps(team_assigner.assignment_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    ball_acquisition_detector = BallAcquisitionDetector(fps=output_fps)
    holder_states = ball_acquisition_detector.detect_holder_states(
        player_tracks,
        semantic_ball_tracks,
    )
    ball_acquisition = [
        state["holder_id"] if state["holder_id"] is not None else -1
        for state in holder_states
    ]

    pass_interception_detector = PassInterceptionDetector(
        max_holder_gap_frames=max(1, round(output_fps * 0.9)),
        minimum_catch_frames=max(2, round(output_fps * 0.1)),
        catch_confirmation_frames=max(3, round(output_fps)),
    )
    ball_acquisition = pass_interception_detector.clean_transient_control_chains(
        ball_acquisition,
        player_assignment,
        holder_states=holder_states,
        discontinuity_frames=scene_discontinuity_frames,
    )
    events = pass_interception_detector.detect_events(
        ball_acquisition,
        player_assignment,
        holder_states=holder_states,
        ball_tracks=semantic_ball_tracks,
        player_tracks=player_tracks,
        discontinuity_frames=scene_discontinuity_frames,
    )
    print(
        "Detected events: "
        f"{sum(event['type'] == 'pass' for event in events)} passes, "
        f"{sum(event['type'] == 'interception' for event in events)} interceptions"
    )

    tactical_view_converter = TacticalViewConverter(
        court_image_path="images/basketball_court.png"
    )
    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(
        court_keypoints_per_frame
    )
    tactical_player_positions = (
        tactical_view_converter.transform_players_to_tactical_view(
            court_keypoints_per_frame,
            player_tracks,
            discontinuity_frames=scene_discontinuity_frames,
        )
    )

    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters,
    )
    tactical_discontinuity_frames = tactical_view_converter.last_diagnostics.get(
        "temporal_discontinuity",
        [],
    )
    tactical_player_positions = speed_and_distance_calculator.smooth_positions(
        tactical_player_positions,
        discontinuity_frames=tactical_discontinuity_frames,
        window_radius=max(1, round(output_fps * 0.1)),
    )
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(
        tactical_player_positions,
        discontinuity_frames=tactical_discontinuity_frames,
    )
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(
        player_distances_per_frame,
        fps=output_fps,
        tactical_player_positions=tactical_player_positions,
    )

    if args.output_analysis:
        write_analysis_manifest(
            args.output_analysis,
            fps=output_fps,
            frame_count=len(video_frames),
            court_width=tactical_view_converter.width,
            court_height=tactical_view_converter.height,
            tactical_player_positions=tactical_player_positions,
            player_assignment=player_assignment,
            ball_acquisition=ball_acquisition,
            events=events,
            tactical_diagnostics=tactical_view_converter.last_diagnostics,
            assignment_metadata=team_assigner.assignment_metadata,
        )
        print(f"Saved analysis manifest to {args.output_analysis}")

    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    frame_number_drawer = FrameNumberDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()
    pass_interception_drawer = PassInterceptionDrawer(
        event_display_frames=round(output_fps * 1.5),
        team_colors=TEAM_DISPLAY_COLORS,
    )
    team_ball_control_drawer = TeamBallControlDrawer(
        team_colors=TEAM_DISPLAY_COLORS,
    )
    passes, interceptions = events_to_overlay_arrays(events, len(video_frames))

    output_video_frames = player_tracks_drawer.draw(
        video_frames,
        player_tracks,
        team_assignments=player_assignment,
        team_colors=TEAM_DISPLAY_COLORS,
        ball_acquisitions=ball_acquisition,
    )
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    output_video_frames = court_keypoint_drawer.draw(
        output_video_frames,
        court_keypoints_per_frame,
    )
    output_video_frames = frame_number_drawer.draw(output_video_frames)
    output_video_frames = speed_and_distance_drawer.draw(
        output_video_frames,
        player_tracks,
        player_distances_per_frame,
        player_speed_per_frame,
    )
    output_video_frames = pass_interception_drawer.draw(
        output_video_frames,
        passes,
        interceptions,
    )
    output_video_frames = team_ball_control_drawer.draw(
        output_video_frames,
        player_assignment,
        ball_acquisition,
    )
    save_video(output_video_frames, output_path, fps=output_fps)
    print(f"Saved annotated video to {output_path}")


if __name__ == "__main__":
    main()
