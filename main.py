import argparse
import json

from backend.app.cache_paths import cache_path, default_job_output_path, video_cache_dir
from backend.app.analytics import (
    BallAcquisitionDetector,
    PassInterceptionDetector,
    SpeedAndDistanceCalculator,
    TacticalViewConverter,
    events_from_arrays,
)
from backend.app.config import OUTPUT_DIR, STUBS_DIR
from backend.app.detection import CourtKeypointDetector
from backend.app.team_assignment import NeedsTeamColorsError, TeamAssigner
from backend.app.tracking import BallTracker, PlayerTracker
from backend.app.utils import probe_video, read_video, save_video
from backend.app.visualization import (
    BallTracksDrawer,
    CourtKeypointDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    PlayerTracksDrawer,
    SpeedAndDistanceDrawer,
    TacticalViewDrawer,
    TeamBallControlDrawer,
)

TEAM_DISPLAY_COLORS = {
    1: (255, 80, 0),   # Bright blue in OpenCV BGR order.
    2: (0, 215, 255),  # Bright yellow in OpenCV BGR order.
}


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
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--max-width", type=int, default=None)
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


def main():
    args = parse_args()
    video_path = args.input_video
    output_path = args.output_video or default_output_path(video_path)
    try:
        team_assigner = TeamAssigner(
            team_1_color=args.team_1_color,
            team_2_color=args.team_2_color,
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
    cache_dir = video_cache_dir(args.stub_path, video_path, processing_identity)
    print(f"Using per-video cache directory: {cache_dir.resolve()}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    assignment_metadata_path = cache_dir / team_assigner.metadata_filename

    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()
    court_keypoint_detector = CourtKeypointDetector()

    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_cache=True,
        cache_path=cache_path(cache_dir, "player_track_stubs.pkl"),
    )
    ball_tracks = ball_tracker.get_object_tracks(
        video_frames,
        read_from_cache=True,
        cache_path=cache_path(cache_dir, "ball_track_stubs.pkl"),
    )
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(
        video_frames,
        read_from_cache=True,
        cache_path=cache_path(cache_dir, "court_key_points_stub.pkl"),
    )

    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_positions(ball_tracks)

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

    ball_acquisition_detector = BallAcquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_acquisitions(
        player_tracks,
        ball_tracks,
    )

    pass_interception_detector = PassInterceptionDetector()
    passes = pass_interception_detector.detect_passes(
        ball_acquisition,
        player_assignment,
    )
    interceptions = pass_interception_detector.detect_interceptions(
        ball_acquisition,
        player_assignment,
    )
    events = events_from_arrays(passes, interceptions, ball_acquisition)

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
        )
    )

    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters,
    )
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(
        tactical_player_positions
    )
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(
        player_distances_per_frame
    )

    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    frame_number_drawer = FrameNumberDrawer()
    pass_interception_drawer = PassInterceptionDrawer()
    tactical_view_drawer = TacticalViewDrawer(
        team_1_color=TEAM_DISPLAY_COLORS[1],
        team_2_color=TEAM_DISPLAY_COLORS[2],
    )
    speed_and_distance_drawer = SpeedAndDistanceDrawer()

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
    output_video_frames = team_ball_control_drawer.draw(
        output_video_frames,
        player_assignment,
        ball_acquisition,
    )
    output_video_frames = pass_interception_drawer.draw(
        output_video_frames,
        passes,
        interceptions,
    )
    output_video_frames = speed_and_distance_drawer.draw(
        output_video_frames,
        player_tracks,
        player_distances_per_frame,
        player_speed_per_frame,
    )
    output_video_frames = tactical_view_drawer.draw(
        output_video_frames,
        tactical_view_converter.court_image_path,
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.key_points,
        tactical_player_positions,
        player_assignment,
        ball_acquisition,
    )

    save_video(output_video_frames, output_path, fps=output_fps)
    print(f"Saved annotated video to {output_path}")


if __name__ == "__main__":
    main()
