import argparse
import logging

from backend.app.config import STUBS_DIR
from backend.app.team_assignment.diagnostics import (
    diagnose_team_track,
    write_diagnostic,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect team-assignment evidence for one cached player track."
    )
    parser.add_argument("video", help="Input video used to create the track cache.")
    parser.add_argument("track_id", type=int, help="Tracked player ID to inspect.")
    parser.add_argument(
        "--cache-root",
        default=str(STUBS_DIR),
        help="Root containing content-keyed per-video caches.",
    )
    parser.add_argument(
        "--assignment-cache-name",
        default=None,
        help="Optional assignment cache filename to compare against.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path; otherwise prints to stdout.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help=(
            "Inspect cached bboxes and emitted assignments without decoding "
            "video frames or importing OpenCV."
        ),
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
    args = parser.parse_args()
    if (args.team_1_color is None) != (args.team_2_color is None):
        parser.error("--team-1-color and --team-2-color must be provided together")
    if args.cache_only and args.team_1_color is not None:
        parser.error("team colors require pixel diagnostics; remove --cache-only")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    diagnostic = diagnose_team_track(
        args.video,
        args.cache_root,
        args.track_id,
        assignment_cache_name=args.assignment_cache_name,
        cache_only=args.cache_only,
        team_1_color=args.team_1_color,
        team_2_color=args.team_2_color,
    )
    write_diagnostic(diagnostic, args.output)


if __name__ == "__main__":
    main()
