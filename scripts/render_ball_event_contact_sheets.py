import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render release-to-catch contact sheets from an event report."
    )
    parser.add_argument("video")
    parser.add_argument("report")
    parser.add_argument("output_dir")
    parser.add_argument("--player-tracks")
    return parser.parse_args()


def render(video_path, report_path, output_dir, player_tracks_path=None):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    frames = _read_frames(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    player_tracks = None
    if player_tracks_path:
        with Path(player_tracks_path).open("rb") as track_file:
            player_tracks = pickle.load(track_file)

    for event in report["events"]:
        release = event["release_frame"]
        catch = event["catch_frame"]
        sample_frames = np.linspace(
            max(0, release - 2),
            min(len(frames) - 1, catch + 2),
            12,
        ).round().astype(int)
        tiles = [_tile(frames[index], index) for index in sample_frames]
        sheet = np.vstack(
            [np.hstack(tiles[start : start + 4]) for start in range(0, 12, 4)]
        )
        output_path = output_dir / (
            f"{Path(video_path).stem}_{event['type']}_{event['frame_index']}.jpg"
        )
        cv2.imwrite(str(output_path), sheet)
        outputs.append(output_path)
        if player_tracks is not None:
            _print_track_diagnostics(event, player_tracks)

    return outputs


def _print_track_diagnostics(event, player_tracks):
    start = event["release_frame"]
    end = event["catch_frame"]
    from_id = event["from_player_id"]
    to_id = event["to_player_id"]
    co_visible = sum(
        from_id in player_tracks[index] and to_id in player_tracks[index]
        for index in range(start, end + 1)
    )
    print(
        f"{event['type']}@{end}: tracks {from_id}->{to_id}, "
        f"co-visible frames={co_visible}/{end - start + 1}"
    )


def _read_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"Could not read frames from {video_path}")
    return frames


def _tile(frame, frame_index):
    tile = cv2.resize(frame, (320, 180))
    cv2.rectangle(tile, (0, 0), (110, 25), (0, 0, 0), -1)
    cv2.putText(
        tile,
        f"frame {frame_index}",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def main():
    args = parse_args()
    outputs = render(
        args.video,
        args.report,
        args.output_dir,
        player_tracks_path=args.player_tracks,
    )
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
