import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = (
    ROOT / "benchmarks" / "courtvision_nba_holdout_v1" / "calibration"
)
DEFAULT_OUTPUT = (
    ROOT / "holdout_sources" / "previews" / "nba_holdout_annotation_review"
)


def _records(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _frame(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index}")
    return frame


def _fit(frame, width=480, height=270):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _with_header(tile, lines, color=(255, 255, 255)):
    height = 25 * len(lines) + 8
    header = np.zeros((height, tile.shape[1], 3), dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(
            header,
            line,
            (7, 22 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
    return np.vstack([header, tile])


def _render_frame_tile(frame, record):
    center = record["ball"]["center_px"]
    if center is not None:
        cv2.drawMarker(
            frame,
            (int(center[0]), int(center[1])),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            22,
            2,
        )
        cv2.circle(
            frame,
            (int(center[0]), int(center[1])),
            14,
            (0, 255, 255),
            2,
        )
    tile = _fit(frame, 640, 360)
    possession = record["possession"]
    team = possession["team_id"] or "-"
    holder = possession["holder"] or "-"
    tile = _with_header(
        tile,
        [
            (
                f"{record['video_id']} f{record['frame_index']} "
                f"{record['scene_id']} [{record['review_status']}]"
            ),
            (
                f"ball={record['ball']['visibility']}/"
                f"{record['ball']['confidence']} state={possession['state']}"
            ),
            f"team={team} holder={holder}",
        ],
    )
    return tile


def _write_pages(clip_id, tiles, output_dir, prefix, columns=4):
    page_size = columns * 4
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), page_size):
        page_tiles = list(tiles[start:start + page_size])
        while len(page_tiles) < page_size:
            page_tiles.append(blank.copy())
        rows = [
            np.hstack(page_tiles[row:row + columns])
            for row in range(0, page_size, columns)
        ]
        sheet = np.vstack(rows)
        page = start // page_size + 1
        path = output_dir / f"{clip_id}_{prefix}_{page:02d}.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise ValueError(f"Could not write {path}")


def _event_frames(event):
    values = [
        event["start_frame"],
        event.get("release_frame"),
        event.get("catch_frame"),
        event["end_frame"],
    ]
    result = []
    for value in values:
        if value is not None and value not in result:
            result.append(value)
    return result


def render(calibration_dir=DEFAULT_CALIBRATION, output_dir=DEFAULT_OUTPUT):
    calibration_dir = Path(calibration_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (calibration_dir / "manifest.json").read_text(encoding="utf-8")
    )
    clips = {clip["id"]: clip for clip in manifest["clips"]}
    frames_by_clip = defaultdict(list)
    for record in _records(calibration_dir / "frames.jsonl"):
        frames_by_clip[record["video_id"]].append(record)
    events_by_clip = defaultdict(list)
    for event in _records(calibration_dir / "events.jsonl"):
        events_by_clip[event["video_id"]].append(event)

    for clip_id, clip in clips.items():
        capture = cv2.VideoCapture(str(ROOT / clip["video_path"]))
        if not capture.isOpened():
            raise ValueError(f"Could not open {clip['video_path']}")
        frame_tiles = [
            _render_frame_tile(
                _frame(capture, record["frame_index"]),
                record,
            )
            for record in frames_by_clip[clip_id]
        ]
        _write_pages(
            clip_id,
            frame_tiles,
            output_dir,
            "frames",
            columns=2,
        )

        event_tiles = []
        for event_index, event in enumerate(events_by_clip[clip_id], 1):
            for frame_index in _event_frames(event):
                tile = _fit(_frame(capture, frame_index), 640, 360)
                tile = _with_header(
                    tile,
                    [
                        (
                            f"event {event_index} {event['event_type']} "
                            f"{event['confidence']} [{event['review_status']}]"
                        ),
                        (
                            f"frame={frame_index} range="
                            f"{event['start_frame']}-{event['end_frame']} "
                            f"release={event.get('release_frame')} "
                            f"catch={event.get('catch_frame')}"
                        ),
                        (
                            f"from={event.get('from_team_id') or '-'} "
                            f"to={event.get('to_team_id') or '-'} "
                            f"outcome={event.get('outcome') or '-'}"
                        ),
                    ],
                    color=(0, 255, 255),
                )
                event_tiles.append(tile)
        if event_tiles:
            _write_pages(
                clip_id,
                event_tiles,
                output_dir,
                "events",
                columns=3,
            )
        capture.release()
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render review sheets for sealed holdout annotations."
    )
    parser.add_argument(
        "--calibration-dir", type=Path, default=DEFAULT_CALIBRATION
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(render(args.calibration_dir.resolve(), args.output_dir.resolve()))
