import argparse
import pickle
from pathlib import Path

import cv2


TEAM_COLORS = {
    -1: (128, 128, 128),
    1: (255, 80, 0),
    2: (0, 215, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one representative crop per cached player track."
    )
    parser.add_argument("video")
    parser.add_argument("track_cache")
    parser.add_argument("assignment_cache")
    parser.add_argument("output_image")
    parser.add_argument("--cell-width", type=int, default=180)
    parser.add_argument("--cell-height", type=int, default=220)
    parser.add_argument("--columns", type=int, default=8)
    return parser.parse_args()


def _load(path):
    with Path(path).open("rb") as file:
        return pickle.load(file)


def _read_frame(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index}")
    return frame


def _representative_tracks(tracks):
    representatives = {}
    for frame_index, frame_tracks in enumerate(tracks):
        for track_id, track in frame_tracks.items():
            x1, y1, x2, y2 = track["bbox"]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            current = representatives.get(track_id)
            if current is None or area > current[0]:
                representatives[track_id] = (area, frame_index, track["bbox"])
    return representatives


def _track_team(assignments, track_id):
    teams = [frame[track_id] for frame in assignments if track_id in frame]
    if not teams:
        return -1
    return max(set(teams), key=teams.count)


def _crop_cell(frame, bbox, track_id, team_id, frame_index, width, height):
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    canvas = frame[:1, :1].copy() * 0
    canvas = cv2.resize(canvas, (width, height))
    if crop.size:
        scale = min((width - 8) / crop.shape[1], (height - 42) / crop.shape[0])
        resized = cv2.resize(
            crop,
            (
                max(1, int(round(crop.shape[1] * scale))),
                max(1, int(round(crop.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        offset_x = (width - resized.shape[1]) // 2
        offset_y = 34 + (height - 34 - resized.shape[0]) // 2
        canvas[
            offset_y : offset_y + resized.shape[0],
            offset_x : offset_x + resized.shape[1],
        ] = resized

    color = TEAM_COLORS.get(team_id, TEAM_COLORS[-1])
    cv2.rectangle(canvas, (1, 1), (width - 2, height - 2), color, 3)
    cv2.putText(
        canvas,
        f"ID {track_id}  team {team_id}  f{frame_index}",
        (7, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def main():
    args = parse_args()
    tracks = _load(args.track_cache)
    assignments = _load(args.assignment_cache)
    representatives = _representative_tracks(tracks)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {args.video}")

    records = sorted(
        representatives.items(),
        key=lambda item: (_track_team(assignments, item[0]), item[0]),
    )
    cells = []
    for track_id, (_, frame_index, bbox) in records:
        frame = _read_frame(capture, frame_index)
        cells.append(
            _crop_cell(
                frame,
                bbox,
                track_id,
                _track_team(assignments, track_id),
                frame_index,
                args.cell_width,
                args.cell_height,
            )
        )
    capture.release()

    blank = cells[0] * 0
    rows = []
    for offset in range(0, len(cells), args.columns):
        row = cells[offset : offset + args.columns]
        row += [blank] * (args.columns - len(row))
        rows.append(cv2.hconcat(row))
    gallery = cv2.vconcat(rows)

    output_path = Path(args.output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), gallery):
        raise ValueError(f"Could not write image: {output_path}")
    print(f"Wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
