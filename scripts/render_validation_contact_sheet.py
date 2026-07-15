import argparse
import math
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create source/render frame pairs for visual validation."
    )
    parser.add_argument("source_video")
    parser.add_argument("rendered_video")
    parser.add_argument("output_image")
    parser.add_argument(
        "--frames",
        required=True,
        help="Comma-separated zero-based frame indices.",
    )
    parser.add_argument("--pair-width", type=int, default=960)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--source-start-seconds", type=float, default=0.0)
    return parser.parse_args()


def _read_selected_frames(path, frame_indices):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    selected = {}
    for frame_index in sorted(frame_indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"Could not read frame {frame_index} from {path}")
        selected[frame_index] = frame
    capture.release()
    return selected


def _label(frame, text):
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _resize_width(frame, width):
    scale = width / frame.shape[1]
    return cv2.resize(
        frame,
        (width, int(round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _video_fps(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps


def build_contact_sheet(
    source_video,
    rendered_video,
    frame_indices,
    pair_width,
    columns,
    source_start_seconds=0.0,
):
    source_fps = _video_fps(source_video)
    rendered_fps = _video_fps(rendered_video)
    source_index_by_rendered = {
        frame_index: int(
            round(
                source_start_seconds * source_fps
                + frame_index * source_fps / rendered_fps
            )
        )
        for frame_index in frame_indices
    }
    source_frames_by_index = _read_selected_frames(
        source_video,
        set(source_index_by_rendered.values()),
    )
    source_frames = {
        rendered_index: source_frames_by_index[source_index]
        for rendered_index, source_index in source_index_by_rendered.items()
    }
    rendered_frames = _read_selected_frames(rendered_video, frame_indices)
    pairs = []

    for frame_index in frame_indices:
        half_width = pair_width // 2
        source_index = source_index_by_rendered[frame_index]
        source = _resize_width(source_frames[frame_index], half_width)
        rendered = _resize_width(rendered_frames[frame_index], half_width)
        height = min(source.shape[0], rendered.shape[0])
        pair = cv2.hconcat(
            [
                _label(source[:height], f"source frame {source_index}"),
                _label(rendered[:height], f"rendered frame {frame_index}"),
            ]
        )
        pairs.append(pair)

    rows = []
    row_count = math.ceil(len(pairs) / columns)
    blank = pairs[0] * 0
    for row_index in range(row_count):
        row_pairs = pairs[row_index * columns : (row_index + 1) * columns]
        row_pairs += [blank] * (columns - len(row_pairs))
        rows.append(cv2.hconcat(row_pairs))
    return cv2.vconcat(rows)


def main():
    args = parse_args()
    frame_indices = [int(value.strip()) for value in args.frames.split(",")]
    if not frame_indices or min(frame_indices) < 0:
        raise ValueError("Frame indices must be non-negative")
    if args.pair_width < 2 or args.columns < 1:
        raise ValueError("Pair width and columns must be positive")

    contact_sheet = build_contact_sheet(
        args.source_video,
        args.rendered_video,
        frame_indices,
        args.pair_width,
        args.columns,
        source_start_seconds=args.source_start_seconds,
    )
    output_path = Path(args.output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), contact_sheet):
        raise ValueError(f"Could not write image: {output_path}")
    print(f"Wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
