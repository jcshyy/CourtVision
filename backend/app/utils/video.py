from pathlib import Path

import cv2


def read_video(video_path: str | Path):
    capture = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)

    capture.release()
    return frames


def save_video(frames, output_path: str | Path, fps: int = 24):
    if not frames:
        raise ValueError("Cannot save an empty frame list.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame in frames:
        writer.write(frame)

    writer.release()
