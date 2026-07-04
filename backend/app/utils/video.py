from pathlib import Path

import cv2


def probe_video(video_path: str | Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    metadata = {
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return metadata


def read_video(
    video_path: str | Path,
    *,
    start_seconds: float = 0,
    duration_seconds: float | None = None,
    target_fps: float | None = None,
    max_width: int | None = None,
    max_decoded_bytes: int | None = None,
):
    metadata = probe_video(video_path)
    source_fps = metadata["fps"]
    if source_fps <= 0:
        raise ValueError(f"Video reports an invalid FPS: {source_fps}")

    output_fps = min(target_fps or source_fps, source_fps)
    scale = min(1.0, (max_width or metadata["width"]) / metadata["width"])
    output_width = max(1, int(round(metadata["width"] * scale)))
    output_height = max(1, int(round(metadata["height"] * scale)))
    available_seconds = max(0.0, metadata["frame_count"] / source_fps - start_seconds)
    selected_seconds = min(duration_seconds or available_seconds, available_seconds)
    estimated_frames = int(selected_seconds * output_fps) + 1
    estimated_bytes = estimated_frames * output_width * output_height * 3
    if max_decoded_bytes is not None and estimated_bytes > max_decoded_bytes:
        estimated_gib = estimated_bytes / 1024**3
        limit_gib = max_decoded_bytes / 1024**3
        raise MemoryError(
            f"Selected video range needs approximately {estimated_gib:.1f} GiB "
            f"of decoded-frame memory (limit: {limit_gib:.1f} GiB). Use "
            "--duration-seconds, --target-fps, and/or --max-width to select a "
            "smaller analysis job."
        )

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    start_frame = int(round(start_seconds * source_fps))
    end_frame = (
        min(metadata["frame_count"], start_frame + int(round(duration_seconds * source_fps)))
        if duration_seconds is not None
        else metadata["frame_count"]
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    next_sample_time = start_seconds
    source_frame = start_frame

    while source_frame < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        frame_time = source_frame / source_fps
        if frame_time + 1e-9 >= next_sample_time:
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(frame)
            next_sample_time += 1.0 / output_fps
        source_frame += 1

    capture.release()
    return frames


def save_video(frames, output_path: str | Path, fps: float = 24):
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
