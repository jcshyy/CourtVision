import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def detect_scene_discontinuities(
    frames,
    *,
    minimum_histogram_distance: float = 0.40,
    transient_return_ratio: float = 0.50,
):
    """Return frame indices that begin a persistently different scene.

    A coarse hue/saturation histogram makes the signal insensitive to player
    motion and resolution. Looking one frame ahead prevents a single-frame
    broadcast flash or graphic from being mistaken for a camera cut.
    """
    if minimum_histogram_distance <= 0 or minimum_histogram_distance > 1:
        raise ValueError("Histogram distance must be in (0, 1]")
    if transient_return_ratio < 0 or transient_return_ratio > 1:
        raise ValueError("Transient return ratio must be in [0, 1]")
    if len(frames) < 2:
        return []

    descriptors = [_scene_histogram(frame) for frame in frames]
    discontinuities = []
    for frame_index in range(1, len(frames)):
        previous_distance = _histogram_total_variation(
            descriptors[frame_index - 1], descriptors[frame_index]
        )
        if previous_distance < minimum_histogram_distance:
            continue

        if frame_index >= 2:
            incoming_distance = _histogram_total_variation(
                descriptors[frame_index - 2], descriptors[frame_index - 1]
            )
            return_distance = _histogram_total_variation(
                descriptors[frame_index - 2], descriptors[frame_index]
            )
            if (
                incoming_distance >= minimum_histogram_distance
                and return_distance
                <= minimum_histogram_distance * transient_return_ratio
            ):
                continue

        if frame_index + 1 < len(frames):
            next_distance = _histogram_total_variation(
                descriptors[frame_index], descriptors[frame_index + 1]
            )
            bridge_distance = _histogram_total_variation(
                descriptors[frame_index - 1], descriptors[frame_index + 1]
            )
            if (
                next_distance >= minimum_histogram_distance
                and bridge_distance
                <= minimum_histogram_distance * transient_return_ratio
            ):
                continue

        discontinuities.append(frame_index)
    return discontinuities


def _scene_histogram(frame):
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("Scene-cut detection requires non-empty video frames")
    height, width = frame.shape[:2]
    y1, y2 = int(height * 0.10), max(int(height * 0.90), 1)
    x1, x2 = int(width * 0.10), max(int(width * 0.90), 1)
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    total = float(histogram.sum())
    return histogram.reshape(-1).astype(np.float32) / max(total, 1.0)


def _histogram_total_variation(first, second):
    return 0.5 * float(np.abs(first - second).sum())


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
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as error:
            raise RuntimeError(
                "FFmpeg is required to create a browser-compatible review video."
            ) from error

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stderr is not None
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("All video frames must have the same dimensions.")
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        error_output = process.stderr.read().decode("utf-8", errors="replace").strip()
        process.stderr.close()
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        process.stderr.close()
        path.unlink(missing_ok=True)
        raise

    if return_code != 0:
        path.unlink(missing_ok=True)
        raise RuntimeError(error_output or "FFmpeg could not encode the review video.")
