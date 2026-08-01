import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "benchmarks" / "courtvision_nba_holdout_v1" / "dataset.json"
DEFAULT_WORK = ROOT / "holdout_sources" / "annotation_work" / "nba_holdout_batch_02"
DEFAULT_PREVIEWS = ROOT / "holdout_sources" / "previews" / "nba_holdout_batch_02"
DEFAULT_CLIPS = ["nba_001", "nba_002", "nba_003", "nba_004", "nba_005"]


def sampled_indices(frame_count, interval):
    return list(range(0, frame_count, interval))


def _read_frame(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index}")
    return frame


def _tile(frame, label, width=480, height=270):
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    header = np.zeros((34, width, 3), dtype=np.uint8)
    cv2.putText(
        header,
        label,
        (7, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, resized])


def _write_pages(clip_id, prefix, tiles, output_dir, columns=4):
    page_size = columns * 4
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), page_size):
        page_tiles = list(tiles[start : start + page_size])
        while len(page_tiles) < page_size:
            page_tiles.append(blank.copy())
        rows = [
            np.hstack(page_tiles[row : row + columns])
            for row in range(0, page_size, columns)
        ]
        page = start // page_size + 1
        path = output_dir / f"{clip_id}_{prefix}_{page:02d}.jpg"
        if not cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise ValueError(f"Could not write {path}")


def extract(
    clip_ids,
    dataset_path=DEFAULT_DATASET,
    work_dir=DEFAULT_WORK,
    preview_dir=DEFAULT_PREVIEWS,
    regular_interval=15,
    dense_interval=5,
):
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    clips = {clip["id"]: clip for clip in dataset["clips"]}
    work_dir = Path(work_dir)
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for clip_id in clip_ids:
        if clip_id not in clips:
            raise ValueError(f"Unknown holdout clip: {clip_id}")
        clip = clips[clip_id]
        video_path = ROOT / clip["video_path"]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        regular = sampled_indices(frame_count, regular_interval)
        dense = sampled_indices(frame_count, dense_interval)
        frame_dir = work_dir / "frames" / clip_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        regular_tiles = []
        for frame_index in regular:
            frame = _read_frame(capture, frame_index)
            frame_path = frame_dir / f"{frame_index:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 97]):
                raise ValueError(f"Could not write {frame_path}")
            regular_tiles.append(_tile(frame, f"{clip_id} f{frame_index}"))
        _write_pages(clip_id, "regular", regular_tiles, preview_dir)

        dense_tiles = [
            _tile(_read_frame(capture, frame_index), f"{clip_id} f{frame_index}")
            for frame_index in dense
        ]
        _write_pages(clip_id, "dense", dense_tiles, preview_dir)
        capture.release()
        summary[clip_id] = {
            "frame_count": frame_count,
            "regular_samples": len(regular),
            "dense_samples": len(dense),
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract raw frames for sealed holdout annotation review."
    )
    parser.add_argument("clip_ids", nargs="*", default=DEFAULT_CLIPS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEWS)
    parser.add_argument("--regular-interval", type=int, default=15)
    parser.add_argument("--dense-interval", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            extract(
                args.clip_ids,
                args.dataset.resolve(),
                args.work_dir.resolve(),
                args.preview_dir.resolve(),
                args.regular_interval,
                args.dense_interval,
            ),
            indent=2,
        )
    )
