"""Adapt authorized official MultiSports annotations to temporal event scoring.

No download or license acceptance is performed. `test_videos` in the released
GT file is validation, not the hidden competition test set. Input may be JSON
or the official NumPy-only pickle (loaded with a restricted global allowlist).
Frame rates must be supplied explicitly; never assume the source-video FPS
matches the distributed frame sequence.
"""
import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path


LABELS = {
    "basketball pass": "pass",
    "basketball 2-point shot": "shot_attempt",
    "basketball 3-point shot": "shot_attempt",
    "basketball pass steal": "interception",
    "basketball dribble steal": "interception",
    "basketball pass-inbound": "throw_in",
}


class AnnotationUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        import numpy as np
        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        for prefix in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed[(prefix, "_reconstruct")] = np._core.multiarray._reconstruct
            allowed[(prefix, "scalar")] = np._core.multiarray.scalar
        for prefix in ("numpy.core.numeric", "numpy._core.numeric"):
            allowed[(prefix, "_frombuffer")] = np._core.numeric._frombuffer
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Disallowed annotation global: {module}.{name}")
        return allowed[(module, name)]


def convert(data, fps_by_video, split="validation"):
    key = "train_videos" if split == "train" else "test_videos"
    selected = sorted(v for v in data[key][0] if v.startswith("basketball/"))
    if not selected:
        raise ValueError("No basketball videos in selected official split")
    videos, events, ignored = [], [], Counter()
    for video_id in selected:
        fps = float(fps_by_video[video_id])
        if not 0 < fps < 1000:
            raise ValueError(f"Invalid FPS for {video_id}")
        count = int(data["nframes"][video_id])
        videos.append({"video_id": video_id, "fps": fps, "frame_count": count,
                       "split": split, "annotations_complete_for_scored_types": True})
        for label_id, tubes in data["gttubes"].get(video_id, {}).items():
            label = data["labels"][int(label_id)]
            event_type = LABELS.get(label)
            if event_type is None:
                ignored[label] += len(tubes)
                continue
            for ordinal, tube in enumerate(tubes):
                frame_ids = [int(row[0]) for row in tube]
                if not frame_ids or min(frame_ids) < 1 or max(frame_ids) > count:
                    raise ValueError(f"Invalid tube frames for {video_id}")
                events.append({"video_id": video_id, "type": event_type,
                               "start_frame": min(frame_ids) - 1, "end_frame": max(frame_ids) - 1,
                               "source_label": label, "source_tube_id": f"{video_id}:{label_id}:{ordinal}"})
    return {"name": "MultiSports basketball temporal event adapter",
            "source": "https://github.com/MCG-NJU/MultiSports",
            "license": "CC-BY-NC-4.0; obtain authorized access before using dataset",
            "scored_types": sorted(set(LABELS.values())),
            "note": "Action tube intervals, not release/rim/catch timestamps. Rebounds and free throws are not scored by the current production contract. This is not official spatial tube mAP.",
            "ignored_source_labels": dict(ignored), "videos": videos, "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--fps-json", type=Path, required=True,
                        help="JSON mapping exact official video IDs to verified distributed-frame FPS")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.annotations.suffix == ".json":
        data = json.loads(args.annotations.read_text(encoding="utf-8"))
    else:
        with args.annotations.open("rb") as source:
            data = AnnotationUnpickler(source).load()
    result = convert(data, json.loads(args.fps_json.read_text()), args.split)
    result["source_sha256"] = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(result['events'])} events in {len(result['videos'])} videos")


if __name__ == "__main__":
    main()
