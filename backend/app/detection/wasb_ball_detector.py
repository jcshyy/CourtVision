"""Basketball-only adapter for the pretrained WASB-SBDT HRNet.

WASB-SBDT is MIT licensed: https://github.com/nttcom/WASB-SBDT
The upstream model consumes three RGB frames at 512x288 and emits three ball
heatmaps. This adapter turns those heatmaps into CourtVision candidate boxes.
"""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.app.config.settings import WASB_BALL_DETECTOR_PATH


WASB_INPUT_WIDTH = 512
WASB_INPUT_HEIGHT = 288
WASB_FRAMES_IN = 3
WASB_HEATMAP_THRESHOLD = 0.50
WASB_MAX_CANDIDATES_PER_FRAME = 8
_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class WasbBallDetector:
    """Run the official basketball WASB checkpoint on consecutive frames."""

    def __init__(
        self,
        model_path=WASB_BALL_DETECTOR_PATH,
        *,
        device=None,
        heatmap_threshold=WASB_HEATMAP_THRESHOLD,
        batch_size=2,
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing WASB TorchScript model: {model_path}. "
                "Run scripts/prepare_wasb_model.py first."
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        self.heatmap_threshold = float(heatmap_threshold)
        self.batch_size = max(1, int(batch_size))

    def detect_frames(self, frames, *, step=1):
        """Return WASB ball candidates aligned one-to-one with ``frames``."""
        if not frames:
            return []
        if step < 1:
            raise ValueError("WASB inference step must be positive")
        for frame in frames:
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("WASB requires non-empty BGR image frames")

        prepared = [_prepare_frame(frame) for frame in frames]
        windows = _window_indices(len(frames), step)
        raw_candidates = defaultdict(list)
        with torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                batch_windows = windows[start : start + self.batch_size]
                inputs = torch.stack([
                    torch.cat([prepared[index][0] for index in window], dim=0)
                    for window in batch_windows
                ]).to(self.device)
                heatmaps = self.model(inputs).sigmoid().cpu().numpy()
                for batch_index, window in enumerate(batch_windows):
                    for channel, frame_index in enumerate(window):
                        inverse = prepared[frame_index][1]
                        candidates = _heatmap_candidates(
                            heatmaps[batch_index, channel],
                            inverse,
                            frames[frame_index].shape,
                            self.heatmap_threshold,
                        )
                        raw_candidates[frame_index].extend(candidates)

        return [
            _merge_window_candidates(raw_candidates[index])
            for index in range(len(frames))
        ]


def _window_indices(frame_count, step):
    if frame_count <= WASB_FRAMES_IN:
        padded = tuple(min(index, frame_count - 1) for index in range(WASB_FRAMES_IN))
        return [padded]
    starts = list(range(0, frame_count - WASB_FRAMES_IN + 1, step))
    final_start = frame_count - WASB_FRAMES_IN
    if starts[-1] != final_start:
        starts.append(final_start)
    return [tuple(range(start, start + WASB_FRAMES_IN)) for start in starts]


def _affine_transform(frame_shape, *, inverse=False):
    height, width = frame_shape[:2]
    center = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
    scale = float(max(height, width))
    source = np.zeros((3, 2), dtype=np.float32)
    destination = np.zeros((3, 2), dtype=np.float32)
    source[0] = center
    source[1] = center + np.asarray([0.0, -scale / 2.0], dtype=np.float32)
    source[2] = source[1] + np.asarray([-scale / 2.0, 0.0], dtype=np.float32)
    destination[0] = [WASB_INPUT_WIDTH / 2.0, WASB_INPUT_HEIGHT / 2.0]
    destination[1] = [WASB_INPUT_WIDTH / 2.0, -WASB_INPUT_WIDTH / 2.0 + WASB_INPUT_HEIGHT / 2.0]
    destination[2] = destination[1] + np.asarray(
        [-WASB_INPUT_WIDTH / 2.0, 0.0], dtype=np.float32
    )
    first, second = (destination, source) if inverse else (source, destination)
    return cv2.getAffineTransform(first, second)


def _prepare_frame(frame):
    transform = _affine_transform(frame.shape)
    inverse = _affine_transform(frame.shape, inverse=True)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    warped = cv2.warpAffine(
        rgb,
        transform,
        (WASB_INPUT_WIDTH, WASB_INPUT_HEIGHT),
        flags=cv2.INTER_LINEAR,
    )
    normalized = (warped.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).contiguous()
    return tensor, inverse


def _heatmap_candidates(heatmap, inverse, frame_shape, threshold):
    mask = (heatmap > threshold).astype(np.uint8)
    component_count, labels = cv2.connectedComponents(mask)
    frame_height, frame_width = frame_shape[:2]
    diameter = float(np.clip(round(frame_width / 100.0), 8, 20))
    candidates = []
    for label in range(1, component_count):
        ys, xs = np.where(labels == label)
        if not len(xs):
            continue
        weights = heatmap[ys, xs]
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            continue
        model_point = np.asarray(
            [float(np.dot(xs, weights) / weight_sum), float(np.dot(ys, weights) / weight_sum), 1.0],
            dtype=np.float32,
        )
        frame_point = inverse @ model_point
        center_x = float(np.clip(frame_point[0], 0, frame_width - 1))
        center_y = float(np.clip(frame_point[1], 0, frame_height - 1))
        confidence = float(weights.max())
        candidates.append({
            "bbox": [
                center_x - diameter / 2,
                center_y - diameter / 2,
                center_x + diameter / 2,
                center_y + diameter / 2,
            ],
            "confidence": confidence,
            "detection_source": "wasb_temporal",
            "wasb_component_score": weight_sum,
        })
    return candidates


def _center(candidate):
    x1, y1, x2, y2 = candidate["bbox"]
    return np.asarray([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def _merge_window_candidates(candidates, merge_radius=10.0):
    merged = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item["confidence"], item["wasb_component_score"]),
        reverse=True,
    ):
        duplicate = next(
            (
                kept
                for kept in merged
                if float(np.linalg.norm(_center(candidate) - _center(kept)))
                <= merge_radius
            ),
            None,
        )
        if duplicate is not None:
            duplicate["wasb_window_support"] = duplicate.get(
                "wasb_window_support", 1
            ) + 1
            continue
        record = dict(candidate)
        record["wasb_window_support"] = 1
        merged.append(record)
        if len(merged) >= WASB_MAX_CANDIDATES_PER_FRAME:
            break
    return merged
