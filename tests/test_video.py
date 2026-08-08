import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.app.utils.video import (
    detect_scene_discontinuities,
    probe_video,
    read_video,
    save_video,
)


class VideoReadTests(unittest.TestCase):
    def _write_video(self, path, frame_count=12, fps=6, size=(64, 48)):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            size,
        )
        for index in range(frame_count):
            writer.write(np.full((size[1], size[0], 3), index, dtype=np.uint8))
        writer.release()

    def test_reads_a_resized_sampled_clip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "clip.mp4"
            self._write_video(video)

            frames = read_video(
                video,
                start_seconds=0.5,
                duration_seconds=1.0,
                target_fps=3,
                max_width=32,
            )

            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0].shape[:2], (24, 32))
            self.assertEqual(probe_video(video)["frame_count"], 12)

    def test_rejects_a_selection_above_the_memory_limit_before_decoding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "clip.mp4"
            self._write_video(video)

            with self.assertRaisesRegex(MemoryError, "--duration-seconds"):
                read_video(video, max_decoded_bytes=1)

    def test_saves_a_browser_compatible_h264_video(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "review.mp4"
            frames = [np.full((48, 64, 3), index * 20, dtype=np.uint8) for index in range(3)]

            save_video(frames, video, fps=6)

            capture = cv2.VideoCapture(str(video))
            fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
            capture.release()
            self.assertIn(codec.lower(), {"avc1", "h264"})

    def test_detects_persistent_cut_but_ignores_single_frame_flash(self):
        green = np.full((60, 80, 3), (0, 180, 0), dtype=np.uint8)
        red = np.full((60, 80, 3), (0, 0, 180), dtype=np.uint8)
        white = np.full((60, 80, 3), 255, dtype=np.uint8)

        self.assertEqual(
            detect_scene_discontinuities([green, green, red, red]),
            [2],
        )
        self.assertEqual(
            detect_scene_discontinuities([green, white, green, green]),
            [],
        )


if __name__ == "__main__":
    unittest.main()
