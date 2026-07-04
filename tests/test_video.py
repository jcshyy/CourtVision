import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.app.utils.video import probe_video, read_video


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


if __name__ == "__main__":
    unittest.main()
