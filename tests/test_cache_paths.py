import tempfile
import unittest
from pathlib import Path

from backend.app.cache_paths import (
    cache_path,
    default_job_output_path,
    video_cache_dir,
)


class VideoCachePathTests(unittest.TestCase):
    def test_replaced_upload_at_same_path_uses_a_new_cache_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "upload.mp4"
            video.write_bytes(b"first-video")
            first_cache = video_cache_dir(root / "stubs", video)

            video.write_bytes(b"other-video")
            replacement_cache = video_cache_dir(root / "stubs", video)

            self.assertEqual(len(b"first-video"), len(b"other-video"))
            self.assertNotEqual(first_cache, replacement_cache)

    def test_different_video_contents_use_different_cache_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_video = root / "upload.mp4"
            second_video = root / "second" / "upload.mp4"
            second_video.parent.mkdir()
            first_video.write_bytes(b"first-video")
            second_video.write_bytes(b"second-video")

            first_cache = video_cache_dir(root / "stubs", first_video)
            second_cache = video_cache_dir(root / "stubs", second_video)

            self.assertNotEqual(first_cache, second_cache)
            self.assertEqual(len(first_cache.name.rsplit("-", 1)[1]), 64)
            self.assertEqual(first_cache.parent, root / "stubs")
            self.assertEqual(second_cache.parent, root / "stubs")

    def test_same_video_contents_reuse_the_same_cache_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_video = root / "first.mp4"
            second_video = root / "first.mov"
            first_video.write_bytes(b"same-video")
            second_video.write_bytes(b"same-video")

            first_cache = video_cache_dir(root / "stubs", first_video)
            second_cache = video_cache_dir(root / "stubs", second_video)

            self.assertEqual(first_cache.name.split("-", 1)[1], second_cache.name.split("-", 1)[1])
            self.assertEqual(
                cache_path(first_cache, "player_track_stubs.pkl"),
                first_cache / "player_track_stubs.pkl",
            )

    def test_processing_options_use_separate_cache_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "game.mp4"
            video.write_bytes(b"same-video")

            first = video_cache_dir(root / "stubs", video, "start=0;duration=20")
            second = video_cache_dir(root / "stubs", video, "start=20;duration=20")

            self.assertNotEqual(first, second)

    def test_default_output_is_reused_for_the_same_video(self):
        output_root = Path("output_videos")

        first_output = default_job_output_path(output_root, "uploads/game.mp4")
        second_output = default_job_output_path(output_root, "uploads/game.mp4")

        self.assertEqual(first_output, second_output)
        self.assertEqual(first_output, Path("output_videos/game.avi"))
        self.assertEqual(first_output.parent, output_root)
        self.assertEqual(first_output.suffix, ".avi")

    def test_default_output_uses_safe_video_stem(self):
        output = default_job_output_path(
            "output_videos",
            "uploads/Game 1 (final).mp4",
        )

        self.assertEqual(output, Path("output_videos/Game_1_final.avi"))


if __name__ == "__main__":
    unittest.main()
