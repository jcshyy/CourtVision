import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "backend" / "app" / "utils" / "cache.py"
MODULE_SPEC = importlib.util.spec_from_file_location("courtvision_cache", MODULE_PATH)
cache_module = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(cache_module)
load_cache = cache_module.load_cache
save_cache = cache_module.save_cache


class CacheTests(unittest.TestCase):
    def test_save_cache_replaces_existing_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "job" / "tracks.pkl"
            save_cache(cache_path, {"version": 1})

            save_cache(cache_path, {"version": 2})

            self.assertEqual(load_cache(cache_path), {"version": 2})
            self.assertEqual(list(cache_path.parent.glob("*.tmp")), [])

    def test_failed_save_preserves_existing_cache_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "tracks.pkl"
            save_cache(cache_path, {"complete": True})

            with patch.object(
                cache_module.pickle,
                "dump",
                side_effect=RuntimeError("simulated write failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated write failure"):
                    save_cache(cache_path, {"complete": False})

            self.assertEqual(load_cache(cache_path), {"complete": True})
            self.assertEqual(list(cache_path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
