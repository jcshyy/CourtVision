import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.batch_job import MODEL_FILENAMES, _last_json_object, _prepare_models


class BatchJobTests(unittest.TestCase):
    def test_last_json_object_skips_pipeline_log_lines(self):
        result = _last_json_object(
            [
                "Loaded 30 frames",
                '{"status":"needs_team_colors","reason":"uncertain"}',
                "cache saved",
            ]
        )

        self.assertEqual(result["status"], "needs_team_colors")
        self.assertEqual(result["reason"], "uncertain")

    def test_prepare_models_downloads_private_weights(self):
        class S3:
            def __init__(self):
                self.downloads = []

            def download_file(self, bucket, key, destination):
                self.downloads.append((bucket, key))
                Path(destination).write_bytes(b"weights")

        s3 = S3()
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "COURTVISION_MODEL_BUCKET": "private-models",
                "COURTVISION_MODEL_PREFIX": "releases/demo-v1",
            },
            clear=False,
        ):
            _prepare_models(s3, temp)

            self.assertEqual(
                s3.downloads,
                [
                    ("private-models", f"releases/demo-v1/{name}")
                    for name in MODEL_FILENAMES
                ],
            )
            for filename in MODEL_FILENAMES:
                self.assertEqual((Path(temp) / filename).read_bytes(), b"weights")

    def test_prepare_models_keeps_existing_weight(self):
        class S3:
            def download_file(self, *_args):
                raise AssertionError("existing weights must not be downloaded")

        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {"COURTVISION_MODEL_BUCKET": "private-models"},
            clear=False,
        ):
            for filename in MODEL_FILENAMES:
                (Path(temp) / filename).write_bytes(b"cached")
            _prepare_models(S3(), temp)


if __name__ == "__main__":
    unittest.main()
