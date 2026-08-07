import unittest

from backend.app.batch_job import _last_json_object


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


if __name__ == "__main__":
    unittest.main()
