import tempfile
import unittest
from hashlib import sha256 as hashlib_sha256
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_ebard_model import install_model, sha256


class PrepareEbardModelTests(unittest.TestCase):
    def test_existing_verified_model_is_reused_without_network(self):
        payload = b"verified model fixture"
        expected = hashlib_sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "ebard.pt"
            model.write_bytes(payload)
            with patch("scripts.prepare_ebard_model.EBARD_SHA256", expected):
                self.assertEqual(sha256(model), expected)
                self.assertEqual(install_model(model), model)

    def test_unverified_existing_model_requires_force(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ebard.pt"
            output.write_bytes(b"not the checkpoint")

            with self.assertRaisesRegex(RuntimeError, "unexpected SHA-256"):
                install_model(output)


if __name__ == "__main__":
    unittest.main()
