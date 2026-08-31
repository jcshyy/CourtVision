import json
import time
import unittest
from unittest.mock import patch

from backend.app import web_api


class WebApiTests(unittest.TestCase):
    def setUp(self):
        web_api._SESSION_SECRET = b"test-session-secret-with-sufficient-length"

    def tearDown(self):
        web_api._SESSION_SECRET = None

    def test_session_round_trip_exposes_csrf_not_cookie_secret(self):
        payload = {
            "v": 1,
            "email": "analyst@example.com",
            "csrf": "csrf-value",
            "exp": int(time.time()) + 60,
        }
        token = web_api._encode_session(payload)
        event = {
            "rawPath": "/api/auth/session",
            "requestContext": {"http": {"method": "GET"}},
            "headers": {"cookie": f"cv_session={token}"},
        }

        response = web_api.lambda_handler(event, None)
        body = json.loads(response["body"])

        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(body["email"], "analyst@example.com")
        self.assertEqual(body["csrfToken"], "csrf-value")
        self.assertNotIn("cv_session", response.get("cookies", []))

    def test_state_change_rejects_missing_csrf(self):
        payload = {
            "v": 1,
            "email": "analyst@example.com",
            "csrf": "expected-token",
            "exp": int(time.time()) + 60,
        }
        token = web_api._encode_session(payload)
        event = {
            "rawPath": "/api/auth/sign-out",
            "requestContext": {"http": {"method": "POST"}},
            "headers": {"cookie": f"cv_session={token}"},
            "body": "{}",
        }

        response = web_api.lambda_handler(event, None)
        body = json.loads(response["body"])

        self.assertEqual(response["statusCode"], 403)
        self.assertEqual(body["error"]["code"], "invalid_request_token")

    def test_state_change_accepts_matching_csrf_and_clears_cookie(self):
        payload = {
            "v": 1,
            "email": "analyst@example.com",
            "csrf": "expected-token",
            "exp": int(time.time()) + 60,
        }
        token = web_api._encode_session(payload)
        event = {
            "rawPath": "/api/auth/sign-out",
            "requestContext": {"http": {"method": "POST"}},
            "headers": {
                "cookie": f"cv_session={token}",
                "x-courtvision-csrf": "expected-token",
            },
            "body": "{}",
        }

        response = web_api.lambda_handler(event, None)

        self.assertEqual(response["statusCode"], 200)
        self.assertIn("Max-Age=0", response["cookies"][0])

    def test_tampered_session_is_rejected(self):
        payload = {
            "v": 1,
            "email": "analyst@example.com",
            "csrf": "csrf-value",
            "exp": int(time.time()) + 60,
        }
        token = web_api._encode_session(payload)
        encoded, signature = token.split(".")
        replacement = "A" if signature[0] != "A" else "B"

        with self.assertRaises(web_api.ApiError) as raised:
            web_api._decode_session(f"{encoded}.{replacement}{signature[1:]}")

        self.assertEqual(raised.exception.code, "invalid_session")

    def test_filename_is_reduced_to_safe_leaf(self):
        self.assertEqual(web_api._safe_filename("../../Game (final)!.mp4"), "Game final.mp4")

    def test_expired_job_is_unavailable_even_before_storage_cleanup(self):
        class JobsTable:
            def get_item(self, **_kwargs):
                return {
                    "Item": {
                        "jobId": "expired-job",
                        "ownerEmail": "analyst@example.com",
                        "expiresAt": int(time.time()) - 1,
                    }
                }

        with patch.object(web_api, "_table", return_value=JobsTable()):
            with self.assertRaises(web_api.ApiError) as raised:
                web_api._owned_job({"email": "analyst@example.com"}, "expired-job")

        self.assertEqual(raised.exception.code, "job_not_found")

    def test_failed_invalid_color_job_can_be_resubmitted_with_black(self):
        job = {
            "jobId": "job-id",
            "status": "failed",
            "errorMessage": "Invalid team-color configuration: rejected: #000000",
        }
        expected = {"statusCode": 202}

        with patch.object(web_api, "_owned_job", return_value=job), patch.object(
            web_api, "_submit_batch", return_value=expected
        ) as submit:
            response = web_api._submit_team_colors(
                {"email": "analyst@example.com"},
                "job-id",
                {"team1Color": "#000000", "team2Color": "#FFFFFF"},
            )

        self.assertEqual(response, expected)
        self.assertEqual(submit.call_args.args[0]["team1Color"], "#000000")

    def test_legacy_worker_color_compatibility_preserves_black_semantics(self):
        self.assertEqual(web_api._worker_compatible_jersey_color("#000000"), "#272727")
        self.assertEqual(web_api._worker_compatible_jersey_color("#001020"), "#001427")
        self.assertEqual(web_api._worker_compatible_jersey_color("#1E55D6"), "#1E55D6")

    def test_nearly_identical_black_swatches_are_rejected_after_compatibility_lift(self):
        job = {"jobId": "job-id", "status": "needs_team_colors"}
        with patch.object(web_api, "_owned_job", return_value=job), self.assertRaises(
            web_api.ApiError
        ) as raised:
            web_api._submit_team_colors(
                {"email": "analyst@example.com"},
                "job-id",
                {"team1Color": "#000000", "team2Color": "#010101"},
            )

        self.assertEqual(raised.exception.code, "invalid_team_colors")


if __name__ == "__main__":
    unittest.main()
