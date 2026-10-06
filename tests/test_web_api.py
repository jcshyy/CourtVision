import json
import time
import unittest
from datetime import UTC, datetime
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

    def test_responses_allow_the_official_site_to_send_credentials(self):
        event = {
            "rawPath": "/api/auth/session",
            "requestContext": {"http": {"method": "OPTIONS"}},
            "headers": {"origin": "https://courtvision.video"},
        }

        response = web_api.lambda_handler(event, None)

        self.assertEqual(response["statusCode"], 204)
        self.assertEqual(
            response["headers"]["access-control-allow-origin"],
            "https://courtvision.video",
        )
        self.assertEqual(
            response["headers"]["access-control-allow-credentials"], "true"
        )

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

    def test_any_valid_email_can_create_a_cognito_account(self):
        class Cognito:
            def __init__(self):
                self.request = None

            def sign_up(self, **kwargs):
                self.request = kwargs
                return {"UserConfirmed": False}

        cognito = Cognito()
        with patch.object(web_api, "_client", return_value=cognito), patch.object(
            web_api, "_env", return_value="client-id"
        ):
            response = web_api._sign_up(
                {"email": "New.User@example.com", "password": "ten-letters"}
            )

        self.assertEqual(response["statusCode"], 201)
        self.assertEqual(cognito.request["Username"], "new.user@example.com")
        self.assertEqual(cognito.request["ClientId"], "client-id")

    def test_confirmed_cognito_credentials_create_a_secure_session(self):
        class Cognito:
            def initiate_auth(self, **_kwargs):
                return {"AuthenticationResult": {"AccessToken": "token"}}

        with patch.object(web_api, "_client", return_value=Cognito()), patch.object(
            web_api, "_env", return_value="client-id"
        ):
            response = web_api._sign_in(
                {"email": "analyst@example.com", "password": "ten-letters"}
            )

        self.assertEqual(response["statusCode"], 200)
        self.assertIn("HttpOnly", response["cookies"][0])
        self.assertIn("SameSite=Strict", response["cookies"][0])

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

    def test_recent_jobs_are_queried_by_signed_in_owner_newest_first(self):
        now = int(time.time())

        class JobsTable:
            def __init__(self):
                self.query_args = None

            def query(self, **kwargs):
                self.query_args = kwargs
                return {
                    "Items": [
                        {
                            "jobId": "new-job",
                            "ownerEmail": "analyst@example.com",
                            "status": "complete",
                            "filename": "fourth-quarter.mp4",
                            "durationSeconds": 20,
                            "createdAt": now,
                            "updatedAt": now,
                            "expiresAt": now + 3600,
                        }
                    ]
                }

        table = JobsTable()
        with patch.object(web_api, "_table", return_value=table):
            response = web_api._list_jobs({"email": "analyst@example.com"})

        body = json.loads(response["body"])
        self.assertEqual(body["jobs"][0]["id"], "new-job")
        self.assertEqual(table.query_args["IndexName"], "OwnerCreatedAtIndex")
        self.assertEqual(
            table.query_args["ExpressionAttributeValues"][":owner"],
            "analyst@example.com",
        )
        self.assertFalse(table.query_args["ScanIndexForward"])

    def test_recent_jobs_endpoint_requires_a_session(self):
        event = {
            "rawPath": "/api/jobs",
            "requestContext": {"http": {"method": "GET"}},
            "headers": {},
        }

        response = web_api.lambda_handler(event, None)

        self.assertEqual(response["statusCode"], 401)

    def test_admin_overview_rejects_a_signed_in_non_admin(self):
        with patch.dict("os.environ", {"ADMIN_EMAIL": "owner@example.com"}, clear=False):
            with self.assertRaises(web_api.ApiError) as raised:
                web_api._admin_overview({"email": "analyst@example.com"})

        self.assertEqual(raised.exception.code, "admin_required")

    def test_admin_overview_lists_signups_and_analysis_activity(self):
        class Cognito:
            def __init__(self):
                self.calls = []

            def list_users(self, **kwargs):
                self.calls.append(kwargs)
                if "PaginationToken" not in kwargs:
                    return {
                        "Users": [
                            {
                                "Username": "owner-id",
                                "UserStatus": "CONFIRMED",
                                "Enabled": True,
                                "UserCreateDate": datetime(2026, 9, 1, 12, tzinfo=UTC),
                                "Attributes": [
                                    {"Name": "email", "Value": "owner@example.com"},
                                    {"Name": "email_verified", "Value": "true"},
                                ],
                            }
                        ],
                        "PaginationToken": "next-users",
                    }
                return {
                    "Users": [
                        {
                            "Username": "viewer@example.com",
                            "UserStatus": "UNCONFIRMED",
                            "Enabled": True,
                            "UserCreateDate": datetime(2026, 9, 2, 12, tzinfo=UTC),
                            "Attributes": [{"Name": "email", "Value": "viewer@example.com"}],
                        }
                    ]
                }

        class ActivityTable:
            def __init__(self):
                self.calls = []

            def scan(self, **kwargs):
                self.calls.append(kwargs)
                if "ExclusiveStartKey" not in kwargs:
                    return {
                        "Items": [
                            {"ownerEmail": "owner@example.com", "analysisCount": 2, "lastAnalysisAt": 200},
                        ],
                        "LastEvaluatedKey": {"jobId": "second"},
                    }
                return {
                    "Items": [
                        {"ownerEmail": "legacy@example.com", "analysisCount": 1, "lastAnalysisAt": 150}
                    ]
                }

        cognito = Cognito()
        activity = ActivityTable()
        with patch.dict(
            "os.environ",
            {"ADMIN_EMAIL": "owner@example.com", "COGNITO_USER_POOL_ID": "pool-id"},
            clear=False,
        ), patch.object(web_api, "_client", return_value=cognito), patch.object(
            web_api, "_table", return_value=activity
        ):
            response = web_api._admin_overview({"email": "OWNER@example.com"})

        body = json.loads(response["body"])
        self.assertEqual(body["totals"], {"signups": 2, "confirmedUsers": 1, "analysisUsers": 2, "analyses": 3})
        self.assertEqual(body["users"][1]["analysisCount"], 2)
        self.assertEqual(body["users"][1]["lastAnalysisAt"], "1970-01-01T00:03:20Z")
        self.assertEqual(cognito.calls[1]["PaginationToken"], "next-users")
        self.assertEqual(activity.calls[1]["ExclusiveStartKey"], {"jobId": "second"})

    def test_analysis_activity_is_recorded_persistently(self):
        class ActivityTable:
            def __init__(self):
                self.request = None

            def update_item(self, **kwargs):
                self.request = kwargs

        table = ActivityTable()
        with patch.object(web_api, "_table", return_value=table):
            web_api._record_analysis_activity(
                {"ownerEmail": "analyst@example.com", "createdAt": 1_799_000_000}
            )

        self.assertEqual(table.request["Key"], {"ownerEmail": "analyst@example.com"})
        self.assertIn("ADD analysisCount", table.request["UpdateExpression"])
        self.assertEqual(table.request["ExpressionAttributeValues"][":one"], 1)

    def test_session_marks_only_the_configured_email_as_admin(self):
        payload = {"v": 1, "email": "owner@example.com", "csrf": "csrf", "exp": int(time.time()) + 60}
        token = web_api._encode_session(payload)
        event = {
            "rawPath": "/api/auth/session",
            "requestContext": {"http": {"method": "GET"}},
            "headers": {"cookie": f"cv_session={token}"},
        }
        with patch.dict("os.environ", {"ADMIN_EMAIL": "OWNER@example.com"}, clear=False):
            response = web_api.lambda_handler(event, None)

        self.assertTrue(json.loads(response["body"])["isAdmin"])

    def test_uncertain_team_continuation_sets_batch_override(self):
        job = {
            "jobId": "job-id",
            "status": "needs_team_colors",
            "team1Color": "#FFFFFF",
            "team2Color": "#C8102E",
        }
        expected = {"statusCode": 202}

        with patch.object(web_api, "_owned_job", return_value=job), patch.object(
            web_api,
            "_submit_batch",
            return_value=expected,
        ) as submit:
            response = web_api._submit_uncertain_teams(
                {"email": "analyst@example.com"},
                "job-id",
            )

        self.assertEqual(response, expected)
        submitted = submit.call_args.args[0]
        self.assertTrue(submitted["allowUncertainTeams"])
        self.assertNotIn("team1Color", submitted)
        self.assertNotIn("team2Color", submitted)

    def test_failed_invalid_color_job_can_be_resubmitted_with_black(self):
        job = {
            "jobId": "job-id",
            "status": "failed",
            "errorMessage": "Invalid team-color configuration: rejected: #000000",
        }
        expected = {"statusCode": 202}

        with patch.object(web_api, "_owned_job", return_value=job), patch.object(
            web_api,
            "_submit_batch",
            return_value=expected,
        ) as submit:
            response = web_api._submit_team_colors(
                {"email": "analyst@example.com"},
                "job-id",
                {"team1Color": "#000000", "team2Color": "#FFFFFF"},
            )

        self.assertEqual(response, expected)
        submitted = submit.call_args.args[0]
        self.assertEqual(submitted["team1Color"], "#000000")
        self.assertFalse(submitted["allowUncertainTeams"])

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

    def test_start_job_requires_uploaded_object_to_match_declared_job(self):
        job = {
            "jobId": "12345678-1234-1234-1234-123456789abc",
            "status": "awaiting_upload",
            "inputKey": "jobs/12345678-1234-1234-1234-123456789abc/input/source.mp4",
            "sizeBytes": 5,
            "contentType": "video/mp4",
        }

        class S3:
            def head_object(self, **_kwargs):
                return {
                    "ContentLength": 6,
                    "ContentType": "video/mp4",
                    "Metadata": {"job-id": job["jobId"]},
                }

        with patch.object(web_api, "_owned_job", return_value=job), patch.object(
            web_api, "_client", return_value=S3()
        ), patch.object(web_api, "_env", return_value="private-artifacts"), patch.object(
            web_api, "_submit_batch"
        ) as submit:
            with self.assertRaises(web_api.ApiError) as raised:
                web_api._start_job({"email": "analyst@example.com"}, job["jobId"])

        self.assertEqual(raised.exception.code, "invalid_upload")
        submit.assert_not_called()

    def test_start_job_submits_matching_uploaded_object(self):
        job = {
            "jobId": "12345678-1234-1234-1234-123456789abc",
            "status": "awaiting_upload",
            "inputKey": "jobs/12345678-1234-1234-1234-123456789abc/input/source.mp4",
            "sizeBytes": 5,
            "contentType": "video/mp4",
        }

        class S3:
            def head_object(self, **_kwargs):
                return {
                    "ContentLength": 5,
                    "ContentType": "video/mp4",
                    "Metadata": {"job-id": job["jobId"]},
                }

        expected = {"statusCode": 202}
        with patch.object(web_api, "_owned_job", return_value=job), patch.object(
            web_api, "_client", return_value=S3()
        ), patch.object(web_api, "_env", return_value="private-artifacts"), patch.object(
            web_api, "_submit_batch", return_value=expected
        ) as submit:
            response = web_api._start_job(
                {"email": "analyst@example.com"},
                job["jobId"],
            )

        self.assertEqual(response, expected)
        submit.assert_called_once_with(job)

    def test_positive_number_rejects_non_finite_values(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(web_api.ApiError):
                web_api._positive_number(value, "Video duration")


if __name__ == "__main__":
    unittest.main()
