import json
import time
import unittest

from backend.app import web_api
from backend.app.flask_api import create_app


class FlaskApiTests(unittest.TestCase):
    def setUp(self):
        web_api._SESSION_SECRET = b"test-session-secret-with-sufficient-length"
        app = create_app()
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def tearDown(self):
        web_api._SESSION_SECRET = None

    @staticmethod
    def _session_token(csrf="csrf-value"):
        return web_api._encode_session(
            {
                "v": 1,
                "email": "analyst@example.com",
                "csrf": csrf,
                "exp": int(time.time()) + 60,
            }
        )

    def test_health_check_does_not_require_authentication(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {"service": "courtvision-api", "status": "healthy"},
        )

    def test_session_contract_matches_lambda_adapter(self):
        token = self._session_token()
        self.client.set_cookie("cv_session", token)

        flask_response = self.client.get("/api/auth/session")
        lambda_response = web_api.lambda_handler(
            {
                "rawPath": "/api/auth/session",
                "requestContext": {"http": {"method": "GET"}},
                "headers": {"cookie": f"cv_session={token}"},
            },
            None,
        )

        self.assertEqual(flask_response.status_code, lambda_response["statusCode"])
        self.assertEqual(flask_response.get_json(), json.loads(lambda_response["body"]))
        self.assertEqual(flask_response.headers["Cache-Control"], "no-store")
        self.assertEqual(flask_response.headers["X-Content-Type-Options"], "nosniff")

    def test_state_change_requires_matching_csrf(self):
        token = self._session_token(csrf="expected-token")
        self.client.set_cookie("cv_session", token)

        rejected = self.client.post("/api/auth/sign-out", json={})
        accepted = self.client.post(
            "/api/auth/sign-out",
            json={},
            headers={"X-CourtVision-CSRF": "expected-token"},
        )

        self.assertEqual(rejected.status_code, 403)
        self.assertEqual(rejected.get_json()["error"]["code"], "invalid_request_token")
        self.assertEqual(accepted.status_code, 200)
        self.assertIn("Max-Age=0", accepted.headers["Set-Cookie"])

    def test_invalid_json_uses_shared_error_contract(self):
        response = self.client.post(
            "/api/auth/sign-up",
            data="{",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["code"], "invalid_json")

    def test_options_does_not_require_authentication(self):
        response = self.client.options("/api/jobs")

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.headers["Cache-Control"], "no-store")


if __name__ == "__main__":
    unittest.main()
