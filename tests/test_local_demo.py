import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from backend.app.local_demo import LOCAL_CSRF, create_app


class LocalDemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runner_calls = []

        def runner(job, source, output, analysis, cache, update_stage):
            self.runner_calls.append(dict(job))
            self.assertTrue(source.is_file())
            cache.mkdir(parents=True, exist_ok=True)
            update_stage("Rendering the review video")
            output.write_bytes(b"local annotated video")
            analysis.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "beta": True,
                        "source": {"fps": 15, "frameCount": 15, "durationSeconds": 1},
                        "court": {"width": 300, "height": 161},
                        "events": [],
                        "frames": [],
                        "diagnostics": {"tacticalView": {}},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess([], 0, "complete")

        app = create_app(
            data_root=self.temp.name,
            pipeline_runner=runner,
            run_jobs_inline=True,
        )
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def csrf_headers():
        return {"X-CourtVision-CSRF": LOCAL_CSRF}

    def create_job(self, *, payload=b"video"):
        response = self.client.post(
            "/api/jobs",
            headers=self.csrf_headers(),
            json={
                "filename": "clip.mp4",
                "contentType": "video/mp4",
                "sizeBytes": len(payload),
                "durationSeconds": 1,
            },
        )
        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        uploaded = self.client.post(
            body["upload"]["url"],
            data={"file": (io.BytesIO(payload), "clip.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(uploaded.status_code, 204)
        return body["job"]["id"]

    def test_local_session_skips_cloud_authentication(self):
        response = self.client.get("/api/auth/session")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["authenticated"])
        self.assertTrue(response.get_json()["localRuntime"])
        self.assertEqual(response.get_json()["csrfToken"], LOCAL_CSRF)

    def test_job_mutations_require_local_csrf(self):
        response = self.client.post(
            "/api/jobs",
            json={
                "filename": "clip.mp4",
                "contentType": "video/mp4",
                "sizeBytes": 5,
                "durationSeconds": 1,
            },
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json()["error"]["code"], "invalid_request_token")

    def test_upload_start_poll_and_download_complete_locally(self):
        job_id = self.create_job()

        started = self.client.post(
            f"/api/jobs/{job_id}/start",
            headers=self.csrf_headers(),
            json={},
        )
        polled = self.client.get(f"/api/jobs/{job_id}")
        downloads = self.client.get(f"/api/jobs/{job_id}/download")
        analysis = self.client.get(downloads.get_json()["analysisUrl"])
        video = self.client.get(downloads.get_json()["playbackUrl"])

        self.assertEqual(started.status_code, 202)
        self.assertEqual(polled.get_json()["job"]["status"], "complete")
        self.assertEqual(downloads.status_code, 200)
        self.assertEqual(analysis.get_json()["schemaVersion"], 1)
        self.assertEqual(video.data, b"local annotated video")
        self.assertEqual(len(self.runner_calls), 1)
        analysis.close()
        video.close()

    def test_team_color_retry_uses_the_same_uploaded_job(self):
        attempts = []

        def runner(job, _source, output, analysis, _cache, _update_stage):
            attempts.append(dict(job))
            if not job.get("team1Color"):
                return subprocess.CompletedProcess(
                    [],
                    2,
                    '{"status":"needs_team_colors","reason":"jerseys overlap"}',
                )
            output.write_bytes(b"video")
            analysis.write_text(
                '{"schemaVersion":1,"source":{"durationSeconds":1},"events":[],"frames":[]}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess([], 0, "complete")

        app = create_app(
            data_root=Path(self.temp.name) / "colors",
            pipeline_runner=runner,
            run_jobs_inline=True,
        )
        app.config.update(TESTING=True)
        client = app.test_client()
        created = client.post(
            "/api/jobs",
            headers=self.csrf_headers(),
            json={
                "filename": "clip.mp4",
                "contentType": "video/mp4",
                "sizeBytes": 5,
                "durationSeconds": 1,
            },
        ).get_json()
        job_id = created["job"]["id"]
        client.post(
            created["upload"]["url"],
            data={"file": (io.BytesIO(b"video"), "clip.mp4")},
            content_type="multipart/form-data",
        )
        client.post(f"/api/jobs/{job_id}/start", headers=self.csrf_headers(), json={})

        needs_colors = client.get(f"/api/jobs/{job_id}").get_json()["job"]
        retried = client.post(
            f"/api/jobs/{job_id}/team-colors",
            headers=self.csrf_headers(),
            json={"team1Color": "#FFFFFF", "team2Color": "#C8102E"},
        )
        completed = client.get(f"/api/jobs/{job_id}").get_json()["job"]

        self.assertEqual(needs_colors["status"], "needs_team_colors")
        self.assertEqual(needs_colors["teamColorReason"], "jerseys overlap")
        self.assertEqual(retried.status_code, 202)
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(attempts[-1]["team1Color"], "#FFFFFF")

    def test_uncertain_team_continuation_retries_the_same_uploaded_job(self):
        attempts = []

        def runner(job, _source, output, analysis, _cache, _update_stage):
            attempts.append(dict(job))
            if not job.get("allowUncertainTeams"):
                return subprocess.CompletedProcess(
                    [],
                    2,
                    '{"status":"needs_team_colors","reason":"too many unknown players"}',
                )
            output.write_bytes(b"video")
            analysis.write_text(
                '{"schemaVersion":1,"source":{"durationSeconds":1},"events":[],"frames":[]}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess([], 0, "complete")

        app = create_app(
            data_root=Path(self.temp.name) / "uncertain",
            pipeline_runner=runner,
            run_jobs_inline=True,
        )
        app.config.update(TESTING=True)
        client = app.test_client()
        created = client.post(
            "/api/jobs",
            headers=self.csrf_headers(),
            json={
                "filename": "clip.mp4",
                "contentType": "video/mp4",
                "sizeBytes": 5,
                "durationSeconds": 1,
            },
        ).get_json()
        job_id = created["job"]["id"]
        client.post(
            created["upload"]["url"],
            data={"file": (io.BytesIO(b"video"), "clip.mp4")},
            content_type="multipart/form-data",
        )
        client.post(f"/api/jobs/{job_id}/start", headers=self.csrf_headers(), json={})

        continued = client.post(
            f"/api/jobs/{job_id}/continue-with-uncertain-teams",
            headers=self.csrf_headers(),
            json={},
        )
        completed = client.get(f"/api/jobs/{job_id}").get_json()["job"]

        self.assertEqual(continued.status_code, 202)
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(len(attempts), 2)
        self.assertTrue(attempts[-1]["allowUncertainTeams"])
        self.assertIsNone(attempts[-1]["team1Color"])

    def test_server_supplies_local_config_and_application(self):
        config = self.client.get("/config.js")
        application = self.client.get("/")

        self.assertEqual(config.status_code, 200)
        self.assertIn(b'"localRuntime": true', config.data)
        self.assertEqual(application.status_code, 200)
        self.assertIn(b'<div id="app"', application.data)
        config.close()
        application.close()


if __name__ == "__main__":
    unittest.main()
