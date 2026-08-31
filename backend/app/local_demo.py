"""Local upload-to-review demo using the production browser API contract.

This server is intentionally bound to loopback by default. It replaces AWS
storage, authentication, and Batch submission with a filesystem job store and
one bounded background worker while preserving the browser-facing workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

from backend.app.batch_job import _last_json_object
from backend.app.web_api import (
    ALLOWED_VIDEO_TYPES,
    HEX_COLOR_PATTERN,
    REPORT_CATEGORIES,
    ApiError,
    _extension_for,
    _format_bytes,
    _can_change_team_colors,
    _iso,
    _positive_int,
    _positive_number,
    _safe_filename,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"
JOB_ID_PATTERN = re.compile(r"^[0-9a-f-]{36}$")
LOCAL_EMAIL = "local@courtvision.dev"
LOCAL_CSRF = "courtvision-local-demo"


class LocalJobStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.jobs_root = self.root / "jobs"
        self.reports_path = self.root / "reports.jsonl"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create(self, body, *, max_bytes, max_duration, retention_seconds):
        filename = _safe_filename(body.get("filename"))
        content_type = str(body.get("contentType", "")).lower()
        size_bytes = _positive_int(body.get("sizeBytes"), "Video size")
        duration = _positive_number(body.get("durationSeconds"), "Video duration")
        if content_type not in ALLOWED_VIDEO_TYPES:
            raise ApiError(400, "Use an MP4, MOV, or WebM video.", code="unsupported_video")
        if size_bytes > max_bytes:
            raise ApiError(
                413,
                f"This video exceeds the {_format_bytes(max_bytes)} local limit.",
                code="video_too_large",
            )
        if duration > max_duration + 0.05:
            raise ApiError(
                400,
                f"Choose a clip no longer than {max_duration:g} seconds.",
                code="video_too_long",
            )

        job_id = str(uuid.uuid4())
        now = int(time.time())
        job = {
            "jobId": job_id,
            "ownerEmail": LOCAL_EMAIL,
            "status": "awaiting_upload",
            "stage": "Waiting for local upload",
            "filename": filename,
            "contentType": content_type,
            "sizeBytes": size_bytes,
            "durationSeconds": round(duration, 3),
            "sourceFilename": f"source{_extension_for(filename, content_type)}",
            "createdAt": now,
            "updatedAt": now,
            "expiresAt": now + retention_seconds,
        }
        self.job_dir(job_id).mkdir(parents=True, exist_ok=False)
        self.write(job)
        return job

    def job_dir(self, job_id):
        self._validate_id(job_id)
        return self.jobs_root / job_id

    def read(self, job_id):
        path = self.job_dir(job_id) / "job.json"
        with self._lock:
            if not path.is_file():
                raise ApiError(404, "This local analysis is unavailable.", code="job_not_found")
            job = json.loads(path.read_text(encoding="utf-8"))
        if int(job.get("expiresAt", 0)) <= int(time.time()):
            raise ApiError(404, "This local analysis has expired.", code="job_not_found")
        return job

    def write(self, job):
        path = self.job_dir(job["jobId"]) / "job.json"
        temporary = path.with_suffix(".json.tmp")
        with self._lock:
            temporary.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)

    def update(self, job_id, **changes):
        with self._lock:
            job = self.read(job_id)
            job.update(changes)
            job["updatedAt"] = int(time.time())
            self.write(job)
            return job

    def append_report(self, report):
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock, self.reports_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(report, separators=(",", ":")) + "\n")

    @staticmethod
    def _validate_id(job_id):
        if not JOB_ID_PATTERN.fullmatch(str(job_id)):
            raise ApiError(404, "This local analysis is unavailable.", code="job_not_found")


def create_app(*, data_root=None, pipeline_runner=None, run_jobs_inline=False):
    data_root = data_root or os.getenv(
        "COURTVISION_LOCAL_DATA",
        str(PROJECT_ROOT / "runs" / "local_demo"),
    )
    store = LocalJobStore(data_root)
    runner = pipeline_runner or _run_pipeline
    executor = None if run_jobs_inline else ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="courtvision-local",
    )

    app = Flask(__name__, static_folder=None)
    app.config.update(
        MAX_CONTENT_LENGTH=_env_int("MAX_UPLOAD_BYTES", 500 * 1024 * 1024) + 1024 * 1024,
        COURTVISION_STORE=store,
        COURTVISION_EXECUTOR=executor,
    )

    def api_response(body=None, status=200):
        response = jsonify(body) if body is not None else Response(status=status)
        response.status_code = status
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def require_csrf():
        if request.headers.get("X-CourtVision-CSRF") != LOCAL_CSRF:
            raise ApiError(403, "Refresh the page and try again.", code="invalid_request_token")

    def submit(job_id):
        job = store.read(job_id)
        if job["status"] not in {"awaiting_upload", "needs_team_colors", "failed"}:
            return job
        source = store.job_dir(job_id) / job["sourceFilename"]
        if not source.is_file() or source.stat().st_size <= 0:
            raise ApiError(409, "The local upload has not finished.", code="upload_incomplete")
        job = store.update(job_id, status="queued", stage="Queued on this computer", errorMessage=None)
        if run_jobs_inline:
            _execute_job(store, job_id, runner)
        else:
            executor.submit(_execute_job, store, job_id, runner)
        return job

    @app.get("/health")
    def health():
        return api_response({"status": "healthy", "service": "courtvision-local-demo"})

    @app.get("/api/auth/session")
    def session():
        return api_response(
            {
                "authenticated": True,
                "email": LOCAL_EMAIL,
                "csrfToken": LOCAL_CSRF,
                "expiresAt": _iso(int(time.time()) + 8 * 60 * 60),
                "localRuntime": True,
            }
        )

    @app.post("/api/auth/sign-out")
    def sign_out():
        require_csrf()
        return api_response({"signedOut": True})

    @app.post("/api/jobs")
    def create_job():
        require_csrf()
        job = store.create(
            request.get_json(silent=False) or {},
            max_bytes=_env_int("MAX_UPLOAD_BYTES", 500 * 1024 * 1024),
            max_duration=_env_float("MAX_DURATION_SECONDS", 30),
            retention_seconds=_env_int("RESULT_RETENTION_SECONDS", 24 * 60 * 60),
        )
        return api_response(
            {
                "job": _public_job(job),
                "upload": {"url": f"/api/jobs/{job['jobId']}/upload", "fields": {}},
            },
            201,
        )

    @app.post("/api/jobs/<job_id>/upload")
    def upload_job(job_id):
        job = store.read(job_id)
        if job["status"] != "awaiting_upload":
            raise ApiError(409, "This local upload is already complete.", code="upload_complete")
        upload = request.files.get("file")
        if upload is None:
            raise ApiError(400, "Choose a video file.", code="missing_upload")
        destination = store.job_dir(job_id) / job["sourceFilename"]
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            upload.save(partial)
            actual_size = partial.stat().st_size
            if actual_size <= 0 or actual_size != int(job["sizeBytes"]):
                raise ApiError(400, "The uploaded file size does not match the selected clip.", code="invalid_upload")
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        store.update(job_id, stage="Local upload received", uploadedBytes=actual_size)
        return Response(status=204)

    @app.post("/api/jobs/<job_id>/start")
    def start_job(job_id):
        require_csrf()
        return api_response({"job": _public_job(submit(job_id))}, 202)

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id):
        return api_response({"job": _public_job(store.read(job_id))})

    @app.post("/api/jobs/<job_id>/team-colors")
    def team_colors(job_id):
        require_csrf()
        job = store.read(job_id)
        if not _can_change_team_colors(job):
            raise ApiError(409, "Team colors are not required for this job.", code="colors_not_required")
        body = request.get_json(silent=False) or {}
        first = str(body.get("team1Color", "")).upper()
        second = str(body.get("team2Color", "")).upper()
        if not HEX_COLOR_PATTERN.fullmatch(first) or not HEX_COLOR_PATTERN.fullmatch(second):
            raise ApiError(400, "Choose two six-digit jersey colors.", code="invalid_team_colors")
        if first == second:
            raise ApiError(400, "Choose two distinct jersey colors.", code="invalid_team_colors")
        store.update(job_id, team1Color=first, team2Color=second)
        return api_response({"job": _public_job(submit(job_id))}, 202)

    @app.get("/api/jobs/<job_id>/download")
    def downloads(job_id):
        job = store.read(job_id)
        if job["status"] != "complete":
            raise ApiError(409, "Results are not ready to download.", code="result_not_ready")
        base = f"/local-artifacts/{job_id}"
        return api_response(
            {
                "videoUrl": f"{base}/annotated.mp4?download=1",
                "playbackUrl": f"{base}/annotated.mp4",
                "analysisUrl": f"{base}/analysis.json",
                "expiresInSeconds": max(0, int(job["expiresAt"]) - int(time.time())),
            }
        )

    @app.post("/api/jobs/<job_id>/reports")
    def create_report(job_id):
        require_csrf()
        job = store.read(job_id)
        body = request.get_json(silent=False) or {}
        category = str(body.get("category", "")).strip()
        notes = str(body.get("notes", "")).strip()
        if category not in REPORT_CATEGORIES:
            raise ApiError(400, "Choose a failure category.", code="invalid_report")
        if not notes or len(notes) > 2000:
            raise ApiError(400, "Describe the failure in 2,000 characters or fewer.", code="invalid_report")
        timestamp = float(body.get("timeSeconds", 0))
        if timestamp < 0 or timestamp > float(job["durationSeconds"]) + 0.1:
            raise ApiError(400, "The report timestamp is outside this clip.", code="invalid_report")
        report_id = str(uuid.uuid4())
        store.append_report(
            {
                "reportId": report_id,
                "jobId": job_id,
                "category": category,
                "notes": notes,
                "timeSeconds": round(timestamp, 3),
                "eventId": str(body.get("eventId") or ""),
                "createdAt": int(time.time()),
            }
        )
        return api_response({"reportId": report_id, "message": "Failure report saved locally."}, 201)

    @app.get("/local-artifacts/<job_id>/<artifact>")
    def artifact(job_id, artifact):
        job = store.read(job_id)
        if job["status"] != "complete" or artifact not in {"annotated.mp4", "analysis.json"}:
            raise ApiError(404, "This local artifact is unavailable.", code="artifact_not_found")
        path = store.job_dir(job_id) / artifact
        if not path.is_file():
            raise ApiError(404, "This local artifact is unavailable.", code="artifact_not_found")
        response = send_file(
            path,
            as_attachment=request.args.get("download") == "1",
            download_name=f"courtvision-{job_id[:8]}.mp4" if artifact.endswith(".mp4") else artifact,
            conditional=True,
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/config.js")
    def local_config():
        config = {
            "apiBaseUrl": "/api",
            "authConnected": True,
            "localRuntime": True,
            "maxDurationSeconds": _env_float("MAX_DURATION_SECONDS", 30),
            "maxUploadBytes": _env_int("MAX_UPLOAD_BYTES", 500 * 1024 * 1024),
            "targetFps": _env_float("TARGET_FPS", 30),
            "maxWidth": _env_int("MAX_WIDTH", 1280),
            "resultRetentionHours": _env_int("RESULT_RETENTION_SECONDS", 86400) / 3600,
            "pollIntervalMs": 1000,
        }
        return Response(
            f"window.COURTVISION_CONFIG = Object.freeze({json.dumps(config)});\n",
            content_type="application/javascript; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/")
    def index():
        return send_from_directory(WEB_ROOT, "app.html")

    @app.get("/<path:asset_path>")
    def static_asset(asset_path):
        return send_from_directory(WEB_ROOT, asset_path)

    @app.errorhandler(ApiError)
    def api_error(error):
        return api_response({"error": {"code": error.code, "message": error.message}}, error.status_code)

    @app.errorhandler(413)
    def upload_too_large(_error):
        return api_response(
            {"error": {"code": "video_too_large", "message": "This upload exceeds the local size limit."}},
            413,
        )

    @app.errorhandler(HTTPException)
    def http_error(error):
        return api_response(
            {"error": {"code": "not_found", "message": "The requested local resource does not exist."}},
            error.code or 500,
        )

    @app.errorhandler(Exception)
    def unexpected_error(error):
        LOGGER.exception("Unhandled local demo error", exc_info=error)
        return api_response(
            {"error": {"code": "internal_error", "message": "The local demo could not complete the request."}},
            500,
        )

    return app


def _execute_job(store, job_id, runner):
    try:
        job = store.update(job_id, status="processing", stage="Preparing the local analysis")
        directory = store.job_dir(job_id)
        source = directory / job["sourceFilename"]
        output = directory / "annotated.mp4"
        analysis = directory / "analysis.json"
        cache = Path(
            os.getenv(
                "COURTVISION_LOCAL_CACHE",
                str(PROJECT_ROOT / "backend" / "stubs"),
            )
        )

        def update_stage(stage):
            store.update(job_id, status="processing", stage=stage)

        result = runner(job, source, output, analysis, cache, update_stage)
        lines = str(result.stdout or "").splitlines()
        if result.returncode == 2:
            detail = _last_json_object(lines) or {}
            store.update(
                job_id,
                status="needs_team_colors",
                stage="Team colors required",
                teamColorReason=str(
                    detail.get("reason")
                    or detail.get("message")
                    or "Automatic jersey discovery was uncertain."
                )[:1000],
            )
            return
        if result.returncode != 0:
            detail = next((line for line in reversed(lines) if line.strip()), "The local pipeline failed.")
            raise RuntimeError(detail[:1000])
        if not output.is_file() or not analysis.is_file():
            raise RuntimeError("The pipeline completed without both review artifacts.")
        store.update(
            job_id,
            status="complete",
            stage="Review ready",
            outputVideo="annotated.mp4",
            analysis="analysis.json",
            errorMessage=None,
        )
    except Exception as error:
        LOGGER.exception("Local CourtVision job %s failed", job_id)
        store.update(
            job_id,
            status="failed",
            stage="Analysis failed",
            errorMessage=str(error)[:1000],
        )


def _run_pipeline(job, source, output, analysis, cache, update_stage):
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        str(source),
        "--output-video",
        str(output),
        "--output-analysis",
        str(analysis),
        "--stub-path",
        str(cache),
        "--duration-seconds",
        str(job["durationSeconds"]),
        "--target-fps",
        str(_env_float("TARGET_FPS", 30)),
        "--max-width",
        str(_env_int("MAX_WIDTH", 1280)),
    ]
    if job.get("team1Color") and job.get("team2Color"):
        command.extend(["--team-1-color", job["team1Color"], "--team-2-color", job["team2Color"]])

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    timed_out = threading.Event()

    def stop_overdue_process():
        timed_out.set()
        process.kill()

    timeout = _env_int("COURTVISION_LOCAL_TIMEOUT_SECONDS", 90 * 60)
    timer = threading.Timer(timeout, stop_overdue_process)
    timer.daemon = True
    timer.start()
    assert process.stdout is not None
    try:
        for line in process.stdout:
            clean = line.rstrip()
            output_lines.append(clean)
            LOGGER.info("local pipeline: %s", clean)
            if clean.startswith("Loaded "):
                update_stage("Analyzing players, ball, and court")
            elif clean.startswith("Detected events:"):
                update_stage("Rendering the review video")
        return_code = process.wait()
    finally:
        timer.cancel()
    if timed_out.is_set():
        output_lines.append(f"Local analysis exceeded the {timeout}-second safety timeout.")
        return_code = 124
    return subprocess.CompletedProcess(command, return_code, "\n".join(output_lines))


def _public_job(job):
    return {
        "id": job["jobId"],
        "status": job["status"],
        "stage": job.get("stage", ""),
        "filename": job.get("filename", ""),
        "durationSeconds": float(job.get("durationSeconds", 0)),
        "createdAt": _iso(int(job.get("createdAt", 0))),
        "updatedAt": _iso(int(job.get("updatedAt", job.get("createdAt", 0)))),
        "expiresAt": _iso(int(job["expiresAt"])),
        "errorMessage": job.get("errorMessage"),
        "teamColorReason": job.get("teamColorReason"),
        "canChangeTeamColors": _can_change_team_colors(job),
    }


def _env_int(name, default):
    return int(os.getenv(name, str(default)))


def _env_float(name, default):
    return float(os.getenv(name, str(default)))


def main():
    parser = argparse.ArgumentParser(description="Run the local CourtVision upload demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    app = create_app(data_root=args.data_dir)
    url = f"http://{args.host}:{args.port}/"
    print(f"CourtVision local demo: {url}")
    print("Uploads and results stay on this computer under runs/local_demo.")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
