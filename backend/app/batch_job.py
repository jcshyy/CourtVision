"""AWS Batch entrypoint for one bounded CourtVision web job."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import boto3
except ImportError:  # Keeps pure helper tests importable without the AWS runtime extras.
    boto3 = None


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_FILENAMES = (
    "player_detector.pt",
    "yolo11n-pose.pt",
    "ball_detector_model.pt",
    "wasb_basketball_torchscript.pt",
    "court_keypoint_detector.pt",
)
BALL_DETECTOR_BACKENDS = {"yolo", "wasb", "hybrid"}


def main():
    if boto3 is None:
        raise RuntimeError("boto3 is required to run the AWS Batch worker")
    job_id = _env("COURTVISION_JOB_ID")
    input_key = _env("COURTVISION_INPUT_KEY")
    bucket = _env("COURTVISION_ARTIFACT_BUCKET")
    jobs_table = boto3.resource("dynamodb").Table(_env("COURTVISION_JOBS_TABLE"))
    s3 = boto3.client("s3")
    retention = int(os.getenv("COURTVISION_RETENTION_SECONDS", "86400"))
    expires_at = int(time.time()) + retention

    try:
        _update_job(jobs_table, job_id, "processing", "Preparing the uploaded clip")
        _prepare_models(s3)
        with tempfile.TemporaryDirectory(prefix=f"courtvision-{job_id[:8]}-") as temp:
            work = Path(temp)
            source_path = work / ("source" + Path(input_key).suffix.lower())
            output_path = work / "annotated.mp4"
            analysis_path = work / "analysis.json"
            cache_path = work / "cache"
            s3.download_file(bucket, input_key, str(source_path))

            _update_job(jobs_table, job_id, "processing", "Analyzing players, ball, and court")
            command = _analysis_command(
                source_path,
                output_path,
                analysis_path,
                cache_path,
            )
            team_one = os.getenv("COURTVISION_TEAM_1_COLOR")
            team_two = os.getenv("COURTVISION_TEAM_2_COLOR")
            if team_one and team_two:
                command.extend(["--team-1-color", team_one, "--team-2-color", team_two])
            if os.getenv("COURTVISION_ALLOW_UNCERTAIN_TEAMS", "").lower() in {
                "1",
                "true",
                "yes",
            }:
                command.append("--allow-uncertain-teams")

            process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            output_lines = []
            assert process.stdout is not None
            for line in process.stdout:
                clean = line.rstrip()
                output_lines.append(clean)
                LOGGER.info("pipeline: %s", clean)
            return_code = process.wait()

            if return_code == 2:
                result = _last_json_object(output_lines)
                if result and result.get("status") == "needs_team_colors":
                    _update_needs_colors(jobs_table, job_id, result, expires_at)
                    return 0
            if return_code != 0:
                detail = next((line for line in reversed(output_lines) if line), "Pipeline failed")
                raise RuntimeError(detail[:1000])
            if not output_path.exists() or not analysis_path.exists():
                raise RuntimeError("Pipeline completed without the required review artifacts")

            _update_job(jobs_table, job_id, "processing", "Finalizing review artifacts")
            output_key = f"jobs/{job_id}/result/annotated.mp4"
            analysis_key = f"jobs/{job_id}/result/analysis.json"
            s3.upload_file(
                str(output_path),
                bucket,
                output_key,
                ExtraArgs={"ContentType": "video/mp4", "Metadata": {"job-id": job_id}},
            )
            s3.upload_file(
                str(analysis_path),
                bucket,
                analysis_key,
                ExtraArgs={"ContentType": "application/json", "Metadata": {"job-id": job_id}},
            )
            now = int(time.time())
            jobs_table.update_item(
                Key={"jobId": job_id},
                UpdateExpression=(
                    "SET #status = :status, stage = :stage, outputVideoKey = :video, "
                    "analysisKey = :analysis, updatedAt = :updated, expiresAt = :expires "
                    "REMOVE errorMessage, teamColorReason"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":status": "complete",
                    ":stage": "Review ready",
                    ":video": output_key,
                    ":analysis": analysis_key,
                    ":updated": now,
                    ":expires": expires_at,
                },
            )
            LOGGER.info("CourtVision job %s completed", job_id)
            return 0
    except Exception as error:
        LOGGER.exception("CourtVision job %s failed", job_id)
        try:
            jobs_table.update_item(
                Key={"jobId": job_id},
                UpdateExpression=(
                    "SET #status = :status, stage = :stage, errorMessage = :error, "
                    "updatedAt = :updated, expiresAt = :expires"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":status": "failed",
                    ":stage": "Analysis failed",
                    ":error": str(error)[:1000],
                    ":updated": int(time.time()),
                    ":expires": expires_at,
                },
            )
        except Exception:
            LOGGER.exception("Could not record failed state for %s", job_id)
        return 1


def _prepare_models(s3, models_dir=None):
    """Materialize private model weights before importing the CLI pipeline."""
    bucket = os.getenv("COURTVISION_MODEL_BUCKET")
    if not bucket:
        return

    project_root = Path(__file__).resolve().parents[2]
    models_dir = Path(models_dir) if models_dir else project_root / "backend" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    prefix = os.getenv("COURTVISION_MODEL_PREFIX", "models").strip("/")

    for filename in MODEL_FILENAMES:
        destination = models_dir / filename
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        key = f"{prefix}/{filename}" if prefix else filename
        partial = destination.with_suffix(destination.suffix + ".part")
        LOGGER.info("Downloading model s3://%s/%s", bucket, key)
        try:
            s3.download_file(bucket, key, str(partial))
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)


def _analysis_command(source_path, output_path, analysis_path, cache_path):
    """Build the bounded worker command from validated deployment settings."""
    detector_backend = os.getenv(
        "COURTVISION_BALL_DETECTOR_BACKEND",
        "hybrid",
    ).strip().lower()
    if detector_backend not in BALL_DETECTOR_BACKENDS:
        raise RuntimeError(
            "COURTVISION_BALL_DETECTOR_BACKEND must be one of "
            f"{sorted(BALL_DETECTOR_BACKENDS)}"
        )
    return [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "main.py"),
        str(source_path),
        "--output-video",
        str(output_path),
        "--output-analysis",
        str(analysis_path),
        "--stub-path",
        str(cache_path),
        "--duration-seconds",
        os.getenv("COURTVISION_MAX_DURATION_SECONDS", "30"),
        "--target-fps",
        os.getenv("COURTVISION_TARGET_FPS", "30"),
        "--max-width",
        os.getenv("COURTVISION_MAX_WIDTH", "1280"),
        "--ball-detector-backend",
        detector_backend,
    ]


def _update_job(table, job_id, status, stage):
    table.update_item(
        Key={"jobId": job_id},
        UpdateExpression="SET #status = :status, stage = :stage, updatedAt = :updated",
        ExpressionAttributeNames={"#status": "status"},
        ExpressionAttributeValues={
            ":status": status,
            ":stage": stage,
            ":updated": int(time.time()),
        },
    )


def _update_needs_colors(table, job_id, result, expires_at):
    reason = result.get("reason") or result.get("message") or "Automatic jersey discovery was uncertain."
    table.update_item(
        Key={"jobId": job_id},
        UpdateExpression=(
            "SET #status = :status, stage = :stage, teamColorReason = :reason, "
            "updatedAt = :updated, expiresAt = :expires"
        ),
        ExpressionAttributeNames={"#status": "status"},
        ExpressionAttributeValues={
            ":status": "needs_team_colors",
            ":stage": "Team colors required",
            ":reason": str(reason)[:1000],
            ":updated": int(time.time()),
            ":expires": expires_at,
        },
    )


def _last_json_object(lines):
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
