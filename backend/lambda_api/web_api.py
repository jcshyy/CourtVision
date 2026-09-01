"""AWS Lambda HTTP API for CourtVision.

Users create an email-and-password account in Amazon Cognito and confirm their
email before uploads or analysis. Uploads go directly to S3, and GPU analysis
is submitted to an existing AWS Batch queue. All limits are environment
configuration.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

try:  # Lambda includes boto3; keeping import optional makes pure helpers testable.
    import boto3
except ImportError:  # pragma: no cover - exercised only outside AWS/dev extras.
    boto3 = None


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
HEX_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm"}
REPORT_CATEGORIES = {
    "ball_tracking",
    "player_tracking",
    "team_assignment",
    "possession",
    "event_detection",
    "tactical_view",
    "rendering",
    "processing",
    "other",
}

_CLIENTS = {}
_SESSION_SECRET = None


class ApiError(Exception):
    def __init__(self, status_code, message, *, code="request_error"):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code


@dataclass(frozen=True)
class ApiRequest:
    """Framework-neutral request consumed by the CourtVision API dispatcher."""

    method: str
    path: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: str | None = None
    cookies: tuple[str, ...] = ()
    is_base64_encoded: bool = False


def lambda_handler(event, _context):
    return handle_request(_request_from_lambda_event(event))


def handle_request(request):
    """Dispatch one normalized API request for Lambda, Flask, or tests."""

    try:
        method = request.method.upper()
        path = request.path or "/"
        if method == "OPTIONS":
            return _response(204, None)

        if method == "POST" and path == "/auth/sign-up":
            return _sign_up(_json_body(request))
        if method == "POST" and path == "/auth/confirm-sign-up":
            return _confirm_sign_up(_json_body(request))
        if method == "POST" and path == "/auth/resend-confirmation":
            return _resend_confirmation(_json_body(request))
        if method == "POST" and path == "/auth/sign-in":
            return _sign_in(_json_body(request))
        if method == "POST" and path == "/auth/forgot-password":
            return _forgot_password(_json_body(request))
        if method == "POST" and path == "/auth/confirm-password":
            return _confirm_password(_json_body(request))

        session = _require_session(request)
        if method not in {"GET", "HEAD", "OPTIONS"}:
            _require_csrf(request, session)

        if method == "GET" and path == "/auth/session":
            return _response(
                200,
                {
                    "authenticated": True,
                    "email": session["email"],
                    "csrfToken": session["csrf"],
                    "expiresAt": _iso(session["exp"]),
                },
            )
        if method == "POST" and path == "/auth/sign-out":
            return _response(200, {"signedOut": True}, cookies=[_expired_cookie()])
        if method == "GET" and path == "/jobs":
            return _list_jobs(session)
        if method == "POST" and path == "/jobs":
            return _create_job(session, _json_body(request))

        job_match = re.fullmatch(
            r"/jobs/([0-9a-f-]+)(?:/(start|download|reports|team-colors|continue-with-uncertain-teams))?",
            path,
        )
        if job_match:
            job_id, action = job_match.groups()
            if method == "GET" and action is None:
                return _get_job(session, job_id)
            if method == "POST" and action == "start":
                return _start_job(session, job_id)
            if method == "GET" and action == "download":
                return _get_downloads(session, job_id)
            if method == "POST" and action == "reports":
                return _create_report(session, job_id, _json_body(request))
            if method == "POST" and action == "team-colors":
                return _submit_team_colors(session, job_id, _json_body(request))
            if method == "POST" and action == "continue-with-uncertain-teams":
                return _submit_uncertain_teams(session, job_id)

        raise ApiError(404, "The requested endpoint does not exist.", code="not_found")
    except ApiError as error:
        return _response(
            error.status_code,
            {"error": {"code": error.code, "message": error.message}},
        )
    except Exception:
        LOGGER.exception("Unhandled CourtVision API error")
        return _response(
            500,
            {
                "error": {
                    "code": "internal_error",
                    "message": "CourtVision could not complete the request. Try again.",
                }
            },
        )


def _request_from_lambda_event(event):
    method, path = _request_target(event)
    return ApiRequest(
        method=method,
        path=path,
        headers=event.get("headers") or {},
        body=event.get("body"),
        cookies=tuple(event.get("cookies") or ()),
        is_base64_encoded=bool(event.get("isBase64Encoded")),
    )


def _sign_up(body):
    email = _normalize_email(body.get("email"))
    password = _password(body.get("password"))
    try:
        result = _client("cognito-idp").sign_up(
            ClientId=_env("COGNITO_APP_CLIENT_ID"),
            Username=email,
            Password=password,
            UserAttributes=[{"Name": "email", "Value": email}],
        )
    except Exception as error:
        if _aws_error_code(error) == "LimitExceededException":
            return _response(
                201,
                {
                    "email": email,
                    "confirmationRequired": True,
                    "message": (
                        "The account was created, but Cognito could not send the confirmation "
                        "email yet. Wait a minute, then select Send a new code."
                    ),
                },
            )
        _raise_cognito_error(error, operation="sign_up")
    return _response(
        201,
        {
            "email": email,
            "confirmationRequired": not bool(result.get("UserConfirmed")),
            "message": "Account created. Enter the confirmation code sent to your email.",
        },
    )


def _confirm_sign_up(body):
    email = _normalize_email(body.get("email"))
    code = _confirmation_code(body.get("code"))
    try:
        _client("cognito-idp").confirm_sign_up(
            ClientId=_env("COGNITO_APP_CLIENT_ID"),
            Username=email,
            ConfirmationCode=code,
        )
    except Exception as error:
        _raise_cognito_error(error, operation="confirm")
    return _response(200, {"confirmed": True, "email": email})


def _resend_confirmation(body):
    email = _normalize_email(body.get("email"))
    try:
        _client("cognito-idp").resend_confirmation_code(
            ClientId=_env("COGNITO_APP_CLIENT_ID"),
            Username=email,
        )
    except Exception as error:
        _raise_cognito_error(error, operation="resend")
    return _response(202, {"message": "A new confirmation code is on the way."})


def _sign_in(body):
    email = _normalize_email(body.get("email"))
    password = _password(body.get("password"))
    try:
        result = _client("cognito-idp").initiate_auth(
            ClientId=_env("COGNITO_APP_CLIENT_ID"),
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={"USERNAME": email, "PASSWORD": password},
        )
    except Exception as error:
        _raise_cognito_error(error, operation="sign_in")
    if not result.get("AuthenticationResult"):
        raise ApiError(401, "CourtVision could not complete sign in.", code="sign_in_incomplete")
    return _session_response(email)


def _forgot_password(body):
    email = _normalize_email(body.get("email"))
    try:
        _client("cognito-idp").forgot_password(
            ClientId=_env("COGNITO_APP_CLIENT_ID"),
            Username=email,
        )
    except Exception as error:
        if _aws_error_code(error) not in {"UserNotFoundException"}:
            _raise_cognito_error(error, operation="forgot_password")
    return _response(
        202,
        {"message": "If an account exists for this email, a reset code is on the way."},
    )


def _confirm_password(body):
    email = _normalize_email(body.get("email"))
    code = _confirmation_code(body.get("code"))
    password = _password(body.get("password"))
    try:
        _client("cognito-idp").confirm_forgot_password(
            ClientId=_env("COGNITO_APP_CLIENT_ID"),
            Username=email,
            ConfirmationCode=code,
            Password=password,
        )
    except Exception as error:
        _raise_cognito_error(error, operation="confirm_password")
    return _response(200, {"passwordReset": True, "email": email})


def _session_response(email):
    now = int(time.time())
    max_age = _env_int("SESSION_TTL_SECONDS", 8 * 60 * 60)
    payload = {
        "v": 1,
        "email": email,
        "csrf": secrets.token_urlsafe(24),
        "exp": now + max_age,
    }
    token = _encode_session(payload)
    cookie = (
        f"cv_session={token}; Path=/; Max-Age={max_age}; HttpOnly; Secure; "
        "SameSite=Strict"
    )
    return _response(
        200,
        {
            "authenticated": True,
            "email": email,
            "csrfToken": payload["csrf"],
            "expiresAt": _iso(payload["exp"]),
        },
        cookies=[cookie],
    )


def _raise_cognito_error(error, *, operation):
    code = _aws_error_code(error)
    if code == "UsernameExistsException":
        raise ApiError(409, "An account already exists for this email. Sign in instead.", code="account_exists") from error
    if code == "UserNotConfirmedException":
        raise ApiError(403, "Confirm your email before signing in.", code="confirmation_required") from error
    if code in {"NotAuthorizedException", "UserNotFoundException"}:
        raise ApiError(401, "The email or password is incorrect.", code="invalid_credentials") from error
    if code == "InvalidPasswordException":
        raise ApiError(400, "Use a password with at least 10 characters.", code="invalid_password") from error
    if code == "CodeMismatchException":
        raise ApiError(400, "That confirmation code is incorrect. Check the email and try again.", code="invalid_code") from error
    if code == "ExpiredCodeException":
        raise ApiError(400, "That confirmation code expired. Request a new code.", code="expired_code") from error
    if code in {"LimitExceededException", "TooManyRequestsException"}:
        raise ApiError(429, "Too many attempts. Wait a moment and try again.", code="rate_limited") from error
    if code == "InvalidParameterException":
        message = "CourtVision could not complete that account request. Check the details and try again."
        raise ApiError(400, message, code="invalid_account_request") from error
    LOGGER.warning("Cognito %s failed with %s", operation, code or type(error).__name__)
    raise ApiError(502, "CourtVision could not reach the account service. Try again.", code="account_service_error") from error


def _aws_error_code(error):
    response = getattr(error, "response", None)
    if isinstance(response, Mapping):
        details = response.get("Error") or {}
        return str(details.get("Code") or "")
    return ""


def _create_job(session, body):
    filename = _safe_filename(body.get("filename"))
    content_type = str(body.get("contentType", "")).lower()
    size_bytes = _positive_int(body.get("sizeBytes"), "Video size")
    duration = _positive_number(body.get("durationSeconds"), "Video duration")
    if content_type not in ALLOWED_VIDEO_TYPES:
        raise ApiError(400, "Use an MP4, MOV, or WebM video.", code="unsupported_video")

    max_bytes = _env_int("MAX_UPLOAD_BYTES", 500 * 1024 * 1024)
    max_duration = _env_float("MAX_DURATION_SECONDS", 30.0)
    if size_bytes > max_bytes:
        raise ApiError(413, f"This video exceeds the {_format_bytes(max_bytes)} beta limit.", code="video_too_large")
    if duration > max_duration + 0.05:
        raise ApiError(400, f"Choose a clip no longer than {max_duration:g} seconds.", code="video_too_long")

    job_id = str(uuid.uuid4())
    extension = _extension_for(filename, content_type)
    input_key = f"jobs/{job_id}/input/source{extension}"
    now = int(time.time())
    retention = _env_int("RESULT_RETENTION_SECONDS", 24 * 60 * 60)
    item = {
        "jobId": job_id,
        "ownerEmail": session["email"],
        "status": "awaiting_upload",
        "stage": "Waiting for upload",
        "filename": filename,
        "contentType": content_type,
        "sizeBytes": size_bytes,
        "durationSeconds": Decimal(str(round(duration, 3))),
        "inputKey": input_key,
        "createdAt": now,
        "updatedAt": now,
        "expiresAt": now + retention,
    }
    _table("JOBS_TABLE").put_item(Item=item)

    upload = _client("s3").generate_presigned_post(
        Bucket=_env("ARTIFACT_BUCKET"),
        Key=input_key,
        Fields={
            "Content-Type": content_type,
            "x-amz-meta-job-id": job_id,
        },
        Conditions=[
            {"Content-Type": content_type},
            {"x-amz-meta-job-id": job_id},
            ["content-length-range", size_bytes, size_bytes],
        ],
        ExpiresIn=_env_int("UPLOAD_URL_TTL_SECONDS", 900),
    )
    return _response(
        201,
        {
            "job": _public_job(item),
            "upload": {"url": upload["url"], "fields": upload["fields"]},
        },
    )


def _list_jobs(session):
    """Return this account's unexpired jobs, newest first."""

    table = _table("JOBS_TABLE")
    now = int(time.time())
    jobs = []
    cursor = None

    # DynamoDB TTL cleanup is asynchronous, so exclude expired rows explicitly.
    # Bound each page and the final response to keep this lightweight for the beta.
    for _ in range(4):
        query = {
            "IndexName": "OwnerCreatedAtIndex",
            "KeyConditionExpression": "ownerEmail = :owner",
            "FilterExpression": "expiresAt > :now",
            "ExpressionAttributeValues": {
                ":owner": session["email"],
                ":now": now,
            },
            "ScanIndexForward": False,
            "Limit": 25,
        }
        if cursor:
            query["ExclusiveStartKey"] = cursor
        page = table.query(**query)
        jobs.extend(page.get("Items", []))
        cursor = page.get("LastEvaluatedKey")
        if len(jobs) >= 25 or not cursor:
            break

    return _response(200, {"jobs": [_public_job(item) for item in jobs[:25]]})


def _start_job(session, job_id):
    job = _owned_job(session, job_id)
    if job["status"] not in {"awaiting_upload", "needs_team_colors", "failed"}:
        return _response(200, {"job": _public_job(job)})
    try:
        uploaded = _client("s3").head_object(
            Bucket=_env("ARTIFACT_BUCKET"),
            Key=job["inputKey"],
        )
    except Exception as error:
        LOGGER.info("Uploaded object not ready for %s: %s", job_id, error)
        raise ApiError(409, "The upload has not finished. Wait a moment and try again.", code="upload_incomplete")
    if (
        int(uploaded.get("ContentLength", -1)) != int(job["sizeBytes"])
        or str(uploaded.get("ContentType", "")).lower() != job["contentType"]
        or uploaded.get("Metadata", {}).get("job-id") != job_id
    ):
        raise ApiError(
            409,
            "The uploaded object does not match this analysis job. Upload the clip again.",
            code="invalid_upload",
        )
    return _submit_batch(job)


def _submit_team_colors(session, job_id, body):
    job = _owned_job(session, job_id)
    if not _can_change_team_colors(job):
        raise ApiError(409, "Team colors are not required for this job.", code="colors_not_required")
    color_one = str(body.get("team1Color", "")).upper()
    color_two = str(body.get("team2Color", "")).upper()
    if not HEX_COLOR_PATTERN.fullmatch(color_one) or not HEX_COLOR_PATTERN.fullmatch(color_two):
        raise ApiError(400, "Choose two six-digit jersey colors.", code="invalid_team_colors")
    if color_one == color_two:
        raise ApiError(400, "Choose two distinct jersey colors.", code="invalid_team_colors")
    if _worker_compatible_jersey_color(color_one) == _worker_compatible_jersey_color(color_two):
        raise ApiError(
            400,
            "Choose two jersey colors with more visible contrast.",
            code="invalid_team_colors",
        )
    job["team1Color"] = color_one
    job["team2Color"] = color_two
    job["allowUncertainTeams"] = False
    return _submit_batch(job)


def _submit_uncertain_teams(session, job_id):
    job = _owned_job(session, job_id)
    if not _can_change_team_colors(job):
        raise ApiError(
            409,
            "Team assignment is not waiting for a decision.",
            code="team_decision_not_required",
        )
    job.pop("team1Color", None)
    job.pop("team2Color", None)
    job["allowUncertainTeams"] = True
    return _submit_batch(job)


def _submit_batch(job):
    job_id = job["jobId"]
    environment = [
        {"name": "COURTVISION_JOB_ID", "value": job_id},
        {"name": "COURTVISION_INPUT_KEY", "value": job["inputKey"]},
        {"name": "COURTVISION_ARTIFACT_BUCKET", "value": _env("ARTIFACT_BUCKET")},
        {"name": "COURTVISION_JOBS_TABLE", "value": _env("JOBS_TABLE")},
        {"name": "COURTVISION_RETENTION_SECONDS", "value": str(_env_int("RESULT_RETENTION_SECONDS", 86400))},
        {"name": "COURTVISION_MAX_DURATION_SECONDS", "value": str(_env_float("MAX_DURATION_SECONDS", 30))},
        {"name": "COURTVISION_TARGET_FPS", "value": str(_env_float("TARGET_FPS", 30))},
        {"name": "COURTVISION_MAX_WIDTH", "value": str(_env_int("MAX_WIDTH", 1280))},
    ]
    if job.get("team1Color") and job.get("team2Color"):
        environment.extend(
            [
                {
                    "name": "COURTVISION_TEAM_1_COLOR",
                    "value": _worker_compatible_jersey_color(job["team1Color"]),
                },
                {
                    "name": "COURTVISION_TEAM_2_COLOR",
                    "value": _worker_compatible_jersey_color(job["team2Color"]),
                },
            ]
        )
    if job.get("allowUncertainTeams"):
        environment.append(
            {"name": "COURTVISION_ALLOW_UNCERTAIN_TEAMS", "value": "true"}
        )
    result = _client("batch").submit_job(
        jobName=f"courtvision-{job_id[:8]}",
        jobQueue=_env("BATCH_JOB_QUEUE"),
        jobDefinition=_env("BATCH_JOB_DEFINITION"),
        containerOverrides={"environment": environment},
        tags={"Project": "CourtVision", "CourtVisionJobId": job_id},
    )
    now = int(time.time())
    set_parts = [
        "#status = :status",
        "stage = :stage",
        "batchJobId = :batch",
        "updatedAt = :updated",
        "allowUncertainTeams = :allow_uncertain",
    ]
    remove_parts = ["errorMessage"]
    values = {
        ":status": "queued",
        ":stage": "Queued for analysis",
        ":batch": result["jobId"],
        ":updated": now,
        ":allow_uncertain": bool(job.get("allowUncertainTeams")),
    }
    if job.get("team1Color") and job.get("team2Color"):
        set_parts.extend(["team1Color = :team_one", "team2Color = :team_two"])
        values.update({":team_one": job["team1Color"], ":team_two": job["team2Color"]})
    else:
        remove_parts.extend(["team1Color", "team2Color"])
    response = _table("JOBS_TABLE").update_item(
        Key={"jobId": job_id},
        UpdateExpression=f"SET {', '.join(set_parts)} REMOVE {', '.join(remove_parts)}",
        ExpressionAttributeNames={"#status": "status"},
        ExpressionAttributeValues=values,
        ReturnValues="ALL_NEW",
    )
    return _response(202, {"job": _public_job(response["Attributes"])})


def _get_job(session, job_id):
    return _response(200, {"job": _public_job(_owned_job(session, job_id))})


def _get_downloads(session, job_id):
    job = _owned_job(session, job_id)
    if job["status"] != "complete":
        raise ApiError(409, "Results are not ready to download.", code="result_not_ready")
    s3 = _client("s3")
    bucket = _env("ARTIFACT_BUCKET")
    expires = _env_int("DOWNLOAD_URL_TTL_SECONDS", 600)
    return _response(
        200,
        {
            "videoUrl": s3.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": bucket,
                    "Key": job["outputVideoKey"],
                    "ResponseContentDisposition": f'attachment; filename="courtvision-{job_id[:8]}.mp4"',
                },
                ExpiresIn=expires,
            ),
            "playbackUrl": s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": job["outputVideoKey"]},
                ExpiresIn=expires,
            ),
            "analysisUrl": s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": job["analysisKey"]},
                ExpiresIn=expires,
            ),
            "expiresInSeconds": expires,
        },
    )


def _create_report(session, job_id, body):
    job = _owned_job(session, job_id)
    category = str(body.get("category", "")).strip()
    if category not in REPORT_CATEGORIES:
        raise ApiError(400, "Choose a failure category.", code="invalid_report")
    notes = str(body.get("notes", "")).strip()
    if not notes:
        raise ApiError(400, "Describe what CourtVision got wrong.", code="invalid_report")
    if len(notes) > 2000:
        raise ApiError(400, "Keep the report under 2,000 characters.", code="invalid_report")
    try:
        timestamp = float(body.get("timeSeconds", 0))
    except (TypeError, ValueError) as error:
        raise ApiError(400, "Enter a valid report timestamp.", code="invalid_report") from error
    duration = float(job.get("durationSeconds", 0))
    if timestamp < 0 or timestamp > duration + 0.1:
        raise ApiError(400, "The report timestamp is outside this clip.", code="invalid_report")

    now = int(time.time())
    report_id = str(uuid.uuid4())
    _table("REPORTS_TABLE").put_item(
        Item={
            "reportId": report_id,
            "jobId": job_id,
            "reporterEmail": session["email"],
            "category": category,
            "notes": notes,
            "timeSeconds": Decimal(str(round(timestamp, 3))),
            "eventId": str(body.get("eventId") or ""),
            "createdAt": now,
            "expiresAt": now + _env_int("REPORT_RETENTION_SECONDS", 90 * 86400),
            "jobExpiresAt": int(job["expiresAt"]),
        }
    )
    return _response(201, {"reportId": report_id, "message": "Failure report saved."})


def _owned_job(session, job_id):
    item = _table("JOBS_TABLE").get_item(Key={"jobId": job_id}, ConsistentRead=True).get("Item")
    if (
        not item
        or item.get("ownerEmail") != session["email"]
        or int(item.get("expiresAt", 0)) <= int(time.time())
    ):
        raise ApiError(404, "This analysis session is unavailable.", code="job_not_found")
    return item


def _public_job(item):
    return {
        "id": item["jobId"],
        "status": item["status"],
        "stage": item.get("stage", ""),
        "filename": item.get("filename", ""),
        "durationSeconds": float(item.get("durationSeconds", 0)),
        "createdAt": _iso(int(item.get("createdAt", 0))),
        "updatedAt": _iso(int(item.get("updatedAt", item.get("createdAt", 0)))),
        "expiresAt": _iso(int(item["expiresAt"])),
        "errorMessage": item.get("errorMessage"),
        "teamColorReason": item.get("teamColorReason"),
        "canChangeTeamColors": _can_change_team_colors(item),
    }


def _can_change_team_colors(job):
    if job.get("status") == "needs_team_colors":
        return True
    if job.get("status") != "failed":
        return False
    error = str(job.get("errorMessage", "")).casefold()
    return (
        "invalid team-color configuration" in error
        or "team jersey colors must" in error
    )


def _worker_compatible_jersey_color(color):
    """Lift explicit near-black swatches above the legacy worker's shadow floor."""
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    brightest = max(red, green, blue)
    minimum_visible_channel = 39
    if brightest >= minimum_visible_channel:
        return color
    if brightest == 0:
        red = green = blue = minimum_visible_channel
    else:
        scale = minimum_visible_channel / brightest
        red, green, blue = (
            min(255, int(round(channel * scale)))
            for channel in (red, green, blue)
        )
    return f"#{red:02X}{green:02X}{blue:02X}"


def _require_session(request):
    token = _cookie_value(request, "cv_session")
    if not token:
        raise ApiError(401, "Sign in to continue.", code="authentication_required")
    payload = _decode_session(token)
    if int(payload.get("exp", 0)) <= int(time.time()):
        raise ApiError(401, "Your session expired. Sign in again.", code="session_expired")
    return payload


def _require_csrf(request, session):
    headers = {str(key).lower(): value for key, value in request.headers.items()}
    if not hmac.compare_digest(str(headers.get("x-courtvision-csrf", "")), session["csrf"]):
        raise ApiError(403, "Refresh the page and try again.", code="invalid_request_token")


def _encode_session(payload):
    encoded = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = _b64url(hmac.new(_session_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
    return f"{encoded}.{signature}"


def _decode_session(token):
    try:
        encoded, signature = token.split(".", 1)
        expected = _b64url(hmac.new(_session_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature")
        return json.loads(_b64url_decode(encoded).decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ApiError(401, "Sign in to continue.", code="invalid_session") from error


def _session_secret():
    global _SESSION_SECRET
    if _SESSION_SECRET is not None:
        return _SESSION_SECRET
    direct = os.getenv("SESSION_SECRET")
    if direct:
        if os.getenv("ENVIRONMENT", "production") == "production":
            LOGGER.warning("SESSION_SECRET is set directly; Secrets Manager is preferred.")
        _SESSION_SECRET = direct.encode("utf-8")
        return _SESSION_SECRET
    result = _client("secretsmanager").get_secret_value(SecretId=_env("SESSION_SECRET_ARN"))
    _SESSION_SECRET = result["SecretString"].encode("utf-8")
    return _SESSION_SECRET


def _request_target(event):
    context = event.get("requestContext", {}).get("http", {})
    method = str(context.get("method") or event.get("httpMethod") or "GET").upper()
    path = str(event.get("rawPath") or event.get("path") or "/")
    prefix = os.getenv("API_PREFIX", "/api").rstrip("/")
    if prefix and path.startswith(prefix):
        path = path[len(prefix) :] or "/"
    return method, path


def _json_body(request):
    body = request.body or "{}"
    if request.is_base64_encoded:
        body = base64.b64decode(body).decode("utf-8")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise ApiError(400, "Send a valid JSON request.", code="invalid_json") from error
    if not isinstance(value, dict):
        raise ApiError(400, "Send a JSON object.", code="invalid_json")
    return value


def _cookie_value(request, name):
    cookies = list(request.cookies)
    header = request.headers.get("cookie") or request.headers.get("Cookie")
    if header:
        cookies.extend(header.split(";"))
    for part in cookies:
        key, separator, value = part.strip().partition("=")
        if separator and key == name:
            return value
    return None


def _response(status_code, body, *, cookies=None):
    allowed_origin = os.getenv("ALLOWED_WEB_ORIGIN", "https://courtvision.video")
    response = {
        "statusCode": status_code,
        "headers": {
            "content-type": "application/json; charset=utf-8",
            "cache-control": "no-store",
            "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
            "x-content-type-options": "nosniff",
            "access-control-allow-origin": allowed_origin,
            "access-control-allow-credentials": "true",
            "access-control-allow-methods": "GET,HEAD,OPTIONS,POST",
            "access-control-allow-headers": "Content-Type,X-CourtVision-CSRF",
            "vary": "Origin",
        },
        "body": "" if body is None else json.dumps(body, default=_decimal_json),
    }
    if cookies:
        response["cookies"] = cookies
    return response


def _client(name):
    if boto3 is None:
        raise RuntimeError("boto3 is required for AWS operations")
    if name not in _CLIENTS:
        _CLIENTS[name] = boto3.client(name)
    return _CLIENTS[name]


def _table(environment_name):
    if boto3 is None:
        raise RuntimeError("boto3 is required for AWS operations")
    key = f"table:{environment_name}"
    if key not in _CLIENTS:
        _CLIENTS[key] = boto3.resource("dynamodb").Table(_env(environment_name))
    return _CLIENTS[key]


def _env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _env_int(name, default):
    return int(os.getenv(name, str(default)))


def _env_float(name, default):
    return float(os.getenv(name, str(default)))


def _normalize_email(value):
    email = str(value or "").strip().lower()
    if len(email) > 254 or not EMAIL_PATTERN.fullmatch(email):
        raise ApiError(400, "Enter a valid email address.", code="invalid_email")
    return email


def _password(value):
    password = str(value or "")
    if len(password) < 10 or len(password) > 256:
        raise ApiError(400, "Use a password with at least 10 characters.", code="invalid_password")
    return password


def _confirmation_code(value):
    code = str(value or "").strip()
    if not re.fullmatch(r"\d{6}", code):
        raise ApiError(400, "Enter the six-digit code from your email.", code="invalid_code")
    return code


def _safe_filename(value):
    filename = str(value or "").strip().replace("\\", "/").split("/")[-1]
    filename = re.sub(r"[^A-Za-z0-9._ -]", "", filename)[:120]
    if not filename:
        raise ApiError(400, "Choose a video file.", code="invalid_filename")
    return filename


def _positive_int(value, label):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ApiError(400, f"{label} is invalid.") from error
    if number <= 0:
        raise ApiError(400, f"{label} must be positive.")
    return number


def _positive_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ApiError(400, f"{label} is invalid.") from error
    if not math.isfinite(number) or number <= 0:
        raise ApiError(400, f"{label} must be positive.")
    return number


def _extension_for(filename, content_type):
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    allowed = {".mp4", ".mov", ".webm"}
    if suffix in allowed:
        return suffix
    return {"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm"}[content_type]


def _format_bytes(value):
    if value >= 1024**3:
        return f"{value / 1024**3:.1f} GiB"
    return f"{round(value / 1024**2)} MiB"


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def _b64url(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value):
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _expired_cookie():
    return "cv_session=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict"


def _decimal_json(value):
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
