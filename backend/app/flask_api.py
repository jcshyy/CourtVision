"""Flask adapter for the CourtVision bounded analysis API contract.

The Flask process is a control plane only. Browser uploads still go directly to
S3, and video inference still runs in isolated AWS Batch workers.
"""

from __future__ import annotations

import os

from flask import Flask, Response, jsonify, request

from backend.app.web_api import ApiRequest, handle_request


API_METHODS = ["GET", "HEAD", "POST", "OPTIONS"]


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 128 * 1024

    @app.get("/health")
    def health():
        return jsonify({"status": "healthy", "service": "courtvision-api"})

    @app.route(
        "/api",
        defaults={"api_path": ""},
        methods=API_METHODS,
        strict_slashes=False,
    )
    @app.route("/api/<path:api_path>", methods=API_METHODS)
    def api(api_path):
        result = handle_request(
            ApiRequest(
                method=request.method,
                path=f"/{api_path}" if api_path else "/",
                headers=dict(request.headers.items()),
                body=request.get_data(as_text=True),
            )
        )
        response = Response(result.get("body", ""), status=result["statusCode"])
        for name, value in result.get("headers", {}).items():
            response.headers[name] = value
        for cookie in result.get("cookies", []):
            response.headers.add("Set-Cookie", cookie)
        return response

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
