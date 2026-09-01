"""Compatibility alias for the dependency-light web API implementation."""

import sys

from backend.lambda_api import web_api as _implementation


sys.modules[__name__] = _implementation
