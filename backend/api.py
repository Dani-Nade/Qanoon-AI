"""Compatibility entrypoint for `uvicorn api:app`.

The real application lives in `qanoon_ai.api.app` so the backend can grow with
an enterprise-grade package structure.
"""

from qanoon_ai.api.app import app
