"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from qanoon_ai.api.routes.case_strength import router as case_strength_router
from qanoon_ai.api.routes.chat import router as chat_router
from qanoon_ai.api.routes.documents import router as documents_router
from qanoon_ai.api.routes.health import router as health_router
from qanoon_ai.api.routes.query import router as query_router
from qanoon_ai.core.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Qanoon AI", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(case_strength_router)
    return app


app = create_app()
