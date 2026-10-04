"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from qanoon_ai.core.config import settings
from qanoon_ai.ingestion.legal_index import index_status
from qanoon_ai.ingestion.postgres_index import production_index_status
from qanoon_ai.api.routes.chat import chat_service
from qanoon_ai.api.routes.query import answer_service
from qanoon_ai.llm.client import adapter_complete

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    search_index = index_status(settings.search_index_file)
    production_index = (
        production_index_status(settings.database_url, embedding_model=settings.embedding_model,
                                embedding_revision=settings.embedding_revision,
                                embedding_dimension=settings.embedding_dimension)
        if settings.retrieval_mode in {"auto", "postgres"}
        else {"ready": False, "documents": 0, "chunks": 0}
    )
    chat = chat_service.llm.status()
    return {
        "status": "ok",
        "retrieval_mode": settings.retrieval_mode,
        "production_index_ready": production_index.get("ready", False),
        "search_index_ready": production_index.get("ready", False)
        or search_index.get("ready", False),
        "indexed_documents": production_index.get("documents", 0)
        or search_index.get("documents", 0),
        "indexed_chunks": production_index.get("chunks", 0)
        or search_index.get("chunks", 0),
        "chunks_index_ready": settings.chunks_file.exists(),
        "vector_index_ready": settings.vector_index_file.exists()
        and settings.vector_metadata_file.exists(),
        # Reported without loading anything: the legacy /query model only loads when that mode is enabled.
        "model_ready": answer_service.model_ready if settings.llm_mode in {"local", "auto", "quotation"} else False,
        "trained_adapter_available": adapter_complete(settings.trained_model_dir),
        "active_model": getattr(answer_service.llm, "model_name", None),
        "llm_mode": settings.llm_mode,
        "chat_model": settings.chat_model,
        "chat_model_ready": chat.get("model_available", False),
        "ollama_reachable": chat.get("reachable", False),
    }
