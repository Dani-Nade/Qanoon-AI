"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the backend."""

    root_dir: Path
    backend_dir: Path
    pdfs_dir: Path
    datasets_dir: Path
    datasets_raw_dir: Path
    datasets_manifest_dir: Path
    canonical_manifest_file: Path
    official_source_catalog_file: Path
    download_manifest_file: Path
    registry_dir: Path
    source_registry_file: Path
    chunks_file: Path
    search_index_file: Path
    reports_dir: Path
    build_report_file: Path
    vector_dir: Path
    vector_index_file: Path
    vector_metadata_file: Path
    model_cache_dir: Path
    trained_model_dir: Path
    database_url: str
    retrieval_mode: str
    embedding_model: str
    embedding_revision: str
    vector_search: bool
    ollama_url: str
    chat_model: str
    chat_think: bool
    chat_max_tokens: int
    chat_num_ctx: int
    chat_max_sources: int
    uploads_dir: Path
    ocr_url: str
    document_max_pages: int
    document_max_bytes: int
    document_retention_hours: int
    reranker_model: str
    reranker_min_score: float
    embedding_dimension: int
    indexing_batch_size: int
    indexing_workers: int
    enable_ocr: bool
    llm_mode: str
    retrieval_top_k: int
    answer_max_sources: int

    @classmethod
    def from_env(cls) -> "Settings":
        root = _env_path("QANOON_ROOT", _project_root())
        backend = _env_path("QANOON_BACKEND_DIR", _backend_root())
        return cls(
            root_dir=root,
            backend_dir=backend,
            pdfs_dir=_env_path("QANOON_PDFS_DIR", root / "pdfs"),
            datasets_dir=_env_path("QANOON_DATASETS_DIR", root / "datasets"),
            datasets_raw_dir=_env_path("QANOON_DATASETS_RAW_DIR", root / "datasets" / "raw"),
            datasets_manifest_dir=_env_path(
                "QANOON_DATASETS_MANIFEST_DIR",
                root / "datasets" / "manifests",
            ),
            canonical_manifest_file=_env_path(
                "QANOON_CANONICAL_MANIFEST_FILE",
                root / "datasets" / "manifests" / "canonical_manifest.jsonl",
            ),
            official_source_catalog_file=_env_path(
                "QANOON_OFFICIAL_SOURCE_CATALOG_FILE",
                root / "datasets" / "registry" / "official_sources.json",
            ),
            download_manifest_file=_env_path(
                "QANOON_DOWNLOAD_MANIFEST_FILE",
                root / "datasets" / "manifests" / "download_manifest.jsonl",
            ),
            registry_dir=_env_path("QANOON_REGISTRY_DIR", root / "data" / "registry"),
            source_registry_file=_env_path(
                "QANOON_SOURCE_REGISTRY_FILE",
                root / "data" / "registry" / "sources.jsonl",
            ),
            chunks_file=_env_path(
                "QANOON_CHUNKS_FILE",
                root / "data" / "chunks" / "chunks.jsonl",
            ),
            search_index_file=_env_path(
                "QANOON_SEARCH_INDEX_FILE",
                root / "data" / "index" / "qanoon_fts.sqlite3",
            ),
            reports_dir=_env_path("QANOON_REPORTS_DIR", root / "data" / "reports"),
            build_report_file=_env_path(
                "QANOON_BUILD_REPORT_FILE",
                root / "data" / "reports" / "system-build-latest.json",
            ),
            vector_dir=_env_path("QANOON_VECTOR_DIR", root / "data" / "index" / "vectors"),
            vector_index_file=_env_path(
                "QANOON_VECTOR_INDEX_FILE",
                root / "data" / "index" / "vectors" / "embeddings.npy",
            ),
            vector_metadata_file=_env_path(
                "QANOON_VECTOR_METADATA_FILE",
                root / "data" / "index" / "vectors" / "manifest.json",
            ),
            model_cache_dir=_env_path("QANOON_MODEL_CACHE_DIR", root / "data" / "cache" / "huggingface" / "hub"),
            trained_model_dir=_env_path(
                "QANOON_TRAINED_MODEL_DIR",
                backend / "training" / "qanoon-model",
            ),
            database_url=os.getenv(
                "QANOON_DATABASE_URL",
                "postgresql://qanoon:qanoon@127.0.0.1:5432/qanoon",
            ).strip(),
            retrieval_mode=os.getenv("QANOON_RETRIEVAL_MODE", "auto").strip().lower(),
            embedding_model=os.getenv(
                "QANOON_EMBEDDING_MODEL", "BAAI/bge-m3"
            ).strip(),
            embedding_revision=os.getenv(
                "QANOON_EMBEDDING_REVISION", "5617a9f61b028005a4858fdac845db406aefb181"
            ).strip(),
            vector_search=_env_bool("QANOON_VECTOR_SEARCH", True),
            ollama_url=os.getenv("QANOON_OLLAMA_URL", "http://127.0.0.1:11434").strip(),
            chat_model=os.getenv("QANOON_CHAT_MODEL", "qwen3.5:9b").strip(),
            # Thinking fixed misread provisos in testing (e.g. CrPC 497 time limits) at ~15-60 s per answer.
            chat_think=_env_bool("QANOON_CHAT_THINK", True),
            chat_max_tokens=_env_int("QANOON_CHAT_MAX_TOKENS", 8192),
            chat_num_ctx=_env_int("QANOON_CHAT_NUM_CTX", 24576),
            # Several planned searches share these slots, so a procedure needs more than one per search.
            chat_max_sources=_env_int("QANOON_CHAT_MAX_SOURCES", 8),
            # Uploaded documents: kept outside the legal index and deleted after the retention period.
            uploads_dir=_env_path("QANOON_UPLOADS_DIR", root / "data" / "uploads"),
            ocr_url=os.getenv("QANOON_OCR_URL", "http://127.0.0.1:8010").strip(),
            document_max_pages=_env_int("QANOON_DOCUMENT_MAX_PAGES", 20),
            document_max_bytes=_env_int("QANOON_DOCUMENT_MAX_BYTES", 25 * 1024 * 1024),
            document_retention_hours=_env_int("QANOON_DOCUMENT_RETENTION_HOURS", 24),
            reranker_model=os.getenv(
                "QANOON_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
            ).strip(),
            reranker_min_score=_env_float("QANOON_RERANKER_MIN_SCORE", 0.35),
            embedding_dimension=_env_int("QANOON_EMBEDDING_DIMENSION", 1024),
            indexing_batch_size=_env_int("QANOON_INDEXING_BATCH_SIZE", 32),
            indexing_workers=_env_int("QANOON_INDEXING_WORKERS", 4),
            enable_ocr=_env_bool("QANOON_ENABLE_OCR", True),
            # The chat endpoint answers through Ollama; the legacy /query model is opt-in so it
            # does not occupy GPU memory next to the chat model.
            llm_mode=os.getenv("QANOON_LLM_MODE", "extracts").strip().lower(),
            retrieval_top_k=_env_int("QANOON_RETRIEVAL_TOP_K", 8),
            answer_max_sources=_env_int("QANOON_ANSWER_MAX_SOURCES", 5),
        )


settings = Settings.from_env()
