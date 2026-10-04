"""Local dense-vector search over the SQLite legal index (BGE-M3, no database server).

Embeddings are stored next to the SQLite index as a float16 matrix keyed by chunk id,
so the keyword index stays the source of truth: chunks added later are embedded
incrementally, and vectors for chunks that no longer exist are ignored at search time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
from threading import Lock
from contextlib import closing

import numpy as np

LOGGER = logging.getLogger(__name__)

EMBEDDINGS_FILE = "embeddings.npy"
CHUNK_IDS_FILE = "chunk_ids.json"
MANIFEST_FILE = "manifest.json"
MAX_TOKENS = 512


def passage_text(title: str | None, section_ref: str | None, text: str) -> str:
    return f"{title or ''} | {section_ref or ''}\n{' '.join(text.split())}"


class Encoder:
    """BGE-M3 dense encoder: CLS pooling, L2-normalised (cosine similarity = dot product)."""

    def __init__(self, model_path: str | Path, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True, dtype=dtype).to(self.device).eval()

    def encode(self, texts: list[str], max_tokens: int = MAX_TOKENS) -> np.ndarray:
        import torch

        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            cls = self.model(**batch).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls.float(), dim=-1)
        return cls.cpu().numpy().astype(np.float16)


def resolve_model_path(cache_dir: Path, model: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision, cache_dir=str(cache_dir), local_files_only=True)


def build_vectors(index_file: Path, vector_dir: Path, model_path: str, model_id: str, batch_size: int = 64, limit: int | None = None) -> dict:
    """Embed chunks that do not have a vector yet; returns a summary."""
    vector_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = vector_dir / MANIFEST_FILE
    existing_ids: list[str] = []
    existing = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model") == model_id:
            existing_ids = json.loads((vector_dir / CHUNK_IDS_FILE).read_text(encoding="utf-8"))
            existing = np.load(vector_dir / EMBEDDINGS_FILE)
        else:
            LOGGER.warning("Embedding model changed (%s -> %s); rebuilding all vectors", manifest.get("model"), model_id)

    with closing(sqlite3.connect(f"file:{index_file}?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT c.chunk_id, d.title, c.section_ref, c.text FROM chunks c JOIN documents d USING(document_id) ORDER BY c.chunk_id"
        ).fetchall()
    known = set(existing_ids)
    pending = [row for row in rows if row[0] not in known]
    if limit is not None:
        pending = pending[:limit]
    # Sorting by length keeps padding (and GPU time) per batch small.
    pending.sort(key=lambda row: len(row[3]))

    encoder = Encoder(model_path) if pending else None
    new_ids: list[str] = []
    new_vectors: list[np.ndarray] = []
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        new_vectors.append(encoder.encode([passage_text(title, ref, text) for _, title, ref, text in batch]))
        new_ids.extend(row[0] for row in batch)
        if (start // batch_size) % 50 == 0:
            LOGGER.info("Embedded %d/%d chunks", start + len(batch), len(pending))

    all_ids = existing_ids + new_ids
    parts = ([existing] if existing is not None and len(existing_ids) else []) + new_vectors
    matrix = np.concatenate(parts) if parts else np.zeros((0, 1024), dtype=np.float16)
    np.save(vector_dir / EMBEDDINGS_FILE, matrix)
    (vector_dir / CHUNK_IDS_FILE).write_text(json.dumps(all_ids), encoding="utf-8")
    manifest_path.write_text(json.dumps({"model": model_id, "dimension": int(matrix.shape[1]), "vectors": len(all_ids)}, indent=2), encoding="utf-8")
    return {"embedded": len(new_ids), "total_vectors": len(all_ids), "index_chunks": len(rows), "missing": len(rows) - len(all_ids)}


class VectorSearcher:
    """Loads vectors and the query encoder lazily on first search."""

    def __init__(self, index_file: Path, vector_dir: Path, cache_dir: Path, model: str, revision: str):
        self.index_file = index_file
        self.vector_dir = vector_dir
        self.cache_dir = cache_dir
        self.model = model
        self.revision = revision
        self._lock = Lock()
        self._loaded = False
        self._failed = False

    @property
    def available(self) -> bool:
        manifest = self.vector_dir / MANIFEST_FILE
        if self._failed or not manifest.exists():
            return False
        try:
            return json.loads(manifest.read_text(encoding="utf-8")).get("model") == self.model
        except (OSError, ValueError):
            return False

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            import torch

            self._chunk_ids = json.loads((self.vector_dir / CHUNK_IDS_FILE).read_text(encoding="utf-8"))
            with closing(sqlite3.connect(f"file:{self.index_file}?mode=ro", uri=True)) as connection:
                jurisdiction_of = dict(connection.execute("SELECT c.chunk_id, d.jurisdiction FROM chunks c JOIN documents d USING(document_id)"))
            # Vectors for chunks that were removed from the index are masked out.
            self._jurisdictions = np.array([jurisdiction_of.get(chunk_id, "__removed__") for chunk_id in self._chunk_ids], dtype=object)
            self._encoder = Encoder(resolve_model_path(self.cache_dir, self.model, self.revision))
            matrix = torch.from_numpy(np.load(self.vector_dir / EMBEDDINGS_FILE)).to(self._encoder.device)
            # float16 matmul is fast on GPU but unsupported on some CPUs.
            self._matrix = matrix if matrix.device.type == "cuda" else matrix.float()
            self._loaded = True

    def search(self, query: str, top_k: int = 50, jurisdictions: tuple[str, ...] | None = None) -> list[tuple[str, float]]:
        """Return [(chunk_id, cosine similarity)] best first."""
        try:
            self._load()
        except Exception:
            LOGGER.exception("Vector search unavailable")
            self._failed = True
            return []
        import torch

        with self._lock:
            query_vector = torch.from_numpy(self._encoder.encode([query], max_tokens=128)).to(self._matrix.device, self._matrix.dtype)
            scores = (self._matrix @ query_vector[0]).float().cpu().numpy()
        allowed = self._jurisdictions != "__removed__"
        if jurisdictions:
            allowed &= np.isin(self._jurisdictions, list(jurisdictions))
        scores = np.where(allowed, scores, -1.0)
        best = np.argsort(-scores)[:top_k]
        return [(self._chunk_ids[i], float(scores[i])) for i in best if scores[i] > -1.0]
