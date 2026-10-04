"""On-disk storage for uploaded documents, with automatic expiry.

Layout: <uploads_dir>/<document_id>/
  meta.json            status, file name, page count, timestamps
  original.<ext>       the uploaded file
  pages/<n>.png        rendered page images
  pages/<n>.json       reading results and user corrections for page n
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
from threading import Lock
import uuid

ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentNotFound(KeyError):
    pass


class DocumentStore:
    def __init__(self, root: Path, retention_hours: int = 24):
        self.root = root
        self.retention = timedelta(hours=retention_hours)
        self._lock = Lock()

    def _dir(self, document_id: str) -> Path:
        # Ids are generated here; anything else (including path tricks) is not a document.
        if not ID_RE.fullmatch(document_id or ""):
            raise DocumentNotFound(document_id)
        path = self.root / document_id
        if not (path / "meta.json").is_file():
            raise DocumentNotFound(document_id)
        return path

    def create(self, filename: str, data: bytes, extension: str) -> str:
        document_id = uuid.uuid4().hex
        path = self.root / document_id
        (path / "pages").mkdir(parents=True)
        (path / f"original{extension}").write_bytes(data)
        created = _now()
        self._write(path / "meta.json", {
            "id": document_id, "filename": filename, "extension": extension, "status": "processing",
            "page_count": 0, "pages_done": 0, "error": None,
            "created_at": created.isoformat(), "expires_at": (created + self.retention).isoformat(),
        })
        return document_id

    def original(self, document_id: str) -> Path:
        path = self._dir(document_id)
        return next(path.glob("original.*"))

    def meta(self, document_id: str) -> dict:
        return json.loads((self._dir(document_id) / "meta.json").read_text(encoding="utf-8"))

    def update_meta(self, document_id: str, **changes) -> dict:
        with self._lock:
            path = self._dir(document_id) / "meta.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta.update(changes)
            self._write(path, meta)
            return meta

    def page_image_path(self, document_id: str, number: int) -> Path:
        return self._dir(document_id) / "pages" / f"{number}.png"

    def save_page(self, document_id: str, number: int, page: dict) -> None:
        self._write(self._dir(document_id) / "pages" / f"{number}.json", page)

    def page(self, document_id: str, number: int) -> dict:
        path = self._dir(document_id) / "pages" / f"{number}.json"
        if not path.is_file():
            raise DocumentNotFound(f"{document_id}/{number}")
        return json.loads(path.read_text(encoding="utf-8"))

    def pages(self, document_id: str) -> list[dict]:
        folder = self._dir(document_id) / "pages"
        files = sorted(folder.glob("*.json"), key=lambda p: int(p.stem))
        return [json.loads(p.read_text(encoding="utf-8")) for p in files]

    def delete(self, document_id: str) -> None:
        shutil.rmtree(self._dir(document_id), ignore_errors=True)

    def purge_expired(self) -> int:
        """Delete documents past their expiry time; returns how many were removed."""
        if not self.root.exists():
            return 0
        removed = 0
        now = _now()
        for path in self.root.iterdir():
            meta_file = path / "meta.json"
            try:
                expires = datetime.fromisoformat(json.loads(meta_file.read_text(encoding="utf-8"))["expires_at"])
            except (OSError, ValueError, KeyError):
                # Unreadable or half-created uploads older than the retention period are removed too.
                expires = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) + self.retention
            if expires <= now:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed

    @staticmethod
    def _write(path: Path, value: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
        temporary.replace(path)
