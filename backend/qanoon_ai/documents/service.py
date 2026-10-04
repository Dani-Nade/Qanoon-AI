"""Upload handling and background page reading."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

from qanoon_ai.core.config import Settings, settings
from qanoon_ai.documents.reading import OcrClient, VisionChecker, read_page
from qanoon_ai.documents.render import SUPPORTED_EXTENSIONS, UnsupportedDocument, page_count, render_pages
from qanoon_ai.documents.store import DocumentStore

LOGGER = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, app_settings: Settings = settings, store: DocumentStore | None = None,
                 ocr: OcrClient | None = None, vision: VisionChecker | None = None):
        self.settings = app_settings
        self.store = store or DocumentStore(app_settings.uploads_dir, app_settings.document_retention_hours)
        self.ocr = ocr or OcrClient(app_settings.ocr_url)
        self.vision = vision or VisionChecker(app_settings.ollama_url, app_settings.chat_model, app_settings.chat_num_ctx)
        # One document at a time: OCR and the vision model share the GPU with chat.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="document-reader")

    def upload(self, filename: str, data: bytes) -> dict:
        self.store.purge_expired()
        extension = Path(filename or "").suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedDocument("Upload a PDF or an image (PNG, JPG, WEBP, TIFF or BMP).")
        if not data:
            raise UnsupportedDocument("The file is empty.")
        if len(data) > self.settings.document_max_bytes:
            raise UnsupportedDocument(f"The file is larger than {self.settings.document_max_bytes // (1024 * 1024)} MB.")
        document_id = self.store.create(Path(filename).name, data, extension)
        try:
            pages = page_count(self.store.original(document_id))
            if pages > self.settings.document_max_pages:
                raise UnsupportedDocument(f"The document has {pages} pages; the limit is {self.settings.document_max_pages}.")
        except UnsupportedDocument:
            self.store.delete(document_id)
            raise
        meta = self.store.update_meta(document_id, page_count=pages)
        self._worker.submit(self._process, document_id)
        return meta

    def _process(self, document_id: str) -> None:
        try:
            pages_dir = self.store.page_image_path(document_id, 1).parent
            rendered = render_pages(self.store.original(document_id), pages_dir, self.settings.document_max_pages)
            ocr_ready = self.ocr.available()
            for page in rendered:
                if not page.text_blocks and not ocr_ready:
                    raise RuntimeError("The OCR service is not running, so scanned pages cannot be read.")
                record = read_page(page.image_path, page.text_blocks, self.ocr, self.vision)
                record.update({"number": page.number, "width": page.width, "height": page.height})
                self.store.save_page(document_id, page.number, record)
                self.store.update_meta(document_id, pages_done=page.number)
            self.store.update_meta(document_id, status="ready")
        except Exception as error:
            LOGGER.exception("Reading document %s failed", document_id)
            try:
                self.store.update_meta(document_id, status="error", error=str(error))
            except KeyError:
                pass  # deleted while processing

    def document(self, document_id: str) -> dict:
        self.store.purge_expired()
        meta = self.store.meta(document_id)
        return {**meta, "pages": self.store.pages(document_id)}

    def correct_page(self, document_id: str, number: int, corrections: dict[int, str]) -> dict:
        page = self.store.page(document_id, number)
        by_index = {block["index"]: block for block in page["blocks"]}
        for index, text in corrections.items():
            if index not in by_index:
                raise KeyError(f"block {index}")
            block = by_index[index]
            block["text"] = text
            block["edited"] = text != block["original_text"]
            if block["edited"]:
                block["uncertain"] = []  # the user has checked this block
        self.store.save_page(document_id, number, page)
        return page
