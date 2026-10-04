from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from dataclasses import replace

import fitz
import pytest

from qanoon_ai.core.config import settings
from qanoon_ai.documents.reading import cross_check_numbers, mark_handwriting, read_page
from qanoon_ai.documents.render import UnsupportedDocument, render_pages
from qanoon_ai.documents.service import DocumentService
from qanoon_ai.documents.store import DocumentNotFound, DocumentStore


def _block(index, text):
    return {"index": index, "label": "Text", "bbox": [0, 0, 10, 10], "confidence": 0.98, "text": text,
            "handwritten": False, "uncertain": [], "edited": False, "original_text": text}


def _pdf(path: Path, pages: list[str]) -> Path:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 560, 800), text, fontsize=11)
    document.save(path)
    return path


class FakeOcr:
    def __init__(self, blocks):
        self.blocks = blocks
        self.calls = 0

    def available(self):
        return True

    def read_page(self, image_path):
        self.calls += 1
        return {"width": 100, "height": 100, "seconds": 0.1, "blocks": [dict(b) for b in self.blocks]}


class FakeVision:
    def __init__(self, check, transcription=""):
        self.check, self.transcription = check, transcription

    def check_page(self, image_path):
        return self.check

    def transcribe(self, image_path):
        return self.transcription


def test_handwritten_items_are_located_in_blocks_and_unlocated_ones_become_notes():
    blocks = [_block(0, "BAIL PETITION NO. 353-B/2016."), _block(1, "FIR NO. 313 DATED 11-05-2016")]
    check = {"handwriting_present": True, "handwritten_items": [
        {"text": "353-B", "location": "petition number"}, {"text": "", "location": "top right corner"}]}
    notes = mark_handwriting(blocks, check)
    assert blocks[0]["handwritten"] and "353-B" in blocks[0]["uncertain"]
    assert not blocks[1]["handwritten"]
    assert len(notes) == 1 and "top right corner" in notes[0]


def test_misread_handwriting_is_still_located_and_signatures_are_not_handwriting():
    blocks = [_block(0, "BAIL PETITION NO. 383-B/2016."), _block(1, "1st - 586 B/2016"),
              _block(2, "[Handwritten signature] FIR NO. 313 DATED 11-05-2016"), _block(3, "Versus")]
    check = {"handwriting_present": True, "handwritten_items": [
        {"text": "353-B/2016", "location": "top right"}, {"text": "P/s-565-B/2016", "location": "middle right"}]}
    notes = mark_handwriting(blocks, check)
    assert blocks[0]["handwritten"] and blocks[1]["handwritten"]
    assert not blocks[2]["handwritten"] and not blocks[3]["handwritten"]
    assert notes == []


def test_inline_handwriting_marks_from_ocr_are_flagged():
    blocks = [_block(0, "Case No. [Handwritten: 245] of 2026")]
    mark_handwriting(blocks, {"handwriting_present": False, "handwritten_items": []})
    assert blocks[0]["handwritten"]


def test_numbers_missing_from_the_second_reading_are_flagged():
    blocks = [_block(0, "FIR No. 313 dated 11-05-2016, hearing on 22-01-2009")]
    flagged = cross_check_numbers(blocks, "FIR No. 313 dated 11-05-2016, hearing on 22-01-2008")
    assert flagged == 1 and blocks[0]["uncertain"] == ["22-01-2009"]


def test_urdu_digits_are_compared_as_numbers():
    blocks = [_block(0, "مقدمہ نمبر ۲۴۵/۲۰۲۶")]
    assert cross_check_numbers(blocks, "Case No. 245/2026") == 0


def test_urdu_pages_are_not_cross_checked_with_the_vision_transcription(tmp_path):
    ocr = FakeOcr([{"index": 0, "label": "Text", "bbox": [0, 0, 1, 1], "confidence": 0.99, "text": "ایکٹ نمبر ۵ بابت ۱۸۹۸ء", "skipped": False}])
    vision = FakeVision({"document_type": "law", "language": "urdu", "handwriting_present": False, "handwritten_items": []},
                        transcription="garbage")
    page = read_page(tmp_path / "1.png", [], ocr, vision)
    assert page["blocks"][0]["uncertain"] == []
    assert any("not cross-checked" in w for w in page["warnings"])


def test_text_layer_pages_skip_ocr(tmp_path):
    ocr = FakeOcr([])
    vision = FakeVision({"document_type": "act", "language": "english", "handwriting_present": False, "handwritten_items": []})
    page = read_page(tmp_path / "1.png", [{"bbox": [0, 0, 1, 1], "text": "Section 1. Short title."}], ocr, vision)
    assert page["method"] == "text-layer" and ocr.calls == 0


def test_render_uses_text_layer_only_when_the_page_has_real_text(tmp_path):
    pdf = _pdf(tmp_path / "doc.pdf", ["Section 7. Talaq. " * 20, "x"])
    out = tmp_path / "pages"
    out.mkdir()
    pages = render_pages(pdf, out, max_pages=20)
    assert [p.number for p in pages] == [1, 2]
    assert pages[0].text_blocks and not pages[1].text_blocks
    assert pages[0].image_path.is_file() and pages[0].width > 2000  # 300 DPI


def test_render_rejects_documents_over_the_page_limit(tmp_path):
    pdf = _pdf(tmp_path / "long.pdf", ["page"] * 3)
    with pytest.raises(UnsupportedDocument):
        render_pages(pdf, tmp_path, max_pages=2)


def test_store_rejects_path_tricks_and_purges_expired_documents(tmp_path):
    store = DocumentStore(tmp_path, retention_hours=24)
    with pytest.raises(DocumentNotFound):
        store.meta("../etc")
    keep = store.create("a.pdf", b"%PDF", ".pdf")
    old = store.create("b.pdf", b"%PDF", ".pdf")
    store.update_meta(old, expires_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat())
    assert store.purge_expired() == 1
    assert store.meta(keep)["filename"] == "a.pdf"
    with pytest.raises(DocumentNotFound):
        store.meta(old)


def test_upload_reads_pages_and_corrections_clear_flags(tmp_path):
    app_settings = replace(settings, uploads_dir=tmp_path, document_max_pages=5)
    ocr = FakeOcr([{"index": 0, "label": "Text", "bbox": [0, 0, 1, 1], "confidence": 0.97, "text": "FIR No. 313 dated 11-05-2016", "skipped": False}])
    vision = FakeVision({"document_type": "FIR", "language": "english", "handwriting_present": False, "handwritten_items": []},
                        transcription="FIR No. 313 dated 11-05-2018")
    service = DocumentService(app_settings, ocr=ocr, vision=vision)
    scanned = _pdf(tmp_path / "scan.pdf", ["x"]).read_bytes()
    meta = service.upload("scan.pdf", scanned)
    assert meta["status"] == "processing" and meta["page_count"] == 1

    deadline = time.monotonic() + 30
    while service.store.meta(meta["id"])["status"] == "processing" and time.monotonic() < deadline:
        time.sleep(0.05)
    document = service.document(meta["id"])
    assert document["status"] == "ready"
    block = document["pages"][0]["blocks"][0]
    assert block["uncertain"] == ["11-05-2016"]

    page = service.correct_page(meta["id"], 1, {0: "FIR No. 313 dated 11-05-2018"})
    assert page["blocks"][0]["edited"] and page["blocks"][0]["uncertain"] == []
    assert page["blocks"][0]["original_text"] == "FIR No. 313 dated 11-05-2016"


def test_upload_rejects_unsupported_files(tmp_path):
    service = DocumentService(replace(settings, uploads_dir=tmp_path), ocr=FakeOcr([]), vision=FakeVision({}))
    with pytest.raises(UnsupportedDocument):
        service.upload("notes.docx", b"data")
    with pytest.raises(UnsupportedDocument):
        service.upload("empty.pdf", b"")
