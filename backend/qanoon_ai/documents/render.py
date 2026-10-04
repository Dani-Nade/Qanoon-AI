"""Turn an uploaded PDF or image into page images, and read an existing PDF text layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

DPI = 300
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
# A page this short has no usable text layer (scans often carry a few stray characters).
MIN_TEXT_LAYER_CHARS = 200


class UnsupportedDocument(ValueError):
    pass


@dataclass(frozen=True)
class RenderedPage:
    number: int
    image_path: Path
    width: int
    height: int
    text_blocks: list[dict]  # from the PDF text layer, empty when the page needs OCR


def page_count(path: Path) -> int:
    try:
        with fitz.open(path) as document:
            return len(document)
    except Exception as error:
        raise UnsupportedDocument(f"The file could not be opened: {error}") from error


def render_pages(path: Path, output_dir: Path, max_pages: int) -> list[RenderedPage]:
    """Render every page at 300 DPI. Images open as single-page documents."""
    pages: list[RenderedPage] = []
    try:
        document = fitz.open(path)
    except Exception as error:
        raise UnsupportedDocument(f"The file could not be opened: {error}") from error
    with document:
        if len(document) > max_pages:
            raise UnsupportedDocument(f"The document has {len(document)} pages; the limit is {max_pages}.")
        is_pdf = document.is_pdf
        for index, page in enumerate(document, start=1):
            # Images carry their own resolution; PDFs are scaled to 300 DPI.
            matrix = fitz.Matrix(DPI / 72, DPI / 72) if is_pdf else fitz.Matrix(1, 1)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"{index}.png"
            pixmap.save(image_path)
            text_blocks = _text_layer(page, DPI / 72) if is_pdf else []
            pages.append(RenderedPage(index, image_path, pixmap.width, pixmap.height, text_blocks))
    return pages


def _text_layer(page: "fitz.Page", scale: float) -> list[dict]:
    blocks = []
    for x0, y0, x1, y1, text, _number, kind in page.get_text("blocks"):
        text = "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())
        if kind == 0 and text:
            blocks.append({"bbox": [round(x0 * scale, 1), round(y0 * scale, 1), round(x1 * scale, 1), round(y1 * scale, 1)],
                           "text": text})
    if sum(len(b["text"]) for b in blocks) < MIN_TEXT_LAYER_CHARS:
        return []
    return blocks
