"""Read one page: Surya text, a vision-model check for handwriting, and a number cross-check.

Surya OCR supplies the page text with block positions. The chat model (qwen3.5:9b) looks
at the same image to judge document type, language and handwriting, because Surya reads
handwritten text as if it were printed. On English pages the vision model also transcribes
the page so that dates and case numbers Surya read can be cross-checked; that transcription
is not used on Urdu pages, where it proved unreliable, and it is never used as page text.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
import re
import unicodedata

import fitz
import requests

LOGGER = logging.getLogger(__name__)

PAGE_CHECK_PROMPT = """Look at this page of a Pakistani legal document and return ONLY a JSON object:
{"document_type": "short description, e.g. FIR, bail petition, court order, judgment, legal notice, nikahnama, agreement, gazette",
 "language": "english" or "urdu" or "mixed",
 "handwriting_present": true or false (handwritten words, numbers or notes; ignore signatures and stamps),
 "handwritten_items": [{"text": "the handwritten characters exactly as you see them, or empty if unreadable", "location": "where on the page"}]}"""

TRANSCRIBE_PROMPT = """Transcribe all printed and typed text on this page exactly as it appears, line by line.
Copy every number, date and case number character for character. Never correct or complete them.
Write [illegible] for anything you cannot read with certainty. Output only the transcription."""

NUMBER_RE = re.compile(r"\d[\d./\-]*\d")
DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
# Surya annotates handwriting inline, e.g. "[Handwritten: 353-B]". Signatures are not handwriting to review.
HANDWRITTEN_MARK = re.compile(r"\[handwritten(?![^\]]*signature)[^\]]*\]", re.IGNORECASE)
MATCH_THRESHOLD = 0.7


def _closest_window(needle: str, haystack: str) -> tuple[float, int]:
    """Best similarity of the needle against same-length windows of the haystack, and where."""
    from difflib import SequenceMatcher

    if not needle or not haystack:
        return 0.0, -1
    size = len(needle)
    best, where = 0.0, -1
    for start in range(0, max(1, len(haystack) - size + 1)):
        ratio = SequenceMatcher(None, needle, haystack[start:start + size]).ratio()
        if ratio > best:
            best, where = ratio, start
    return best, where


def _compact(text: str) -> str:
    """Letters and digits only, lower case: tolerant of spacing and punctuation differences."""
    text = unicodedata.normalize("NFKC", text).translate(DIGITS).casefold()
    return "".join(ch for ch in text if ch.isalnum())


def _strip_fence(raw: str) -> str:
    return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


class OcrClient:
    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=3).ok
        except requests.RequestException:
            return False

    def read_page(self, image_path: Path) -> dict:
        with image_path.open("rb") as handle:
            response = requests.post(f"{self.base_url}/ocr/page", files={"image": (image_path.name, handle, "image/png")},
                                     timeout=self.timeout)
        response.raise_for_status()
        return response.json()


class VisionChecker:
    """Asks the local vision-capable chat model about a page image."""

    def __init__(self, ollama_url: str, model: str, num_ctx: int, timeout: float = 600.0):
        self.url = ollama_url.rstrip("/") + "/api/chat"
        self.model = model
        self.num_ctx = num_ctx  # same as chat, so Ollama does not reload the model between uses
        self.timeout = timeout

    @staticmethod
    def _image(image_path: Path) -> str:
        pixmap = fitz.Pixmap(str(image_path))
        while pixmap.width > 1400:
            pixmap.shrink(1)
        return base64.b64encode(pixmap.tobytes("png")).decode()

    def _ask(self, prompt: str, image: str, json_output: bool, max_tokens: int) -> str:
        payload = {"model": self.model, "stream": False, "think": False,
                   "messages": [{"role": "user", "content": prompt, "images": [image]}],
                   "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": max_tokens}}
        if json_output:
            payload["format"] = "json"
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def check_page(self, image_path: Path) -> dict:
        raw = self._ask(PAGE_CHECK_PROMPT, self._image(image_path), json_output=True, max_tokens=800)
        value = json.loads(_strip_fence(raw))
        items = [item for item in value.get("handwritten_items") or [] if isinstance(item, dict)]
        return {
            "document_type": str(value.get("document_type") or "").strip() or None,
            "language": value.get("language") if value.get("language") in {"english", "urdu", "mixed"} else None,
            "handwriting_present": value.get("handwriting_present") is True,
            "handwritten_items": [{"text": str(i.get("text") or "").strip(), "location": str(i.get("location") or "").strip()} for i in items],
        }

    def transcribe(self, image_path: Path) -> str:
        return self._ask(TRANSCRIBE_PROMPT, self._image(image_path), json_output=False, max_tokens=4096)


def mark_handwriting(blocks: list[dict], check: dict) -> list[str]:
    """Mark blocks containing handwritten items; return notes for items that could not be located."""
    notes = []
    for block in blocks:
        if HANDWRITTEN_MARK.search(block["text"]):
            block["handwritten"] = True
    for item in check.get("handwritten_items", []):
        needle = _compact(item["text"])
        located = False
        if len(needle) >= 3:
            for block in blocks:
                if needle in _compact(block["text"]):
                    block["handwritten"] = True
                    block["uncertain"].append(item["text"])
                    located = True
            if not located:
                # OCR often misreads handwriting ("353-B" as "383-B"); find the closest block instead.
                scored = [(_closest_window(needle, _compact(block["text"]))[0], block) for block in blocks]
                score, block = max(scored, key=lambda pair: pair[0], default=(0.0, None))
                if block is not None and score >= MATCH_THRESHOLD:
                    block["handwritten"] = True
                    located = True
        if not located:
            where = f" ({item['location']})" if item["location"] else ""
            notes.append(f"Handwriting detected{where}" + (f": \"{item['text']}\"" if item["text"] else "") +
                         ". It is not read reliably; check it on the page image.")
    if check.get("handwriting_present") and not notes and not any(b.get("handwritten") for b in blocks):
        notes.append("Handwriting was detected on this page but could not be located in the text; check the page image.")
    return notes


def cross_check_numbers(blocks: list[dict], transcription: str) -> int:
    """Flag dates and case numbers Surya read that the independent transcription does not contain."""
    reference = _compact(transcription)
    flagged = 0
    for block in blocks:
        for number in set(NUMBER_RE.findall(block["text"].translate(DIGITS))):
            if _compact(number) not in reference and number not in block["uncertain"]:
                block["uncertain"].append(number)
                flagged += 1
    return flagged


def read_page(image_path: Path, text_blocks: list[dict], ocr: OcrClient, vision: VisionChecker | None) -> dict:
    """Produce the stored page record: blocks with text, positions, and review flags."""
    warnings: list[str] = []
    if text_blocks:
        method, blocks = "text-layer", [
            {"index": i, "label": "Text", "bbox": b["bbox"], "confidence": None, "text": b["text"]} for i, b in enumerate(text_blocks)
        ]
    else:
        result = ocr.read_page(image_path)
        method = "ocr"
        blocks = [b for b in result["blocks"] if not b.get("skipped")]
        for i, block in enumerate(blocks):
            block["index"] = i
    for block in blocks:
        block.update({"handwritten": False, "uncertain": [], "edited": False, "original_text": block["text"]})

    check = {"document_type": None, "language": None, "handwriting_present": False, "handwritten_items": []}
    if vision is not None:
        try:
            check = vision.check_page(image_path)
            warnings += mark_handwriting(blocks, check)
        except Exception:
            LOGGER.exception("Vision page check failed for %s", image_path)
            warnings.append("The handwriting check could not run on this page; handwritten parts may not be marked.")
        if check["language"] == "english" and method == "ocr":
            try:
                flagged = cross_check_numbers(blocks, vision.transcribe(image_path))
                if flagged:
                    warnings.append(f"{flagged} number(s) could not be confirmed by a second reading and are highlighted.")
            except Exception:
                LOGGER.exception("Number cross-check failed for %s", image_path)
                warnings.append("Numbers on this page could not be cross-checked.")
        elif check["language"] in {"urdu", "mixed"}:
            warnings.append("Numbers on Urdu pages are not cross-checked yet; check dates and case numbers against the page image.")
    else:
        warnings.append("The handwriting check is unavailable; handwritten parts are not marked.")

    return {
        "method": method,
        "document_type": check["document_type"],
        "language": check["language"],
        "handwriting_present": check["handwriting_present"] or any(b["handwritten"] for b in blocks),
        "handwritten_items": check["handwritten_items"],
        "blocks": blocks,
        "warnings": warnings,
    }
