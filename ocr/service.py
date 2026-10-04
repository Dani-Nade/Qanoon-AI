"""Local OCR service: Surya OCR 2 behind a small HTTP API.

Runs in its own environment (ocr/.venv) so Surya's pinned dependencies never touch the
backend. Start with:

    ocr/.venv/Scripts/python.exe -m uvicorn service:app --app-dir ocr --host 127.0.0.1 --port 8010

POST /ocr/page  (multipart "image": a PNG or JPEG of one page)
  -> {"width", "height", "seconds", "blocks": [{"index", "label", "bbox": [x0, y0, x1, y1],
      "confidence", "text", "skipped"}]}
"""

from __future__ import annotations

import html
import io
import os
from pathlib import Path
import re
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
# Models and caches stay on the project drive; C: has little free space.
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("MODEL_CACHE_DIR", str(ROOT / "data" / "cache" / "ocr-models"))
os.environ.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")
os.environ.setdefault("DISABLE_TQDM", "true")
# One page at a time keeps GPU memory small next to the chat model.
os.environ.setdefault("SURYA_INFERENCE_PARALLEL", "1")
_server = next((ROOT / "ocr" / "llama.cpp").rglob("llama-server.exe"), None)
if _server:
    os.environ.setdefault("LLAMA_CPP_BINARY", str(_server))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

TAG = re.compile(r"<[^>]+>")
BREAK = re.compile(r"<\s*(br|/p|/div|/li|/tr|/h\d)\s*/?>", re.IGNORECASE)
MAX_IMAGE_BYTES = 40 * 1024 * 1024

app = FastAPI(title="Qanoon AI OCR", version="0.1.0")
_lock = threading.Lock()
_state: dict = {"manager": None, "predictor": None, "error": None}


def block_text(markup: str) -> str:
    text = html.unescape(TAG.sub(" ", BREAK.sub("\n", markup)))
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def predictor():
    """Start llama-server and Surya once, on first use."""
    with _lock:
        if _state["predictor"] is None:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor

            manager = SuryaInferenceManager(method="llamacpp", lazy=False)
            _state["manager"], _state["predictor"] = manager, RecognitionPredictor(manager)
        return _state["predictor"]


@app.on_event("shutdown")
def shutdown() -> None:
    if _state["manager"] is not None:
        _state["manager"].stop()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": "surya-ocr-2", "loaded": _state["predictor"] is not None,
            "llama_server": str(_server) if _server else None}


@app.post("/ocr/page")
async def ocr_page(image: UploadFile = File(...)) -> dict:
    data = await image.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Page image is larger than 40 MB")
    try:
        page_image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "The file is not a readable image")
    started = time.monotonic()
    ocr = predictor()
    with _lock:  # llama-server is started with one slot
        page = ocr([page_image], full_page=True)[0]
    blocks = []
    for index, block in enumerate(sorted(page.blocks, key=lambda b: b.reading_order)):
        xs = [point[0] for point in block.polygon]
        ys = [point[1] for point in block.polygon]
        blocks.append({
            "index": index,
            "label": block.label,
            "bbox": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)],
            "confidence": round(float(block.confidence), 4) if getattr(block, "confidence", None) is not None else None,
            "text": block_text(block.html),
            "skipped": bool(block.skipped),
        })
    return {"width": page_image.width, "height": page_image.height,
            "seconds": round(time.monotonic() - started, 2), "blocks": blocks}
