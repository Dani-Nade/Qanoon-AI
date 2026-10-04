"""Head-to-head, OCR side: read the test pages with Surya OCR 2 through a local llama-server.

Run with the OCR environment: ocr/.venv/Scripts/python.exe ocr/bench/surya_run.py
Writes data/ocr-benchmark/test6/results/surya/<page>.json with text, blocks (position,
label, confidence) and timing.
"""

import html
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "data" / "ocr-benchmark" / "test6"
OUT = TEST / "results" / "surya"

# Keep models off the nearly full C: drive and use the downloaded CUDA build of llama-server.
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("MODEL_CACHE_DIR", str(ROOT / "data" / "cache" / "ocr-models"))
server = next((ROOT / "ocr" / "llama.cpp").rglob("llama-server.exe"), None)
if server:
    os.environ.setdefault("LLAMA_CPP_BINARY", str(server))
os.environ.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")

from PIL import Image  # noqa: E402
from surya.inference import SuryaInferenceManager  # noqa: E402
from surya.recognition import RecognitionPredictor  # noqa: E402

TAG = re.compile(r"<[^>]+>")
BREAK = re.compile(r"<\s*(br|/p|/div|/li|/tr|/h\d)\s*/?>", re.IGNORECASE)


def block_text(markup: str) -> str:
    return html.unescape(TAG.sub(" ", BREAK.sub("\n", markup))).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manager = SuryaInferenceManager(method="llamacpp", lazy=False)
    predictor = RecognitionPredictor(manager)
    try:
        for entry in json.loads((TEST / "manifest.json").read_text(encoding="utf-8")):
            image = Image.open(TEST / entry["image"]).convert("RGB")
            started = time.monotonic()
            page = predictor([image], full_page=True)[0]
            seconds = round(time.monotonic() - started, 1)
            blocks = [
                {"label": b.label, "reading_order": b.reading_order, "polygon": b.polygon,
                 "confidence": getattr(b, "confidence", None), "text": block_text(b.html),
                 "skipped": b.skipped, "error": b.error}
                for b in sorted(page.blocks, key=lambda b: b.reading_order)
            ]
            text = "\n".join(b["text"] for b in blocks if b["text"])
            result = {"page": entry["name"], "engine": "surya-ocr-2", "text": text, "blocks": blocks, "seconds": seconds}
            (OUT / f"{entry['name']}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{entry['name']}: {seconds}s, {len(blocks)} blocks, {len(text)} chars", flush=True)
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
