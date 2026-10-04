"""Head-to-head, vision-model side: transcribe the test pages with qwen3.5:9b through Ollama.

Writes data/ocr-benchmark/test6/results/qwen/<page>.json with the transcription, a
document-type / handwriting judgement, and timings.
"""

import base64
import json
from pathlib import Path
import time

import fitz
import requests

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "data" / "ocr-benchmark" / "test6"
OUT = TEST / "results" / "qwen"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.5:9b"

TRANSCRIBE = """Transcribe ALL text on this scanned Pakistani legal document page exactly as it appears.
- Keep the original language and script (English stays English, Urdu stays in Urdu script).
- Copy every number, date, case number and section reference character for character. Never correct or complete them.
- Put text that is handwritten (not printed) inside [handwritten: ...].
- If a word or number cannot be read with certainty, write [illegible] instead of guessing.
- Keep one line of output per printed line, in reading order. Output only the transcription."""

CLASSIFY = """Look at this Pakistani legal document page and return ONLY a JSON object:
{"document_type": short description, "language": "english" or "urdu" or "mixed",
 "handwriting_present": true or false (any handwritten text, numbers or notes, excluding signatures alone),
 "handwritten_items": [short descriptions of each handwritten item and where it is]}"""


def page_image(path: Path) -> str:
    # Halve the 300 DPI render (about 1240 px wide) to stay within the vision encoder's budget.
    pixmap = fitz.Pixmap(str(path))
    pixmap.shrink(1)
    return base64.b64encode(pixmap.tobytes("png")).decode()


def ask(prompt: str, image: str, json_output: bool = False) -> tuple[str, float]:
    started = time.monotonic()
    payload = {
        "model": MODEL, "stream": False, "think": False,
        "messages": [{"role": "user", "content": prompt, "images": [image]}],
        "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 4096},
    }
    if json_output:
        payload["format"] = "json"
    response = requests.post(OLLAMA, json=payload, timeout=1800)
    response.raise_for_status()
    return response.json()["message"]["content"], round(time.monotonic() - started, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for entry in json.loads((TEST / "manifest.json").read_text(encoding="utf-8")):
        image = page_image(TEST / entry["image"])
        text, transcribe_s = ask(TRANSCRIBE, image)
        raw, classify_s = ask(CLASSIFY, image, json_output=True)
        try:
            # The model sometimes wraps JSON in a ``` fence even when JSON output is requested.
            judgement = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
        except ValueError:
            judgement = {"unparsed": raw}
        result = {"page": entry["name"], "engine": MODEL, "text": text, "judgement": judgement,
                  "seconds": {"transcribe": transcribe_s, "classify": classify_s}}
        (OUT / f"{entry['name']}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{entry['name']}: {transcribe_s}s transcribe, {classify_s}s classify, {len(text)} chars, handwriting={judgement.get('handwriting_present')}", flush=True)


if __name__ == "__main__":
    main()
