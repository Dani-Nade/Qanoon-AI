"""
Step 2: Clean raw extracted text.
Reads from data/extracted/, writes to data/cleaned/.

Removes:
- Page headers/footers (e.g. "Page 1 of 6", "THE GAZETTE OF PAKISTAN...")
- Repeated document title lines that appear on every page
- Excessive whitespace and blank lines
- Non-printable / control characters
Preserves:
- Section/article numbering (critical for legal text)
- Punctuation and legal terminology
"""

import json
import re
import unicodedata
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_DIR = PROJECT_ROOT / "data" / "extracted"
OUT_DIR = PROJECT_ROOT / "data" / "cleaned"

# Patterns that appear as running headers/footers in Pakistani law PDFs
_NOISE_PATTERNS = [
    re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"THE\s+GAZETTE\s+OF\s+PAKISTAN.*", re.IGNORECASE),
    re.compile(r"EXTRAORDINARY\s+GAZETTE.*", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),          # lone page numbers
    re.compile(r"\[Part\s+[IVX]+\]", re.IGNORECASE),
    re.compile(r"Islamabad,\s+the\s+\d+.*?\d{4}", re.IGNORECASE),
]


def _remove_noise(text: str) -> str:
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)
    return text


def _normalize_unicode(text: str) -> str:
    # Normalize to NFC, remove control characters except newline/tab
    text = unicodedata.normalize("NFC", text)
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in ("\n", "\t") or cat[0] != "C":
            cleaned.append(ch)
    return "".join(cleaned)


def _normalize_whitespace(text: str) -> str:
    # Collapse 3+ consecutive newlines to 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def clean_text(text: str) -> str:
    text = _normalize_unicode(text)
    text = _remove_noise(text)
    text = _normalize_whitespace(text)
    return text


def run(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(IN_DIR.glob("*.json"))
    if not json_files:
        print("No extracted files found. Run extract.py first.")
        return

    print(f"Found {len(json_files)} extracted files")
    skipped = 0

    for in_path in tqdm(json_files, desc="Cleaning text"):
        out_path = OUT_DIR / in_path.name

        if out_path.exists() and not force:
            skipped += 1
            continue

        with open(in_path, encoding="utf-8") as f:
            data = json.load(f)

        data["full_text"] = clean_text(data["full_text"])
        for page in data["pages"]:
            page["text"] = clean_text(page["text"])

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    success = len(json_files) - skipped
    print(f"\nCleaning complete:")
    print(f"  Cleaned : {success}")
    print(f"  Skipped : {skipped} (already exist)")


if __name__ == "__main__":
    run()
