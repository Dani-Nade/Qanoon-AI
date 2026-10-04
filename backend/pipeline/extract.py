"""
Step 1: Extract raw text from all PDFs in the pdfs/ directory.
Outputs one JSON file per PDF into data/extracted/.

JSON structure per file:
{
  "filename": "II of 1947 - Prevention of Corruption Act, 1947.pdf",
  "year": "1947",
  "title": "II of 1947 - Prevention of Corruption Act, 1947",
  "num_pages": 6,
  "pages": [
    {"page_num": 1, "text": "..."},
    ...
  ],
  "full_text": "combined text of all pages"
}
"""

import json
from pathlib import Path

import pdfplumber
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDFS_DIR = PROJECT_ROOT / "pdfs"
OUT_DIR = PROJECT_ROOT / "data" / "extracted"


def extract_pdf(pdf_path: Path) -> dict:
    year = pdf_path.parent.name
    title = pdf_path.stem
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page_num": i, "text": text})

    full_text = "\n\n".join(p["text"] for p in pages if p["text"].strip())

    return {
        "filename": pdf_path.name,
        "year": year,
        "title": title,
        "num_pages": len(pages),
        "pages": pages,
        "full_text": full_text,
    }


def run(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDFS_DIR.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs")

    skipped = 0
    errors = []

    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        out_path = OUT_DIR / f"{pdf_path.stem}.json"

        if out_path.exists() and not force:
            skipped += 1
            continue

        try:
            data = extract_pdf(pdf_path)
            if not data["full_text"].strip():
                errors.append((str(pdf_path), "empty text"))
                continue
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            errors.append((str(pdf_path), str(e)))

    total = len(pdf_files)
    success = total - len(errors) - skipped
    print(f"\nExtraction complete:")
    print(f"  Extracted : {success}")
    print(f"  Skipped   : {skipped} (already exist)")
    print(f"  Errors    : {len(errors)}")

    if errors:
        print("\nFailed files:")
        for path, reason in errors:
            print(f"  {Path(path).name}: {reason}")


if __name__ == "__main__":
    run()
