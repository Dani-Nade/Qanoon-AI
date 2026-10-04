"""Grade engine transcriptions against the key fields in reference.json.

Usage: python ocr/bench/grade.py            (compares every engine found under results/)
"""

import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "data" / "ocr-benchmark" / "test6"

DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
LETTERS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ة": "ہ", "ۃ": "ہ", "ـ": None})
MARKS = re.compile(r"[ً-ٰٟ‌‍‏‎]")


def normalize(text: str) -> str:
    """Remove differences that do not change what the text says: case, spacing, digit and letter forms."""
    text = unicodedata.normalize("NFKC", text).translate(DIGITS).translate(LETTERS)
    text = MARKS.sub("", text).casefold()
    return re.sub(r"\s+", "", text)


def main() -> None:
    reference = json.loads((TEST / "reference.json").read_text(encoding="utf-8"))
    engines = sorted(p.name for p in (TEST / "results").iterdir() if p.is_dir())
    totals = {engine: {"found": 0, "fields": 0, "hw_correct": 0, "pages": 0, "seconds": 0.0} for engine in engines}
    for page, expected in reference.items():
        if page.startswith("_"):
            continue
        print(f"\n== {page}")
        for engine in engines:
            path = TEST / "results" / engine / f"{page}.json"
            if not path.exists():
                print(f"  {engine:<8} (no result)")
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            haystack = normalize(result["text"])
            missing = [field for field in expected["printed"] if normalize(field) not in haystack]
            found = len(expected["printed"]) - len(missing)
            flagged = result.get("handwriting_detected")
            judgement = result.get("judgement")
            if isinstance(judgement, dict) and "unparsed" in judgement:
                try:
                    judgement = result["judgement"] = json.loads(
                        judgement["unparsed"].strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
                except ValueError:
                    pass
            if flagged is None and isinstance(result.get("judgement"), dict):
                flagged = bool(result["judgement"].get("handwriting_present")) or "[handwritten" in result["text"]
            hw = "n/a" if flagged is None else ("correct" if flagged == expected["handwriting_present"] else "WRONG")
            seconds = sum(result.get("seconds", {}).values()) if isinstance(result.get("seconds"), dict) else result.get("seconds", 0)
            t = totals[engine]
            t["found"] += found
            t["fields"] += len(expected["printed"])
            t["pages"] += 1
            t["seconds"] += seconds or 0
            t["hw_correct"] += hw == "correct"
            print(f"  {engine:<8} fields {found}/{len(expected['printed'])}  handwriting {hw:<7} {seconds or 0:.0f}s  missing: {missing}")
    print("\n== summary")
    for engine, t in totals.items():
        if t["pages"]:
            print(f"  {engine:<8} key fields {t['found']}/{t['fields']} ({100 * t['found'] / t['fields']:.0f}%)  "
                  f"handwriting judged right on {t['hw_correct']}/{t['pages']} pages  total {t['seconds']:.0f}s")


if __name__ == "__main__":
    main()
