"""
Step 3: Split cleaned documents into overlapping chunks.
Reads from data/cleaned/, writes all chunks to data/chunks/chunks.jsonl

Strategy:
- Split document into paragraphs (double-newline boundaries)
- Accumulate paragraphs until chunk reaches ~CHUNK_WORDS words
- Carry OVERLAP_WORDS words from the end of the previous chunk into the next
  so context is not lost at chunk boundaries

Each output record (one JSON per line):
{
  "chunk_id":    "1947__II_of_1947__0",
  "source_file": "II of 1947 - Prevention of Corruption Act, 1947.pdf",
  "year":        "1947",
  "title":       "II of 1947 - Prevention of Corruption Act, 1947",
  "chunk_index": 0,
  "text":        "..."
}
"""

import json
import re
from pathlib import Path

from tqdm import tqdm

IN_DIR = Path(__file__).parent.parent / "data" / "cleaned"
OUT_DIR = Path(__file__).parent.parent / "data" / "chunks"
OUT_FILE = OUT_DIR / "chunks.jsonl"

CHUNK_WORDS = 350      # target words per chunk (~450 tokens)
OVERLAP_WORDS = 50     # words carried over from previous chunk


def _word_count(text: str) -> int:
    return len(text.split())


def _split_paragraphs(text: str) -> list[str]:
    paras = re.split(r"\n\n+", text)
    return [p.strip() for p in paras if p.strip()]


def chunk_document(data: dict) -> list[dict]:
    paragraphs = _split_paragraphs(data["full_text"])
    chunks = []
    current_words: list[str] = []
    chunk_index = 0

    for para in paragraphs:
        para_words = para.split()
        current_words.extend(para_words)

        if _word_count(" ".join(current_words)) >= CHUNK_WORDS:
            chunk_text = " ".join(current_words)
            slug = re.sub(r"[^\w]+", "_", data["title"])[:60]
            chunks.append({
                "chunk_id": f"{data['year']}__{slug}__{chunk_index}",
                "source_file": data["filename"],
                "year": data["year"],
                "title": data["title"],
                "chunk_index": chunk_index,
                "text": chunk_text,
            })
            # carry overlap into next chunk
            current_words = current_words[-OVERLAP_WORDS:]
            chunk_index += 1

    # flush remaining words as the final chunk
    if current_words:
        remaining = " ".join(current_words)
        if _word_count(remaining) > 20:   # skip tiny trailing fragments
            slug = re.sub(r"[^\w]+", "_", data["title"])[:60]
            chunks.append({
                "chunk_id": f"{data['year']}__{slug}__{chunk_index}",
                "source_file": data["filename"],
                "year": data["year"],
                "title": data["title"],
                "chunk_index": chunk_index,
                "text": remaining,
            })

    return chunks


def run(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_FILE.exists() and not force:
        print(f"Chunks file already exists at {OUT_FILE}")
        print("Pass force=True or delete the file to re-chunk.")
        return

    json_files = sorted(IN_DIR.glob("*.json"))
    if not json_files:
        print("No cleaned files found. Run clean.py first.")
        return

    print(f"Chunking {len(json_files)} documents...")

    total_chunks = 0
    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        for in_path in tqdm(json_files, desc="Chunking"):
            with open(in_path, encoding="utf-8") as f:
                data = json.load(f)

            chunks = chunk_document(data)
            for chunk in chunks:
                out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)

    print(f"\nChunking complete:")
    print(f"  Documents : {len(json_files)}")
    print(f"  Chunks    : {total_chunks}")
    print(f"  Output    : {OUT_FILE}")


if __name__ == "__main__":
    run()
