"""
Step 4: Generate sentence embeddings and build a FAISS index.
Reads  : data/chunks/chunks.jsonl
Writes :
  vector_db/faiss.index     — FAISS index (inner-product on L2-normalised vectors = cosine sim)
  vector_db/metadata.jsonl  — parallel metadata for every vector (same order as index)

Model: paraphrase-multilingual-mpnet-base-v2
  - 768-dimensional embeddings
  - Supports Urdu, English, and 50+ other languages
  - ~420 MB download on first run (cached by sentence-transformers after that)

Query usage (next semester, RAG):
  1. Embed the user query with the same model
  2. faiss.index.search(query_vec, k=5) -> top-5 chunk indices
  3. Look up indices in metadata.jsonl to get the actual text
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
OUT_DIR = PROJECT_ROOT / "vector_db"
INDEX_FILE = OUT_DIR / "faiss.index"
META_FILE = OUT_DIR / "metadata.jsonl"

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 64
EMBEDDING_DIM = 768


def load_chunks(path: Path) -> tuple[list[str], list[dict]]:
    texts, metadata = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            # store everything except the full text in metadata (saves RAM)
            metadata.append({k: v for k, v in record.items() if k != "text"})
    return texts, metadata


def build_index(embeddings: np.ndarray) -> faiss.Index:
    # Normalise to unit length so inner-product == cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    return index


def run(force: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_FILE.exists() and META_FILE.exists() and not force:
        print(f"FAISS index already exists at {INDEX_FILE}")
        print("Pass force=True or delete the files to rebuild.")
        return

    if not CHUNKS_FILE.exists():
        print("Chunks file not found. Run chunk.py first.")
        return

    print("Loading chunks...")
    texts, metadata = load_chunks(CHUNKS_FILE)
    print(f"  Loaded {len(texts)} chunks")

    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"\nGenerating embeddings in batches of {BATCH_SIZE}...")
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + BATCH_SIZE]
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(vecs)

    embeddings = np.vstack(all_embeddings).astype("float32")
    print(f"  Embedding matrix shape: {embeddings.shape}")

    print("\nBuilding FAISS index...")
    index = build_index(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"  Index saved to {INDEX_FILE}")

    print("Saving metadata...")
    with open(META_FILE, "w", encoding="utf-8") as f:
        for record in metadata:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Metadata saved to {META_FILE}")

    print(f"\nFAISS index ready:")
    print(f"  Vectors  : {index.ntotal}")
    print(f"  Dimension: {EMBEDDING_DIM}")


if __name__ == "__main__":
    run()
