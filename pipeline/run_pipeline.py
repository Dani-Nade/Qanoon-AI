"""
Orchestrator — runs all pipeline steps in order.

Usage:
  python pipeline/run_pipeline.py            # run all steps
  python pipeline/run_pipeline.py --force    # re-run even if outputs exist
  python pipeline/run_pipeline.py --step extract
  python pipeline/run_pipeline.py --step clean
  python pipeline/run_pipeline.py --step chunk
  python pipeline/run_pipeline.py --step embed
  python pipeline/run_pipeline.py --search "punishment for theft"
"""

import argparse
import json
import time
from pathlib import Path


def _header(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def run_extract(force: bool) -> None:
    _header("STEP 1 — PDF Extraction")
    from pipeline.extract import run
    run(force=force)


def run_clean(force: bool) -> None:
    _header("STEP 2 — Text Cleaning")
    from pipeline.clean import run
    run(force=force)


def run_chunk(force: bool) -> None:
    _header("STEP 3 — Chunking")
    from pipeline.chunk import run
    run(force=force)


def run_embed(force: bool) -> None:
    _header("STEP 4 — Embedding + FAISS")
    from pipeline.embed import run
    run(force=force)


def run_search(query: str, top_k: int = 5) -> None:
    """Quick sanity-check search against the built FAISS index."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    _header(f"SEARCH: \"{query}\"")

    index_path = Path(__file__).parent.parent / "vector_db" / "faiss.index"
    meta_path = Path(__file__).parent.parent / "vector_db" / "metadata.jsonl"
    chunks_path = Path(__file__).parent.parent / "data" / "chunks" / "chunks.jsonl"

    if not index_path.exists():
        print("FAISS index not found. Run the full pipeline first.")
        return

    index = faiss.read_index(str(index_path))

    metadata = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))

    # Build a quick chunk_id -> text lookup
    chunk_texts = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            chunk_texts[r["chunk_id"]] = r["text"]

    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)

    distances, indices = index.search(vec, top_k)

    print(f"\nTop {top_k} results:\n")
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        meta = metadata[idx]
        text = chunk_texts.get(meta["chunk_id"], "")[:300]
        print(f"[{rank}] Score: {score:.4f}")
        print(f"    Source : {meta['source_file']} (Year: {meta['year']})")
        print(f"    Chunk  : {meta['chunk_id']}")
        print(f"    Text   : {text}...")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="QanoonAI data pipeline")
    parser.add_argument(
        "--step",
        choices=["extract", "clean", "chunk", "embed"],
        help="Run only a specific step",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run step even if output already exists",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Test a search query against the built FAISS index",
    )
    args = parser.parse_args()

    if args.search:
        run_search(args.search)
        return

    start = time.time()

    steps = {
        "extract": run_extract,
        "clean": run_clean,
        "chunk": run_chunk,
        "embed": run_embed,
    }

    if args.step:
        steps[args.step](args.force)
    else:
        for fn in steps.values():
            fn(args.force)

    elapsed = time.time() - start
    _header(f"PIPELINE COMPLETE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
