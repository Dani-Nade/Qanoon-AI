"""Measure retrieval alone: is an expected section among the top results for each eval question?

No model or GPU is needed, so retrieval changes can be measured in seconds.

Usage: python evals/run_retrieval_checks.py [--top-k 5] [--show-misses]
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).parent))

from qanoon_ai.core.config import settings  # noqa: E402
from qanoon_ai.language.detection import build_retrieval_queries, detect_language  # noqa: E402
from qanoon_ai.retrieval.sqlite import SQLiteFTSRetriever  # noqa: E402
from qanoon_ai.retrieval.vectors import VectorSearcher  # noqa: E402


from run_prompt_checks import cites_expected  # noqa: E402


def retrieve(retriever, question, jurisdiction, top_k):
    # Mirrors LegalAnswerService.answer: merge results across normalized queries.
    merged = {}
    for query in build_retrieval_queries(question, detect_language(question, "auto")) or [""]:
        for chunk in retriever.search(query, top_k=max(top_k, settings.retrieval_top_k), jurisdiction=jurisdiction, semantic_query=question):
            if chunk.chunk_id not in merged or (merged[chunk.chunk_id].score or 0) < (chunk.score or 0):
                merged[chunk.chunk_id] = chunk
    return sorted(merged.values(), key=lambda chunk: chunk.score or 0, reverse=True)[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument("--keyword-only", action="store_true", help="ignore the vector index")
    args = parser.parse_args()

    cases = json.loads((Path(__file__).parent / "prompt_cases.json").read_text(encoding="utf-8"))
    vectors = None if args.keyword_only else VectorSearcher(
        settings.search_index_file, settings.vector_dir, settings.model_cache_dir,
        settings.embedding_model, settings.embedding_revision,
    )
    retriever = SQLiteFTSRetriever(settings.search_index_file, vectors)
    by_category = defaultdict(lambda: [0, 0])
    empty_declines = 0
    declines = 0
    started = time.monotonic()
    for case in cases:
        chunks = retrieve(retriever, case["question"], case.get("jurisdiction"), args.top_k)
        if case.get("expect") == "decline":
            declines += 1
            empty_declines += not chunks
            continue
        if not case.get("sections"):
            continue
        hit = any(cites_expected(case, chunk.section_ref, chunk.title) for chunk in chunks)
        by_category[case["category"]][0] += hit
        by_category[case["category"]][1] += 1
        if args.show_misses and not hit:
            got = "; ".join(f"{chunk.title[:28]} {chunk.section_ref}" for chunk in chunks) or "nothing"
            print(f"MISS {case['id']:<38} want {case['sections']} got {got}")
    hits = sum(h for h, _ in by_category.values())
    total = sum(t for _, t in by_category.values())
    for name, (h, t) in by_category.items():
        print(f"  {name:<24} {h}/{t}")
    print(f"HIT@{args.top_k} {hits}/{total}   declines with no sources retrieved: {empty_declines}/{declines}   ({time.monotonic() - started:.1f}s)")


if __name__ == "__main__":
    main()
