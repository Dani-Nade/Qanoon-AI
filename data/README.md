# Generated Data

`datasets/` is the only legal source of truth. This `data/` directory contains
derived, reproducible artifacts only:

```text
index/       Local development indexes
reports/     Ingestion, indexing, and evaluation reports
cache/       Downloaded model and extraction caches
registry/    Legacy/generated registries
training/    Legacy training material pending formal validation
```

The production builder reads `datasets/manifests/canonical_manifest.jsonl`
directly. It never reads `data/training/qa_pairs.jsonl` or old registries.
