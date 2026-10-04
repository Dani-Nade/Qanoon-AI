from __future__ import annotations

import json

from qanoon_ai.ingestion.source_catalog import load_source_catalog


def test_source_catalog_loads_and_filters(tmp_path):
    catalog_file = tmp_path / "official_sources.json"
    catalog_file.write_text(
        json.dumps(
            {
                "version": "test",
                "sources": [
                    {
                        "source_id": "punjab_test",
                        "name": "Punjab Test",
                        "jurisdiction": "punjab",
                        "homepage": "https://example.com",
                        "seed_urls": ["https://example.com/laws"],
                        "allowed_domains": ["example.com"],
                        "document_scope": ["acts"],
                    },
                    {
                        "source_id": "sindh_test",
                        "name": "Sindh Test",
                        "jurisdiction": "sindh",
                        "homepage": "https://example.org",
                        "seed_urls": ["https://example.org/laws"],
                        "allowed_domains": ["example.org"],
                        "document_scope": ["rules"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    catalog = load_source_catalog(catalog_file)

    assert catalog.version == "test"
    assert len(catalog.sources) == 2
    assert [source.source_id for source in catalog.select(jurisdictions={"punjab"})] == [
        "punjab_test"
    ]
    assert [source.jurisdiction for source in catalog.select(source_ids={"sindh_test"})] == [
        "sindh"
    ]
