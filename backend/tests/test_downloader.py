from __future__ import annotations

from qanoon_ai.ingestion import downloader
from qanoon_ai.ingestion.source_catalog import OfficialSource


def test_discovery_supports_post_data_src_filters_and_rewrites(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        body = b"""
        <a data-src="https://files.example.gov/judgments/one.pdf">One</a>
        <a href="https://files.example.gov/notices/tender.pdf">Tender</a>
        """
        return body, "text/html", url

    monkeypatch.setattr(downloader, "_request", fake_request)
    source = OfficialSource(
        source_id="court_test",
        name="Court Test",
        jurisdiction="court_test",
        homepage="https://example.gov/judgments",
        seed_urls=("https://example.gov/judgments",),
        allowed_domains=("example.gov", "files.example.gov"),
        document_scope=("reported_judgments",),
        seed_method="POST",
        seed_form_data=(("year", "0"),),
        document_url_patterns=(r"/judgments/.*\.pdf$",),
        url_rewrites=(("https://files.example.gov/", "http://files.example.gov/"),),
    )

    urls, errors = downloader.discover_document_links(source)

    assert errors == []
    assert urls == ["http://files.example.gov/judgments/one.pdf"]
    assert calls[0][1]["method"] == "POST"
    assert calls[0][1]["form_data"] == (("year", "0"),)


def test_document_signature_validation_rejects_html_pdf_response():
    assert downloader._has_valid_document_signature("https://example.gov/law.pdf", b"%PDF-1.7")
    assert not downloader._has_valid_document_signature(
        "https://example.gov/law.pdf", b"<html>Access denied</html>"
    )
    assert not downloader._has_valid_document_signature(
        "https://example.gov/law.pdf", b"<html>Error</html>\n%PDF-1.7"
    )


def test_url_hash_filename_strategy_is_deterministic():
    source = OfficialSource(
        source_id="test",
        name="Test",
        jurisdiction="test",
        homepage="https://example.gov",
        seed_urls=("https://example.gov",),
        allowed_domains=("example.gov",),
        document_scope=("acts",),
        filename_strategy="url_hash_prefix",
    )

    first = downloader._target_filename(source, "https://example.gov/files/law.pdf", "application/pdf")
    second = downloader._target_filename(source, "https://example.gov/files/law.pdf", "application/pdf")

    assert first == second
    assert first.endswith("__law.pdf")


def test_request_encodes_spaces_after_url_semicolon(monkeypatch):
    requested_urls = []

    class Response:
        headers = {"Content-Type": "application/pdf"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b"%PDF-1.7"

        def geturl(self):
            return requested_urls[0]

    def fake_urlopen(request, **kwargs):
        requested_urls.append(request.full_url)
        return Response()

    monkeypatch.setattr(downloader, "urlopen", fake_urlopen)

    downloader._request(
        "https://example.gov/Judgments/Appeal; No. 31 of 1997.pdf",
        timeout=1,
    )

    assert "%20No.%2031%20of%201997.pdf" in requested_urls[0]


def test_canonical_url_treats_encoded_and_raw_spaces_as_same_document():
    raw = "https://EXAMPLE.gov/Judgments/Appeal No. 1.pdf"
    encoded = "https://example.gov/Judgments/Appeal%20No.%201.pdf"

    assert downloader._canonical_url(raw) == downloader._canonical_url(encoded)
