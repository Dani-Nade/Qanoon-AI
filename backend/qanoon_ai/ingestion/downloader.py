"""Download official legal documents with provenance manifests."""

from __future__ import annotations

import hashlib
import html
import json
import re
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

from qanoon_ai.ingestion.source_catalog import OfficialSource

DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx"}
HTML_EXTENSIONS = {"", ".html", ".htm", ".php", ".aspx"}
USER_AGENT = "QanoonAI-DatasetBuilder/0.1 (+https://local.qanoon.ai)"
_MANIFEST_LOCK = threading.Lock()


@dataclass(frozen=True)
class DownloadManifestRecord:
    timestamp_utc: str
    source_id: str
    source_name: str
    jurisdiction: str
    document_url: str
    local_path: str | None
    status: str
    content_type: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SourceDownloadSummary:
    source_id: str
    discovered: int
    downloaded: int
    skipped_existing: int
    dry_run: int
    errors: int


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        for attr in ("href", "src", "data-href", "data-src"):
            value = attrs_map.get(attr)
            if value:
                self.links.append(value.strip())


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str, *, max_length: int = 96) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_length] or "source"


def _safe_filename(name: str) -> str:
    name = unquote(name).strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[\x00-\x1f<>:\"|?*]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180] or "document"


def _content_extension(content_type: str | None) -> str:
    if not content_type:
        return ".pdf"
    lowered = content_type.lower()
    if "pdf" in lowered:
        return ".pdf"
    if "wordprocessingml" in lowered:
        return ".docx"
    if "msword" in lowered:
        return ".doc"
    return ".pdf"


def _filename_from_url(url: str, content_type: str | None) -> str:
    parsed = urlparse(url)
    filename = Path(unquote(parsed.path)).name
    if not filename or "." not in filename:
        fingerprint = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        filename = f"document_{fingerprint}{_content_extension(content_type)}"
    return _safe_filename(filename)


def _is_allowed_url(url: str, allowed_domains: tuple[str, ...]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    for domain in allowed_domains:
        normalized = domain.lower().lstrip(".")
        if host == normalized or host.endswith(f".{normalized}"):
            return True
    return False


def _extension_from_url(url: str) -> str:
    return Path(urlparse(url).path.lower()).suffix


def _looks_like_document_url(url: str) -> bool:
    return _extension_from_url(url) in DOCUMENT_EXTENSIONS


def _looks_like_html_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    return _extension_from_url(url) in HTML_EXTENSIONS


def _request(
    url: str,
    *,
    timeout: float,
    method: str = "GET",
    form_data: tuple[tuple[str, str], ...] = (),
    verify_tls: bool = True,
) -> tuple[bytes, str | None, str]:
    parsed = urlparse(url)
    encoded_url = parsed._replace(
        path=quote(parsed.path, safe="/%"),
        params=quote(parsed.params, safe="=;,%"),
        query=quote(parsed.query, safe="=&%/:+?,"),
    ).geturl()
    encoded_form = urlencode(dict(form_data)).encode("utf-8") if form_data else None
    request = Request(
        encoded_url,
        data=encoded_form,
        method=method,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,*/*",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    context = None if verify_tls else ssl._create_unverified_context()
    with urlopen(request, timeout=timeout, context=context) as response:
        content = response.read()
        content_type = response.headers.get("Content-Type")
        final_url = response.geturl()
    return content, content_type, final_url


def _matches_patterns(url: str, patterns: tuple[str, ...]) -> bool:
    return not patterns or any(re.search(pattern, url, re.IGNORECASE) for pattern in patterns)


def _rewrite_url(url: str, rewrites: tuple[tuple[str, str], ...]) -> str:
    for source_prefix, target_prefix in rewrites:
        if url.startswith(source_prefix):
            return target_prefix + url[len(source_prefix) :]
    return url


def _canonical_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=quote(unquote(parsed.path), safe="/%"),
        params=quote(unquote(parsed.params), safe="=;,%"),
    ).geturl()


def _has_valid_document_signature(url: str, content: bytes) -> bool:
    extension = _extension_from_url(url)
    if extension == ".pdf":
        return content.startswith(b"%PDF-")
    if extension == ".docx":
        return content.startswith(b"PK\x03\x04")
    if extension == ".doc":
        return content.startswith(bytes.fromhex("D0CF11E0A1B11AE1"))
    return False


def _target_filename(source: OfficialSource, url: str, content_type: str | None) -> str:
    filename = _filename_from_url(url, content_type)
    if source.filename_strategy == "url_hash_prefix":
        fingerprint = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        return f"{fingerprint}__{filename}"
    return filename


def _is_document_response(url: str, content_type: str | None) -> bool:
    if _looks_like_document_url(url):
        return True
    if not content_type:
        return False
    lowered = content_type.lower()
    return "pdf" in lowered or "msword" in lowered or "wordprocessingml" in lowered


def _parse_links(base_url: str, html_bytes: bytes) -> list[str]:
    text = html_bytes.decode("utf-8", errors="ignore")
    link_candidates: list[str] = []

    for candidate_text in (text, html.unescape(text)):
        parser = _LinkParser()
        parser.feed(candidate_text)
        link_candidates.extend(parser.links)
        link_candidates.extend(
            re.findall(
                r"(?:https?://[^\s'\"<>]+\.pdf|Document\.aspx\?[^\s'\"<>]+|pdfviewer\.aspx\?[^\s'\"<>]+|lawdir/[^\s'\"<>]+\.pdf)",
                candidate_text,
                flags=re.IGNORECASE,
            )
        )

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_link in link_candidates:
        if raw_link.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute, _fragment = urldefrag(urljoin(base_url, raw_link))
        absolute = html.unescape(absolute).rstrip("&")
        if absolute in seen:
            continue
        seen.add(absolute)
        normalized.append(absolute)
    return normalized


def _link_priority(url: str) -> int:
    lowered = url.lower()
    direct_document_tokens = (
        "wise=opendoc",
        "lawdetails",
        "lawdir",
        "uploads/",
    )
    legal_listing_tokens = (
        "document.aspx",
        "gazettedetail",
        "lawspdf",
    )
    if _looks_like_document_url(url) or any(token in lowered for token in direct_document_tokens):
        return 0
    if any(token in lowered for token in legal_listing_tokens):
        return 1
    return 2


def discover_document_links(
    source: OfficialSource,
    *,
    max_depth: int = 1,
    max_pages: int = 50,
    timeout: float = 30.0,
) -> tuple[list[str], list[dict[str, str]]]:
    """Discover document links from a configured source without downloading them yet."""

    queue: list[tuple[str, int]] = [(url, 0) for url in source.seed_urls]
    seen_pages: set[str] = set()
    document_urls: set[str] = set()
    errors: list[dict[str, str]] = []

    while queue and len(seen_pages) < max_pages:
        page_url, depth = queue.pop(0)
        page_url, _fragment = urldefrag(page_url)
        if page_url in seen_pages:
            continue
        if not _is_allowed_url(page_url, source.allowed_domains):
            continue
        seen_pages.add(page_url)

        try:
            is_seed = depth == 0 and page_url in source.seed_urls
            body, content_type, final_url = _request(
                page_url,
                timeout=timeout,
                method=source.seed_method if is_seed else "GET",
                form_data=source.seed_form_data if is_seed else (),
                verify_tls=source.verify_tls,
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, HTTPException) as exc:
            errors.append({"url": page_url, "error": str(exc)})
            continue

        if _is_document_response(final_url, content_type):
            rewritten_url = _rewrite_url(final_url, source.url_rewrites)
            if _matches_patterns(rewritten_url, source.document_url_patterns):
                document_urls.add(rewritten_url)
            continue

        if content_type and "html" not in content_type.lower() and not _looks_like_html_url(final_url):
            continue

        for link in sorted(_parse_links(final_url, body), key=_link_priority):
            if not _is_allowed_url(link, source.allowed_domains):
                continue
            if _looks_like_document_url(link):
                rewritten_link = _rewrite_url(link, source.url_rewrites)
                if _matches_patterns(rewritten_link, source.document_url_patterns):
                    document_urls.add(rewritten_link)
            elif (
                depth < max_depth
                and _looks_like_html_url(link)
                and link not in seen_pages
                and _matches_patterns(link, source.crawl_url_patterns)
            ):
                queue.append((link, depth + 1))
        queue.sort(key=lambda item: (_link_priority(item[0]), item[1]))

    return sorted(document_urls), errors


def _write_manifest_record(manifest_file: Path, record: DownloadManifestRecord) -> None:
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with _MANIFEST_LOCK:
        with manifest_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _successful_downloads(manifest_file: Path, dataset_dir: Path, source_id: str) -> set[str]:
    successful_urls: set[str] = set()
    if not manifest_file.exists():
        return successful_urls
    with manifest_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("source_id") != source_id:
                continue
            if record.get("status") not in {"downloaded", "skipped_existing"}:
                continue
            local_path = record.get("local_path")
            if local_path and (dataset_dir / local_path).exists():
                successful_urls.add(_canonical_url(record.get("document_url", "")))
    return successful_urls


def _write_file(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def download_source(
    source: OfficialSource,
    *,
    raw_dir: Path,
    manifest_file: Path,
    limit: int | None = 25,
    dry_run: bool = False,
    force: bool = False,
    max_depth: int = 1,
    max_pages: int = 50,
    timeout: float = 30.0,
    sleep_seconds: float = 0.3,
    workers: int = 1,
) -> SourceDownloadSummary:
    """Download official documents for one source and append provenance records."""

    document_urls, discovery_errors = discover_document_links(
        source,
        max_depth=max_depth,
        max_pages=max_pages,
        timeout=timeout,
    )
    selected_urls = document_urls if limit is None else document_urls[:limit]
    target_dir = raw_dir / _slug(source.jurisdiction) / _slug(source.source_id)
    successful_urls = (
        set()
        if force
        else _successful_downloads(manifest_file, raw_dir.parent, source.source_id)
    )

    downloaded = 0
    skipped_existing = 0
    dry_run_count = 0
    error_count = len(discovery_errors)

    for error in discovery_errors:
        _write_manifest_record(
            manifest_file,
            DownloadManifestRecord(
                timestamp_utc=_now_utc(),
                source_id=source.source_id,
                source_name=source.name,
                jurisdiction=source.jurisdiction,
                document_url=error["url"],
                local_path=None,
                status="discovery_error",
                error=error["error"],
            ),
        )

    if dry_run:
        for url in selected_urls:
            dry_run_count += 1
            _write_manifest_record(
                manifest_file,
                DownloadManifestRecord(
                    timestamp_utc=_now_utc(),
                    source_id=source.source_id,
                    source_name=source.name,
                    jurisdiction=source.jurisdiction,
                    document_url=url,
                    local_path=None,
                    status="dry_run",
                ),
            )
        return SourceDownloadSummary(
            source_id=source.source_id,
            discovered=len(document_urls),
            downloaded=downloaded,
            skipped_existing=skipped_existing,
            dry_run=dry_run_count,
            errors=error_count,
        )

    def acquire(url: str) -> str:
        if _canonical_url(url) in successful_urls:
            return "skipped_existing"
        try:
            content, content_type, final_url = _request(
                url,
                timeout=timeout,
                verify_tls=source.verify_tls,
            )
            if not _has_valid_document_signature(final_url, content):
                raise ValueError("Response did not contain the expected document signature")
            filename = _target_filename(source, final_url, content_type)
            output_path = target_dir / filename
            relative_path = output_path.resolve().relative_to(raw_dir.parent.resolve())

            if output_path.exists() and not force:
                _write_manifest_record(
                    manifest_file,
                    DownloadManifestRecord(
                        timestamp_utc=_now_utc(),
                        source_id=source.source_id,
                        source_name=source.name,
                        jurisdiction=source.jurisdiction,
                        document_url=final_url,
                        local_path=str(relative_path),
                        status="skipped_existing",
                        content_type=content_type,
                        bytes=output_path.stat().st_size,
                    ),
                )
                return "skipped_existing"

            digest = _write_file(output_path, content)
            _write_manifest_record(
                manifest_file,
                DownloadManifestRecord(
                    timestamp_utc=_now_utc(),
                    source_id=source.source_id,
                    source_name=source.name,
                    jurisdiction=source.jurisdiction,
                    document_url=final_url,
                    local_path=str(relative_path),
                    status="downloaded",
                    content_type=content_type,
                    bytes=len(content),
                    sha256=digest,
                ),
            )
            return "downloaded"
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, HTTPException) as exc:
            _write_manifest_record(
                manifest_file,
                DownloadManifestRecord(
                    timestamp_utc=_now_utc(),
                    source_id=source.source_id,
                    source_name=source.name,
                    jurisdiction=source.jurisdiction,
                    document_url=url,
                    local_path=None,
                    status="download_error",
                    error=str(exc),
                ),
            )
            return "error"
        finally:
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        outcomes = list(executor.map(acquire, selected_urls))
    downloaded += outcomes.count("downloaded")
    skipped_existing += outcomes.count("skipped_existing")
    error_count += outcomes.count("error")

    return SourceDownloadSummary(
        source_id=source.source_id,
        discovered=len(document_urls),
        downloaded=downloaded,
        skipped_existing=skipped_existing,
        dry_run=dry_run_count,
        errors=error_count,
    )


def download_sources(
    sources: list[OfficialSource],
    *,
    raw_dir: Path,
    manifest_file: Path,
    limit: int | None = 25,
    dry_run: bool = False,
    force: bool = False,
    max_depth: int = 1,
    max_pages: int = 50,
    timeout: float = 30.0,
    sleep_seconds: float = 0.3,
    workers: int = 1,
) -> list[SourceDownloadSummary]:
    """Download documents for multiple sources."""

    summaries: list[SourceDownloadSummary] = []
    for source in sources:
        summaries.append(
            download_source(
                source,
                raw_dir=raw_dir,
                manifest_file=manifest_file,
                limit=limit,
                dry_run=dry_run,
                force=force,
                max_depth=max_depth,
                max_pages=max_pages,
                timeout=timeout,
                sleep_seconds=sleep_seconds,
                workers=workers,
            )
        )
    return summaries


def summary_to_dict(summary: SourceDownloadSummary) -> dict[str, Any]:
    """Return a serializable summary for CLI output or tests."""

    return asdict(summary)
