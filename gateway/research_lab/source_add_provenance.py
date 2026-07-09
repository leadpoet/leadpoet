"""SOURCE_ADD provenance precheck.

This is a pre-manual-review spam/fake-source filter. It does not accept sources
into the catalog and it intentionally fails open to manual review on provider
or network errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request


PRECHECK_PASSED = "provenance_precheck_passed"
PRECHECK_MANUAL = "needs_manual_review"
PRECHECK_REJECTED = "rejected_precheck"

_SD_BASE_URL = "https://api.scrapingdog.com"
_MAX_DOC_EXCERPT = 1200
_FAKE_TEST_PATTERNS = (
    "free fake rest api",
    "fake rest api",
    "placeholder api",
    "dummy api",
    "for testing",
    "prototyping",
    "not persisted",
    "lorem ipsum",
)
_DOC_MARKERS = (
    "api reference",
    "quickstart",
    "endpoint",
    "authentication",
    "authorization",
    "rate limit",
    "status code",
    "curl",
    "http",
)


@dataclass(frozen=True)
class SourceAddProvenanceResult:
    precheck_status: str
    reasons: tuple[str, ...]
    doc: dict[str, Any] = field(default_factory=dict)

    def to_record_doc(self) -> dict[str, Any]:
        return {
            "precheck_status": self.precheck_status,
            "reasons": list(self.reasons),
            **self.doc,
        }


def sanitize_source_add_precheck_doc(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Return a bounded, DB-safe precheck doc without raw secrets or page bodies."""

    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for key, item in value.items():
                lowered_key = str(key).lower()
                if any(marker in lowered_key for marker in ("api_key", "secret", "password", "token", "credential")):
                    out[str(key)[:80]] = "[redacted]"
                else:
                    out[str(key)[:80]] = clean(item)
            return out
        if isinstance(value, list):
            return [clean(item) for item in value[:20]]
        if isinstance(value, tuple):
            return [clean(item) for item in value[:20]]
        if isinstance(value, str):
            text = value
            lowered = text.lower()
            for marker in ("sk-or-", "sb_secret", "service_role", "raw_secret", "api_key="):
                if marker in lowered:
                    return "[redacted]"
            return text[:2000]
        return value

    return clean(dict(doc))


def evaluate_source_add_provenance(
    *,
    source_name: str,
    source_kind: str,
    declared_base_domains: Sequence[str],
    source_metadata: Mapping[str, Any],
    scrapingdog_api_key: str | None = None,
    timeout_seconds: int = 45,
) -> SourceAddProvenanceResult:
    """Run a lightweight, provider-backed legitimacy precheck.

    Obvious fake/test/non-docs sources are rejected. Missing credentials,
    provider failures, and ambiguous evidence become manual review.
    """

    started = time.time()
    key = str(scrapingdog_api_key or os.getenv("RESEARCH_LAB_SCRAPINGDOG_API_KEY") or os.getenv("SCRAPINGDOG_API_KEY") or "").strip()
    metadata = dict(source_metadata or {})
    api_base_url = str(metadata.get("api_base_url") or "").strip()
    documentation_url = str(metadata.get("documentation_url") or "").strip()
    auth_type = str(metadata.get("auth_type") or "").strip()
    endpoint_examples = metadata.get("endpoint_examples") if isinstance(metadata.get("endpoint_examples"), list) else []
    rate_limit_notes = str(metadata.get("rate_limit_notes") or "").strip()
    third_party_refs = [str(item) for item in metadata.get("third_party_refs", []) if str(item or "").strip()]

    reasons: list[str] = []
    doc: dict[str, Any] = {
        "source_name": str(source_name or "")[:160],
        "source_kind": str(source_kind or "")[:80],
        "api_base_domain": _domain(api_base_url),
        "documentation_domain": _domain(documentation_url),
        "declared_base_domains": sorted({_normalize_domain(item) for item in declared_base_domains if _normalize_domain(item)})[:20],
        "auth_type": auth_type[:80],
        "endpoint_example_count": len(endpoint_examples),
        "rate_limit_notes_present": bool(rate_limit_notes),
        "third_party_ref_domains": sorted({_domain(item) for item in third_party_refs if _domain(item)})[:20],
    }

    required_errors = _metadata_required_errors(metadata)
    if required_errors:
        reasons.extend(required_errors)
        doc["duration_ms"] = int((time.time() - started) * 1000)
        return SourceAddProvenanceResult(PRECHECK_REJECTED, tuple(reasons), sanitize_source_add_precheck_doc(doc))

    if not _docs_match_api_domain(api_base_url, documentation_url):
        reasons.append("docs_domain_not_related_to_api_domain")

    docs_fetch = _scrapingdog_scrape(documentation_url, api_key=key, timeout_seconds=timeout_seconds) if key else {
        "provider_status": "missing_scrapingdog_key"
    }
    doc["docs_fetch"] = _summarize_fetch(docs_fetch)
    docs_text = _extract_text(docs_fetch)
    lowered_docs = docs_text.lower()
    fake_hits = _fake_pattern_hits(lowered_docs)
    doc["fake_test_markers"] = fake_hits[:8]
    doc["docs_completeness"] = _docs_completeness(lowered_docs)

    if fake_hits:
        reasons.append("documentation_contains_fake_or_test_markers")
    if docs_fetch.get("provider_status") == "error":
        reasons.append("documentation_provider_error")
    if docs_fetch.get("status") and int(docs_fetch.get("status") or 0) >= 400:
        reasons.append("documentation_fetch_failed")
    if docs_fetch.get("provider_status") == "missing_scrapingdog_key":
        reasons.append("scrapingdog_key_missing")

    ai_query = _build_ai_query(source_name, api_base_url, documentation_url, third_party_refs)
    ai_result = _scrapingdog_ai_mode(ai_query, api_key=key, timeout_seconds=timeout_seconds) if key else {
        "provider_status": "missing_scrapingdog_key"
    }
    ai_summary = _summarize_ai(ai_result)
    doc["ai_mode"] = ai_summary
    ai_text = str(ai_summary.get("markdown_excerpt") or "").lower()
    ai_fake_hits = _fake_pattern_hits(ai_text)
    if ai_fake_hits:
        reasons.append("ai_mode_identified_fake_or_test_api")
    if int(ai_summary.get("reference_count") or 0) == 0:
        reasons.append("ai_mode_no_references")

    doc["duration_ms"] = int((time.time() - started) * 1000)

    # Obvious fake/test sources are rejected early. Provider/config failures and
    # weak provenance remain reviewable by an operator.
    if "documentation_contains_fake_or_test_markers" in reasons or "ai_mode_identified_fake_or_test_api" in reasons:
        status = PRECHECK_REJECTED
    elif any(
        reason in reasons
        for reason in ("documentation_fetch_failed", "documentation_provider_error", "scrapingdog_key_missing", "ai_mode_no_references")
    ):
        status = PRECHECK_MANUAL
    elif doc["docs_completeness"]["score"] >= 2 and int(ai_summary.get("reference_count") or 0) >= 2:
        status = PRECHECK_PASSED
    else:
        status = PRECHECK_MANUAL
        if "low_docs_completeness" not in reasons:
            reasons.append("low_docs_completeness")

    if not reasons:
        reasons.append("provenance_reference_backed")
    return SourceAddProvenanceResult(status, tuple(sorted(set(reasons))), sanitize_source_add_precheck_doc(doc))


def _metadata_required_errors(metadata: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in ("api_base_url", "documentation_url", "auth_type", "rate_limit_notes"):
        if not str(metadata.get(field_name) or "").strip():
            errors.append(f"missing_{field_name}")
    examples = metadata.get("endpoint_examples")
    if not isinstance(examples, list) or not examples:
        errors.append("missing_endpoint_examples")
    return errors


def _scrapingdog_scrape(target_url: str, *, api_key: str, timeout_seconds: int) -> dict[str, Any]:
    return _scrapingdog_get(
        "/scrape",
        {"url": target_url},
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )


def _scrapingdog_ai_mode(query: str, *, api_key: str, timeout_seconds: int) -> dict[str, Any]:
    return _scrapingdog_get(
        "/google/ai_mode",
        {"query": query, "country": "us"},
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )


def _scrapingdog_get(path: str, params: Mapping[str, str], *, api_key: str, timeout_seconds: int) -> dict[str, Any]:
    query = dict(params)
    query["api_key"] = api_key
    url = _SD_BASE_URL + path + "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(url, headers={"User-Agent": "leadpoet-source-add-precheck/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(240_000)
            status = int(response.status)
            content_type = str(response.headers.get("content-type") or "")
    except urllib.error.HTTPError as exc:
        body = exc.read(120_000)
        status = int(exc.code)
        content_type = str(exc.headers.get("content-type") or "") if exc.headers else ""
    except Exception as exc:
        return {"provider_status": "error", "error_type": type(exc).__name__, "error": str(exc)[:200]}
    text = body.decode("utf-8", "replace")
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    return {
        "provider_status": "ok",
        "status": status,
        "content_type": content_type[:120],
        "body_text": text[:_MAX_DOC_EXCERPT],
        "json": parsed if isinstance(parsed, Mapping) else None,
    }


def _summarize_fetch(result: Mapping[str, Any]) -> dict[str, Any]:
    parsed = result.get("json") if isinstance(result.get("json"), Mapping) else {}
    text = _extract_text(result)
    title = _html_title(text)
    return {
        "provider_status": str(result.get("provider_status") or "")[:80],
        "status": int(result.get("status") or 0),
        "content_type": str(result.get("content_type") or "")[:120],
        "title": title[:200],
        "text_excerpt": _strip_html(text)[:_MAX_DOC_EXCERPT],
        "json_keys": sorted(str(key)[:80] for key in parsed.keys())[:20] if isinstance(parsed, Mapping) else [],
        "error_type": str(result.get("error_type") or "")[:80],
    }


def _summarize_ai(result: Mapping[str, Any]) -> dict[str, Any]:
    parsed = result.get("json") if isinstance(result.get("json"), Mapping) else {}
    markdown = str(parsed.get("markdown") or parsed.get("answer") or "")
    references = parsed.get("references") if isinstance(parsed.get("references"), list) else []
    ref_docs: list[dict[str, str]] = []
    for item in references[:12]:
        if not isinstance(item, Mapping):
            continue
        link = str(item.get("link") or item.get("url") or "")
        ref_docs.append(
            {
                "title": str(item.get("title") or "")[:160],
                "source": str(item.get("source") or "")[:120],
                "domain": _domain(link),
            }
        )
    return {
        "provider_status": str(result.get("provider_status") or "")[:80],
        "status": int(result.get("status") or 0),
        "reference_count": len(references),
        "reference_domains": sorted({ref["domain"] for ref in ref_docs if ref["domain"]})[:20],
        "references": ref_docs,
        "markdown_excerpt": markdown[:1600],
        "error_type": str(result.get("error_type") or "")[:80],
    }


def _extract_text(result: Mapping[str, Any]) -> str:
    parsed = result.get("json") if isinstance(result.get("json"), Mapping) else {}
    for key in ("text", "html", "markdown", "body", "message"):
        value = parsed.get(key) if isinstance(parsed, Mapping) else None
        if isinstance(value, str) and value:
            return value[:_MAX_DOC_EXCERPT]
    return str(result.get("body_text") or "")[:_MAX_DOC_EXCERPT]


def _docs_completeness(text: str) -> dict[str, Any]:
    hits = [marker for marker in _DOC_MARKERS if marker in text]
    return {"score": len(hits), "markers": hits[:12]}


def _fake_pattern_hits(text: str) -> list[str]:
    hits: list[str] = []
    lowered = str(text or "").lower()
    for pattern in _FAKE_TEST_PATTERNS:
        start = 0
        while True:
            index = lowered.find(pattern, start)
            if index < 0:
                break
            prefix = lowered[max(0, index - 24) : index]
            if not re.search(r"\b(not|isn't|neither)\b[\w\s,;-]{0,18}$", prefix):
                hits.append(pattern)
                break
            start = index + len(pattern)
    return hits[:8]


def _build_ai_query(source_name: str, api_base_url: str, documentation_url: str, refs: Sequence[str]) -> str:
    return (
        "Assess if this submitted API source is a legitimate established API, not a fake/test API or miner-owned wrapper. "
        "Return concise evidence and cite official/third-party sources. "
        f"Source name: {str(source_name or '')[:120]}. "
        f"API base URL: {api_base_url}. Documentation URL: {documentation_url}. "
        f"Submitted third-party refs: {', '.join(refs[:5])}."
    )[:1800]


def _html_title(text: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.I | re.S)
    return _strip_html(match.group(1)) if match else ""


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text or ""))).strip()


def _domain(value: str) -> str:
    return _normalize_domain(value)


def _normalize_domain(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urllib.parse.urlparse(raw if "://" in raw else "https://" + raw)
    domain = (parsed.hostname or raw).lower().split(":", 1)[0]
    return domain[4:] if domain.startswith("www.") else domain


def _docs_match_api_domain(api_url: str, docs_url: str) -> bool:
    api = _domain(api_url)
    docs = _domain(docs_url)
    if not api or not docs:
        return False
    return api == docs or api.endswith("." + docs) or docs.endswith("." + api) or _root_domain(api) == _root_domain(docs)


def _root_domain(domain: str) -> str:
    parts = [part for part in str(domain or "").split(".") if part]
    return ".".join(parts[-2:]) if len(parts) >= 2 else str(domain or "")


__all__ = [
    "PRECHECK_MANUAL",
    "PRECHECK_PASSED",
    "PRECHECK_REJECTED",
    "SourceAddProvenanceResult",
    "evaluate_source_add_provenance",
    "sanitize_source_add_precheck_doc",
]
