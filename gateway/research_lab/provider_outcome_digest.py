"""W2: sanitized provider-outcome digest for generation context.

Candidate generation previously flew blind on provider behavior: recorded
truth (bug-35 provider-error markers, ``provider_usage`` rows, the W3 usage
ledger, day-cache outcomes) never reached the planner/inspection/draft
prompts. This module projects that truth into a small, sanitized digest that
is injected as a sibling of ``benchmark_public_summary``.

Sanitization contract: no request/response bodies, no query strings, no URLs
beyond the provider host class, no ICP/company identifiers (anything
identifier-shaped is hashed before it gets here or dropped). Only aggregate
counts, histograms, and rates leave this module.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json

PROVIDER_OUTCOME_DIGEST_SCHEMA_VERSION = "1.0"
PROVIDER_OUTCOME_DIGEST_ENV = "RESEARCH_LAB_PROVIDER_OUTCOME_DIGEST"

_MAX_LEDGER_ROWS_READ = 20_000

_MARKER_PREFIX = "research_lab_private_runtime_provider_error"
_STATUS_RE = re.compile(r"\bstatus=(\d{3})\b")
_URL_RE = re.compile(r"\burl=(\S+)")
_ERROR_CLASS_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]{2,60})(?=[:;\s])")

_MAX_PROVIDERS = 12
_MAX_HISTOGRAM_KEYS = 10
_MAX_TOP_ERRORS = 8


def _provider_for_url(url: str) -> str:
    lowered = str(url or "").lower()
    if "exa.ai" in lowered:
        return "exa"
    if "scrapingdog" in lowered:
        return "scrapingdog"
    if "openrouter" in lowered:
        return "openrouter"
    host = re.sub(r"^[a-z]+://", "", lowered).split("/", 1)[0].split("?", 1)[0]
    host = host.rsplit("@", 1)[-1].split(":", 1)[0]
    # Keep only the registrable-suffix shape; never echo full hosts that might
    # embed tenant identifiers.
    parts = [part for part in host.split(".") if part]
    return ".".join(parts[-2:]) if len(parts) >= 2 else (host or "unknown")


def parse_provider_error_marker_line(line: str) -> dict[str, Any] | None:
    """Parse one bug-35 marker line into {provider, error_class, http_status}.

    Bodies and query strings in the marker are dropped — only the exception
    class, status code, and provider class survive into the digest.
    """

    text = str(line or "")
    index = text.find(_MARKER_PREFIX)
    if index < 0:
        return None
    payload = text[index + len(_MARKER_PREFIX) :].strip()
    if not payload:
        return None
    status_match = _STATUS_RE.search(payload)
    url_match = _URL_RE.search(payload)
    error_match = _ERROR_CLASS_RE.match(payload)
    return {
        "provider": _provider_for_url(url_match.group(1) if url_match else ""),
        "error_class": (error_match.group(1) if error_match else "unknown_error")[:60],
        "http_status": int(status_match.group(1)) if status_match else 0,
    }


def _bucket(providers: dict[str, dict[str, Counter]], provider: str) -> dict[str, Counter]:
    if provider not in providers:
        providers[provider] = {
            "stages": Counter(),
            "status_histogram": Counter(),
            "error_classes": Counter(),
            "endpoint_classes": Counter(),
            "meta": Counter(),
        }
    return providers[provider]


def build_provider_outcome_digest(
    *,
    provider_usage_rows: Sequence[Mapping[str, Any]] = (),
    usage_ledger_rows: Sequence[Mapping[str, Any]] = (),
    provider_error_marker_lines: Sequence[str] = (),
    day_cache_entries: Mapping[str, Mapping[str, Any]] | None = None,
    candidate_snapshot_miss_counts: Mapping[str, int] | None = None,
    utc_day: str = "",
) -> dict[str, Any]:
    """Aggregate recorded provider truth into the generation-context digest.

    Inputs (all optional, all already-recorded truth):
    - ``provider_usage_rows``: loop LLM usage entries
      (``provider``/``call_stage``/``failed_request.error_class``/``http_status``);
    - ``usage_ledger_rows``: W3 proxy ledger rows
      (``provider_id``/``endpoint_class``/``status``/``evidence``);
    - ``provider_error_marker_lines``: bug-35 stderr marker lines;
    - ``day_cache_entries``: evidence day-cache entries (``status``/``outcome``);
    - ``candidate_snapshot_miss_counts``: per-candidate dev-eval miss counts so
      replay artifacts are not read as bad hypotheses (§6).
    """

    providers: dict[str, dict[str, Counter]] = {}

    for row in provider_usage_rows:
        if not isinstance(row, Mapping):
            continue
        provider = str(row.get("provider") or "unknown")[:40]
        bucket = _bucket(providers, provider)
        bucket["meta"]["call_count"] += 1
        stage = str(row.get("call_stage") or row.get("stage") or "")[:60]
        if stage:
            bucket["stages"][stage] += 1
        failed = row.get("failed_request")
        if isinstance(failed, Mapping):
            bucket["meta"]["error_count"] += 1
            error_class = str(failed.get("error_class") or "")[:60]
            if error_class:
                bucket["error_classes"][error_class] += 1
            try:
                status = int(failed.get("http_status") or 0)
            except (TypeError, ValueError):
                status = 0
            if status:
                bucket["status_histogram"][str(status)] += 1

    for row in usage_ledger_rows:
        if not isinstance(row, Mapping):
            continue
        provider = str(row.get("provider_id") or "unknown")[:40]
        bucket = _bucket(providers, provider)
        bucket["meta"]["call_count"] += 1
        endpoint_class = str(row.get("endpoint_class") or "")[:80]
        if endpoint_class:
            bucket["endpoint_classes"][endpoint_class] += 1
        try:
            status = int(row.get("status") or 0)
        except (TypeError, ValueError):
            status = 0
        if status:
            bucket["status_histogram"][str(status)] += 1
            if status >= 400:
                bucket["meta"]["error_count"] += 1
        evidence = str(row.get("evidence") or "")
        if evidence in {"error", "quota_exhausted", "credential_missing"}:
            bucket["error_classes"][evidence] += 1

    for line in provider_error_marker_lines:
        parsed = parse_provider_error_marker_line(line)
        if not parsed:
            continue
        bucket = _bucket(providers, parsed["provider"])
        bucket["meta"]["call_count"] += 1
        bucket["meta"]["error_count"] += 1
        bucket["error_classes"][parsed["error_class"]] += 1
        if parsed["http_status"]:
            bucket["status_histogram"][str(parsed["http_status"])] += 1

    day_outcomes = Counter()
    for entry in (day_cache_entries or {}).values():
        if not isinstance(entry, Mapping):
            continue
        outcome = str(entry.get("outcome") or "")
        if outcome:
            day_outcomes[outcome] += 1

    provider_docs: dict[str, dict[str, Any]] = {}
    total_errors: Counter = Counter()
    for provider, bucket in sorted(providers.items())[:_MAX_PROVIDERS]:
        call_count = int(bucket["meta"].get("call_count", 0))
        error_count = int(bucket["meta"].get("error_count", 0))
        provider_docs[provider] = {
            "call_count": call_count,
            "error_count": error_count,
            "error_rate": round(error_count / call_count, 4) if call_count else 0.0,
            "stages": dict(bucket["stages"].most_common(_MAX_HISTOGRAM_KEYS)),
            "status_histogram": dict(bucket["status_histogram"].most_common(_MAX_HISTOGRAM_KEYS)),
            "error_classes": dict(bucket["error_classes"].most_common(_MAX_HISTOGRAM_KEYS)),
            "endpoint_classes": dict(bucket["endpoint_classes"].most_common(_MAX_HISTOGRAM_KEYS)),
        }
        for error_class, count in bucket["error_classes"].items():
            total_errors[f"{provider}:{error_class}"] += count

    digest: dict[str, Any] = {
        "schema_version": PROVIDER_OUTCOME_DIGEST_SCHEMA_VERSION,
        "utc_day": str(utc_day or "")[:10],
        "providers": provider_docs,
        "top_error_classes": [
            {"key": key, "count": count} for key, count in total_errors.most_common(_MAX_TOP_ERRORS)
        ],
        "day_cache_outcomes": dict(day_outcomes.most_common(4)),
    }
    if candidate_snapshot_miss_counts:
        digest["candidate_snapshot_miss_counts"] = {
            str(node_id)[:80]: max(0, int(count))
            for node_id, count in list(candidate_snapshot_miss_counts.items())[:20]
        }
        digest["snapshot_miss_note"] = (
            "High snapshot_miss_count means the candidate changed provider request "
            "shapes and dev replay could not score it — treat its dev score as "
            "not-comparable, not as a refuted hypothesis."
        )
    digest["digest_hash"] = sha256_json({key: value for key, value in digest.items() if key != "digest_hash"})
    return digest


def _utc_day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _read_todays_ledger_rows(path: str) -> list[dict[str, Any]]:
    today = _utc_day()
    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= _MAX_LEDGER_ROWS_READ:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and str(row.get("utc_day") or "") == today:
                    rows.append(row)
    except OSError:
        return []
    return rows


def _read_day_cache_entries(path: str) -> dict[str, dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(doc, Mapping) or str(doc.get("utc_day") or "") != _utc_day():
        return {}
    entries = doc.get("entries")
    if not isinstance(entries, Mapping):
        return {}
    # Only outcome/status leave the cache — bodies never enter the digest.
    return {
        str(key): {"status": value.get("status"), "outcome": value.get("outcome")}
        for key, value in entries.items()
        if isinstance(value, Mapping)
    }


def build_run_provider_outcome_digest(
    *,
    candidate_snapshot_miss_counts: Mapping[str, int] | None = None,
) -> dict[str, Any] | None:
    """Worker-side digest from on-disk recorded truth. Flag-gated, default on.

    Reads today's W3 usage-ledger rows and evidence day-cache outcomes; returns
    None (prompt-inert) when ``RESEARCH_LAB_PROVIDER_OUTCOME_DIGEST=false`` or
    nothing recorded yet.
    """

    raw = str(os.getenv(PROVIDER_OUTCOME_DIGEST_ENV, "true") or "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return None
    ledger_path = str(os.getenv("RESEARCH_LAB_PROVIDER_USAGE_LEDGER_PATH") or "").strip()
    day_cache_path = str(os.getenv("RESEARCH_LAB_PROVIDER_EVIDENCE_DAY_CACHE") or "").strip()
    ledger_rows = _read_todays_ledger_rows(ledger_path) if ledger_path else []
    day_entries = _read_day_cache_entries(day_cache_path) if day_cache_path else {}
    if not ledger_rows and not day_entries and not candidate_snapshot_miss_counts:
        return None
    return build_provider_outcome_digest(
        usage_ledger_rows=ledger_rows,
        day_cache_entries=day_entries,
        candidate_snapshot_miss_counts=candidate_snapshot_miss_counts,
        utc_day=_utc_day(),
    )
