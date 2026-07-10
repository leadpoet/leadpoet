"""W2: sanitized provider-outcome digest for generation context.

Candidate generation previously flew blind on provider behavior. The evidence
proxy now maintains one bounded current-day sidecar while it records calls;
hosted workers project that compact file into planner/inspection/draft context
without parsing the much larger replay cache or usage ledger.

Sanitization contract: no request/response bodies, no query strings, no URLs
beyond the provider host class, no ICP/company identifiers (anything
identifier-shaped is hashed before it gets here or dropped). Only aggregate
counts, histograms, and rates leave this module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import threading
import time
from collections import Counter
from copy import deepcopy
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json

PROVIDER_OUTCOME_DIGEST_SCHEMA_VERSION = "1.0"
PROVIDER_OUTCOME_DIGEST_ENV = "RESEARCH_LAB_PROVIDER_OUTCOME_DIGEST"
PROVIDER_OUTCOME_SIDECAR_ENV = "RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH"
PROVIDER_OUTCOME_SIDECAR_STALE_SECONDS_ENV = (
    "RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_STALE_SECONDS"
)
PROVIDER_OUTCOME_SIDECAR_SCHEMA_VERSION = "1.0"

MAX_PROVIDER_OUTCOME_SIDECAR_BYTES = 128 * 1024
MAX_PROVIDER_OUTCOME_ENDPOINTS = 24
MAX_PROVIDER_OUTCOME_STATUSES = 12
MAX_PROVIDER_OUTCOME_EVIDENCE = 12
DEFAULT_PROVIDER_OUTCOME_SIDECAR_STALE_SECONDS = 3600
DEFAULT_PROVIDER_OUTCOME_SIDECAR_FLUSH_SECONDS = 1.0

_MARKER_PREFIX = "research_lab_private_runtime_provider_error"
_STATUS_RE = re.compile(r"\bstatus=(\d{3})\b")
_URL_RE = re.compile(r"\burl=(\S+)")
_ERROR_CLASS_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]{2,60})(?=[:;\s])")

_MAX_PROVIDERS = 12
_MAX_HISTOGRAM_KEYS = 10
_MAX_TOP_ERRORS = 8
_SAFE_BUCKET_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,80}$")
_SAFE_ENDPOINT_RE = re.compile(r"^/[A-Za-z0-9_./:\-]{0,159}$")

logger = logging.getLogger(__name__)
_warning_lock = threading.Lock()
_warning_keys: set[str] = set()


def _warn_once(reason: str, *, path: str = "", detail: str = "") -> None:
    key = f"{reason}:{path}"
    with _warning_lock:
        if key in _warning_keys:
            return
        _warning_keys.add(key)
    logger.warning(
        "research_lab_provider_outcome_sidecar_unavailable reason=%s path_hash=%s detail=%s",
        str(reason or "unknown")[:80],
        sha256_json({"path": str(path or "")}) if path else "",
        str(detail or "")[:120],
    )


def _empty_sidecar_doc(utc_day: str) -> dict[str, Any]:
    return {
        "schema_version": PROVIDER_OUTCOME_SIDECAR_SCHEMA_VERSION,
        "utc_day": str(utc_day or "")[:10],
        "generated_at": "",
        "generated_at_epoch": 0.0,
        "sequence": 0,
        "providers": {},
        "totals": _empty_outcome_bucket(),
    }


def _empty_outcome_bucket() -> dict[str, Any]:
    return {
        "call_count": 0,
        "cache_hit_count": 0,
        "live_call_count": 0,
        "blocked_count": 0,
        "quota_exhausted_count": 0,
        "credential_missing_count": 0,
        "replay_miss_count": 0,
        "error_count": 0,
        "measured_spend_microusd": 0,
        "estimated_spend_microusd": 0,
        "status_counts": {},
        "evidence_counts": {},
    }


def _safe_provider_bucket(value: Any) -> str:
    text = str(value or "unknown")[:80]
    return text if _SAFE_BUCKET_RE.fullmatch(text) else "other"


def _safe_endpoint_bucket(value: Any) -> str:
    text = str(value or "/")[:160]
    if "?" in text or "#" in text or ".." in text:
        return "other"
    return text if _SAFE_ENDPOINT_RE.fullmatch(text) else "other"


def _safe_evidence_bucket(value: Any) -> str:
    text = str(value or "unknown")[:80]
    return text if _SAFE_BUCKET_RE.fullmatch(text) else "other"


def _increment_bounded(counter: dict[str, Any], key: str, *, cap: int) -> None:
    bucket = key if key in counter or len(counter) < max(1, cap - 1) else "other"
    counter[bucket] = max(0, int(counter.get(bucket) or 0)) + 1


def _record_outcome_bucket(
    bucket: dict[str, Any],
    *,
    evidence: str,
    status: int,
    live_call: bool,
    spend_microusd: int,
    spend_kind: str,
) -> None:
    bucket["call_count"] = max(0, int(bucket.get("call_count") or 0)) + 1
    if evidence == "hit":
        bucket["cache_hit_count"] = max(0, int(bucket.get("cache_hit_count") or 0)) + 1
    if live_call:
        bucket["live_call_count"] = max(0, int(bucket.get("live_call_count") or 0)) + 1
    if evidence in {"blocked", "budget_soft_stop"}:
        bucket["blocked_count"] = max(0, int(bucket.get("blocked_count") or 0)) + 1
    if evidence == "quota_exhausted":
        bucket["quota_exhausted_count"] = max(
            0, int(bucket.get("quota_exhausted_count") or 0)
        ) + 1
    if evidence == "credential_missing":
        bucket["credential_missing_count"] = max(
            0, int(bucket.get("credential_missing_count") or 0)
        ) + 1
    if evidence == "replay_miss":
        bucket["replay_miss_count"] = max(0, int(bucket.get("replay_miss_count") or 0)) + 1
    if (status > 0 and not 200 <= status < 300) or evidence in {
        "error",
        "quota_exhausted",
        "credential_missing",
    }:
        bucket["error_count"] = max(0, int(bucket.get("error_count") or 0)) + 1
    _increment_bounded(
        bucket.setdefault("status_counts", {}),
        str(max(0, int(status))),
        cap=MAX_PROVIDER_OUTCOME_STATUSES,
    )
    _increment_bounded(
        bucket.setdefault("evidence_counts", {}),
        _safe_evidence_bucket(evidence),
        cap=MAX_PROVIDER_OUTCOME_EVIDENCE,
    )
    spend = max(0, int(spend_microusd)) if live_call and 200 <= status < 300 else 0
    if spend and spend_kind == "measured":
        bucket["measured_spend_microusd"] = max(
            0, int(bucket.get("measured_spend_microusd") or 0)
        ) + spend
    elif spend:
        bucket["estimated_spend_microusd"] = max(
            0, int(bucket.get("estimated_spend_microusd") or 0)
        ) + spend


def _sidecar_document_hash(doc: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in doc.items() if key != "document_hash"})


def _validated_sidecar_doc(
    doc: Any,
    *,
    now: float | None = None,
    stale_seconds: int = 0,
) -> dict[str, Any] | None:
    if not isinstance(doc, Mapping):
        return None
    if str(doc.get("schema_version") or "") != PROVIDER_OUTCOME_SIDECAR_SCHEMA_VERSION:
        return None
    if str(doc.get("utc_day") or "") != _utc_day():
        return None
    if str(doc.get("document_hash") or "") != _sidecar_document_hash(doc):
        return None
    try:
        generated_at_epoch = float(doc.get("generated_at_epoch") or 0.0)
        sequence = max(0, int(doc.get("sequence") or 0))
    except (TypeError, ValueError):
        return None
    current_time = time.time() if now is None else float(now)
    if generated_at_epoch <= 0 or generated_at_epoch > current_time + 60:
        return None
    if stale_seconds > 0 and current_time - generated_at_epoch > stale_seconds:
        return None
    providers = doc.get("providers")
    totals = doc.get("totals")
    if not isinstance(providers, Mapping) or not isinstance(totals, Mapping):
        return None
    if len(providers) > _MAX_PROVIDERS:
        return None
    normalized_providers: dict[str, Any] = {}
    for raw_provider, raw_bucket in providers.items():
        provider = _safe_provider_bucket(raw_provider)
        if provider != str(raw_provider) or not isinstance(raw_bucket, Mapping):
            return None
        endpoints = raw_bucket.get("endpoints") or {}
        if not isinstance(endpoints, Mapping) or len(endpoints) > MAX_PROVIDER_OUTCOME_ENDPOINTS:
            return None
        normalized_endpoints: dict[str, Any] = {}
        for raw_endpoint, endpoint_bucket in endpoints.items():
            endpoint = _safe_endpoint_bucket(raw_endpoint)
            if endpoint != str(raw_endpoint) or not isinstance(endpoint_bucket, Mapping):
                return None
            normalized = _normalize_outcome_bucket(endpoint_bucket)
            if normalized is None:
                return None
            normalized_endpoints[endpoint] = normalized
        normalized_bucket = _normalize_outcome_bucket(raw_bucket)
        if normalized_bucket is None:
            return None
        normalized_bucket["endpoints"] = normalized_endpoints
        normalized_providers[provider] = normalized_bucket
    normalized_totals = _normalize_outcome_bucket(totals)
    if normalized_totals is None:
        return None
    normalized_doc = {
        "schema_version": PROVIDER_OUTCOME_SIDECAR_SCHEMA_VERSION,
        "utc_day": _utc_day(),
        "generated_at": str(doc.get("generated_at") or "")[:32],
        "generated_at_epoch": generated_at_epoch,
        "sequence": sequence,
        "providers": normalized_providers,
        "totals": normalized_totals,
    }
    normalized_doc["document_hash"] = _sidecar_document_hash(normalized_doc)
    if normalized_doc["document_hash"] != str(doc.get("document_hash") or ""):
        return None
    return normalized_doc


def _normalize_outcome_bucket(value: Mapping[str, Any]) -> dict[str, Any] | None:
    normalized = _empty_outcome_bucket()
    for key in (
        "call_count",
        "cache_hit_count",
        "live_call_count",
        "blocked_count",
        "quota_exhausted_count",
        "credential_missing_count",
        "replay_miss_count",
        "error_count",
        "measured_spend_microusd",
        "estimated_spend_microusd",
    ):
        try:
            raw = int(value.get(key) or 0)
        except (TypeError, ValueError):
            return None
        if raw < 0:
            return None
        normalized[key] = raw
    for field_name, cap in (
        ("status_counts", MAX_PROVIDER_OUTCOME_STATUSES),
        ("evidence_counts", MAX_PROVIDER_OUTCOME_EVIDENCE),
    ):
        raw_counter = value.get(field_name) or {}
        if not isinstance(raw_counter, Mapping) or len(raw_counter) > cap:
            return None
        counter: dict[str, int] = {}
        for raw_key, raw_count in raw_counter.items():
            key = _safe_evidence_bucket(raw_key)
            if key != str(raw_key):
                return None
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                return None
            if count < 0:
                return None
            counter[key] = count
        normalized[field_name] = counter
    return normalized


def load_provider_outcome_sidecar(
    path: str,
    *,
    now: float | None = None,
    stale_seconds: int = DEFAULT_PROVIDER_OUTCOME_SIDECAR_STALE_SECONDS,
    warn: bool = True,
) -> dict[str, Any] | None:
    """Read one compact, mode-0600, hash-verified current-day sidecar."""

    sidecar_path = str(path or "").strip()
    if not sidecar_path:
        if warn:
            _warn_once("path_unset")
        return None
    try:
        path_stat = os.stat(sidecar_path)
    except OSError as exc:
        if warn:
            _warn_once("missing", path=sidecar_path, detail=type(exc).__name__)
        return None
    if not stat.S_ISREG(path_stat.st_mode) or stat.S_IMODE(path_stat.st_mode) & 0o077:
        if warn:
            _warn_once("unsafe_mode", path=sidecar_path)
        return None
    if path_stat.st_size <= 0 or path_stat.st_size > MAX_PROVIDER_OUTCOME_SIDECAR_BYTES:
        if warn:
            _warn_once("invalid_size", path=sidecar_path, detail=str(path_stat.st_size))
        return None
    try:
        with open(sidecar_path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        if warn:
            _warn_once("malformed", path=sidecar_path, detail=type(exc).__name__)
        return None
    validated = _validated_sidecar_doc(doc, now=now, stale_seconds=stale_seconds)
    if validated is None and warn:
        _warn_once("invalid_or_stale", path=sidecar_path)
    return validated


class ProviderOutcomeSidecarAccumulator:
    """Thread-safe bounded daily aggregates written atomically at most once/second."""

    def __init__(
        self,
        path: str,
        *,
        flush_interval_seconds: float = DEFAULT_PROVIDER_OUTCOME_SIDECAR_FLUSH_SECONDS,
    ) -> None:
        self._path = str(path or "").strip()
        self._flush_interval_seconds = max(0.01, float(flush_interval_seconds))
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self._last_flush_monotonic = 0.0
        self._dirty = False
        loaded = (
            load_provider_outcome_sidecar(self._path, stale_seconds=0, warn=False)
            if self._path
            else None
        )
        self._doc = loaded or _empty_sidecar_doc(_utc_day())

    def record(
        self,
        *,
        provider_id: str,
        endpoint_class: str,
        evidence: str,
        status: int,
        live_call: bool,
        spend_microusd: int,
        spend_kind: str,
    ) -> None:
        if not self._path:
            return
        with self._lock:
            self._roll_day_locked()
            providers = self._doc.setdefault("providers", {})
            provider = _safe_provider_bucket(provider_id)
            if provider not in providers and len(providers) >= max(1, _MAX_PROVIDERS - 1):
                provider = "other"
            provider_bucket = providers.setdefault(provider, _empty_outcome_bucket())
            endpoints = provider_bucket.setdefault("endpoints", {})
            endpoint = _safe_endpoint_bucket(endpoint_class)
            if endpoint not in endpoints and len(endpoints) >= max(
                1, MAX_PROVIDER_OUTCOME_ENDPOINTS - 1
            ):
                endpoint = "other"
            endpoint_bucket = endpoints.setdefault(endpoint, _empty_outcome_bucket())
            safe_evidence = _safe_evidence_bucket(evidence)
            _record_outcome_bucket(
                provider_bucket,
                evidence=safe_evidence,
                status=max(0, int(status)),
                live_call=bool(live_call),
                spend_microusd=max(0, int(spend_microusd)),
                spend_kind="measured" if spend_kind == "measured" else "estimated",
            )
            _record_outcome_bucket(
                endpoint_bucket,
                evidence=safe_evidence,
                status=max(0, int(status)),
                live_call=bool(live_call),
                spend_microusd=max(0, int(spend_microusd)),
                spend_kind="measured" if spend_kind == "measured" else "estimated",
            )
            _record_outcome_bucket(
                self._doc.setdefault("totals", _empty_outcome_bucket()),
                evidence=safe_evidence,
                status=max(0, int(status)),
                live_call=bool(live_call),
                spend_microusd=max(0, int(spend_microusd)),
                spend_kind="measured" if spend_kind == "measured" else "estimated",
            )
            self._doc["sequence"] = max(0, int(self._doc.get("sequence") or 0)) + 1
            self._dirty = True
            elapsed = time.monotonic() - self._last_flush_monotonic
            if self._last_flush_monotonic <= 0 or elapsed >= self._flush_interval_seconds:
                self._schedule_locked(0.01)
            else:
                self._schedule_locked(self._flush_interval_seconds - elapsed)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._doc)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._flush_locked()

    def _roll_day_locked(self) -> None:
        if str(self._doc.get("utc_day") or "") != _utc_day():
            self._doc = _empty_sidecar_doc(_utc_day())
            self._dirty = True

    def _schedule_locked(self, delay: float) -> None:
        if self._timer is not None:
            return
        timer = threading.Timer(max(0.01, delay), self._flush_from_timer)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _flush_from_timer(self) -> None:
        with self._lock:
            self._timer = None
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._path or not self._dirty:
            return
        now = time.time()
        self._doc["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        self._doc["generated_at_epoch"] = round(now, 6)
        self._doc["document_hash"] = _sidecar_document_hash(self._doc)
        encoded = json.dumps(
            self._doc,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > MAX_PROVIDER_OUTCOME_SIDECAR_BYTES:
            logger.warning(
                "research_lab_provider_outcome_sidecar_write_failed reason=oversized size=%s",
                len(encoded),
            )
            return
        directory = os.path.dirname(self._path) or "."
        temp_path = (
            f"{self._path}.tmp.{os.getpid()}.{threading.get_ident()}.{int(now * 1_000_000)}"
        )
        try:
            os.makedirs(directory, mode=0o700, exist_ok=True)
            fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
            os.replace(temp_path, self._path)
            os.chmod(self._path, 0o600)
            self._last_flush_monotonic = time.monotonic()
            self._dirty = False
        except Exception as exc:
            logger.warning(
                "research_lab_provider_outcome_sidecar_write_failed path_hash=%s error=%s",
                sha256_json({"path": self._path}),
                type(exc).__name__,
            )
            try:
                os.unlink(temp_path)
            except OSError:
                pass


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


def _digest_from_sidecar(
    sidecar: Mapping[str, Any],
    *,
    candidate_snapshot_miss_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    providers: dict[str, Any] = {}
    top_errors: Counter = Counter()
    for provider, raw_bucket in sorted((sidecar.get("providers") or {}).items()):
        if not isinstance(raw_bucket, Mapping):
            continue
        call_count = max(0, int(raw_bucket.get("call_count") or 0))
        error_count = max(0, int(raw_bucket.get("error_count") or 0))
        evidence_counts = {
            str(key): max(0, int(value))
            for key, value in (raw_bucket.get("evidence_counts") or {}).items()
        }
        error_classes = {
            key: count
            for key, count in evidence_counts.items()
            if key in {"error", "blocked", "budget_soft_stop", "quota_exhausted", "credential_missing"}
        }
        for error_class, count in error_classes.items():
            top_errors[f"{provider}:{error_class}"] += count
        endpoints = raw_bucket.get("endpoints") or {}
        providers[str(provider)] = {
            "call_count": call_count,
            "error_count": error_count,
            "error_rate": round(error_count / call_count, 4) if call_count else 0.0,
            "cache_hit_count": max(0, int(raw_bucket.get("cache_hit_count") or 0)),
            "live_call_count": max(0, int(raw_bucket.get("live_call_count") or 0)),
            "blocked_count": max(0, int(raw_bucket.get("blocked_count") or 0)),
            "quota_exhausted_count": max(
                0, int(raw_bucket.get("quota_exhausted_count") or 0)
            ),
            "credential_missing_count": max(
                0, int(raw_bucket.get("credential_missing_count") or 0)
            ),
            "replay_miss_count": max(0, int(raw_bucket.get("replay_miss_count") or 0)),
            "status_histogram": {
                str(key): max(0, int(value))
                for key, value in (raw_bucket.get("status_counts") or {}).items()
            },
            "evidence_counts": evidence_counts,
            "error_classes": error_classes,
            "endpoint_classes": {
                str(endpoint): max(0, int(endpoint_bucket.get("call_count") or 0))
                for endpoint, endpoint_bucket in sorted(endpoints.items())
                if isinstance(endpoint_bucket, Mapping)
            },
            "measured_spend_microusd": max(
                0, int(raw_bucket.get("measured_spend_microusd") or 0)
            ),
            "estimated_spend_microusd": max(
                0, int(raw_bucket.get("estimated_spend_microusd") or 0)
            ),
        }
    totals = sidecar.get("totals") if isinstance(sidecar.get("totals"), Mapping) else {}
    digest: dict[str, Any] = {
        "schema_version": PROVIDER_OUTCOME_DIGEST_SCHEMA_VERSION,
        "utc_day": str(sidecar.get("utc_day") or "")[:10],
        "source": "provider_outcome_sidecar",
        "sidecar_sequence": max(0, int(sidecar.get("sequence") or 0)),
        "sidecar_document_hash": str(sidecar.get("document_hash") or ""),
        "generated_at": str(sidecar.get("generated_at") or "")[:32],
        "providers": providers,
        "top_error_classes": [
            {"key": key, "count": count}
            for key, count in top_errors.most_common(_MAX_TOP_ERRORS)
        ],
        "day_cache_outcomes": {
            "hit": max(0, int(totals.get("cache_hit_count") or 0)),
            "live": max(0, int(totals.get("live_call_count") or 0)),
        },
        "aggregate_spend": {
            "measured_microusd": max(0, int(totals.get("measured_spend_microusd") or 0)),
            "estimated_microusd": max(0, int(totals.get("estimated_spend_microusd") or 0)),
        },
    }
    if candidate_snapshot_miss_counts:
        digest["candidate_snapshot_miss_counts"] = {
            str(node_id)[:80]: max(0, int(count))
            for node_id, count in list(candidate_snapshot_miss_counts.items())[:20]
        }
        digest["snapshot_miss_note"] = (
            "High snapshot_miss_count means the candidate changed provider request "
            "shapes and dev replay could not score it - treat its dev score as "
            "not-comparable, not as a refuted hypothesis."
        )
    digest["digest_hash"] = sha256_json(
        {key: value for key, value in digest.items() if key != "digest_hash"}
    )
    return digest


def build_run_provider_outcome_digest(
    *,
    candidate_snapshot_miss_counts: Mapping[str, int] | None = None,
) -> dict[str, Any] | None:
    """Load only the proxy's compact sidecar; never scan evidence/cache files."""

    raw = str(os.getenv(PROVIDER_OUTCOME_DIGEST_ENV, "true") or "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return None
    sidecar_path = str(os.getenv(PROVIDER_OUTCOME_SIDECAR_ENV) or "").strip()
    try:
        stale_seconds = int(
            os.getenv(
                PROVIDER_OUTCOME_SIDECAR_STALE_SECONDS_ENV,
                str(DEFAULT_PROVIDER_OUTCOME_SIDECAR_STALE_SECONDS),
            )
            or DEFAULT_PROVIDER_OUTCOME_SIDECAR_STALE_SECONDS
        )
    except (TypeError, ValueError):
        stale_seconds = DEFAULT_PROVIDER_OUTCOME_SIDECAR_STALE_SECONDS
    sidecar = load_provider_outcome_sidecar(
        sidecar_path,
        stale_seconds=max(1, min(86_400, stale_seconds)),
    )
    if sidecar is None:
        return None
    return _digest_from_sidecar(
        sidecar,
        candidate_snapshot_miss_counts=candidate_snapshot_miss_counts,
    )
