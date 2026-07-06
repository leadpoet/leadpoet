"""Provider evidence cache: pin candidate provider I/O to baseline evidence.

The daily baseline run already records every in-container provider call
(request and response) in its trace artifact. This module turns those trace
entries into a per-ICP request-fingerprint -> canonical-response cache. When
a cache is handed to a candidate container (see the provider diagnostics
bootstrap in private_runtime.py), a candidate that issues the same provider
request as the baseline receives the recorded canonical baseline response
instead of a fresh live call, so provider result variance between the
baseline run and a later candidate run cannot change the comparison. Requests
without a recorded response fall through to a live call and are marked in the
trace as cache misses, which downstream scoring uses to distinguish
same-evidence evaluations from fresh-evidence evaluations.

Each fingerprint maps to exactly one canonical response: the baseline's
settled outcome for that request. When the baseline recorded several outcomes
for one fingerprint (a retried transient failure followed by a success, or a
repeated sampling prompt with several completions), the last success wins,
and only a request that never succeeded keeps its final failure. Replay is
therefore fully deterministic and order-independent — the same request always
returns the same response, transient failures the baseline retried through
are never replayed, and repeated sampling of one prompt cannot reintroduce
output variance.

Caches are built per ICP and must be mounted per ICP: a single cache spanning
the whole window would hand every candidate container the recorded evidence
for all sealed ICPs at once, which candidate code could read directly. A
per-ICP cache only ever exposes evidence the running candidate could obtain
itself by issuing the same requests live.

The fingerprint canonicalization here MUST stay in sync with the inline
implementation inside _PROVIDER_DIAGNOSTICS_BOOTSTRAP (private_runtime.py):
the bootstrap computes fingerprints on raw in-container request values, this
module computes them on sanitized trace values, and the two only agree
because credential-bearing query parameters are dropped and provider request
bodies carry no secrets.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
import urllib.parse
from typing import Any, Iterable, Mapping

EVIDENCE_CACHE_PATH_ENV = "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH"
EVIDENCE_CACHE_SCHEMA_VERSION = "1.1"

# Query parameters that carry credentials or per-run noise; excluded from the
# fingerprint so redacted trace URLs and raw in-container URLs canonicalize to
# the same value.
_AUTH_QUERY_PARAMS = frozenset(
    {"api_key", "apikey", "api-key", "key", "token", "access_token", "x-api-key"}
)

# The trace writer caps recorded URLs at 2000 characters. A capped URL cannot
# be fingerprinted faithfully (its query string may be cut mid-parameter), so
# entries at or beyond this length are not cacheable from trace data.
_TRACE_URL_CAP = 2000


def canonical_request_fingerprint(method: str, url: str, body: bytes | str | None) -> str:
    """Stable fingerprint of one provider request.

    method: upper-cased. url: scheme/host lower-cased, credential query params
    dropped, remaining query params sorted. body: canonical JSON when the body
    parses as JSON, raw bytes otherwise.
    """
    method_part = str(method or "GET").upper()
    try:
        split = urllib.parse.urlsplit(str(url or ""))
        pairs = [
            (k, v)
            for k, v in urllib.parse.parse_qsl(split.query, keep_blank_values=True)
            if k.lower() not in _AUTH_QUERY_PARAMS
        ]
        pairs.sort()
        url_part = "|".join(
            (
                (split.scheme or "").lower(),
                (split.netloc or "").lower(),
                split.path or "",
                urllib.parse.urlencode(pairs),
            )
        )
    except Exception:
        url_part = str(url or "")
    if body is None:
        body_bytes = b""
    elif isinstance(body, (bytes, bytearray)):
        body_bytes = bytes(body)
    else:
        body_bytes = str(body).encode("utf-8", "replace")
    if body_bytes:
        try:
            parsed = json.loads(body_bytes.decode("utf-8"))
            body_bytes = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        except (ValueError, UnicodeDecodeError):
            pass
    digest = hashlib.sha256()
    digest.update(method_part.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(url_part.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(body_bytes)
    return digest.hexdigest()


def icp_evidence_cache_key(canonical_icp: Mapping[str, Any]) -> str:
    """Cache-file key for one ICP.

    Both sides derive this from the exact payload the runner sends the
    container (``canonicalize_private_model_icp`` output): the worker names
    each per-ICP cache file with this key, and the runner picks the file for
    the ICP in its stdin payload by recomputing it.
    """
    encoded = json.dumps(dict(canonical_icp), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _entry_request_is_complete(entry: Mapping[str, Any]) -> bool:
    """True when the recorded request reproduces the raw request faithfully.

    A fingerprint computed from a partially recorded request can never match
    the live request, so such entries are not cacheable. Requests with no body
    are complete by definition.
    """
    url = str(entry.get("url_redacted") or "")
    if len(url) >= _TRACE_URL_CAP:
        return False
    body_b64 = str(entry.get("request_body_b64") or "")
    byte_len = entry.get("request_byte_len")
    if not isinstance(byte_len, int) or byte_len <= 0:
        return not body_b64
    if not body_b64:
        return False
    try:
        return len(base64.b64decode(body_b64)) == byte_len
    except Exception:
        return False


def build_evidence_cache_from_trace_entries(
    entries: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fold trace entries into fingerprint -> canonical recorded response.

    Only replayable entries are kept: a faithfully recorded request, a usable
    status, and, for successes, a complete (untruncated) response body. Error
    responses (4xx/5xx) are cached body-optional so a request that failed for
    the baseline replays the same failure for candidates. When a fingerprint
    recorded several outcomes, the settled one wins: the last success if any
    exists, otherwise the final failure — so transient failures the baseline
    retried through never replay, and repeated sampling of one prompt yields
    one deterministic completion.
    """
    cache: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("phase") or "call") not in ("call", ""):
            continue
        status = entry.get("response_status")
        if not isinstance(status, int):
            continue
        if not _entry_request_is_complete(entry):
            continue
        body_b64 = str(entry.get("response_body_b64") or "")
        if status < 400:
            # A success is only replayable with its complete body.
            if not body_b64 or bool(entry.get("truncated")):
                continue
            byte_len = entry.get("response_byte_len")
            try:
                if isinstance(byte_len, int) and byte_len >= 0:
                    if len(base64.b64decode(body_b64)) != byte_len:
                        continue
            except Exception:
                continue
        fingerprint = canonical_request_fingerprint(
            str(entry.get("method") or "GET"),
            str(entry.get("url_redacted") or ""),
            base64.b64decode(entry.get("request_body_b64") or "") or None,
        )
        outcome = str(entry.get("outcome") or "")
        record = {
            "status": int(status),
            "body_b64": body_b64,
            "outcome": outcome or ("error" if status >= 400 else "success"),
        }
        existing = cache.get(fingerprint)
        # Settled-outcome rule: a success always supersedes and later
        # outcomes of the same kind supersede earlier ones; a failure never
        # replaces a success.
        if existing is None or status < 400 or existing["status"] >= 400:
            cache[fingerprint] = record
    return cache


def merge_evidence_caches(
    *caches: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """First cache to record a fingerprint owns its canonical response."""
    merged: dict[str, dict[str, Any]] = {}
    for cache in caches:
        for fingerprint, record in cache.items():
            if fingerprint in merged:
                continue
            normalized = _normalize_record(record)
            if normalized is not None:
                merged[str(fingerprint)] = normalized
    return merged


def _normalize_record(record: Any) -> dict[str, Any] | None:
    """Accept a canonical record; settle a legacy sequence to one record."""
    if isinstance(record, list):
        settled = None
        for item in record:
            if not isinstance(item, Mapping) or not isinstance(item.get("status"), int):
                continue
            if settled is None or item["status"] < 400 or settled["status"] >= 400:
                settled = dict(item)
        return settled
    if isinstance(record, Mapping) and isinstance(record.get("status"), int):
        return dict(record)
    return None


def save_evidence_cache(
    cache: Mapping[str, Any],
    path: str,
    *,
    window_hash: str = "",
    icp_ref: str = "",
) -> None:
    doc = {
        "schema_version": EVIDENCE_CACHE_SCHEMA_VERSION,
        "rolling_window_hash": window_hash,
        "icp_ref": icp_ref,
        # Evidence is valid only for the UTC day it was recorded: at 00:00
        # UTC the window rotates and every input must be re-recorded fresh.
        "utc_day": time.strftime("%Y-%m-%d", time.gmtime()),
        "entries": {
            str(k): v
            for k, v in ((str(k), _normalize_record(v)) for k, v in cache.items())
            if v is not None
        },
    }
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, sort_keys=True, separators=(",", ":"))
    os.replace(tmp_path, path)


def load_evidence_cache(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if isinstance(doc, Mapping):
        stamp = str(doc.get("utc_day") or "")
        if stamp and stamp != time.strftime("%Y-%m-%d", time.gmtime()):
            return {}
    entries = doc.get("entries") if isinstance(doc, Mapping) else None
    if not isinstance(entries, Mapping):
        return {}
    loaded: dict[str, dict[str, Any]] = {}
    for key, record in entries.items():
        normalized = _normalize_record(record)
        if normalized is not None:
            loaded[str(key)] = normalized
    return loaded


def build_cache_file_from_trace_files(
    trace_paths: Iterable[str],
    out_path: str,
    *,
    window_hash: str = "",
    icp_ref: str = "",
) -> int:
    """CLI/worker helper: fold one or more trace artifacts into a cache file."""
    caches = []
    for trace_path in trace_paths:
        with open(trace_path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
        entries = doc.get("entries") if isinstance(doc, Mapping) else doc
        caches.append(build_evidence_cache_from_trace_entries(entries or []))
    merged = merge_evidence_caches(*caches)
    save_evidence_cache(merged, out_path, window_hash=window_hash, icp_ref=icp_ref)
    return len(merged)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a provider evidence cache from trace artifacts")
    parser.add_argument("--trace", action="append", required=True, help="trace JSON file (repeatable)")
    parser.add_argument("--out", required=True, help="output cache file path")
    parser.add_argument("--window-hash", default="", help="rolling window hash label")
    parser.add_argument("--icp-ref", default="", help="ICP reference label")
    args = parser.parse_args(argv)
    count = build_cache_file_from_trace_files(
        args.trace, args.out, window_hash=args.window_hash, icp_ref=args.icp_ref
    )
    print(json.dumps({"entries": count, "out": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
