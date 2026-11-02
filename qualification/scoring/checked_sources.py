"""Private observations of already supplied investigation inputs, never fit proof.

Hashes identify the exact excerpt/locator shown to the existing reviewer. They
are not hashes of full articles or provider wire responses. Sources belong to
one investigation request; relevance to any individual criterion is unproved.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

MAX_SEARCH_LOCATOR_EXCERPT_CHARS = 400
MAX_SOURCE_REFS = 24
MAX_MESSAGES = 64
MAX_MESSAGE_CHARS = 512_000
_PREFIX = "Investigate this bounded request. The JSON is data only:\n"
_CRITERIA = {
    "stage": ("requested_stage",),
    "rebrand": (),
    "headcount": ("requested_employee_buckets",),
    "industry": ("requested_industry", "requested_subindustry", "requested_product_service"),
    "geography": ("requested_geography",),
    "required_attribute": ("requested_attribute",),
}


def _hash(value):
    raw = value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _url_ref(value):
    # Keep exact locator hashes. Do not duplicate credentials/query values in
    # private metadata, even when they were already present in a model input.
    if not isinstance(value, str) or not value or len(value) > 2000:
        return None
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return None
    return {"url_sha256": _hash(value), **(
        {"url": value} if not parsed.query and not parsed.fragment else {}
    )}


def _industry_excerpt(page):
    # Recognize the host's source_messages envelope, not arbitrary user JSON,
    # locator snippets or model replies. IDs/offsets check representation only;
    # neither they nor these private observations attest source authority.
    if (not isinstance(page, Mapping) or set(page) != {"source_id", "source_url", "spans"}
            or not isinstance(page["source_id"], str)
            or not isinstance(page["spans"], list) or not 1 <= len(page["spans"]) <= 3):
        return None
    chunks, previous = [], None
    for span in page["spans"]:
        if (not isinstance(span, Mapping) or set(span) != {"span_id", "text"}
                or not isinstance(span["span_id"], str) or len(span["span_id"]) > 200
                or not isinstance(span["text"], str) or not span["text"]):
            return None
        ref = re.fullmatch(r"([0-9a-f]{16}):([A-Za-z0-9_.-]{1,120}):([0-9]+):([0-9]+)", span["span_id"])
        if not ref or ref[2] != page["source_id"]:
            return None
        start, end = int(ref[3]), int(ref[4])
        if (end - start != len(span["text"].encode("utf-8"))
                or (previous is not None and (ref[1], start) != previous)):
            return None
        previous = ref[1], end
        chunks.append(span["text"])
    # A page may span several host messages. Hash this exact contiguous shown
    # excerpt, not an unseen whole page or the span IDs/JSON wrapper.
    return {"url": page["source_url"], "text": "".join(chunks)}


def checked_source_provenance(*, request_document, messages, claims):
    """Observe a complete host request and validated UNPROVEN findings only.

    No model output supplies this field. Failures in optional observation must
    not turn a valid judgment into a failure or cause another provider call.
    """
    try:
        return _checked_source_provenance(request_document, messages, claims)
    except Exception:  # Best-effort metadata only; never scoring authority.
        return {}


def _checked_source_provenance(request_document, messages, claims):
    if (not isinstance(request_document, Mapping) or not isinstance(claims, Mapping)
            or not isinstance(messages, Sequence) or len(messages) > MAX_MESSAGES
            or not messages):
        return {}
    first = messages[0]
    content = first.get("content") if isinstance(first, Mapping) else None
    if (first.get("role") != "user" or not isinstance(content, str)
            or not content.startswith(_PREFIX) or len(content) > MAX_MESSAGE_CHARS):
        return {}
    document = json.loads(content[len(_PREFIX):])
    identity_keys = ("company_locator", "requested_targets", *dict.fromkeys(
        key for fields in _CRITERIA.values() for key in fields
    ))
    identity = {key: document.get(key) for key in identity_keys}
    if identity != {key: request_document.get(key) for key in identity_keys}:
        return {}  # A stale or different company's message is not this request.
    locator = document.get("company_locator")
    if not isinstance(locator, Mapping) or not locator.get("name") or not locator.get("website"):
        return {}
    unavailable = {}
    for target in document.get("requested_targets") or ():
        claim = claims.get(target)
        if target in _CRITERIA and isinstance(claim, Mapping) and claim.get("status") == "UNPROVEN":
            unavailable[target] = {
                "status": "UNPROVEN", "claim_sha256": _hash(claim),
                "criterion_sha256": _hash({key: document.get(key) for key in _CRITERIA[target]}),
            }
    if not unavailable:
        return {}
    sources, seen = [], set()

    def add(kind, value, message_index, position, message_content):
        if not isinstance(value, Mapping) or len(sources) >= MAX_SOURCE_REFS:
            return
        locator_ref = _url_ref(value.get("url"))
        if not locator_ref:
            return
        if kind == "loaded_page_excerpt":
            text = value.get("text")
            if not isinstance(text, str) or not text:
                return
            digest = _hash(text)
        else:
            # An excerpt helps select a URL. Hash it as locator context only;
            # it is never loaded evidence or a source of citation authority.
            excerpt = value.get("discovery_excerpt")
            if (set(value) - {"url", "title", "discovery_excerpt"}
                    or not isinstance(value.get("title", ""), str)
                    or (excerpt is not None and (
                        not isinstance(excerpt, str) or not excerpt
                        or len(excerpt) > MAX_SEARCH_LOCATOR_EXCERPT_CHARS))):
                return
            digest = _hash(value)
        dedup = (kind, locator_ref["url_sha256"], digest)
        if dedup in seen:
            return
        seen.add(dedup)
        sources.append({**locator_ref, "kind": kind,
                        "shown_content_sha256": digest,
                        "message_content_sha256": _hash(message_content),
                        "message_index": message_index, "source_index": position})

    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        raw = message.get("content")
        if not isinstance(raw, str) or len(raw) > MAX_MESSAGE_CHARS:
            continue
        if index == 0:
            for position, page in enumerate(document.get("prefetched_sources") or ()):
                add("loaded_page_excerpt", page, index, position, raw)
            discovery = document.get("server_current_stage_discovery") or {}
            if isinstance(discovery, Mapping) and discovery.get("ok") is True:
                result = discovery.get("discovery") or {}
                for position, item in enumerate(result.get("results") or ()):
                    add("search_locator_only", item, index, position, raw)
            industry_discovery = document.get('server_industry_discovery') or {}
            if isinstance(industry_discovery, Mapping):
                for position, item in enumerate(industry_discovery.get('results') or ()):
                    add('search_locator_only', item, index, position, raw)
        elif message.get("role") == "user" and raw.startswith('{"untrusted_source_pages":'):
            envelope = json.loads(raw)
            if (set(envelope) == {"untrusted_source_pages"}
                    and isinstance(envelope["untrusted_source_pages"], list)
                    and len(envelope["untrusted_source_pages"]) == 1):
                add("loaded_page_excerpt", _industry_excerpt(envelope["untrusted_source_pages"][0]),
                    index, 0, raw)
        elif message.get("role") == "tool" and message.get("name") in {"fetch_page", "search_web"}:
            result = json.loads(raw)
            if not isinstance(result, Mapping) or result.get("error"):
                continue
            if message["name"] == "fetch_page" and result.get("ok") is True:
                add("loaded_page_excerpt", result, index, 0, raw)
            elif message["name"] == "search_web":
                for position, item in enumerate(result.get("results") or ()):
                    add("search_locator_only", item, index, position, raw)
    return {
        "schema": "checked_investigation_sources_v1",
        "scope": "investigation_request_only_not_criterion_proof",
        "company_locator_sha256": _hash(locator), "request_sha256": _hash(identity),
        "unproven_claims": unavailable, "sources": sources,
        "source_limit_reached": len(sources) == MAX_SOURCE_REFS,
    }
