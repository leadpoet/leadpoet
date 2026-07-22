"""Deepline evidence repair for Research Lab intent verification.

When every intent signal on an autoresearch company scores zero, the company
is zeroed as a false positive — but a real company whose submitted evidence
URL was weak (aggregator page, dead link, wrong article) dies as a false
negative. Before finalizing that zero, this client asks the
``leadpoet-company-evidence-repair`` play for a small, bounded pack of
better evidence sources for the same claim; the caller re-verifies against
those sources through the normal scorer, which remains the only authority.

Lab-scoped and feature-gated: requires
``RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_ENABLED`` plus ``DEEPLINE_API_KEY``,
otherwise ``repair_sources`` returns [] and scoring behaves exactly as
before. Runs are agentic workflows that legitimately take on the order of a
minute — the budget is 90s (bounded 10-180) with 2s polls; short budgets
abandon runs that were about to settle and re-pay for the same lookup later.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

PLAY_NAME = "leadpoet-company-evidence-repair"
_TERMINAL_OK = {"completed", "complete", "succeeded", "success"}
_TERMINAL_FAIL = {"failed", "error", "cancelled", "canceled", "timed_out", "timeout"}

MAX_SOURCES = 2  # each repaired source pays a full re-verification


def enabled() -> bool:
    flag = os.environ.get(
        "RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_ENABLED", ""
    ).strip().lower()
    return flag in {"1", "true", "yes", "on"} and bool(
        os.environ.get("DEEPLINE_API_KEY", "").strip()
    )


def _budget_s() -> float:
    try:
        raw = float(
            os.environ.get("RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_TIMEOUT_SECONDS", "90")
            or "90"
        )
    except ValueError:
        raw = 90.0
    # Provider-published completion times: median 15s, P95 84s, P99 184s;
    # recovery polling is sanctioned up to 5 minutes.
    return max(10.0, min(raw, 300.0))


def _poll_s() -> float:
    try:
        raw = float(
            os.environ.get("RESEARCH_LAB_DEEPLINE_EVIDENCE_REPAIR_POLL_SECONDS", "2")
            or "2"
        )
    except ValueError:
        raw = 2.0
    return max(0.1, min(raw, 10.0))


def _host() -> str:
    return os.environ.get("DEEPLINE_HOST_URL", "https://code.deepline.com").rstrip("/")


def _first_text(value: Any, keys: tuple) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def _run_id(payload: Dict[str, Any]) -> Optional[str]:
    nested = payload.get("run") if isinstance(payload.get("run"), dict) else None
    for source in (nested, payload):
        found = _first_text(source, ("id", "workflowId", "workflow_id", "runId", "run_id"))
        if found:
            return found
    return None


def _status(payload: Dict[str, Any]) -> str:
    nested = payload.get("run") if isinstance(payload.get("run"), dict) else None
    for source in (nested, payload):
        found = _first_text(source, ("status", "run_status", "state"))
        if found:
            return found.casefold()
    return ""


def _sources(value: Any, depth: int = 0) -> List[Dict[str, Any]]:
    """Recursive scan for a plausible sources list anywhere in the payload."""
    if depth > 8:
        return []
    if isinstance(value, dict):
        raw = value.get("sources")
        if isinstance(raw, list):
            found = [item for item in raw if isinstance(item, dict) and item.get("url")]
            if found:
                return found[:MAX_SOURCES]
        for nested in value.values():
            found = _sources(nested, depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value[:50]:
            found = _sources(nested, depth + 1)
            if found:
                return found
    return []


async def _request(
    client: httpx.AsyncClient, method: str, url: str, body: Optional[dict]
) -> Dict[str, Any]:
    headers = {
        "Authorization": "Bearer " + os.environ.get("DEEPLINE_API_KEY", "").strip(),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(method, url, headers=headers, json=body)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


async def _call(
    client: httpx.AsyncClient, method: str, url: str, body: Optional[dict]
) -> Dict[str, Any]:
    """One bounded retry for transient failures only; config errors fail fast."""
    for attempt in range(2):
        try:
            return await _request(client, method, url, body)
        except httpx.HTTPStatusError as exc:
            transient = exc.response.status_code == 429 or exc.response.status_code >= 500
            if transient and attempt == 0:
                await asyncio.sleep(0.25)
                continue
            raise
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == 0:
                await asyncio.sleep(0.25)
                continue
            raise
    return {}


async def repair_sources(
    *,
    company_name: str,
    company_domain: str,
    requested_criterion: str,
    evidence_kind: str = "intent",
    existing_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Ask for replacement evidence sources. Returns [] on any failure —
    the caller's zero stands unless repair produces something verifiable."""
    if not enabled():
        return []
    if evidence_kind not in {"industry", "required_attribute", "intent"}:
        return []
    payload = {
        "name": PLAY_NAME,
        "input": {
            "company_name": (company_name or "")[:200],
            "company_domain": (company_domain or "")[:500],
            "requested_criterion": (requested_criterion or "")[:2000],
            "evidence_kind": evidence_kind,
            "existing_url": (existing_url or "")[:2000] or None,
        },
    }
    deadline = time.monotonic() + _budget_s()
    try:
        timeout = httpx.Timeout(min(_budget_s(), 60.0))
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            started = await _call(
                client, "POST", _host() + "/api/v2/plays/run", payload
            )
            found = _sources(started)
            if found:
                return found
            if _status(started) in _TERMINAL_FAIL:
                logger.warning("deepline_evidence_repair_failed company=%s", company_name)
                return []
            workflow_id = _run_id(started)
            if not workflow_id:
                logger.warning("deepline_evidence_repair_no_run_id company=%s", company_name)
                return []
            encoded = quote(workflow_id, safe="")
            poll_url = _host() + f"/api/v2/runs/{encoded}?full=true"
            fallback_url = _host() + f"/api/v2/plays/run/{encoded}?full=true"
            while time.monotonic() < deadline:
                await asyncio.sleep(_poll_s())
                try:
                    snapshot = await _call(client, "GET", poll_url, None)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404:
                        poll_url, fallback_url = fallback_url, poll_url
                        continue
                    raise
                found = _sources(snapshot)
                if found:
                    return found
                status = _status(snapshot)
                if status in _TERMINAL_OK:
                    return []
                if status in _TERMINAL_FAIL:
                    logger.warning(
                        "deepline_evidence_repair_failed company=%s", company_name
                    )
                    return []
            logger.warning(
                "deepline_evidence_repair_timeout company=%s budget=%.0fs",
                company_name,
                _budget_s(),
            )
    except Exception as exc:  # noqa: BLE001 — repair must never break scoring
        logger.warning(
            "deepline_evidence_repair_error company=%s error=%s",
            company_name,
            str(exc)[:160],
        )
    return []
