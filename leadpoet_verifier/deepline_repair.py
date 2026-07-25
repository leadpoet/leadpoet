from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import quote

import httpx


PLAY_NAME = "leadpoet-company-evidence-repair"
TERMINAL_SUCCESS = {"completed", "complete", "succeeded", "success"}
TERMINAL_FAILURE = {"failed", "error", "cancelled", "canceled", "timed_out", "timeout"}


class DeeplineEvidenceRepairUnavailable(RuntimeError):
    """A bounded, sanitized Deepline failure safe to persist in receipts."""

    def __init__(
        self,
        code: str,
        *,
        status_code: int | None = None,
        endpoint: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.endpoint = endpoint
        self.retryable = retryable

    def receipt(self) -> dict[str, Any]:
        return {
            "reason_code": self.code,
            "status_code": self.status_code,
            "endpoint": self.endpoint,
            "retryable": self.retryable,
        }


Transport = Callable[
    [str, str, dict[str, Any] | None],
    Awaitable[dict[str, Any]],
]


def _first_text(value: Any, keys: tuple[str, ...]) -> str | None:
    if not isinstance(value, dict):
        return None
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def _run_id(value: dict[str, Any]) -> str | None:
    return _first_text(value, ("workflowId", "workflow_id", "runId", "run_id", "id"))


def _status(value: dict[str, Any]) -> str:
    return (_first_text(value, ("status", "run_status", "state")) or "").casefold()


def _sources(value: Any, depth: int = 0) -> list[dict[str, Any]]:
    if depth > 8:
        return []
    if isinstance(value, dict):
        raw = value.get("sources")
        if isinstance(raw, list):
            materialized = [item for item in raw if isinstance(item, dict)]
            if materialized and any(item.get("url") for item in materialized):
                return materialized[:6]
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


class DeeplineEvidenceRepairClient:
    def __init__(
        self,
        *,
        api_key: str,
        host_url: str = "https://code.deepline.com",
        timeout_seconds: float = 90,
        poll_seconds: float = 2,
        transport: Transport | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._host_url = host_url.rstrip("/")
        self._timeout_seconds = max(10.0, min(timeout_seconds, 180.0))
        self._poll_seconds = max(0.1, min(poll_seconds, 10.0))
        self._transport = transport or self._request

    @classmethod
    def from_env(cls) -> "DeeplineEvidenceRepairClient | None":
        enabled = os.getenv(
            "VERIFIER_DEEPLINE_EVIDENCE_REPAIR_ENABLED", "false"
        ).strip().casefold() in {"1", "true", "yes", "on"}
        if not enabled:
            return None
        api_key = os.getenv("DEEPLINE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "VERIFIER_DEEPLINE_EVIDENCE_REPAIR_ENABLED requires DEEPLINE_API_KEY"
            )
        return cls(
            api_key=api_key,
            host_url=os.getenv("DEEPLINE_HOST_URL", "https://code.deepline.com"),
            timeout_seconds=float(
                os.getenv("VERIFIER_DEEPLINE_EVIDENCE_REPAIR_TIMEOUT_SECONDS", "90")
            ),
            poll_seconds=float(
                os.getenv("VERIFIER_DEEPLINE_EVIDENCE_REPAIR_POLL_SECONDS", "2")
            ),
        )

    async def _request(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(min(self._timeout_seconds, 60.0))
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.request(method, url, headers=headers, json=body)
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                raise DeeplineEvidenceRepairUnavailable(
                    "invalid_deepline_response",
                ) from exc
        if not isinstance(payload, dict):
            raise DeeplineEvidenceRepairUnavailable("invalid_deepline_response")
        return payload

    async def _call_transport(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None,
        *,
        endpoint: str,
    ) -> dict[str, Any]:
        """Call Deepline with one bounded retry for transient failures only."""

        for attempt in range(2):
            try:
                payload = await self._transport(method, url, body)
                if not isinstance(payload, dict):
                    raise DeeplineEvidenceRepairUnavailable(
                        "invalid_deepline_response",
                        endpoint=endpoint,
                    )
                return payload
            except DeeplineEvidenceRepairUnavailable:
                raise
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                retryable = status_code == 429 or status_code >= 500
                if retryable and attempt == 0:
                    await asyncio.sleep(0.25)
                    continue
                raise DeeplineEvidenceRepairUnavailable(
                    f"deepline_http_{status_code}",
                    status_code=status_code,
                    endpoint=endpoint,
                    retryable=retryable,
                ) from exc
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == 0:
                    await asyncio.sleep(0.25)
                    continue
                raise DeeplineEvidenceRepairUnavailable(
                    "deepline_transport_error",
                    endpoint=endpoint,
                    retryable=True,
                ) from exc
        raise DeeplineEvidenceRepairUnavailable(
            "deepline_transport_error",
            endpoint=endpoint,
            retryable=True,
        )

    async def repair(
        self,
        *,
        company_name: str,
        company_domain: str,
        requested_criterion: str,
        evidence_kind: str,
        existing_url: str | None,
    ) -> list[dict[str, Any]]:
        if evidence_kind not in {"industry", "required_attribute", "intent"}:
            raise ValueError("unsupported evidence repair kind")
        payload = {
            "name": PLAY_NAME,
            "input": {
                "company_name": company_name[:200],
                "company_domain": company_domain[:500],
                "requested_criterion": requested_criterion[:2_000],
                "evidence_kind": evidence_kind,
                "existing_url": existing_url[:2_000] if existing_url else None,
            },
        }
        started = await self._call_transport(
            "POST",
            f"{self._host_url}/api/v2/plays/run",
            payload,
            endpoint="plays_run_start",
        )
        found = _sources(started)
        if found:
            return found
        workflow_id = _run_id(started)
        status = _status(started)
        if status in TERMINAL_FAILURE:
            raise DeeplineEvidenceRepairUnavailable("deepline_repair_failed")
        if not workflow_id:
            raise DeeplineEvidenceRepairUnavailable("missing_deepline_workflow_id")

        deadline = time.monotonic() + self._timeout_seconds
        encoded = quote(workflow_id, safe="")
        while time.monotonic() < deadline:
            await asyncio.sleep(self._poll_seconds)
            try:
                snapshot = await self._call_transport(
                    "GET",
                    f"{self._host_url}/api/v2/runs/{encoded}?full=true",
                    None,
                    endpoint="runs_status",
                )
            except DeeplineEvidenceRepairUnavailable as exc:
                if exc.status_code != 404:
                    raise
                snapshot = await self._call_transport(
                    "GET",
                    f"{self._host_url}/api/v2/plays/run/{encoded}?full=true",
                    None,
                    endpoint="plays_run_status",
                )
            found = _sources(snapshot)
            if found:
                return found
            status = _status(snapshot)
            if status in TERMINAL_SUCCESS:
                return []
            if status in TERMINAL_FAILURE:
                raise DeeplineEvidenceRepairUnavailable("deepline_repair_failed")
        raise DeeplineEvidenceRepairUnavailable("deepline_repair_timeout")
