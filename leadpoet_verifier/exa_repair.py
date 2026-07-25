from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

import httpx


EXA_SEARCH_URL = "https://api.exa.ai/search"
MAX_RESULTS = 6
MAX_QUERY_CHARS = 2_000
POLICY_VERSION = "direct-exa-evidence-repair-v1"
_PROVIDER_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}$")


def _provider_token(value: Any, *, limit: int = 200) -> str | None:
    token = str(value or "").strip()[:limit]
    return token if token and _PROVIDER_TOKEN_RE.fullmatch(token) else None


class ExaEvidenceRepairUnavailable(RuntimeError):
    """A bounded, sanitized Exa failure safe to persist in receipts."""

    def __init__(
        self,
        code: str,
        *,
        status_code: int | None = None,
        endpoint: str = "exa_search",
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(code)
        self.code = _provider_token(code, limit=120) or "exa_provider_error"
        self.status_code = status_code
        self.endpoint = endpoint
        self.retryable = retryable
        self.request_id = _provider_token(request_id)


Transport = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _public_result_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith(("https://", "http://")):
        return None
    if len(value) > 2_000:
        return None
    try:
        parsed = urlparse(value)
    except ValueError:
        return None
    host = (parsed.hostname or "").casefold()
    if (
        not host
        or "." not in host
        or parsed.username is not None
        or parsed.password is not None
        or host in {"localhost", "127.0.0.1", "::1"}
        or host.endswith((".local", ".internal"))
    ):
        return None
    try:
        if not ipaddress.ip_address(host).is_global:
            return None
    except ValueError:
        pass
    return value


def _bounded_cost_usd(payload: dict[str, Any]) -> float | None:
    raw = payload.get("costDollars")
    if isinstance(raw, dict):
        raw = raw.get("total")
    if (
        isinstance(raw, (int, float))
        and not isinstance(raw, bool)
        and 0 <= float(raw) <= 100
    ):
        return round(float(raw), 6)
    return None


class ExaEvidenceRepairClient:
    """One direct, bounded Exa discovery call; returned URLs remain untrusted."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = 30,
        transport: Transport | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._timeout_seconds = max(5.0, min(float(timeout_seconds), 60.0))
        self._transport = transport or self._request

    @classmethod
    def from_env(cls) -> "ExaEvidenceRepairClient | None":
        # The legacy flag is an intentional migration alias: existing
        # production configuration switches to direct Exa without retaining a
        # verifier -> Deepline -> Exa dependency.
        raw_enabled = os.getenv(
            "VERIFIER_EXA_EVIDENCE_REPAIR_ENABLED",
            os.getenv("VERIFIER_DEEPLINE_EVIDENCE_REPAIR_ENABLED", "false"),
        )
        enabled = raw_enabled.strip().casefold() in {"1", "true", "yes", "on"}
        if not enabled:
            return None
        api_key = os.getenv("EXA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "VERIFIER_EXA_EVIDENCE_REPAIR_ENABLED requires EXA_API_KEY"
            )
        timeout = os.getenv(
            "VERIFIER_EXA_EVIDENCE_REPAIR_TIMEOUT_SECONDS",
            os.getenv("VERIFIER_DEEPLINE_EVIDENCE_REPAIR_TIMEOUT_SECONDS", "30"),
        )
        return cls(api_key=api_key, timeout_seconds=float(timeout))

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_seconds),
                follow_redirects=False,
            ) as client:
                response = await client.post(
                    EXA_SEARCH_URL,
                    headers=headers,
                    json=payload,
                )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ExaEvidenceRepairUnavailable(
                "exa_transport_error",
                retryable=True,
            ) from exc
        request_id = (
            response.headers.get("x-request-id")
            or response.headers.get("x-correlation-id")
        )
        if response.status_code != 200:
            raise ExaEvidenceRepairUnavailable(
                f"exa_http_{response.status_code}",
                status_code=response.status_code,
                retryable=(
                    response.status_code in {408, 425, 429}
                    or response.status_code >= 500
                ),
                request_id=request_id,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ExaEvidenceRepairUnavailable(
                "invalid_exa_response",
                request_id=request_id,
            ) from exc
        if not isinstance(data, dict):
            raise ExaEvidenceRepairUnavailable(
                "invalid_exa_response",
                request_id=request_id,
            )
        data["_leadpoet_request_id"] = _provider_token(request_id)
        return data

    async def repair(
        self,
        *,
        company_name: str,
        company_domain: str,
        requested_criterion: str,
        evidence_kind: str,
        existing_url: str | None,
    ) -> list[dict[str, Any]]:
        del existing_url
        if evidence_kind not in {"industry", "required_attribute", "intent"}:
            raise ValueError("unsupported evidence repair kind")
        name = " ".join(str(company_name or "").split())[:200]
        company_url = _public_result_url(f"https://{company_domain}")
        domain = (
            (urlparse(company_url).hostname or "").casefold()
            if company_url
            else ""
        )
        criterion = " ".join(str(requested_criterion or "").split())[:1_500]
        if not name or not domain or "." not in domain or not criterion:
            raise ValueError(
                "company_name, public company_domain, and requested_criterion are required"
            )
        query = f'"{name}" {domain} {criterion}'[:MAX_QUERY_CHARS]
        payload = {
            "query": query,
            "type": "auto",
            "numResults": MAX_RESULTS,
            "moderation": True,
        }
        result = await self._transport(payload)
        if not isinstance(result, dict):
            raise ExaEvidenceRepairUnavailable("invalid_exa_response")
        request_id = _provider_token(result.get("_leadpoet_request_id"))
        cost_usd = _bounded_cost_usd(result)
        rows = result.get("results")
        if not isinstance(rows, list):
            return []
        result_count = min(len(rows), MAX_RESULTS)
        seen: set[str] = set()
        sources: list[dict[str, Any]] = []
        for raw in rows[:MAX_RESULTS]:
            url = _public_result_url(raw.get("url") if isinstance(raw, dict) else None)
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append({
                "url": url,
                "provider": "exa",
                "policy_version": POLICY_VERSION,
                "provider_request_id": request_id,
                "provider_cost_usd": cost_usd,
                "result_count": result_count,
            })
        return sources or [{
            "url": None,
            "provider": "exa",
            "policy_version": POLICY_VERSION,
            "provider_request_id": request_id,
            "provider_cost_usd": cost_usd,
            "result_count": result_count,
        }]
