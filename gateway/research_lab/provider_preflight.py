"""Provider preflight probes for Research Lab scoring and paid loops.

Before the scoring worker starts a daily baseline (or claims a candidate)
and before the hosted worker claims a paid loop ticket, a cheap cached probe
checks that ScrapingDog and Exa are reachable and credited. A provider that
is out of credits (HTTP 402), quota-exhausted (429), auth-broken (401/403),
or persistently unreachable would otherwise turn the whole run into zeros
and burn the miner's budget on work that cannot succeed.

Verdicts are cached per process for a TTL so probes stay cheap. On a
credit/quota/auth failure — or a streak of transport/5xx failures — the
probe auto-pauses the relevant maintenance control (scoring or
autoresearch) with a ``provider_preflight:`` reason marker. When a later
probe comes back healthy, a pause that carries that marker (and only such a
pause) is auto-resumed, so operator-set pauses are never overridden.

Provider request rejections (404/400/other 4xx) count as HEALTHY: the
provider answered, which is all the preflight needs to know.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

PREFLIGHT_REASON_PREFIX = "provider_preflight:"
_PAUSE_STATUSES = {401, 402, 403, 429}
_QUOTA_TEXT_MARKERS = (
    "out of credits",
    "insufficient credits",
    "payment required",
    "quota",
    "rate limit",
    "too many requests",
)


def _env_flag(name: str, default: str) -> bool:
    return str(os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def preflight_enabled() -> bool:
    return _env_flag("RESEARCH_LAB_PROVIDER_PREFLIGHT_ENABLED", "true")


def preflight_auto_pause_enabled() -> bool:
    return _env_flag("RESEARCH_LAB_PROVIDER_PREFLIGHT_AUTO_PAUSE", "true")


def preflight_auto_resume_enabled() -> bool:
    return _env_flag("RESEARCH_LAB_PROVIDER_PREFLIGHT_AUTO_RESUME", "true")


def _preflight_ttl_seconds() -> float:
    try:
        return max(60.0, float(os.getenv("RESEARCH_LAB_PROVIDER_PREFLIGHT_TTL_SECONDS", "600")))
    except ValueError:
        return 600.0


def _preflight_timeout_seconds() -> float:
    try:
        return max(2.0, float(os.getenv("RESEARCH_LAB_PROVIDER_PREFLIGHT_TIMEOUT_SECONDS", "12")))
    except ValueError:
        return 12.0


def _preflight_failure_streak_threshold() -> int:
    try:
        return max(1, int(os.getenv("RESEARCH_LAB_PROVIDER_PREFLIGHT_FAILURE_STREAK", "3")))
    except ValueError:
        return 3


def provider_preflight_settings() -> dict[str, Any]:
    """Return the existing non-secret runtime knobs as a canonical document."""

    return {
        "enabled": preflight_enabled(),
        "ttl_seconds": _preflight_ttl_seconds(),
        "timeout_seconds": _preflight_timeout_seconds(),
        "failure_streak_threshold": _preflight_failure_streak_threshold(),
    }


def _scrapingdog_key() -> str:
    for name in ("RESEARCH_LAB_SCRAPINGDOG_API_KEY", "SCRAPINGDOG_API_KEY"):
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
    return ""


def _exa_key() -> str:
    for name in ("RESEARCH_LAB_BENCHMARK_EXA_API_KEY", "RESEARCH_LAB_EXA_API_KEY", "EXA_API_KEY"):
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
    return ""


@dataclass
class ProviderVerdict:
    provider: str
    healthy: bool
    status: str  # healthy | no_credential | credit_or_auth | transport_failure
    detail: str = ""
    http_status: int = 0
    checked_at: float = field(default_factory=time.time)

    def to_doc(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "healthy": self.healthy,
            "status": self.status,
            "detail": self.detail[:300],
            "http_status": self.http_status,
        }


def _classify_http_error(provider: str, exc: urllib.error.HTTPError) -> ProviderVerdict:
    body = ""
    try:
        body = exc.read().decode("utf-8", "replace")[:500]
    except Exception:  # noqa: BLE001 - probe must never raise
        body = ""
    lowered = f"{exc.reason} {body}".lower()
    if exc.code in _PAUSE_STATUSES or any(marker in lowered for marker in _QUOTA_TEXT_MARKERS):
        return ProviderVerdict(
            provider=provider,
            healthy=False,
            status="credit_or_auth",
            detail=f"HTTP {exc.code}: {str(exc.reason)[:120]}",
            http_status=int(exc.code),
        )
    if exc.code >= 500:
        return ProviderVerdict(
            provider=provider,
            healthy=False,
            status="transport_failure",
            detail=f"HTTP {exc.code}: {str(exc.reason)[:120]}",
            http_status=int(exc.code),
        )
    # Any other 4xx means the provider answered a (possibly malformed) probe
    # request — reachable and credited enough to reject it.
    return ProviderVerdict(
        provider=provider,
        healthy=True,
        status="healthy",
        detail=f"HTTP {exc.code} (request rejected, provider up)",
        http_status=int(exc.code),
    )


def probe_scrapingdog(timeout_seconds: float | None = None) -> ProviderVerdict:
    """Account-status probe: cheap, authenticated, no scrape credits burned."""
    key = _scrapingdog_key()
    if not key:
        return ProviderVerdict(provider="scrapingdog", healthy=True, status="no_credential")
    url = "https://api.scrapingdog.com/account?" + urllib.parse.urlencode({"api_key": key})
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="GET"),
            timeout=timeout_seconds or _preflight_timeout_seconds(),
        ) as response:
            status = int(getattr(response, "status", None) or getattr(response, "code", 0) or 0)
            body = response.read().decode("utf-8", "replace")[:500]
            lowered = body.lower()
            if any(marker in lowered for marker in ("out of credits", "insufficient credits")):
                return ProviderVerdict(
                    provider="scrapingdog",
                    healthy=False,
                    status="credit_or_auth",
                    detail="account reports exhausted credits",
                    http_status=status,
                )
            return ProviderVerdict(
                provider="scrapingdog", healthy=True, status="healthy", http_status=status
            )
    except urllib.error.HTTPError as exc:
        return _classify_http_error("scrapingdog", exc)
    except Exception as exc:  # noqa: BLE001 - URLError, timeout, resets
        return ProviderVerdict(
            provider="scrapingdog",
            healthy=False,
            status="transport_failure",
            detail=str(exc)[:200],
        )


def probe_exa(timeout_seconds: float | None = None) -> ProviderVerdict:
    """Minimal authenticated search (numResults=1): surfaces 402/429 directly."""
    key = _exa_key()
    if not key:
        return ProviderVerdict(provider="exa", healthy=True, status="no_credential")
    payload = json.dumps({"query": "provider preflight", "numResults": 1}).encode("utf-8")
    request = urllib.request.Request(
        "https://api.exa.ai/search",
        data=payload,
        headers={"x-api-key": key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout_seconds or _preflight_timeout_seconds()
        ) as response:
            status = int(getattr(response, "status", None) or getattr(response, "code", 0) or 0)
            response.read()
            return ProviderVerdict(provider="exa", healthy=True, status="healthy", http_status=status)
    except urllib.error.HTTPError as exc:
        return _classify_http_error("exa", exc)
    except Exception as exc:  # noqa: BLE001
        return ProviderVerdict(
            provider="exa",
            healthy=False,
            status="transport_failure",
            detail=str(exc)[:200],
        )


_PROBES: dict[str, Callable[[float | None], ProviderVerdict]] = {
    "scrapingdog": probe_scrapingdog,
    "exa": probe_exa,
}


class ProviderPreflight:
    """Per-process cached preflight with failure-streak tracking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._verdicts: dict[str, ProviderVerdict] = {}
        self._failure_streaks: dict[str, int] = {}

    def check(
        self,
        *,
        force: bool = False,
        settings: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return {"healthy": bool, "verdicts": [...], "pause_worthy": bool}."""
        effective = dict(settings or provider_preflight_settings())
        if not bool(effective.get("enabled")):
            return {"healthy": True, "verdicts": [], "pause_worthy": False, "disabled": True}
        ttl = max(60.0, float(effective["ttl_seconds"]))
        timeout = max(2.0, float(effective["timeout_seconds"]))
        now = time.time()
        verdicts: list[ProviderVerdict] = []
        for name, probe in _PROBES.items():
            with self._lock:
                cached = self._verdicts.get(name)
            if cached is not None and not force and now - cached.checked_at < ttl:
                verdicts.append(cached)
                continue
            verdict = probe(timeout)
            with self._lock:
                self._verdicts[name] = verdict
                if verdict.status == "transport_failure":
                    self._failure_streaks[name] = self._failure_streaks.get(name, 0) + 1
                else:
                    self._failure_streaks[name] = 0
            if not verdict.healthy:
                logger.warning(
                    "research_lab_provider_preflight_unhealthy provider=%s status=%s detail=%s streak=%d",
                    verdict.provider,
                    verdict.status,
                    verdict.detail[:200],
                    self._failure_streaks.get(name, 0),
                )
            verdicts.append(verdict)
        streak_threshold = max(1, int(effective["failure_streak_threshold"]))
        pause_worthy = False
        healthy = True
        for verdict in verdicts:
            if verdict.status == "credit_or_auth":
                healthy = False
                pause_worthy = True
            elif verdict.status == "transport_failure":
                healthy = False
                if self._failure_streaks.get(verdict.provider, 0) >= streak_threshold:
                    pause_worthy = True
        return {
            "healthy": healthy,
            "pause_worthy": pause_worthy,
            "verdicts": [verdict.to_doc() for verdict in verdicts],
        }


_shared_preflight = ProviderPreflight()


def shared_preflight() -> ProviderPreflight:
    return _shared_preflight


async def preflight_gate(
    *,
    scope: str,
    actor_ref: str,
    is_paused: Callable[[], Any],
    set_paused: Callable[..., Any],
    worker_index: int = 0,
    authority_check: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Async preflight gate for one maintenance scope (scoring/autoresearch).

    Returns {"proceed": bool, "reason": str, "verdicts": [...]}. When the
    verdict is pause-worthy and auto-pause is on, pauses the scope with a
    ``provider_preflight:`` marker. When healthy and the scope was paused by
    a previous preflight (marker present), auto-resumes it. Operator pauses
    (no marker) are never resumed here.
    """
    if authority_check is None:
        from gateway.research_lab.v2_authority import (
            execute_provider_preflight_v2,
        )

        authority_check = execute_provider_preflight_v2
    result = authority_check(
        scope_key="%s:%s" % (str(scope), str(actor_ref)),
        worker_index=int(worker_index),
        settings=provider_preflight_settings(),
        force=False,
        provider_credential_profile="benchmark_model",
    )
    if hasattr(result, "__await__"):
        result = await result
    if not isinstance(result, Mapping):
        raise RuntimeError("attested provider preflight result is invalid")
    if result.get("disabled"):
        return {"proceed": True, "reason": "preflight_disabled", "verdicts": []}
    verdicts = result.get("verdicts") or []
    if result.get("healthy"):
        if preflight_auto_resume_enabled():
            try:
                state = await is_paused()
                if isinstance(state, dict):
                    paused = bool(state.get("paused"))
                    reason = str(state.get("reason") or "")
                else:
                    paused = bool(getattr(state, "paused", False))
                    reason = str(getattr(state, "reason", "") or "")
                if paused and reason.startswith(PREFLIGHT_REASON_PREFIX):
                    await set_paused(
                        paused=False,
                        reason=f"{PREFLIGHT_REASON_PREFIX}providers_recovered",
                        actor_ref=actor_ref,
                        event_doc={"scope": scope, "verdicts": verdicts},
                    )
                    logger.warning(
                        "research_lab_provider_preflight_auto_resumed scope=%s", scope
                    )
            except Exception as exc:  # noqa: BLE001 - resume is best-effort
                logger.warning(
                    "research_lab_provider_preflight_resume_check_failed scope=%s error=%s",
                    scope,
                    str(exc)[:200],
                )
        return {"proceed": True, "reason": "healthy", "verdicts": verdicts}
    if result.get("pause_worthy") and preflight_auto_pause_enabled():
        unhealthy = [v for v in verdicts if not v.get("healthy")]
        marker = ",".join(
            f"{v.get('provider')}={v.get('status')}" for v in unhealthy
        )
        try:
            await set_paused(
                paused=True,
                reason=f"{PREFLIGHT_REASON_PREFIX}{marker}"[:200],
                actor_ref=actor_ref,
                event_doc={"scope": scope, "verdicts": verdicts},
            )
            logger.warning(
                "research_lab_provider_preflight_auto_paused scope=%s providers=%s",
                scope,
                marker,
            )
        except Exception as exc:  # noqa: BLE001 - pause is best-effort
            logger.warning(
                "research_lab_provider_preflight_pause_failed scope=%s error=%s",
                scope,
                str(exc)[:200],
            )
    return {"proceed": False, "reason": "provider_preflight_unhealthy", "verdicts": verdicts}
