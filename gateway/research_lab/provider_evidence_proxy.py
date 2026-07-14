"""Host-side provider evidence proxy: record-once, replay-for-all.

Scoring containers send their provider traffic here instead of calling
providers directly (the in-container bootstrap rewrites provider hosts to
this proxy, and containers carry no provider credentials). For each request
the proxy answers from, in order:

1. the baseline tape — the reference run's recorded evidence for the day;
2. the shared day cache — responses already recorded today for the same
   request by any earlier run;
3. the live provider — called once, and the response is recorded into the
   shared day cache so every later identical request replays it.

Because recording happens at the host boundary, the day cache only ever
contains what providers actually returned.
Everything expires at 00:00 UTC (the recording's utc_day must match today),
after which the same inputs are recorded fresh.

Credentials live only in this process: inbound requests have credential
parameters stripped, and upstream calls are re-authenticated from the
proxy's own environment, so a container never needs (or sees) a real key.

W3 (sourceexperiments.md): upstream routing is a validated, hash-audited
registry instead of a hardcoded table; every call appends a usage-ledger row
with caller attribution bound at proxy spawn (or via worker-issued tokens —
never trusted from container-supplied identity claims); lab/fulfillment keys
are split behind ``RESEARCH_LAB_PROVIDER_KEY_SPLIT`` which also removes the
silent ``QUALIFICATION_*`` fallback.

Cost caps: paid live calls are metered per cost scope (the
``X-Research-Lab-Cost-Scope`` header, e.g. one ICP) against
``RESEARCH_LAB_PROVIDER_COST_CAP_USD_PER_ICP``; a scope over its cap gets a
typed zero-cost soft stop for private-model traffic before any upstream
contact, and every response carries the cost event headers so the
container-side trace tee can attribute spend. Non-model/debug callers can
still receive the hard 402 behavior by omitting the soft-stop header.
"""

from __future__ import annotations

import base64
from decimal import Decimal
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json
from research_lab.eval.provider_costs import (
    DEFAULT_PROVIDER_COST_CAP_USD_PER_ICP,
    DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    ProviderCostEstimate,
    ProviderCostLedger,
    _deepline_status_from_response,
    decimal_from_env,
    estimate_provider_cost,
    exa_agent_run_status,
    extract_openrouter_cost_dollars,
    redacted_endpoint,
)
from research_lab.eval.provider_evidence_cache import (
    canonical_request_fingerprint,
    load_evidence_cache,
)
from gateway.research_lab.provider_capabilities import (
    EffectiveProviderCapabilities,
    LiveTextModelCatalog,
    capability_catalog_enabled,
    capability_enforcement_mode,
    capability_refresh_seconds,
    load_effective_provider_capabilities_sync,
    normalize_candidate_route,
    provider_request_allowed,
)
from gateway.research_lab.provider_outcome_digest import (
    PROVIDER_OUTCOME_SIDECAR_ENV,
    ProviderOutcomeSidecarAccumulator,
)

PROXY_URL_ENV = "RESEARCH_LAB_EVIDENCE_PROXY_URL"
REGISTRY_PATH_ENV = "RESEARCH_LAB_PROVIDER_REGISTRY_PATH"
USAGE_LEDGER_PATH_ENV = "RESEARCH_LAB_PROVIDER_USAGE_LEDGER_PATH"
KEY_SPLIT_ENV = "RESEARCH_LAB_PROVIDER_KEY_SPLIT"
CALLER_TOKEN_HEADER = "X-Research-Lab-Caller-Token"
# W4: a request carrying this header replays from tape/day-cache only — a miss
# returns 409 instead of a live upstream call (probe live-flag off).
REPLAY_ONLY_HEADER = "X-Research-Lab-Replay-Only"
BUDGET_SOFT_STOP_HEADER = "X-Research-Lab-Budget-Soft-Stop"
BUDGET_SOFT_STOP_RESPONSE_HEADER = "X-Research-Lab-Budget-Soft-Stopped"
OPENROUTER_MANAGEMENT_CREDENTIAL_REFS = (
    "RESEARCH_LAB_OPENROUTER_MANAGEMENT_KEY",
    "OPENROUTER_MANAGEMENT_KEY",
    "OPENROUTER_API_MANAGEMENT_KEY",
    "OR_MANAGEMENT_KEY",
)

logger = logging.getLogger(__name__)

_VALID_AUTH_KINDS = ("header", "query", "bearer", "none")
_MEASURED_PROVIDER_COST_SOURCES = {
    "exa_cost_dollars",
    "openrouter_response_usage",
    "openrouter_generation_reconciliation",
    "deepline_response_cost",
}

# Legacy fallback env chains, used ONLY while the key split is off. With
# RESEARCH_LAB_PROVIDER_KEY_SPLIT on, lab traffic authenticates exclusively
# from lab-scoped keys and a missing key is a hard, attributed failure —
# never a silent borrow of fulfillment (QUALIFICATION_*) credentials.
_LEGACY_CREDENTIAL_FALLBACKS: dict[str, tuple[str, ...]] = {
    "exa": ("EXA_API_KEY",),
    "sd": ("SCRAPINGDOG_API_KEY", "QUALIFICATION_SCRAPINGDOG_API_KEY"),
    "or": ("OPENROUTER_API_KEY", "QUALIFICATION_OPENROUTER_API_KEY", "OPENROUTER_KEY"),
    "deepline": ("DEEPLINE_API_KEY",),
}


@dataclass(frozen=True)
class ProviderRegistryEntry:
    """One validated upstream in the evidence-proxy routing registry."""

    id: str
    base_url: str
    auth_kind: str  # header | query | bearer | none
    auth_name: str = ""  # header or query-parameter name carrying the credential
    credential_ref: tuple[str, ...] = ()  # ordered env-var names, first hit wins
    per_day_quota: int = 0  # live upstream calls per UTC day; 0 = unlimited
    cost_model: dict[str, Any] = field(default_factory=dict)
    active_window: str = ""
    active: bool = True
    capability_policy: dict[str, Any] = field(default_factory=dict)
    planner_summary: dict[str, Any] = field(default_factory=dict)
    probe_endpoints: tuple[dict[str, Any], ...] = ()
    origin: str = "legacy_fallback"
    reward_eligible: bool = False
    credential_ready: bool | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProviderRegistryEntry":
        credential_ref = data.get("credential_ref") or ()
        if isinstance(credential_ref, str):
            credential_ref = (credential_ref,)
        return cls(
            id=str(data.get("id") or ""),
            base_url=str(data.get("base_url") or ""),
            auth_kind=str(data.get("auth_kind") or "none"),
            auth_name=str(data.get("auth_name") or ""),
            credential_ref=tuple(str(item) for item in credential_ref),
            per_day_quota=int(data.get("per_day_quota") or 0),
            cost_model=dict(data.get("cost_model") or {}),
            active_window=str(data.get("active_window") or ""),
            active=bool(data.get("active", True)),
            capability_policy=dict(data.get("capability_policy") or {}),
            planner_summary=dict(data.get("planner_summary") or {}),
            probe_endpoints=tuple(
                dict(item)
                for item in (data.get("probe_endpoints") or [])
                if isinstance(item, Mapping)
            ),
            origin=str(data.get("origin") or "legacy_fallback"),
            reward_eligible=bool(data.get("reward_eligible", False)),
            credential_ready=(
                bool(data.get("credential_ready"))
                if data.get("credential_ready") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["credential_ref"] = list(self.credential_ref)
        data["probe_endpoints"] = [dict(item) for item in self.probe_endpoints]
        return data

    def est_cost_microusd(self) -> int:
        try:
            return max(0, int(self.cost_model.get("est_cost_microusd_per_call") or 0))
        except (TypeError, ValueError):
            return 0


def validate_provider_registry_entries(entries: Sequence[ProviderRegistryEntry]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for entry in entries:
        label = entry.id or "<missing id>"
        if not entry.id or not entry.id.replace("_", "").replace("-", "").isalnum():
            errors.append(f"{label}: id must be a non-empty slug")
        if entry.id in seen_ids:
            errors.append(f"{label}: duplicate registry id")
        seen_ids.add(entry.id)
        if not entry.base_url.startswith("https://") and not entry.base_url.startswith(
            ("http://127.0.0.1", "http://localhost")
        ):
            # Loopback http is allowed for test fakes and local trial doubles;
            # anything routable must be https.
            errors.append(f"{label}: base_url must be https")
        if entry.auth_kind not in _VALID_AUTH_KINDS:
            errors.append(f"{label}: unknown auth_kind {entry.auth_kind}")
        if entry.auth_kind in {"header", "query"} and not entry.auth_name:
            errors.append(f"{label}: auth_name required for {entry.auth_kind} auth")
        if entry.auth_kind != "none" and not entry.credential_ref:
            errors.append(f"{label}: credential_ref required for authenticated upstream")
        for env_name in entry.credential_ref:
            if not env_name or "=" in env_name or env_name != env_name.strip():
                errors.append(f"{label}: credential_ref entries must be env-var names")
        if entry.per_day_quota < 0:
            errors.append(f"{label}: per_day_quota must be >= 0")
    return errors


def provider_registry_hash(entries: Sequence[ProviderRegistryEntry]) -> str:
    return sha256_json({"registry": [entry.to_dict() for entry in sorted(entries, key=lambda e: e.id)]})


def _openrouter_chat_completion_path(upstream_url: str) -> bool:
    try:
        path = urllib.parse.urlsplit(str(upstream_url or "")).path.rstrip("/")
    except Exception:
        return False
    return path == "/api/v1/chat/completions"


def _openrouter_request_with_usage_metadata(request_body: bytes) -> bytes:
    """Ask OpenRouter to return usage/cost metadata without changing prompts.

    The replay fingerprint remains the caller's original request body; this
    helper only changes the live upstream request made by the host proxy so
    successful OpenRouter calls can be cost-accounted deterministically.
    """

    if not request_body:
        return request_body
    try:
        decoded = json.loads(request_body.decode("utf-8"))
    except Exception:
        return request_body
    if not isinstance(decoded, dict):
        return request_body
    usage = decoded.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    usage["include"] = True
    decoded["usage"] = usage
    if decoded.get("stream") is True:
        stream_options = decoded.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        stream_options["include_usage"] = True
        decoded["stream_options"] = stream_options
    return json.dumps(
        decoded,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _truthy_header(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _budget_soft_stop_body(provider: str, upstream_url: str) -> bytes:
    """Provider-shaped empty payload for private-model budget exhaustion.

    This is intentionally synthetic and is never written to the evidence day
    cache. The goal is to let the private model stop paid work and return any
    companies it already has instead of crashing on a hard HTTP 402.
    """

    try:
        path = urllib.parse.urlsplit(str(upstream_url or "")).path
    except Exception:
        path = ""
    base = {
        "research_lab_budget_exhausted": True,
        "research_lab_provider_cost_cap_blocked": True,
    }
    if provider == "or":
        doc: dict[str, Any] = {
            **base,
            "id": "research-lab-budget-soft-stop",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "[]"},
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    elif provider == "exa":
        doc = {**base, "results": [], "data": [], "costDollars": 0}
        if path.startswith("/agent/runs/"):
            doc.update({"status": "completed", "object": "agent.run"})
    elif provider == "sd":
        doc = {**base, "results": [], "data": [], "organic_results": [], "answer": ""}
    elif provider == "deepline":
        doc = {
            **base,
            "status": "completed",
            "result": None,
            "data": [],
            "billing": {"credits": 0, "cost_usd": 0},
        }
    else:
        doc = {**base, "results": [], "data": []}
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _openrouter_generation_id_from_headers(headers: Mapping[str, Any]) -> str:
    lowered = {str(k).lower(): str(v) for k, v in dict(headers or {}).items()}
    for name in (
        "X-OpenRouter-Generation-Id",
        "X-Openrouter-Generation-Id",
        "OpenRouter-Generation-Id",
        "X-OpenRouter-Response-Id",
        "OpenRouter-Response-Id",
        "X-Generation-Id",
    ):
        value = str(headers.get(name) or lowered.get(name.lower()) or "").strip()
        if value:
            return value[:200]
    return ""


def _normalize_openrouter_key(value: str) -> str:
    raw = str(value or "").strip()
    prefix = "sk-or-v1-"
    if raw.lower().startswith(prefix):
        return prefix + raw[len(prefix):]
    return raw


def resolve_openrouter_management_credential() -> tuple[str, str]:
    for env_name in OPENROUTER_MANAGEMENT_CREDENTIAL_REFS:
        value = _normalize_openrouter_key(os.getenv(env_name) or "")
        if value:
            return value, env_name
    return "", ""


def _reconcile_openrouter_generation_cost(
    *,
    entry: ProviderRegistryEntry,
    credential: str,
    management_credential: str = "",
    estimate: ProviderCostEstimate,
) -> ProviderCostEstimate:
    generation_id = str(estimate.generation_id or "").strip()
    if not generation_id:
        return estimate
    gen_url = entry.base_url + "/api/v1/generation?id=" + urllib.parse.quote(generation_id, safe="")
    credentials: list[str] = []
    for candidate in (management_credential, credential):
        normalized = _normalize_openrouter_key(candidate)
        if normalized and normalized not in credentials:
            credentials.append(normalized)
    if not credentials:
        return estimate
    last_error: Exception | None = None
    for attempt, delay in enumerate((0.0, 2.0, 5.0, 10.0, 20.0)):
        if delay:
            time.sleep(delay)
        for token in credentials:
            gen_headers = {"Authorization": "Bearer " + token}
            gen_req = urllib.request.Request(gen_url, headers=gen_headers, method="GET")
            try:
                with urllib.request.urlopen(gen_req, timeout=30) as gen_response:
                    if int(getattr(gen_response, "status", None) or getattr(gen_response, "code", 0) or 0) >= 400:
                        continue
                    generation_body = gen_response.read()
                    reconciled_cost, reconciled_metadata = extract_openrouter_cost_dollars(generation_body)
                    if reconciled_cost is not None:
                        return ProviderCostEstimate(
                            provider="or",
                            endpoint=estimate.endpoint,
                            model=estimate.model or str(reconciled_metadata.get("model") or "")[:160],
                            billable=True,
                            cost_usd=reconciled_cost,
                            cost_source="openrouter_generation_reconciliation",
                            prompt_tokens=int(reconciled_metadata.get("prompt_tokens") or 0),
                            completion_tokens=int(reconciled_metadata.get("completion_tokens") or 0),
                            generation_id=generation_id,
                        )
                    return estimate
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code in {401, 403} and token == credentials[-1]:
                    return estimate
                continue
            except Exception as exc:
                last_error = exc
                if attempt == 4 and token == credentials[-1]:
                    break
    if last_error is not None:
        logger.warning(
            "research_lab_openrouter_generation_cost_reconcile_failed generation_id_prefix=%s error=%s",
            generation_id[:16],
            last_error,
        )
    return estimate


def _headers_to_dict(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    try:
        items = headers.items()
    except Exception:
        return {}
    try:
        return {str(k): str(v) for k, v in items}
    except Exception:
        return {}


def seed_provider_registry() -> list[ProviderRegistryEntry]:
    """The three pre-registry upstreams, expressed as registry seed entries."""

    return [
        ProviderRegistryEntry(
            id="exa",
            base_url="https://api.exa.ai",
            auth_kind="header",
            auth_name="x-api-key",
            credential_ref=("RESEARCH_LAB_EXA_API_KEY", "EXA_API_KEY"),
            cost_model={"currency": "usd", "est_cost_microusd_per_call": 5_000},
        ),
        ProviderRegistryEntry(
            id="sd",
            base_url="https://api.scrapingdog.com",
            auth_kind="query",
            auth_name="api_key",
            credential_ref=("RESEARCH_LAB_SCRAPINGDOG_API_KEY", "SCRAPINGDOG_API_KEY"),
            cost_model={"currency": "usd", "est_cost_microusd_per_call": 1_000},
        ),
        ProviderRegistryEntry(
            id="or",
            base_url="https://openrouter.ai",
            auth_kind="bearer",
            auth_name="Authorization",
            credential_ref=("RESEARCH_LAB_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
            cost_model={"currency": "usd", "est_cost_microusd_per_call": 2_000},
        ),
        ProviderRegistryEntry(
            id="deepline",
            base_url="https://code.deepline.com",
            auth_kind="bearer",
            auth_name="Authorization",
            credential_ref=("RESEARCH_LAB_DEEPLINE_API_KEY", "DEEPLINE_API_KEY"),
            cost_model={"currency": "usd", "est_cost_microusd_per_call": 100_000},
        ),
    ]


def _load_static_provider_registry(path: str = "") -> list[ProviderRegistryEntry]:
    """Load the continuity registry without consulting remote capability state."""

    resolved = str(path or os.getenv(REGISTRY_PATH_ENV) or "").strip()
    if not resolved:
        return seed_provider_registry()
    with open(resolved, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    raw_entries = doc.get("providers") if isinstance(doc, Mapping) else doc
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("provider registry file must contain a non-empty providers list")
    entries = [ProviderRegistryEntry.from_mapping(item) for item in raw_entries]
    errors = validate_provider_registry_entries(entries)
    if errors:
        raise ValueError("invalid provider registry: " + "; ".join(errors))
    return entries


def _entries_from_capabilities(
    capabilities: EffectiveProviderCapabilities,
) -> list[ProviderRegistryEntry]:
    entries = [ProviderRegistryEntry.from_mapping(item) for item in capabilities.providers]
    errors = validate_provider_registry_entries(entries)
    if errors:
        raise ValueError("invalid effective provider registry: " + "; ".join(errors))
    return entries


def load_provider_registry_with_capabilities(
    path: str = "",
    *,
    strict_remote: bool = False,
) -> tuple[list[ProviderRegistryEntry], EffectiveProviderCapabilities]:
    """Load private capabilities + ready SOURCE_ADD rows over continuity routes."""

    static_entries = _load_static_provider_registry(path)
    if capability_catalog_enabled():
        capabilities = load_effective_provider_capabilities_sync(
            [entry.to_dict() for entry in static_entries],
            strict_remote=strict_remote,
        )
    else:
        capabilities = load_effective_provider_capabilities_sync(
            [entry.to_dict() for entry in static_entries],
            strict_remote=False,
            private_row_loader=lambda: None,
            source_row_loader=lambda: (),
        )
    return _entries_from_capabilities(capabilities), capabilities


def load_provider_registry(path: str = "") -> list[ProviderRegistryEntry]:
    """Compatibility wrapper returning the current effective routing entries."""

    entries, _capabilities = load_provider_registry_with_capabilities(path)
    return entries


def reserved_builtin_provider_ids_sync(path: str = "") -> set[str]:
    """Provider IDs that SOURCE_ADD may never replace."""

    static_ids = {entry.id for entry in _load_static_provider_registry(path)}
    try:
        _entries, capabilities = load_provider_registry_with_capabilities(path)
    except Exception:
        return static_ids
    return static_ids | {
        str(item.get("id") or "")
        for item in capabilities.providers
        if str(item.get("origin") or "") == "builtin"
    }


def _key_split_enabled(override: bool | None = None) -> bool:
    if override is not None:
        return bool(override)
    return str(os.getenv(KEY_SPLIT_ENV, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_provider_credential(
    entry: ProviderRegistryEntry,
    *,
    key_split: bool | None = None,
) -> tuple[str, str]:
    """Return (credential_value, source_env_name).

    Key split ON: only lab-scoped (``RESEARCH_LAB_*``) refs are honored — no
    fallback to shared or ``QUALIFICATION_*`` keys.
    Key split OFF: the entry's refs are tried in order, then the legacy
    fallback chain for the pre-registry providers (pre-W3 behavior).
    """

    split = _key_split_enabled(key_split)
    refs = list(entry.credential_ref)
    if split:
        refs = [name for name in refs if name.startswith("RESEARCH_LAB_")]
    else:
        for legacy in _LEGACY_CREDENTIAL_FALLBACKS.get(entry.id, ()):
            if legacy not in refs:
                refs.append(legacy)
    for env_name in refs:
        value = (os.getenv(env_name) or "").strip()
        if value:
            return value, env_name
    try:
        from gateway.research_lab.source_add_catalog import decrypt_source_add_registry_credential

        credential, source_ref = decrypt_source_add_registry_credential(entry)
        if credential:
            return credential, source_ref
    except Exception as exc:  # noqa: BLE001 - caller handles missing credential
        logger.warning("source_add_registry_credential_decrypt_failed provider=%s error=%s", entry.id, str(exc)[:200])
    return "", ""


class ProviderUsageLedger:
    """Per-call usage ledger + per-day live-call counters.

    Rows append to a JSONL file when a path is configured; counters always
    accumulate in memory so per-day quotas are enforceable either way.
    """

    def __init__(self, path: str = "", *, outcome_sidecar_path: str = "") -> None:
        self._path = str(path or "")
        self._lock = threading.Lock()
        self._live_day = _utc_day()
        self._live_calls: dict[str, int] = {}
        self._outcomes = ProviderOutcomeSidecarAccumulator(
            outcome_sidecar_path or os.getenv(PROVIDER_OUTCOME_SIDECAR_ENV) or ""
        )

    def _roll_day_locked(self) -> None:
        today = _utc_day()
        if self._live_day != today:
            self._live_day = today
            self._live_calls = {}

    def live_calls_today(self, provider_id: str) -> int:
        with self._lock:
            self._roll_day_locked()
            return self._live_calls.get(provider_id, 0)

    def record(
        self,
        *,
        provider_id: str,
        endpoint_class: str,
        fingerprint: str,
        evidence: str,
        status: int,
        est_cost_microusd: int,
        caller: Mapping[str, Any] | None,
        outcome_evidence: str = "",
        live_call: bool | None = None,
        sidecar_spend_microusd: int | None = None,
        sidecar_spend_kind: str = "estimated",
    ) -> None:
        with self._lock:
            self._roll_day_locked()
            if evidence == "recorded":
                self._live_calls[provider_id] = self._live_calls.get(provider_id, 0) + 1
            self._outcomes.record(
                provider_id=provider_id,
                endpoint_class=endpoint_class,
                evidence=outcome_evidence or evidence,
                status=int(status),
                live_call=(evidence == "recorded" if live_call is None else bool(live_call)),
                spend_microusd=(
                    int(est_cost_microusd)
                    if sidecar_spend_microusd is None
                    else int(sidecar_spend_microusd)
                ),
                spend_kind=sidecar_spend_kind,
            )
            if self._path:
                row = {
                    "schema_version": "1.0",
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "utc_day": self._live_day,
                    "provider_id": provider_id,
                    "endpoint_class": endpoint_class,
                    "request_fingerprint": fingerprint,
                    "evidence": evidence,
                    "status": int(status),
                    "est_cost_microusd": int(est_cost_microusd),
                    "caller": dict(caller or {}),
                }
                try:
                    with open(self._path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                except Exception as exc:
                    logger.warning(
                        "research_lab_provider_usage_ledger_append_failed path_hash=%s error=%s",
                        sha256_json({"path": self._path}),
                        type(exc).__name__,
                    )

    def close(self) -> None:
        self._outcomes.close()


def _endpoint_class(rest: str) -> str:
    """Path-only endpoint class: no query strings, bounded length."""

    path = urllib.parse.urlsplit(rest).path or "/"
    return path[:200]


class CallerTokenMap:
    """Worker-issued caller tokens → caller context.

    The worker (host side) writes ``{token: {caller fields...}}`` to a file it
    owns and hands tokens to runs; a container presents the opaque token and
    the proxy resolves identity from the host-side map. Container-supplied
    identity FIELDS are never trusted — an unknown token attributes as
    ``{"caller_kind": "unknown_token"}``.
    """

    def __init__(self, path: str = "") -> None:
        self._path = str(path or "")
        self._lock = threading.Lock()
        self._mtime = 0.0
        self._tokens: dict[str, dict[str, Any]] = {}

    def resolve(self, token: str) -> dict[str, Any] | None:
        if not token or not self._path:
            return None
        with self._lock:
            try:
                mtime = os.stat(self._path).st_mtime
            except OSError:
                return None
            if mtime != self._mtime:
                try:
                    with open(self._path, "r", encoding="utf-8") as handle:
                        doc = json.load(handle)
                    self._tokens = {
                        str(key): dict(value)
                        for key, value in (doc or {}).items()
                        if isinstance(value, Mapping)
                    }
                    self._mtime = mtime
                except Exception:
                    return None
            context = self._tokens.get(token)
            return dict(context) if context else None


def _utc_day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


_EXA_AGENT_NONTERMINAL_STATUSES = {"queued", "running", "in_progress", "pending"}
_EXA_TRANSIENT_ERROR_TAGS = {"NO_MORE_CREDITS"}

# Deepline play runs are polled at GET /api/v2/runs/<id> until terminal. A
# nonterminal snapshot recorded into the day cache freezes every later poll of
# the same URL at that snapshot, so the poller never observes "completed" and
# burns its whole budget (same failure mode the Exa /agent/runs/ exemption
# fixes). Only terminal run states may be recorded.
_DEEPLINE_RUN_NONTERMINAL_STATUSES = {"queued", "running", "in_progress", "pending", "started"}


def _response_is_recordable(provider: str, upstream_url: str, status: int, body: bytes) -> bool:
    # Never record error responses into the day cache.  The cache replays a
    # recorded response to every identical request for the rest of the UTC
    # day, so recording a transient provider failure (a 5xx burst, an
    # unstable endpoint's 410/404 window) poisons that query for the whole
    # day: retry attempts replay the cached failure instantly instead of
    # going live, and the day ends with no benchmark even after the provider
    # recovers.  Failures always retry live; the per-scope cost ledger
    # bounds what those retries can spend.  (Baseline-tape replay of
    # recorded errors is unaffected — this gates only NEW day-cache writes.)
    if status >= 400:
        return False
    try:
        path = urllib.parse.urlsplit(upstream_url).path
    except Exception:
        path = ""
    if provider == "exa":
        if _exa_response_is_transient_error(body):
            return False
        if not path.startswith("/agent/runs/"):
            return True
        agent_status = exa_agent_run_status(body)
        return agent_status not in _EXA_AGENT_NONTERMINAL_STATUSES
    if provider == "deepline":
        # A play-start response mints a fresh one-time run id. Recording it
        # replays that same id to every later identical start request for
        # the rest of the day, and polling the expired id then 404s until
        # the poller's whole budget burns. Start requests always go live.
        if path.startswith("/api/v2/plays"):
            return False
        if not path.startswith("/api/v2/runs/"):
            return True
        run_status = _deepline_status_from_response(body)
        return run_status not in _DEEPLINE_RUN_NONTERMINAL_STATUSES
    return True


def _exa_response_is_transient_error(body: bytes) -> bool:
    try:
        doc = json.loads(body.decode("utf-8"))
    except Exception:
        return False
    if not isinstance(doc, Mapping):
        return False
    tag = str(doc.get("tag") or "").strip().upper()
    if tag in _EXA_TRANSIENT_ERROR_TAGS:
        return True
    error = doc.get("error")
    if isinstance(error, str) and error.strip():
        return True
    return False


def _record_is_replayable(record: Mapping[str, Any]) -> bool:
    try:
        status = int(record.get("status") or 0)
    except Exception:
        status = 0
    if status >= 400:
        return True
    try:
        body = base64.b64decode(record.get("body_b64") or "")
    except Exception:
        body = b""
    agent_status = exa_agent_run_status(body)
    return agent_status not in _EXA_AGENT_NONTERMINAL_STATUSES


class EvidenceStore:
    """Baseline tape + shared day cache with single-flight live calls.

    A Condition guards all state. Single-flight guarantees that for any one
    request fingerprint, exactly one caller makes the live provider call while
    every concurrent identical caller waits and then replays the recorded
    result — so parallel same-request work (e.g. two candidates on the same
    ICP) shares one live call instead of each calling the provider.
    """

    def __init__(self, baseline_dir: str = "", day_cache_path: str = "") -> None:
        # Reentrant so a lookup that triggers a midnight rollover can reload
        # under the same held lock without deadlocking.
        self._cond = threading.Condition(threading.RLock())
        self._baseline: dict[str, dict[str, Any]] = {}
        self._day: dict[str, dict[str, Any]] = {}
        self._inflight: set[str] = set()
        self._cost_ledgers: dict[str, ProviderCostLedger] = {}
        self._day_path = day_cache_path
        self._loaded_day = ""
        self._baseline_dir = baseline_dir
        with self._cond:
            self._reload_locked()

    def _reload_locked(self) -> None:
        self._loaded_day = _utc_day()
        self._baseline = {}
        self._inflight.clear()
        self._cost_ledgers.clear()
        if self._baseline_dir and os.path.isdir(self._baseline_dir):
            for name in sorted(os.listdir(self._baseline_dir)):
                if not name.endswith(".json"):
                    continue
                try:
                    loaded = load_evidence_cache(os.path.join(self._baseline_dir, name))
                except Exception:
                    continue
                for key, record in loaded.items():
                    if _record_is_replayable(record):
                        self._baseline.setdefault(key, record)
        self._day = {}
        if self._day_path and os.path.isfile(self._day_path):
            try:
                with open(self._day_path, "r", encoding="utf-8") as handle:
                    doc = json.load(handle)
                if str(doc.get("utc_day") or "") == self._loaded_day:
                    entries = doc.get("entries")
                    if isinstance(entries, Mapping):
                        self._day = {
                            str(k): dict(v)
                            for k, v in entries.items()
                            if (
                                isinstance(v, Mapping)
                                and isinstance(v.get("status"), int)
                                and _record_is_replayable(v)
                            )
                        }
            except Exception:
                self._day = {}

    def reload(self) -> None:
        with self._cond:
            self._reload_locked()

    def _rollover_if_needed_locked(self) -> None:
        if self._loaded_day != _utc_day():
            # Midnight UTC: all recorded evidence expires; start the day empty
            # and wake any waiters so they re-lead against the fresh day.
            self._reload_locked()
            self._cond.notify_all()

    def _cached_locked(self, fingerprint: str) -> dict[str, Any] | None:
        record = self._baseline.get(fingerprint)
        if record is not None and not _record_is_replayable(record):
            record = None
        if record is None:
            record = self._day.get(fingerprint)
            if record is not None and not _record_is_replayable(record):
                self._day.pop(fingerprint, None)
                record = None
        return dict(record) if record else None

    def lookup(self, fingerprint: str) -> dict[str, Any] | None:
        with self._cond:
            self._rollover_if_needed_locked()
            return self._cached_locked(fingerprint)

    def acquire_or_wait(self, fingerprint: str, timeout: float = 175.0) -> tuple[dict[str, Any] | None, bool]:
        """Single-flight gate.

        Returns (record, is_leader):
        - (record, False): already recorded — replay it, do not call live.
        - (None, True): you are the leader — make the live call, then call
          record() (or release_lead() on failure).
        - (None, False): timed out waiting for the leader — fall back to a
          live call without leadership (rare; keeps a stuck leader from
          blocking forever).
        """
        with self._cond:
            deadline = None
            while True:
                self._rollover_if_needed_locked()
                cached = self._cached_locked(fingerprint)
                if cached is not None:
                    return cached, False
                if fingerprint not in self._inflight:
                    self._inflight.add(fingerprint)
                    return None, True
                # Another caller is leading this fingerprint; wait for it.
                if deadline is None:
                    deadline = timeout
                if not self._cond.wait(timeout=deadline):
                    return None, False

    def release_lead(self, fingerprint: str) -> None:
        """Leader's live call failed with no recordable result: let a waiter
        take over rather than block."""
        with self._cond:
            self._inflight.discard(fingerprint)
            self._cond.notify_all()

    def record(self, fingerprint: str, status: int, body: bytes) -> None:
        with self._cond:
            self._rollover_if_needed_locked()
            self._inflight.discard(fingerprint)
            self._cond.notify_all()
            if fingerprint in self._baseline or fingerprint in self._day:
                return
            self._day[fingerprint] = {
                "status": int(status),
                "body_b64": base64.b64encode(body).decode("ascii"),
                "outcome": "error" if status >= 400 else "success",
            }
            if self._day_path:
                doc = {
                    "schema_version": "1.1",
                    "utc_day": self._loaded_day,
                    "entries": self._day,
                }
                tmp = f"{self._day_path}.tmp.{os.getpid()}"
                try:
                    with open(tmp, "w", encoding="utf-8") as handle:
                        json.dump(doc, handle, sort_keys=True, separators=(",", ":"))
                    os.replace(tmp, self._day_path)
                except Exception:
                    pass

    def cost_ledger(self, scope: str, cap_usd: Decimal) -> ProviderCostLedger:
        with self._cond:
            self._rollover_if_needed_locked()
            key = str(scope or "unscoped")
            ledger = self._cost_ledgers.get(key)
            if ledger is None:
                ledger = ProviderCostLedger(scope=key, cap_usd=cap_usd)
                self._cost_ledgers[key] = ledger
            return ledger


class ProviderRegistryState:
    """Atomic effective-registry holder with last-known-good refresh."""

    def __init__(
        self,
        *,
        entries: Sequence[ProviderRegistryEntry],
        capabilities: EffectiveProviderCapabilities,
        registry_path: str = "",
        refresh_enabled: bool = False,
        refresh_seconds: int | None = None,
        loader: Any | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._entries = {entry.id: entry for entry in entries}
        self._capabilities = capabilities
        self._registry_path = str(registry_path or "")
        self._refresh_enabled = bool(refresh_enabled)
        self._refresh_seconds = max(10, int(refresh_seconds or capability_refresh_seconds()))
        self._loader = loader or (
            lambda: load_provider_registry_with_capabilities(
                self._registry_path,
                strict_remote=True,
            )
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def resolve(self, provider_id: str) -> ProviderRegistryEntry | None:
        with self._lock:
            return self._entries.get(str(provider_id or ""))

    def capabilities(self) -> EffectiveProviderCapabilities:
        with self._lock:
            return self._capabilities

    def diagnostic(self) -> dict[str, Any]:
        return self.capabilities().diagnostic()

    def refresh_once(self) -> bool:
        try:
            entries, capabilities = self._loader()
            errors = validate_provider_registry_entries(entries)
            if errors:
                raise ValueError("invalid refreshed provider registry: " + "; ".join(errors))
        except Exception as exc:
            logger.warning(
                "research_lab_provider_capability_refresh_failed retained_last_known_good=true error=%s",
                str(exc)[:200],
            )
            return False
        with self._lock:
            self._entries = {entry.id: entry for entry in entries}
            self._capabilities = capabilities
        diagnostic = capabilities.diagnostic()
        logger.info(
            "research_lab_provider_capability_refresh_succeeded capability_hash=%s provider_count=%d credential_ready_count=%d private_snapshot_loaded=%s",
            diagnostic["capability_hash"],
            diagnostic["provider_count"],
            diagnostic["credential_ready_count"],
            diagnostic["private_snapshot_loaded"],
        )
        return True

    def start(self) -> None:
        if not self._refresh_enabled or self._thread is not None:
            return

        def _run() -> None:
            while not self._stop.wait(self._refresh_seconds):
                self.refresh_once()

        self._thread = threading.Thread(
            target=_run,
            name="provider-capability-refresh",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


class _CapabilityAwareHTTPServer(ThreadingHTTPServer):
    registry_state: ProviderRegistryState
    usage_ledger: ProviderUsageLedger

    def handle_error(self, request, client_address) -> None:
        # Recycled/timing-out worker clients drop their sockets mid-request;
        # that is routine churn, not a proxy fault, and the default handler
        # prints a full traceback for each one.
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            return
        super().handle_error(request, client_address)

    def server_close(self) -> None:
        self.registry_state.stop()
        try:
            super().server_close()
        finally:
            # ThreadingHTTPServer waits for active handlers in server_close;
            # flush only after no request can append another outcome.
            self.usage_ledger.close()


_HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "host", "content-length", "authorization", "x-api-key"}


class _ProxyHandler(BaseHTTPRequestHandler):
    store: EvidenceStore
    registry: dict[str, ProviderRegistryEntry]
    registry_state: ProviderRegistryState
    ledger: ProviderUsageLedger
    caller_context: dict[str, Any]
    caller_tokens: CallerTokenMap
    key_split: bool | None
    # W5 trials: in-memory credential overrides by provider id (e.g. a
    # KMS-decrypted miner key for a per-trial proxy instance). Never read from
    # env, never exposed to the container.
    credential_overrides: dict[str, str]
    model_catalog: LiveTextModelCatalog
    enforcement_mode: str
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # quiet; the gateway logs enough
        pass

    def _provider(self) -> tuple[ProviderRegistryEntry, str] | None:
        parts = self.path.lstrip("/").split("/", 1)
        name = parts[0] if parts else ""
        entry = self.registry_state.resolve(name)
        if not entry or not entry.active:
            return None
        rest = "/" + (parts[1] if len(parts) > 1 else "")
        return entry, rest

    def _caller(self) -> dict[str, Any]:
        token = str(self.headers.get(CALLER_TOKEN_HEADER) or "").strip()
        if token:
            resolved = self.caller_tokens.resolve(token)
            if resolved is not None:
                return resolved
            return {"caller_kind": "unknown_token"}
        return dict(self.caller_context)

    def _respond(
        self,
        status: int,
        body: bytes,
        evidence: str = "",
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            if evidence:
                self.send_header("X-Research-Lab-Evidence", evidence)
            for key, value in dict(headers or {}).items():
                if key and value is not None:
                    self.send_header(str(key), str(value))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            pass

    def _ledger_row(
        self,
        entry: ProviderRegistryEntry,
        rest: str,
        fingerprint: str,
        *,
        evidence: str,
        status: int,
        live_cost: bool,
        est_cost_microusd: int | None = None,
        outcome_evidence: str = "",
        live_call: bool | None = None,
        sidecar_spend_microusd: int | None = None,
        sidecar_spend_kind: str = "estimated",
    ) -> None:
        if est_cost_microusd is None:
            est_cost_microusd = entry.est_cost_microusd() if live_cost else 0
        self.ledger.record(
            provider_id=entry.id,
            endpoint_class=_endpoint_class(rest),
            fingerprint=fingerprint,
            evidence=evidence,
            status=status,
            est_cost_microusd=est_cost_microusd,
            caller=self._caller(),
            outcome_evidence=outcome_evidence,
            live_call=live_call,
            sidecar_spend_microusd=sidecar_spend_microusd,
            sidecar_spend_kind=sidecar_spend_kind,
        )

    def _handle(self) -> None:
        routed = self._provider()
        if routed is None:
            self._respond(404, b'{"error":"unknown provider route"}')
            return
        entry, rest = routed
        length = int(self.headers.get("Content-Length") or 0)
        request_body = self.rfile.read(length) if length else b""
        normalized_route = normalize_candidate_route(rest)
        if normalized_route is None:
            self._respond(400, b'{"error":"unsafe provider route"}')
            return
        normalized_path, normalized_query = normalized_route
        rest = normalized_path + (("?" + normalized_query) if normalized_query else "")
        route_allowed, route_reason, _normalized_path = provider_request_allowed(
            entry.to_dict(),
            self.command,
            rest,
        )
        # Fingerprint on the UPSTREAM shape of the request so it matches
        # evidence recorded from direct provider calls (baseline tapes).
        upstream_url = entry.base_url + rest
        fingerprint = canonical_request_fingerprint(self.command, upstream_url, request_body or None)
        enforce_route = (
            self.enforcement_mode == "enforce"
            or entry.origin == "source_add"
            or route_reason in {"unsafe_route", "blocked_route"}
        )
        if not route_allowed and enforce_route:
            self._ledger_row(entry, rest, fingerprint, evidence="blocked", status=403, live_cost=False)
            self._respond(403, b'{"error":"provider route not allowed"}', evidence="blocked")
            return
        if not route_allowed:
            logger.warning(
                "research_lab_provider_capability_route_observed provider_hash=%s route_hash=%s reason=%s",
                sha256_json({"provider": entry.id}),
                sha256_json({"method": self.command, "path": normalized_path}),
                route_reason,
            )
        model_policy = (
            entry.capability_policy.get("model_policy")
            if isinstance(entry.capability_policy, Mapping)
            and isinstance(entry.capability_policy.get("model_policy"), Mapping)
            else {}
        )
        if str(model_policy.get("kind") or "none") == "live_text_catalog":
            try:
                request_doc = json.loads(request_body.decode("utf-8")) if request_body else {}
            except Exception:
                request_doc = {}
            model_id = str(request_doc.get("model") or "") if isinstance(request_doc, Mapping) else ""
            model_allowed, model_status = self.model_catalog.validate_model(
                entry.to_dict(),
                model_id,
            )
            if not model_allowed and self.enforcement_mode == "enforce":
                self._ledger_row(entry, rest, fingerprint, evidence="blocked", status=403, live_cost=False)
                self._respond(403, b'{"error":"text model not allowed"}', evidence="blocked")
                return
            if not model_allowed:
                logger.warning(
                    "research_lab_provider_model_observed provider_hash=%s model_hash=%s status=%s",
                    sha256_json({"provider": entry.id}),
                    sha256_json({"model": model_id}),
                    model_status,
                )
        # Per-scope cost accounting: the container names its scope (one ICP)
        # and paid live calls in that scope are capped.
        scope = str(self.headers.get("X-Research-Lab-Cost-Scope") or "unscoped").strip() or "unscoped"
        cap_usd = decimal_from_env(
            "RESEARCH_LAB_PROVIDER_COST_CAP_USD_PER_ICP",
            DEFAULT_PROVIDER_COST_CAP_USD_PER_ICP,
        )
        header_cap = str(self.headers.get("X-Research-Lab-Cost-Cap-Usd") or "").strip()
        if header_cap:
            try:
                parsed_cap = Decimal(header_cap)
                if parsed_cap >= 0:
                    cap_usd = parsed_cap
            except Exception:
                pass
        credit_price = decimal_from_env(
            "RESEARCH_LAB_SCRAPINGDOG_COST_PER_CREDIT_USD",
            DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
        )
        cost_ledger = self.store.cost_ledger(scope, cap_usd)
        # Global read-through with single-flight: the first run of the day to
        # make a request calls the provider live while every concurrent
        # identical request (e.g. another candidate on the same ICP) waits and
        # replays the recorded result, so one live call is shared by all.
        cached, is_leader = self.store.acquire_or_wait(fingerprint)
        if cached is not None:
            try:
                body = base64.b64decode(cached.get("body_b64") or "")
            except Exception:
                body = b""
            status = int(cached.get("status") or 502)
            event = cost_ledger.cache_hit_event(
                provider=entry.id,
                endpoint=redacted_endpoint(entry.id, upstream_url),
                request_fingerprint=fingerprint,
                status_code=status,
            )
            self._ledger_row(entry, rest, fingerprint, evidence="hit", status=status, live_cost=False)
            self._respond(status, body, evidence="hit", headers=event.to_headers())
            return
        if str(self.headers.get(REPLAY_ONLY_HEADER) or "").strip().lower() in {"1", "true", "yes", "on"}:
            if is_leader:
                self.store.release_lead(fingerprint)
            self._ledger_row(entry, rest, fingerprint, evidence="replay_miss", status=409, live_cost=False)
            self._respond(409, b'{"error":"replay_miss"}')
            return
        endpoint = redacted_endpoint(entry.id, upstream_url)
        if cost_ledger.should_block_paid_call():
            block_reason = cost_ledger.block_reason()
            error_code = (
                "research_lab_provider_cost_cap_exceeded"
                if block_reason == "cost_cap_reached"
                else "research_lab_provider_cost_tracking_failed"
            )
            soft_stop = block_reason == "cost_cap_reached" and _truthy_header(
                self.headers.get(BUDGET_SOFT_STOP_HEADER)
            )
            status_code = 200 if soft_stop else 402
            evidence_label = "budget_soft_stop" if soft_stop else "blocked"
            event = cost_ledger.block_event(
                provider=entry.id,
                endpoint=endpoint,
                request_fingerprint=fingerprint,
                reason=block_reason,
                status_code=status_code,
                evidence=evidence_label,
            )
            body = json.dumps(
                {
                    "error": error_code,
                    "provider": entry.id,
                    "endpoint": endpoint,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if is_leader:
                self.store.release_lead(fingerprint)
            self._ledger_row(entry, rest, fingerprint, evidence=evidence_label, status=status_code, live_cost=False)
            headers = event.to_headers()
            if soft_stop:
                body = _budget_soft_stop_body(entry.id, upstream_url)
                headers[BUDGET_SOFT_STOP_RESPONSE_HEADER] = "1"
            self._respond(status_code, body, evidence=evidence_label, headers=headers)
            return
        # Live call: enforce the per-day quota before touching the upstream.
        if entry.per_day_quota > 0 and self.ledger.live_calls_today(entry.id) >= entry.per_day_quota:
            if is_leader:
                self.store.release_lead(fingerprint)
            self._ledger_row(entry, rest, fingerprint, evidence="quota_exhausted", status=429, live_cost=False)
            self._respond(429, b'{"error":"provider day quota exhausted"}')
            return
        credential = self.credential_overrides.get(entry.id) or ""
        if not credential:
            credential, _source = resolve_provider_credential(entry, key_split=self.key_split)
        if entry.auth_kind != "none" and not credential:
            # No silent fallback: a missing lab key is an attributed failure.
            if is_leader:
                self.store.release_lead(fingerprint)
            self._ledger_row(entry, rest, fingerprint, evidence="credential_missing", status=502, live_cost=False)
            self._respond(502, b'{"error":"lab provider credential not configured"}')
            return
        # Re-authenticate from the proxy's own credentials.
        target = upstream_url
        if entry.auth_kind == "query":
            split = urllib.parse.urlsplit(target)
            pairs = [
                (k, v)
                for k, v in urllib.parse.parse_qsl(split.query, keep_blank_values=True)
                if k.lower() != entry.auth_name.lower()
            ]
            pairs.append((entry.auth_name, credential))
            target = urllib.parse.urlunsplit(split._replace(query=urllib.parse.urlencode(pairs)))
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in _HOP_HEADERS and not k.lower().startswith("x-research-lab")
        }
        # The day cache records upstream bodies as raw bytes with no header
        # metadata, so a compressed body replays as undecodable garbage to any
        # client (the replay carries no Content-Encoding). Force identity on
        # the upstream request so recorded bodies are always plain: a client's
        # forwarded Accept-Encoding: gzip must never decide what gets cached.
        headers = {k: v for k, v in headers.items() if k.lower() != "accept-encoding"}
        headers["Accept-Encoding"] = "identity"
        if entry.auth_kind == "header" and credential:
            headers[entry.auth_name] = credential
        elif entry.auth_kind == "bearer" and credential:
            headers[entry.auth_name or "Authorization"] = "Bearer " + credential
        upstream_request_body = request_body
        if entry.id == "or" and _openrouter_chat_completion_path(upstream_url):
            upstream_request_body = _openrouter_request_with_usage_metadata(request_body)
        request = urllib.request.Request(
            target,
            data=upstream_request_body or None,
            headers=headers,
            method=self.command,
        )
        response_headers: dict[str, str] = {}
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = int(getattr(response, "status", None) or getattr(response, "code", 0) or 0)
                response_headers = _headers_to_dict(getattr(response, "headers", None))
                body = response.read()
        except urllib.error.HTTPError as exc:
            # An HTTP error status is a real, recordable provider outcome.
            status = int(exc.code)
            response_headers = _headers_to_dict(getattr(exc, "headers", None))
            try:
                body = exc.read()
            except Exception:
                body = b""
        except Exception:
            # No recordable result (transport failure): release leadership so a
            # waiting caller can retry rather than block on us.
            if is_leader:
                self.store.release_lead(fingerprint)
            event = cost_ledger.record_live_event(
                provider=entry.id,
                request_fingerprint=fingerprint,
                status_code=502,
                estimate=ProviderCostEstimate(
                    provider=entry.id,
                    endpoint=endpoint,
                    billable=False,
                    cost_source="transport_failure_zero_cost",
                ),
            )
            self._ledger_row(
                entry,
                rest,
                fingerprint,
                evidence="error",
                status=502,
                live_cost=False,
                live_call=True,
                sidecar_spend_microusd=0,
            )
            self._respond(502, b'{"error":"upstream unreachable"}', evidence="error", headers=event.to_headers())
            return
        # Nonterminal Exa agent polls are real spend but not replayable
        # evidence: never cache them (a later identical poll must go live).
        recordable = _response_is_recordable(entry.id, upstream_url, status, body)
        evidence_label = "recorded" if recordable else "live_unrecorded"
        if recordable:
            self.store.record(fingerprint, status, body)
        elif is_leader:
            self.store.release_lead(fingerprint)
        estimate = estimate_provider_cost(
            provider=entry.id,
            upstream_url=upstream_url,
            status=status,
            response_body=body,
            request_body=upstream_request_body or request_body or None,
            scrapingdog_credit_price_usd=credit_price,
        )
        if entry.id == "or" and not estimate.generation_id:
            generation_id = _openrouter_generation_id_from_headers(response_headers)
            if generation_id:
                estimate = replace(estimate, generation_id=generation_id)
        if (
            entry.id == "or"
            and estimate.generation_id
            and (
                estimate.tracking_reason == "missing_openrouter_cost"
                or estimate.cost_source == "openrouter_missing_cost_zero_cost"
            )
        ):
            management_credential, _management_source = resolve_openrouter_management_credential()
            estimate = _reconcile_openrouter_generation_cost(
                entry=entry,
                credential=credential,
                management_credential=management_credential,
                estimate=estimate,
            )
        event = cost_ledger.record_live_event(
            provider=entry.id,
            request_fingerprint=fingerprint,
            status_code=status,
            estimate=estimate,
            evidence=evidence_label,
        )
        # The usage row carries the measured cost when the estimator priced the
        # call; the registry's static per-call estimate is the fallback.
        measured_microusd: int | None = None
        if estimate.billable and estimate.cost_usd is not None:
            try:
                measured_microusd = max(0, int(Decimal(estimate.cost_usd) * 1_000_000))
            except Exception:
                measured_microusd = None
        sidecar_spend_microusd = 0
        if estimate.billable and 200 <= status < 300:
            sidecar_spend_microusd = (
                measured_microusd
                if measured_microusd is not None
                else max(0, entry.est_cost_microusd())
            )
        self._ledger_row(
            entry,
            rest,
            fingerprint,
            evidence="recorded",
            status=status,
            live_cost=True,
            est_cost_microusd=measured_microusd,
            outcome_evidence=evidence_label,
            live_call=True,
            sidecar_spend_microusd=sidecar_spend_microusd,
            sidecar_spend_kind=(
                "measured"
                if estimate.cost_source in _MEASURED_PROVIDER_COST_SOURCES
                else "estimated"
            ),
        )
        self._respond(status, body, evidence=evidence_label, headers=event.to_headers())

    do_GET = _handle
    do_POST = _handle


def serve_evidence_proxy(
    *,
    host: str = "0.0.0.0",
    port: int = 0,
    baseline_dir: str = "",
    day_cache_path: str = "",
    registry: Sequence[ProviderRegistryEntry] | None = None,
    usage_ledger_path: str = "",
    caller_context: Mapping[str, Any] | None = None,
    caller_token_map_path: str = "",
    key_split: bool | None = None,
    credential_overrides: Mapping[str, str] | None = None,
    dynamic_registry: bool | None = None,
    registry_path: str = "",
    registry_loader: Any | None = None,
    registry_refresh_seconds: int | None = None,
    enforcement_mode: str | None = None,
    model_catalog: LiveTextModelCatalog | None = None,
    outcome_sidecar_path: str = "",
) -> tuple[ThreadingHTTPServer, EvidenceStore, threading.Thread]:
    """Start the proxy; returns (server, store, thread). Caller owns shutdown.

    ``caller_context`` is the spawn-bound identity stamped onto ledger rows
    when a request carries no worker-issued token. ``key_split=None`` defers
    to the ``RESEARCH_LAB_PROVIDER_KEY_SPLIT`` env flag at call time.
    """

    if registry is not None:
        entries = list(registry)
        capabilities = load_effective_provider_capabilities_sync(
            [entry.to_dict() for entry in entries],
            strict_remote=False,
            private_row_loader=lambda: None,
            source_row_loader=lambda: (),
        )
    else:
        entries, capabilities = load_provider_registry_with_capabilities(registry_path)
    errors = validate_provider_registry_entries(entries)
    if errors:
        raise ValueError("invalid provider registry: " + "; ".join(errors))
    refresh_enabled = (
        bool(dynamic_registry)
        if dynamic_registry is not None
        else registry is None and capability_catalog_enabled()
    )
    registry_state = ProviderRegistryState(
        entries=entries,
        capabilities=capabilities,
        registry_path=registry_path,
        refresh_enabled=refresh_enabled,
        refresh_seconds=registry_refresh_seconds,
        loader=registry_loader,
    )
    store = EvidenceStore(baseline_dir=baseline_dir, day_cache_path=day_cache_path)
    ledger = ProviderUsageLedger(
        usage_ledger_path or os.getenv(USAGE_LEDGER_PATH_ENV) or "",
        outcome_sidecar_path=(
            outcome_sidecar_path or os.getenv(PROVIDER_OUTCOME_SIDECAR_ENV) or ""
        ),
    )
    handler = type(
        "BoundProxyHandler",
        (_ProxyHandler,),
        {
            "store": store,
            "registry": {entry.id: entry for entry in entries},
            "registry_state": registry_state,
            "ledger": ledger,
            "caller_context": dict(caller_context or {}),
            "caller_tokens": CallerTokenMap(caller_token_map_path),
            "key_split": key_split,
            "credential_overrides": {str(k): str(v) for k, v in (credential_overrides or {}).items()},
            "model_catalog": model_catalog or LiveTextModelCatalog(),
            "enforcement_mode": str(enforcement_mode or capability_enforcement_mode()),
        },
    )
    server = _CapabilityAwareHTTPServer((host, port), handler)
    server.registry_state = registry_state
    server.usage_ledger = ledger
    registry_state.start()
    thread = threading.Thread(target=server.serve_forever, name="evidence-proxy", daemon=True)
    thread.start()
    return server, store, thread


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Provider evidence proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--baseline-dir", default=os.getenv("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR") or "")
    parser.add_argument("--day-cache", default=os.getenv("RESEARCH_LAB_PROVIDER_EVIDENCE_DAY_CACHE") or "")
    parser.add_argument("--registry", default=os.getenv(REGISTRY_PATH_ENV) or "")
    parser.add_argument("--usage-ledger", default=os.getenv(USAGE_LEDGER_PATH_ENV) or "")
    parser.add_argument(
        "--outcome-sidecar",
        default=os.getenv(PROVIDER_OUTCOME_SIDECAR_ENV) or "",
    )
    parser.add_argument(
        "--caller-token-map",
        default=os.getenv("RESEARCH_LAB_PROXY_CALLER_TOKEN_MAP") or "",
    )
    args = parser.parse_args()
    caller_context: dict[str, Any] = {"caller_kind": "evidence_proxy_default"}
    raw_context = os.getenv("RESEARCH_LAB_PROXY_CALLER_CONTEXT") or ""
    if raw_context:
        try:
            loaded = json.loads(raw_context)
            if isinstance(loaded, Mapping):
                caller_context = dict(loaded)
        except Exception:
            pass
    server, _store, thread = serve_evidence_proxy(
        host=args.host,
        port=args.port,
        baseline_dir=args.baseline_dir,
        day_cache_path=args.day_cache,
        registry_path=args.registry,
        dynamic_registry=True,
        usage_ledger_path=args.usage_ledger,
        outcome_sidecar_path=args.outcome_sidecar,
        caller_context=caller_context,
        caller_token_map_path=args.caller_token_map,
    )
    print(
        json.dumps(
            {
                "listening": f"{args.host}:{args.port}",
                **server.registry_state.diagnostic(),
            }
        )
    )
    thread.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
