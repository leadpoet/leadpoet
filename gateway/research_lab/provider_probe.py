"""Gateway-side resolution for the loop's `probe_provider` operation (W4).

A probe is one metered provider call made on the loop's behalf through the
evidence proxy (baseline tape → shared day cache → live). Guard rails, in
order, before any upstream contact:

1. typed catalog only — the endpoint must exist in the probe catalog and the
   params must validate against its schema (no free-form URLs);
2. query guard — normalized param values are hash-compared against the
   private-window ICP/company term set and substring-checked against
   ``FORBIDDEN_CODE_EDIT_TERMS``; a hit blocks the probe with no upstream
   contact (``probe_blocked``);
3. per-loop budget — probe count and estimated cost caps.

The verbatim response records into the proxy's day cache (benchmark
reproducibility) and dual-writes a snapshot-store-format overlay so
probe-motivated candidates stay dev-replay-scorable; the loop only ever sees
the sanitized, size-capped projection.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from gateway.research_lab.code_build import _redact_secret_values
from gateway.research_lab.provider_evidence_proxy import (
    CALLER_TOKEN_HEADER,
    REPLAY_ONLY_HEADER,
)
from research_lab.canonical import sha256_json
from research_lab.code_editing import FORBIDDEN_CODE_EDIT_TERMS, CodeEditSourceInspectionRequest
from research_lab.probe_catalog import (
    ProviderProbeEndpoint,
    find_probe_endpoint,
    validate_probe_params,
)

PROBE_RESPONSE_MAX_CHARS = 12_000

DEFAULT_PROBE_MAX_PROBES = 4
DEFAULT_PROBE_MAX_COST_MICROUSD = 250_000  # $0.25


def _normalize_probe_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _term_hash(value: str) -> str:
    return hashlib.sha256(_normalize_probe_text(value).encode("utf-8")).hexdigest()


def hash_private_window_terms(terms: Iterable[str]) -> frozenset[str]:
    """Hash the private-window ICP/company term set for probe screening.

    The engine holds only these hashes — plaintext window terms never sit in
    probe-guard state.
    """

    hashes = set()
    for term in terms:
        normalized = _normalize_probe_text(term)
        if len(normalized) >= 3:
            hashes.add(_term_hash(normalized))
    return frozenset(hashes)


# ICP-item keys whose string values are identifier-shaped (company names,
# domains) and belong in the guard term set. Long free-text ICP descriptions
# are deliberately excluded: the guard compares probe n-grams (≤4 words), so
# multi-sentence text can never match and would only bloat the set.
_GUARD_TERM_KEY_MARKERS = ("company", "domain", "website", "account", "brand", "employer")


def probe_guard_terms_from_icp_items(items: Iterable[Mapping[str, Any]]) -> set[str]:
    """Extract identifier-shaped private-window terms from benchmark ICP items."""

    terms: set[str] = set()

    def _walk(value: Any, key: str) -> None:
        if isinstance(value, Mapping):
            for nested_key, nested in value.items():
                _walk(nested, str(nested_key).lower())
        elif isinstance(value, (list, tuple)):
            for nested in value:
                _walk(nested, key)
        elif isinstance(value, str):
            text = value.strip()
            if 3 <= len(text) <= 120 and any(marker in key for marker in _GUARD_TERM_KEY_MARKERS):
                terms.add(text)

    for item in items:
        if isinstance(item, Mapping):
            _walk(item, "")
    return terms


async def build_probe_guard_term_hashes() -> frozenset[str]:
    """Fetch the current private ICP window and hash its guard terms.

    Called by the hosted worker once per run when probes are enabled. Raises
    on window unavailability — the caller fails CLOSED (probes stay off for
    the run) rather than probing unscreened.
    """

    from gateway.research_lab.icp_window import fetch_rolling_icp_window

    window = await fetch_rolling_icp_window(allow_partial=True)
    return hash_private_window_terms(probe_guard_terms_from_icp_items(window.benchmark_items))


def probe_query_guard(
    params: Mapping[str, Any],
    *,
    private_window_term_hashes: frozenset[str] = frozenset(),
) -> str:
    """Return a block reason ('' = allowed). Never returns the offending text."""

    for value in params.values():
        if not isinstance(value, str):
            continue
        normalized = _normalize_probe_text(value)
        if not normalized:
            continue
        for term in FORBIDDEN_CODE_EDIT_TERMS:
            if term in normalized:
                return "forbidden_term"
        if private_window_term_hashes:
            candidates = {normalized}
            candidates.update(token for token in normalized.split(" ") if len(token) >= 3)
            # Sliding word n-grams (2..4) so multi-word company/ICP names are
            # caught inside longer queries, not only as exact full-value matches.
            tokens = normalized.split(" ")
            for size in (2, 3, 4):
                for index in range(0, max(0, len(tokens) - size + 1)):
                    candidates.add(" ".join(tokens[index : index + size]))
            for candidate in candidates:
                if _term_hash(candidate) in private_window_term_hashes:
                    return "private_window_term"
    return ""


@dataclass
class ProbeBudgetState:
    """Per-loop probe accounting (mutable; one instance per run)."""

    max_probes: int = DEFAULT_PROBE_MAX_PROBES
    max_cost_microusd: int = DEFAULT_PROBE_MAX_COST_MICROUSD
    probes_used: int = 0
    cost_used_microusd: int = 0

    def can_spend(self, est_cost_microusd: int) -> bool:
        if self.probes_used >= self.max_probes:
            return False
        return self.cost_used_microusd + max(0, int(est_cost_microusd)) <= self.max_cost_microusd

    def charge(self, est_cost_microusd: int) -> None:
        self.probes_used += 1
        self.cost_used_microusd += max(0, int(est_cost_microusd))

    def to_context(self) -> dict[str, Any]:
        return {
            "max_probes": self.max_probes,
            "probes_used": self.probes_used,
            "max_cost_microusd": self.max_cost_microusd,
            "cost_used_microusd": self.cost_used_microusd,
        }


@dataclass(frozen=True)
class ProbeResolution:
    outcome: str  # resolved | blocked | budget_exhausted | upstream_error | replay_miss | invalid
    model_result: dict[str, Any]
    event_doc: dict[str, Any]
    cost_microusd: int = 0
    snapshot_overlay_written: bool = False


def _sanitize_probe_body(body: bytes) -> tuple[str, bool]:
    text = body.decode("utf-8", errors="replace")
    truncated = len(text) > PROBE_RESPONSE_MAX_CHARS
    return _redact_secret_values(text[:PROBE_RESPONSE_MAX_CHARS]), truncated


def _write_snapshot_overlay(
    *,
    overlay_uri: str,
    upstream_base_url: str,
    endpoint: ProviderProbeEndpoint,
    query_params: Mapping[str, Any],
    body_params: Mapping[str, Any] | None,
    status: int,
    body: bytes,
) -> bool:
    """Dual-write the verbatim response in dev-snapshot-store format.

    Best-effort by design: a failed overlay write must not fail the probe —
    it only means a request-shape-changing candidate scores as a snapshot
    miss later (surfaced by W2's ``snapshot_miss_count``).
    """

    if not overlay_uri:
        return False
    try:
        from research_lab.eval.snapshot_store import (
            MODE_RECORD,
            ProviderSnapshotStore,
            build_snapshot_request,
        )

        url = upstream_base_url.rstrip("/") + endpoint.path
        if query_params:
            url += "?" + urllib.parse.urlencode(sorted((str(k), str(v)) for k, v in query_params.items()))
        request = build_snapshot_request(
            endpoint.method,
            url,
            body=json.dumps(dict(body_params), sort_keys=True) if body_params else None,
        )
        store = ProviderSnapshotStore(overlay_uri, mode=MODE_RECORD)
        body_text = body.decode("utf-8", errors="replace")
        try:
            store.record_response(request, status=int(status), body_text=body_text)
        except Exception:
            # The snapshot store refuses secret-bearing bodies; a redacted
            # overlay still keeps the candidate replay-scorable.
            store.record_response(request, status=int(status), body_text=_redact_secret_values(body_text))
        return True
    except Exception:
        return False


def resolve_provider_probe(
    request: CodeEditSourceInspectionRequest,
    *,
    catalog: list[ProviderProbeEndpoint],
    proxy_url: str,
    budget: ProbeBudgetState,
    live_enabled: bool,
    private_window_term_hashes: frozenset[str] = frozenset(),
    registry_base_urls: Mapping[str, str] | None = None,
    snapshot_overlay_uri: str = "",
    caller_token: str = "",
    timeout_seconds: int = 60,
) -> ProbeResolution:
    """Resolve one probe_provider request. Synchronous (callers thread it)."""

    params_hash = sha256_json({"params": dict(request.params)}) if request.params else ""
    base_event: dict[str, Any] = {
        "operation": "probe_provider",
        "endpoint_id": request.endpoint,
        "provider": request.provider,
        "params_hash": params_hash,
        "rationale_hash": sha256_json({"rationale": request.rationale}) if request.rationale else "",
        # Axis-B labeling: probes are external side-information for trajectory
        # capture, never axis-A agentic tool output.
        "axis": "B_external_side_information",
    }

    endpoint = find_probe_endpoint(catalog, request.endpoint)
    if endpoint is None or (request.provider and request.provider != endpoint.provider_id):
        return ProbeResolution(
            outcome="blocked",
            model_result={
                "operation": "probe_provider",
                "endpoint": request.endpoint,
                "blocked": True,
                "block_reason": "unknown_endpoint",
            },
            event_doc={**base_event, "blocked": True, "block_reason": "unknown_endpoint"},
        )
    normalized_params, param_errors = validate_probe_params(endpoint, request.params)
    if param_errors:
        return ProbeResolution(
            outcome="blocked",
            model_result={
                "operation": "probe_provider",
                "endpoint": endpoint.endpoint_id,
                "blocked": True,
                "block_reason": "param_schema_violation",
                "param_errors": param_errors[:6],
            },
            event_doc={**base_event, "blocked": True, "block_reason": "param_schema_violation"},
        )
    guard_reason = probe_query_guard(normalized_params, private_window_term_hashes=private_window_term_hashes)
    if guard_reason:
        return ProbeResolution(
            outcome="blocked",
            model_result={
                "operation": "probe_provider",
                "endpoint": endpoint.endpoint_id,
                "blocked": True,
                "block_reason": guard_reason,
            },
            event_doc={**base_event, "blocked": True, "block_reason": guard_reason},
        )
    if not budget.can_spend(endpoint.est_cost_microusd):
        return ProbeResolution(
            outcome="budget_exhausted",
            model_result={
                "operation": "probe_provider",
                "endpoint": endpoint.endpoint_id,
                "skipped": "probe_budget_exhausted",
                "budget": budget.to_context(),
            },
            event_doc={**base_event, "skipped": "probe_budget_exhausted", "budget": budget.to_context()},
        )
    if not proxy_url:
        return ProbeResolution(
            outcome="upstream_error",
            model_result={
                "operation": "probe_provider",
                "endpoint": endpoint.endpoint_id,
                "error_class": "probe_proxy_unconfigured",
            },
            event_doc={**base_event, "error_class": "probe_proxy_unconfigured"},
        )

    query_params = {
        name: value for name, value in normalized_params.items()
        if _param_location(endpoint, name) == "query"
    }
    body_params = {
        name: value for name, value in normalized_params.items()
        if _param_location(endpoint, name) == "body"
    }
    target = proxy_url.rstrip("/") + "/" + endpoint.provider_id + endpoint.path
    if query_params:
        target += "?" + urllib.parse.urlencode(sorted((str(k), str(v)) for k, v in query_params.items()))
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if caller_token:
        headers[CALLER_TOKEN_HEADER] = caller_token
    if not live_enabled:
        headers[REPLAY_ONLY_HEADER] = "1"
    data = json.dumps(body_params, sort_keys=True).encode("utf-8") if endpoint.method == "POST" else None
    http_request = urllib.request.Request(target, data=data, headers=headers, method=endpoint.method)
    try:
        with urllib.request.urlopen(http_request, timeout=max(5, int(timeout_seconds))) as response:
            status = int(getattr(response, "status", 0) or 0)
            body = response.read()
            evidence = str(response.headers.get("X-Research-Lab-Evidence") or "")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            body = exc.read()
        except Exception:
            body = b""
        evidence = str(exc.headers.get("X-Research-Lab-Evidence") or "") if exc.headers else ""
        if status == 409 and not live_enabled:
            return ProbeResolution(
                outcome="replay_miss",
                model_result={
                    "operation": "probe_provider",
                    "endpoint": endpoint.endpoint_id,
                    "error_class": "probe_replay_miss",
                    "detail": "live probes are disabled and no recorded evidence matched",
                },
                event_doc={**base_event, "error_class": "probe_replay_miss"},
            )
    except Exception:
        return ProbeResolution(
            outcome="upstream_error",
            model_result={
                "operation": "probe_provider",
                "endpoint": endpoint.endpoint_id,
                "error_class": "probe_upstream_error",
            },
            event_doc={**base_event, "error_class": "probe_upstream_error"},
        )

    # Charge only calls that reached evidence resolution (replayed or live) —
    # blocked/miss paths above spent nothing upstream.
    cost = endpoint.est_cost_microusd if evidence == "recorded" else 0
    budget.charge(cost)
    overlay_written = False
    if status < 500:
        overlay_written = _write_snapshot_overlay(
            overlay_uri=snapshot_overlay_uri,
            upstream_base_url=(registry_base_urls or {}).get(endpoint.provider_id, ""),
            endpoint=endpoint,
            query_params=query_params,
            body_params=body_params if endpoint.method == "POST" else None,
            status=status,
            body=body,
        )
    sanitized, truncated = _sanitize_probe_body(body)
    if status >= 500:
        return ProbeResolution(
            outcome="upstream_error",
            model_result={
                "operation": "probe_provider",
                "endpoint": endpoint.endpoint_id,
                "status": status,
                "error_class": "probe_upstream_error",
            },
            event_doc={**base_event, "status": status, "error_class": "probe_upstream_error", "evidence": evidence},
            cost_microusd=cost,
        )
    return ProbeResolution(
        outcome="resolved",
        model_result={
            "operation": "probe_provider",
            "endpoint": endpoint.endpoint_id,
            "provider": endpoint.provider_id,
            "status": status,
            "content": sanitized,
            "content_truncated": truncated,
            "budget": budget.to_context(),
        },
        event_doc={
            **base_event,
            "status": status,
            "evidence": evidence,
            "response_hash": sha256_json({"body": sanitized}),
            "bytes_returned": len(body),
            "est_cost_microusd": cost,
            "snapshot_overlay_written": overlay_written,
            "budget": budget.to_context(),
        },
        cost_microusd=cost,
        snapshot_overlay_written=overlay_written,
    )


def _param_location(endpoint: ProviderProbeEndpoint, name: str) -> str:
    for spec in endpoint.params:
        if spec.name == name:
            return spec.location
    return "query"
