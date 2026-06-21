"""Local/staged Notary Proxy v2 contract for Research Lab calls.

This module is deliberately inert unless a verifier/test enables local mock
calls. It does not make network requests, write Supabase, touch gateway
runtime, or alter fulfillment behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .canonical import sha256_json
from .evidence import build_evidence_bundle, evidence_refs_from_bundle
from .fabric import build_notary_proxy_egress_stub, validate_notary_proxy_egress_stub
from .loop_start_contract import (
    LoopStartPolicy,
    MinerOpenRouterKeyReference,
    build_provider_key_sources,
)
from .notary import LocalSnapshotStore, NotarySigner, capture_snapshot
from .schema_validation import validate_schema_record
from .trace import build_execution_trace, hash_payload, make_trace_call


NOTARY_PROXY_V2_VERSION = "notary-proxy-v2-local-v0.1.0"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "notary_proxy_v2_fixtures.json"

ALLOWED_PROVIDERS = ("openrouter", "exa", "scrapingdog")
SERVER_SIDE_PROVIDERS = ("exa", "scrapingdog")

PROTECTED_PROXY_KEYS = {
    "api_key",
    "openrouter_api_key",
    "openrouter_key",
    "raw_key",
    "raw_secret",
    "secret",
    "token",
    "credential",
    "credential_value",
}
PROTECTED_PROXY_MARKERS = (
    "sk-or-",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
)


@dataclass(frozen=True)
class NotaryProxyRuntimeFlags:
    local_mock_calls_enabled: bool = False
    production_network_enabled: bool = False
    gateway_runtime_enabled: bool = False
    supabase_writes_enabled: bool = False
    validator_integration_enabled: bool = False
    fulfillment_touch_enabled: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "NotaryProxyRuntimeFlags":
        data = data or {}
        return cls(
            local_mock_calls_enabled=bool(data.get("local_mock_calls_enabled", False)),
            production_network_enabled=bool(data.get("production_network_enabled", False)),
            gateway_runtime_enabled=bool(data.get("gateway_runtime_enabled", False)),
            supabase_writes_enabled=bool(data.get("supabase_writes_enabled", False)),
            validator_integration_enabled=bool(data.get("validator_integration_enabled", False)),
            fulfillment_touch_enabled=bool(data.get("fulfillment_touch_enabled", False)),
        )

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class NotaryProxyBudget:
    job_budget_cents: int
    per_provider_budget_cents: dict[str, int]
    max_calls_per_provider: dict[str, int]
    hard_stop_on_overrun: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "NotaryProxyBudget":
        return cls(
            job_budget_cents=int(data["job_budget_cents"]),
            per_provider_budget_cents={
                str(provider): int(cents)
                for provider, cents in dict(data.get("per_provider_budget_cents", {})).items()
            },
            max_calls_per_provider={
                str(provider): int(count)
                for provider, count in dict(data.get("max_calls_per_provider", {})).items()
            },
            hard_stop_on_overrun=bool(data.get("hard_stop_on_overrun", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_budget_cents": self.job_budget_cents,
            "per_provider_budget_cents": dict(sorted(self.per_provider_budget_cents.items())),
            "max_calls_per_provider": dict(sorted(self.max_calls_per_provider.items())),
            "hard_stop_on_overrun": self.hard_stop_on_overrun,
        }


@dataclass(frozen=True)
class NotaryProxyProviderRoute:
    provider: str
    key_source: str
    key_ref: str
    usage: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "NotaryProxyProviderRoute":
        return cls(
            provider=str(data["provider"]),
            key_source=str(data["key_source"]),
            key_ref=str(data.get("key_ref", "")),
            usage=str(data.get("usage", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "key_source": self.key_source,
            "key_ref_hash": hash_payload(self.key_ref),
            "usage": self.usage,
        }


@dataclass(frozen=True)
class MockProviderCall:
    call_id: str
    provider: str
    method: str
    url: str
    endpoint: str
    purpose: str
    model: str
    component: str | None
    request_payload: dict[str, Any]
    response_payload: dict[str, Any]
    response_snapshot_content: str
    status_code: int
    cost_cents: int
    tokens_in: int = 0
    tokens_out: int = 0
    call_emitter: str = "code"
    teacher_model_flag: bool = False
    fetch_ts: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MockProviderCall":
        return cls(
            call_id=str(data["call_id"]),
            provider=str(data["provider"]),
            method=str(data.get("method", "GET")).upper(),
            url=str(data["url"]),
            endpoint=str(data.get("endpoint", "")),
            purpose=str(data.get("purpose", "other")),
            model=str(data.get("model", data["provider"])),
            component=data.get("component"),
            request_payload=dict(data.get("request_payload", {})),
            response_payload=dict(data.get("response_payload", {})),
            response_snapshot_content=str(data.get("response_snapshot_content", "")),
            status_code=int(data.get("status_code", 200)),
            cost_cents=int(data.get("cost_cents", 0)),
            tokens_in=int(data.get("tokens_in", 0)),
            tokens_out=int(data.get("tokens_out", 0)),
            call_emitter=str(data.get("call_emitter", "code")),
            teacher_model_flag=bool(data.get("teacher_model_flag", False)),
            fetch_ts=data.get("fetch_ts"),
        )

    @property
    def cost_usd(self) -> float:
        return round(self.cost_cents / 100.0, 6)


def validate_runtime_flags(flags: NotaryProxyRuntimeFlags | Mapping[str, Any]) -> list[str]:
    if not isinstance(flags, NotaryProxyRuntimeFlags):
        flags = NotaryProxyRuntimeFlags.from_mapping(flags)
    errors: list[str] = []
    if not flags.local_mock_calls_enabled:
        errors.append("local_mock_calls_disabled")
    if flags.production_network_enabled:
        errors.append("production_network_must_remain_disabled")
    if flags.gateway_runtime_enabled:
        errors.append("gateway_runtime_must_remain_disabled")
    if flags.supabase_writes_enabled:
        errors.append("supabase_writes_must_remain_disabled")
    if flags.validator_integration_enabled:
        errors.append("validator_integration_must_remain_disabled")
    if flags.fulfillment_touch_enabled:
        errors.append("fulfillment_touch_must_remain_disabled")
    return errors


def validate_budget(budget: NotaryProxyBudget | Mapping[str, Any]) -> list[str]:
    if not isinstance(budget, NotaryProxyBudget):
        budget = NotaryProxyBudget.from_mapping(budget)
    errors: list[str] = []
    if budget.job_budget_cents < 0:
        errors.append("job_budget_cents must be non-negative")
    for provider, cents in budget.per_provider_budget_cents.items():
        if provider not in ALLOWED_PROVIDERS:
            errors.append(f"budget provider not allowlisted: {provider}")
        if cents < 0:
            errors.append(f"provider budget must be non-negative: {provider}")
    for provider, count in budget.max_calls_per_provider.items():
        if provider not in ALLOWED_PROVIDERS:
            errors.append(f"rate-limit provider not allowlisted: {provider}")
        if count < 0:
            errors.append(f"provider call limit must be non-negative: {provider}")
    return errors


def build_proxy_provider_routes(
    *,
    miner_openrouter_key_ref: MinerOpenRouterKeyReference | Mapping[str, Any],
    loop_start_policy: LoopStartPolicy | Mapping[str, Any],
) -> tuple[NotaryProxyProviderRoute, ...]:
    sources = build_provider_key_sources(miner_openrouter_key_ref, loop_start_policy)
    routes = tuple(NotaryProxyProviderRoute.from_mapping(source) for source in sources)
    errors = validate_provider_routes(routes)
    if errors:
        raise ValueError("; ".join(errors))
    return routes


def validate_provider_routes(routes: Sequence[NotaryProxyProviderRoute | Mapping[str, Any]]) -> list[str]:
    route_records = [
        route if isinstance(route, NotaryProxyProviderRoute) else NotaryProxyProviderRoute.from_mapping(route)
        for route in routes
    ]
    errors: list[str] = []
    seen = {route.provider for route in route_records}
    for provider in ALLOWED_PROVIDERS:
        if provider not in seen:
            errors.append(f"missing provider route: {provider}")
    for route in route_records:
        if route.provider not in ALLOWED_PROVIDERS:
            errors.append(f"provider not allowlisted: {route.provider}")
        if route.provider == "openrouter" and route.key_source != "miner":
            errors.append("openrouter must route through miner key reference")
        if route.provider == "openrouter" and not route.key_ref:
            errors.append("openrouter miner key reference is required")
        if route.provider in SERVER_SIDE_PROVIDERS and route.key_source != "leadpoet_server_side":
            errors.append(f"{route.provider} must route through Leadpoet server-side key source")
        if route.provider in SERVER_SIDE_PROVIDERS and not route.key_ref:
            errors.append(f"{route.provider} server-side key reference is required")
        if contains_secret_material(route.to_dict()):
            errors.append(f"route contains raw secret material: {route.provider}")
    return _dedupe(errors)


def run_local_notary_proxy_v2(
    *,
    run_id: str,
    artifact_hash: str,
    icp_set_hash: str,
    eval_version: Mapping[str, str],
    flags: NotaryProxyRuntimeFlags | Mapping[str, Any],
    budget: NotaryProxyBudget | Mapping[str, Any],
    routes: Sequence[NotaryProxyProviderRoute | Mapping[str, Any]],
    calls: Sequence[MockProviderCall | Mapping[str, Any]],
    snapshot_store: LocalSnapshotStore,
    signer: NotarySigner,
    delete_snapshot_ref: str | None = None,
) -> dict[str, Any]:
    if not isinstance(flags, NotaryProxyRuntimeFlags):
        flags = NotaryProxyRuntimeFlags.from_mapping(flags)
    if not isinstance(budget, NotaryProxyBudget):
        budget = NotaryProxyBudget.from_mapping(budget)
    route_records = tuple(
        route if isinstance(route, NotaryProxyProviderRoute) else NotaryProxyProviderRoute.from_mapping(route)
        for route in routes
    )
    call_records = tuple(call if isinstance(call, MockProviderCall) else MockProviderCall.from_mapping(call) for call in calls)

    errors = []
    errors.extend(validate_runtime_flags(flags))
    errors.extend(validate_budget(budget))
    errors.extend(validate_provider_routes(route_records))
    for call in call_records:
        errors.extend(_validate_mock_call(call))
    if errors:
        raise ValueError("; ".join(_dedupe(errors)))

    routes_by_provider = {route.provider: route for route in route_records}
    provider_spend: dict[str, int] = {}
    provider_calls: dict[str, int] = {}
    total_spend = 0
    trace_calls = []
    evidence_bundles = []
    evidence_refs = []
    call_logs = []
    blocked_calls = []
    egress_stubs = []
    deletion_bundle = None
    last_snapshot_ref = ""

    for seq, call in enumerate(call_records, start=1):
        route = routes_by_provider[call.provider]
        budget_denial = _budget_denial(
            call=call,
            budget=budget,
            provider_spend=provider_spend,
            provider_calls=provider_calls,
            total_spend=total_spend,
        )
        if budget_denial is not None:
            blocked_calls.append(budget_denial)
            if budget.hard_stop_on_overrun:
                break
            continue

        request_hash = hash_payload(call.request_payload)
        response_hash = hash_payload(call.response_payload)
        snapshot = capture_snapshot(
            url=call.url,
            content=call.response_snapshot_content,
            store=snapshot_store,
            signer=signer,
            fetch_ts=call.fetch_ts,
            metadata={
                "schema_version": "notary_proxy_v2.snapshot_metadata.v1",
                "call_id": call.call_id,
                "provider": call.provider,
                "method": call.method,
                "endpoint": call.endpoint,
                "status_code": call.status_code,
                "request_hash": request_hash,
                "response_hash": response_hash,
                "key_source": route.key_source,
                "key_ref_hash": hash_payload(route.key_ref),
                "cost_cents": call.cost_cents,
                "live_network_performed": False,
            },
        )
        last_snapshot_ref = snapshot.snapshot_ref

        bundle = build_evidence_bundle(
            artifact_hash=artifact_hash,
            snapshots=[snapshot],
            run_id=run_id,
            created_at=snapshot.fetch_ts,
            retention_class="live_verification",
            verification_state="active",
            merkle_anchor_ref="arweave:pending",
        )
        evidence_bundles.append(bundle)
        evidence_refs.extend(evidence_refs_from_bundle(bundle, signal_indices={snapshot.snapshot_ref: seq - 1}))

        trace_call = make_trace_call(
            seq=seq,
            ts=snapshot.fetch_ts,
            provider=call.provider,
            model=call.model,
            purpose=call.purpose,
            component=call.component,
            request=call.request_payload,
            response=call.response_payload,
            call_emitter=call.call_emitter,
            tokens_in=call.tokens_in,
            tokens_out=call.tokens_out,
            cost_usd=call.cost_usd,
            teacher_model_flag=call.teacher_model_flag,
        )
        trace_calls.append(trace_call)

        egress_stub = build_notary_proxy_egress_stub(
            egress_id=f"egress:notary-proxy-v2:{run_id}:{seq}",
            run_id=run_id,
            url=call.url,
            snapshot=snapshot.to_schema_snapshot(),
            evidence_bundle_ref=bundle["bundle_hash"],
            egress_policy_ref="egress-policy:notary-proxy-v2:local",
            vsock_route_ref="vsock:notary-proxy-v2:local-disabled",
            method=call.method,
        )
        egress_errors = validate_notary_proxy_egress_stub(egress_stub)
        if egress_errors:
            raise ValueError("; ".join(egress_errors))
        egress_stubs.append(egress_stub.to_dict())

        provider_spend[call.provider] = provider_spend.get(call.provider, 0) + call.cost_cents
        provider_calls[call.provider] = provider_calls.get(call.provider, 0) + 1
        total_spend += call.cost_cents
        call_logs.append(
            {
                "seq": seq,
                "call_id": call.call_id,
                "provider": call.provider,
                "purpose": call.purpose,
                "method": call.method,
                "url": call.url,
                "endpoint": call.endpoint,
                "status_code": call.status_code,
                "request_hash": request_hash,
                "response_hash": response_hash,
                "snapshot_ref": snapshot.snapshot_ref,
                "evidence_bundle_ref": bundle["bundle_hash"],
                "key_source": route.key_source,
                "key_ref_hash": hash_payload(route.key_ref),
                "cost_cents": call.cost_cents,
                "cost_usd": call.cost_usd,
            }
        )

    trace = build_execution_trace(
        run_id=run_id,
        artifact_hash=artifact_hash,
        role="candidate",
        rung="L1",
        status="completed",
        icp_set_hash=icp_set_hash,
        eval_version=dict(eval_version),
        calls=trace_calls,
        evidence_refs=evidence_refs,
        outputs_payload={"notary_proxy_v2_call_count": len(call_logs)},
        score_bundle_payload={"notary_proxy_v2_total_cost_cents": total_spend},
        attestation_ref="attestation:notary-proxy-v2:local-only",
    )

    if delete_snapshot_ref or last_snapshot_ref:
        target_ref = delete_snapshot_ref or last_snapshot_ref
        deleted = snapshot_store.delete_content(
            target_ref,
            deletion_reason_ref=f"deletion-request:notary-proxy-v2:{run_id}",
            deleted_at="2026-06-20T00:20:00Z",
        )
        deletion_bundle = build_evidence_bundle(
            artifact_hash=artifact_hash,
            snapshots=[deleted],
            run_id=run_id,
            created_at="2026-06-20T00:20:00Z",
            retention_class="live_verification",
            verification_state="content_deleted",
            deletion_request_ref=f"deletion-request:notary-proxy-v2:{run_id}",
            merkle_anchor_ref="arweave:pending",
        )

    cost_ledger = {
        "schema_version": "notary_proxy_v2.cost_ledger.v1",
        "total_cents": total_spend,
        "total_usd": round(total_spend / 100.0, 6),
        "by_provider_cents": dict(sorted(provider_spend.items())),
        "by_provider_usd": {
            provider: round(cents / 100.0, 6)
            for provider, cents in sorted(provider_spend.items())
        },
        "blocked_call_count": len(blocked_calls),
    }

    public_bundle_without_hash = {
        "schema_version": "notary_proxy_v2.public_bundle.v1",
        "proxy_version": NOTARY_PROXY_V2_VERSION,
        "run_id": run_id,
        "artifact_hash": artifact_hash,
        "icp_set_hash": icp_set_hash,
        "status": "completed_with_budget_denial" if blocked_calls else "completed",
        "runtime_flags": flags.to_dict(),
        "provider_routes": [route.to_public_dict() for route in route_records],
        "budget": budget.to_dict(),
        "call_log": call_logs,
        "blocked_calls": blocked_calls,
        "cost_ledger": cost_ledger,
        "execution_trace_hash": hash_payload(trace),
        "evidence_bundle_hashes": [bundle["bundle_hash"] for bundle in evidence_bundles],
        "deletion_bundle_hash": deletion_bundle["bundle_hash"] if deletion_bundle else None,
        "egress_stubs": egress_stubs,
        "live_network_performed": False,
        "production_writes_performed": False,
        "fulfillment_touched": False,
    }
    assert_no_secret_material(public_bundle_without_hash, label="public_bundle")
    public_bundle = {
        **public_bundle_without_hash,
        "public_bundle_hash": hash_payload(public_bundle_without_hash),
    }

    result = {
        "schema_version": "notary_proxy_v2.result.v1",
        "public_bundle": public_bundle,
        "execution_trace": trace,
        "evidence_bundles": evidence_bundles,
        "deletion_bundle": deletion_bundle,
        "cost_ledger": cost_ledger,
        "call_log": call_logs,
        "blocked_calls": blocked_calls,
        "provider_routes": [route.to_public_dict() for route in route_records],
    }
    errors = validate_notary_proxy_v2_result(result)
    if errors:
        raise ValueError("; ".join(errors))
    return result


def validate_notary_proxy_v2_result(result: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        assert_no_secret_material(result, label="notary_proxy_v2_result")
    except ValueError as exc:
        errors.append(str(exc))

    public_bundle = dict(result.get("public_bundle", {}))
    if public_bundle.get("live_network_performed"):
        errors.append("live network must not be performed in staged/local mode")
    if public_bundle.get("production_writes_performed"):
        errors.append("production writes must not be performed in staged/local mode")
    if public_bundle.get("fulfillment_touched"):
        errors.append("fulfillment must not be touched by Notary Proxy v2 local contract")
    flags = NotaryProxyRuntimeFlags.from_mapping(public_bundle.get("runtime_flags", {}))
    if flags.production_network_enabled or flags.gateway_runtime_enabled or flags.supabase_writes_enabled:
        errors.append("runtime flags expose a production path")

    trace = dict(result.get("execution_trace", {}))
    errors.extend(validate_schema_record("execution_trace.schema.json", trace))
    for bundle in result.get("evidence_bundles", []):
        errors.extend(validate_schema_record("evidence_bundle.schema.json", dict(bundle)))
    deletion_bundle = result.get("deletion_bundle")
    if deletion_bundle is not None:
        errors.extend(validate_schema_record("evidence_bundle.schema.json", dict(deletion_bundle)))
        if deletion_bundle.get("verification_state") != "content_deleted":
            errors.append("deletion bundle must retain content_deleted verification state")

    trace_total = int(round(float(trace.get("cost_ledger", {}).get("total_usd", 0)) * 100))
    ledger_total = int(dict(result.get("cost_ledger", {})).get("total_cents", -1))
    if trace_total != ledger_total:
        errors.append("execution trace cost ledger does not match proxy cost ledger")
    if public_bundle.get("cost_ledger", {}).get("total_cents") != ledger_total:
        errors.append("public bundle cost ledger does not match proxy cost ledger")

    for route in public_bundle.get("provider_routes", []):
        if "key_ref" in route:
            errors.append("public provider routes must expose only key_ref_hash, not key_ref")
        if route.get("provider") == "openrouter" and route.get("key_source") != "miner":
            errors.append("public OpenRouter route must show miner key source")
        if route.get("provider") in SERVER_SIDE_PROVIDERS and route.get("key_source") != "leadpoet_server_side":
            errors.append(f"public {route.get('provider')} route must show Leadpoet server-side key source")
    return _dedupe(errors)


def verify_research_lab_notary_proxy_v2(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    fixture = _load_fixture(Path(fixture_path))

    disabled_errors = validate_runtime_flags(NotaryProxyRuntimeFlags())
    _assert("local_mock_calls_disabled" in disabled_errors, "default flags block execution")

    unsafe_errors = validate_runtime_flags(
        NotaryProxyRuntimeFlags.from_mapping(
            {
                "local_mock_calls_enabled": True,
                "production_network_enabled": True,
                "supabase_writes_enabled": True,
            }
        )
    )
    _assert("production_network_must_remain_disabled" in unsafe_errors, "production network flag is rejected")
    _assert("supabase_writes_must_remain_disabled" in unsafe_errors, "Supabase writes flag is rejected")

    policy = LoopStartPolicy.from_mapping(fixture["loop_start_policy"])
    key_ref = MinerOpenRouterKeyReference.from_mapping(fixture["miner_openrouter_key_ref"])
    routes = build_proxy_provider_routes(miner_openrouter_key_ref=key_ref, loop_start_policy=policy)
    _assert(_route_source(routes, "openrouter") == "miner", "OpenRouter routes to miner key reference")
    _assert(_route_source(routes, "exa") == "leadpoet_server_side", "Exa routes to Leadpoet server-side key")
    _assert(
        _route_source(routes, "scrapingdog") == "leadpoet_server_side",
        "ScrapingDog routes to Leadpoet server-side key",
    )

    with tempfile.TemporaryDirectory(prefix="leadpoet_notary_proxy_v2_") as tmpdir:
        result = run_local_notary_proxy_v2(
            run_id=fixture["run"]["run_id"],
            artifact_hash=fixture["run"]["artifact_hash"],
            icp_set_hash=fixture["run"]["icp_set_hash"],
            eval_version=fixture["run"]["eval_version"],
            flags=fixture["runtime_flags"],
            budget=fixture["budget"],
            routes=routes,
            calls=fixture["mock_provider_calls"],
            snapshot_store=LocalSnapshotStore(Path(tmpdir) / "snapshots"),
            signer=NotarySigner("notary-proxy-v2-local-test", key_id="notary-proxy-v2"),
        )

        _assert(not validate_notary_proxy_v2_result(result), "proxy result validates offline")
        _assert(result["cost_ledger"]["total_cents"] == fixture["expectations"]["total_success_cost_cents"], "cost totals match")
        _assert(len(result["call_log"]) == fixture["expectations"]["successful_call_count"], "successful calls logged")
        _assert(len(result["blocked_calls"]) == fixture["expectations"]["blocked_call_count"], "budget overrun blocks call")
        _assert(
            result["blocked_calls"][0]["reason"] == fixture["expectations"]["blocked_reason"],
            "blocked call records budget reason",
        )
        _assert(
            result["deletion_bundle"]["verification_state"] == "content_deleted",
            "snapshot deletion leaves content_deleted evidence bundle",
        )

        deleted_ref = result["deletion_bundle"]["snapshots"][0]["snapshot_ref"]
        _assert_raises(
            lambda: LocalSnapshotStore(Path(tmpdir) / "snapshots").read_content(deleted_ref),
            "deleted snapshot content is unavailable while hashes remain",
        )

    with tempfile.TemporaryDirectory(prefix="leadpoet_notary_proxy_v2_repeat_") as tmpdir:
        repeat = run_local_notary_proxy_v2(
            run_id=fixture["run"]["run_id"],
            artifact_hash=fixture["run"]["artifact_hash"],
            icp_set_hash=fixture["run"]["icp_set_hash"],
            eval_version=fixture["run"]["eval_version"],
            flags=fixture["runtime_flags"],
            budget=fixture["budget"],
            routes=routes,
            calls=fixture["mock_provider_calls"],
            snapshot_store=LocalSnapshotStore(Path(tmpdir) / "snapshots"),
            signer=NotarySigner("notary-proxy-v2-local-test", key_id="notary-proxy-v2"),
        )
    _assert(
        repeat["public_bundle"]["public_bundle_hash"] == result["public_bundle"]["public_bundle_hash"],
        "public bundle is deterministic",
    )

    return {
        "public_bundle_hash": result["public_bundle"]["public_bundle_hash"],
        "execution_trace_hash": result["public_bundle"]["execution_trace_hash"],
        "evidence_bundle_count": len(result["evidence_bundles"]),
        "successful_call_count": len(result["call_log"]),
        "blocked_call_count": len(result["blocked_calls"]),
        "total_cost_cents": result["cost_ledger"]["total_cents"],
        "deletion_bundle_hash": result["deletion_bundle"]["bundle_hash"],
        "provider_routes": {
            route["provider"]: route["key_source"]
            for route in result["provider_routes"]
        },
    }


def _validate_mock_call(call: MockProviderCall) -> list[str]:
    errors: list[str] = []
    if call.provider not in ALLOWED_PROVIDERS:
        errors.append(f"provider not allowlisted: {call.provider}")
    if call.method not in {"GET", "POST"}:
        errors.append("method must be GET or POST")
    if call.status_code < 100 or call.status_code > 599:
        errors.append("status_code must be an HTTP status code")
    if call.cost_cents < 0:
        errors.append("cost_cents must be non-negative")
    if not call.response_snapshot_content:
        errors.append("response_snapshot_content is required")
    if contains_secret_material(call.request_payload) or contains_secret_material(call.response_payload):
        errors.append(f"mock call contains raw secret material: {call.call_id}")
    return errors


def _budget_denial(
    *,
    call: MockProviderCall,
    budget: NotaryProxyBudget,
    provider_spend: Mapping[str, int],
    provider_calls: Mapping[str, int],
    total_spend: int,
) -> dict[str, Any] | None:
    current_provider_calls = int(provider_calls.get(call.provider, 0))
    max_calls = budget.max_calls_per_provider.get(call.provider)
    if max_calls is not None and current_provider_calls + 1 > max_calls:
        return _blocked_call(call, "provider_rate_limit_exceeded", total_spend, provider_spend)

    projected_total = total_spend + call.cost_cents
    if projected_total > budget.job_budget_cents:
        return _blocked_call(call, "job_budget_exceeded", total_spend, provider_spend)

    provider_budget = budget.per_provider_budget_cents.get(call.provider)
    projected_provider = int(provider_spend.get(call.provider, 0)) + call.cost_cents
    if provider_budget is not None and projected_provider > provider_budget:
        return _blocked_call(call, "provider_budget_exceeded", total_spend, provider_spend)
    return None


def _blocked_call(
    call: MockProviderCall,
    reason: str,
    total_spend: int,
    provider_spend: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "call_id": call.call_id,
        "provider": call.provider,
        "purpose": call.purpose,
        "reason": reason,
        "attempted_cost_cents": call.cost_cents,
        "spent_before_cents": total_spend,
        "provider_spent_before_cents": int(provider_spend.get(call.provider, 0)),
        "request_hash": hash_payload(call.request_payload),
        "response_hash": hash_payload(call.response_payload),
    }


def contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_lower = str(key).lower()
            if key_lower in PROTECTED_PROXY_KEYS or key_lower.startswith("raw_") or key_lower.endswith("_api_key"):
                return True
            if contains_secret_material(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in PROTECTED_PROXY_MARKERS)
    return False


def assert_no_secret_material(value: Any, *, label: str) -> None:
    if contains_secret_material(value):
        raise ValueError(f"{label} contains raw provider secret material")


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _route_source(routes: Sequence[NotaryProxyProviderRoute], provider: str) -> str:
    for route in routes:
        if route.provider == provider:
            return route.key_source
    return ""


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def _assert_raises(fn, label: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(label)
