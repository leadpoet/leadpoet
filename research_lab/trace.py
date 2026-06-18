"""Execution trace request/response lineage ledger helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional
import uuid

from .canonical import coerce_iso_z, sha256_bytes, sha256_json, utc_now_iso
from .schema_validation import assert_schema_record


def hash_payload(payload: Any) -> str:
    if isinstance(payload, bytes):
        return sha256_bytes(payload)
    if isinstance(payload, str):
        return sha256_bytes(payload.encode("utf-8"))
    return sha256_json(payload)


@dataclass(frozen=True)
class TraceCall:
    seq: int
    ts: str
    provider: str
    model: str
    purpose: str
    call_emitter: str
    component: Optional[str]
    request_hash: str
    response_hash: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    teacher_model_flag: bool = False

    def to_schema_call(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "provider": self.provider,
            "model": self.model,
            "purpose": self.purpose,
            "call_emitter": self.call_emitter,
            "component": self.component,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "tokens": {"in": self.tokens_in, "out": self.tokens_out},
            "cost_usd": round(float(self.cost_usd), 6),
            "teacher_model_flag": self.teacher_model_flag,
        }


def make_trace_call(
    *,
    seq: int,
    provider: str,
    model: str,
    purpose: str,
    request: Any,
    response: Any,
    ts: str | None = None,
    call_emitter: str = "code",
    component: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    teacher_model_flag: bool = False,
) -> TraceCall:
    return TraceCall(
        seq=seq,
        ts=coerce_iso_z(ts) if ts else utc_now_iso(),
        provider=provider,
        model=model,
        purpose=purpose,
        call_emitter=call_emitter,
        component=component,
        request_hash=hash_payload(request),
        response_hash=hash_payload(response),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        teacher_model_flag=teacher_model_flag,
    )


def build_execution_trace(
    *,
    artifact_hash: str,
    role: str,
    rung: str,
    status: str,
    icp_set_hash: str,
    eval_version: dict[str, str],
    calls: Iterable[TraceCall | dict[str, Any]],
    evidence_refs: Iterable[dict[str, Any]],
    run_id: str | None = None,
    lane_id: str | None = None,
    judge_verdicts: Optional[list[dict[str, Any]]] = None,
    outputs_payload: Any = None,
    score_bundle_payload: Any = None,
    attestation_ref: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    call_records = [
        call.to_schema_call() if isinstance(call, TraceCall) else dict(call)
        for call in calls
    ]
    run_uuid = run_id or str(uuid.uuid4())
    evidence = [dict(ref) for ref in evidence_refs]
    outputs_ref = hash_payload(outputs_payload if outputs_payload is not None else {})
    score_bundle_ref = hash_payload(score_bundle_payload if score_bundle_payload is not None else {})
    cost_ledger = _cost_ledger(call_records)
    trace = {
        "run_id": run_uuid,
        "schema_version": "1.0",
        "artifact_hash": artifact_hash,
        "role": role,
        "rung": rung,
        "status": status,
        "lane_id": lane_id,
        "icp_set_hash": icp_set_hash,
        "eval_version": dict(eval_version),
        "calls": call_records,
        "evidence_bundles": evidence,
        "judge_verdicts": judge_verdicts or [],
        "outputs_ref": outputs_ref,
        "score_bundle_ref": score_bundle_ref,
        "cost_ledger": cost_ledger,
        "attestation_ref": attestation_ref,
    }
    assert_no_raw_lineage(trace)
    if validate:
        assert_schema_record("execution_trace.schema.json", trace)
    return trace


def assert_no_raw_lineage(trace: dict[str, Any]) -> None:
    forbidden = {"request", "response", "raw_request", "raw_response", "prompt", "judge_prompt"}
    _assert_forbidden_keys_absent(trace, forbidden, "$")


def _assert_forbidden_keys_absent(value: Any, forbidden: set[str], path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in forbidden:
                raise ValueError(f"execution trace must not contain raw field {key!r} at {path}")
            _assert_forbidden_keys_absent(item, forbidden, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _assert_forbidden_keys_absent(item, forbidden, f"{path}[{idx}]")


def _cost_ledger(calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_provider: dict[str, float] = {}
    by_purpose: dict[str, float] = {}
    total = 0.0
    for call in calls:
        cost = round(float(call.get("cost_usd", 0.0)), 6)
        total += cost
        provider = str(call.get("provider") or "unknown")
        purpose = str(call.get("purpose") or "other")
        by_provider[provider] = round(by_provider.get(provider, 0.0) + cost, 6)
        by_purpose[purpose] = round(by_purpose.get(purpose, 0.0) + cost, 6)
    return {
        "total_usd": round(total, 6),
        "by_provider": by_provider,
        "by_purpose": by_purpose,
    }
