"""Bounded scorer-only control over the existing worker socket.

This is judgment coordination, not a provider call. It has no provider action
sequence, quota debit, or synthetic provider receipt.
"""
from __future__ import annotations

import asyncio
import json
import socket
import time
from typing import Any, Mapping

SCHEMA = "leadpoet.lab_arena.signal_judgment_control.v1"
POLICY = "signal_judgment_v1"
MAX_BYTES = 1_048_576
REQUEST_FIELDS = {"operation", "company_index", "signal_index", "verified_company_identity", "verified_funding_stage_context"}
FIT_CAPABILITY_OPERATION = "fit_checkpoint_capability"


def fit_checkpoint_capability() -> dict:
    from lab_arena import company_judgments
    return {"status": "supported", "checkpoint_kind": company_judgments.FIT_CHECKPOINT_KIND,
            "schema_version": company_judgments.FIT_CHECKPOINT_SCHEMA_VERSION}


def validate_request(value: Any) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("invalid signal cache request")
    from lab_arena import company_judgments, evidence_judgments
    if (value.get("checkpoint_kind") == evidence_judgments.FIT_KIND
        or value.get("judgment_kind") == evidence_judgments.SIGNAL_KIND):
        return validate_evidence_request(value)
    if value.get("checkpoint_kind") == company_judgments.FIT_CHECKPOINT_KIND:
        expected = {"operation", "company_index", "signal_index", "checkpoint_kind"}
        if value.get("operation") == "complete":
            expected |= {"reference", "judgment"}
        if (set(value) != expected or value.get("operation") not in {"claim", "complete", FIT_CAPABILITY_OPERATION}
            or type(value.get("company_index")) is not int or not 0 <= value["company_index"] < 5
            or type(value.get("signal_index")) is not int
            or value["signal_index"] != company_judgments.FIT_CHECKPOINT_INDEX):
            raise ValueError("invalid verified fit checkpoint request")
        from lab_arena import contracts
        encoded = contracts.canonical_json(value).encode("utf-8")
        if len(encoded) > MAX_BYTES:
            raise ValueError("signal cache request too large")
        return json.loads(encoded)
    expected = REQUEST_FIELDS | ({"reference", "judgment"} if value.get("operation") == "complete" else set())
    if "verified_source_pages" in value:
        expected.add("verified_source_pages")
        if not isinstance(value["verified_source_pages"], Mapping):
            raise ValueError("invalid retained source context")
    if set(value) != expected or value.get("operation") not in {"claim", "complete"}:
        raise ValueError("invalid signal cache operation")
    if any(type(value.get(key)) is not int or not 0 <= value[key] < 100 for key in ("company_index", "signal_index")):
        raise ValueError("invalid signal cache position")
    if not isinstance(value.get("verified_company_identity"), Mapping) or not isinstance(value.get("verified_funding_stage_context"), list):
        raise ValueError("invalid signal cache context")
    from lab_arena import contracts
    encoded = contracts.canonical_json(value).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise ValueError("signal cache request too large")
    return json.loads(encoded)


def validate_evidence_request(value: Any) -> dict:
    from lab_arena import contracts, evidence_judgments as evidence
    if not isinstance(value, Mapping):
        raise ValueError("invalid evidence request")
    fit = value.get("checkpoint_kind") == evidence.FIT_KIND
    expected = {"operation", "company_index", "signal_index", "checkpoint_kind" if fit else "judgment_kind"}
    operation = value.get("operation")
    if "evidence_judgment_policy" in value:
        expected.add("evidence_judgment_policy")
        if value["evidence_judgment_policy"] != evidence.POLICY:
            raise ValueError("invalid evidence policy")
    if "claim" in value:
        expected.add("claim")
        if fit or not isinstance(value["claim"], Mapping):
            raise ValueError("invalid current claim")
    if not fit:
        expected |= {"verified_company_identity", "verified_funding_stage_context"}
        if "verified_source_pages" in value:
            expected.add("verified_source_pages")
    if operation != "candidates" or not fit or "evidence_input" in value:
        expected |= {"evidence_input", "evidence_hash"}
    if operation == "complete":
        expected |= {"reference", "judgment"}
    if "candidate_reference" in value:
        expected.add("candidate_reference")
        if "equivalence_receipt" in value:
            expected.add("equivalence_receipt")
    if (set(value) != expected or operation not in {"candidates", "claim", "complete"}
        or type(value.get("company_index")) is not int or not 0 <= value["company_index"] < 5
        or type(value.get("signal_index")) is not int
        or (value["signal_index"] != -1 if fit else not 0 <= value["signal_index"] < 100)
        or ("candidate_reference" in value and operation != "claim")):
        raise ValueError("invalid evidence cache request")
    encoded = contracts.canonical_json(value).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise ValueError("evidence cache request too large")
    return json.loads(encoded)


def _exchange_once(request: Mapping[str, Any]) -> dict:
    from lab_arena import contracts, shim
    from qualification.scoring.signal_judgment_cache import SignalJudgmentCacheError
    if not shim.trusted_scorer_mode():
        raise SignalJudgmentCacheError("signal cache requires scorer mode")
    encoded = contracts.canonical_json({"schema_version": SCHEMA, "request": validate_request(request)}).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise SignalJudgmentCacheError("signal cache frame too large")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(35.0)
        connection.connect(shim.worker_socket_path())
        connection.sendall(len(encoded).to_bytes(4, "big") + encoded)
        size = int.from_bytes(shim._recv_exact(connection, 4), "big")
        if not 2 <= size <= MAX_BYTES:
            raise SignalJudgmentCacheError("signal cache response too large")
        result = json.loads(shim._recv_exact(connection, size))
    if not isinstance(result, dict):
        raise SignalJudgmentCacheError("invalid signal cache response")
    return result


async def exchange(request: Mapping[str, Any]) -> dict:
    """Retry the same control request only; never replay a provider operation."""
    from qualification.scoring.signal_judgment_cache import SignalJudgmentCacheError
    deadline = time.monotonic() + 120.0
    while True:
        try:
            result = await asyncio.to_thread(_exchange_once, request)
        except (OSError, ValueError) as exc:
            raise SignalJudgmentCacheError("signal cache transport unavailable") from exc
        if request.get("operation") == FIT_CAPABILITY_OPERATION:
            if result == {"status": "unavailable"} or result == fit_checkpoint_capability():
                return result
            raise SignalJudgmentCacheError("verified fit checkpoint capability is invalid")
        if result.get("status") not in {"busy", "pending"}:
            if result.get("status") not in {"hit", "miss", "stored", "existing", "released", "candidates"}:
                raise SignalJudgmentCacheError("signal cache request unavailable")
            return result
        if time.monotonic() >= deadline:
            raise SignalJudgmentCacheError("signal cache wait incomplete")
        await asyncio.sleep(2.0)
