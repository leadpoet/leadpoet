"""Bounded worker observations for outcomes that paid receipts cannot prove.

This is worker-side evidence, never a provider receipt or cost authority.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from lab_arena import contracts, operations, training_capture, training_input

SCHEMA_VERSION = "leadpoet.lab_arena.training_observations.v1"
MAX_OBJECT_BYTES = 8 * 1024 * 1024
MAX_COMPRESSED_BYTES = MAX_OBJECT_BYTES + 64 * 1024
MAX_OBSERVATIONS = 16_384
KINDS = frozenset({"local_catalog", "local_refusal", "transport_failure", "invalid_response", "nonstandard_response", "local_parse_error"})


class ObservationError(ValueError):
    """Worker evidence is incomplete or has an invalid identity."""


def object_ref(run_id: str, generation: int, claim_hash: str) -> str:
    training_input.object_ref(run_id, generation)
    training_input._validate_claim_hash(claim_hash)
    return "private/arena-training/runs/%s/leases/%d/observations-%s.json.gz" % (run_id, generation, claim_hash[7:])


def _canonical(value: Any) -> bytes:
    return contracts.canonical_json(value).encode("utf-8")


def clean_observation(value: Mapping[str, Any], secrets: Sequence[str] = ()) -> dict:
    """Decode response bytes before privacy filtering; never truncate evidence."""
    if len(_canonical(value)) > MAX_OBJECT_BYTES - 65_536:
        raise ObservationError("size_limit")
    if set(value) != {"ordinal", "action_sequence", "operation_id", "kind", "request", "outcome", "call_identity", "privacy", "contributes_to_provider_call_count"}:
        raise ObservationError("observation_fields_invalid")
    if type(value["ordinal"]) is not int or value["ordinal"] < 0:
        raise ObservationError("ordinal_invalid")
    if value["action_sequence"] is not None and (type(value["action_sequence"]) is not int or value["action_sequence"] < 0):
        raise ObservationError("action_invalid")
    if value["kind"] not in KINDS or not isinstance(value["operation_id"], str):
        raise ObservationError("kind_invalid")
    if value["operation_id"] != "unknown" and operations.get_operation(value["operation_id"]) is None:
        raise ObservationError("operation_invalid")
    if type(value["contributes_to_provider_call_count"]) is not bool:
        raise ObservationError("contribution_invalid")
    identity = value["call_identity"]
    if identity is not None and (not isinstance(identity, str) or not contracts.SHA256_RE.fullmatch(identity)):
        raise ObservationError("call_identity_invalid")
    if value["kind"] == "local_catalog" and (value["operation_id"] not in operations.CATALOG_OPERATIONS or identity is not None or value["contributes_to_provider_call_count"] is not True or value["action_sequence"] is None):
        raise ObservationError("catalog_identity_invalid")
    if value["kind"] == "local_refusal" and (identity is not None or value["contributes_to_provider_call_count"] is not False or value["action_sequence"] is None):
        raise ObservationError("refusal_identity_invalid")
    if value["kind"] == "local_parse_error" and (identity is not None or value["contributes_to_provider_call_count"] is not False or value["action_sequence"] is not None):
        raise ObservationError("parse_identity_invalid")
    request = value["request"]
    if not isinstance(request, dict) or set(request) != {"admitted_parameters", "timeout_ms"} or not isinstance(request["admitted_parameters"], dict) or type(request["timeout_ms"]) is not int:
        raise ObservationError("request_invalid")
    request, redactions = training_capture.sanitize_training_value(request, secrets)
    outcome = value["outcome"]
    if not isinstance(outcome, dict) or not set(outcome) <= {"protocol", "response", "exception_class", "error_code"} or outcome.get("protocol") not in {"http", "frame"}:
        raise ObservationError("outcome_invalid")
    outcome = dict(outcome)
    response = outcome.pop("response", None)
    outcome, changes = training_capture.sanitize_training_value(outcome, secrets)
    redactions = set(redactions) | set(changes)
    if response is not None:
        if not isinstance(response, dict):
            raise ObservationError("response_invalid")
        if set(response) == {"error"} and outcome["protocol"] == "frame":
            response, changes = training_capture.sanitize_training_value(response, secrets)
        elif set(response) == {"status", "headers", "body_b64"}:
            if type(response["status"]) is not int or not isinstance(response["headers"], dict) or not isinstance(response["body_b64"], str):
                raise ObservationError("response_invalid")
            raw = base64.b64decode(response["body_b64"], validate=True)
            text = raw.decode("utf-8")
            try:
                body = json.loads(text)
            except ValueError:
                body = text
            cleaned, body_changes = training_capture.sanitize_training_value(body, secrets)
            # Typed provider reasoning summaries/text are hidden content too.
            def hide_typed(value):
                changed = False
                if isinstance(value, dict):
                    for key, child in value.items():
                        if value.get("type") == "reasoning" and key.lower() in {"content", "summary", "text"}:
                            value[key] = "[REDACTED]"
                            changed = True
                        else:
                            changed = hide_typed(child) or changed
                elif isinstance(value, list):
                    for child in value:
                        changed = hide_typed(child) or changed
                return changed
            if hide_typed(cleaned):
                body_changes = sorted(set(body_changes) | {"hidden_reasoning"})
            headers, changes = training_capture.sanitize_training_value(response["headers"], secrets)
            changes = set(changes) | set(body_changes)
            # Preserve literal response bytes when privacy did not change them.
            if body_changes:
                raw = (_canonical(cleaned) if not isinstance(body, str) else cleaned.encode("utf-8"))
            response = {"status": response["status"], "headers": headers,
                        "body_b64": base64.b64encode(raw).decode("ascii")}
        else:
            raise ObservationError("response_invalid")
        outcome["response"] = response
        redactions.update(changes)
    elif not isinstance(outcome.get("exception_class"), str):
        raise ObservationError("outcome_missing")
    privacy = value["privacy"]
    if not isinstance(privacy, dict) or set(privacy) != {"redacted", "redactions"} or type(privacy["redacted"]) is not bool or not isinstance(privacy["redactions"], list) or any(x not in {"credential", "hidden_reasoning"} for x in privacy["redactions"]) or privacy["redacted"] != bool(privacy["redactions"]):
        raise ObservationError("privacy_invalid")
    redactions.update(privacy["redactions"])
    return {**value, "request": request, "outcome": outcome,
            "privacy": {"redacted": bool(redactions), "redactions": sorted(redactions)}}


@dataclass
class Buffer:
    observations: list[dict] = field(default_factory=list)
    size_bytes: int = 0
    attempted: int = 0
    local_refusals: int = 0
    failures: set[str] = field(default_factory=set)

    def append(self, value: Mapping[str, Any], secrets: Sequence[str] = ()) -> None:
        """Caller holds the run lock; every capture failure stays observational."""
        self.attempted += 1
        self.local_refusals += value.get("kind") == "local_refusal"
        try:
            cleaned = clean_observation(value, secrets)
            size = len(_canonical(cleaned)) + 1
            if len(self.observations) >= MAX_OBSERVATIONS or self.size_bytes + size > MAX_OBJECT_BYTES - 65_536:
                raise ObservationError("size_limit")
            self.observations.append(cleaned)
            self.size_bytes += size
        except Exception:
            self.failures.add("capture_incomplete")

    def document(self) -> dict:
        return {"observations": sorted(self.observations, key=lambda row: row["ordinal"]),
                "observation_count": self.attempted, "local_refusal_count": self.local_refusals,
                "complete": not self.failures, "failures": sorted(self.failures)}


def _validate_payload(document: Mapping[str, Any], secrets: Sequence[str] = ()) -> dict:
    if not isinstance(document, Mapping) or set(document) != {"observations", "observation_count", "local_refusal_count", "complete", "failures"}:
        raise ObservationError("payload_invalid")
    rows = document["observations"]
    if not isinstance(rows, list) or len(rows) > MAX_OBSERVATIONS:
        raise ObservationError("count_invalid")
    if any(type(document[key]) is not int or document[key] < 0 for key in ("observation_count", "local_refusal_count")) or type(document["complete"]) is not bool or not isinstance(document["failures"], list) or any(x != "capture_incomplete" for x in document["failures"]):
        raise ObservationError("counts_invalid")
    cleaned = [clean_observation(row, secrets) for row in rows]
    ordinals = [row["ordinal"] for row in cleaned]
    actions = [row["action_sequence"] for row in cleaned if row["action_sequence"] is not None]
    if ordinals != sorted(set(ordinals)) or len(set(actions)) != len(actions):
        raise ObservationError("duplicate_or_unordered_observation")
    if document["observation_count"] < len(rows) or document["local_refusal_count"] < sum(row["kind"] == "local_refusal" for row in rows):
        raise ObservationError("counts_invalid")
    if document["complete"] and (document["failures"] or document["observation_count"] != len(rows) or document["local_refusal_count"] != sum(row["kind"] == "local_refusal" for row in rows)):
        raise ObservationError("completeness_invalid")
    return {**document, "observations": cleaned}


def build_snapshot(run_id: str, kind: str, document: Mapping[str, Any], *, lease_generation: int,
                   claim_request_hash: str, secrets: Sequence[str] = ()) -> tuple[dict, bytes]:
    object_ref(run_id, lease_generation, claim_request_hash)
    if kind not in {"execute", "score"} or len(_canonical(document)) > MAX_OBJECT_BYTES:
        raise ObservationError("snapshot_invalid")
    payload = _validate_payload(document, secrets)
    snapshot = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "kind": kind,
                "lease_generation": lease_generation, "claim_request_hash": claim_request_hash,
                "payload": payload, "payload_sha256": contracts.document_hash(payload)}
    raw = _canonical(snapshot)
    if len(raw) > MAX_OBJECT_BYTES:
        raise ObservationError("size_limit")
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise ObservationError("size_limit")
    return snapshot, compressed


def read_snapshot(objects: Any, run_id: str, kind: str, lease_generation: int, claim_request_hash: str) -> dict:
    try:
        compressed = objects.get_bounded(object_ref(run_id, lease_generation, claim_request_hash), MAX_COMPRESSED_BYTES)
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw = stream.read(MAX_OBJECT_BYTES + 1)
        if len(raw) > MAX_OBJECT_BYTES:
            raise ObservationError("size_limit")
        snapshot = json.loads(raw)
        if (not isinstance(snapshot, dict) or set(snapshot) != {"schema_version", "run_id", "kind", "lease_generation", "claim_request_hash", "payload", "payload_sha256"}
                or snapshot["schema_version"] != SCHEMA_VERSION or snapshot["run_id"] != run_id
                or kind not in {"execute", "score"} or snapshot["kind"] != kind
                or type(snapshot["lease_generation"]) is not int or snapshot["lease_generation"] != lease_generation
                or snapshot["claim_request_hash"] != claim_request_hash
                or contracts.document_hash(snapshot["payload"]) != snapshot["payload_sha256"]):
            raise ObservationError("identity_or_hash_invalid")
        cleaned = _validate_payload(snapshot["payload"])
        if cleaned != snapshot["payload"]:
            raise ObservationError("privacy_invalid")
        return snapshot
    except ObservationError:
        raise
    except Exception:
        raise ObservationError("observations_unavailable") from None
