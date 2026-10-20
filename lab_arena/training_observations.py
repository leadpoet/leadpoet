"""Bounded worker observations for outcomes that paid receipts cannot prove.

This is worker-side evidence, never a provider receipt or cost authority.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from lab_arena import contracts, lab_arena_checkpoint, operations, training_capture, training_input

LEGACY_SCHEMA_VERSION = "leadpoet.lab_arena.training_observations.v1"
SCHEMA_VERSION = "leadpoet.lab_arena.training_observations.v2"
# One run already admits up to 192 MiB of explicit agent control frames.
# Retain that evidence plus 64 MiB for host receipts/catalogs and JSON overhead.
# Single events keep their original bound; this is only an aggregate increase.
MAX_EVENT_BYTES = 8 * 1024 * 1024
MAX_OBJECT_BYTES = 256 * 1024 * 1024
# Deflate framing for incompressible 256 MiB can exceed the old 64 KiB reserve.
MAX_COMPRESSED_BYTES = MAX_OBJECT_BYTES + 1024 * 1024
MAX_OBSERVATIONS = 16_384
MAX_AGENT_OBSERVATIONS = 4096
MAX_WEB_BODY_BYTES = 6 * 1024 * 1024  # A 2 MiB decoded fetch can expand through U+FFFD.
_V1_FIELDS = frozenset({"observations", "observation_count", "local_refusal_count", "complete", "failures"})
_V2_FIELDS = _V1_FIELDS | {"response_receipts", "response_receipt_count", "agent_observations", "agent_observation_count"}
_AGENT_REF_RE = re.compile(r"agent:(0|[1-9][0-9]*)\Z")
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
    if len(_canonical(value)) > MAX_EVENT_BYTES - 65_536:
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


def _timestamp(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > 40:
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def _url(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > 2000:
        return False
    try:
        parts = urlsplit(value)
        return parts.scheme in {"http", "https"} and bool(parts.hostname)
    except ValueError:
        return False


def clean_response_receipt(value: Mapping[str, Any]) -> dict:
    """A socket receipt proves a write, not application consumption or billing."""
    required = {"ordinal", "action_sequence", "operation_id", "call_identity", "protocol",
                "status", "body_sha256", "body_bytes", "write_state", "socket_written_at"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ObservationError("response_receipt_invalid")
    if (type(value["ordinal"]) is not int or value["ordinal"] < 0
            or type(value["action_sequence"]) is not int or value["action_sequence"] < 0
            or not isinstance(value["operation_id"], str)
            or operations.get_operation(value["operation_id"]) is None
            or not isinstance(value["call_identity"], str)
            or not contracts.SHA256_RE.fullmatch(value["call_identity"])
            or value["protocol"] not in {"http", "frame"}
            or type(value["status"]) is not int or not 100 <= value["status"] <= 599
            or not isinstance(value["body_sha256"], str)
            or not contracts.SHA256_RE.fullmatch(value["body_sha256"])
            or type(value["body_bytes"]) is not int or value["body_bytes"] < 0
            or value["write_state"] not in {"prepared", "socket_written", "socket_write_failed"}
            or (value["write_state"] == "socket_written") != _timestamp(value["socket_written_at"])
            or value["write_state"] != "socket_written" and value["socket_written_at"] is not None):
        raise ObservationError("response_receipt_invalid")
    return dict(value)


def clean_agent_observation(value: Mapping[str, Any], secrets: Sequence[str] = ()) -> dict:
    """Keep only explicit, bounded agent reports; never infer hidden state."""
    required = {"id", "kind", "observed_at", "after_action_sequence", "depends_on", "data", "privacy"}
    if not isinstance(value, Mapping) or set(value) != required or len(_canonical(value)) > MAX_EVENT_BYTES - 65_536:
        raise ObservationError("agent_observation_invalid")
    if (type(value["id"]) is not int or value["id"] < 0
            or value["kind"] not in {"web_fetch", "response_received", "state_snapshot", "capture_failure"}
            or not _timestamp(value["observed_at"])
            or value["after_action_sequence"] is not None and
            (type(value["after_action_sequence"]) is not int or value["after_action_sequence"] < 0)
            or not isinstance(value["depends_on"], list) or len(value["depends_on"]) > 32
            or any(not isinstance(item, str) or len(item) > 96 or not
                   (_AGENT_REF_RE.fullmatch(item) or contracts.SHA256_RE.fullmatch(item))
                   for item in value["depends_on"])
            or len(set(value["depends_on"])) != len(value["depends_on"])
            or not isinstance(value["data"], dict)):
        raise ObservationError("agent_observation_invalid")
    data = value["data"]
    if value["kind"] == "web_fetch":
        required_web = {"url", "final_url", "retrieved_at", "status", "content_type",
                        "body_sha256", "body_bytes", "stored_body_sha256", "stored_body_bytes",
                        "body", "body_ref"}
        if (set(data) not in (required_web, required_web | {"capture"})
                or not _url(data["url"]) or not _url(data["final_url"])
                or not _timestamp(data["retrieved_at"])
                or type(data["status"]) is not int or not 100 <= data["status"] <= 599
                or not isinstance(data["content_type"], str) or len(data["content_type"]) > 256
                or not isinstance(data["body_sha256"], str) or not contracts.SHA256_RE.fullmatch(data["body_sha256"])
                or type(data["body_bytes"]) is not int or not 0 <= data["body_bytes"] <= MAX_WEB_BODY_BYTES
                or data["stored_body_sha256"] is not None and
                (not isinstance(data["stored_body_sha256"], str) or not contracts.SHA256_RE.fullmatch(data["stored_body_sha256"]))
                or data["stored_body_bytes"] is not None and
                (type(data["stored_body_bytes"]) is not int or data["stored_body_bytes"] < 0)
                or (data["body"] is None) == (data["body_ref"] is None)
                or data["body"] is not None and not isinstance(data["body"], str)
                or data["body_ref"] is not None and (type(data["body_ref"]) is not int or data["body_ref"] < 0)):
            raise ObservationError("web_fetch_invalid")
        if "capture" in data:
            try:
                lab_arena_checkpoint.validate_web_capture_metadata(data["capture"])
            except (ValueError, LookupError):
                raise ObservationError("web_capture_invalid") from None
    elif value["kind"] == "response_received":
        if (set(data) != {"call_identity", "status", "body_sha256", "body_bytes"}
                or not isinstance(data["call_identity"], str) or not contracts.SHA256_RE.fullmatch(data["call_identity"])
                or type(data["status"]) is not int or not 100 <= data["status"] <= 599
                or not isinstance(data["body_sha256"], str) or not contracts.SHA256_RE.fullmatch(data["body_sha256"])
                or type(data["body_bytes"]) is not int or data["body_bytes"] < 0):
            raise ObservationError("response_received_invalid")
    elif value["kind"] == "state_snapshot" and (set(data) != {"name", "value"} or not isinstance(data["name"], str)
          or not 1 <= len(data["name"]) <= 80 or len(_canonical(data["value"])) > 65_536):
        raise ObservationError("state_snapshot_invalid")
    elif value["kind"] == "capture_failure" and (set(data) != {"category", "reason"}
          or data["category"] not in {"web_fetch", "response_received", "state_snapshot"}
          or data["reason"] not in {"size_limit", "invalid_value"}):
        raise ObservationError("agent_failure_invalid")
    cleaned, redactions = training_capture.sanitize_training_value(dict(data), secrets)
    privacy = value["privacy"]
    if (not isinstance(privacy, dict) or set(privacy) != {"redacted", "redactions"}
            or type(privacy["redacted"]) is not bool or not isinstance(privacy["redactions"], list)
            or any(item not in {"credential", "hidden_reasoning"} for item in privacy["redactions"])
            or privacy["redacted"] != bool(privacy["redactions"])):
        raise ObservationError("agent_privacy_invalid")
    redactions = sorted(set(redactions) | set(privacy["redactions"]))
    if value["kind"] == "web_fetch" and cleaned["body"] is not None:
        if "capture" in data and not privacy["redacted"]:
            try:
                lab_arena_checkpoint.validate_web_capture_metadata(data["capture"], data["body"])
            except (ValueError, LookupError, UnicodeError):
                raise ObservationError("web_capture_bytes_invalid") from None
        stored = cleaned["body"].encode("utf-8")
        if len(stored) > MAX_WEB_BODY_BYTES:
            raise ObservationError("web_fetch_size_limit")
        # The digest describes the original agent-returned text; a privacy
        # change is separately marked non-exact by the artifact.
        if not redactions and (len(stored) != data["body_bytes"]
                               or "sha256:" + hashlib.sha256(stored).hexdigest() != data["body_sha256"]):
            raise ObservationError("web_fetch_hash_invalid")
        stored_hash = "sha256:" + hashlib.sha256(stored).hexdigest()
        if (data["stored_body_sha256"] not in {None, stored_hash}
                or data["stored_body_bytes"] not in {None, len(stored)}):
            raise ObservationError("stored_web_fetch_hash_invalid")
        cleaned["stored_body_sha256"] = stored_hash
        cleaned["stored_body_bytes"] = len(stored)
    return {**value, "data": cleaned,
            "privacy": {"redacted": bool(redactions), "redactions": redactions}}


@dataclass
class Buffer:
    observations: list[dict] = field(default_factory=list)
    response_receipts: list[dict] = field(default_factory=list)
    response_receipt_count: int = 0
    agent_observations: list[dict] = field(default_factory=list)
    agent_observation_count: int = 0
    web_body_by_hash: dict[tuple[str, str], int] = field(default_factory=dict)
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

    def append_response_receipt(self, value: Mapping[str, Any]) -> bool:
        self.response_receipt_count += 1
        try:
            cleaned = clean_response_receipt(value)
            size = len(_canonical(cleaned)) + 1
            if (len(self.response_receipts) >= MAX_OBSERVATIONS
                    or self.size_bytes + size > MAX_OBJECT_BYTES - 65_536):
                raise ObservationError("size_limit")
            self.response_receipts.append(cleaned)
            self.size_bytes += size
            return True
        except Exception:
            self.failures.add("capture_incomplete")
            return False

    def mark_socket_write(self, ordinal: int, *, written_at: str | None) -> None:
        for receipt in reversed(self.response_receipts):
            if receipt["ordinal"] == ordinal:
                receipt["write_state"] = "socket_written" if written_at is not None else "socket_write_failed"
                receipt["socket_written_at"] = written_at
                clean_response_receipt(receipt)
                if written_at is None:
                    self.failures.add("capture_incomplete")
                return
        self.failures.add("capture_incomplete")

    def append_agent_observation(self, value: Mapping[str, Any], secrets: Sequence[str] = ()) -> int | None:
        self.agent_observation_count += 1
        try:
            cleaned = clean_agent_observation(value, secrets)
            if cleaned["kind"] == "response_received":
                # A runtime observer and an existing agent hook can acknowledge
                # the same read. Keep one exact receipt and its stable ID.
                for previous in reversed(self.agent_observations):
                    if previous["kind"] == "response_received" and previous["data"] == cleaned["data"]:
                        self.agent_observation_count -= 1
                        return previous["id"]
            web_key = None
            if cleaned["kind"] == "web_fetch" and cleaned["data"]["body"] is not None:
                web_key = (cleaned["data"]["body_sha256"], cleaned["data"]["stored_body_sha256"])
                previous = self.web_body_by_hash.get(web_key)
                if previous is not None:
                    cleaned["data"]["body"] = None
                    cleaned["data"]["body_ref"] = previous
            size = len(_canonical(cleaned)) + 1
            if (len(self.agent_observations) >= MAX_AGENT_OBSERVATIONS
                    or self.size_bytes + size > MAX_OBJECT_BYTES - 65_536):
                raise ObservationError("size_limit")
            self.agent_observations.append(cleaned)
            self.size_bytes += size
            if web_key is not None and cleaned["data"]["body"] is not None:
                self.web_body_by_hash[web_key] = cleaned["id"]
            if (cleaned["kind"] == "capture_failure"
                    or cleaned["kind"] == "web_fetch"
                    and cleaned["data"].get("capture", {}).get("complete") is False
                    or cleaned["kind"] == "state_snapshot"
                    and cleaned["data"]["name"] == "automatic_capture_summary"
                    and isinstance(cleaned["data"]["value"], dict)
                    and cleaned["data"]["value"].get("complete") is False):
                self.failures.add("capture_incomplete")
            return cleaned["id"]
        except Exception:
            self.failures.add("capture_incomplete")
            return None

    def document(self) -> dict:
        failures = set(self.failures)
        if any(row["write_state"] != "socket_written" for row in self.response_receipts):
            failures.add("capture_incomplete")
        return {"observations": sorted(self.observations, key=lambda row: row["ordinal"]),
                "observation_count": self.attempted, "local_refusal_count": self.local_refusals,
                "response_receipts": sorted(self.response_receipts, key=lambda row: row["ordinal"]),
                "response_receipt_count": self.response_receipt_count,
                "agent_observations": list(self.agent_observations),
                "agent_observation_count": self.agent_observation_count,
                "complete": not failures, "failures": sorted(failures)}


def _validate_payload(document: Mapping[str, Any], secrets: Sequence[str] = ()) -> dict:
    if not isinstance(document, Mapping) or frozenset(document) not in {_V1_FIELDS, _V2_FIELDS}:
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
    if set(document) == _V1_FIELDS:
        return {**document, "observations": cleaned}
    receipts = document["response_receipts"]
    agent_rows = document["agent_observations"]
    if (not isinstance(receipts, list) or len(receipts) > MAX_OBSERVATIONS
            or not isinstance(agent_rows, list) or len(agent_rows) > MAX_AGENT_OBSERVATIONS
            or type(document["response_receipt_count"]) is not int or document["response_receipt_count"] < len(receipts)
            or type(document["agent_observation_count"]) is not int or document["agent_observation_count"] < len(agent_rows)):
        raise ObservationError("extra_counts_invalid")
    cleaned_receipts = [clean_response_receipt(row) for row in receipts]
    receipt_ordinals = [row["ordinal"] for row in cleaned_receipts]
    if receipt_ordinals != sorted(set(receipt_ordinals)) or set(receipt_ordinals) & set(ordinals):
        raise ObservationError("response_receipt_order_invalid")
    cleaned_agent = [clean_agent_observation(row, secrets) for row in agent_rows]
    ids = [row["id"] for row in cleaned_agent]
    if ids != list(range(len(ids))):
        raise ObservationError("agent_observation_order_invalid")
    bodies = {}
    for row in cleaned_agent:
        if any(_AGENT_REF_RE.fullmatch(reference) and int(reference[6:]) >= row["id"]
               for reference in row["depends_on"]):
            raise ObservationError("agent_dependency_invalid")
        if row["kind"] == "web_fetch":
            data = row["data"]
            if data["body_ref"] is not None:
                previous = bodies.get(data["body_ref"])
                if (previous is None or previous["body_sha256"] != data["body_sha256"]
                        or previous["body_bytes"] != data["body_bytes"]
                        or previous["stored_body_sha256"] != data["stored_body_sha256"]
                        or previous["stored_body_bytes"] != data["stored_body_bytes"]):
                    raise ObservationError("web_body_ref_invalid")
            else:
                bodies[row["id"]] = data
    if document["complete"] and (document["response_receipt_count"] != len(receipts)
            or document["agent_observation_count"] != len(agent_rows)
            or any(row["write_state"] != "socket_written" for row in receipts)
            or any(row["kind"] == "web_fetch" and row["data"].get("capture", {}).get("complete") is False
                   or row["kind"] == "state_snapshot"
                   and row["data"]["name"] == "automatic_capture_summary"
                   and isinstance(row["data"]["value"], dict)
                   and row["data"]["value"].get("complete") is False
                   for row in cleaned_agent)):
        raise ObservationError("completeness_invalid")
    return {**document, "observations": cleaned, "response_receipts": cleaned_receipts,
            "agent_observations": cleaned_agent}


def build_snapshot(run_id: str, kind: str, document: Mapping[str, Any], *, lease_generation: int,
                   claim_request_hash: str, secrets: Sequence[str] = ()) -> tuple[dict, bytes]:
    object_ref(run_id, lease_generation, claim_request_hash)
    if kind not in {"execute", "score"} or len(_canonical(document)) > MAX_OBJECT_BYTES:
        raise ObservationError("snapshot_invalid")
    payload = _validate_payload(document, secrets)
    schema_version = SCHEMA_VERSION if set(payload) == _V2_FIELDS else LEGACY_SCHEMA_VERSION
    snapshot = {"schema_version": schema_version, "run_id": run_id, "kind": kind,
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
                or snapshot["schema_version"] not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION} or snapshot["run_id"] != run_id
                or kind not in {"execute", "score"} or snapshot["kind"] != kind
                or type(snapshot["lease_generation"]) is not int or snapshot["lease_generation"] != lease_generation
                or snapshot["claim_request_hash"] != claim_request_hash
                or contracts.document_hash(snapshot["payload"]) != snapshot["payload_sha256"]):
            raise ObservationError("identity_or_hash_invalid")
        cleaned = _validate_payload(snapshot["payload"])
        if (snapshot["schema_version"] == SCHEMA_VERSION) != (set(cleaned) == _V2_FIELDS):
            raise ObservationError("schema_payload_mismatch")
        if cleaned != snapshot["payload"]:
            raise ObservationError("privacy_invalid")
        return snapshot
    except ObservationError:
        raise
    except Exception:
        raise ObservationError("observations_unavailable") from None
