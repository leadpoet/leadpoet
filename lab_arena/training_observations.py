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
JOURNAL_SCHEMA_VERSION = "leadpoet.lab_arena.training_observation_journal.v1"
MAX_SEGMENT_BYTES = 16 * 1024 * 1024
MAX_SEGMENTS = 1024
MAX_JOURNAL_ATTEMPTS = 10_000_000
OPTIONAL_CAPTURE_INCOMPLETE = "optional_capture_incomplete"
_FAILURES = frozenset({"capture_incomplete", OPTIONAL_CAPTURE_INCOMPLETE})
_V1_FIELDS = frozenset({"observations", "observation_count", "local_refusal_count", "complete", "failures"})
_V2_FIELDS = _V1_FIELDS | {"response_receipts", "response_receipt_count", "agent_observations", "agent_observation_count"}
_AGENT_REF_RE = re.compile(r"agent:(0|[1-9][0-9]*)\Z")
KINDS = frozenset({"local_catalog", "local_refusal", "transport_failure", "invalid_response", "nonstandard_response", "local_parse_error"})
BROKER_NOT_DISPATCHED_REASONS = frozenset({
    "provider_admission_deadline", "provider_admission_cancelled",
    "judgment_cache_cancelled", "judgment_cache_busy",
    "provider_cost_uncertain", "provider_calls_inflight", "budget_busy",
})
_BROKER_PROVIDER_UNAVAILABLE_BODY = b'{"error":{"code":"provider_unavailable"}}'


class ObservationError(ValueError):
    """Worker evidence is incomplete or has an invalid identity."""


def object_ref(run_id: str, generation: int, claim_hash: str) -> str:
    training_input.object_ref(run_id, generation)
    training_input._validate_claim_hash(claim_hash)
    return "private/arena-training/runs/%s/leases/%d/observations-%s.json.gz" % (run_id, generation, claim_hash[7:])


def segment_ref(run_id: str, generation: int, claim_hash: str, number: int) -> str:
    object_ref(run_id, generation, claim_hash)
    if type(number) is not int or not 0 <= number < MAX_SEGMENTS:
        raise ObservationError("segment_number_invalid")
    return "private/arena-training/runs/%s/leases/%d/observations-%s/segment-%04d.json.gz" % (
        run_id, generation, claim_hash[7:], number)


def _canonical(value: Any) -> bytes:
    return contracts.canonical_json(value).encode("utf-8")


def _expected_not_dispatched_response(response: Any) -> bool:
    if (not isinstance(response, Mapping) or response.get("status") != 502
            or type(response.get("status")) is not int
            or not isinstance(response.get("body_b64"), str)):
        return False
    try:
        return base64.b64decode(response["body_b64"], validate=True) == _BROKER_PROVIDER_UNAVAILABLE_BODY
    except (ValueError, TypeError):
        return False


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
    if not isinstance(outcome, dict) or not set(outcome) <= {"protocol", "response", "exception_class", "error_code", "broker_disposition"} or outcome.get("protocol") not in {"http", "frame"}:
        raise ObservationError("outcome_invalid")
    if "broker_disposition" in outcome:
        disposition = outcome["broker_disposition"]
        if (value["kind"] != "nonstandard_response" or identity is None
                or value["action_sequence"] is None
                or not isinstance(disposition, dict)
                or set(disposition) != {"outcome", "reason"}
                or disposition.get("outcome") != "not_dispatched"
                or not isinstance(disposition.get("reason"), str)
                or disposition.get("reason") not in BROKER_NOT_DISPATCHED_REASONS
                or not _expected_not_dispatched_response(outcome.get("response"))):
            raise ObservationError("broker_disposition_invalid")
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
            or value["kind"] not in {"web_fetch", "web_fetch_binary", "http_response_metadata", "response_received", "state_snapshot", "capture_failure"}
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
    elif value["kind"] == "web_fetch_binary":
        required_binary = {"url", "final_url", "retrieved_at", "status", "content_type",
                           "body_sha256", "body_bytes", "stored_body_sha256", "stored_body_bytes",
                           "body_b64", "body_ref", "capture", "privacy_review"}
        if (set(data) != required_binary
                or not _url(data["url"]) or not _url(data["final_url"])
                or not _timestamp(data["retrieved_at"])
                or type(data["status"]) is not int or not 100 <= data["status"] <= 599
                or not isinstance(data["content_type"], str) or len(data["content_type"]) > 256
                or not isinstance(data["body_sha256"], str) or not contracts.SHA256_RE.fullmatch(data["body_sha256"])
                or type(data["body_bytes"]) is not int or not 0 <= data["body_bytes"] <= lab_arena_checkpoint.MAX_TRAINING_BINARY_BODY_BYTES
                or data["stored_body_sha256"] is not None and
                (not isinstance(data["stored_body_sha256"], str) or not contracts.SHA256_RE.fullmatch(data["stored_body_sha256"]))
                or data["stored_body_bytes"] is not None and
                (type(data["stored_body_bytes"]) is not int or data["stored_body_bytes"] < 0)
                or (data["body_b64"] is None) == (data["body_ref"] is None)
                or data["body_b64"] is not None and not isinstance(data["body_b64"], str)
                or data["body_ref"] is not None and (type(data["body_ref"]) is not int or data["body_ref"] < 0)
                or data["privacy_review"] != "unreviewed_binary"):
            raise ObservationError("binary_fetch_invalid")
        try:
            lab_arena_checkpoint.validate_binary_capture_metadata(data["capture"])
        except ValueError:
            raise ObservationError("binary_capture_invalid") from None
        if (data["capture"]["received_body_sha256"] != data["body_sha256"]
                or data["capture"]["received_body_bytes"] != data["body_bytes"]):
            raise ObservationError("binary_capture_bytes_invalid")
    elif value["kind"] == "http_response_metadata":
        try:
            lab_arena_checkpoint.validate_http_response_metadata(data)
        except ValueError:
            raise ObservationError("http_response_metadata_invalid") from None
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
          or data["category"] not in {"web_fetch", "web_fetch_binary", "http_response_metadata", "response_received", "state_snapshot"}
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
    if value["kind"] == "web_fetch_binary" and data["body_b64"] is not None:
        try:
            raw = base64.b64decode(data["body_b64"], validate=True)
        except (ValueError, TypeError):
            raise ObservationError("binary_body_invalid") from None
        if base64.b64encode(raw).decode("ascii") != data["body_b64"]:
            raise ObservationError("binary_body_invalid")
        if len(raw) > lab_arena_checkpoint.MAX_TRAINING_BINARY_BODY_BYTES:
            raise ObservationError("binary_fetch_size_limit")
        if not privacy["redacted"] and (len(raw) != data["body_bytes"]
                or "sha256:" + hashlib.sha256(raw).hexdigest() != data["body_sha256"]):
            raise ObservationError("binary_fetch_hash_invalid")
        if not privacy["redacted"]:
            try:
                lab_arena_checkpoint.validate_binary_capture_metadata(data["capture"], raw)
            except ValueError:
                raise ObservationError("binary_capture_bytes_invalid") from None
        # Latin-1 is a reversible byte mapping. The existing credential
        # sanitizer sees literal tokens before they can be hidden in base64.
        binary_text, binary_changes = training_capture.sanitize_training_value(raw.decode("latin-1"), secrets)
        stored = binary_text.encode("latin-1")
        if len(stored) > lab_arena_checkpoint.MAX_TRAINING_BINARY_BODY_BYTES:
            raise ObservationError("binary_fetch_size_limit")
        stored_hash = "sha256:" + hashlib.sha256(stored).hexdigest()
        if (data["stored_body_sha256"] not in {None, stored_hash}
                or data["stored_body_bytes"] not in {None, len(stored)}):
            raise ObservationError("stored_binary_fetch_hash_invalid")
        cleaned["body_b64"] = base64.b64encode(stored).decode("ascii")
        cleaned["stored_body_sha256"] = stored_hash
        cleaned["stored_body_bytes"] = len(stored)
        redactions = sorted(set(redactions) | set(binary_changes))
        if len(_canonical(cleaned)) > MAX_EVENT_BYTES - 65_536:
            raise ObservationError("binary_fetch_size_limit")
    return {**value, "data": cleaned,
            "privacy": {"redacted": bool(redactions), "redactions": redactions}}


def optional_capture_failure(row: Mapping[str, Any]) -> bool:
    """Classify only validated optional HTTP evidence, never an unknown fault."""
    kind, data = row["kind"], row["data"]
    if kind == "http_response_metadata":
        return True
    if kind in {"web_fetch", "web_fetch_binary"}:
        return data.get("capture", {}).get("complete") is False
    if kind != "state_snapshot" or data["name"] != "automatic_capture_summary":
        return False
    value = data["value"]
    if (not isinstance(value, dict) or value.get("source") != "runtime_http_v1"
            or value.get("complete") is not False or type(value.get("pending_queue")) is not int
            or value["pending_queue"] != 0 or type(value.get("inflight")) is not int
            or value["inflight"] != 0 or not isinstance(value.get("counts"), dict)):
        return False
    counts = value["counts"]
    normal = {"responses_started", "responses_complete", "responses_partial", "checkpoint_failure",
              "provider_ack_written", "response_metadata_written", "web_written"}
    optional = {"checkpoint_failure_" + category + "_" + reason
                for category in ("web_fetch", "web_fetch_binary", "http_response_metadata")
                for reason in ("timeout", "host_rejected", "transport_error", "invalid_response",
                               "invalid_frame", "socket_unavailable")}
    if any(not isinstance(key, str) or type(count) is not int or count < 0
           or key not in normal | optional and not key.startswith("client_")
           for key, count in counts.items()):
        return False
    failures = sum(counts.get(key, 0) for key in optional)
    return (counts.get("checkpoint_failure", 0) == failures
            and counts.get("responses_started", 0) == counts.get("responses_complete", 0) + counts.get("responses_partial", 0)
            and counts.get("responses_started", 0) == failures + sum(counts.get(key, 0) for key in (
                "provider_ack_written", "response_metadata_written", "web_written"))
            and (failures > 0 or counts.get("responses_partial", 0) > 0))


def optional_capture_only(document: Mapping[str, Any]) -> bool:
    """A new explicit failure tag cannot excuse missing required host evidence."""
    try:
        if (set(document) != _V2_FIELDS or document["complete"] is not False
                or document["failures"] != [OPTIONAL_CAPTURE_INCOMPLETE]
                or _validate_payload(document) != document
                or any(document[count] != len(document[rows]) for count, rows in (
                    ("observation_count", "observations"), ("response_receipt_count", "response_receipts"),
                    ("agent_observation_count", "agent_observations")))
                or document["local_refusal_count"] != sum(row["kind"] == "local_refusal" for row in document["observations"])
                or any(row["write_state"] != "socket_written" for row in document["response_receipts"])
                or any(row["privacy"]["redacted"] for row in document["observations"])
                or any(row["privacy"]["redacted"] for row in document["agent_observations"]
                       if row["kind"] == "response_received")):
            return False
        rows = [row for row in document["agent_observations"]
                if row["kind"] in {"capture_failure", "http_response_metadata"}
                or row["kind"] in {"web_fetch", "web_fetch_binary"} and row["data"].get("capture", {}).get("complete") is False
                or row["kind"] == "state_snapshot" and row["data"]["name"] == "automatic_capture_summary"
                and isinstance(row["data"]["value"], dict) and row["data"]["value"].get("complete") is False]
        return bool(rows) and all(optional_capture_failure(row) for row in rows)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


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

    @staticmethod
    def prepare_observation(value: Mapping[str, Any], secrets: Sequence[str] = ()) -> tuple[dict, int] | None:
        """Validate detached evidence before acquiring the shared run lock."""
        try:
            cleaned = clean_observation(value, secrets)
            return cleaned, len(_canonical(cleaned)) + 1
        except Exception:
            return None

    def append(self, value: Mapping[str, Any], secrets: Sequence[str] = ()) -> None:
        """Caller holds the run lock; every capture failure stays observational."""
        self.append_prepared(self.prepare_observation(value, secrets),
                             local_refusal=value.get("kind") == "local_refusal")

    def append_prepared(self, prepared: tuple[dict, int] | None, *, local_refusal: bool) -> None:
        """Commit against current counters and limits while the caller holds the lock."""
        self.attempted += 1
        self.local_refusals += local_refusal
        try:
            if prepared is None:
                raise ObservationError("observation_invalid")
            cleaned, size = prepared
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
            body_field = ("body" if cleaned["kind"] == "web_fetch" else "body_b64"
                          if cleaned["kind"] == "web_fetch_binary" else None)
            if body_field is not None and cleaned["data"][body_field] is not None:
                web_key = (cleaned["kind"], cleaned["data"]["body_sha256"],
                           cleaned["data"]["stored_body_sha256"])
                previous = self.web_body_by_hash.get(web_key)
                if previous is not None:
                    cleaned["data"][body_field] = None
                    cleaned["data"]["body_ref"] = previous
            size = len(_canonical(cleaned)) + 1
            if (len(self.agent_observations) >= MAX_AGENT_OBSERVATIONS
                    or self.size_bytes + size > MAX_OBJECT_BYTES - 65_536):
                raise ObservationError("size_limit")
            self.agent_observations.append(cleaned)
            self.size_bytes += size
            if web_key is not None and cleaned["data"][body_field] is not None:
                self.web_body_by_hash[web_key] = cleaned["id"]
            if (cleaned["kind"] in {"capture_failure", "http_response_metadata"}
                    or cleaned["kind"] in {"web_fetch", "web_fetch_binary"}
                    and cleaned["data"].get("capture", {}).get("complete") is False
                    or cleaned["kind"] == "state_snapshot"
                    and cleaned["data"]["name"] == "automatic_capture_summary"
                    and isinstance(cleaned["data"]["value"], dict)
                    and cleaned["data"]["value"].get("complete") is False):
                self.failures.add(OPTIONAL_CAPTURE_INCOMPLETE if optional_capture_failure(cleaned)
                                  else "capture_incomplete")
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
    if any(type(document[key]) is not int or document[key] < 0 for key in ("observation_count", "local_refusal_count")) or type(document["complete"]) is not bool or not isinstance(document["failures"], list) or any(not isinstance(x, str) or x not in _FAILURES for x in document["failures"]):
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
    if receipt_ordinals != sorted(set(receipt_ordinals)):
        raise ObservationError("response_receipt_order_invalid")
    observations_by_ordinal = {row["ordinal"]: row for row in cleaned}
    for receipt in cleaned_receipts:
        observation = observations_by_ordinal.get(receipt["ordinal"])
        if observation is None:
            continue
        response = observation["outcome"].get("response")
        if (observation["kind"] in {"local_catalog", "local_refusal", "local_parse_error"}
                or observation["call_identity"] != receipt["call_identity"]
                or observation["action_sequence"] != receipt["action_sequence"]
                or observation["operation_id"] != receipt["operation_id"]
                or observation["outcome"]["protocol"] != receipt["protocol"]
                or not isinstance(response, dict)
                or response.get("status") != receipt["status"]):
            raise ObservationError("response_receipt_observation_invalid")
        try:
            raw = base64.b64decode(response["body_b64"], validate=True)
        except (KeyError, TypeError, ValueError):
            raise ObservationError("response_receipt_observation_invalid") from None
        if (not observation["privacy"]["redacted"]
                and (len(raw) != receipt["body_bytes"]
                     or "sha256:" + hashlib.sha256(raw).hexdigest() != receipt["body_sha256"])):
            raise ObservationError("response_receipt_observation_invalid")
    cleaned_agent = [clean_agent_observation(row, secrets) for row in agent_rows]
    ids = [row["id"] for row in cleaned_agent]
    if ids != list(range(len(ids))):
        raise ObservationError("agent_observation_order_invalid")
    bodies = {}
    for row in cleaned_agent:
        if any(_AGENT_REF_RE.fullmatch(reference) and int(reference[6:]) >= row["id"]
               for reference in row["depends_on"]):
            raise ObservationError("agent_dependency_invalid")
        if row["kind"] in {"web_fetch", "web_fetch_binary"}:
            data = row["data"]
            if data["body_ref"] is not None:
                previous = bodies.get(data["body_ref"])
                if (previous is None or previous.get("privacy_review") != data.get("privacy_review")
                        or previous["body_sha256"] != data["body_sha256"]
                        or previous["body_bytes"] != data["body_bytes"]
                        or previous["stored_body_sha256"] != data["stored_body_sha256"]
                        or previous["stored_body_bytes"] != data["stored_body_bytes"]):
                    raise ObservationError("web_body_ref_invalid")
            else:
                bodies[row["id"]] = data
    if document["complete"] and (document["response_receipt_count"] != len(receipts)
            or document["agent_observation_count"] != len(agent_rows)
            or any(row["write_state"] != "socket_written" for row in receipts)
            or any(row["kind"] == "http_response_metadata"
                   or row["kind"] in {"web_fetch", "web_fetch_binary"} and row["data"].get("capture", {}).get("complete") is False
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


_SEGMENT_FIELDS = frozenset({
    "observation_from", "observations", "receipt_from", "response_receipts",
    "receipt_updates", "agent_from", "agent_observations", "observation_count",
    "local_refusal_count", "response_receipt_count", "agent_observation_count",
    "failures",
})


def _clean_segment_payload(value: Mapping[str, Any], secrets: Sequence[str] = ()) -> dict:
    if not isinstance(value, Mapping) or set(value) != _SEGMENT_FIELDS:
        raise ObservationError("segment_payload_invalid")
    for key in ("observation_from", "receipt_from", "agent_from", "observation_count",
                "local_refusal_count", "response_receipt_count", "agent_observation_count"):
        if type(value[key]) is not int or not 0 <= value[key] <= MAX_JOURNAL_ATTEMPTS:
            raise ObservationError("segment_count_invalid")
    if not isinstance(value["failures"], list) or any(not isinstance(x, str) or x not in _FAILURES for x in value["failures"]):
        raise ObservationError("segment_failure_invalid")
    for key in ("observations", "response_receipts", "receipt_updates", "agent_observations"):
        if not isinstance(value[key], list) or len(value[key]) > MAX_OBSERVATIONS:
            raise ObservationError("segment_rows_invalid")
    if (value["observation_from"] + len(value["observations"]) > MAX_OBSERVATIONS
            or value["receipt_from"] + len(value["response_receipts"]) > MAX_OBSERVATIONS
            or value["agent_from"] + len(value["agent_observations"]) > MAX_AGENT_OBSERVATIONS):
        raise ObservationError("segment_rows_invalid")
    cleaned = dict(value)
    cleaned["observations"] = [clean_observation(row, secrets) for row in value["observations"]]
    cleaned["response_receipts"] = [clean_response_receipt(row) for row in value["response_receipts"]]
    cleaned["receipt_updates"] = [clean_response_receipt(row) for row in value["receipt_updates"]]
    cleaned["agent_observations"] = [clean_agent_observation(row, secrets) for row in value["agent_observations"]]
    if (any(row["write_state"] == "prepared" for row in cleaned["receipt_updates"])
            or len(_canonical(cleaned)) > MAX_SEGMENT_BYTES - 4096):
        raise ObservationError("segment_size_or_transition_invalid")
    return cleaned


def build_segment(run_id: str, kind: str, value: Mapping[str, Any], *,
                  lease_generation: int, claim_request_hash: str, number: int,
                  previous_sha256: str | None, cumulative_bytes: int,
                  secrets: Sequence[str] = ()) -> tuple[dict, bytes]:
    segment_ref(run_id, lease_generation, claim_request_hash, number)
    if (kind not in {"execute", "score"} or
            (number == 0) != (previous_sha256 is None) or
            previous_sha256 is not None and not contracts.SHA256_RE.fullmatch(previous_sha256)):
        raise ObservationError("segment_identity_invalid")
    payload = _clean_segment_payload(value, secrets)
    payload_bytes = len(_canonical(payload))
    if (type(cumulative_bytes) is not int or not payload_bytes <= cumulative_bytes <= MAX_OBJECT_BYTES):
        raise ObservationError("journal_capacity_invalid")
    segment = {"schema_version": JOURNAL_SCHEMA_VERSION, "run_id": run_id,
               "kind": kind, "lease_generation": lease_generation,
               "claim_request_hash": claim_request_hash, "number": number,
               "previous_sha256": previous_sha256, "cumulative_bytes": cumulative_bytes,
               "payload": payload,
               "payload_sha256": contracts.document_hash(payload)}
    segment["segment_sha256"] = contracts.document_hash(segment)
    raw = _canonical(segment)
    if len(raw) > MAX_SEGMENT_BYTES:
        raise ObservationError("segment_size_limit")
    return segment, gzip.compress(raw, compresslevel=6, mtime=0)


def read_segment(objects: Any, run_id: str, kind: str, lease_generation: int,
                 claim_request_hash: str, number: int) -> dict | None:
    ref = segment_ref(run_id, lease_generation, claim_request_hash, number)
    getter = getattr(objects, "get_optional_bounded", None)
    if not callable(getter):
        raise ObservationError("journal_store_unavailable")
    try:
        compressed = getter(ref, MAX_SEGMENT_BYTES + 65536)
        if compressed is None:
            return None
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw = stream.read(MAX_SEGMENT_BYTES + 1)
        if len(raw) > MAX_SEGMENT_BYTES:
            raise ObservationError("segment_size_limit")
        segment = json.loads(raw)
        if (not isinstance(segment, dict) or set(segment) != {
                "schema_version", "run_id", "kind", "lease_generation", "claim_request_hash",
                "number", "previous_sha256", "cumulative_bytes", "payload", "payload_sha256", "segment_sha256"}
                or segment["schema_version"] != JOURNAL_SCHEMA_VERSION
                or segment["run_id"] != run_id or segment["kind"] != kind
                or type(segment["lease_generation"]) is not int
                or segment["lease_generation"] != lease_generation
                or segment["claim_request_hash"] != claim_request_hash
                or type(segment["number"]) is not int
                or segment["number"] != number
                or (number > 0 and not isinstance(segment["previous_sha256"], str))
                or (number > 0 and not contracts.SHA256_RE.fullmatch(segment["previous_sha256"]))
                or type(segment["cumulative_bytes"]) is not int
                or not len(_canonical(segment["payload"])) <= segment["cumulative_bytes"] <= MAX_OBJECT_BYTES
                or (number == 0) != (segment["previous_sha256"] is None)
                or contracts.document_hash(segment["payload"]) != segment["payload_sha256"]
                or contracts.document_hash({key: value for key, value in segment.items()
                                            if key != "segment_sha256"}) != segment["segment_sha256"]
                or _clean_segment_payload(segment["payload"]) != segment["payload"]):
            raise ObservationError("journal_invalid")
        return segment
    except ObservationError:
        raise
    except Exception:
        raise ObservationError("journal_unavailable") from None


def read_journal_prefix(objects: Any, run_id: str, kind: str, lease_generation: int,
                        claim_request_hash: str) -> tuple[list[dict], dict | None, str | None]:
    """Return only a verified contiguous prefix; a missing first segment is empty."""
    segments, observations, receipts, agent_rows = [], [], [], []
    receipt_positions: dict[int, int] = {}
    previous = None
    cumulative_bytes = 0
    counts = (0, 0, 0, 0)
    states = []
    failure = None
    for number in range(MAX_SEGMENTS):
        try:
            segment = read_segment(objects, run_id, kind, lease_generation, claim_request_hash, number)
        except ObservationError as exc:
            failure = str(exc)
            break
        if segment is None:
            break
        payload = segment["payload"]
        if (segment["previous_sha256"] != previous
                or segment["cumulative_bytes"] != cumulative_bytes + len(_canonical(payload))
                or payload["observation_from"] != len(observations)
                or payload["receipt_from"] != len(receipts)
                or payload["agent_from"] != len(agent_rows)):
            failure = "journal_chain_invalid"
            break
        next_receipts = list(receipts)
        next_positions = dict(receipt_positions)
        next_observations = observations + payload["observations"]
        next_agent_rows = agent_rows + payload["agent_observations"]
        next_counts = (payload["observation_count"], payload["local_refusal_count"],
                       payload["response_receipt_count"], payload["agent_observation_count"])
        if any(current < prior for current, prior in zip(next_counts, counts)):
            failure = "journal_count_invalid"
            break
        for row in payload["receipt_updates"]:
            position = next_positions.get(row["ordinal"])
            if (position is None or next_receipts[position]["write_state"] != "prepared"
                    or row["write_state"] == "prepared"
                    or any(row[key] != next_receipts[position][key] for key in row
                           if key not in {"write_state", "socket_written_at"})):
                failure = "journal_transition_invalid"
                break
            next_receipts[position] = row
        if failure:
            break
        for row in payload["response_receipts"]:
            if row["ordinal"] in next_positions:
                failure = "journal_receipt_duplicate"
                break
            next_positions[row["ordinal"]] = len(next_receipts)
            next_receipts.append(row)
        if failure:
            break
        if (next_counts[0] < len(next_observations)
                or next_counts[1] < sum(row["kind"] == "local_refusal" for row in next_observations)
                or next_counts[2] < len(next_receipts) or next_counts[3] < len(next_agent_rows)):
            failure = "journal_count_invalid"
            break
        observations, receipts, agent_rows = next_observations, next_receipts, next_agent_rows
        receipt_positions, counts = next_positions, next_counts
        previous = segment["segment_sha256"]
        cumulative_bytes = segment["cumulative_bytes"]
        segments.append(segment)
        states.append((observations, receipts, agent_rows, counts))
    if len(segments) == MAX_SEGMENTS:
        failure = "journal_segment_limit"
    if not segments:
        return [], None, failure
    while states:
        observations, receipts, agent_rows, counts = states[-1]
        document = {"observations": sorted(observations, key=lambda row: row["ordinal"]),
                    "observation_count": counts[0], "local_refusal_count": counts[1],
                    "response_receipts": sorted(receipts, key=lambda row: row["ordinal"]),
                    "response_receipt_count": counts[2], "agent_observations": agent_rows,
                    "agent_observation_count": counts[3], "complete": False,
                    "failures": ["capture_incomplete"]}
        try:
            if len(_canonical(document)) > MAX_OBJECT_BYTES:
                raise ObservationError("journal_prefix_size_limit")
            return segments, _validate_payload(document), failure
        except ObservationError:
            failure = "journal_prefix_invalid"
            states.pop()
            segments.pop()
    return [], None, failure
