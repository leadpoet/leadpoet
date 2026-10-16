"""Exact, private runtime input documents; absent evidence is never replayable."""

from __future__ import annotations

import gzip
import io
import json
import re
from typing import Any, Mapping, Sequence

from lab_arena import contracts, training_capture

SCHEMA_VERSION = "leadpoet.lab_arena.training_input.v1"
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_OBJECT_BYTES = MAX_INPUT_BYTES + 64 * 1024
MAX_COMPRESSED_BYTES = MAX_OBJECT_BYTES + 64 * 1024
_RUN_RE = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class InputCaptureError(ValueError):
    """Input evidence is missing, conflicting, or invalid."""


def object_ref(run_id: str, lease_generation: int) -> str:
    if not isinstance(run_id, str) or not _RUN_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise InputCaptureError("input_run_invalid")
    if type(lease_generation) is not int or lease_generation < 1:
        raise InputCaptureError("input_generation_invalid")
    return "private/arena-training/runs/%s/leases/%d/input.json.gz" % (run_id, lease_generation)


def _validate_claim_hash(claim_request_hash: str) -> None:
    if not isinstance(claim_request_hash, str) or not _HASH_RE.fullmatch(claim_request_hash):
        raise InputCaptureError("input_claim_invalid")


def build_snapshot(
    run_id: str, kind: str, input_document: Mapping[str, Any],
    *, lease_generation: int, claim_request_hash: str, secrets: Sequence[str] = (),
) -> tuple[dict[str, Any], bytes]:
    """Retain the complete JSON document except explicit privacy redactions."""
    object_ref(run_id, lease_generation)
    _validate_claim_hash(claim_request_hash)
    if kind not in {"execute", "score"} or not isinstance(input_document, Mapping):
        raise InputCaptureError("input_document_invalid")
    if len(contracts.canonical_json(input_document).encode("utf-8")) > MAX_INPUT_BYTES:
        raise InputCaptureError("input_size_limit")
    cleaned, redactions = training_capture.sanitize_training_value(input_document, secrets)
    document = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "kind": kind,
        "lease_generation": lease_generation, "claim_request_hash": claim_request_hash,
        "input": cleaned,
        "input_sha256": contracts.document_hash(cleaned),
        "privacy": {"redacted": bool(redactions), "redactions": redactions},
    }
    raw = contracts.canonical_json(document).encode("utf-8")
    if len(raw) > MAX_OBJECT_BYTES:
        raise InputCaptureError("input_size_limit")
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise InputCaptureError("input_size_limit")
    return document, compressed


def read_input(objects: Any, run_id: str, kind: str, lease_generation: int, claim_request_hash: str) -> dict[str, Any]:
    """Read a bounded snapshot and verify its run, schema, hash, and privacy."""
    try:
        _validate_claim_hash(claim_request_hash)
        compressed = objects.get_bounded(object_ref(run_id, lease_generation), MAX_COMPRESSED_BYTES)
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw = stream.read(MAX_OBJECT_BYTES + 1)
        if len(raw) > MAX_OBJECT_BYTES:
            raise InputCaptureError("input_size_limit")
        document = json.loads(raw)
        if (
            not isinstance(document, dict)
            or set(document) != {"schema_version", "run_id", "kind", "lease_generation", "claim_request_hash", "input", "input_sha256", "privacy"}
            or document.get("schema_version") != SCHEMA_VERSION
            or document.get("run_id") != run_id
            or type(document.get("lease_generation")) is not int
            or document.get("lease_generation") != lease_generation
            or document.get("claim_request_hash") != claim_request_hash
            or kind not in {"execute", "score"} or document.get("kind") != kind
            or not isinstance(document.get("input"), dict)
            or document.get("input_sha256") != contracts.document_hash(document["input"])
        ):
            raise InputCaptureError("input_identity_or_hash_invalid")
        privacy = document.get("privacy")
        if (
            not isinstance(privacy, dict) or set(privacy) != {"redacted", "redactions"}
            or type(privacy.get("redacted")) is not bool
            or not isinstance(privacy.get("redactions"), list)
            or any(reason not in {"credential", "hidden_reasoning"} for reason in privacy["redactions"])
            or privacy["redactions"] != sorted(set(privacy["redactions"]))
            or privacy["redacted"] != bool(privacy["redactions"])
        ):
            raise InputCaptureError("input_privacy_invalid")
        return document
    except InputCaptureError:
        raise
    except Exception:
        raise InputCaptureError("input_unavailable") from None
