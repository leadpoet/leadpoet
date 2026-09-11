"""Versioned, salted commitments to the exact private Arena benchmark inputs.

This module does no I/O. Public verification hashes the exported canonical
UTF-8 preimages; display-only scores and labels never enter those preimages.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Sequence

from lab_arena import contracts

POLICY = "commit_reveal_day2_v1"
MANIFEST_VERSION = "leadpoet.lab_arena.benchmark_commitment.v1"
LEAF_VERSION = "leadpoet.lab_arena.benchmark_leaf.v1"
ARTIFACT_VERSION = "leadpoet.lab_arena.benchmark.v2"
COUNT = contracts.BENCHMARK_ICP_COUNT
# Generated ICPs are small structured records. These bounds permit substantial
# qualification text without borrowing the much smaller signed-request limit.
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
LIMITS = contracts.StrictLimits(
    max_depth=16, max_list_items=512, max_object_keys=256,
    max_string_bytes=256 * 1024, max_total_bytes=MAX_ARTIFACT_BYTES,
)
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NONCE = re.compile(r"[0-9a-f]{64}\Z")
_CONTEXT_FIELDS = (
    "network_name", "netuid", "round_id", "icp_set_date", "evaluation_date",
)
_MANIFEST_FIELDS = {
    "schema_version", *_CONTEXT_FIELDS, "public_at", "disclosure_policy",
    "icp_count", "entries",
}
_LEAF_FIELDS = {"schema_version", *_CONTEXT_FIELDS, "icp_position", "nonce", "icp"}


class BenchmarkCommitmentError(contracts.ArenaContractError):
    """An invalid commitment; messages deliberately contain no private input."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise BenchmarkCommitmentError(code)


def instant(value: Any) -> datetime:
    try:
        _require(isinstance(value, str), "benchmark_time_invalid")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        _require(parsed.tzinfo is not None, "benchmark_time_invalid")
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError) as exc:
        raise BenchmarkCommitmentError("benchmark_time_invalid") from exc


def iso(value: datetime) -> str:
    _require(value.tzinfo is not None, "benchmark_time_invalid")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _date(value: Any) -> date:
    try:
        _require(isinstance(value, str), "benchmark_date_invalid")
        parsed = date.fromisoformat(value)
        _require(parsed.isoformat() == value, "benchmark_date_invalid")
        return parsed
    except (ValueError, TypeError) as exc:
        raise BenchmarkCommitmentError("benchmark_date_invalid") from exc


def policy(round_row: Mapping[str, Any]) -> str | None:
    config = round_row.get("configuration_doc") or {}
    _require(isinstance(config, Mapping), "benchmark_policy_invalid")
    if "benchmark_disclosure_policy" not in config:
        return None
    _require(config["benchmark_disclosure_policy"] == POLICY, "benchmark_policy_invalid")
    return POLICY


def _next_day(value: date | datetime):
    try:
        return value + timedelta(days=1)
    except OverflowError as exc:
        raise BenchmarkCommitmentError("benchmark_date_invalid") from exc


def _context_valid(context: Mapping[str, Any]) -> None:
    _require(
        isinstance(context.get("network_name"), str)
        and bool(context["network_name"])
        and len(context["network_name"]) <= 64,
        "benchmark_scope_invalid",
    )
    _require(type(context.get("netuid")) is int and context["netuid"] > 0, "benchmark_scope_invalid")
    _require(isinstance(context.get("round_id"), str) and bool(contracts.ROUND_ID_RE.fullmatch(context["round_id"])), "benchmark_scope_invalid")
    bank = _date(context.get("icp_set_date"))
    evaluation = _date(context.get("evaluation_date"))
    _require(evaluation == _next_day(bank), "benchmark_date_invalid")


def round_metadata(round_row: Mapping[str, Any]) -> dict:
    """Validate frozen schedule metadata, including an open round's plan."""

    _require(policy(round_row) == POLICY, "benchmark_policy_invalid")
    config = round_row["configuration_doc"]
    schedule = config.get("schedule") or {}
    _require(isinstance(schedule, Mapping), "benchmark_time_invalid")
    opened, cutoff = instant(schedule.get("submission_open")), instant(schedule.get("submission_cutoff"))
    _require(cutoff > opened and cutoff.date() == _next_day(opened.date()), "benchmark_time_invalid")
    reveal = instant(round_row.get("benchmark_reveal_at"))
    _require(reveal == _next_day(cutoff), "benchmark_time_invalid")
    context = {
        "network_name": config.get("network_name", "finney"),
        "netuid": config.get("netuid", 71),
        "round_id": round_row.get("round_id"),
        "icp_set_date": opened.date().isoformat(),
        "evaluation_date": cutoff.date().isoformat(),
    }
    _context_valid(context)
    for field in ("icp_set_date", "evaluation_date"):
        if round_row.get(field) is not None:
            _require(round_row[field] == context[field], "benchmark_date_invalid")
    return {**context, "public_at": iso(reveal), "disclosure_policy": POLICY}


def _bounded(value: Any) -> None:
    try:
        contracts.check_strict_document(value, LIMITS)
    except (contracts.ArenaContractError, UnicodeError) as exc:
        raise BenchmarkCommitmentError("benchmark_document_invalid") from exc


def _keys(value: Any, keys: set[str]) -> None:
    _require(isinstance(value, Mapping) and set(value) == keys, "benchmark_document_invalid")


def validate_commitment(document: Any, *, round_row: Mapping[str, Any] | None = None) -> dict:
    _bounded(document)
    _keys(document, {"manifest", "manifest_hash", "canonical_manifest"})
    manifest = document["manifest"]
    _keys(manifest, _MANIFEST_FIELDS)
    _require(manifest["schema_version"] == MANIFEST_VERSION and manifest["disclosure_policy"] == POLICY, "benchmark_policy_invalid")
    _context_valid(manifest)
    _require(instant(manifest["public_at"]).date() == _next_day(_date(manifest["evaluation_date"])), "benchmark_time_invalid")
    _require(iso(instant(manifest["public_at"])) == manifest["public_at"], "benchmark_time_invalid")
    _require(type(manifest["icp_count"]) is int and manifest["icp_count"] == COUNT, "benchmark_count_invalid")
    entries = manifest["entries"]
    _require(isinstance(entries, list) and len(entries) == COUNT, "benchmark_count_invalid")
    for position, entry in enumerate(entries):
        _keys(entry, {"icp_position", "icp_hash"})
        _require(type(entry["icp_position"]) is int and entry["icp_position"] == position, "benchmark_position_invalid")
        _require(isinstance(entry["icp_hash"], str) and bool(_HASH.fullmatch(entry["icp_hash"])), "benchmark_hash_invalid")
    _require(document["canonical_manifest"] == contracts.canonical_json(manifest), "benchmark_canonical_invalid")
    _require(document["manifest_hash"] == contracts.document_hash(manifest), "benchmark_hash_invalid")
    if round_row is not None:
        expected = round_metadata(round_row)
        _require(all(manifest[key] == value for key, value in expected.items()), "benchmark_round_mismatch")
    return dict(document)


def committed_document(round_row: Mapping[str, Any]) -> dict:
    """Validate the committed row without reading its private object."""

    document = validate_commitment(round_row.get("benchmark_commitment_doc"), round_row=round_row)
    _require(round_row.get("status") != "open", "benchmark_not_committed")
    manifest = document["manifest"]
    _require(all(round_row.get(key) == manifest[key] for key in ("icp_set_date", "evaluation_date")), "benchmark_date_invalid")
    _require(isinstance(round_row.get("benchmark_ref"), str) and bool(round_row["benchmark_ref"]), "benchmark_not_committed")
    committed_at = instant(round_row.get("benchmark_committed_at"))
    cutoff = instant(round_row["configuration_doc"]["schedule"]["submission_cutoff"])
    _require(committed_at >= cutoff, "benchmark_time_invalid")
    return document


def build_artifact(
    round_row: Mapping[str, Any], icps: Sequence[Mapping[str, Any]], *,
    nonce_factory: Callable[[], str] | None = None,
) -> dict:
    metadata = round_metadata(round_row)
    _require(len(icps) == COUNT, "benchmark_count_invalid")
    nonce_factory = nonce_factory or (lambda: secrets.token_hex(32))
    context = {key: metadata[key] for key in _CONTEXT_FIELDS}
    preimages = [contracts.canonical_json({
        "schema_version": LEAF_VERSION, **context, "icp_position": position,
        "nonce": nonce_factory(), "icp": dict(icp),
    }) for position, icp in enumerate(icps)]
    manifest = {
        "schema_version": MANIFEST_VERSION, **metadata, "icp_count": COUNT,
        "entries": [{"icp_position": position, "icp_hash": contracts.hash_bytes(value.encode("utf-8"))} for position, value in enumerate(preimages)],
    }
    document = {
        "manifest": manifest, "manifest_hash": contracts.document_hash(manifest),
        "canonical_manifest": contracts.canonical_json(manifest),
    }
    artifact = {"schema_version": ARTIFACT_VERSION, "commitment": document, "canonical_preimages": preimages}
    validate_artifact(artifact, commitment=document)
    return artifact


def _leaf(preimage: Any, document: Mapping[str, Any], position: int) -> dict:
    _require(isinstance(preimage, str) and len(preimage.encode("utf-8")) <= LIMITS.max_string_bytes, "benchmark_document_invalid")
    try:
        leaf = json.loads(preimage)
    except (TypeError, ValueError, RecursionError) as exc:
        raise BenchmarkCommitmentError("benchmark_document_invalid") from exc
    _bounded(leaf)
    _keys(leaf, _LEAF_FIELDS)
    manifest = document["manifest"]
    _require(leaf["schema_version"] == LEAF_VERSION, "benchmark_document_invalid")
    _require(type(leaf["icp_position"]) is int and leaf["icp_position"] == position, "benchmark_position_invalid")
    _require(all(leaf[key] == manifest[key] for key in _CONTEXT_FIELDS), "benchmark_round_mismatch")
    _require(isinstance(leaf["nonce"], str) and bool(_NONCE.fullmatch(leaf["nonce"])), "benchmark_nonce_invalid")
    _require(isinstance(leaf["icp"], dict) and isinstance(leaf["icp"].get("icp_id"), str) and bool(leaf["icp"]["icp_id"].strip()), "benchmark_icp_invalid")
    _require(not ({"icp_position", "baseline_score"} & set(leaf["icp"])), "benchmark_icp_invalid")
    _require(preimage == contracts.canonical_json(leaf), "benchmark_canonical_invalid")
    _require(contracts.hash_bytes(preimage.encode("utf-8")) == manifest["entries"][position]["icp_hash"], "benchmark_hash_invalid")
    return leaf


def validate_artifact(artifact: Any, *, commitment: Mapping[str, Any]) -> list[dict]:
    _bounded(artifact)
    _keys(artifact, {"schema_version", "commitment", "canonical_preimages"})
    _require(artifact["schema_version"] == ARTIFACT_VERSION, "benchmark_document_invalid")
    document = validate_commitment(commitment)
    _require(artifact["commitment"] == document, "benchmark_hash_invalid")
    preimages = artifact["canonical_preimages"]
    _require(isinstance(preimages, list) and len(preimages) == COUNT, "benchmark_count_invalid")
    leaves = [_leaf(value, document, position) for position, value in enumerate(preimages)]
    _require(len({leaf["icp"]["icp_id"].strip() for leaf in leaves}) == COUNT, "benchmark_icp_invalid")
    _require(len({leaf["nonce"] for leaf in leaves}) == COUNT, "benchmark_nonce_invalid")
    return [leaf["icp"] for leaf in leaves]


def verify_assignment(proof: Any, *, round_id: str, position: int, evaluation_date: str, icp: Mapping[str, Any]) -> None:
    _keys(proof, {"commitment", "canonical_preimage"})
    document = validate_commitment(proof["commitment"])
    _require(type(position) is int and position in range(COUNT), "benchmark_position_invalid")
    manifest = document["manifest"]
    _require(manifest["round_id"] == round_id and manifest["evaluation_date"] == evaluation_date, "benchmark_round_mismatch")
    leaf = _leaf(proof["canonical_preimage"], document, position)
    _require(contracts.canonical_json(leaf["icp"]) == contracts.canonical_json(icp), "benchmark_icp_mismatch")


def verify_reveal(commitment_response: Mapping[str, Any], reveal_response: Mapping[str, Any]) -> str:
    """Verify downloaded Day 1 and Day 2 responses, including decorated ICPs."""

    _bounded(commitment_response)
    _bounded(reveal_response)
    _keys(commitment_response, {"round_id", "manifest", "manifest_hash", "canonical_manifest", "committed_at"})
    instant(commitment_response["committed_at"])
    document = validate_commitment({key: commitment_response.get(key) for key in ("manifest", "manifest_hash", "canonical_manifest")})
    manifest = document["manifest"]
    _require(commitment_response.get("round_id") == manifest["round_id"] == reveal_response.get("round_id"), "benchmark_round_mismatch")
    _require(reveal_response.get("public_at") == manifest["public_at"] and reveal_response.get("icp_set_date") == manifest["icp_set_date"] and reveal_response.get("disclosure_policy") == POLICY, "benchmark_round_mismatch")
    revealed_commitment = reveal_response.get("commitment")
    _require(revealed_commitment == commitment_response, "benchmark_hash_invalid")
    verification = reveal_response.get("verification")
    _keys(verification, {"manifest_hash", "canonical_preimages"})
    _require(verification["manifest_hash"] == document["manifest_hash"], "benchmark_hash_invalid")
    icps = validate_artifact({"schema_version": ARTIFACT_VERSION, "commitment": document, "canonical_preimages": verification["canonical_preimages"]}, commitment=document)
    displayed = reveal_response.get("icps")
    _require(isinstance(displayed, list) and len(displayed) == COUNT, "benchmark_count_invalid")
    _require(type(reveal_response.get("public_icp_count")) is int and reveal_response["public_icp_count"] == COUNT and type(reveal_response.get("private_icp_count")) is int and reveal_response["private_icp_count"] == 0, "benchmark_count_invalid")
    for position, (raw, view) in enumerate(zip(icps, displayed)):
        _require(isinstance(view, Mapping) and type(view.get("icp_position")) is int and view["icp_position"] == position, "benchmark_position_invalid")
        # These reserved display fields never appear in the committed input.
        # All raw input fields must match exactly.
        expected = {**raw, "icp_position": position, "baseline_score": view.get("baseline_score")}
        _require(contracts.canonical_json(view) == contracts.canonical_json(expected), "benchmark_icp_mismatch")
    return document["manifest_hash"]
