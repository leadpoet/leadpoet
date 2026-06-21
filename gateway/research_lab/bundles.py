"""Research Lab gateway bundle builders.

These helpers intentionally stay under ``gateway/`` so the production gateway
does not import validator-only Research Lab modules when deployed by gateway
file sync.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping, Sequence


SECRET_MARKERS = (
    "sk-or-",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
    "service_role",
)
SECRET_KEY_MARKERS = (
    "api_key",
    "raw_secret",
    "raw_openrouter",
    "credential",
    "token",
)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_json(data: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_shadow_report_bundle(
    *,
    epoch: int,
    weight_input_snapshots: Sequence[Mapping[str, Any]],
    ticket_rows: Sequence[Mapping[str, Any]],
    queue_rows: Sequence[Mapping[str, Any]],
    receipt_rows: Sequence[Mapping[str, Any]],
    reimbursement_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic, read-only Research Lab shadow bundle.

    The current production-safe shape is deliberately non-mutating:
    ``submission_allowed`` stays false and the Research Lab component defaults
    to a zero/empty vector unless a future writer emits an explicit verified
    weight vector into ``research_weight_input_snapshots.snapshot_doc``.
    """
    source_state = {
        "epoch": int(epoch),
        "weight_input_snapshots": _stable_rows(weight_input_snapshots),
        "ticket_rows": _stable_rows(ticket_rows),
        "queue_rows": _stable_rows(queue_rows),
        "receipt_rows": _stable_rows(receipt_rows),
        "reimbursement_rows": _stable_rows(reimbursement_rows),
    }
    if contains_secret_material(source_state):
        raise ValueError("Research Lab shadow source state contains raw secret material")

    weight_vector = _extract_verified_shadow_weight_vector(weight_input_snapshots)
    weight_vector_hash = sha256_json(weight_vector)
    source_state_hash = sha256_json(source_state)

    observability = {
        "ticket_count": len(ticket_rows),
        "queued_run_count": sum(1 for row in queue_rows if row.get("current_queue_status") == "queued"),
        "completed_receipt_count": sum(1 for row in receipt_rows if row.get("current_receipt_status") == "completed"),
        "failed_receipt_count": sum(1 for row in receipt_rows if row.get("current_receipt_status") == "failed"),
        "reimbursement_award_count": len(reimbursement_rows),
        "weight_input_snapshot_count": len(weight_input_snapshots),
        "weight_sum": int(weight_vector.get("weight_sum", 0)),
    }

    bundle_without_id = {
        "bundle_id": "",
        "schema_version": "1.0",
        "bundle_type": "research_lab_shadow_report",
        "epoch": int(epoch),
        "generated_at": utc_now_iso(),
        "shadow_only": True,
        "read_only": True,
        "submission_allowed": False,
        "on_chain_submission_allowed": False,
        "source_state_hash": source_state_hash,
        "source_state": source_state,
        "weight_vector": weight_vector,
        "weight_vector_hash": weight_vector_hash,
        "observability": observability,
        "verifier_contract": {
            "required_checks": [
                "shadow_flags_enabled",
                "live_mutation_flags_false",
                "no_raw_secret_material",
                "source_state_hash_matches",
                "weight_vector_hash_matches",
                "open_verifier_golden_vectors_pass",
            ],
            "live_weight_mutation": "forbidden_for_shadow_bundle",
        },
    }
    bundle_id = "research_lab_shadow_bundle:" + sha256_json(bundle_without_id).split(":", 1)[1][:32]
    return {**bundle_without_id, "bundle_id": bundle_id}


def contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered_key = str(key).lower()
            if any(marker in lowered_key for marker in SECRET_KEY_MARKERS):
                return True
            if contains_secret_material(item):
                return True
    elif isinstance(value, list):
        return any(contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in SECRET_MARKERS)
    return False


def _stable_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in rows]
    return sorted(normalized, key=lambda row: canonical_json(row))


def _extract_verified_shadow_weight_vector(weight_input_snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    explicit_vectors: list[dict[str, Any]] = []
    for row in weight_input_snapshots:
        doc = row.get("snapshot_doc") if isinstance(row.get("snapshot_doc"), Mapping) else {}
        candidate = doc.get("weight_vector") or doc.get("shadow_weight_vector")
        if not isinstance(candidate, Mapping):
            continue
        normalized = {
            "epoch": int(row.get("epoch", candidate.get("epoch", 0))),
            "u16_weights": {str(uid): int(weight) for uid, weight in dict(candidate.get("u16_weights", {})).items()},
            "weight_sum": int(candidate.get("weight_sum", sum(int(v) for v in dict(candidate.get("u16_weights", {})).values()))),
            "source": "research_weight_input_snapshots.snapshot_doc",
        }
        expected_hash = row.get("weight_vector_hash") or candidate.get("weight_vector_hash")
        if expected_hash and expected_hash != sha256_json(normalized):
            normalized["source_hash_warning"] = "snapshot_weight_vector_hash_mismatch"
        explicit_vectors.append(normalized)

    if explicit_vectors:
        return explicit_vectors[0]

    return {
        "epoch": int(weight_input_snapshots[0]["epoch"]) if weight_input_snapshots else 0,
        "u16_weights": {},
        "weight_sum": 0,
        "source": "no_verified_research_lab_weight_vector_emitted",
    }
