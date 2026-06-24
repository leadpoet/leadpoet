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
    "hidden_icp",
    "icp_plaintext",
    ".dkr.ecr.",
    "image_digest",
    "private_repo",
    "judge_prompt",
)
SECRET_KEY_MARKERS = (
    "api_key",
    "raw_secret",
    "raw_openrouter",
    "credential",
    "token",
    "private_model_manifest_doc",
    "candidate_patch_manifest",
    "image_digest",
    "proxy_url",
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


def build_research_lab_audit_bundle(
    *,
    epoch: int,
    ticket_rows: Sequence[Mapping[str, Any]],
    queue_rows: Sequence[Mapping[str, Any]],
    receipt_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    candidate_event_rows: Sequence[Mapping[str, Any]],
    loop_event_rows: Sequence[Mapping[str, Any]],
    dispatch_event_rows: Sequence[Mapping[str, Any]],
    rolling_window_rows: Sequence[Mapping[str, Any]],
    benchmark_rows: Sequence[Mapping[str, Any]],
    private_model_version_rows: Sequence[Mapping[str, Any]] = (),
    promotion_event_rows: Sequence[Mapping[str, Any]] = (),
    private_repo_commit_event_rows: Sequence[Mapping[str, Any]] = (),
    public_benchmark_report_rows: Sequence[Mapping[str, Any]] = (),
    score_bundle_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a redacted, hash-linked Research Lab audit bundle.

    Validators use this to verify gateway lifecycle and score-bundle math
    without seeing private model code, private image refs, patch manifests, or
    hidden ICP plaintext.
    """
    source_state = {
        "ticket_rows": _stable_rows(_redact_rows(ticket_rows, _AUDIT_TICKET_FIELDS)),
        "queue_rows": _stable_rows(_redact_rows(queue_rows, _AUDIT_QUEUE_FIELDS)),
        "receipt_rows": _stable_rows(_redact_rows(receipt_rows, _AUDIT_RECEIPT_FIELDS)),
        "candidate_rows": _stable_rows(_redact_rows(candidate_rows, _AUDIT_CANDIDATE_FIELDS)),
        "candidate_event_rows": _stable_rows(_redact_rows(candidate_event_rows, _AUDIT_CANDIDATE_EVENT_FIELDS)),
        "auto_research_loop_event_rows": _stable_rows(_redact_rows(loop_event_rows, _AUDIT_LOOP_EVENT_FIELDS)),
        "scoring_dispatch_event_rows": _stable_rows(_redact_rows(dispatch_event_rows, _AUDIT_DISPATCH_EVENT_FIELDS)),
        "rolling_icp_window_rows": _stable_rows(_redact_rows(rolling_window_rows, _AUDIT_ROLLING_WINDOW_FIELDS)),
        "private_baseline_benchmark_rows": _stable_rows(_redact_rows(benchmark_rows, _AUDIT_BENCHMARK_FIELDS)),
        "private_model_version_rows": _stable_rows(_redact_rows(private_model_version_rows, _AUDIT_PRIVATE_MODEL_VERSION_FIELDS)),
        "candidate_promotion_event_rows": _stable_rows(_redact_rows(promotion_event_rows, _AUDIT_PROMOTION_EVENT_FIELDS)),
        "private_repo_commit_event_rows": _stable_rows(_redact_rows(private_repo_commit_event_rows, _AUDIT_PRIVATE_REPO_COMMIT_FIELDS)),
        "public_benchmark_report_rows": _stable_rows(_redact_rows(public_benchmark_report_rows, _AUDIT_PUBLIC_BENCHMARK_FIELDS)),
        "score_bundle_rows": _stable_rows(_redact_rows(score_bundle_rows, _AUDIT_SCORE_BUNDLE_FIELDS)),
    }
    if contains_secret_material(source_state):
        raise ValueError("Research Lab audit source state contains private or secret material")

    source_state_hash = sha256_json(source_state)
    bundle_without_id = {
        "bundle_id": "",
        "schema_version": "1.0",
        "bundle_type": "research_lab_signed_audit_bundle",
        "epoch": int(epoch),
        "generated_at": utc_now_iso(),
        "read_only": True,
        "on_chain_submission_allowed": False,
        "private_model_execution_boundary": "gateway_owned",
        "validator_private_artifact_access": "forbidden",
        "source_state_hash": source_state_hash,
        "source_state": source_state,
        "observability": {
            "ticket_count": len(ticket_rows),
            "run_queue_count": len(queue_rows),
            "receipt_count": len(receipt_rows),
            "candidate_count": len(candidate_rows),
            "candidate_event_count": len(candidate_event_rows),
            "auto_research_loop_event_count": len(loop_event_rows),
            "scoring_dispatch_event_count": len(dispatch_event_rows),
            "rolling_icp_window_count": len(rolling_window_rows),
            "private_baseline_benchmark_count": len(benchmark_rows),
            "private_model_version_count": len(private_model_version_rows),
            "candidate_promotion_event_count": len(promotion_event_rows),
            "private_repo_commit_event_count": len(private_repo_commit_event_rows),
            "public_benchmark_report_count": len(public_benchmark_report_rows),
            "score_bundle_count": len(score_bundle_rows),
        },
        "verifier_contract": {
            "required_checks": [
                "no_raw_secret_material",
                "no_private_model_manifest_doc",
                "no_candidate_patch_manifest",
                "no_private_image_ref",
                "no_hidden_icp_plaintext",
                "source_state_hash_matches",
                "score_bundle_hashes_match",
                "score_bundle_aggregates_recompute",
                "promotion_events_reference_score_bundles",
                "public_benchmark_reports_are_sanitized",
            ],
            "private_model_recompute": "forbidden_for_main_validators_v1",
        },
    }
    bundle_id = "research_lab_audit:" + sha256_json(bundle_without_id).split(":", 1)[1]
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


def _redact_rows(rows: Sequence[Mapping[str, Any]], allowed_fields: set[str]) -> list[dict[str, Any]]:
    redacted: list[dict[str, Any]] = []
    for row in rows:
        item = {key: _redact_value(value) for key, value in dict(row).items() if key in allowed_fields}
        redacted.append(item)
    return redacted


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _redact_value(item)
            for key, item in value.items()
            if str(key) not in _FORBIDDEN_AUDIT_KEYS
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


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


_FORBIDDEN_AUDIT_KEYS = {
    "private_model_manifest_doc",
    "private_model_manifest",
    "candidate_patch_manifest",
    "image_digest",
    "icp",
    "icp_data",
    "hidden_icp",
    "icp_plaintext",
    "proxy_url",
}

_AUDIT_TICKET_FIELDS = {
    "ticket_id",
    "miner_hotkey",
    "island",
    "requested_loop_count",
    "ticket_hash",
    "ticket_doc",
    "current_ticket_status",
    "current_event_seq",
    "current_reason",
    "created_at",
}
_AUDIT_QUEUE_FIELDS = {
    "run_id",
    "ticket_id",
    "queue_priority",
    "current_queue_status",
    "current_event_seq",
    "current_reason",
    "current_status_at",
}
_AUDIT_RECEIPT_FIELDS = {
    "receipt_id",
    "ticket_id",
    "run_id",
    "miner_hotkey",
    "island",
    "receipt_hash",
    "current_receipt_status",
    "current_event_seq",
    "current_status_at",
    "provider_usage",
    "cost_ledger",
}
_AUDIT_CANDIDATE_FIELDS = {
    "candidate_id",
    "run_id",
    "ticket_id",
    "receipt_id",
    "miner_hotkey",
    "island",
    "parent_artifact_hash",
    "candidate_artifact_hash",
    "private_model_manifest_hash",
    "candidate_patch_hash",
    "anchored_hash",
    "current_candidate_status",
    "current_event_seq",
    "current_score_bundle_id",
    "current_status_at",
    "redacted_public_summary",
}
_AUDIT_CANDIDATE_EVENT_FIELDS = {
    "event_id",
    "candidate_id",
    "run_id",
    "ticket_id",
    "seq",
    "event_type",
    "candidate_status",
    "evaluator_ref",
    "reason",
    "score_bundle_id",
    "anchored_hash",
    "event_doc",
    "created_at",
}
_AUDIT_LOOP_EVENT_FIELDS = {
    "event_id",
    "run_id",
    "ticket_id",
    "receipt_id",
    "seq",
    "event_type",
    "loop_status",
    "node_id",
    "worker_ref",
    "elapsed_seconds",
    "candidate_artifact_hash",
    "candidate_patch_hash",
    "provider_usage",
    "cost_ledger",
    "event_doc",
    "anchored_hash",
    "created_at",
}
_AUDIT_DISPATCH_EVENT_FIELDS = {
    "dispatch_event_id",
    "dispatch_type",
    "dispatch_status",
    "candidate_id",
    "run_id",
    "ticket_id",
    "rolling_window_hash",
    "score_bundle_id",
    "benchmark_bundle_id",
    "worker_ref",
    "proxy_ref_hash",
    "event_doc",
    "anchored_hash",
    "created_at",
}
_AUDIT_ROLLING_WINDOW_FIELDS = {
    "rolling_window_hash",
    "required_days",
    "icps_per_day",
    "selected_set_count",
    "selected_icp_count",
    "window_doc",
    "anchored_hash",
    "created_at",
}
_AUDIT_BENCHMARK_FIELDS = {
    "benchmark_bundle_id",
    "benchmark_date",
    "private_model_artifact_hash",
    "private_model_manifest_hash",
    "rolling_window_hash",
    "evaluation_epoch",
    "aggregate_score",
    "scoring_worker_ref",
    "proxy_ref_hash",
    "signature_ref",
    "score_summary_doc",
    "benchmark_bundle_hash",
    "anchored_hash",
    "created_at",
}
_AUDIT_PRIVATE_MODEL_VERSION_FIELDS = {
    "private_model_version_id",
    "model_artifact_hash",
    "private_model_manifest_hash",
    "git_commit_sha",
    "config_hash",
    "component_registry_version",
    "scoring_adapter_version",
    "source_candidate_id",
    "source_score_bundle_id",
    "source_benchmark_bundle_id",
    "signature_ref",
    "build_id",
    "redacted_version_doc",
    "version_hash",
    "anchored_hash",
    "current_event_seq",
    "current_event_type",
    "current_version_status",
    "current_reason",
    "current_status_at",
    "created_at",
}
_AUDIT_PROMOTION_EVENT_FIELDS = {
    "promotion_event_id",
    "candidate_id",
    "derived_candidate_id",
    "source_score_bundle_id",
    "derived_score_bundle_id",
    "private_model_version_id",
    "event_type",
    "promotion_status",
    "active_parent_artifact_hash",
    "candidate_parent_artifact_hash",
    "rolling_window_hash",
    "improvement_points",
    "threshold_points",
    "worker_ref",
    "event_doc",
    "anchored_hash",
    "created_at",
}
_AUDIT_PRIVATE_REPO_COMMIT_FIELDS = {
    "commit_event_id",
    "candidate_id",
    "score_bundle_id",
    "private_model_version_id",
    "commit_status",
    "git_commit_sha",
    "branch_name",
    "private_repo_ref_hash",
    "event_doc",
    "anchored_hash",
    "created_at",
}
_AUDIT_PUBLIC_BENCHMARK_FIELDS = {
    "report_id",
    "benchmark_date",
    "benchmark_bundle_id",
    "private_model_artifact_hash",
    "private_model_manifest_hash",
    "rolling_window_hash",
    "aggregate_score",
    "report_doc",
    "report_hash",
    "anchored_hash",
    "current_event_seq",
    "current_event_type",
    "current_report_status",
    "current_status_at",
    "created_at",
}
_AUDIT_SCORE_BUNDLE_FIELDS = {
    "score_bundle_id",
    "run_id",
    "ticket_id",
    "receipt_id",
    "miner_hotkey",
    "island",
    "evaluation_epoch",
    "bundle_status",
    "parent_artifact_hash",
    "candidate_artifact_hash",
    "private_model_manifest_hash",
    "candidate_patch_hash",
    "icp_set_hash",
    "scoring_version",
    "evaluator_version",
    "score_bundle_hash",
    "anchored_hash",
    "signature_ref",
    "score_bundle_doc",
    "current_event_status",
    "current_status_at",
}
