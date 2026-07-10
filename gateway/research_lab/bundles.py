"""Research Lab gateway bundle builders.

These helpers intentionally stay under ``gateway/`` so the production gateway
does not import validator-only Research Lab modules when deployed by gateway
file sync.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from typing import Any, Mapping, Sequence


logger = logging.getLogger(__name__)

AUDIT_SECRET_SCAN_MODE_ENV_VAR = "RESEARCH_LAB_AUDIT_SECRET_SCAN_MODE"
AUDIT_SECRET_SCAN_MODE_REDACT = "redact"
AUDIT_SECRET_SCAN_MODE_RAISE = "raise"
AUDIT_INLINE_VALUE_MAX_BYTES = 16 * 1024
_AUDIT_COMPACTION_EXEMPT_FIELDS = {
    ("score_bundle_rows", "score_bundle_doc"),
}
SHADOW_REPORT_ALLOWLIST_ENV_VAR = "RESEARCH_LAB_SHADOW_REPORT_ALLOWLIST"
_TRUTHY = {"1", "true", "yes", "on"}

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
    "private_model_manifest_doc",
    "candidate_patch_manifest",
    "image_digest",
    "proxy_url",
)
SECRET_TOKEN_KEY_MARKERS = (
    "access_token",
    "api_token",
    "auth_token",
    "bearer_token",
    "refresh_token",
    "session_token",
    "token_key",
    "token_secret",
    "token_value",
)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_json(data: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def audit_secret_scan_mode() -> str:
    """Return the bundle secret-scan mode: ``redact`` (default) or ``raise``.

    ``raise`` restores the legacy strict behavior for auditors: any secret
    marker anywhere in the source state throws the whole bundle away. The
    default ``redact`` mode replaces offending values with deterministic
    placeholders so one poisoned historical row cannot wedge every future
    bundle build.
    """
    mode = os.getenv(AUDIT_SECRET_SCAN_MODE_ENV_VAR, AUDIT_SECRET_SCAN_MODE_REDACT).strip().lower()
    if mode == AUDIT_SECRET_SCAN_MODE_RAISE:
        return AUDIT_SECRET_SCAN_MODE_RAISE
    return AUDIT_SECRET_SCAN_MODE_REDACT


def shadow_report_allowlist_enabled() -> bool:
    """Return whether the shadow report source state is allowlist-filtered."""
    return os.getenv(SHADOW_REPORT_ALLOWLIST_ENV_VAR, "true").strip().lower() in _TRUTHY


def redact_secret_material(value: Any) -> tuple[Any, dict[str, Any]]:
    """Return a redacted deep copy of ``value`` plus a redaction summary.

    String values containing a ``SECRET_MARKERS`` substring are replaced with
    ``[REDACTED:<marker-label>:sha256:<first-12-hex-of-sha256-of-original>]``
    so auditors can correlate incidents without seeing content. Mapping
    entries whose key looks secret are dropped and counted. The input is
    never mutated, and the redacted copy passes ``contains_secret_material``.
    """
    marker_counts: dict[str, int] = {}
    redacted_value_count = 0

    def _redact(item: Any) -> Any:
        nonlocal redacted_value_count
        if isinstance(item, Mapping):
            redacted: dict[str, Any] = {}
            for key, nested in item.items():
                key_marker = _first_secret_key_marker(str(key).lower())
                if key_marker is not None:
                    label = _secret_marker_label(key_marker)
                    marker_counts[label] = marker_counts.get(label, 0) + 1
                    redacted_value_count += 1
                    continue
                redacted[key] = _redact(nested)
            return redacted
        if isinstance(item, list):
            return [_redact(nested) for nested in item]
        if isinstance(item, str):
            lowered = item.lower()
            hit_markers = [marker for marker in SECRET_MARKERS if marker in lowered]
            if hit_markers:
                for marker in hit_markers:
                    label = _secret_marker_label(marker)
                    marker_counts[label] = marker_counts.get(label, 0) + 1
                redacted_value_count += 1
                return _redaction_placeholder(hit_markers[0], item)
        return item

    redacted_copy = _redact(value)
    summary = {
        "redaction_count": redacted_value_count,
        "redactions": [
            {"marker": label, "count": count}
            for label, count in sorted(marker_counts.items())
        ],
    }
    return redacted_copy, summary


def _redaction_placeholder(marker: str, original: str) -> str:
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
    return f"[REDACTED:{_secret_marker_label(marker)}:sha256:{digest}]"


_MARKER_LABEL_TRANSLATION = str.maketrans({"-": "_", "_": "-", ".": "-"})


def _secret_marker_label(marker: str) -> str:
    """Return a defanged marker label safe to embed in redacted output.

    Swapping ``-``/``_``/``.`` keeps labels readable while guaranteeing the
    placeholder never re-trips this module's or downstream verifier scans.
    """
    label = marker.translate(_MARKER_LABEL_TRANSLATION).strip("-")
    if not label or any(existing in label for existing in SECRET_MARKERS):
        return "marker-" + hashlib.sha256(marker.encode("utf-8")).hexdigest()[:8]
    return label


def _apply_secret_scan(source_state: Any, *, error_message: str) -> tuple[Any, dict[str, Any]]:
    mode = audit_secret_scan_mode()
    if mode == AUDIT_SECRET_SCAN_MODE_RAISE:
        if contains_secret_material(source_state):
            raise ValueError(error_message)
        return source_state, {"mode": mode, "redaction_count": 0, "redactions": []}
    redacted_state, summary = redact_secret_material(source_state)
    if summary["redaction_count"]:
        logger.warning(
            "%s; redacted %s value(s) (markers: %s)",
            error_message,
            summary["redaction_count"],
            ",".join(entry["marker"] for entry in summary["redactions"]),
        )
    return redacted_state, {"mode": mode, **summary}


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

    Unless ``RESEARCH_LAB_SHADOW_REPORT_ALLOWLIST`` is disabled, the source
    state rows are allowlist-filtered like the audit bundle rows so the
    public shadow report never exposes unfiltered internal rows.
    """
    if shadow_report_allowlist_enabled():
        source_state = {
            "epoch": int(epoch),
            "weight_input_snapshots": _stable_rows(_redact_rows(weight_input_snapshots, _SHADOW_WEIGHT_INPUT_FIELDS)),
            "ticket_rows": _stable_rows(_redact_rows(ticket_rows, _AUDIT_TICKET_FIELDS)),
            "queue_rows": _stable_rows(_redact_rows(queue_rows, _AUDIT_QUEUE_FIELDS)),
            "receipt_rows": _stable_rows(_redact_rows(receipt_rows, _AUDIT_RECEIPT_FIELDS)),
            "reimbursement_rows": _stable_rows(_redact_rows(reimbursement_rows, _SHADOW_REIMBURSEMENT_AWARD_FIELDS)),
        }
    else:
        source_state = {
            "epoch": int(epoch),
            "weight_input_snapshots": _stable_rows(weight_input_snapshots),
            "ticket_rows": _stable_rows(ticket_rows),
            "queue_rows": _stable_rows(queue_rows),
            "receipt_rows": _stable_rows(receipt_rows),
            "reimbursement_rows": _stable_rows(reimbursement_rows),
        }
    source_state, secret_scan = _apply_secret_scan(
        source_state,
        error_message="Research Lab shadow source state contains raw secret material",
    )

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
        "secret_scan": secret_scan,
        "weight_vector": weight_vector,
        "weight_vector_hash": weight_vector_hash,
        "observability": observability,
        "verifier_contract": {
            "required_checks": [
                "shadow_flags_enabled",
                "live_mutation_flags_false",
                "secret_payload_absent",
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
        "repo_commit_event_rows": _stable_rows(_redact_rows(private_repo_commit_event_rows, _AUDIT_PRIVATE_REPO_COMMIT_FIELDS)),
        "public_benchmark_report_rows": _stable_rows(_redact_rows(public_benchmark_report_rows, _AUDIT_PUBLIC_BENCHMARK_FIELDS)),
        "score_bundle_rows": _stable_rows(_redact_rows(score_bundle_rows, _AUDIT_SCORE_BUNDLE_FIELDS)),
    }
    source_state, secret_scan = _apply_secret_scan(
        source_state,
        error_message="Research Lab audit source state contains private or secret material",
    )
    source_state, value_commitments = _compact_oversized_audit_values(source_state)

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
        "secret_scan": secret_scan,
        "value_commitments": value_commitments,
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
            "repo_commit_event_count": len(private_repo_commit_event_rows),
            "public_benchmark_report_count": len(public_benchmark_report_rows),
            "score_bundle_count": len(score_bundle_rows),
        },
        "verifier_contract": {
            "required_checks": [
                "secret_payload_absent",
                "private_manifest_payload_absent",
                "candidate_patch_payload_absent",
                "private_image_reference_absent",
                "sealed_icp_payload_absent",
                "source_state_hash_matches",
                "score_bundle_hashes_match",
                "score_bundle_aggregates_recompute",
                "promotion_events_reference_score_bundles",
                "public_benchmark_reports_are_sanitized",
                "oversized_value_commitments_match_source_rows",
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
            if _looks_like_secret_key(lowered_key):
                return True
            if contains_secret_material(item):
                return True
    elif isinstance(value, list):
        return any(contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in SECRET_MARKERS)
    return False


def _looks_like_secret_key(lowered_key: str) -> bool:
    return _first_secret_key_marker(lowered_key) is not None


def _first_secret_key_marker(lowered_key: str) -> str | None:
    for marker in SECRET_KEY_MARKERS:
        if marker in lowered_key:
            return marker
    for marker in SECRET_TOKEN_KEY_MARKERS:
        if marker in lowered_key:
            return marker
    return None


def _stable_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in rows]
    return sorted(normalized, key=lambda row: canonical_json(row))


def _redact_rows(rows: Sequence[Mapping[str, Any]], allowed_fields: set[str]) -> list[dict[str, Any]]:
    redacted: list[dict[str, Any]] = []
    for row in rows:
        item = {
            _audit_output_key(key): _redact_value(value)
            for key, value in dict(row).items()
            if key in allowed_fields
        }
        redacted.append(item)
    return redacted


def _audit_output_key(key: str) -> str:
    if key == "private_repo_ref_hash":
        return "repo_ref_hash"
    return key


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


def _compact_oversized_audit_values(
    source_state: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace oversized row fields with deterministic content commitments.

    The append-only source tables remain authoritative. The signed audit row
    commits to large nested documents without duplicating hundreds of
    megabytes into one JSONB value, which would make PostgreSQL's safety check
    and insert exceed the statement timeout.
    """

    compacted: dict[str, Any] = {}
    field_stats: dict[tuple[str, str], dict[str, int]] = {}
    value_count = 0
    canonical_bytes_committed = 0
    for source_name, rows in source_state.items():
        if not isinstance(rows, list):
            compacted[str(source_name)] = rows
            continue
        compacted_rows: list[Any] = []
        for row in rows:
            if not isinstance(row, Mapping):
                compacted_rows.append(row)
                continue
            compacted_row: dict[str, Any] = {}
            for field_name, value in row.items():
                if (str(source_name), str(field_name)) in _AUDIT_COMPACTION_EXEMPT_FIELDS:
                    compacted_row[str(field_name)] = value
                    continue
                canonical_value = canonical_json(value).encode("utf-8")
                canonical_size = len(canonical_value)
                if canonical_size <= AUDIT_INLINE_VALUE_MAX_BYTES:
                    compacted_row[str(field_name)] = value
                    continue
                value_type = (
                    "object"
                    if isinstance(value, Mapping)
                    else "array"
                    if isinstance(value, list)
                    else "string"
                    if isinstance(value, str)
                    else type(value).__name__
                )
                compacted_row[str(field_name)] = {
                    "_audit_value_commitment": {
                        "schema_version": "research_lab_audit_value_commitment.v1",
                        "sha256": "sha256:" + hashlib.sha256(canonical_value).hexdigest(),
                        "canonical_bytes": canonical_size,
                        "value_type": value_type,
                    }
                }
                key = (str(source_name), str(field_name))
                stats = field_stats.setdefault(key, {"count": 0, "canonical_bytes": 0})
                stats["count"] += 1
                stats["canonical_bytes"] += canonical_size
                value_count += 1
                canonical_bytes_committed += canonical_size
            compacted_rows.append(compacted_row)
        compacted[str(source_name)] = compacted_rows
    fields = [
        {
            "source": source_name,
            "field": field_name,
            "count": stats["count"],
            "canonical_bytes": stats["canonical_bytes"],
        }
        for (source_name, field_name), stats in sorted(field_stats.items())
    ]
    return compacted, {
        "schema_version": "research_lab_audit_value_commitment_summary.v1",
        "inline_max_bytes": AUDIT_INLINE_VALUE_MAX_BYTES,
        "value_count": value_count,
        "canonical_bytes_committed": canonical_bytes_committed,
        "fields": fields,
    }


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

_SHADOW_WEIGHT_INPUT_FIELDS = {
    "weight_input_snapshot_id",
    "epoch",
    "netuid",
    "snapshot_status",
    "fulfillment_weight_ref",
    "leaderboard_weight_ref",
    "improvement_grant_ref",
    "reimbursement_weight_ref",
    "active_researcher_floor_ref",
    "source_bundle_ref",
    "input_state_hash",
    "weight_vector_hash",
    "snapshot_doc",
    "created_at",
}
_SHADOW_REIMBURSEMENT_AWARD_FIELDS = {
    "award_id",
    "receipt_id",
    "run_id",
    "miner_hotkey",
    "island",
    "run_day",
    "policy_id",
    "award_status",
    "participation_score",
    "participation_fraction",
    "rebate_rate",
    "eligible_cost_microusd",
    "target_reimbursement_microusd",
    "reimbursement_epochs",
    "loop_start_fee_included",
    "input_hash",
    "current_event_seq",
    "current_event_type",
    "current_award_status",
    "current_status_at",
    "created_at",
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
