"""Tests for Research Lab bundle secret-scan redaction and shadow allowlist.

Covers the bug #36 fix (redact-and-log instead of raise, so one poisoned
historical row can no longer wedge every future audit bundle) and the P2
shadow report allowlist conversion in ``gateway/research_lab/bundles.py``.
"""

import copy
import hashlib
import json

import pytest

from gateway.research_lab.bundles import (
    AUDIT_INLINE_VALUE_MAX_BYTES,
    AUDIT_SECRET_SCAN_MODE_ENV_VAR,
    SECRET_MARKERS,
    SHADOW_REPORT_ALLOWLIST_ENV_VAR,
    audit_secret_scan_mode,
    build_research_lab_audit_bundle,
    build_shadow_report_bundle,
    contains_secret_material,
    redact_secret_material,
    sha256_json,
    shadow_report_allowlist_enabled,
)


POISONED_ECR_ERROR = (
    "pull access denied: 123456789.dkr.ecr.us-east-1.amazonaws.com/leadpoet-private@sha256:"
    + "a" * 64
)
POISONED_SUPABASE_ERROR = "permission denied for role service_role on table research_loop_receipts"


def _expected_placeholder(marker_label: str, original: str) -> str:
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
    return f"[REDACTED:{marker_label}:sha256:{digest}]"


def _fake_store_rows() -> dict[str, list[dict]]:
    """Row fixtures shaped like the scoring worker's full-history audit fetch,
    including poisoned historical rows with raw exception text (bug #36)."""
    return {
        "ticket_rows": [
            {
                "ticket_id": "11111111-1111-4111-8111-111111111111",
                "miner_hotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX",
                "island": "generalist",
                "requested_loop_count": 1,
                "ticket_hash": "sha256:" + "1" * 64,
                "ticket_doc": {"note": "clean ticket"},
                "current_ticket_status": "completed",
                "current_event_seq": 3,
                "current_reason": "",
                "created_at": "2026-06-01T00:00:00Z",
            }
        ],
        "queue_rows": [
            {
                "run_id": "22222222-2222-4222-8222-222222222222",
                "ticket_id": "11111111-1111-4111-8111-111111111111",
                "queue_priority": 10,
                "current_queue_status": "queued",
                "current_event_seq": 1,
                "current_reason": "",
                "current_status_at": "2026-06-01T01:00:00Z",
            }
        ],
        "receipt_rows": [
            {
                "receipt_id": "33333333-3333-4333-8333-333333333333",
                "ticket_id": "11111111-1111-4111-8111-111111111111",
                "run_id": "22222222-2222-4222-8222-222222222222",
                "miner_hotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX",
                "island": "generalist",
                "receipt_hash": "sha256:" + "2" * 64,
                "current_receipt_status": "completed",
                "current_event_seq": 2,
                "current_status_at": "2026-06-01T02:00:00Z",
                "provider_usage": [],
                "cost_ledger": {},
            }
        ],
        "candidate_rows": [],
        # Historical poisoned rows: raw str(exc) error text in audit-visible
        # event docs, exactly the live-confirmed 07-02 failure shape.
        "candidate_event_rows": [
            {
                "event_id": "44444444-4444-4444-8444-444444444444",
                "candidate_id": "55555555-5555-4555-8555-555555555555",
                "seq": 1,
                "event_type": "failed",
                "candidate_status": "failed",
                "reason": POISONED_SUPABASE_ERROR,
                "event_doc": {
                    "error": POISONED_ECR_ERROR,
                    "detail": "prompt leak: judge_prompt text should never appear",
                },
                "created_at": "2026-06-02T00:00:00Z",
            }
        ],
        "loop_event_rows": [
            {
                "event_id": "66666666-6666-4666-8666-666666666666",
                "run_id": "22222222-2222-4222-8222-222222222222",
                "seq": 1,
                "event_type": "failed",
                "loop_status": "failed",
                "provider_usage": [{"provider": "openrouter", "cost": 1}],
                "cost_ledger": {"total": 1},
                "event_doc": {
                    "errors": [
                        {"error": POISONED_ECR_ERROR},
                        {"error": "prompt leak: judge_prompt text should never appear"},
                    ]
                },
                "anchored_hash": "sha256:" + "6" * 64,
                "created_at": "2026-06-02T01:00:00Z",
            }
        ],
        "dispatch_event_rows": [
            {
                "dispatch_event_id": "77777777-7777-4777-8777-777777777777",
                "dispatch_type": "audit_bundle_build",
                "dispatch_status": "failed",
                "event_doc": {"error": POISONED_SUPABASE_ERROR},
                "created_at": "2026-06-02T02:00:00Z",
            }
        ],
        "rolling_window_rows": [],
        "benchmark_rows": [],
    }


def _build_audit_bundle_from_fake_store(rows: dict[str, list[dict]]) -> dict:
    return build_research_lab_audit_bundle(
        epoch=123,
        ticket_rows=rows["ticket_rows"],
        queue_rows=rows["queue_rows"],
        receipt_rows=rows["receipt_rows"],
        candidate_rows=rows["candidate_rows"],
        candidate_event_rows=rows["candidate_event_rows"],
        loop_event_rows=rows["loop_event_rows"],
        dispatch_event_rows=rows["dispatch_event_rows"],
        rolling_window_rows=rows["rolling_window_rows"],
        benchmark_rows=rows["benchmark_rows"],
    )


@pytest.fixture(autouse=True)
def _default_scan_env(monkeypatch):
    monkeypatch.delenv(AUDIT_SECRET_SCAN_MODE_ENV_VAR, raising=False)
    monkeypatch.delenv(SHADOW_REPORT_ALLOWLIST_ENV_VAR, raising=False)


def test_scan_mode_defaults_to_redact():
    assert audit_secret_scan_mode() == "redact"


def test_scan_mode_parses_raise_and_falls_back_to_redact(monkeypatch):
    monkeypatch.setenv(AUDIT_SECRET_SCAN_MODE_ENV_VAR, "RAISE")
    assert audit_secret_scan_mode() == "raise"
    monkeypatch.setenv(AUDIT_SECRET_SCAN_MODE_ENV_VAR, "garbage")
    assert audit_secret_scan_mode() == "redact"


def test_redact_secret_material_produces_deterministic_placeholders():
    source = {"error": POISONED_ECR_ERROR}
    redacted_first, summary_first = redact_secret_material(source)
    redacted_second, summary_second = redact_secret_material(source)
    assert redacted_first == redacted_second
    assert summary_first == summary_second
    assert redacted_first["error"] == _expected_placeholder("dkr-ecr", POISONED_ECR_ERROR)


def test_redact_secret_material_counts_markers():
    source = {
        "a": POISONED_ECR_ERROR,
        "b": POISONED_SUPABASE_ERROR,
        "c": [POISONED_SUPABASE_ERROR],
    }
    redacted, summary = redact_secret_material(source)
    assert summary["redaction_count"] == 3
    assert {entry["marker"]: entry["count"] for entry in summary["redactions"]} == {
        "dkr-ecr": 1,
        "service-role": 2,
    }
    assert redacted["b"] == _expected_placeholder("service-role", POISONED_SUPABASE_ERROR)


def test_redact_secret_material_handles_multiple_markers_in_one_value():
    combined = POISONED_SUPABASE_ERROR + " while pulling " + POISONED_ECR_ERROR
    redacted, summary = redact_secret_material({"error": combined})
    # One redacted value, both markers recorded; placeholder uses the first
    # matching marker in SECRET_MARKERS order (service_role before .dkr.ecr.).
    assert summary["redaction_count"] == 1
    assert {entry["marker"] for entry in summary["redactions"]} == {"dkr-ecr", "service-role"}
    assert redacted["error"] == _expected_placeholder("service-role", combined)


def test_redact_secret_material_traverses_nested_dicts_and_lists():
    source = {
        "outer": [
            {"inner": {"deep": [POISONED_ECR_ERROR, "clean"]}},
            "also clean",
        ]
    }
    redacted, summary = redact_secret_material(source)
    assert summary["redaction_count"] == 1
    assert redacted["outer"][0]["inner"]["deep"][0] == _expected_placeholder(
        "dkr-ecr", POISONED_ECR_ERROR
    )
    assert redacted["outer"][0]["inner"]["deep"][1] == "clean"
    assert redacted["outer"][1] == "also clean"


def test_redact_secret_material_drops_secret_looking_keys():
    source = {"event_doc": {"openrouter_api_key": "sk-live-whatever", "kept": "value"}}
    redacted, summary = redact_secret_material(source)
    assert "openrouter_api_key" not in redacted["event_doc"]
    assert redacted["event_doc"]["kept"] == "value"
    assert summary["redaction_count"] == 1
    assert {entry["marker"] for entry in summary["redactions"]} == {"api-key"}


def test_redact_secret_material_does_not_mutate_input():
    source = {
        "error": POISONED_ECR_ERROR,
        "nested": [{"reason": POISONED_SUPABASE_ERROR, "api_key": "raw"}],
    }
    snapshot = copy.deepcopy(source)
    redact_secret_material(source)
    assert source == snapshot


def test_redacted_output_passes_strict_scan():
    source = {
        "error": POISONED_ECR_ERROR,
        "reason": POISONED_SUPABASE_ERROR,
        "event_doc": {"openrouter_api_key": "x", "list": [POISONED_ECR_ERROR]},
    }
    redacted, _summary = redact_secret_material(source)
    assert contains_secret_material(redacted) is False


def test_redaction_placeholders_never_retrip_scan_for_any_marker():
    for marker in SECRET_MARKERS:
        poisoned = f"prefix {marker} suffix"
        redacted, summary = redact_secret_material({"value": poisoned})
        assert summary["redaction_count"] == 1, marker
        assert contains_secret_material(redacted) is False, marker
        assert contains_secret_material(summary) is False, marker


def test_audit_bundle_builds_over_poisoned_fake_store_rows():
    rows = _fake_store_rows()
    snapshot = copy.deepcopy(rows)
    bundle = _build_audit_bundle_from_fake_store(rows)

    # Whole-bundle build no longer wedges on one poisoned historical row.
    assert bundle["bundle_type"] == "research_lab_signed_audit_bundle"
    assert bundle["secret_scan"]["mode"] == "redact"
    assert bundle["secret_scan"]["redaction_count"] >= 4
    markers_hit = {entry["marker"] for entry in bundle["secret_scan"]["redactions"]}
    assert {"dkr-ecr", "service-role", "judge-prompt"} <= markers_hit

    # Placeholders landed where the raw error text used to be.
    candidate_events = bundle["source_state"]["candidate_event_rows"]
    assert candidate_events[0]["event_doc"]["error"] == _expected_placeholder(
        "dkr-ecr", POISONED_ECR_ERROR
    )
    assert candidate_events[0]["reason"] == _expected_placeholder(
        "service-role", POISONED_SUPABASE_ERROR
    )

    # The hash is computed over the redacted content and the bundle is clean.
    assert bundle["source_state_hash"] == sha256_json(bundle["source_state"])
    assert contains_secret_material(bundle) is False

    # Rows destined for storage elsewhere are never mutated.
    assert rows == snapshot


def test_audit_bundle_raise_mode_still_raises(monkeypatch):
    monkeypatch.setenv(AUDIT_SECRET_SCAN_MODE_ENV_VAR, "raise")
    with pytest.raises(ValueError, match="private or secret material"):
        _build_audit_bundle_from_fake_store(_fake_store_rows())


def test_audit_bundle_raise_mode_passes_clean_rows(monkeypatch):
    monkeypatch.setenv(AUDIT_SECRET_SCAN_MODE_ENV_VAR, "raise")
    rows = _fake_store_rows()
    rows["candidate_event_rows"] = []
    rows["loop_event_rows"] = []
    rows["dispatch_event_rows"] = []
    bundle = _build_audit_bundle_from_fake_store(rows)
    assert bundle["secret_scan"] == {"mode": "raise", "redaction_count": 0, "redactions": []}
    assert bundle["source_state_hash"] == sha256_json(bundle["source_state"])


def test_audit_bundle_clean_rows_report_zero_redactions():
    rows = _fake_store_rows()
    rows["candidate_event_rows"] = []
    rows["loop_event_rows"] = []
    rows["dispatch_event_rows"] = []
    bundle = _build_audit_bundle_from_fake_store(rows)
    assert bundle["secret_scan"] == {"mode": "redact", "redaction_count": 0, "redactions": []}


def test_audit_bundle_commits_oversized_nested_values_without_mutating_source():
    rows = _fake_store_rows()
    oversized_doc = {"trace": "x" * (AUDIT_INLINE_VALUE_MAX_BYTES + 1024)}
    rows["candidate_event_rows"][0]["event_doc"] = oversized_doc
    snapshot = copy.deepcopy(rows)

    bundle = _build_audit_bundle_from_fake_store(rows)

    stored = bundle["source_state"]["candidate_event_rows"][0]["event_doc"]
    commitment = stored["_audit_value_commitment"]
    assert commitment == {
        "schema_version": "research_lab_audit_value_commitment.v1",
        "sha256": sha256_json(oversized_doc),
        "canonical_bytes": len(
            json.dumps(
                oversized_doc,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ),
        "value_type": "object",
    }
    summary = bundle["value_commitments"]
    assert summary["value_count"] == 1
    assert summary["canonical_bytes_committed"] == commitment["canonical_bytes"]
    assert summary["fields"] == [
        {
            "source": "candidate_event_rows",
            "field": "event_doc",
            "count": 1,
            "canonical_bytes": commitment["canonical_bytes"],
        }
    ]
    assert bundle["source_state_hash"] == sha256_json(bundle["source_state"])
    assert rows == snapshot


def test_audit_bundle_loop_events_use_anchored_refs_not_duplicate_trace_docs():
    rows = _fake_store_rows()

    bundle = _build_audit_bundle_from_fake_store(rows)

    stored = bundle["source_state"]["auto_research_loop_event_rows"][0]
    assert stored["anchored_hash"] == "sha256:" + "6" * 64
    assert "event_doc" not in stored
    assert "provider_usage" not in stored
    assert "cost_ledger" not in stored


def test_audit_bundle_keeps_score_bundle_docs_inline_for_validator_recompute():
    rows = _fake_store_rows()
    score_bundle_doc = {"payload": "x" * (AUDIT_INLINE_VALUE_MAX_BYTES + 1024)}

    bundle = build_research_lab_audit_bundle(
        epoch=123,
        ticket_rows=rows["ticket_rows"],
        queue_rows=rows["queue_rows"],
        receipt_rows=rows["receipt_rows"],
        candidate_rows=rows["candidate_rows"],
        candidate_event_rows=rows["candidate_event_rows"],
        loop_event_rows=rows["loop_event_rows"],
        dispatch_event_rows=rows["dispatch_event_rows"],
        rolling_window_rows=rows["rolling_window_rows"],
        benchmark_rows=rows["benchmark_rows"],
        score_bundle_rows=[{"score_bundle_doc": score_bundle_doc}],
    )

    assert bundle["source_state"]["score_bundle_rows"][0]["score_bundle_doc"] == score_bundle_doc


def _shadow_rows() -> dict[str, list[dict]]:
    return {
        "weight_input_snapshots": [
            {
                "weight_input_snapshot_id": "11111111-1111-4111-8111-111111111111",
                "epoch": 123,
                "netuid": 71,
                "snapshot_status": "shadow",
                "source_bundle_ref": "bundle:abc",
                "input_state_hash": "sha256:" + "3" * 64,
                "weight_vector_hash": None,
                "snapshot_doc": {},
                "created_at": "2026-06-01T00:00:00Z",
            }
        ],
        "ticket_rows": [
            {
                "ticket_id": "11111111-1111-4111-8111-111111111111",
                "miner_hotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX",
                "island": "generalist",
                "requested_loop_count": 1,
                "ticket_hash": "sha256:" + "1" * 64,
                "ticket_doc": {"note": "clean"},
                "current_ticket_status": "completed",
                "current_event_seq": 3,
                "current_reason": "",
                "created_at": "2026-06-01T00:00:00Z",
                # Internal fields that must never reach the public report.
                "miner_openrouter_key_ref": "vault:ref:123",
                "brief_sanitized_ref": "internal-brief-ref",
                "loop_start_fee_payment_ref": "payment-ref",
                "current_actor_hotkey": "internal-actor",
            }
        ],
        "queue_rows": [
            {
                "run_id": "22222222-2222-4222-8222-222222222222",
                "ticket_id": "11111111-1111-4111-8111-111111111111",
                "queue_priority": 10,
                "current_queue_status": "queued",
                "current_event_seq": 1,
                "current_reason": "",
                "current_status_at": "2026-06-01T01:00:00Z",
                "worker_ref": "internal-worker-ref",
                "current_event_hash": "internal-hash",
            }
        ],
        "receipt_rows": [
            {
                "receipt_id": "33333333-3333-4333-8333-333333333333",
                "ticket_id": "11111111-1111-4111-8111-111111111111",
                "run_id": "22222222-2222-4222-8222-222222222222",
                "miner_hotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX",
                "island": "generalist",
                "receipt_hash": "sha256:" + "2" * 64,
                "current_receipt_status": "completed",
                "current_event_seq": 2,
                "current_status_at": "2026-06-01T02:00:00Z",
                "provider_usage": [],
                "cost_ledger": {},
                "receipt_doc": {"internal": "doc"},
                "loop_start_payment_id": "44444444-4444-4444-8444-444444444444",
            }
        ],
        "reimbursement_rows": [
            {
                "award_id": "award-1",
                "run_id": "run-1",
                "miner_hotkey": "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX",
                "island": "generalist",
                "run_day": "2026-06-01",
                "policy_id": "policy-1",
                "award_status": "awarded",
                "current_award_status": "awarded",
                "participation_score": 1.0,
                "participation_fraction": 0.5,
                "rebate_rate": 0.1,
                "eligible_cost_microusd": 1000,
                "target_reimbursement_microusd": 100,
                "reimbursement_epochs": 3,
                "loop_start_fee_included": False,
                "input_hash": "sha256:" + "4" * 64,
                "created_at": "2026-06-01T03:00:00Z",
                "award_doc": {"internal": "doc"},
                "participation_snapshot_id": "55555555-5555-4555-8555-555555555555",
            }
        ],
    }


def _build_shadow_bundle(rows: dict[str, list[dict]]) -> dict:
    return build_shadow_report_bundle(
        epoch=123,
        weight_input_snapshots=rows["weight_input_snapshots"],
        ticket_rows=rows["ticket_rows"],
        queue_rows=rows["queue_rows"],
        receipt_rows=rows["receipt_rows"],
        reimbursement_rows=rows["reimbursement_rows"],
    )


def test_shadow_report_allowlist_enabled_by_default():
    assert shadow_report_allowlist_enabled() is True


def test_shadow_report_exposes_only_allowlisted_fields():
    bundle = _build_shadow_bundle(_shadow_rows())
    source_state = bundle["source_state"]

    ticket = source_state["ticket_rows"][0]
    assert set(ticket) <= {
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
    assert "miner_openrouter_key_ref" not in ticket
    assert "brief_sanitized_ref" not in ticket
    assert "current_actor_hotkey" not in ticket

    queue = source_state["queue_rows"][0]
    assert "worker_ref" not in queue
    assert "current_event_hash" not in queue

    receipt = source_state["receipt_rows"][0]
    assert "receipt_doc" not in receipt
    assert "loop_start_payment_id" not in receipt

    award = source_state["reimbursement_rows"][0]
    assert "award_doc" not in award
    assert "participation_snapshot_id" not in award
    assert award["award_status"] == "awarded"

    snapshot = source_state["weight_input_snapshots"][0]
    assert snapshot["snapshot_status"] == "shadow"
    assert bundle["source_state_hash"] == sha256_json(source_state)
    assert bundle["secret_scan"]["mode"] == "redact"


def test_shadow_report_legacy_rows_when_allowlist_disabled(monkeypatch):
    monkeypatch.setenv(SHADOW_REPORT_ALLOWLIST_ENV_VAR, "false")
    bundle = _build_shadow_bundle(_shadow_rows())
    ticket = bundle["source_state"]["ticket_rows"][0]
    assert ticket["miner_openrouter_key_ref"] == "vault:ref:123"
    assert ticket["brief_sanitized_ref"] == "internal-brief-ref"


def test_shadow_report_redacts_poisoned_ticket_doc_instead_of_raising():
    rows = _shadow_rows()
    rows["ticket_rows"][0]["ticket_doc"] = {"error": POISONED_ECR_ERROR}
    snapshot = copy.deepcopy(rows)
    bundle = _build_shadow_bundle(rows)
    ticket = bundle["source_state"]["ticket_rows"][0]
    assert ticket["ticket_doc"]["error"] == _expected_placeholder("dkr-ecr", POISONED_ECR_ERROR)
    assert bundle["secret_scan"]["redaction_count"] == 1
    assert contains_secret_material(bundle) is False
    assert rows == snapshot


def test_shadow_report_raise_mode_still_raises(monkeypatch):
    monkeypatch.setenv(AUDIT_SECRET_SCAN_MODE_ENV_VAR, "raise")
    rows = _shadow_rows()
    rows["ticket_rows"][0]["ticket_doc"] = {"error": POISONED_SUPABASE_ERROR}
    with pytest.raises(ValueError, match="raw secret material"):
        _build_shadow_bundle(rows)
