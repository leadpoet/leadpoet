"""Disposable PostgreSQL proof for source-bound, miner-billed review metadata."""

from __future__ import annotations

import json

import pytest

from lab_arena import code_review, code_review_policy
from lab_arena.store import ArenaStoreError, new_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    accepted_recovery_submission, recovery_connect, recovery_store,
    recovery_superuser,
)
from tests.postgres_migration_harness import SCRIPTS


MIGRATION = SCRIPTS / "565-lab-arena-research-review-result.sql"
SIGNATURE = "public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)"
ARCHIVE_HASH = "sha256:" + "a" * 64
MANIFEST_HASH = "sha256:" + "b" * 64


@pytest.fixture(scope="module")
def recovery_database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + (
            "492-lab-arena-review-provider-observations.sql",
            "562-lab-arena-review-context-integrity-reason.sql",
            "563-lab-arena-review-source-digest-reason.sql", MIGRATION.name,
        )
    )


def _claim(store, superuser, suffix, *, active=True):
    round_id, submission_id, miner = accepted_recovery_submission(store, suffix)
    with superuser.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_submissions SET "
            "source_archive_sha256=%s,source_normalized_sha256=%s "
            "WHERE submission_id=%s",
            (ARCHIVE_HASH, MANIFEST_HASH, submission_id),
        )
        if active:
            # The fixture's minimal historical round config cannot pass the
            # active-policy creation contract. Set only this frozen selector in
            # the disposable database, then restore ordinary trigger behavior.
            cursor.execute("SET session_replication_role='replica'")
            try:
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET configuration_doc="
                    "configuration_doc||jsonb_build_object("
                    "'reasoning_capture_policy','full_generated_thinking_v1') "
                    "WHERE round_id=%s",
                    (round_id,),
                )
            finally:
                cursor.execute("SET session_replication_role='origin'")
    token = new_lease_token()
    assert store.begin_submission_review(
        submission_id, miner, token, 50_000,
        code_review.DEFAULT_REVIEW_MODEL, 3, 1200,
    )["status"] == "claimed"
    return submission_id, miner, token


def _research(classification="model_directed"):
    claims = {
        "model_directed": ["finish_continue", "observation_to_action"],
        "workflow": ["fixed_control"],
        "unverified": ["uncertainty"],
    }[classification]
    return {
        "classification": classification,
        "policy_id": code_review.RESEARCH_POLICY_ID,
        "policy_hash": code_review.RESEARCH_POLICY_HASH,
        "reviewer_version": code_review.RESEARCH_REVIEWER_VERSION,
        "source_archive_sha256": ARCHIVE_HASH,
        "source_manifest_sha256": MANIFEST_HASH,
        "evidence_claims": claims,
        "evidence_count": len(claims),
    }


def _document(classification="model_directed", *, status="passed"):
    document = {
        "model": code_review.DEFAULT_REVIEW_MODEL,
        "file_count": 3,
        "source_bytes": 1200,
        "passed": status == "passed",
        "verdict": "pass" if status == "passed" else "reject",
        "categories": [],
        "research_control": _research(classification),
    }
    if status == "error":
        document["error_code"] = "code_review_research_unverified"
        document["retryable"] = True
    return document


def test_active_pass_persists_only_safe_binding_and_settles_once(recovery_store, recovery_superuser):
    submission, miner, token = _claim(recovery_store, recovery_superuser, "e10")
    document = {**_document(), "provider_prose": "private prose"}
    assert recovery_store.finish_submission_review(
        submission, miner, token, "passed", document, 12_345,
    )["status"] == "passed"
    row = recovery_store.get_submission(submission)
    assert row["code_review_doc"]["research_control"] == _research()
    assert row["code_review_doc"]["cost_status"] == "settled"
    assert row["code_review_doc"]["cost_microusd"] == 12_345
    assert "private prose" not in json.dumps(row["code_review_doc"])
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert [entry["entry_kind"] for entry in ledger] == [
        "reservation", "dispatch", "settlement",
    ]
    assert {entry["funding_source"] for entry in ledger} == {"miner_key"}
    assert ledger[-1]["amount_microusd"] == 12_345
    assert "research_control" not in ledger[-1]["entry_doc"]
    assert recovery_store.finish_submission_review(
        submission, miner, token, "passed", document, 12_345,
    )["status"] == "existing"
    assert len(recovery_store.list_ledger(submission_id=submission)) == 3


def test_unverified_assessment_finishes_and_settles(recovery_store, recovery_superuser):
    submission, miner, token = _claim(recovery_store, recovery_superuser, "e11")
    assert recovery_store.finish_submission_review(
        submission, miner, token, "error", _document("unverified", status="error"), 1_000,
    )["status"] == "error"
    row = recovery_store.get_submission(submission)
    assert row["code_review_status"] == "error"
    assert row["code_review_doc"]["error_code"] == "code_review_research_unverified"
    assert row["code_review_doc"]["retryable"] is True
    assert row["code_review_doc"]["research_control"]["classification"] == "unverified"
    assert row["code_review_doc"]["cost_status"] == "settled"
    assert recovery_store.list_ledger(submission_id=submission)[-1]["amount_microusd"] == 1_000
    assert code_review_policy.review_retry_available(row)
    assert code_review_policy.retry_ready_at(row) is not None

    assert recovery_store.begin_submission_review(
        submission, miner, new_lease_token(), 50_000,
        code_review.DEFAULT_REVIEW_MODEL, 3, 1200,
    )["status"] == "backoff"
    with recovery_superuser.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_submissions SET "
            "code_review_started_at=clock_timestamp() - interval '61 seconds' "
            "WHERE submission_id=%s",
            (submission,),
        )
    claimed = recovery_store.begin_submission_review(
        submission, miner, new_lease_token(), 50_000,
        code_review.DEFAULT_REVIEW_MODEL, 3, 1200,
    )
    assert claimed["status"] == "claimed"
    assert claimed["attempt"] == 2
    retried = recovery_store.get_submission(submission)
    assert retried["code_review_status"] == "reviewing"
    assert retried["code_review_attempts"] == 2
    assert [entry["entry_kind"] for entry in recovery_store.list_ledger(submission_id=submission)] == [
        "reservation", "dispatch", "settlement", "reservation", "dispatch",
    ]


def test_main_digest_failure_remains_permanent_after_research_migration(
    recovery_store, recovery_superuser,
):
    submission, miner, token = _claim(recovery_store, recovery_superuser, "e25")
    document = {
        **code_review_policy.source_diagnostic("code_review_source_digest_mismatch"),
        "model": code_review.DEFAULT_REVIEW_MODEL,
        "file_count": 3,
        "source_bytes": 1200,
    }
    assert recovery_store.finish_submission_review(
        submission, miner, token, "error", document, 0,
    )["status"] == "error"
    row = recovery_store.get_submission(submission)
    assert row["code_review_doc"]["error_code"] == "code_review_source_digest_mismatch"
    assert row["code_review_doc"]["retryable"] is False
    assert row["code_review_doc"]["cost_microusd"] == 0
    assert not code_review_policy.review_retry_available(row)
    assert [entry["entry_kind"] for entry in recovery_store.list_ledger(submission_id=submission)] == [
        "reservation", "dispatch", "settlement",
    ]


@pytest.mark.parametrize("suffix,error_code", [
    ("e28", "code_review_provider_authentication"),
    ("e29", "unknown_error_code"),
])
def test_retryable_research_error_does_not_admit_other_codes(
    recovery_store, recovery_superuser, suffix, error_code,
):
    submission, miner, token = _claim(recovery_store, recovery_superuser, suffix)
    document = _document("unverified", status="error")
    document["error_code"] = error_code
    document.pop("research_control")
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        recovery_store.finish_submission_review(
            submission, miner, token, "error", document, 1_000,
        )
    assert recovery_store.get_submission(submission)["code_review_status"] == "reviewing"
    assert [entry["entry_kind"] for entry in recovery_store.list_ledger(submission_id=submission)] == [
        "reservation", "dispatch",
    ]


def test_workflow_rejection_is_billed_without_a_misconduct_category(recovery_store, recovery_superuser):
    submission, miner, token = _claim(recovery_store, recovery_superuser, "e17")
    assert recovery_store.finish_submission_review(
        submission, miner, token, "rejected", _document("workflow", status="rejected"), 650,
    )["status"] == "rejected"
    row = recovery_store.get_submission(submission)
    assert row["code_review_doc"]["research_control"]["classification"] == "workflow"
    assert row["code_review_doc"]["categories"] == []
    assert recovery_store.list_ledger(submission_id=submission)[-1]["amount_microusd"] == 650


@pytest.mark.parametrize("category,actual", [
    ("duplicate_submission", 0),
    ("malicious_behavior", 650),
])
def test_independent_rejection_without_research_assessment_settles(
    recovery_store, recovery_superuser, category, actual,
):
    submission, miner, token = _claim(
        recovery_store, recovery_superuser, "e23" if actual == 0 else "e24",
    )
    document = _document(status="rejected")
    document.pop("research_control")
    document["categories"] = [category]

    assert recovery_store.finish_submission_review(
        submission, miner, token, "rejected", document, actual,
    )["status"] == "rejected"
    row = recovery_store.get_submission(submission)
    assert row["code_review_status"] == "rejected"
    assert row["code_review_doc"]["categories"] == [category]
    assert "research_control" not in row["code_review_doc"]
    assert row["code_review_doc"]["cost_microusd"] == actual
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert ledger[-1]["entry_kind"] == "settlement"
    assert ledger[-1]["amount_microusd"] == actual
    assert recovery_store.finish_submission_review(
        submission, miner, token, "rejected", document, actual,
    )["status"] == "existing"
    assert len(recovery_store.list_ledger(submission_id=submission)) == len(ledger)


@pytest.mark.parametrize("index,change", list(enumerate((
    lambda doc: doc["research_control"].update(source_archive_sha256="sha256:" + "c" * 64),
    lambda doc: doc["research_control"].update(evidence_claims=["fixed_control"]),
    lambda doc: doc["research_control"].update(excerpt="source text"),
    lambda doc: doc["research_control"].update(evidence_count=17),
    lambda doc: doc.pop("research_control"),
), start=18)))
def test_invalid_active_metadata_cannot_settle_or_pass(recovery_store, recovery_superuser, index, change):
    submission, miner, token = _claim(recovery_store, recovery_superuser, f"e{index:02x}")
    document = _document()
    change(document)
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        recovery_store.finish_submission_review(submission, miner, token, "passed", document, 10)
    assert recovery_store.get_submission(submission)["code_review_status"] == "reviewing"
    assert [entry["entry_kind"] for entry in recovery_store.list_ledger(submission_id=submission)] == [
        "reservation", "dispatch",
    ]


def test_legacy_round_rejects_research_field_but_keeps_old_pass(recovery_store, recovery_superuser):
    submission, miner, token = _claim(recovery_store, recovery_superuser, "e20", active=False)
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        recovery_store.finish_submission_review(submission, miner, token, "passed", _document(), 20)
    legacy = _document()
    legacy.pop("research_control")
    assert recovery_store.finish_submission_review(
        submission, miner, token, "passed", legacy, 20,
    )["status"] == "passed"
    assert "research_control" not in recovery_store.get_submission(submission)["code_review_doc"]


@pytest.mark.parametrize("suffix,reason", [
    ("e21", "context_integrity"),
    ("e1b", "research_binding"),
    ("e1c", "research_classification"),
    ("e1d", "research_evidence"),
    ("e1e", "research_evidence_mismatch"),
    ("e1f", "research_keys"),
    ("e22", "research_limitations"),
])
def test_new_parser_reason_settles_existing_failure_contract(
    recovery_store, recovery_superuser, suffix, reason,
):
    submission, miner, token = _claim(recovery_store, recovery_superuser, suffix)
    document = {
        "model": code_review.DEFAULT_REVIEW_MODEL,
        "file_count": 3,
        "source_bytes": 1200,
        "error_code": "review_response_invalid",
        "error_reason": reason,
    }
    assert recovery_store.finish_submission_review(
        submission, miner, token, "error", document, None,
    )["status"] == "error"
    row = recovery_store.get_submission(submission)
    assert row["code_review_doc"]["error_reason"] == reason
    assert row["code_review_doc"]["cost_status"] == "uncertain"
    assert recovery_store.list_ledger(submission_id=submission)[-1]["entry_kind"] == "uncertain"


def test_migration_replay_preserves_function_identity_and_security(recovery_superuser):
    with recovery_superuser.cursor() as cursor:
        cursor.execute("SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p WHERE p.oid=%s::regprocedure", (SIGNATURE,))
        before = cursor.fetchone()[0]
        cursor.execute(MIGRATION.read_text())
        cursor.execute(MIGRATION.read_text())
        cursor.execute("SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p WHERE p.oid=%s::regprocedure", (SIGNATURE,))
        assert cursor.fetchone()[0] == before
        cursor.execute(
            "SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE'),"
            "has_function_privilege('service_role',%s,'EXECUTE'),"
            "has_function_privilege('anon',%s,'EXECUTE')",
            (SIGNATURE,) * 3,
        )
        assert cursor.fetchone() == (True, False, False)


def test_migration_refuses_unreviewed_function_definition(recovery_superuser):
    with recovery_superuser.cursor() as cursor:
        cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
        original = cursor.fetchone()[0]
        changed = original.replace("BEGIN\n", "BEGIN\n  -- unrelated fixture change\n", 1)
        assert changed != original
        try:
            cursor.execute(changed)
            with pytest.raises(Exception, match="Arena review556 preimage differs") as caught:
                cursor.execute(MIGRATION.read_text())
            assert caught.value.pgcode == "55000"
            cursor.execute("ROLLBACK")
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
            assert cursor.fetchone()[0] == changed
        finally:
            cursor.execute("ROLLBACK")
            cursor.execute(original)
