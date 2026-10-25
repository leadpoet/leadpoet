"""Persist every typed review error and exercise digest failures without spend."""

import hashlib
from unittest.mock import Mock

import pytest

from lab_arena import code_review, code_review_policy, submission_similarity
from lab_arena.code_review_runtime import SubmissionCodeReviewer
from lab_arena.store import ArenaStoreError, new_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.review_context_integrity562_postgres_test import (
    SIGNATURE, _claimed, context,
)
from tests.lab_arena.submission_similarity_runtime_test import archive
from tests.lab_arena.test_lab_arena_code_review_postgres import accepted_recovery_submission
from tests.lab_arena.test_lab_arena_code_review_runtime import Objects, _price_table
from tests.postgres_migration_harness import SCRIPTS


MIGRATION = SCRIPTS / "563-lab-arena-review-source-digest-reason.sql"
PREIMAGE = "f8b5171ec7584721ae879a66e450136967e63a91e7a80c03ea9fb97e2d15c1d8"
POSTIMAGE = "573ee0b15a4952e4d25e2943c79578e44993c5e8117d2fdcc2a208fbb623cba1"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS + (
        "492-lab-arena-review-provider-observations.sql",
        "562-lab-arena-review-context-integrity-reason.sql",
        MIGRATION.name,
    ))


def _document(code):
    diagnostic = (
        code_review_policy.source_diagnostic(code)
        if code in code_review_policy.SOURCE_ERROR_CODES
        else code_review_policy.diagnostic(code)
    )
    return {**diagnostic, "model": code_review.DEFAULT_REVIEW_MODEL,
            "file_count": 71, "source_bytes": 2_118_944}


def test_digest_error_fails_before_migration_then_settles_same_claim(context):
    store, connection = context
    submission, miner, token = _claimed(store, "digestbeforeafter")
    document = _document("code_review_source_digest_mismatch")
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
        original = cursor.fetchone()[0]
        assert hashlib.sha256(original.encode()).hexdigest() == POSTIMAGE
        predecessor = original.replace(
            "'code_review_source_size_mismatch',\n         'code_review_source_digest_mismatch',",
            "'code_review_source_size_mismatch',",
        )
        assert hashlib.sha256(predecessor.encode()).hexdigest() == PREIMAGE
        cursor.execute(predecessor)
    try:
        with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
            store.finish_submission_review(submission, miner, token, "error", document, 0)
        assert store.get_submission(submission)["code_review_status"] == "reviewing"
        assert len(store.list_ledger(submission_id=submission)) == 2
    finally:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
    assert store.finish_submission_review(
        submission, miner, token, "error", document, 0,
    )["status"] == "error"
    row = store.get_submission(submission)
    assert row["code_review_expires_at"] is None
    assert row["code_review_doc"]["retryable"] is False
    assert row["code_review_doc"]["cost_microusd"] == 0


@pytest.mark.parametrize("code", sorted(
    code for code in code_review_policy.ERROR_CODES
    if code != "code_review_research_unverified"
))
def test_every_canonical_error_code_persists_through_real_rpc(context, code):
    store, _ = context
    assert len(code_review_policy.ERROR_CODES - {"code_review_research_unverified"}) == 45
    submission, miner, token = _claimed(store, code)
    document = _document(code)
    assert store.finish_submission_review(
        submission, miner, token, "error", document, 0,
    )["status"] == "error"
    row = store.get_submission(submission)
    assert row["code_review_doc"]["error_code"] == code
    assert row["code_review_expires_at"] is None
    assert row["code_review_doc"]["cost_microusd"] == 0
    if code in code_review_policy.SOURCE_ERROR_CODES:
        assert row["code_review_doc"]["retryable"] is False


@pytest.mark.parametrize("field,suffix", [
    ("source_archive_sha256", "d00"), ("source_normalized_sha256", "d01"),
])
def test_saved_digest_mismatch_is_permanent_zero_cost_without_provider_call(context, field, suffix):
    store, connection = context
    _, submission, miner = accepted_recovery_submission(store, suffix)
    payload = archive("def run(icp):\n    return icp.industry\n")
    identity = submission_similarity.inspect_archive(payload)
    digests = {"source_archive_sha256": identity.archive_sha256,
               "source_normalized_sha256": identity.normalized_sha256}
    digests[field] = "sha256:" + "0" * 64
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_submissions SET source_size_bytes=%s,"
            "source_archive_sha256=%s,source_normalized_sha256=%s WHERE submission_id=%s",
            (len(payload), digests["source_archive_sha256"],
             digests["source_normalized_sha256"], submission),
        )
    row = store.get_submission(submission)
    objects = Objects(payload)
    transport = Mock()
    credential_for = Mock(side_effect=AssertionError("source failure cannot read credentials"))
    reviewer = SubmissionCodeReviewer(
        store=store, objects=objects, credential_for=credential_for,
        price_table=_price_table(), transport=transport,
    )
    result = reviewer.review(row)
    assert result["status"] == "error"
    assert result["ledger_status"] == "settled"
    assert result["actual_microusd"] == 0
    terminal = store.get_submission(submission)
    assert terminal["code_review_status"] == "error"
    assert terminal["code_review_attempts"] == 1
    assert terminal["code_review_expires_at"] is None
    # Preserve the claim binding for the existing idempotent completion path.
    assert terminal["code_review_claim"] is not None
    assert terminal["code_review_doc"]["error_code"] == "code_review_source_digest_mismatch"
    assert terminal["code_review_doc"]["retryable"] is False
    assert terminal["code_review_doc"]["cost_microusd"] == 0
    ledger = store.list_ledger(submission_id=submission)
    assert [entry["entry_kind"] for entry in ledger] == ["reservation", "dispatch", "settlement"]
    assert all(entry["amount_microusd"] == 0 for entry in ledger)
    assert ledger[-1]["entry_doc"]["error_code"] == "code_review_source_digest_mismatch"
    assert ledger[-1]["entry_doc"]["retryable"] is False
    assert reviewer.review(terminal)["status"] == "exhausted"
    assert store.begin_submission_review(
        submission, miner, new_lease_token(), 0, code_review.DEFAULT_REVIEW_MODEL, 2, 1,
    )["status"] == "exhausted"
    assert len(store.list_ledger(submission_id=submission)) == 3
    assert len(objects.calls) == 1
    credential_for.assert_not_called()
    transport.send.assert_not_called()


def test_unknown_error_code_cannot_settle_or_clear_the_claim(context):
    store, _ = context
    submission, miner, token = _claimed(store, "unknowncode")
    document = {**_document("review_source_invalid"), "error_code": "source_digest_mismatch"}
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        store.finish_submission_review(submission, miner, token, "error", document, 0)
    row = store.get_submission(submission)
    assert row["code_review_status"] == "reviewing"
    assert row["code_review_expires_at"] is not None
    assert len(store.list_ledger(submission_id=submission)) == 2


def test_migration_replay_preserves_metadata_and_rejects_an_unknown_preimage(context):
    _, connection = context
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p WHERE p.oid=%s::regprocedure", (SIGNATURE,))
        before = cursor.fetchone()[0]
        cursor.execute(MIGRATION.read_text())
        cursor.execute(MIGRATION.read_text())
        cursor.execute("SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p WHERE p.oid=%s::regprocedure", (SIGNATURE,))
        assert cursor.fetchone()[0] == before
        cursor.execute("SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE'),has_function_privilege('service_role',%s,'EXECUTE'),has_function_privilege('anon',%s,'EXECUTE'),has_function_privilege('authenticated',%s,'EXECUTE')", (SIGNATURE,) * 4)
        assert cursor.fetchone() == (True, False, False, False)
        cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
        original = cursor.fetchone()[0]
        assert hashlib.sha256(original.encode()).hexdigest() == POSTIMAGE
        changed = original.replace("BEGIN\n", "BEGIN\n  -- unknown fixture change\n", 1)
        try:
            cursor.execute(changed)
            with pytest.raises(Exception, match="Arena review563 preimage differs"):
                cursor.execute(MIGRATION.read_text())
            cursor.execute("ROLLBACK")
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
            assert cursor.fetchone()[0] == changed
        finally:
            cursor.execute("ROLLBACK")
            cursor.execute(original)
