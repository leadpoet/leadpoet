"""The paid E2E runner starts from current, guarded PostgreSQL authority."""

import pytest

from scripts.verify_arena_full_thinking_e2e import full_thinking_database
from tests.lab_arena.complete_baseline_final_authority_postgres_test import (
    test_normal_complete_baseline_publishes_and_activates_with_all_guards as _prove_round,
)


@pytest.fixture()
def database():
    yield from full_thinking_database()


def test_complete_mainline_schema_starts_and_publishes(database, tmp_path, monkeypatch):
    pg, dsn = database
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        expected = {
            "public.lab_arena__participant_incomplete_reason_v1(text,text)":
                "f0947c0d204d0883d926781cfe03ba22ad2b940ea16cf7618de90a9533cd44c4",
            "public.lab_arena_reserve_judgment_call(text,text,text,text,text,text,bigint,jsonb,integer)":
                "f73e8cd3839372e5bdb0cfefd2269e9bd1abcd578d0502cda14948e81acb843d",
            "public.lab_arena_miner_credit_failure_v1(text)":
                "23fbb6e5d40450916b75110d16d9b3c9c369264014dea51e17db7f3f978786c6",
            "public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)":
                "9762e92771dfa3333811caabeaec56d01b5c02c888869193c258660796d29bdd",
            "public.lab_arena_cancel_round(text,text)":
                "c984c3ef7cb6f8ab8970b152e44336bc7eb3135a9e31ebda8f00c331c7aa1700",
        }
        for signature, digest in expected.items():
            cursor.execute(
                "SELECT encode(extensions.digest(pg_get_functiondef(%s::regprocedure),"
                "'sha256'),'hex')", (signature,),
            )
            assert cursor.fetchone()[0] == digest, signature
        for index in (
            "lab_arena_ledger_paid_native_request_idx",
            "lab_arena_ledger_deepline_uncertain_round_entry_idx",
            "lab_arena_ledger_openrouter_uncertain_run_entry_idx",
            "lab_arena_ledger_round_call_head_idx",
        ):
            cursor.execute(
                "SELECT indisvalid,indisready FROM pg_index WHERE indexrelid="
                "to_regclass(%s)", ("public." + index,),
            )
            assert cursor.fetchone() == (True, True), index
        cursor.execute(
            "SELECT pg_get_functiondef('public.lab_arena__per_icp_publication_valid"
            "(text,jsonb)'::regprocedure)"
        )
        assert "lab_arena_training_publication_563" in cursor.fetchone()[0]

    # This uses the production service and transition RPCs against the exact
    # E2E schema. Providers are in-process test doubles, so it spends nothing.
    _prove_round(database, tmp_path, monkeypatch, False, True, 17)
