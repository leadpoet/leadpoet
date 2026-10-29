"""The paid E2E runner starts from current, guarded PostgreSQL authority."""

from pathlib import Path

import pytest

from scripts.verify_arena_full_thinking_e2e import full_thinking_database
from tests.lab_arena.complete_baseline_final_authority_postgres_test import (
    test_normal_complete_baseline_publishes_and_activates_with_all_guards as _prove_round,
)


@pytest.fixture(params=["production", "numeric"])
def database(request):
    yield from full_thinking_database(production_order=request.param == "production")


# These two independently released migrations already share 585. Their exact
# filenames remain ordered explicitly by the current service fixture. Preserve
# applied history, reject any extra collision, and do not count rollback plans.
_RELEASED_SEQUENCE_COLLISIONS = {
    585: {'585-lab-arena-champion-baseline-reasoning.sql',
          '585-lab-arena-review-deadline-recovery.sql'},
}


def _unexpected_migration_collisions(paths):
    groups = {}
    for path in paths:
        number = int(path.name.split('-', 1)[0])
        if number >= 564 and not path.name.endswith('-rollback.sql'):
            groups.setdefault(number, set()).add(path.name)
    return {number: sorted(names) for number, names in groups.items()
            if len(names) > 1 and names != _RELEASED_SEQUENCE_COLLISIONS.get(number)}


def test_new_migrations_have_unique_sequence_numbers():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    assert not _unexpected_migration_collisions(scripts.glob("[0-9][0-9][0-9]-*.sql"))


def test_sequence_guard_keeps_new_collisions_and_extra_585_files_blocked():
    released = [Path(name) for name in _RELEASED_SEQUENCE_COLLISIONS[585]]
    assert not _unexpected_migration_collisions(released + [
        Path('585-lab-arena-review-deadline-recovery-rollback.sql')])
    assert 585 in _unexpected_migration_collisions(released + [Path('585-unreviewed.sql')])
    assert 589 in _unexpected_migration_collisions([
        Path('589-first.sql'), Path('589-second.sql')])


def test_complete_mainline_schema_starts_and_publishes(database, tmp_path, monkeypatch):
    pg, dsn = database
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        # These targets come from the guarded migrations: 569 publication,
        # 573 billing materialization, and 572 concurrent stage/claim behavior.
        expected = {
            "public.lab_arena__participant_incomplete_reason_v1(text,text)":
                "f0947c0d204d0883d926781cfe03ba22ad2b940ea16cf7618de90a9533cd44c4",
            "public.lab_arena_reserve_judgment_call(text,text,text,text,text,text,bigint,jsonb,integer)":
                "f73e8cd3839372e5bdb0cfefd2269e9bd1abcd578d0502cda14948e81acb843d",
            "public.lab_arena_miner_credit_failure_v1(text)":
                "23fbb6e5d40450916b75110d16d9b3c9c369264014dea51e17db7f3f978786c6",
            "public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)":
                "ce8691e166d24a87ff6bbf475dca8cc0eb526b6688f3af22fad6ad8b0a8b6c89",
            "public.lab_arena_open_stage(text,smallint,jsonb,integer[])":
                "fd1d9ea1870b749014e0685c10a6c3c1a3fdf68c8042dc02bf8d32162851b05f",
            "public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)":
                "a01a0cd11bba2abb77f74d04ee780cab32a04ff2c08eabbb448b90ae280e92f7",
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


@pytest.mark.parametrize("retry_backoff", [False, True])
def test_concurrent_baseline_judging_uses_complete_e2e_schema(
    database, tmp_path_factory, retry_backoff,
):
    from tests.lab_arena import company_judgment_validation_projection548_postgres_test as authority
    from tests.lab_arena import concurrent_baseline_judging572_postgres_test as concurrent

    # Reuse the real runner, stage-transition and final-publication checks on
    # the complete schema, including the full-thinking and closed-billing tail.
    completed = authority.completed.__wrapped__(database, tmp_path_factory)
    try:
        harness, row = next(completed)
        saved = {}
        with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
            for table in ("lab_arena_rounds", "lab_arena_runs", "lab_arena_ledger",
                          "lab_arena_judgment_cache"):
                cursor.execute(f"SELECT to_jsonb(t) FROM public.{table} t")
                saved[table] = [item[0] for item in cursor.fetchall()]
        scenario = database, harness, row, saved
        if retry_backoff:
            concurrent.test_miners_finish_while_baseline_retry_keeps_its_backoff(scenario)
        else:
            concurrent.test_real_overlap_and_final_authority(scenario, 1, 0)
    finally:
        completed.close()


def test_review_guard_read_is_installed_and_preserves_restart_authority(database):
    from tests.lab_arena import review_guard_row_read_postgres_test as guard

    # The current reviewer uses this RPC before reserving review work. It must
    # exist in the same complete schema used by publication and paid E2E.
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT to_regprocedure('public.lab_arena_review_guard_state_v1()')")
        assert cursor.fetchone()[0] is not None, 'current E2E tail omits migration 574'
    guard.test_idempotent_install_preserves_control_canonical_function_and_permissions(database)
    guard.test_review_guard_does_not_queue_behind_claim_control_but_canonical_state_still_does(database)
    for commit in (False, True):
        guard.test_real_guard_update_blocks_read_until_commit_or_rollback(database, commit)
    guard.test_retained_expired_guard_still_blocks_and_operator_pause_semantics_stay_same(database)
