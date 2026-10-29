"""The paid E2E runner starts from current, guarded PostgreSQL authority."""

from collections import defaultdict
import hashlib
from pathlib import Path

import pytest

from scripts.verify_arena_full_thinking_e2e import full_thinking_database
from tests.lab_arena.complete_baseline_final_authority_postgres_test import (
    test_normal_complete_baseline_publishes_and_activates_with_all_guards as _prove_round,
)


@pytest.fixture(params=["production", "numeric"])
def database(request):
    yield from full_thinking_database(production_order=request.param == "production")


_RECORDED_SHARED_SEQUENCES = {
    # Historical independent releases retain their full filename identity.
    # Exact introduction commits: 5595ae7b2e, 4654b49f24, f520c7e3d9, 8a43dab759.
    "478-lab-arena-baseline-closed-settlement-recovery.sql":
        "850c3ecb8967d219a6ed46cc7218bbf8eccf875c4f5d799d5c2f9a2887b48325",
    "478-lab-arena-infrastructure-retry-cost.sql":
        "7c56999a360aa29bc1f0e18598c69321edeeb662c428d69d89abb46c462cd017",
    "489-lab-arena-chain-outcome-resigning.sql":
        "98627134f3ba469d26dc588f5a4ae48d40680e7223ba82f0ece2df94dd410fa3",
    "489-lab-arena-weight-write-rpc-privileges.sql":
        "73e19a8fa99a468f262846f734630fe1994264e06d2113f9bbf9ea5aac005a9c",
    # Exact bytes at their introduction commits: 4f90c5172 and 16933f72f.
    # These already-applied identities must not be renamed to repair CI.
    "585-lab-arena-champion-baseline-reasoning.sql":
        "4c070d53c5a02cb0b4d44846d13041c49e647bca165dfc415f93aaf7b79bc3db",
    "585-lab-arena-review-deadline-recovery.sql":
        "e54b999380190722216c0f78a84e3ca175f9eadcf51fa68d2387a53669c3bc5a",
    "585-lab-arena-review-deadline-recovery-rollback.sql":
        "9c78bc154eb93482cf29f6be10609260efa8d395c6db2969edaaced52ff57e3d",
}


def _assert_migration_identities(paths):
    names = [path.name for path in paths]
    assert len(set(names)) == len(names), "duplicate migration identity"
    forward = defaultdict(set)
    for name in names:
        if name.endswith("-rollback.sql"):
            companion = name.removesuffix("-rollback.sql") + ".sql"
            assert companion in names, f"rollback has no forward migration: {name}"
        else:
            forward[int(name.split("-", 1)[0])].add(name)
    recorded_forward = defaultdict(set)
    for name in _RECORDED_SHARED_SEQUENCES:
        if not name.endswith("-rollback.sql"):
            recorded_forward[int(name.split("-", 1)[0])].add(name)
    collisions = {number: sorted(identities) for number, identities in forward.items()
                  if len(identities) > 1
                  and identities != recorded_forward.get(number)}
    assert not collisions, f"new forward migration sequence collision: {collisions}"


def test_new_migrations_preserve_recorded_identities_and_unique_forward_sequences():
    # The E2E runner uses full filenames. A rollback is a companion, not
    # another forward release. Only exact recorded collisions are allowed;
    # every new forward sequence must remain unique.
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    paths = [
        path
        for path in scripts.glob("[0-9][0-9][0-9]-*.sql")
        if int(path.name.split("-", 1)[0]) >= 564
    ]
    _assert_migration_identities(paths)
    for name, digest in _RECORDED_SHARED_SEQUENCES.items():
        assert hashlib.sha256((scripts / name).read_bytes()).hexdigest() == digest, name


def test_forward_migration_and_rollback_have_one_release_position():
    _assert_migration_identities([Path("589-example.sql"), Path("589-example-rollback.sql")])


@pytest.mark.parametrize("names,reason", [
    (["589-first.sql", "589-second.sql"], "new forward migration sequence collision"),
    ([*_RECORDED_SHARED_SEQUENCES, "585-unrecorded-forward.sql"], "new forward migration sequence collision"),
    (["589-example-rollback.sql"], "rollback has no forward migration"),
    (["589-example.sql", "589-example.sql"], "duplicate migration identity"),
])
def test_migration_identity_guard_rejects_new_ambiguity(names, reason):
    with pytest.raises(AssertionError, match=reason):
        _assert_migration_identities([Path(name) for name in names])


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
