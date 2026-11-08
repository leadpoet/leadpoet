"""Forward-only readiness proof for the669 wrapper and unchanged direct probes."""
from contextlib import contextmanager
from datetime import timedelta

import pytest

from lab_arena.service import ServiceError
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, new_lease_token
from tests.lab_arena import prospective_admission669_postgres_test as alternate
from tests.lab_arena.queued_submission_replacement_postgres_test import _config, _doc, _accept
from tests.lab_arena.test_lab_arena_service_round import Harness
from tests.lab_arena.test_lab_arena_source_migration_postgres import _hotkey
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '670-lab-arena-replacement-schema-review-delegation.sql'
PROBE = 'public.lab_arena_submission_replacement_schema_v1()'
BEGIN = 'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'
V2 = 'public.lab_arena_begin_submission_review_v2(text,text,text,bigint,text,integer,bigint,text)'
EXPECTED = {'schema_version': 'leadpoet.lab_arena.submission_replacement_schema.v1',
            'version': 258, 'replacement_freeze_seconds': 3600, 'max_replacement_attempts': 1,
            'per_round_replacement_freeze_seconds': True, 'default_replacement_freeze_seconds': 600}


def definition(connection, signature):
    with connection.cursor() as cursor:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        return cursor.fetchone()[0]


@pytest.fixture(scope='module')
def database(tmp_path_factory):
    fixture = alternate.primary.recovery_database.__wrapped__()
    store = connection = harness = None
    try:
        pg, dsn = next(fixture)
        connection = pg.connect(**dsn)
        connection.autocommit = True
        direct = definition(connection, BEGIN)
        harness = Harness(lambda: pg.connect(**dsn), tmp_path_factory.mktemp('probe670'),
                          challengers=[], runners=['alpha'])
        assert harness.service.startup_checks()['schema_version'] == 197
        store = harness.service.store
        assert store.submission_replacement_schema(require_per_round_cutoff=True) == EXPECTED
        with connection.cursor() as cursor:
            cursor.execute(alternate.MIGRATION.read_text())
        with pytest.raises(ServiceError, match='submission_replacement_schema_unavailable'):
            harness.service.startup_checks()
        before = alternate.primary.migration_state_snapshot(connection)
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
        after = alternate.primary.migration_state_snapshot(connection)
        assert after[1] == before[1]
        assert set(after[0]) == set(before[0])
        changed = {name for name in before[0] if before[0][name] != after[0][name]}
        assert changed == {PROBE.removeprefix('public.')}
        assert before[0][PROBE.removeprefix('public.')][1] == after[0][PROBE.removeprefix('public.')][1]
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
        assert alternate.primary.migration_state_snapshot(connection) == after
        yield connection, store, harness, direct
    finally:
        if store is not None:
            store.close()
        if connection is not None:
            connection.close()
        fixture.close()


def test_full_gateway_startup_and_both_review_contracts_pass(database):
    _, store, harness, _ = database
    assert harness.service.startup_checks()['schema_version'] == 197
    assert store.submission_replacement_schema(require_per_round_cutoff=True) == EXPECTED
    assert store.code_review_schema(require_refusal_fallback=True,
        require_invalid_review_budget=True, require_prospective_primary=True)['version'] == 207
    assert store.code_review_schema(require_prospective_alternate=True)['prospective_alternate']['model'] == alternate.KIMI


@contextmanager
def altered(connection, signature, old, new):
    original = definition(connection, signature)
    assert old in original
    with connection.cursor() as cursor:
        cursor.execute(original.replace(old, new))
    try:
        yield
    finally:
        with connection.cursor() as cursor:
            cursor.execute(original)


@pytest.mark.parametrize('signature,old,new', [
    (BEGIN, 'BEGIN\n', "BEGIN\n  -- lab_arena_queued_replacement_review_deferral\n"),
    (V2, 'lab_arena_queued_replacement_review_deferral', 'untrusted_deferral_marker'),
    (V2, 'BEGIN\n', 'BEGIN\n  -- different delegate source\n'),
])
def test_modified_wrapper_or_delegate_cannot_pass_with_marker_comments(database, signature, old, new):
    connection, store, _, _ = database
    with altered(connection, signature, old, new):
        with pytest.raises(ArenaStoreError, match='lab_arena_queued_replacement_schema_incomplete'):
            store.submission_replacement_schema()
    assert store.submission_replacement_schema() == EXPECTED


def test_missing_delegate_is_rejected(database):
    connection, store, _, _ = database
    with connection.cursor() as cursor:
        cursor.execute(f'ALTER FUNCTION {V2} RENAME TO lab_arena_begin_submission_review_v2_hidden670')
    try:
        with pytest.raises(ArenaStoreError, match='lab_arena_queued_replacement_schema_incomplete'):
            store.submission_replacement_schema()
    finally:
        with connection.cursor() as cursor:
            cursor.execute('ALTER FUNCTION public.lab_arena_begin_submission_review_v2_hidden670'
                           '(text,text,text,bigint,text,integer,bigint,text) RENAME TO lab_arena_begin_submission_review_v2')


@pytest.mark.parametrize('signature', [BEGIN, V2])
@pytest.mark.parametrize('change,restore', [
    ('OWNER TO postgres', 'OWNER TO lab_arena_owner'),
    ('GRANT', 'REVOKE'),
])
def test_wrapper_and_delegate_security_drift_is_rejected(database, signature, change, restore):
    connection, store, _, _ = database
    with connection.cursor() as cursor:
        if change == 'GRANT':
            cursor.execute(f'GRANT EXECUTE ON FUNCTION {signature} TO PUBLIC')
        else:
            cursor.execute(f'ALTER FUNCTION {signature} {change}')
    try:
        with pytest.raises(ArenaStoreError, match='lab_arena_queued_replacement_schema_incomplete'):
            store.submission_replacement_schema()
    finally:
        with connection.cursor() as cursor:
            if restore == 'REVOKE':
                cursor.execute(f'REVOKE EXECUTE ON FUNCTION {signature} FROM PUBLIC')
            else:
                cursor.execute(f'ALTER FUNCTION {signature} {restore}')


@pytest.mark.parametrize('delegate_installed', [False, True])
def test_legacy_direct_body_retains_original_marker_based_compatibility(database, delegate_installed):
    connection, store, harness, direct = database
    wrapper = definition(connection, BEGIN)
    with connection.cursor() as cursor:
        if not delegate_installed:
            cursor.execute(f'ALTER FUNCTION {V2} RENAME TO lab_arena_begin_submission_review_v2_hidden670')
        # A benign legacy source variation must not need a new approved hash.
        cursor.execute(direct.replace('BEGIN\n', 'BEGIN\n  -- legacy direct implementation\n', 1))
    try:
        assert store.submission_replacement_schema(require_per_round_cutoff=True) == EXPECTED
        assert harness.service.startup_checks()['schema_version'] == 197
    finally:
        with connection.cursor() as cursor:
            cursor.execute(wrapper)
            if not delegate_installed:
                cursor.execute('ALTER FUNCTION public.lab_arena_begin_submission_review_v2_hidden670'
                               '(text,text,text,bigint,text,integer,bigint,text) RENAME TO lab_arena_begin_submission_review_v2')


@pytest.mark.parametrize('signature,marker,replacement', [
    ('public.lab_arena__register_submission_v2(text,text,text,jsonb,text,bigint,text)', 'replacement_limit_reached', 'untrusted_limit670'),
    ('public.lab_arena__register_submission_v2(text,text,text,jsonb,text,bigint,text)', 'v_prior_original', 'v_legacy_source670'),
    ('public.lab_arena_accept_submission_with_credentials(text,text,text,jsonb)', 'v_prior.replaces_submission_id IS NOT NULL', 'v_prior.replaces_submission_id IS NULL'),
])
def test_existing_registration_and_finalization_guards_remain_required(database, signature, marker, replacement):
    connection, store, _, _ = database
    # Alter only a string/identifier in this disposable definition, never invoke it.
    with altered(connection, signature, marker, replacement):
        with pytest.raises(ArenaStoreError, match='lab_arena_queued_replacement_schema_incomplete'):
            store.submission_replacement_schema()


@pytest.mark.parametrize('index', ['lab_arena_submissions_one_current_per_miner_uq',
                                   'lab_arena_submissions_one_uploading_per_miner_uq'])
def test_both_existing_indexes_remain_required(database, index):
    connection, store, _, _ = database
    with connection.cursor() as cursor:
        cursor.execute(f'ALTER INDEX public.{index} RENAME TO {index}_hidden670')
    try:
        with pytest.raises(ArenaStoreError, match='lab_arena_queued_replacement_schema_incomplete'):
            store.submission_replacement_schema()
    finally:
        with connection.cursor() as cursor:
            cursor.execute(f'ALTER INDEX public.{index}_hidden670 RENAME TO {index}')


def test_queued_review_still_defers_both_entrypoints_without_spend(database):
    connection, store, _, _ = database
    rid = 'arena-2098-05-04-probe670'
    miner = _hotkey('queued-review-probe670')
    store.create_round(rid, _config(rid, cutoff_after=timedelta(hours=3)))
    assert store.register_submission(rid, 'probe670-queued', miner, _doc(rid, 'probe670-queued', 123))['status'] == 'registered'
    assert _accept(store, rid, 'probe670-queued', miner)['status'] == 'ok'
    before = alternate.primary.migration_state_snapshot(connection)
    for selected_alternate in (None, alternate.KIMI):
        result = store.begin_submission_review('probe670-queued', miner, new_lease_token(),
            50_000, alternate.DS, 3, 1200, selected_alternate)
        assert result['status'] == 'deferred'
    assert store.get_submission('probe670-queued')['code_review_status'] == 'pending'
    assert store.list_ledger(submission_id='probe670-queued') == []
    assert alternate.primary.migration_state_snapshot(connection) == before


def test_migration_refuses_unknown_probe_source_and_preserves_metadata(database):
    connection, _, _, _ = database
    with altered(connection, PROBE, 'BEGIN\n', 'BEGIN\n  -- unexpected probe source\n'):
        before = alternate.primary.migration_state_snapshot(connection)
        try:
            with connection.cursor() as cursor, pytest.raises(Exception, match='probe670 preimage differs'):
                cursor.execute(MIGRATION.read_text())
        finally:
            with connection.cursor() as cursor:
                cursor.execute('ROLLBACK')
        assert alternate.primary.migration_state_snapshot(connection) == before
