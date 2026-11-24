"""Native SQL proof that an expired restart guard stops withholding paid review."""
import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena.store import ArenaStore, PsycopgTransport, new_lease_token
from tests.lab_arena import prospective_admission669_postgres_test as alternate
from tests.lab_arena.prospective_admission660_postgres_test import migration_state_snapshot
from tests.lab_arena import review_infrastructure_hold701_postgres_test as hold
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '741-lab-arena-review-reservation-guard-lease.sql'
PRIOR = ('705-lab-arena-review-guard-lease-expiry.sql',)
GATE = 'lab_arena_begin_submission_review_v2(text,text,text,bigint,text,integer,bigint,text)'
PROBE = 'lab_arena_submission_replacement_schema_v1()'
POSTIMAGE = '7153f9bf3687dd149f92c0239045e9970bda1522276a0307d0173962a803f791'
COMMITMENT = 'sha256:' + 'a' * 64
CANDIDATE = 'b' * 40


def set_guard(connection, *, lease_seconds):
    """Install a drain guard whose lease expires lease_seconds from now."""
    expires = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        try:
            cursor.execute(
                "UPDATE public.lab_arena_restart_claim_control SET guard_commitment=%s,"
                "owner_commitment=%s,guard_generation=guard_generation+1,guard_expires_at=%s,"
                "candidate_commit=%s,restart_scope='gateway',restart_phase='draining' "
                "WHERE singleton",
                (COMMITMENT, COMMITMENT, expires, CANDIDATE),
            )
        finally:
            cursor.execute("SET session_replication_role=DEFAULT")
    return expires


def clear_guard(connection):
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        try:
            cursor.execute(
                "UPDATE public.lab_arena_restart_claim_control SET guard_commitment='',"
                "owner_commitment='',guard_expires_at=NULL,candidate_commit='',"
                "restart_scope='',restart_phase='',captured_leases='[]'::JSONB WHERE singleton"
            )
        finally:
            cursor.execute("SET session_replication_role=DEFAULT")


def reserve(store, suffix):
    _, sid, miner = alternate.new(store, suffix)
    return alternate.begin(store, sid, miner, token=new_lease_token())


@pytest.fixture(scope='module')
def database():
    generator = hold.database.__wrapped__()
    try:
        pg, dsn = next(generator)
        with pg.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                for name in PRIOR:
                    cursor.execute((SCRIPTS / name).read_text())
            store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
            try:
                # The defect: an abandoned guard whose lease lapsed an hour ago
                # still refuses every paid reservation, so review never resumes.
                set_guard(connection, lease_seconds=-3600)
                assert reserve(store, 'lease741-before')['status'] == 'paused'
                before = migration_state_snapshot(connection)
                with connection.cursor() as cursor:
                    cursor.execute(MIGRATION.read_text())
                after = migration_state_snapshot(connection)
                assert after[1] == before[1]  # No historical row changes.
                changed = {name for name in before[0] if before[0][name] != after[0][name]}
                assert changed == {GATE, PROBE}
                assert set(after[0]) == set(before[0])  # No new or dropped functions.
                for name in changed:
                    assert after[0][name][1] == before[0][name][1]  # Owner, ACL, volatility.
                assert hashlib.sha256(after[0][GATE][0].encode()).hexdigest() == POSTIMAGE
                assert POSTIMAGE in after[0][PROBE][0]
                with connection.cursor() as cursor:
                    cursor.execute(MIGRATION.read_text())
                assert migration_state_snapshot(connection) == after
                clear_guard(connection)
            finally:
                store.close()
        yield pg, dsn
    finally:
        generator.close()


@pytest.fixture
def context(database):
    pg, dsn = database
    connection = pg.connect(**dsn)
    connection.autocommit = True
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    try:
        yield store, connection
    finally:
        clear_guard(connection)
        store.close()
        connection.close()


def test_an_expired_guard_no_longer_withholds_paid_review(context):
    store, connection = context
    set_guard(connection, lease_seconds=-1)
    assert reserve(store, 'lease741-expired')['status'] == 'claimed'


def test_a_live_guard_still_withholds_paid_review(context):
    store, connection = context
    set_guard(connection, lease_seconds=900)
    assert reserve(store, 'lease741-live')['status'] == 'paused'


def test_no_guard_reserves_as_before(context):
    store, _ = context
    assert reserve(store, 'lease741-none')['status'] == 'claimed'


def test_an_expiring_guard_releases_review_without_a_restart(context):
    store, connection = context
    set_guard(connection, lease_seconds=900)
    assert reserve(store, 'lease741-holds')['status'] == 'paused'
    # Nothing restarts and the commitment is still recorded; only time passes.
    set_guard(connection, lease_seconds=-1)
    with connection.cursor() as cursor:
        cursor.execute("SELECT guard_commitment FROM public.lab_arena_restart_claim_control"
                       " WHERE singleton")
        assert cursor.fetchone()[0] == COMMITMENT
    assert reserve(store, 'lease741-resumes')['status'] == 'claimed'


def test_readiness_probes_accept_the_rewritten_gate(context):
    store, _ = context
    store.code_review_schema(require_refusal_fallback=True, require_invalid_review_budget=True,
                             require_prospective_primary=True, require_prospective_alternate=True)
    store.submission_replacement_schema(require_per_round_cutoff=True)
