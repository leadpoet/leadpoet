"""Native SQL proof that an outage hold keeps reviews retrying past the deadline."""
import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena import code_review, code_review_policy
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, new_lease_token
from tests.lab_arena import prospective_admission669_postgres_test as alternate
from tests.lab_arena.prospective_admission660_postgres_test import migration_state_snapshot
from tests.lab_arena.refusal_review_fallback599_postgres_test import verdict
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '701-lab-arena-review-infrastructure-hold.sql'
PRIOR = ('670-lab-arena-replacement-schema-review-delegation.sql',
         '687-lab-arena-private-owner-review-findings.sql')
DS = code_review.PROSPECTIVE_REVIEW_MODEL
V2 = 'lab_arena_begin_submission_review_v2(text,text,text,bigint,text,integer,bigint,text)'
FINISH = 'lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)'
PROBE = 'lab_arena_submission_replacement_schema_v1()'
POSTIMAGES = {
    V2: 'd5bf3a7c86e97dfb1b7080f5d6e959b05c0fa3463a10ecf88f93e67fddbaba5a',
    FINISH: '0c195df953c91fab3d7a1267d66b80cfc1d10a419cf349641915afd3b88504d9',
    PROBE: 'e39f8dd8162715b476dd02f7a6535512500fa0d29e8ccf9f43b2903361d3adbc',
}
NEW_FUNCTIONS = {
    'lab_arena__review_hold_deadline701_v1(lab_arena_rounds)',
    'lab_arena_hold_round_review_v1(text)',
    'lab_arena_decide_review_hold_v1(text,text)',
}


def past_deadline(connection, round_id):
    """Move only this disposable round to just after its benchmark deadline."""
    now = datetime.now(timezone.utc)
    schedule = {'submission_cutoff': now - timedelta(minutes=40),
                'benchmark_deadline': now - timedelta(minutes=1),
                'stage_1_start': now - timedelta(seconds=59)}
    with connection.cursor() as cursor:
        # The write-once trigger protects real rounds; skip it for this fixture only.
        cursor.execute("SET session_replication_role=replica")
        try:
            for key, moment in schedule.items():
                cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
                               "configuration_doc,%s,to_jsonb(%s::TEXT)) WHERE round_id=%s",
                               ('{schedule,%s}' % key, moment.isoformat(), round_id))
        finally:
            cursor.execute("SET session_replication_role=DEFAULT")
    return schedule


def claim(store, sid, miner):
    token = new_lease_token()
    return alternate.begin(store, sid, miner, DS, token), token


@pytest.fixture(scope='module')
def database():
    generator = alternate.recovery_database.__wrapped__()
    try:
        pg, dsn = next(generator)
        with pg.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                for name in PRIOR:
                    cursor.execute((SCRIPTS / name).read_text())
            store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
            try:
                # Before 701 an outage past the deadline can only end in rejection.
                rid, sid, miner = alternate.new(store, 'hold701-before')
                past_deadline(connection, rid)
                assert claim(store, sid, miner)[0]['status'] == 'deadline'
                before = migration_state_snapshot(connection)
                with connection.cursor() as cursor:
                    cursor.execute(MIGRATION.read_text())
                after = migration_state_snapshot(connection)
                assert after[1] == before[1]  # No historical row changes.
                changed = {name for name in before[0] if before[0][name] != after[0][name]}
                assert changed == set(POSTIMAGES)
                assert set(after[0]) - set(before[0]) == NEW_FUNCTIONS
                for name in changed:
                    assert after[0][name][1] == before[0][name][1]  # Owner, ACL, volatility.
                    assert hashlib.sha256(after[0][name][0].encode()).hexdigest() == POSTIMAGES[name]
                with connection.cursor() as cursor:
                    cursor.execute(MIGRATION.read_text())
                assert migration_state_snapshot(connection) == after
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
        store.close()
        connection.close()


def test_readiness_probes_accept_the_rewritten_gate(context):
    store, _ = context
    store.code_review_schema(require_refusal_fallback=True, require_invalid_review_budget=True,
                             require_prospective_primary=True, require_prospective_alternate=True)
    store.submission_replacement_schema(require_per_round_cutoff=True)


def test_hold_keeps_an_outage_retrying_past_the_deadline_until_it_passes(context):
    store, connection = context
    rid, sid, miner = alternate.new(store, 'hold701-outage')
    result, token = claim(store, sid, miner)
    assert result['status'] == 'claimed'
    alternate.finish_error(store, connection, sid, miner, token, DS, 'code_review_provider_unavailable')
    schedule = past_deadline(connection, rid)
    # Without a hold the historical deadline still applies.
    assert claim(store, sid, miner)[0]['status'] == 'deadline'
    held = store.hold_round_review(rid)
    assert held['status'] == 'held' and held['released_at'] is None
    stage_1 = schedule['stage_1_start']
    assert datetime.fromisoformat(held['review_deadline']) == stage_1 + timedelta(minutes=60)
    assert datetime.fromisoformat(held['max_review_deadline']) == stage_1 + timedelta(minutes=120)
    assert store.hold_round_review(rid) == {**held, 'status': 'existing'}
    # The provider recovered: the same submission is claimed and passes.
    result, token = claim(store, sid, miner)
    assert result['status'] == 'claimed'
    receipt = store.finish_submission_review(sid, miner, token, 'passed', verdict(DS), 1234)
    assert receipt['code_review_status'] == 'passed'
    assert store.get_submission(sid)['code_review_status'] == 'passed'


def test_released_or_absent_hold_keeps_the_historical_deadline(context):
    store, connection = context
    rid, sid, miner = alternate.new(store, 'hold701-release')
    past_deadline(connection, rid)
    assert store.decide_review_hold(rid, 'release') == {'status': 'no_hold'}
    assert store.hold_round_review(rid)['status'] == 'held'
    released = store.decide_review_hold(rid, 'release')
    assert released['status'] == 'released' and released['released_at'] is not None
    assert store.decide_review_hold(rid, 'extend')['status'] == 'released'
    assert claim(store, sid, miner)[0]['status'] == 'deadline'
    with pytest.raises(ArenaStoreError):
        store.decide_review_hold(rid, 'reopen')


def test_extend_moves_once_to_the_fixed_cap(context):
    store, connection = context
    rid, _, _ = alternate.new(store, 'hold701-extend')
    past_deadline(connection, rid)
    held = store.hold_round_review(rid)
    extended = store.decide_review_hold(rid, 'extend')
    assert extended['status'] == 'extended'
    assert extended['review_deadline'] == held['max_review_deadline']
    assert store.decide_review_hold(rid, 'extend') == extended


def test_hold_is_not_due_before_cutoff_and_never_for_a_closed_round(context):
    store, connection = context
    rid, _, _ = alternate.new(store, 'hold701-early')
    assert store.hold_round_review(rid) == {'status': 'not_due'}
    with connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM public.lab_arena_review_holds WHERE round_id=%s", (rid,))
        assert cursor.fetchone()[0] == 0
    rid, sid, miner = alternate.new(store, 'hold701-closed')
    past_deadline(connection, rid)
    assert store.hold_round_review(rid)['status'] == 'held'
    store.cancel_round(rid, 'operator')
    assert store.hold_round_review(rid)['status'] == 'existing'
    assert store.decide_review_hold(rid, 'extend') == {'status': 'not_open'}
    assert claim(store, sid, miner)[0]['status'] in ('deadline', 'round_closed')


def test_claim_cut_off_by_a_held_window_stays_retryable(context):
    store, connection = context
    rid, sid, miner = alternate.new(store, 'hold701-window')
    past_deadline(connection, rid)
    assert store.hold_round_review(rid)['status'] == 'held'
    result, token = claim(store, sid, miner)
    assert result['status'] == 'claimed'
    with connection.cursor() as cursor:
        cursor.execute("UPDATE public.lab_arena_review_holds SET review_deadline=clock_timestamp()"
                       "-interval '1 second' WHERE round_id=%s", (rid,))
    receipt = store.finish_submission_review(sid, miner, token, 'passed', verdict(DS), 1234)
    assert receipt['code_review_status'] == 'error'
    row = store.get_submission(sid)
    assert row['code_review_doc']['error_code'] == 'code_review_transport_error'
    assert code_review_policy.review_retry_available(row)
    assert code_review_policy.infrastructure_blocked(row)


def test_hold_table_is_owner_private(context):
    _, connection = context
    with connection.cursor() as cursor:
        cursor.execute("SELECT has_table_privilege('lab_arena_service','public.lab_arena_review_holds','SELECT'),"
                       "has_function_privilege('lab_arena_service','public.lab_arena_hold_round_review_v1(text)','EXECUTE'),"
                       "has_function_privilege('lab_arena_service','public.lab_arena_decide_review_hold_v1(text,text)','EXECUTE'),"
                       "has_function_privilege('lab_arena_service',"
                       "'public.lab_arena__review_hold_deadline701_v1(lab_arena_rounds)','EXECUTE')")
        assert cursor.fetchone() == (False, True, True, False)

