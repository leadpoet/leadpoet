"""Actual claim/settlement/completion RPCs obey frozen score admission windows."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import confirmed_score_parallel455_postgres_test as prior
from tests.lab_arena import stage_cutoff_drain420_postgres_test as cutoff
from tests.lab_arena import test_lab_arena_restart_claim_drain_postgres as guards
from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import _uncertain_call
from tests.lab_arena.score_submission_serialization_postgres_test import _reserve_dynamic
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete

base_database = prior.base_database
current_database = prior.current_database
database = prior.database
ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/458-lab-arena-score-claim-cutoff.sql'
SIG = prior.SIG
PRE = prior.POST
POST = '4e5335081b6fb3f8007f9e3a74c61a84bee47c07f7260616c36cd39ede272621'
ANCHOR = '    -- lab_arena_stage_cutoff_drain_v1: this is admission, not an extension'
ADDITION = """    -- Frozen score windows gate admission even while the driver is delayed.
    AND (
      runs.kind <> 'score'
      OR pg_catalog.clock_timestamp() <
           (v_round.configuration_doc #>> ARRAY[
             'schedule', CASE v_stage WHEN 1 THEN 'stage_1_scoring_close'
               ELSE 'final_scoring_close' END
           ])::TIMESTAMPTZ
    )
"""
DEADLINE = datetime(2026, 10, 10, 20, 30, 2, 123456, timezone.utc)


def _seed(conn, *, stage=2, current=True, deadline=DEADLINE, kind='score', policy=True):
    # Existing harness uses a complete frozen configuration, source submission,
    # signed runner and pending jobs. Only fixture stage/window/policy differ.
    config = cutoff._seed(conn, kind=kind, policy=policy, positions=3)
    fields = [field.name for field in contracts.STAGE_SCHEDULE_FIELDS
              if field.name in config['schedule']]
    selected = fields.index('stage_1_scoring_close' if stage == 1 else 'final_scoring_close')
    config['schedule'] = {name: cutoff._stamp(deadline + timedelta(days=index-selected))
                          for index, name in enumerate(fields)}
    if not current:
        config.pop('sourcing_cost_eligibility_policy', None)
    contracts.validate_round_configuration(config)
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status=%s,configuration_doc=%s::jsonb WHERE round_id=%s",
                    (f'stage{stage}' + ('_scoring' if kind == 'score' else ''),
                     json.dumps(config), cutoff.ROUND))
        if stage == 1:
            cur.execute('UPDATE public.lab_arena_runs SET stage=1,icp_position=icp_position-10 WHERE round_id=%s', (cutoff.ROUND,))
    conn.commit()


def _state(conn, *, commit=True):
    state = cutoff._state(conn)
    with conn.cursor() as cur:
        cur.execute('SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c')
        state['control'] = cur.fetchall()
    if commit:
        conn.commit()
    return state


@pytest.fixture(scope='module')
def migrated(database):
    pg, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    with pg.connect(**dsn) as conn:
        with conn.cursor() as cur:
            cur.execute((ROOT / 'scripts/455-lab-arena-confirmed-score-parallel.sql').read_text())
            original = prior._definition(cur)
            security = prior._security(cur)
            assert original.count(ANCHOR) == 1
            cur.execute("SELECT encode(extensions.digest(%s,'sha256'),'hex')", (original,))
            assert cur.fetchone()[0] == PRE
        conn.commit()
        _seed(conn)
    # Regression proof against the exact original function: old code admits
    # new score work after cutoff. Keep that real lease across migration.
    active, token, request_id, _ = claim(store, cutoff.ROUND, cutoff.RUNNER, parallelism=8)
    assert active['status'] == 'leased' and active['kind'] == 'score'
    unknown, *_ = _uncertain_call(store, active, token, label='cutoff-unknown')
    identity, reserved = _reserve_dynamic(store, active, token, 'cutoff-paid')
    assert reserved['status'] == 'reserved'
    assert store.mark_dispatched(run_id=active['run_id'], lease_token_hash=hash_lease_token(token), call_identity=identity)['status'] == 'dispatched'
    accepted_old, accepted_token = claim(store, cutoff.ROUND, cutoff.RUNNER, parallelism=8)[:2]
    assert accepted_old['status'] == 'leased'
    assert complete(store, accepted_old['run_id'], hash_lease_token(accepted_token), 'accepted', output_ref=f'arena/{cutoff.ROUND}/outputs/old-accepted.json')['status'] == 'accepted'
    with pg.connect(**dsn) as conn:
        before = _state(conn)
        assert any(row['status'] == 'pending' for row in before['lab_arena_runs'])
        assert any(row['status'] == 'accepted' for row in before['lab_arena_runs'])
        for old_hash, error in [(PRE, 'preimage differs'), (POST, 'postimage differs')]:
            with conn.cursor() as cur:
                body = SQL.read_text().replace('BEGIN;\nSET LOCAL', 'SET LOCAL', 1).rsplit('COMMIT;', 1)[0]
                with pytest.raises(pg.Error, match=error):
                    cur.execute(body.replace(old_hash, '0'*64))
            conn.rollback()
            with conn.cursor() as cur:
                assert prior._definition(cur) == original
                assert prior._security(cur) == security
            assert _state(conn) == before
        with conn.cursor() as cur:
            cur.execute(SQL.read_text())
            after = prior._definition(cur)
            assert after == original.replace(ANCHOR, ADDITION + ANCHOR)
            assert prior._security(cur) == security
            cur.execute("SELECT encode(extensions.digest(%s,'sha256'),'hex')", (after,))
            assert cur.fetchone()[0] == POST
            cur.execute(SQL.read_text())
            assert prior._definition(cur) == after
        conn.commit()
        assert _state(conn) == before
    assert claim(store, cutoff.ROUND, cutoff.RUNNER, parallelism=8, request_id=request_id, token=token)[0] == active
    with pg.connect(**dsn) as conn:
        assert _state(conn) == before
    assert claim(store, cutoff.ROUND, cutoff.RUNNER, parallelism=8)[0] == {'status': 'no_pending'}
    with pg.connect(**dsn) as conn:
        assert _state(conn) == before
    # A frozen admission deadline must not stop existing paid work or drop an
    # unrelated unknown liability. Settlement and acceptance use real RPCs.
    unknown_before = store.list_ledger(call_identity=unknown)
    assert store.settle_call(run_id=active['run_id'], lease_token_hash=hash_lease_token(token), call_identity=identity, actual_microusd=2000, terminal_response={'status': 200, 'call_succeeded': True})['status'] == 'settled'
    assert complete(store, active['run_id'], hash_lease_token(token), 'accepted', output_ref=f'arena/{cutoff.ROUND}/outputs/preserved.json')['status'] == 'accepted'
    assert store.list_ledger(call_identity=unknown) == unknown_before
    with pg.connect(**dsn) as conn:
        accepted = _state(conn)
        assert accepted['lab_arena_submissions'] == before['lab_arena_submissions']
        with conn.cursor() as cur:
            cur.execute(SQL.read_text())
        conn.commit()
        assert _state(conn) == accepted
    yield store
    store.close()


@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('current', [False, True], ids=['legacy', 'current'])
@pytest.mark.parametrize('offset', [-1, 0, 1], ids=['before', 'equal', 'after'])
def test_exact_score_cutoff_boundary(database, migrated, stage, current, offset):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        _seed(conn, stage=stage, current=current)
        before = _state(conn)
        with conn.cursor() as cur:
            # Transaction-only clock controls ONLY the inserted predicate.
            # Exercise the actual RPC, locks and write path without a time race.
            original = prior._definition(cur)
            controlled = ADDITION.replace('pg_catalog.clock_timestamp()', 'public.lab_arena_test_claim_clock()')
            clock = DEADLINE + timedelta(microseconds=offset)
            cur.execute('CREATE FUNCTION public.lab_arena_test_claim_clock() RETURNS timestamptz LANGUAGE sql AS %s', ("SELECT '" + cutoff._stamp(clock) + "'::timestamptz",))
            cur.execute(original.replace(ADDITION, controlled))
            cur.execute('SET LOCAL ROLE lab_arena_service')
            cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,8,8,%s,%s,%s,%s,4500)',
                        (cutoff.ROUND, cutoff.RUNNER, [], 'd'*32, 'sha256:' + 'b'*64, 'sha256:' + 'c'*64))
            result = cur.fetchone()[0]
            assert result['status'] == ('leased' if offset < 0 else 'no_pending')
            if offset < 0:
                assert result['kind'] == 'score' and result['stage'] == stage
            cur.execute('RESET ROLE')
            if offset >= 0:
                assert _state(conn, commit=False) == before
        conn.rollback()
        with conn.cursor() as cur:
            assert prior._definition(cur) == original


@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('deadline', [None, 'invalid'], ids=['missing', 'invalid'])
def test_missing_or_invalid_selected_deadline_fails_closed(database, migrated, stage, deadline):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        _seed(conn, stage=stage)
        key = 'stage_1_scoring_close' if stage == 1 else 'final_scoring_close'
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,ARRAY['schedule',%s],%s::jsonb)", (key, json.dumps(deadline)))
        conn.commit()
        before = _state(conn)
        if deadline is None:
            assert cutoff._claim(conn) == {'status': 'no_pending'}
        else:
            with pytest.raises(pg.Error):
                cutoff._claim(conn)
            conn.rollback()
        assert _state(conn) == before


@pytest.mark.parametrize('past_cutoff', [False, True], ids=['before-cutoff', 'after-cutoff'])
def test_restart_guard_and_existing_exact_replay(database, migrated, past_cutoff):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        _seed(conn, deadline=datetime(2099, 1, 1, tzinfo=timezone.utc))
        active = cutoff._claim(conn)
        assert active['status'] == 'leased'
        with conn.cursor() as cur:
            cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control')
            generation = cur.fetchone()[0]
        acquired = guards._acquire(conn, generation=generation)
        assert acquired['guard_present'] is True and acquired['drain']['captured_count'] == 1
        conn.commit()
        if past_cutoff:
            with conn.cursor() as cur:
                cur.execute('SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (cutoff.ROUND,))
                config = cur.fetchone()[0]
                config['schedule'] = {key: cutoff._stamp(datetime.fromisoformat(value.replace('Z', '+00:00')) - timedelta(days=40000)) for key, value in config['schedule'].items()}
                contracts.validate_round_configuration(config)
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s', (json.dumps(config), cutoff.ROUND))
            conn.commit()
        before = _state(conn)
        assert cutoff._claim(conn) == active
        assert cutoff._claim(conn, 'b')['status'] == 'paused'
        assert _state(conn) == before
        guards._rpc(conn, 'lab_arena_abort_restart_guard_v1', guards.GUARD, guards.OWNER, acquired['guard_generation'], 'test-abort')
        conn.commit()


@pytest.mark.parametrize('policy', [False, True], ids=['legacy', 'baseline-first'])
def test_execute_cutoff_unchanged(database, migrated, policy):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        _seed(conn, kind='execute', policy=policy)
        cutoff._boundary(conn, -1)
        before = _state(conn)
        result = cutoff._claim(conn)
        assert result['status'] == ('no_pending' if policy else 'leased')
        if policy:
            assert _state(conn) == before


@pytest.mark.parametrize('mutation,error', [('acl', 'security shape differs'), ('body', 'preimage differs')])
def test_migration_replay_rejects_tampering(database, migrated, mutation, error):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        before = _state(conn)
        with conn.cursor() as cur:
            original, security = prior._definition(cur), prior._security(cur)
            if mutation == 'acl':
                cur.execute(f'GRANT EXECUTE ON FUNCTION {SIG} TO PUBLIC')
            else:
                cur.execute(original.replace('Frozen score windows', 'Unexpected score windows'))
            body = SQL.read_text().replace('BEGIN;\nSET LOCAL', 'SET LOCAL', 1)
            body = body.rsplit('COMMIT;', 1)[0]
            with pytest.raises(pg.Error, match=error):
                cur.execute(body)
        conn.rollback()
        with conn.cursor() as cur:
            assert prior._definition(cur) == original
            assert prior._security(cur) == security
        assert _state(conn) == before
