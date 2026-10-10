"""Exact local terminal hold: active judges drain through current RPCs."""
from datetime import datetime, timezone
import uuid

import pytest

from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena import oct10_uniform_pe_rejudge445_postgres_test as recovery
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete, hash_lease_token, hotkey

database = recovery.database
migrated = recovery.migrated
ROUND = recovery.ROUND


@pytest.fixture
def unheld(database, migrated):
    yield from recovery._seeded(database, apply_hold=False)


def test_terminal_hold_preserves_active_judge_blocks_new_work_and_allows_rpc_completion(unheld):
    psycopg, dsn = unheld
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    lease, token, _, _ = claim(store, ROUND, hotkey('hold444-runner'), parallelism=20, ceiling=20)
    assert lease['status'] == 'leased' and lease['kind'] == 'score'
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            sql = recovery._hold_render(cur)
            before = recovery._snapshot(cur)
            cur.execute(sql)
            held = recovery._snapshot(cur)
            assert {k:v for k,v in held.items() if k != 'hold'} == {k:v for k,v in before.items() if k != 'hold'}
            assert (held['hold']['operator_paused'], held['hold']['actor_ref'], held['hold']['pause_reason']) == (True, 'oct10-uniform-pe-hold444', 'oct10_uniform_pe_boundary_recovery')
            cur.execute(sql)
            assert recovery._snapshot(cur) == held
            # Existing trigger/API guards must work without new framework code.
            with pytest.raises(psycopg.Error, match='round_progression_paused'):
                cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged' WHERE round_id=%s", (ROUND,))
            cur.execute('ROLLBACK')
            with pytest.raises(psycopg.Error, match='claims_paused'):
                cur.execute("UPDATE public.lab_arena_runs SET status='leased' WHERE run_id='old445-score-0'")
            cur.execute('ROLLBACK')
            rerun = recovery._render(cur)
            with pytest.raises(psycopg.Error, match='terminal hold or inventory invalid'):
                cur.execute(rerun)
            cur.execute('ROLLBACK')
    denied, _, _, _ = claim(store, ROUND, hotkey('hold444-other'), parallelism=20, ceiling=20)
    assert denied['status'] in ('paused', 'no_pending')
    assert complete(store, lease['run_id'], hash_lease_token(token), 'accepted', output_ref='arena/test/drained444.json')['status'] == 'accepted'
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)
            assert recovery._snapshot(cur)['hold'] == held['hold']
            cur.execute(recovery._render(cur))
            cur.execute('SELECT operator_paused,guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
            assert cur.fetchone() == (False, held['hold']['guard_generation'])


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_rounds SET stage_generation=99 WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_rounds SET status_generation=99 WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{max_challengers}','68') WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_runs SET status='pending' WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-10' AND stage=2 AND kind='execute' AND status='accepted')",
    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,pause_reason='foreign',actor_ref='foreign' WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('a',64),owner_commitment='sha256:'||repeat('b',64),candidate_commit=repeat('c',40),restart_scope='all',restart_phase='draining',guard_expires_at=now()+interval '1 hour' WHERE singleton",
    "UPDATE public.lab_arena_submissions SET source_size_bytes=999 WHERE submission_id='miner440-4'",
    "UPDATE public.lab_arena_runs SET output_ref='arena/tampered.json' WHERE round_id='arena-2026-10-10' AND stage=2 AND kind='execute' AND icp_position=0",
    "UPDATE public.qualification_private_icp_sets SET icps='[]' WHERE set_id=20261009",
    "UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id='arena-2026-10-10-r440archive'",
])
def test_hold_invalid_state_and_reviewed_preimage_drift_fail_closed(unheld, mutation):
    psycopg, dsn = unheld
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            sql = recovery._hold_render(cur)
            original = recovery._snapshot(cur)
            cur.execute('BEGIN')
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute(mutation)
            cur.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(psycopg.Error, match='Oct10 uniform hold444'):
                cur.execute(sql)
            cur.execute('ROLLBACK')
            assert recovery._snapshot(cur) == original


@pytest.mark.parametrize('mode,status', [('shadow','stage2'), ('live','committed')])
def test_unrelated_non_live_or_idle_committed_round_does_not_block_hold(unheld, mode, status):
    psycopg, dsn = unheld
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('SET session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET status=%s,configuration_doc=jsonb_set(configuration_doc,'{mode}',to_jsonb(%s::text)) WHERE round_id=%s", (status, mode, recovery.old.ARCHIVE))
            cur.execute('SET session_replication_role=origin')
            cur.execute(recovery._hold_render(cur))
            assert recovery._snapshot(cur)['hold']['operator_paused']
            cur.execute(recovery._render(cur))
            assert not recovery._snapshot(cur)['hold']['operator_paused']


def test_hold_placeholder_and_function_drift_fail_closed(unheld):
    psycopg, dsn = unheld
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            before = recovery._snapshot(cur)
            sql = recovery._hold_render(cur)
            template = (recovery.ROOT / 'scripts/444-arena-2026-10-10-uniform-pe-hold.sql.template').read_text()
            with pytest.raises(psycopg.Error):
                cur.execute(template)
            cur.execute('ROLLBACK')
            cur.execute('BEGIN')
            cur.execute(before['stage_function'].replace('  v_generation := v_round.stage_generation + 1;', '  v_generation := v_round.stage_generation + 2;'))
            with pytest.raises(psycopg.Error, match='reviewed preimage differs'):
                cur.execute(sql)
            cur.execute('ROLLBACK')
            assert recovery._snapshot(cur) == before


@pytest.mark.parametrize('status', ['open', 'committed'])
def test_foreign_live_lease_blocks_hold_and_rejudge_even_on_idle_round(unheld, status):
    psycopg, dsn = unheld
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for rejudge in (False, True):
                cur.execute('BEGIN')
                if rejudge:
                    cur.execute(recovery._hold_render(cur).replace('BEGIN;', '').replace('COMMIT;', ''))
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute('UPDATE public.lab_arena_rounds SET status=%s WHERE round_id=%s', (status, recovery.old.ARCHIVE))
                cur.execute("UPDATE public.lab_arena_runs SET status='leased',lease_expires_at=now()+interval '1 hour' WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score')", (recovery.old.ARCHIVE,))
                assert cur.rowcount == 1
                cur.execute('SET LOCAL session_replication_role=origin')
                sql = recovery._render(cur) if rejudge else recovery._hold_render(cur)
                with pytest.raises(psycopg.Error, match='terminal.*(invalid|differs)'):
                    cur.execute(sql)
                cur.execute('ROLLBACK')
                assert not recovery._snapshot(cur)['hold']['operator_paused']


def test_reviewed_hold_allows_normal_claim_progress_completion_and_round_close_before_apply(unheld):
    psycopg, dsn = unheld
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            sql = recovery._hold_render(cur)
    # These ordinary changes occur after review, before the hold takes locks.
    lease, token, _, _ = claim(store, ROUND, hotkey('prehold444'), parallelism=20, ceiling=20)
    assert lease['status'] == 'leased'
    token_hash = hash_lease_token(token)
    assert store.append_trajectory_events(lease['run_id'], token_hash, [{
        'event_id': str(uuid.uuid4()), 'kind': 'runtime.started',
        'occurred_at': datetime.now(timezone.utc).isoformat(),
        'content': {'status': 'starting', 'runtime': 'runsc', 'lease_generation': lease['lease_generation']},
    }])['status'] == 'accepted'
    call = 'sha256:' + '9' * 64
    assert store.reserve_call(run_id=lease['run_id'], lease_token_hash=token_hash,
        call_identity=call, operation_id='exa.contents', provider='deepline',
        funding_source='miner_key', amount_microusd=100, call_doc={})['status'] == 'reserved'
    assert store.mark_dispatched(run_id=lease['run_id'], lease_token_hash=token_hash, call_identity=call)['status'] == 'dispatched'
    assert store.settle_call(run_id=lease['run_id'], lease_token_hash=token_hash, call_identity=call,
        actual_microusd=100, terminal_response={'status': 200, 'call': {'call_succeeded': True}})['status'] == 'settled'
    assert complete(store, lease['run_id'], token_hash, 'accepted', output_ref='arena/test/prehold444.json')['status'] == 'accepted'
    assert store.close_scoring(ROUND, 2)['status'] == 'closed'
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            before = recovery._snapshot(cur)
            cur.execute(sql)
            after = recovery._snapshot(cur)
            assert {k:v for k,v in after.items() if k != 'hold'} == {k:v for k,v in before.items() if k != 'hold'}
            assert after['hold']['operator_paused']
            cur.execute(sql)
            assert recovery._snapshot(cur) == after
