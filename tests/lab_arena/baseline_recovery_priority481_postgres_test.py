"""Existing eligible baseline recovery precedes miners without bypassing guards."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import uuid

import pytest

from tests.lab_arena import bounded_baseline_recovery470_postgres_test as recovery
from tests.lab_arena import paid_host_quarantine480_postgres_test as previous
from tests.lab_arena.execute_host_cooldown396_postgres_test import SIGNATURE, _hash, _security
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/481-lab-arena-baseline-recovery-claim-priority.sql'
PRE = previous.POST
POST = '6432e422112cb17f0e487a18d355f881cc8d4e6a17889f93b89ad55a21017de1'
ROUND, BASELINE, RUNNER = recovery.ROUND, recovery.SUB, recovery.RUNNER
OTHER_RUNNER = hotkey('baseline481-other-runner')
MINERS = ('priority481-miner-a', 'priority481-miner-b')


def claim(conn, *, runner=RUNNER):
    request = uuid.uuid4().hex
    return recovery.rpc(conn,
        'lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,4500)',
        (ROUND, runner, request, 'sha256:' + request * 2, recovery.TOKEN))


def preserved(conn):
    """Every completed baseline row and frozen input, including failed parents."""
    with conn.cursor() as cur:
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s', (ROUND,))
        row = cur.fetchone()[0]
        cur.execute('SELECT to_jsonb(s) FROM public.lab_arena_submissions s ORDER BY submission_id')
        submissions = cur.fetchall()
        cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r "
                    "WHERE submission_id=%s AND status IN ('accepted','failed') ORDER BY run_id", (BASELINE,))
        completed = cur.fetchall()
        cur.execute('SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id')
        ledger = cur.fetchall()
    return row, submissions, completed, ledger


def seed(conn):
    with conn.cursor() as cur:
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,"
                    "guard_commitment='',owner_commitment='',guard_expires_at=NULL,"
                    "candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'")
    conn.commit()
    assignment, cfg = recovery.seed(conn)
    cfg.update(stage_1_icp_count=5, stage_2_icp_count=5,
               runner_hotkeys=[RUNNER, OTHER_RUNNER])
    participants = [dict(submission_id=sub, miner_hotkey=hotkey(sub), is_king=False)
                    for sub in MINERS]
    participants.append(dict(submission_id=BASELINE, miner_hotkey=recovery.MINER, is_king=True))
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb,participants=%s::jsonb '
                    'WHERE round_id=%s', (json.dumps(cfg), json.dumps(participants), ROUND))
        for sub in MINERS:
            cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,"
                        "status,is_king,source_ref,source_size_bytes,code_review_status,code_review_doc,"
                        "code_review_claim,code_review_started_at,code_review_attempts) "
                        "VALUES (%s,%s,%s,'frozen',false,%s,100,'passed','{}',"
                        "'sha256:'||repeat('a',64),now(),1)",
                        (sub, ROUND, hotkey(sub), f'arena/{ROUND}/sources/{sub}.tar.gz'))
            for position in (0, 1):
                item = f'{ROUND}:{sub}:2:{position}'
                cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,"
                            "miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation) "
                            "VALUES (%s,%s,%s,%s,%s,2,%s,1,'execute','pending',5)",
                            (item + ':1', item, ROUND, sub, hotkey(sub), position))
        # Nine valid completed members remain fixed; only the missing position 0
        # is queued through the real recovery RPC after these fixture inserts.
        for position in range(1, 10):
            item = f'{ROUND}:{BASELINE}:1:{position}'
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,"
                        "miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,"
                        "output_ref,per_icp_score) VALUES (%s,%s,%s,%s,%s,1,%s,2,'execute','accepted',1,"
                        "'accepted',%s,%s)",
                        (item + ':2', item, ROUND, BASELINE, recovery.MINER, position,
                         f'arena/{ROUND}/outputs/{position}.json', position))
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,"
                        "miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,"
                        "output_ref,scored_run_id,judgment_cache_key,judgment_input_hash) "
                        "VALUES (%s,%s,%s,%s,%s,1,%s,1,'score','accepted',3,'accepted',%s,%s,%s,%s)",
                        (item + ':score:1', item + ':score', ROUND, BASELINE, recovery.MINER, position,
                         f'arena/{ROUND}/judgments/{position}.json', item + ':2',
                         'sha256:' + str(position) * 64, 'sha256:' + str(position) * 64))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    assert recovery.recover(conn) == {'status': 'ok', 'queued': 1}
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena__baseline_recovery_phase_v1(d,r) '
                    'FROM public.lab_arena_runs r JOIN public.lab_arena_rounds d USING(round_id) '
                    'WHERE r.run_id=%s', (assignment + ':3',))
        assert cur.fetchone()[0] is True
    return assignment + ':3'


@pytest.fixture(scope='module')
def database():
    generator = previous.database.__wrapped__()
    db = next(generator)
    try:
        with db[0].connect(**db[1]) as conn:
            with conn.cursor() as cur:
                cur.execute(previous.SQL.read_text())
                assert _hash(cur) == PRE
                pending = seed(conn)
                before = preserved(conn)
                # Reproduce the exact production starvation: the baseline is
                # eligible, but the first frozen miner wins before this patch.
                old_claim = claim(conn)
                assert old_claim['submission_id'] == MINERS[0]
                assert old_claim['run_id'] != pending
                assert preserved(conn) == before
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURE,))
                old_definition = cur.fetchone()[0]
                security = _security(cur)
                sql = SQL.read_text()
                for wrong, message in ((sql.replace(PRE, '0' * 64, 1), 'preimage differs'),
                                       (sql.replace(POST, '0' * 64, 1), 'readback differs')):
                    with pytest.raises(db[0].Error, match=message):
                        cur.execute(wrong)
                    cur.execute('ROLLBACK')
                    assert _hash(cur) == PRE and preserved(conn) == before
                cur.execute(sql)
                assert _hash(cur) == POST
                cur.execute(sql)
                assert _hash(cur) == POST and _security(cur) == security
                assert preserved(conn) == before
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURE,))
                new_definition = cur.fetchone()[0]
                # Every WHERE gate, lock, accounting path and lease write is
                # verbatim. The only difference is the leading sort key.
                addition = """    -- lab_arena_baseline_recovery_priority_v1: finish the existing baseline
    -- recovery before ordinary miner work; every admission guard stays above.
    CASE WHEN runs.kind = 'execute' AND runs.stage = 1
              AND public.lab_arena__baseline_recovery_phase_v1(v_round,runs)
      THEN 0 ELSE 1 END,
"""
                assert new_definition == old_definition.replace('  ORDER BY\n', '  ORDER BY\n' + addition)
        yield db
    finally:
        generator.close()


def test_existing_baseline_first_then_original_miner_order_and_preserved_nine(database):
    with database[0].connect(**database[1]) as conn:
        pending = seed(conn)
        before = preserved(conn)
        first = claim(conn)
        assert (first['run_id'], first['stage'], first['attempt']) == (pending, 1, 3)
        assert preserved(conn) == before
        miners = [claim(conn) for _ in range(4)]
        assert [(r['submission_id'], r['icp_position']) for r in miners] == [
            (MINERS[0], 0), (MINERS[0], 1), (MINERS[1], 0), (MINERS[1], 1)]
        assert preserved(conn) == before
        with conn.cursor() as cur:
            cur.execute("SELECT kind,count(*) FROM public.lab_arena_runs WHERE submission_id=%s "
                        "AND status='accepted' GROUP BY kind ORDER BY kind", (BASELINE,))
            assert cur.fetchall() == [('execute', 9), ('score', 9)]


@pytest.mark.parametrize('fault', ['parent_model_error', 'stale_generation', 'expired', 'paused', 'guarded'])
def test_priority_never_admits_ineligible_expired_or_paused_work(database, fault):
    with database[0].connect(**database[1]) as conn:
        pending = seed(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault == 'parent_model_error':
                cur.execute("UPDATE public.lab_arena_runs SET terminal_cause='model_error' "
                            "WHERE run_id=%s", (pending[:-1] + '2',))
            elif fault == 'stale_generation':
                cur.execute('UPDATE public.lab_arena_runs SET stage_generation=4 WHERE run_id=%s', (pending,))
            elif fault == 'expired':
                cutoff = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                            "jsonb_set(configuration_doc,'{schedule,stage_2_close}',to_jsonb(%s::text))",
                            (cutoff,))
            elif fault == 'paused':
                cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true')
            else:
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET "
                            "guard_commitment='sha256:'||repeat('a',64),"
                            "owner_commitment='sha256:'||repeat('b',64),"
                            "guard_expires_at=now()+interval '1 hour',candidate_commit=repeat('c',40),"
                            "restart_scope='all',restart_phase='draining'")
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        before = preserved(conn)
        response = claim(conn)
        if fault in ('paused', 'guarded'):
            assert response == {'status': 'paused'}
        elif fault == 'expired':
            assert response == {'status': 'no_pending'}
        else:
            assert response['submission_id'] == MINERS[0]
        assert preserved(conn) == before
        with conn.cursor() as cur:
            cur.execute('SELECT status FROM public.lab_arena_runs WHERE run_id=%s', (pending,))
            assert cur.fetchone()[0] == 'pending'


def test_previous_runner_handoff_remains_authoritative(database):
    with database[0].connect(**database[1]) as conn:
        pending = seed(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_runs SET previous_runner_hotkey=%s WHERE run_id=%s',
                        (RUNNER, pending))
            cur.execute("UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=%s,"
                        "lease_expires_at=now()+interval '5 minutes' WHERE run_id=%s",
                        (OTHER_RUNNER, f'{ROUND}:{MINERS[1]}:2:0:1'))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert claim(conn)['run_id'] != pending
        assert claim(conn, runner=OTHER_RUNNER)['run_id'] == pending


@pytest.mark.parametrize('paid', [False, True])
def test_round_host_quarantine_blocks_priority_but_healthy_runner_can_claim(database, paid):
    with database[0].connect(**database[1]) as conn:
        pending = seed(conn)
        # Three distinct failed host assignments satisfy the existing quarantine
        # predicate. They remain historical; the new priority cannot override it.
        faults = []
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            for position in range(3):
                item = f'{ROUND}:{MINERS[0]}:2:{position}:2'
                faults.append(item)
                cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,"
                            "miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,runner_hotkey,"
                            "lease_expires_at,terminal_cause,lease_generation) VALUES (%s,%s,%s,%s,%s,2,%s,2,'execute',"
                            "'failed',5,%s,now()-interval '1 second','lease_expired',1)",
                            (item, item[:-2], ROUND, MINERS[0], hotkey(MINERS[0]), position, RUNNER))
                cur.execute("INSERT INTO public.lab_arena_trajectory_events(run_id,event_id,round_id,"
                            "submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_identifier,stage,"
                            "icp_position,attempt,run_kind,model_role,event_kind,occurred_at,content) "
                            "SELECT run_id,%s,round_id,submission_id,miner_hotkey,runner_hotkey,assignment_id,"
                            "'icp-'||icp_position,stage,icp_position,attempt,kind,'miner','runtime.error',now(),"
                            "'{\"status\":\"abandoned\",\"failure_stage\":\"runtime\",\"error_class\":\"RuntimeHostError\"}'::jsonb "
                            "FROM public.lab_arena_runs WHERE run_id=%s", (str(uuid.uuid4()), item))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        if paid:
            previous._paid_faults(conn, faults, expired=True)
            assert previous._proofs(conn, faults) == [True] * 3
        before = preserved(conn)
        assert claim(conn) == {'status': 'no_pending'}
        assert claim(conn, runner=OTHER_RUNNER)['run_id'] == pending
        assert preserved(conn) == before


def test_concurrent_claims_lease_the_same_recovery_only_once(database):
    with database[0].connect(**database[1]) as conn:
        pending = seed(conn)
        before = preserved(conn)
    barrier = threading.Barrier(4)

    def one():
        with database[0].connect(**database[1]) as conn:
            barrier.wait(timeout=10)
            return claim(conn)

    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(lambda _: one(), range(4)))
    assert all(r['status'] == 'leased' for r in claims)
    assert len({r['run_id'] for r in claims}) == 4
    assert sum(r['run_id'] == pending for r in claims) == 1
    with database[0].connect(**database[1]) as conn:
        assert preserved(conn) == before


@pytest.mark.parametrize('mutation,restoration', [
    (f'ALTER FUNCTION {SIGNATURE} OWNER TO postgres',
     f'ALTER FUNCTION {SIGNATURE} OWNER TO lab_arena_owner'),
    (f'GRANT EXECUTE ON FUNCTION {SIGNATURE} TO PUBLIC',
     f'REVOKE EXECUTE ON FUNCTION {SIGNATURE} FROM PUBLIC'),
])
def test_replay_rejects_changed_owner_or_acl(database, mutation, restoration):
    with database[0].connect(**database[1]) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(mutation)
            try:
                with pytest.raises(database[0].Error, match='security shape differs'):
                    cur.execute(SQL.read_text())
                cur.execute('ROLLBACK')
            finally:
                cur.execute(restoration)
            assert _hash(cur) == POST


@pytest.mark.parametrize('recover_successfully', [True, False])
def test_priority_release_signed_runner_completes_or_blocks_publication(
    database, tmp_path, monkeypatch, recover_successfully,
):
    # Use the exact 481-upgraded claim RPC through execution, provider billing,
    # completion and judging. The negative keeps publication/rewards blocked.
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert _hash(cur) == POST
    recovery.exercise_signed_baseline_recovery(
        database, tmp_path, monkeypatch, recover_successfully, current_publication=False,
    )
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert _hash(cur) == POST
