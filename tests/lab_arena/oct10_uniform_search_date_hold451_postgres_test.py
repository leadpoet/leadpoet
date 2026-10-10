"""The fresh post448 owner hold permits paid judge drain and no new work."""
import pytest

from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import oct10_uniform_search_date_hold451_fixtures as recovery
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete, hotkey

database = recovery.database
migrated = recovery.migrated


@pytest.fixture
def unheld(database, migrated):
    yield from recovery._seeded(database, apply_hold=False)


def test_hold_preserves_outputs_costs_and_allows_current_judge_drain(unheld):
    psycopg, dsn = unheld
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            sql = recovery._hold_render(cur)
            lease, token, _, _ = claim(store,recovery.ROUND,hotkey('judge-hold451'))
            assert lease['status'] == 'leased', lease
            before = recovery._snapshot(cur)
            cur.execute(sql)
            after = recovery._snapshot(cur)
            assert {k:v for k,v in before.items() if k!='hold'} == {k:v for k,v in after.items() if k!='hold'}
            assert after['hold']['operator_paused']
            cur.execute(sql)
            assert recovery._snapshot(cur) == after
            refused, _, _, _ = claim(store,recovery.ROUND,hotkey('judge-hold451-next'))
            assert refused['status'] == 'paused', refused
            # A normal in-flight bill call extends the lease after this hold.
            identity='sha256:'+'4'*64
            assert store.reserve_call(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),
                call_identity=identity,operation_id='openrouter.chat',provider='openrouter',
                funding_source='miner_key',amount_microusd=100,call_doc={},lease_ttl_seconds=120)['status']=='reserved'
            assert store.mark_dispatched(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),call_identity=identity)['status']=='dispatched'
            assert store.mark_uncertain(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),call_identity=identity,call_doc={})['status']=='uncertain'
            assert complete(store,lease['run_id'],hash_lease_token(token),'accepted',output_ref='arena/test/drained451.json')['status']=='accepted'
            cur.execute(sql)
            assert recovery._snapshot(cur)['hold'] == after['hold']
            with pytest.raises(psycopg.Error,match='lab_arena_round_progression_paused'):
                cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged' WHERE round_id=%s",(recovery.ROUND,))
            cur.execute('ROLLBACK')


@pytest.mark.parametrize('mutation',[
    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,actor_ref='foreign',pause_reason='foreign' WHERE singleton",
    "UPDATE public.lab_arena_restart_claim_control SET guard_generation=999 WHERE singleton",
    "UPDATE public.lab_arena_rounds SET published_at=now() WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_rounds SET reward_activated_at=now() WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{scorer_image_digest}','\"wrong\"') WHERE round_id='arena-2026-10-10'",
    "UPDATE public.lab_arena_runs SET output_ref='arena/wrong.json' WHERE round_id='arena-2026-10-10' AND kind='execute' AND status='accepted'",
    "UPDATE public.lab_arena_runs SET result_doc='{}' WHERE round_id='arena-2026-10-10-r445archive' AND kind='score'",
    "UPDATE public.lab_arena_runs SET result_doc='{}' WHERE round_id='arena-2026-10-10-r448archive' AND kind='score'",
    "UPDATE public.lab_arena_ledger SET entry_doc='{\"retained_provider_body\":\"changed\"}' WHERE round_id='arena-2026-10-10-r445archive'",
    "UPDATE public.lab_arena_ledger SET entry_doc='{\"retained_provider_body\":\"changed\"}' WHERE round_id='arena-2026-10-10-r448archive'",
    "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal",
])
def test_hold_rejects_wrong_proof_with_no_writes(unheld,mutation):
    psycopg,dsn=unheld
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            sql=recovery._hold_render(cur)
            before=recovery._snapshot(cur)
            cur.execute('BEGIN')
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute(mutation)
            cur.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(psycopg.Error,match='Oct10 uniform hold451'):
                cur.execute(sql)
            cur.execute('ROLLBACK')
            assert recovery._snapshot(cur)==before


@pytest.mark.parametrize('mutation',[
    "ALTER FUNCTION public.lab_arena_open_stage(text,smallint,jsonb,integer[]) SET search_path=public,pg_catalog",
    "ALTER FUNCTION public.lab_arena_open_scoring_v2(text,smallint,jsonb) SECURITY INVOKER",
])
def test_fresh_hold_snapshot_cannot_authorize_changed_canonical_function(unheld,mutation):
    psycopg,dsn=unheld
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            before=recovery._snapshot(cur)
            cur.execute('BEGIN')
            cur.execute(mutation)
            # The independently captured448 postimage still binds canonical SQL.
            sql=recovery._hold_render(cur)
            with pytest.raises(psycopg.Error,match='Oct10 uniform hold451 terminal inventory'):
                cur.execute(sql)
            cur.execute('ROLLBACK')
            assert recovery._snapshot(cur)==before


def test_archive_hash_phase_keeps_live_writes_free_and_archives_locked(unheld):
    import queue
    import threading
    import time

    psycopg, dsn = unheld
    barrier = 451202610
    result, pid_queue = [], queue.Queue()
    with psycopg.connect(**dsn) as owner:
        owner.autocommit = True
        with owner.cursor() as cur:
            sql = recovery._hold_render(cur)
            owner.commit()  # Release the capture transaction before the worker CAS.
            marker = '  SELECT (WITH archive AS MATERIALIZED ('
            assert sql.count(marker) == 1
            sql = sql.replace(marker, f'  PERFORM pg_advisory_xact_lock({barrier});\n'+marker)
            cur.execute('SELECT pg_advisory_lock(%s)', (barrier,))

            def apply_hold():
                try:
                    with psycopg.connect(**dsn) as conn:
                        conn.autocommit = True
                        with conn.cursor() as worker:
                            worker.execute('SELECT pg_backend_pid()')
                            pid_queue.put(worker.fetchone()[0])
                            worker.execute(sql)
                    result.append(None)
                except Exception as exc:
                    result.append(exc)

            thread = threading.Thread(target=apply_hold)
            thread.start()
            try:
                pid = pid_queue.get(timeout=2)
                deadline = time.monotonic()+2
                while True:
                    cur.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=%s AND locktype='advisory' AND NOT granted)", (pid,))
                    if cur.fetchone()[0]:
                        break
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                with psycopg.connect(**dsn) as active, active.cursor() as write:
                    write.execute("SET LOCAL lock_timeout='300ms'")
                    # A normal score completion field and normal source-score
                    # projection both remain writable during historical proof.
                    write.execute("UPDATE public.lab_arena_runs SET updated_at=updated_at+interval '1 second' WHERE run_id='old452-score-2'")
                    assert write.rowcount == 1
                    write.execute("UPDATE public.lab_arena_runs SET qualification_doc='{\"companies\":[]}',per_icp_score=0 WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' AND status='accepted')", (recovery.ROUND,))
                    assert write.rowcount == 1
                    write.execute("SELECT pg_try_advisory_xact_lock(pg_catalog.hashtextextended('lab-arena-claim-control',0))")
                    assert write.fetchone()[0] is True
                for statement in (
                    "UPDATE public.lab_arena_rounds SET updated_at=updated_at+interval '1 second' WHERE round_id='arena-2026-10-10-r448archive'",
                    "UPDATE public.lab_arena_runs SET updated_at=updated_at+interval '1 second' WHERE run_id='old448-score-1'",
                    "UPDATE public.lab_arena_submissions SET source_size_bytes=source_size_bytes+1 WHERE round_id='arena-2026-10-10-r448archive'",
                ):
                    with psycopg.connect(**dsn) as archived, archived.cursor() as write:
                        write.execute("SET LOCAL lock_timeout='100ms'")
                        write.execute('SET LOCAL session_replication_role=replica')
                        with pytest.raises(psycopg.Error, match='lock timeout'):
                            write.execute(statement)
                        archived.rollback()
            finally:
                cur.execute('SELECT pg_advisory_unlock(%s)', (barrier,))
                thread.join(timeout=10)
            assert not thread.is_alive()
            assert result == [None], result
            assert recovery._snapshot(cur)['hold']['operator_paused'] is True


@pytest.mark.parametrize("pathway", ["trajectory", "score_projection"])
def test_hold_fence_allows_trajectory_and_round_share_score_writer(unheld, pathway):
    """Reproduce both production lock orders without live-table/round locks."""
    import json
    import queue
    import threading
    import time

    from lab_arena.trajectory import event

    psycopg, dsn = unheld
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    barrier = 451202611
    results, hold_pid, score_pid = {}, queue.Queue(), queue.Queue()
    with psycopg.connect(**dsn) as owner:
        owner.autocommit = True
        with owner.cursor() as cur:
            lease, token, _, _ = claim(store,recovery.ROUND,hotkey('judge-race451'))
            assert lease['status'] == 'leased'
            # record_run_scores accepts the ordinary judged stage. The hold
            # binds this fresh state and permits no further progression.
            if pathway == 'score_projection':
                cur.execute('SET session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged',stage_generation=stage_generation+1,status_generation=status_generation+1 WHERE round_id=%s", (recovery.ROUND,))
                cur.execute('SET session_replication_role=origin')
            cur.execute("SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' AND stage=2 AND status='accepted' AND per_icp_score IS NULL", (recovery.ROUND,))
            source_run = cur.fetchone()[0]
            assert source_run
            sql = recovery._hold_render(cur)
            owner.commit()
            marker = '  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;'
            assert sql.count(marker) == 1
            sql = sql.replace(marker, f'  PERFORM pg_advisory_xact_lock({barrier});\n'+marker)
            cur.execute('SELECT pg_advisory_lock(%s)', (barrier,))

            def apply_hold():
                try:
                    with psycopg.connect(**dsn) as conn:
                        conn.autocommit = True
                        with conn.cursor() as worker:
                            worker.execute('SELECT pg_backend_pid()')
                            hold_pid.put(worker.fetchone()[0])
                            worker.execute(sql)
                    results['hold'] = 'applied'
                except Exception as exc:
                    results['hold'] = exc

            def record_scores():
                try:
                    with psycopg.connect(**dsn) as conn, conn.cursor() as worker:
                        worker.execute('SELECT pg_backend_pid()')
                        score_pid.put(worker.fetchone()[0])
                        # This is the exact round-before-advisory order used by
                        # the production score RPC, not a changed test function.
                        worker.execute('SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR SHARE', (recovery.ROUND,))
                        worker.execute('SELECT public.lab_arena_record_run_scores(%s,2::smallint,%s::jsonb)',
                            (recovery.ROUND,json.dumps([{'run_id':source_run,'per_icp_score':0,'qualification_doc':{'companies':[]}}])))
                    results['score'] = 'unexpected write'
                except psycopg.Error as exc:
                    results['score'] = (exc.pgcode, 'lab_arena_round_progression_paused' in str(exc))

            def wait_advisory(pid):
                deadline = time.monotonic()+3
                while True:
                    cur.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=%s AND locktype='advisory' AND NOT granted)", (pid,))
                    if cur.fetchone()[0]:
                        return
                    assert time.monotonic() < deadline
                    time.sleep(0.01)

            hold = threading.Thread(target=apply_hold)
            scores = threading.Thread(target=record_scores)
            hold.start()
            try:
                wait_advisory(hold_pid.get(timeout=2))
                # Exact authenticated production trajectory RPC must remain
                # usable while the hold owns the claim-control advisory fence.
                if pathway == 'trajectory':
                    assert store.append_trajectory_events(lease['run_id'],hash_lease_token(token),
                        [event('runtime.stdout',{'text':'concurrent hold451 proof'})])['status'] == 'accepted'
                    assert complete(store,lease['run_id'],hash_lease_token(token),'accepted',
                        output_ref='arena/test/concurrent-drain451.json')['status'] == 'accepted'
                else:
                    scores.start()
                    wait_advisory(score_pid.get(timeout=2))
            finally:
                cur.execute('SELECT pg_advisory_unlock(%s)', (barrier,))
                hold.join(timeout=10)
                if scores.ident is not None:
                    scores.join(timeout=10)
            assert not hold.is_alive() and not scores.is_alive()
            expected = {'hold':'applied'}
            if pathway == 'score_projection':
                expected['score'] = ('55000',True)
            assert results == expected, results
            assert recovery._snapshot(cur)['hold']['operator_paused'] is True
            cur.execute('SELECT per_icp_score FROM public.lab_arena_runs WHERE run_id=%s', (source_run,))
            assert cur.fetchone()[0] is None
