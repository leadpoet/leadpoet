"""Exact progression RPCs use control -> round/run, with unchanged final guards."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import pytest

from tests.lab_arena import baseline_infrastructure_continuation485_postgres_test as prior
from tests.lab_arena import publication_transition_timeout437_postgres_test as publication
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/490-lab-arena-progression-lock-order.sql'
HISTORICAL = ROOT / 'tests/fixtures/arena_progression490_historical_functions.sql'
TARGETS = re.findall(r"\('([^']+)','([0-9a-f]{64})','([0-9a-f]{64})',", SQL.read_text())
CONTROL = "SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0))"
CLAIM = "SELECT public.lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,4500)"

def _claim_args(rid):
    return rid, hotkey('alpha'), 'a'*32, 'sha256:'+'b'*64, 'sha256:'+'c'*64
ROUND = 'arena-2099-10-15-lock490'


def _functions(cur):
    cur.execute("SELECT oid::regprocedure::text,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' "
                "FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
    return {name: (definition, metadata) for name, definition, metadata in cur.fetchall()}


def _apply(conn):
    with conn.cursor() as cur:
        cur.execute(SQL.read_text())
    conn.commit()


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        assert db[1]['host'] == '127.0.0.1'
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            # Source-only capture from the read-only2026-10-15 catalog audit.
            # Ordinary485 fixtures omit Sep18/Oct01/Oct10 historical function
            # branches and both renamed backups. No live row or provider body
            # is copied. Every definition must match the independently reviewed
            # production preimage in the unchanged migration below.
            cur.execute(HISTORICAL.read_text())
            functions = _functions(cur)
            assert len(TARGETS) == 13
            for signature, before, _ in TARGETS:
                assert hashlib.sha256(functions[signature.removeprefix('public.')][0].encode()).hexdigest() == before
        yield db
    finally:
        gen.close()


@pytest.fixture(scope='module')
def originals(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        return _functions(cur)


@pytest.fixture(autouse=True)
def restore(database, originals):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for signature, _, _ in TARGETS:
            cur.execute(originals[signature.removeprefix('public.')][0])
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,"
                    "guard_commitment='',owner_commitment='',guard_expires_at=NULL,captured_leases='[]'")


@pytest.fixture()
def migrated(database):
    with database[0].connect(**database[1]) as conn:
        _apply(conn)
        with conn.cursor() as cur:
            return _functions(cur)


def _seed(conn, status='cancelled'):
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc,participants,benchmark_ref,rewards_enabled) "
                    "VALUES (%s,%s,'{\"integrity_policy\":\"arena_integrity_v1\"}', '[]','benchmark',false)", (ROUND, status))
    conn.commit()


def _catalog(cur):
    cur.execute("SELECT oid,to_jsonb(t) FROM pg_trigger t ORDER BY oid")
    triggers = cur.fetchall()
    cur.execute("SELECT oid,relacl,relowner,relrowsecurity,relforcerowsecurity FROM pg_class WHERE relnamespace='public'::regnamespace ORDER BY oid")
    tables = cur.fetchall()
    cur.execute('SELECT to_jsonb(a) FROM pg_default_acl a ORDER BY oid')
    return triggers, tables, cur.fetchall()


def test_exact_migration_replay_metadata_and_non_superuser(database, originals):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        before = _functions(cur)
        other = _catalog(cur)
        cur.execute('CREATE ROLE progression490_migrator NOLOGIN NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO progression490_migrator')
        cur.execute('GRANT CREATE ON SCHEMA public TO progression490_migrator')
        conn.commit()
        cur.execute('SET ROLE progression490_migrator')
        cur.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
        assert cur.fetchone() == (False,)
        _apply(conn)
        cur.execute('RESET ROLE')
        after = _functions(cur)
        assert before.keys() == after.keys()
        targets = {s.removeprefix('public.'): post for s, _, post in TARGETS}
        changed = 0
        for signature, (definition, metadata) in before.items():
            if signature in targets:
                assert after[signature][1] == metadata
                assert hashlib.sha256(after[signature][0].encode()).hexdigest() == targets[signature]
                assert after[signature][0].replace(
                    "-- Keep claims and progression in one order: control, then round/run.\n"
                    "  -- Final transition and score triggers still enforce operator holds.\n"
                    "  PERFORM pg_catalog.pg_advisory_xact_lock(\n"
                    "    pg_catalog.hashtextextended('lab-arena-claim-control', 0));\n  ", '', 1) == definition
                changed += 1
            else:
                assert after[signature] == before[signature]
        assert changed == 13 and _catalog(cur) == other
        _apply(conn)
        assert _functions(cur) == after


@pytest.mark.parametrize('drift', ['source', 'config', 'grants', 'owner', 'definer', 'posthash'])
def test_unknown_final_target_rolls_back_all_prior_changes(database, drift):
    pg, dsn = database
    signature = TARGETS[-1][0]
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        if drift == 'source':
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
            cur.execute(cur.fetchone()[0].replace('DECLARE', '-- unexpected490\nDECLARE', 1))
        elif drift == 'config':
            cur.execute('ALTER FUNCTION '+signature+" SET statement_timeout='601s'")
        elif drift == 'grants':
            cur.execute('GRANT EXECUTE ON FUNCTION '+signature+' TO anon')
        elif drift == 'owner':
            cur.execute('ALTER FUNCTION '+signature+' OWNER TO postgres')
        elif drift == 'definer':
            cur.execute('ALTER FUNCTION '+signature+' SECURITY INVOKER')
        conn.commit()
        before = _functions(cur)
        text = SQL.read_text()
        if drift == 'posthash':
            text = text.replace(TARGETS[-1][2], '0'*64)
        with pytest.raises(pg.Error, match='Arena progression490 .* differs'):
            cur.execute(text)
        conn.rollback()
        assert _functions(cur) == before
        # Restore metadata drift too; no task-owned changes survive this test.
        if drift in ('config', 'grants', 'owner', 'definer', 'source'):
            cur.execute('ALTER FUNCTION '+signature+' OWNER TO lab_arena_owner')
            cur.execute('REVOKE ALL ON FUNCTION '+signature+' FROM anon')
            cur.execute('ALTER FUNCTION '+signature+' SECURITY DEFINER')
            cur.execute('ALTER FUNCTION '+signature+" SET statement_timeout='600s'")


def _wait(pg, dsn, pid, event):
    until = time.monotonic()+4
    with pg.connect(**dsn) as observer:
        observer.autocommit = True
        with observer.cursor() as cur:
            while time.monotonic() < until:
                cur.execute('SELECT wait_event_type,wait_event FROM pg_stat_activity WHERE pid=%s', (pid,))
                row = cur.fetchone()
                if row and row[0] == 'Lock' and (event is None or row[1] == event):
                    return
                time.sleep(.005)
    raise AssertionError(f'session did not wait on {event}')


def _thread(conn, sql, params=(), commit=True):
    results, errors = [], []
    def run():
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                results.append(cur.fetchone()[0])
            if commit:
                conn.commit()
        except BaseException as error:
            errors.append(error)
            conn.rollback()
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return worker, results, errors


def _join(worker, errors, *, allowed=()):
    worker.join(5)
    assert not worker.is_alive()
    assert all(getattr(error, 'pgcode', None) in allowed for error in errors), errors


CALLS = (
    ('lab_arena_activate_preexecuted_stage2_v1(%s)', ()),
    ('lab_arena_cancel_round(%s,%s)', ('lock490-test',)),
    ('lab_arena_close_parallel_execution_v1(%s)', ()),
    ('lab_arena_close_stage(%s,1::smallint)', ()),
    ("lab_arena_open_parallel_execution_v1(%s,'[{\"submission_id\":\"x\"}]')", ()),
    ("lab_arena_open_scoring(%s,1::smallint,'[]')", ()),
    ("lab_arena_open_scoring_sep15_baseline_only_v1(%s,1::smallint,'[]')", ()),
    ("lab_arena_open_scoring_sep16_baseline_only_v1(%s,1::smallint,'[]')", ()),
    ("lab_arena_open_scoring_v2(%s,1::smallint,'[]')", ()),
    ("lab_arena_open_scoring_v3(%s,1::smallint,'[]')", ()),
    ("lab_arena_open_stage(%s,1::smallint,'[]',ARRAY[0])", ()),
    ("lab_arena_record_run_scores(%s,1::smallint,'[]')", ()),
    ("lab_arena_transition_round(%s,'scored','published','{}')", ()),
)


@pytest.mark.parametrize('call,tail', CALLS)
def test_every_target_waits_before_taking_round_lock(database, migrated, call, tail):
    pg, dsn = database
    with pg.connect(**dsn) as operation, pg.connect(**dsn) as claimant:
        _seed(claimant)
        with claimant.cursor() as b, operation.cursor() as a:
            b.execute(CONTROL)
            a.execute("SET statement_timeout='4s'")
            a.execute('SELECT pg_backend_pid()')
            pid = a.fetchone()[0]
            worker, _, errors = _thread(operation, 'SELECT public.'+call, (ROUND, *tail))
            _wait(pg, dsn, pid, 'advisory')
            b.execute('SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE NOWAIT', (ROUND,))
            assert b.fetchone() == (1,)
            claimant.commit()
            _join(worker, errors, allowed=('22023', '55000'))


@pytest.fixture(scope='module')
def proven_round(database, tmp_path_factory):
    # Build signed source evidence on the established complete service fixture.
    # Its historical helpers are intentionally absent from the isolated485
    # catalog; the tested transition/projection then runs on exact490 bodies.
    generator = current_service_database()
    source = next(generator)
    try:
        return publication.current_proven_round(source, tmp_path_factory)
    finally:
        generator.close()


def _publication_row(pg, dsn, proven, count=25):
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        return publication._expand(cur, proven, count)


def test_real_publication_old_deadlock_then_fixed_claim_progress(database, proven_round, record_property):
    pg, dsn = database
    rid, document = _publication_row(pg, dsn, proven_round)
    sql = "SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)"
    args = (rid, json.dumps(dict(publication_doc=document, published_at=document['published_at'])))
    with pg.connect(**dsn) as publisher, pg.connect(**dsn) as claimant:
        with publisher.cursor() as a, claimant.cursor() as b:
            a.execute("SET deadlock_timeout='100ms'")
            b.execute("SET deadlock_timeout='100ms'")
            a.execute('SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE', (rid,))
            b.execute(CONTROL)
            b.execute('SELECT pg_backend_pid()')
            pid = b.fetchone()[0]
            worker, _, errors = _thread(claimant, CLAIM, _claim_args(rid))
            _wait(pg, dsn, pid, None)
            failure = None
            try:
                a.execute(sql, args)
            except pg.Error as error:
                failure = error
            publisher.rollback()
            worker.join(5)
            assert not worker.is_alive()
            assert any(getattr(e, 'pgcode', None) == '40P01' for e in [failure, *errors])
    with pg.connect(**dsn) as conn:
        _apply(conn)
    with pg.connect(**dsn) as publisher, pg.connect(**dsn) as claimant:
        with publisher.cursor() as a, claimant.cursor() as b:
            b.execute(CONTROL)
            a.execute('SET ROLE lab_arena_service')
            a.execute('SELECT pg_backend_pid()')
            pid = a.fetchone()[0]
            worker, results, errors = _thread(publisher, sql, args)
            _wait(pg, dsn, pid, 'advisory')
            b.execute(CLAIM, _claim_args(rid))
            assert b.fetchone()[0]['status'] == 'no_pending'
            started = time.monotonic()
            claimant.commit()
            _join(worker, errors)
            elapsed = time.monotonic()-started
            assert len(results) == 1 and results[0]['status'] == 'ok'
            assert results[0]['round_status'] == 'published'
            record_property('publication490_seconds', elapsed)
            print(f'publication490 participants=25 scored_runs=500 elapsed_seconds={elapsed:.4f}')
            assert elapsed < 5  # Existing claim lock timeout, never raised.
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SELECT status,publication_doc FROM public.lab_arena_rounds WHERE round_id=%s', (rid,))
        assert cur.fetchone() == ('published', document)


def _score_row(pg, dsn, proven):
    rid, _ = _publication_row(pg, dsn, proven, 2)
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT run_id,per_icp_score,qualification_doc FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND stage=2 AND kind='execute' AND status='accepted' "
                    "AND per_icp_score IS NOT NULL ORDER BY run_id LIMIT 1", (rid,))
        run_id, score, qualification = cur.fetchone()
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged' WHERE round_id=%s", (rid,))
        cur.execute('UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s', (run_id,))
    return rid, run_id, json.dumps([dict(run_id=run_id, per_icp_score=float(score), qualification_doc=qualification)])


def test_real_score_projection_old_control_writer_cycle_then_fixed_progress(database, proven_round):
    pg, dsn = database
    rid, run_id, scores = _score_row(pg, dsn, proven_round)
    sql = 'SELECT public.lab_arena_record_run_scores(%s,2::smallint,%s::jsonb)'
    cancel = "SELECT public.lab_arena_cancel_round(%s,'lock490-control-writer')"
    # The other real progression RPC owns control first. The old score writer
    # holds round SHARE and its run UPDATE before its final score trigger takes
    # control; cancellation needs round UPDATE. Claims use SKIP LOCKED for run
    # candidates, so this is deliberately not claimed as a claim/run deadlock.
    with pg.connect(**dsn) as scorer, pg.connect(**dsn) as owner:
        with scorer.cursor() as a, owner.cursor() as b:
            a.execute("SET deadlock_timeout='100ms'")
            b.execute("SET deadlock_timeout='100ms'")
            a.execute('SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR SHARE', (rid,))
            a.execute('SELECT 1 FROM public.lab_arena_runs WHERE run_id=%s FOR UPDATE', (run_id,))
            b.execute(CONTROL)
            b.execute('SELECT pg_backend_pid()')
            pid = b.fetchone()[0]
            worker, _, errors = _thread(owner, cancel, (rid,), commit=False)
            _wait(pg, dsn, pid, None)
            failure = None
            try:
                a.execute(sql, (rid, scores))
            except pg.Error as error:
                failure = error
            scorer.rollback()
            worker.join(5)
            assert not worker.is_alive()
            owner.rollback()
            assert any(getattr(e, 'pgcode', None) == '40P01' for e in [failure, *errors])
    with pg.connect(**dsn) as conn:
        _apply(conn)
    with pg.connect(**dsn) as scorer, pg.connect(**dsn) as owner:
        with scorer.cursor() as a, owner.cursor() as b:
            b.execute(CONTROL)
            a.execute('SET ROLE lab_arena_service')
            a.execute('SELECT pg_backend_pid()')
            pid = a.fetchone()[0]
            worker, results, errors = _thread(scorer, sql, (rid, scores))
            _wait(pg, dsn, pid, 'advisory')
            b.execute(cancel, (rid,))
            assert b.fetchone()[0]['status'] == 'cancelled'
            owner.commit()
            _join(worker, errors)
            assert results == [{'status': 'stale', 'round_status': 'cancelled'}]
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SELECT per_icp_score FROM public.lab_arena_runs WHERE run_id=%s', (run_id,))
        assert cur.fetchone() == (None,)
        # Restore only this disposable fixture to its prior legitimate judged
        # state to verify successful score, replay and write-once controls.
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged' WHERE round_id=%s", (rid,))
        cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute('SET LOCAL ROLE lab_arena_service')
        cur.execute(sql, (rid, scores))
        assert cur.fetchone()[0] == {'status': 'ok', 'recorded': 1, 'existing': 0}
        cur.execute(sql, (rid, scores))
        assert cur.fetchone()[0] == {'status': 'ok', 'recorded': 0, 'existing': 1}
        conn.commit()
        changed = json.loads(scores)
        changed[0]['per_icp_score'] += 1
        with pytest.raises(pg.Error, match='lab_arena_score_write_once'):
            cur.execute(sql, (rid, json.dumps(changed)))
        conn.rollback()


@pytest.mark.parametrize('hold', ['paused', 'missing', 'foreign'])
@pytest.mark.parametrize('kind', ['publication', 'score'])
def test_final_hold_guards_remain_fail_closed(database, migrated, proven_round, hold, kind):
    pg, dsn = database
    if kind == 'publication':
        rid, document = _publication_row(pg, dsn, proven_round, 2)
        sql = "SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)"
        args = (rid, json.dumps(dict(publication_doc=document, published_at=document['published_at'])))
    else:
        rid, run_id, scores = _score_row(pg, dsn, proven_round)
        sql = 'SELECT public.lab_arena_record_run_scores(%s,2::smallint,%s::jsonb)'
        args = (rid, scores)
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s', (rid,))
        original_round = cur.fetchone()
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s ORDER BY run_id', (rid,))
        original_runs = cur.fetchall()
        if hold == 'paused':
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true')
        elif hold == 'missing':
            cur.execute('DELETE FROM public.lab_arena_restart_claim_control')
        else:
            cur.execute("UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('e',64),"
                        "owner_commitment='sha256:'||repeat('f',64),guard_expires_at=now()+interval '5 minutes',"
                        "candidate_commit=repeat('d',40),restart_scope='gateway',restart_phase='draining'")
        cur.execute('SET LOCAL ROLE lab_arena_service')
        with pytest.raises(pg.Error) as error:
            cur.execute(sql, args)
        assert error.value.pgcode == '55000'
        conn.rollback()
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s', (rid,))
        assert cur.fetchone() == original_round
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s ORDER BY run_id', (rid,))
        assert cur.fetchall() == original_runs


@pytest.mark.parametrize('drift', ['score', 'cost', 'identity', 'anon'])
def test_real_publication_auth_and_evidence_guards_unchanged(database, migrated, proven_round, drift):
    pg, dsn = database
    rid, document = _publication_row(pg, dsn, proven_round, 2)
    forged = deepcopy(document)
    if drift == 'score':
        forged['final_ranking'][0]['final_score'] += 1
    elif drift == 'cost':
        forged['final_ranking'][0]['cost_summary']['competition_sourcing_microusd'] += 1
    elif drift == 'identity':
        forged['final_ranking'][0]['submission_id'] = 'foreign490'
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL ROLE '+('anon' if drift == 'anon' else 'lab_arena_service'))
        with pytest.raises(pg.Error) as error:
            cur.execute("SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)",
                        (rid, json.dumps(dict(publication_doc=forged, published_at=forged['published_at']))))
        if drift == 'anon':
            assert error.value.pgcode == '42501'
        conn.rollback()
        cur.execute('SELECT status,publication_doc FROM public.lab_arena_rounds WHERE round_id=%s', (rid,))
        assert cur.fetchone() == ('scored', None)
