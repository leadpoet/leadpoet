"""A lost claim response is not an authenticated failed score runner."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
from unittest.mock import patch
import uuid

import pytest

from tests.lab_arena import claim_eligibility_fast_paths509_postgres_test as fast
from tests.lab_arena import claim_admission_transition509_510_postgres_test as transition

current_database = transition.current_database

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/511-lab-arena-score-expiry-start-proof.sql'
PRE = fast.POST
POST = 'ccf306a3d6fddbe92f22f90cc2c0c401847eb4d80eb3a160cf06fa6ba6aa9987'


@pytest.fixture(scope='module')
def database():
    # The concurrent510index and complete current company/signal schema are
    # part of this fixture; our tests change only the511claim source.
    gen = transition.database.__wrapped__()
    db = next(gen)
    try:
        yield db
    finally:
        gen.close()
        # The nested hermetic fixture must stop its own local PostgreSQL.
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex((db[1]['host'], int(db[1]['port']))) != 0


@pytest.fixture(scope='module')
def originals(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        return {sig: fast.definition(cur, sig) for sig in (fast.SIG, fast.CREDIT, fast.CONTINUATION)}


@pytest.fixture(autouse=True)
def restore(database, originals):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for definition in originals.values():
            cur.execute(definition)
        cur.execute('TRUNCATE public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
    yield
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for definition in originals.values():
            cur.execute(definition)


def apply(conn):
    with conn.cursor() as cur:
        cur.execute(SQL.read_text())
    conn.commit()


def event(conn, lease, suffix, *, kind='runtime.started', content=None):
    content = content or {'status': 'starting', 'lease_generation': lease['lease_generation']}
    body = [{'event_id': str(uuid.uuid4()), 'kind': kind, 'content': content,
             'occurred_at': datetime.now(timezone.utc).replace(microsecond=0).isoformat()}]
    return fast.seed._rpc(conn, 'SELECT public.lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)',
                          (lease['run_id'], 'sha256:'+suffix*64, json.dumps(body)))


def cohort(conn, *, count=3, extra=1, started=None, explicit_errors=False):
    cfg = fast.score_seed(conn, positions=count+extra)
    # Ordinary retry is authorized by the materialized scoring plan and its
    # accepted source, as in production; a bare pending score fixture is not.
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("SELECT run_id,assignment_id,icp_position FROM public.lab_arena_runs WHERE kind='score' ORDER BY icp_position")
        items = []
        for run_id, assignment, position in cur.fetchall():
            source_assignment = assignment.removesuffix(':score')
            source_id = source_assignment+':1'
            output_ref = 'fixture/'+source_id
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,output_ref) SELECT %s,%s,round_id,submission_id,miner_hotkey,stage,icp_position,1,'execute','accepted',stage_generation,'accepted',%s FROM public.lab_arena_runs WHERE run_id=%s", (source_id, source_assignment, output_ref, run_id))
            input_hash = 'sha256:'+hashlib.sha256(source_id.encode()).hexdigest()
            cache_key = 'sha256:'+hashlib.sha256(assignment.encode()).hexdigest()
            scope = {'scorer_image_digest': cfg['scorer_image_digest'],
                     'scorer_image_reference': cfg['scorer_image_reference'],
                     'round_id': fast.seed.ROUND, 'scoring_input_hash': input_hash,
                     'cache_key': cache_key}
            cur.execute('UPDATE public.lab_arena_runs SET scored_run_id=%s,judgment_input_hash=%s,judgment_cache_key=%s,judgment_scope_doc=%s::jsonb WHERE run_id=%s', (source_id, input_hash, cache_key, json.dumps(scope), run_id))
            items.append({'scored_run_id': source_id, 'submission_id': fast.seed.SUB,
                          'icp_position': position, 'output_ref': output_ref})
        cur.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-20',stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s", (json.dumps({'work_items': items}), fast.seed.ROUND))
    conn.commit()
    leases = []
    for n, suffix in enumerate('abcdef0'[:count]):
        lease = fast.rpc(conn, suffix=suffix)
        assert lease['status'] == 'leased'
        leases.append(lease)
        if started is not None and n < started:
            assert event(conn, lease, suffix)['status'] == 'accepted'
        if explicit_errors:
            assert event(conn, lease, suffix, kind='runtime.error', content={
                'status': 'abandoned', 'failure_stage': 'runtime',
                'error_class': 'RuntimeHostError'})['status'] == 'accepted'
    return leases


def expire_clock(conn):
    """Advance only fixture lease timestamps; the production expiry RPC is real."""
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '2 seconds' WHERE status='leased'")
        cur.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{lease_expires_at}',to_jsonb(lease_expires_at)) WHERE status='leased'")
        cur.execute("UPDATE public.lab_arena_trajectory_events t SET created_at=r.lease_expires_at-interval '1 second',occurred_at=date_trunc('second',r.lease_expires_at-interval '1 second') FROM public.lab_arena_runs r WHERE r.run_id=t.run_id")
    conn.commit()


def expire(conn):
    return fast.seed._rpc(conn, 'SELECT public.lab_arena_expire_leases(%s)', (fast.seed.ROUND,))


def mutate_start(conn, fault):
    string_fields = {'runner_hotkey': fast.hotkey('wrong-runner'), 'round_id': 'wrong-round',
                     'submission_id': 'wrong-submission', 'miner_hotkey': fast.hotkey('wrong-miner'),
                     'assignment_id': 'wrong-assignment', 'run_kind': 'execute',
                     'event_kind': 'runtime.error'}
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        if fault in string_fields:
            cur.execute('UPDATE public.lab_arena_trajectory_events SET '+fault+'=%s', (string_fields[fault],))
        elif fault in ('stage', 'icp_position', 'attempt'):
            cur.execute('UPDATE public.lab_arena_trajectory_events SET '+fault+'='+fault+'+1')
        elif fault == 'run_id':
            cur.execute("UPDATE public.lab_arena_trajectory_events SET run_id='unrelated-run'")
        elif fault == 'generation':
            cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,'{lease_generation}','999')")
        elif fault == 'missing_generation':
            cur.execute("UPDATE public.lab_arena_trajectory_events SET content=content-'lease_generation'")
        elif fault == 'status':
            cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,'{status}','\"other\"')")
        elif fault in ('created_early', 'created_late', 'occurred_early', 'occurred_late'):
            column = 'created_at' if fault.startswith('created') else 'occurred_at'
            interval = '-4501 seconds' if fault.endswith('early') else '1 second'
            cur.execute('UPDATE public.lab_arena_trajectory_events t SET '+column+"=(r.claim_response->>'lease_expires_at')::timestamptz+%s::interval FROM public.lab_arena_runs r WHERE t.run_id=r.run_id", (interval,))
    conn.commit()


def test_exact_migration_security_replay_and_only_two_inference_predicates_change(database):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        cohort(conn, count=1, extra=1)
        with conn.cursor() as cur:
            before = fast.definition(cur)
            assert hashlib.sha256(before.encode()).hexdigest() == PRE
            metadata = fast.metadata(cur)
            cur.execute("SELECT pg_get_indexdef(indexrelid),indisvalid,indisready FROM pg_index WHERE indexrelid='public.lab_arena_runs_score_cooldown_claim_idx'::regclass")
            index_metadata = cur.fetchone()
            assert index_metadata[1:] == (True, True)
            rows = fast.latest.rows(cur)
            expire_before = fast.definition(cur, 'public.lab_arena_expire_leases(text)')
            cur.execute('CREATE ROLE score511_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO score511_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO score511_migrator')
        conn.commit()
        for digest, error in ((PRE, 'preimage differs'), (POST, 'postimage differs')):
            with conn.cursor() as cur, pytest.raises(pg.Error, match=error):
                cur.execute(SQL.read_text().replace(digest, '0'*64))
            conn.rollback()
            with conn.cursor() as cur:
                assert fast.definition(cur) == before and fast.metadata(cur) == metadata
                assert fast.latest.rows(cur) == rows
        with conn.cursor() as cur:
            cur.execute('GRANT EXECUTE ON FUNCTION '+fast.SIG+' TO anon')
            with pytest.raises(pg.Error, match='security shape differs'):
                cur.execute(SQL.read_text())
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute('SET ROLE score511_migrator')
            cur.execute(SQL.read_text())
            after = fast.definition(cur)
            cur.execute(SQL.read_text())
            assert fast.definition(cur) == after and fast.metadata(cur) == metadata
            cur.execute('RESET ROLE')
            assert hashlib.sha256(after.encode()).hexdigest() == POST
            expected = before
            for label in ('cooldown', 'handoff'):
                old = re.search(r'\$'+label+r'_old\$(.*?)\$'+label+r'_old\$', SQL.read_text(), re.S)[1]
                new = re.search(r'\$'+label+r'_new\$(.*?)\$'+label+r'_new\$', SQL.read_text(), re.S)[1]
                assert expected.count(old) == 1
                expected = expected.replace(old, new)
            assert after == expected
            assert fast.definition(cur, 'public.lab_arena_expire_leases(text)') == expire_before
            cur.execute("SELECT pg_get_indexdef(indexrelid),indisvalid,indisready FROM pg_index WHERE indexrelid='public.lab_arena_runs_score_cooldown_claim_idx'::regclass")
            assert cur.fetchone() == index_metadata
            assert fast.latest.rows(cur) == rows
            host_guard = '    -- lab_arena_active_host_fault_claim_guard_v1:'
            end_guard = '    -- lab_arena_zero_call_score_runner_cooldown_v1:'
            assert before[before.index(host_guard):before.index(end_guard)] == after[after.index(host_guard):after.index(end_guard)]


def test_seven_undelivered_claims_expire_normally_then_same_runner_continues(database):
    with database[0].connect(**database[1]) as conn:
        leases = cohort(conn, count=7, extra=0)
        with conn.cursor() as cur:
            before = fast.latest.rows(cur)
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE kind='execute' AND status='accepted' ORDER BY run_id")
            accepted_sources = cur.fetchall()
            cur.execute('SELECT configuration_doc,participants,stage2_scoring_plan_doc FROM public.lab_arena_rounds')
            frozen = cur.fetchall()
            cur.execute("SELECT assignment_id,scored_run_id,judgment_input_hash,judgment_cache_key,judgment_scope_doc,stage_generation FROM public.lab_arena_runs WHERE kind='score'")
            bindings = {row[0]: row[1:] for row in cur.fetchall()}
        assert expire(conn)['retried'] == 0
        with conn.cursor() as cur:
            assert fast.latest.rows(cur) == before  # no early expiry or new work
        expire_clock(conn)
        result = expire(conn)
        assert result['retried'] == 7, result
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        conn.commit()
        assert fast.rpc(conn, suffix='8') == {'status': 'no_pending'}  # reproduced post509 bug
        apply(conn)
        successor = fast.rpc(conn, suffix='9')
        assert successor['status'] == 'leased' and successor['attempt'] == 2
        assert successor['lease_generation'] == 2
        assert fast.rpc(conn, suffix='9') == successor
        with conn.cursor() as cur:
            cur.execute('SELECT scored_run_id,judgment_input_hash,judgment_cache_key,judgment_scope_doc,stage_generation FROM public.lab_arena_runs WHERE run_id=%s', (successor['run_id'],))
            assert cur.fetchone() == bindings[successor['assignment_id']]
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE kind='execute' AND status='accepted' ORDER BY run_id")
            assert cur.fetchall() == accepted_sources
            cur.execute('SELECT configuration_doc,participants,stage2_scoring_plan_doc FROM public.lab_arena_rounds')
            assert cur.fetchall() == frozen
            cur.execute('SELECT assignment_id,count(*) FILTER(WHERE status=\'leased\') FROM public.lab_arena_runs GROUP BY assignment_id')
            assert all(count <= 1 for _, count in cur.fetchall())
            cur.execute('SELECT count(*) FROM public.lab_arena_ledger')
            assert cur.fetchone() == (0,)
        assert event(conn, leases[0], 'a')['status'] == 'stale'


@pytest.mark.parametrize('started,blocked', [(0, False), (1, False), (2, False), (3, True)])
def test_three_distinct_matching_started_expiries_still_cool_down(database, started, blocked):
    with database[0].connect(**database[1]) as conn:
        cohort(conn, started=started)
        expire_clock(conn)
        assert expire(conn)['retried'] == 3
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        conn.commit()
        apply(conn)
        result = fast.rpc(conn, suffix='8')
        assert (result['status'] == 'no_pending') is blocked
        if blocked:
            assert fast.rpc(conn, suffix='9', runner=fast.hotkey('healthy-other'))['status'] == 'leased'


@pytest.mark.parametrize('fault', ['run_id', 'runner_hotkey', 'round_id', 'submission_id',
    'miner_hotkey', 'assignment_id', 'stage', 'icp_position', 'attempt', 'run_kind',
    'event_kind', 'generation', 'missing_generation', 'status', 'created_early',
    'created_late', 'occurred_early', 'occurred_late'])
def test_unbound_started_events_do_not_infer_failed_runner(database, fault):
    with database[0].connect(**database[1]) as conn:
        cohort(conn, started=3)
        expire_clock(conn)
        mutate_start(conn, fault)
        assert expire(conn)['retried'] == 3
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        conn.commit()
        apply(conn)
        assert fast.rpc(conn, suffix='8')['status'] == 'leased'


def test_single_matching_started_expiry_still_hands_off_to_another_runner(database):
    with database[0].connect(**database[1]) as conn:
        cohort(conn, count=1, extra=0, started=1)
        expire_clock(conn)
        assert expire(conn)['retried'] == 1
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        conn.commit()
        apply(conn)
        assert fast.rpc(conn, suffix='8') == {'status': 'no_pending'}
        assert fast.rpc(conn, suffix='9', runner=fast.hotkey('healthy-other'))['status'] == 'leased'


def test_explicit_authenticated_host_errors_still_quarantine_without_started(database):
    with database[0].connect(**database[1]) as conn:
        cohort(conn, explicit_errors=True)
        expire_clock(conn)
        expire(conn)
        apply(conn)
        assert fast.rpc(conn, suffix='8') == {'status': 'no_pending'}


def test_existing_cache_and_funding_fences_are_retained(database):
    with patch.object(fast, 'apply', apply):
        fast.test_active_company_cache_reservation_remains_a_claim_fence(database)
        fast.test_real_confirmed_credit_proof_remains_terminal(database)


def test_a_start_after_original_claim_window_does_not_gain_renewed_window_authority(database):
    with database[0].connect(**database[1]) as conn:
        cohort(conn, started=3)
        expire_clock(conn)
        with conn.cursor() as cur:
            # Later heartbeat expiry must not replace the original signed
            # claim-response window used to authenticate the start.
            cur.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{lease_expires_at}',to_jsonb(lease_expires_at-interval '5 minutes')) WHERE status='leased'")
        conn.commit()
        assert expire(conn)['retried'] == 3
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        conn.commit()
        apply(conn)
        assert fast.rpc(conn, suffix='8')['status'] == 'leased'


def test_retained_paid_ledger_excludes_zero_call_inference_without_erasing_costs(database):
    with database[0].connect(**database[1]) as conn:
        leases = cohort(conn, started=3)
        expire_clock(conn)
        assert expire(conn)['retried'] == 3
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
            for index, lease in enumerate(leases):
                identity = 'sha256:'+hashlib.sha256(str(index).encode()).hexdigest()
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,terminal_response) SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','miner_key',123,%s::jsonb FROM public.lab_arena_runs WHERE run_id=%s", (identity, json.dumps({'status': 200, 'call_succeeded': True}), lease['run_id']))
            cur.execute('SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id')
            before = cur.fetchall()
        conn.commit()
        apply(conn)
        successor = fast.rpc(conn, suffix='8')
        assert successor['status'] == 'leased' and successor['attempt'] == 2
        with conn.cursor() as cur:
            cur.execute('SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id')
            assert cur.fetchall() == before


def test_after511_cached_failure_retry_keeps_completed_company_authorities(database):
    with database[0].connect(**database[1]) as conn:
        apply(conn)
    transition.test_company_failure_retry_preserves_completed_authorities_and_finishes(database)


def test_after511_full_current_schema_completes_and_publishes_normal_round(database, current_database, tmp_path, monkeypatch):
    with database[0].connect(**database[1]) as conn:
        apply(conn)
    transition._assert_complete_current_round(current_database, tmp_path, monkeypatch)
