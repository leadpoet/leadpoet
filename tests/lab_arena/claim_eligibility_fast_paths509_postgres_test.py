"""Real claim RPC retains admission while skipping already-satisfied proofs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import threading

import pytest

from tests.lab_arena import flat_per_icp_cost500_postgres_test as latest
from tests.lab_arena import stage_cutoff_drain420_postgres_test as seed
from tests.lab_arena import miner_credit_terminal484_postgres_test as credit
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/509-lab-arena-claim-eligibility-fast-paths.sql'
SIG = 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'
CREDIT = 'public.lab_arena_miner_credit_failure_v1(text)'
CONTINUATION = 'public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)'
PRE = 'aa00d7f981f0c255d85e84d650b6986ed0d400feccebacb1fc9aa702192c3b52'
POST = '6d9bdb660a476838c42bee9dd9de5588be9f19f41ee32ea167b1f47bcafbf2ef'


@pytest.fixture(scope='module')
def database():
    yield from latest.database_with_flat_cost_migration()


def definition(cur, signature=SIG):
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
    return cur.fetchone()[0]


def metadata(cur):
    cur.execute("SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (SIG,))
    return cur.fetchone()[0]


@pytest.fixture(scope='module')
def originals(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        return {sig: definition(cur, sig) for sig in (SIG, CREDIT, CONTINUATION)}


@pytest.fixture(autouse=True)
def restore(database, originals):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for text in originals.values():
            cur.execute(text)
    yield
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for text in originals.values():
            cur.execute(text)


def apply(conn):
    with conn.cursor() as cur:
        cur.execute(SQL.read_text())
    conn.commit()


def score_seed(conn, *, deadline=None, attempt=1, generation=7, positions=3):
    cfg = seed._seed(conn, kind='score', positions=positions)
    cfg['schedule']['final_scoring_close'] = seed._stamp(deadline or datetime.now(timezone.utc) + timedelta(hours=1))
    cfg['max_attempts_per_assignment'] = 2
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s', (json.dumps(cfg), seed.ROUND))
        if attempt != 1:
            cur.execute("UPDATE public.lab_arena_runs SET attempt=%s,run_id=assignment_id||':'||%s::text WHERE round_id=%s", (attempt, attempt, seed.ROUND))
        cur.execute('UPDATE public.lab_arena_runs SET stage_generation=%s,created_at=clock_timestamp()-interval \'20 minutes\' WHERE round_id=%s', (generation, seed.ROUND))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    return cfg


def rpc(conn, *, suffix='a', runner=seed.RUNNER, excluded=()):
    with conn.cursor() as cur:
        cur.execute('SET LOCAL ROLE lab_arena_service')
        cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,7,20,%s,%s,%s,%s,4500)',
                    (seed.ROUND, runner, list(excluded), suffix*32, 'sha256:'+suffix*64, 'sha256:'+suffix*64))
        result = cur.fetchone()[0]
    conn.commit()
    return result


def helper(conn, signature, result):
    """Local-only control: fail if an unnecessary proof is evaluated."""
    with conn.cursor() as cur:
        old = definition(cur, signature)
        start = old.index('AS $function$')
        body = "BEGIN RAISE EXCEPTION 'unexpected expensive proof'; END;" if result == 'raise' else 'BEGIN RETURN '+result+'; END;'
        cur.execute(old[:start]+'AS $function$\n'+body+'\n$function$\n')
    conn.commit()


def test_exact_authority_idempotence_and_rows_unchanged(database):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        score_seed(conn)
        active = rpc(conn)
        with conn.cursor() as cur:
            before = definition(cur)
            assert hashlib.sha256(before.encode()).hexdigest() == PRE
            security = metadata(cur)
            rows = latest.rows(cur)
            cur.execute('CREATE ROLE claim509_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO claim509_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO claim509_migrator')
            cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE claim509_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
        conn.commit()
        for old, message in ((PRE, 'preimage differs'), (POST, 'postimage differs')):
            with conn.cursor() as cur, pytest.raises(pg.Error, match=message):
                cur.execute(SQL.read_text().replace(old, '0'*64))
            conn.rollback()
            with conn.cursor() as cur:
                assert definition(cur) == before and metadata(cur) == security
                assert latest.rows(cur) == rows
        with conn.cursor() as cur:
            cur.execute('GRANT EXECUTE ON FUNCTION '+SIG+' TO anon')
            with pytest.raises(pg.Error, match='security shape differs'):
                cur.execute(SQL.read_text())
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute('SET ROLE claim509_migrator')
            cur.execute(SQL.read_text())
            after = definition(cur)
            cur.execute(SQL.read_text())
            assert definition(cur) == after and metadata(cur) == security
            cur.execute('RESET ROLE')
            assert hashlib.sha256(after.encode()).hexdigest() == POST
            expected = before
            for label in ('credit', 'window'):
                old = re.search(r'\$'+label+r'_old\$(.*?)\$'+label+r'_old\$', SQL.read_text(), re.S)[1]
                new = re.search(r'\$'+label+r'_new\$(.*?)\$'+label+r'_new\$', SQL.read_text(), re.S)[1]
                assert expected.count(old) == 1
                expected = expected.replace(old, new)
            assert after == expected  # all other locks/gates/cache/funding code unchanged
            assert latest.rows(cur) == rows
            for role in ('anon', 'authenticated', 'service_role'):
                cur.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, SIG))
                assert cur.fetchone() == (False,)
        conn.commit()
        assert rpc(conn) == active  # exact pre-migration signed claim replay


def test_current_first_attempt_skips_both_expensive_helpers(database):
    with database[0].connect(**database[1]) as conn:
        score_seed(conn)
        apply(conn)
        helper(conn, CREDIT, 'raise')
        helper(conn, CONTINUATION, 'raise')
        assert rpc(conn)['status'] == 'leased'


@pytest.mark.parametrize('attempt,deadline_seconds,credited,expected', [
    (1, 3600, True, 'leased'), (2, 3600, True, 'leased'),
    (3, 3600, True, 'no_pending'), (3, 3600, False, 'leased'),
    (2, -1, True, 'no_pending'), (1, -1, False, 'no_pending'),
])
def test_credit_and_frozen_window_fallbacks(database, attempt, deadline_seconds, credited, expected):
    with database[0].connect(**database[1]) as conn:
        score_seed(conn, attempt=attempt, deadline=datetime.now(timezone.utc)+timedelta(seconds=deadline_seconds))
        apply(conn)
        helper(conn, CREDIT, 'TRUE' if credited else 'FALSE')
        assert rpc(conn)['status'] == expected


@pytest.mark.parametrize('generation', [6, 8])
def test_stale_generation_needs_existing_continuation_proof(database, generation):
    with database[0].connect(**database[1]) as conn:
        score_seed(conn, generation=generation)
        apply(conn)
        assert rpc(conn) == {'status': 'no_pending'}


@pytest.mark.parametrize('fault', ['excluded', 'unreviewed', 'paused'])
def test_independent_admission_fences_still_reject(database, fault):
    with database[0].connect(**database[1]) as conn:
        score_seed(conn)
        apply(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault == 'unreviewed':
                cur.execute("UPDATE public.lab_arena_submissions SET code_review_status='error'")
            if fault == 'paused':
                cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true')
        conn.commit()
        excluded = [hotkey(seed.SUB)] if fault == 'excluded' else []
        assert rpc(conn, excluded=excluded)['status'] == ('paused' if fault == 'paused' else 'no_pending')


def test_real_confirmed_credit_proof_remains_terminal(database):
    with database[0].connect(**database[1]) as conn:
        assignment, _ = credit.seed(conn)
        credit.pending(conn, assignment, 3)
        apply(conn)
        assert credit.proved(conn, assignment+':3') is True
        # Real funding proof and expired owner window keep the successor closed.
        assert credit.base.claim(conn)['status'] == 'no_pending'


def test_where_true_null_truth_table_and_strict_boundary(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('''SELECT count(*) FROM(VALUES(TRUE),(FALSE),(NULL::boolean)) a(cheap)
          CROSS JOIN(VALUES(TRUE),(FALSE),(NULL::boolean)) b(proof)
          WHERE ((NOT proof OR cheap) IS TRUE) IS DISTINCT FROM
                ((CASE WHEN cheap THEN TRUE ELSE NOT proof END) IS TRUE)''')
        assert cur.fetchone() == (0,)
        cur.execute('''SELECT count(*) FROM(VALUES(TRUE),(FALSE),(NULL::boolean)) a(non_score)
          CROSS JOIN(VALUES(TRUE),(FALSE),(NULL::boolean)) b(continuation)
          CROSS JOIN(VALUES(TRUE),(FALSE),(NULL::boolean)) c(before_deadline)
          WHERE ((non_score OR continuation OR before_deadline) IS TRUE) IS DISTINCT FROM
            ((CASE WHEN non_score OR before_deadline THEN TRUE ELSE continuation END) IS TRUE)''')
        assert cur.fetchone() == (0,)
        cur.execute("SELECT moment < deadline FROM (VALUES ('2026-10-20T14:00:01.999999Z'::timestamptz),('2026-10-20T14:00:02Z'::timestamptz),('2026-10-20T14:00:02.000001Z'::timestamptz)) t(moment) CROSS JOIN (VALUES ('2026-10-20T14:00:02Z'::timestamptz)) c(deadline)")
        assert cur.fetchall() == [(True,), (False,), (False,)]


@pytest.mark.parametrize('fault', ['missing_deadline', 'invalid_deadline', 'invalid_max'])
def test_missing_or_invalid_configuration_cannot_open_score_work(database, fault):
    with database[0].connect(**database[1]) as conn:
        cfg = score_seed(conn)
        if fault == 'missing_deadline':
            cfg['schedule'].pop('final_scoring_close')
        elif fault == 'invalid_deadline':
            cfg['schedule']['final_scoring_close'] = 'invalid'
        else:
            cfg['max_attempts_per_assignment'] = 'invalid'
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb', (json.dumps(cfg),))
        conn.commit()
        apply(conn)
        if fault == 'missing_deadline':
            assert rpc(conn)['status'] == 'no_pending'
        else:
            with pytest.raises(database[0].Error):
                rpc(conn)
            conn.rollback()


def test_concurrent_claims_keep_distinct_leases_and_replay_identity(database):
    with database[0].connect(**database[1]) as conn:
        score_seed(conn, positions=4)
        apply(conn)
    barrier = threading.Barrier(2)
    def one(index):
        with database[0].connect(**database[1]) as conn:
            barrier.wait(timeout=10)
            result = rpc(conn, suffix='bc'[index], runner=hotkey('claim509-'+str(index)))
            assert rpc(conn, suffix='bc'[index], runner=hotkey('claim509-'+str(index))) == result
            return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(one, (0, 1)))
    assert all(x['status'] == 'leased' for x in claims)
    assert len({x['run_id'] for x in claims}) == 2


def test_active_company_cache_reservation_remains_a_claim_fence(database):
    with database[0].connect(**database[1]) as conn:
        score_seed(conn, positions=2)
        apply(conn)
        cache_key = 'sha256:'+'d'*64
        other = hotkey('claim509-cache-owner')
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s ORDER BY icp_position', (seed.ROUND,))
            candidate, owner = [row[0] for row in cur.fetchall()]
            cur.execute('UPDATE public.lab_arena_runs SET company_judgment_refs=%s::jsonb WHERE run_id=%s',
                        (json.dumps([{'cache_key': cache_key}]), candidate))
            cur.execute("UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=%s,lease_generation=1,lease_expires_at=clock_timestamp()+interval '1 hour' WHERE run_id=%s", (other, owner))
            cur.execute("INSERT INTO public.lab_arena_company_judgment_reservations(cache_key,authority_slot,company_input_hash,run_id,lease_generation,runner_hotkey,lease_expires_at) VALUES(%s,0,%s,%s,1,%s,clock_timestamp()+interval '1 hour')", (cache_key, 'sha256:'+'e'*64, owner, other))
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_company_judgment_reservations r')
            before = cur.fetchall()
        conn.commit()
        assert rpc(conn)['status'] == 'no_pending'
        with conn.cursor() as cur:
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_company_judgment_reservations r')
            assert cur.fetchall() == before


def test_large_frozen_round_still_skips_expensive_helpers(database):
    with database[0].connect(**database[1]) as conn:
        cfg = score_seed(conn, positions=3)
        # Synthetic local-only padding approximates the observed 629KB frozen
        # composite. It contains no production inputs or provider data.
        cfg['claim509_fixture_padding'] = ''.join(hashlib.sha256(str(n).encode()).hexdigest()
                                                 for n in range(9820))
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb', (json.dumps(cfg),))
        conn.commit()
        apply(conn)
        helper(conn, CREDIT, 'raise')
        helper(conn, CONTINUATION, 'raise')
        assert rpc(conn)['status'] == 'leased'
