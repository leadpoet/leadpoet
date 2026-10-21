"""Only live local heads need global checks; cancellation authority is unchanged."""
from pathlib import Path
import hashlib
import re
import pytest
from tests.lab_arena import judge_final_deadline530_postgres_test as prior

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/533-lab-arena-cancel-latest-call-heads.sql'
INDEX_SQL = ROOT / 'scripts/532-lab-arena-round-call-head-index.sql'
SIG = 'public.lab_arena_cancel_round(text,text)'
POST = 'c984c3ef7cb6f8ab8970b152e44336bc7eb3135a9e31ebda8f00c331c7aa1700'


def body():
    return SQL.read_text().split('DO $round_call_heads$')[1].split('$round_call_heads$;')[0]


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    pg, dsn = next(gen)
    conn = pg.connect(**dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(INDEX_SQL.read_text())
            cur.execute(SQL.read_text()); cur.execute(SQL.read_text())
        yield pg, dsn
    finally:
        conn.close()
        gen.close()


@pytest.mark.parametrize('active', ['leased', 'submitted', 'reservation', 'dispatch'])
def test_actual_cancel_still_drains_live_work(database, active):
    prior.test_atomic_deadline_cancel_does_not_terminate_active_work(database, active)


@pytest.mark.parametrize('kind', ['settlement', 'uncertain', 'recovery', 'refusal'])
def test_actual_closed_and_unknown_cost_heads_still_allow_safe_failure(database, kind):
    prior.test_closed_or_unknown_heads_do_not_block_safe_failure_or_change_costs(database, kind)


def test_actual_completion_and_uncommitted_call_serialization(database):
    prior.test_completion_holding_round_share_finishes_before_safe_cancel(database)
    prior.test_uncommitted_open_call_under_round_share_is_not_missed(database)


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('drift', ['source', 'owner', 'acl', 'secdef', 'volatility', 'path', 'index_missing', 'index_wrong', 'index_invalid', 'index_not_ready', 'index_not_live'])
def test_security_source_index_drift_refused_before_replay(database, installed, drift):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIG,))
        post = cur.fetchone()[0]
        assert hashlib.sha256(post.encode()).hexdigest() == POST
        old = re.search(r'\$old\$(.*?)\$old\$', SQL.read_text(), re.S)[1]
        new = re.search(r'\$new\$(.*?)\$new\$', SQL.read_text(), re.S)[1]
        if not installed:
            cur.execute(post.replace(new, old))
        cur.execute('SAVEPOINT hostile')
        if drift == 'source':
            cur.execute((post if installed else post.replace(new, old)).replace('BEGIN\n', 'BEGIN\n -- unexpected\n', 1))
        elif drift in ('index_invalid', 'index_not_ready', 'index_not_live'):
            field = {'index_invalid':'indisvalid', 'index_not_ready':'indisready', 'index_not_live':'indislive'}[drift]
            cur.execute('UPDATE pg_index SET '+field+'=false WHERE indexrelid=\'public.lab_arena_ledger_round_call_head_idx\'::regclass')
        elif drift.startswith('index_'):
            cur.execute('DROP INDEX public.lab_arena_ledger_round_call_head_idx')
            if drift == 'index_wrong':
                cur.execute('CREATE INDEX lab_arena_ledger_round_call_head_idx ON public.lab_arena_ledger(round_id)WHERE call_identity IS NOT NULL AND entry_kind=\'reservation\'')
        else:
            cur.execute({'owner': 'ALTER FUNCTION '+SIG+' OWNER TO postgres',
                'acl': 'GRANT EXECUTE ON FUNCTION '+SIG+' TO anon',
                'secdef': 'ALTER FUNCTION '+SIG+' SECURITY INVOKER',
                'volatility': 'ALTER FUNCTION '+SIG+' STABLE',
                'path': 'ALTER FUNCTION '+SIG+' SET search_path=public,pg_catalog'}[drift])
        with pytest.raises(pg.Error):
            cur.execute('DO $round_call_heads$'+body()+'$round_call_heads$;')
        cur.execute('ROLLBACK TO SAVEPOINT hostile')
        cur.execute(post)


def test_obsolete_index_cleanup_keeps_replacement_and_function(database):
    pg, dsn = database
    conn = pg.connect(**dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute((ROOT / 'scripts/531-lab-arena-round-open-call-index.sql').read_text())
            cur.execute((ROOT / 'scripts/534-lab-arena-remove-open-call-index.sql').read_text())
            cur.execute("SELECT to_regclass('public.lab_arena_ledger_round_open_call_idx'),indisvalid,indisready,indislive FROM pg_index WHERE indexrelid='public.lab_arena_ledger_round_call_head_idx'::regclass")
            assert cur.fetchone() == (None, True, True, True)
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIG,))
            assert hashlib.sha256(cur.fetchone()[0].encode()).hexdigest() == POST
            with pytest.raises(pg.Error):
                cur.execute((ROOT / 'scripts/534-lab-arena-remove-open-call-index.sql').read_text())
    finally:
        conn.close()
