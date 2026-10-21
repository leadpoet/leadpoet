"""Round-local cancellation heads stay exact and avoid a global ledger scan."""
from pathlib import Path
import json
import pytest
from tests.lab_arena.lab_arena_pg_harness import _local_database

SQL = Path(__file__).parents[2] / 'scripts/532-lab-arena-round-call-head-index.sql'
INDEX = 'lab_arena_ledger_round_call_head_idx'
OLD_QUERY = """SELECT EXISTS(SELECT 1 FROM public.lab_arena_ledger opened
 WHERE opened.round_id=%s AND opened.call_identity IS NOT NULL
 AND opened.entry_kind IN ('reservation','dispatch')
 AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger later
 WHERE later.call_identity=opened.call_identity AND later.entry_id>opened.entry_id))"""
QUERY = """WITH round_rows AS MATERIALIZED (
 SELECT call_identity,entry_id,entry_kind FROM public.lab_arena_ledger
 WHERE round_id=%s AND call_identity IS NOT NULL)
 SELECT EXISTS(SELECT 1 FROM (
 SELECT DISTINCT ON(call_identity)call_identity,entry_id,entry_kind FROM round_rows
 ORDER BY call_identity,entry_id DESC)opened
 WHERE opened.entry_kind IN('reservation','dispatch')AND NOT EXISTS(
 SELECT 1 FROM public.lab_arena_ledger later
 WHERE later.call_identity=opened.call_identity AND later.entry_id>opened.entry_id))"""


def nodes(plan):
    yield plan
    for child in plan.get('Plans', []):
        yield from nodes(child)


@pytest.fixture(scope='module')
def database():
    gen = _local_database([])
    pg, dsn = next(gen)
    try:
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            # Deliberately include orphan and cross-round heads: no run join or
            # later.round_id filter may silently narrow the cancellation guard.
            cur.execute('CREATE UNLOGGED TABLE public.lab_arena_ledger(entry_id bigint,round_id text,run_id text,entry_kind text,call_identity text,pad text)')
            cur.execute('CREATE INDEX lab_arena_ledger_call_idx ON public.lab_arena_ledger(call_identity,entry_id DESC)WHERE call_identity IS NOT NULL')
        yield pg, dsn
    finally:
        gen.close()


def insert(cur, entry, round_id, kind, cid, run=None):
    cur.execute('INSERT INTO public.lab_arena_ledger VALUES(%s,%s,%s,%s,%s,NULL)', (entry, round_id, run, kind, cid))


def test_global_head_semantics_and_standalone_creation(database):
    pg, dsn = database
    cases = {'open_reservation': True, 'open_dispatch': True, 'settled': False,
             'uncertain': False, 'recovery': False, 'refusal': False,
             'null_identity': False, 'other_round': False,
             'cross_round_later': False, 'orphan_open': True}
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        for n, case in enumerate(cases):
            cid = 'call-' + case
            insert(cur, n * 3, case, 'reservation', None if case == 'null_identity' else cid)
            if case in ('settled', 'uncertain', 'recovery', 'refusal'):
                insert(cur, n * 3 + 1, case, 'settlement' if case == 'settled' else case, cid)
            elif case == 'open_dispatch':
                insert(cur, n * 3 + 1, case, 'dispatch', cid)
            elif case == 'cross_round_later':
                insert(cur, n * 3 + 1, 'another-round', 'settlement', cid)
            elif case == 'other_round':
                cur.execute('DELETE FROM public.lab_arena_ledger WHERE round_id=%s', (case,))
                insert(cur, n * 3, 'another-round', 'reservation', cid)
        for offset, (case, sequence, expected) in enumerate([
            ('terminal_then_open', [('local','settlement'),('local','reservation')], True),
            ('open_terminal_open', [('local','reservation'),('local','refusal'),('local','dispatch')], True),
            ('local_closed_foreign_open', [('local','reservation'),('local','settlement'),('foreign','dispatch')], False),
            ('foreign_terminal_local_open', [('local','reservation'),('foreign','uncertain'),('local','dispatch')], True),
        ]):
            cases[case] = expected
            for step, (where, kind) in enumerate(sequence):
                insert(cur, 1000+offset*10+step, case if where == 'local' else 'foreign', kind, 'call-'+case, 'mismatched-run')
        cur.execute('SELECT jsonb_agg(to_jsonb(l)ORDER BY entry_id)FROM public.lab_arena_ledger l')
        before = cur.fetchone()[0]
        for rid, expected in cases.items():
            cur.execute(OLD_QUERY, (rid,)); assert cur.fetchone()[0] is expected
            cur.execute(QUERY, (rid,)); assert cur.fetchone()[0] is expected
    conn = pg.connect(**dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(SQL.read_text())
            for rid, expected in cases.items():
                cur.execute(QUERY, (rid,)); assert cur.fetchone()[0] is expected
            cur.execute('SELECT jsonb_agg(to_jsonb(l)ORDER BY entry_id)FROM public.lab_arena_ledger l')
            assert cur.fetchone()[0] == before
            cur.execute('SELECT indisvalid,indisready,pg_get_indexdef(indexrelid)FROM pg_index WHERE indexrelid=%s::regclass', (INDEX,))
            valid, ready, definition = cur.fetchone()
            assert valid and ready and 'btree (round_id)' in definition
            assert 'call_identity IS NOT NULL' in definition
            assert 'entry_kind' not in definition
            # No IF NOT EXISTS may silently accept a pre-existing invalid or
            # wrong-shaped index. The operator verifies the exact catalog.
            with pytest.raises(pg.Error):
                cur.execute(SQL.read_text())
    finally:
        conn.close()


def test_production_count_settled_queue_uses_round_index(database):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('TRUNCATE public.lab_arena_ledger')
        cur.execute('DROP INDEX IF EXISTS ' + INDEX)
        cur.execute("INSERT INTO public.lab_arena_ledger SELECT i,CASE WHEN i<=123000 THEN 'target' ELSE 'other-'||(i%1000)::text END,'run-'||(((i-1)/3)%700)::text,(ARRAY['reservation','dispatch','settlement'])[((i-1)%3)+1],'sha256:'||md5(((i-1)/3)::text)||md5(((i-1)/3)::text),repeat(md5(i::text),8)FROM generate_series(1,5400000)i")
        cur.execute('ANALYZE public.lab_arena_ledger')
        cur.execute('EXPLAIN(ANALYZE,BUFFERS,TIMING OFF,FORMAT JSON)' + OLD_QUERY, ('target',))
        old = cur.fetchone()[0][0]
        assert any(n['Node Type'] == 'Seq Scan' and n.get('Alias') == 'opened' for n in nodes(old['Plan']))
    conn = pg.connect(**dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(SQL.read_text())
            cur.execute('ANALYZE public.lab_arena_ledger')
            cur.execute('EXPLAIN(ANALYZE,BUFFERS,TIMING OFF,FORMAT JSON)' + QUERY, ('target',))
            new = cur.fetchone()[0][0]
            outer = next(n for n in nodes(new['Plan']) if n.get('Index Name') == INDEX)
            assert outer.get('Index Name') == INDEX
            assert outer['Actual Rows'] == 123000
            later = next(n for n in nodes(new['Plan']) if n.get('Alias') == 'later')
            assert later['Actual Loops'] == 0
            assert outer['Shared Hit Blocks'] + outer['Shared Read Blocks'] < 15000
            # Cold index builds and visibility/cache state can reverse a
            # single wall-time sample. Assert the bounded access path and
            # absolute local budget; production timing remains a release gate.
            assert new['Execution Time'] < 8000
            cur.execute('EXPLAIN(ANALYZE,BUFFERS,TIMING OFF,FORMAT JSON)' + QUERY, ('target',))
            warm = cur.fetchone()[0][0]
            print(json.dumps({'rows': 5400000, 'round_candidates': 82000, 'round_all_kind_rows': 123000,
                'old_ms': old['Execution Time'], 'new_cold_ms': new['Execution Time'],
                'new_warm_ms': warm['Execution Time'],
                'outer_buffers': outer['Shared Hit Blocks'] + outer['Shared Read Blocks'],
                'limitation': 'local unlogged fixture is not production heap, visibility or cache state'}))
            cur.execute(QUERY, ('target',)); assert cur.fetchone()[0] is False
            # A single latest open head remains visible, including at the end
            # of the target's physical range. No settlement or fee is edited.
            cur.execute("INSERT INTO public.lab_arena_ledger VALUES(6000001,'target',NULL,'reservation','orphan-last',NULL)")
            cur.execute(QUERY, ('target',)); assert cur.fetchone()[0] is True
    finally:
        conn.close()
