"""Indexed interrupted-response proof; disposable PostgreSQL, no provider I/O."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from threading import Event
import time
import re
from pathlib import Path

import pytest
from tests.lab_arena import interrupted_judge_response689_postgres_test as previous

SQL = Path(__file__).parents[2] / 'scripts/690-lab-arena-retained-proof-call-lookup.sql'
SIG = 'public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger)'
PRE = 'b971cc230ccc60162c21325040e291663feceb6ef5a18ac19f8cbac974dbabc0'
POST = '4d359779d99e492d325f8d2e55cd71c00e9156dee6cfa566ea1db34431e4b57f'
old, base, finite = previous.old, previous.base, previous.finite
database = previous.database
prepared = previous.prepared
prepared_company = previous.prepared_company
case = previous.case
base_extended = previous.base_extended
extended = previous.extended
provider_case = previous.provider_case
plain_finite = previous.plain_finite
before = previous.before
post = previous.post
installed = previous.installed


@pytest.fixture
def current(installed):
    p = installed
    previous.budget.upgrade(p)
    previous.previous.upgrade(p)
    previous.upgrade(p)
    return p


def upgrade(p):
    with p.c.cursor() as q:
        q.execute(SQL.read_text())
    p.c.commit()


def helper(q):
    return old.scalar(q, 'SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure', (SIG,))


def test_function_only_exact_patch_is_idempotent_and_preserves_every_proof_check(current):
    p = current
    with p.c.cursor() as q:
        source = helper(q)
        metadata = old.scalar(q, "SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (SIG,))
    p.c.commit()
    assert hashlib.sha256(source.encode()).hexdigest() == PRE
    rows = base.snapshot(p)
    upgrade(p)
    upgrade(p)
    with p.c.cursor() as q:
        changed = helper(q)
        assert old.scalar(q, "SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (SIG,)) == metadata
    p.c.commit()
    assert changed == source.replace("   WHERE r.entry_kind='reservation' AND d.entry_kind='dispatch'\n", "   WHERE r.entry_kind='reservation' AND d.entry_kind='dispatch'\n    AND r.call_identity=p_entry.call_identity\n", 1)
    assert hashlib.sha256(changed.encode()).hexdigest() == POST
    assert base.snapshot(p) == rows


@pytest.mark.parametrize('error', ['ReadTimeout', 'ReadError', 'RemoteProtocolError'])
def test_indexed_proof_keeps_native_dry_run_grant_checkpoints_payer_and_ledger(current, error):
    p = current
    previous.budget.diagnostic(p)
    entry = previous.transport(p, changes={'transport_error_class': error})
    assert previous.proved(p, entry) is True
    rows = base.snapshot(p)
    upgrade(p)
    assert previous.proved(p, entry) is True
    preview = finite.grant(p, dry=True)
    assert base.snapshot(p) == rows
    result = finite.grant(p)
    assert result['child_identity'] == preview['child_identity']
    assert result['checkpoints'] == p.checkpoints
    assert result['signal_checkpoints'] == p.signal_checkpoints
    after = base.snapshot(p)
    for table in set(rows) - {'lab_arena_runs', 'lab_arena_scoring_supersessions'}:
        assert after[table] == rows[table]
    child = result['child_identity']['run_id']
    assert old.funding(p, child)['funding_source'] == ('host' if p.organizer else 'miner_key')


@pytest.mark.parametrize('field,value', [
    ('run_id', 'other'), ('round_id', 'other'), ('submission_id', 'other'),
    ('miner_hotkey', 'other'), ('stage', 1), ('call_identity', 'sha256:' + 'e' * 64),
    ('provider', 'deepline'), ('operation_id', 'exa.search'),
    ('funding_source', None),
])
def test_indexed_lookup_preserves_all_identity_and_payer_guards(current, field, value):
    p = current
    entry = previous.transport(p, mismatch={field: value})
    assert previous.proved(p, entry) is False
    upgrade(p)
    assert previous.proved(p, entry) is False


@pytest.mark.parametrize('fault', ['request', 'order', 'no_reservation', 'no_dispatch'])
def test_indexed_lookup_preserves_request_and_order_guards(current, fault):
    p = current
    kwargs = {'omit': [fault[3:]]} if fault.startswith('no_') else {}
    entry = previous.transport(p, **kwargs)
    if fault in ('request', 'order'):
        with p.c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            if fault == 'request':
                q.execute("UPDATE lab_arena_ledger SET entry_doc='{}'::jsonb WHERE entry_kind='reservation' AND call_identity=(SELECT call_identity FROM lab_arena_ledger WHERE entry_id=%s)", (entry,))
            else:
                q.execute("UPDATE lab_arena_ledger SET entry_id=%s+1 WHERE entry_kind='reservation' AND call_identity=(SELECT call_identity FROM lab_arena_ledger WHERE entry_id=%s)", (entry, entry))
            q.execute('SET LOCAL session_replication_role=origin')
        p.c.commit()
    assert previous.proved(p, entry) is False
    upgrade(p)
    assert previous.proved(p, entry) is False


def test_indexed_proof_keeps_one_replacement_and_no_third_exception(current):
    p = current
    previous.make_failed_budget_child(p)
    upgrade(p)
    preview = finite.grant(p, request_id='replacement', approval='approval:replacement', dry=True)
    child = finite.grant(p, request_id='replacement', approval='approval:replacement')['child_identity']['run_id']
    assert child == preview['child_identity']['run_id']
    previous.prior.claim_child(p, child)
    previous.prior.complete_failed(p, child)
    previous.prior.saved_parent(p, child)
    previous.budget.diagnostic(p)
    previous.transport(p)
    rows = base.snapshot(p)
    with pytest.raises(p.pg.Error):
        finite.grant(p, request_id='no-third', approval='approval:no-third')
    p.c.rollback()
    assert base.snapshot(p) == rows


def nodes(plan):
    yield plan
    for child in plan.get('Plans', []):
        yield from nodes(child)


def explain(q, source, entry):
    # Extract the actual helper's EXISTS predicate. A generic plan matches
    # PL/pgSQL's cached composite fields, without constant-folding the proof.
    query = re.search(r'AND EXISTS\((SELECT 1 FROM public.lab_arena_ledger r\n.*?p_entry.funding_source\)\n.*?r.funding_source\))\)\) IS TRUE', source, re.S).group(1)
    fields = ['entry_id', 'run_id', 'round_id', 'submission_id', 'miner_hotkey', 'stage',
              'call_identity', 'provider', 'operation_id', 'funding_source']
    for i, field in enumerate(fields, 1):
        query = query.replace('p_entry.' + field, '$' + str(i))
    row = old.scalar(q, 'SELECT to_jsonb(l) FROM lab_arena_ledger l WHERE entry_id=%s', (entry,))
    q.execute('DEALLOCATE ALL')
    q.execute('PREPARE retained_lookup(BIGINT,TEXT,TEXT,TEXT,TEXT,SMALLINT,TEXT,TEXT,TEXT,TEXT) AS SELECT EXISTS(' + query + ')')
    q.execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) EXECUTE retained_lookup(' + ','.join(['%s'] * len(fields)) + ')', [row[field] for field in fields])
    return q.fetchone()[0][0]


def test_realistic_ledger_generic_plan_reads_only_the_bound_call_after_fix(current, record_property):
    p = current
    if p.organizer:
        pytest.skip('Planner regression is payer-independent; run once for miner payer.')
    with p.c.cursor() as q:
        # Native schema and indexes, 40,000 unrelated calls / 80,000 rows.
        # Fill before the target call so pre-fix scanning cannot stop early.
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("""INSERT INTO lab_arena_ledger(entry_kind,miner_hotkey,round_id,
          submission_id,run_id,stage,call_identity,provider,operation_id,
          funding_source,amount_microusd,entry_doc)
          SELECT kind,'unrelated-miner','unrelated-round','unrelated-submission',
           'unrelated-run-'||n,2,'sha256:'||lpad(to_hex(n),64,'0'),'openrouter',
           'openrouter.chat','miner_key',0,jsonb_build_object('request_hash','sha256:'||repeat('4',64))
          FROM generate_series(1,40000) n CROSS JOIN (VALUES('reservation'),('dispatch')) k(kind)""")
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    entry = previous.transport(p)
    with p.c.cursor() as q:
        q.execute('ANALYZE lab_arena_ledger')
        q.execute("SET LOCAL plan_cache_mode=force_generic_plan")
        q.execute('SET LOCAL max_parallel_workers_per_gather=0')
        q.execute('SET LOCAL jit=off')
        original = helper(q)
        slow = explain(q, original, entry)
        assert any(n.get('Rows Removed by Filter', 0) >= 40000 for n in nodes(slow['Plan']))
        assert not any('Index Cond' in n and '$7' in n['Index Cond'] for n in nodes(slow['Plan']))
    p.c.commit()
    upgrade(p)
    with p.c.cursor() as q:
        q.execute('SET LOCAL plan_cache_mode=force_generic_plan')
        q.execute('SET LOCAL max_parallel_workers_per_gather=0')
        q.execute('SET LOCAL jit=off')
        fast = explain(q, helper(q), entry)
        indexes = {n.get('Index Name') for n in nodes(fast['Plan']) if '$7' in n.get('Index Cond', '')}
        assert {'lab_arena_ledger_reservation_uq', 'lab_arena_ledger_dispatch_uq'} <= indexes
        assert fast['Plan'].get('Shared Hit Blocks', 0) < 100
        assert slow['Plan'].get('Shared Hit Blocks', 0) > 1000
    p.c.commit()
    assert previous.proved(p, entry) is True

    record_property('before_execution_ms', slow['Execution Time'])
    record_property('after_execution_ms', fast['Execution Time'])
    record_property('before_shared_hit_blocks', slow['Plan'].get('Shared Hit Blocks', 0))
    record_property('after_shared_hit_blocks', fast['Plan'].get('Shared Hit Blocks', 0))

    # Exercise a native grant and lease, then a native reservation on that
    # separate run while the proof executes under the grant's canonical lock.
    # The bulk ledger is retained. No upstream request is dispatched.
    previous.budget.diagnostic(p)
    child = finite.grant(p)['child_identity']['run_id']
    lease = previous.prior.claim_child(p, child)
    request = 'sha256:' + 'a' * 64
    identity = old.contracts.provider_call_identity(
        assignment_id=lease['assignment_id'], attempt=lease['attempt'],
        icp_position=lease['icp_position'], action_sequence=0,
        operation_id='openrouter.chat', request_hash=request)
    funding = old.funding(p, child)['funding_source']
    proof_locked = Event()

    def locked_proof():
        with p.pg.connect(**p.dsn) as connection, connection.cursor() as q:
            q.execute("SET LOCAL statement_timeout='2s'")
            q.execute("SET LOCAL plan_cache_mode=force_generic_plan")
            q.execute("SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0))")
            proof_locked.set()
            # Repeated work increases overlap without an artificial sleep or
            # changing the existing recovery/provider synchronization rules.
            q.execute('SELECT bool_and(public.lab_arena__retained_provider_transient_v1(l)) FROM lab_arena_ledger l CROSS JOIN generate_series(1,1000) n WHERE l.entry_id=%s', (entry,))
            assert q.fetchone()[0] is True

    with ThreadPoolExecutor(max_workers=1) as executor:
        proof = executor.submit(locked_proof)
        assert proof_locked.wait(2)
        started = time.monotonic()
        with p.pg.connect(**p.dsn) as connection, connection.cursor() as q:
            q.execute("SET LOCAL statement_timeout='2s'")
            q.execute('SET LOCAL ROLE lab_arena_service')
            doc = dict(request_hash=request, base_call_identity=identity,
                       provider_attempt=1, action_sequence=0, max_output_tokens=8)
            reserved = old.scalar(q, 'SELECT lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)',
                                  (child, old.TOKEN, identity, 'openrouter.chat', 'openrouter',
                                   funding, 500, json.dumps(doc), 1500))
            assert reserved['status'] == 'reserved'
        proof.result(timeout=3)
        elapsed = time.monotonic() - started
        assert elapsed < 3
        record_property('concurrent_native_reservation_ms', elapsed * 1000)
