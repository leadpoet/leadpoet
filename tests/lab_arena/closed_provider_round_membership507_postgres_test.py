"""Generic closed-cost plans scan uncertainty history once, with identical bills."""
from pathlib import Path

from lab_arena import broker as br, contracts
from lab_arena.store import hash_lease_token
from functools import partial
import base64
import hashlib
import json
import re

import pytest

from tests import postgres_migration_harness as pg_harness
from tests.lab_arena import lab_arena_pg_harness as arena_pg
from tests.lab_arena import closed_provider_candidate_query_postgres_test as prior
from tests.lab_arena import closed_openrouter_billing_postgres_test as router
from tests.lab_arena.flat_per_icp_cost500_postgres_test import (
    catalog, rows,
)

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / 'scripts/507-lab-arena-closed-provider-round-membership.sql'
SIGNATURE = 'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)'
PREIMAGE = 'a65adc61115dbb1fa7258bffcadc96218cf889534f3de189b82134e2c5effaf0'
POSTGRES_IMAGE = 'postgres:17.9@sha256:2a0d0fe14825b0939f78a8cad5cd4e6aa68bf94d0e5dd96e24b6d23af4315545'


@pytest.fixture(scope='module')
def database():
    # Keep historical lifecycle fixtures; upgrade exactly the closed billing
    # seam. Full500 also changes claim authority, which these fixtures predate.
    from tests.lab_arena.lab_arena_pg_harness import (
        CURRENT_PROVIDER_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
    )
    gen = database_with_lab_arena_migration(CURRENT_PROVIDER_SERVICE_MIGRATIONS + (
        '435-lab-arena-deepline-candidate-query.sql',
        '436-lab-arena-closed-provider-candidate-query.sql',
        '438-lab-arena-accepted-host-score-billing.sql',
        '439-lab-arena-closed-miner-success-score-billing.sql',
        '261-lab-arena-settlement-failure-deepline-cost-reconciliation.sql',
        '453-lab-arena-deepline-adaptation-cost-reconciliation.sql',
        '456-lab-arena-score-deepline-cost-priority.sql',
        '457-lab-arena-closed-adaptation-score-billing.sql',
        '462-lab-arena-deepline-native-settlement-binding.sql',
    ))
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            # The three canonical source substitutions from500, restricted
            # to this read-only seam. Exact deployed hashes below bind them.
            names = prior.NAMES + ['lab_arena__closed_score_dynamic_uncertainty_v1',
                'lab_arena__closed_host_score_success_uncertainty_v1',
                'lab_arena__closed_miner_score_success_uncertainty_v1',
                'lab_arena__closed_score_adaptation_uncertainty_v1']
            helper_hashes = {
                'lab_arena__closed_host_score_success_uncertainty_v1':'e347886710c2461366607941044af1123d092622cfb65ecae1c5b70a1a22f914',
                'lab_arena__closed_miner_score_success_uncertainty_v1':'67fcd9dc83d29491aa4608e069b6680ac652e85b93b1feac64cc01a72b1f93f8',
                'lab_arena__closed_score_adaptation_uncertainty_v1':'0b876ab01b0c9e88ca9dc4a1cfcd3eb9e8cc13dc453a2bd418593e5dd3a96f82',
                'lab_arena__closed_score_dynamic_uncertainty_v1':'d5885be424024b95f7aac98b175872857baa6b0856e407ee9108b2382581ff0d',
            }
            for name in names:
                suffix = prior.SUFFIX if name in prior.NAMES else '(bigint)'
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', ('public.'+name+suffix,))
                definition = cur.fetchone()[0]
                definition = re.sub(r"((?:[A-Za-z_][A-Za-z_0-9]*\.)?[A-Za-z_][A-Za-z_0-9]*\s*->>\s*'sourcing_cost_eligibility_policy')\s+IS DISTINCT FROM 'successful_calls_per_icp_v1'",
                    r"COALESCE(\1, '') NOT IN ('successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')", definition)
                definition = re.sub(r"=\s*'successful_calls_per_icp_v1'",
                    "IN ('successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')", definition)
                definition = re.sub(r"'successful_calls_v1',\s*'successful_calls_per_icp_v1'",
                    "'successful_calls_v1', 'successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1'", definition)
                cur.execute(definition)
                if name in helper_hashes:
                    assert hashlib.sha256(definition.encode()).hexdigest()==helper_hashes[name]
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',
                ('public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)',))
            assert hashlib.sha256(cur.fetchone()[0].encode()).hexdigest() == '8a973077fb121ef9f01866b301aba0ec5dd05da8143f87644ad8e28afa530f49'
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURE,))
            assert hashlib.sha256(cur.fetchone()[0].encode()).hexdigest() == '58f74de8066a3c9b0300a66c32820f92a098d584a964574d886c830dbc7f2547'
            # The follow-on must reject500, then apply only after exact506.
            cur.execute('SAVEPOINT before506')
            with pytest.raises(db[0].Error,match='exact preimage differs'):
                cur.execute(MIGRATION.read_text())
            cur.execute('ROLLBACK TO SAVEPOINT before506')
            cur.execute((ROOT/'scripts/506-lab-arena-closed-reconciliation-eligible-run-scope.sql').read_text())
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURE,))
            assert hashlib.sha256(cur.fetchone()[0].encode()).hexdigest() == PREIMAGE
            for name in prior.NAMES:
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', ('public.'+name+prior.SUFFIX,))
                definition = cur.fetchone()[0].replace('FUNCTION public.'+name+'(',
                    'FUNCTION public.test_before_'+name+'(', 1)
                if name == prior.NAMES[1]:
                    definition = definition.replace('public.'+prior.NAMES[0]+'(',
                        'public.test_before_'+prior.NAMES[0]+'(')
                cur.execute(definition)
                cur.execute('ALTER FUNCTION public.test_before_'+name+prior.SUFFIX+' OWNER TO lab_arena_owner')
            before, before_rows = catalog(cur), rows(cur)
            before_relations = prior._catalog(cur)[1]
            cur.execute(MIGRATION.read_text())
            after = catalog(cur)
            assert rows(cur) == before_rows
            assert prior._catalog(cur)[1] == before_relations
            changed = 0
            for old, new in zip(before, after):
                assert old[0] == new[0] and old[2:] == new[2:]
                if old[1] != new[1]:
                    changed += 1
                    assert old[0] == SIGNATURE.removeprefix('public.')
            assert changed == 1 and len(before) == len(after)
            cur.execute(MIGRATION.read_text())
            assert catalog(cur) == after and rows(cur) == before_rows
        yield db
    finally:
        gen.close()


def test_scope_policy_cause_and_proof_controls(database):
    prior.test_miner_policies_causes_cancelled_liabilities_and_scope(database)


def test_cursor_latest_head_and_settlement_controls(database):
    prior.test_cursor_wrap_latest_head_and_settlement(database)


def test_host_success_bindings(database, tmp_path, monkeypatch):
    monkeypatch.setattr(prior.br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    h, _, connect, _, _, candidate = prior.host.closed_call(database, tmp_path, '23')
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            for funding in ('host', 'miner_key'):
                for status, cause in (('failed','judge_error'), ('accepted','accepted')):
                    cur.execute('SAVEPOINT valid_receipt')
                    cur.execute('UPDATE lab_arena_ledger SET funding_source=%s WHERE call_identity=%s',
                                (funding,candidate['call_identity']))
                    cur.execute('UPDATE lab_arena_runs SET status=%s,terminal_cause=%s WHERE run_id=%s',
                                (status,cause,candidate['run_id']))
                    assert all(v['status']=='ok' for v in prior._equivalent(cur,h.round_id))
                    cur.execute('ROLLBACK TO SAVEPOINT valid_receipt')
            for path,value in [('call,call_succeeded','null'),('call,call_succeeded','"bogus"'),
                    ('call,provider_status','402'),('call,provider_status','"200"'),
                    ('call,reason','"transport_failure"'),('call,deepline_job_id','null'),
                    ('call,deepline_job_id','123'),('call,deepline_job_id','"bad/id"'),
                    ('call,credential_fingerprint','"sha256:'+'b'*64+'"')]:
                cur.execute('SAVEPOINT invalid_receipt')
                cur.execute('UPDATE lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,%s::text[],%s::jsonb) '
                            'WHERE entry_id=%s',('{'+path+'}',value,candidate['uncertain_entry_id']))
                prior._equivalent(cur,h.round_id)
                cur.execute('SELECT public.lab_arena__closed_host_score_success_uncertainty_v1(%s)',
                            (candidate['uncertain_entry_id'],))
                assert cur.fetchone()==(False,)
                cur.execute('ROLLBACK TO SAVEPOINT invalid_receipt')
    finally:
        h.service.store.close()


@pytest.mark.parametrize('status', ['accepted', 'failed'])
@pytest.mark.parametrize('closed', ['published', 'cancelled'])
def test_openrouter_independent(database, status, closed):
    prior.test_openrouter_lane_remains_independent(database, status, closed)


def test_mixed_provider_global_cursor(database):
    prior.test_mixed_provider_cursor_keeps_one_global_order(database)


@pytest.mark.parametrize('policy', ['successful_calls_v1', 'successful_calls_per_icp_v1',
                                   'successful_calls_flat_per_icp_v1', 'legacy', None])
def test_flat_policy_and_null_scope_equivalence(database, policy):
    store, rid, candidate = prior._closed_call(database, '507'+hashlib.sha256(str(policy).encode()).hexdigest()[:6])
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc || "
                        "jsonb_build_object('sourcing_cost_eligibility_policy',%s::text) WHERE round_id=%s",
                        (policy, rid))
            for scope in ('', rid, None, 'missing'):
                prior._equivalent(cur, scope, candidate['uncertain_entry_id'])
            for kwargs in ({'mode':None}, {'network':None}, {'netuid':None}):
                assert prior._equivalent(cur, rid, **kwargs) == [{'status':'none'}]*2
    finally:
        store.close()


def test_active_rounds_malformed_metadata_and_failed_unknowns(database):
    store,rid,candidate = prior._closed_call(database,'507metadata')
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            for status in ('open','stage2','scored'):
                cur.execute('SAVEPOINT scope')
                cur.execute('UPDATE lab_arena_rounds SET status=%s WHERE round_id=%s',(status,rid))
                assert prior._equivalent(cur,rid)==[{'status':'none'}]*2
                cur.execute('ROLLBACK TO SAVEPOINT scope')
            for field in ('mode','network_name','netuid'):
                for value in (None,{},[],True):
                    cur.execute('SAVEPOINT scope')
                    try:
                        cur.execute('UPDATE lab_arena_rounds SET configuration_doc=jsonb_set('
                            'configuration_doc,%s::text[],%s::jsonb) WHERE round_id=%s',
                            ('{'+field+'}',json.dumps(value),rid))
                    except (database[0].errors.CheckViolation, database[0].errors.InvalidTextRepresentation) as error:
                        # The existing generated chain-pair constraint rejects
                        # impossible stored scopes before any selector runs.
                        if isinstance(error,database[0].errors.CheckViolation):
                            assert error.diag.constraint_name=='lab_arena_round_network_pair'
                        else:
                            assert field=='netuid' and error.pgcode=='22P02'
                    else:
                        assert prior._equivalent(cur,rid)==[{'status':'none'}]*2
                    cur.execute('ROLLBACK TO SAVEPOINT scope')
            for value in (None,[],{'call':None},{'call':[]}):
                cur.execute('SAVEPOINT bad_proof')
                cur.execute('UPDATE lab_arena_ledger SET entry_doc=%s::jsonb WHERE entry_id=%s',
                            (json.dumps(value),candidate['uncertain_entry_id']))
                prior._equivalent(cur,rid)
                cur.execute('ROLLBACK TO SAVEPOINT bad_proof')
    finally:
        store.close()
    from tests.lab_arena.closed_scoring_billing_postgres_test import (
        test_fixed_unknown_is_not_admitted_to_published_mutation_path,
    )
    test_fixed_unknown_is_not_admitted_to_published_mutation_path(database,'successful_calls_per_icp_v1')


def test_closed_bill_full_transition(database):
    router.test_exact_closed_judge_bill_settles_once_without_changing_accepted_state(
        database, 'missing_provider_cost', 'accepted', 'published')
    router.test_failed_lookup_and_rotated_credentials_never_invent_or_release_cost(database)


@pytest.mark.parametrize('mutation', ['source', 'security', 'target'])
def test_migration_guard_rejects_drift(database, mutation):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',
                    ('public.test_before_'+prior.NAMES[0]+prior.SUFFIX,))
        old = cur.fetchone()[0].replace('FUNCTION public.test_before_', 'FUNCTION public.', 1)
        cur.execute(old)
        if mutation == 'source':
            cur.execute(old.replace('candidate_query_436', 'changed_source'))
        elif mutation == 'security':
            cur.execute('ALTER FUNCTION '+SIGNATURE+" SET work_mem='1MB'")
        sql = MIGRATION.read_text()
        if mutation == 'target':
            # Corrupt only the expected postimage, leaving the exact preimage.
            import re
            target = re.search(r"hex'\)='([a-f0-9]{64})'", sql)[1]
            sql = sql.replace(target, '0'*64)
        cur.execute('SAVEPOINT mutation')
        with pytest.raises(database[0].Error, match='Closed provider round-array'):
            cur.execute(sql)
        conn.rollback()


@pytest.fixture
def plan_database():
    # Keep representative cardinalities independent of earlier scope controls.
    # Benchmark the actual deployed PG17 planner, as in the funding-throughput
    # proof. All ordinary historical fixtures keep their existing PG15 runtime.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(arena_pg, '_database_with_migrations',
                      partial(pg_harness._database_with_migrations, image=POSTGRES_IMAGE))
        yield from database.__wrapped__()


def _plan_call_templates(database):
    """Get each ledger kind from the actual broker and native SQL writers."""
    from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import _store, _open_scoring
    from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete
    from tests.lab_arena.test_lab_arena_broker import price_table, DL_KEY, CHAT, KEY

    class Transport:
        def __init__(self, provider, settled):
            self.provider, self.settled = provider, settled

        def send(self, **request):
            if not self.settled:
                if request['method'] == 'POST':
                    raise br.ProviderTransportError('ReadTimeout')
                if '/executions/by-key/' in request['url']:
                    return br.ProviderResponse(404,
                        {'x-deepline-idempotency-supported': 'true'}, b'{}')
                return br.ProviderResponse(200, {}, b'{"recent":{"entries":[]}}')
            assert request['method'] == 'POST'
            if self.provider == 'openrouter':
                doc = {'id': 'gen-template507', 'choices': [],
                       'usage': {'cost': '0.000013', 'prompt_tokens': 20, 'completion_tokens': 50}}
            else:
                doc = {'job_id': request['headers']['x-deepline-request-id'], 'status': 'completed',
                       'billing': {'credits_charged': .02, 'pricing_status': 'final'},
                       'result': {'data': {'rawHtml': '<html>synthetic</html>', 'metadata': {
                           'url': 'https://example.com/', 'sourceURL': 'https://example.com/',
                           'statusCode': 200}}}}
            return br.ProviderResponse(200, {}, json.dumps(doc).encode())

    templates = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
        for provider in ('openrouter', 'deepline'):
            for settled in (False, True):
                store = _store(database)
                try:
                    rid = f'arena-2026-09-14-{provider[:2]}{int(settled)}507'
                    runners, _ = _open_scoring(store, rid, participants=2, runners=2)
                    run, token, _, _ = claim(store, rid, runners[0], parallelism=8, ceiling=8)
                    context = br.RunContext(run_id=run['run_id'], assignment_id=run['assignment_id'],
                        icp_position=run['icp_position'], lease_token_hash=hash_lease_token(token),
                        miner_hotkey=run['miner_hotkey'], submission_id=run['submission_id'],
                        stage=run['stage'], kind='score', attempt=run['attempt'], round_id=rid)
                    broker = br.Broker(store=store,
                        key_for=lambda p: DL_KEY if p == 'deepline' else KEY,
                        credential_for=lambda c, p: DL_KEY if p == 'deepline' else KEY,
                        funding_source_for=lambda _: 'miner_key',
                        provider_funding_source_for=lambda *_: 'miner_key',
                        price_table=price_table(), judge_models=['openai/gpt-4o-mini'],
                        transport=Transport(provider, settled))
                    result = broker.execute(context,
                        operation_id='openrouter.chat' if provider == 'openrouter' else 'scrapingdog.scrape',
                        parameters=CHAT if provider == 'openrouter' else {'url': 'https://example.com/'},
                        action_sequence=0, timeout_ms=5000)
                    assert result.call['outcome'] == ('settled' if settled else 'uncertain')
                    ledger = store.list_ledger(call_identity=result.call['call_identity'])
                    assert [row['entry_kind'] for row in ledger] == [
                        'reservation', 'dispatch', 'settlement' if settled else 'uncertain']
                    # The native completion writer permits a successful score
                    # after a failed provider request; keep this lifecycle proof.
                    completed = complete(store, run['run_id'], hash_lease_token(token),
                        'accepted', output_ref='s3://fixture/score.json')
                    assert completed['status'] == 'accepted'
                    templates.extend((provider, settled, row, result.call) for row in ledger)
                finally:
                    store.close()
    return templates


@pytest.mark.parametrize('work_mem,random_page_cost', [('12MB', '1.1'), ('4MB', '4')],
                         ids=['source-planner', 'target-planner'])
def test_generic_function_plan_reads_spill_once(plan_database, tmp_path, work_mem, random_page_cost):
    """Actual SQL functions plus exposed generic body; no literal-plan proxy."""
    database = plan_database
    templates = _plan_call_templates(database)
    pg, dsn = database
    store, rid, run, cid, _ = router._call(database, '507plan')
    try:
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("SET LOCAL statement_timeout='20s'")
            cur.execute('SET LOCAL work_mem=%s', (work_mem,))
            cur.execute('SET LOCAL jit=off')
            cur.execute('SET LOCAL random_page_cost=%s', (random_page_cost,))
            # Sixty closed rounds mirror the observed production cardinality.
            cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc || "
                        "'{\"network_name\":\"plan507\"}'::jsonb WHERE round_id=%s", (rid,))
            cur.execute("SELECT attname FROM pg_attribute WHERE attrelid='lab_arena_rounds'::regclass "
                        "AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum")
            cols = [r[0] for r in cur.fetchall()]
            repl = {'round_id':"'arena-2026-10-20-plan507'||s", 'reward_basis_hash':'NULL',
                    'effective_reward_epoch':'NULL'}
            cur.execute('INSERT INTO lab_arena_rounds ('+','.join(cols)+') SELECT '+
                        ','.join(repl.get(k,'r.'+k) for k in cols)+
                        ' FROM lab_arena_rounds r CROSS JOIN generate_series(1,59) s WHERE round_id=%s', (rid,))
            cur.execute("SELECT attname FROM pg_attribute WHERE attrelid='lab_arena_runs'::regclass "
                        "AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum")
            cols = [r[0] for r in cur.fetchall()]
            repl = {'run_id':"repeat('r',48)||'plan507-'||s", 'assignment_id':"repeat('r',48)||'plan507-'||s",
                    'round_id':"CASE WHEN s%%60=0 THEN r.round_id ELSE 'arena-2026-10-20-plan507'||(s%%60) END",
                    'status':"'accepted'", 'terminal_cause':"'accepted'",
                    'result_doc':"'{\"terminal_status\":\"accepted\"}'::jsonb",
                    'output_ref':"'s3://fixture/score.json'",
                    'claim_request_id':'NULL'}
            cur.execute('INSERT INTO lab_arena_runs ('+','.join(cols)+') SELECT '+
                        ','.join(repl.get(k,'r.'+k) for k in cols)+
                        ' FROM lab_arena_runs r CROSS JOIN generate_series(1,40000) s WHERE run_id=%s',
                        (run['run_id'],))
            cur.execute('SELECT max(entry_id) FROM lab_arena_ledger')
            maximum = cur.fetchone()[0]
            cur.execute("SELECT attname FROM pg_attribute WHERE attrelid='lab_arena_ledger'::regclass "
                        'AND attnum>0 AND NOT attisdropped ORDER BY attnum')
            cols = [r[0] for r in cur.fetchall()]
            # Keep one native-proven terminal shape for both call outcomes.
            # Mixing correlated status/cause pairs underestimates eligible runs
            # and loses the historical 60-round rescan this benchmark measures.
            # Keep the unresolved/settled and provider mix, using real per-kind
            # broker documents rather than relabelled OpenRouter uncertainties.
            cur.execute('CREATE TEMP TABLE plan507_templates (provider text, settled boolean, '
                        'entry_kind text, amount bigint, doc jsonb, terminal jsonb, cid text, request_hash text)')
            request_hashes = {}
            for provider, settled, row, call in templates:
                request_hashes.setdefault(provider, call['request_hash'])
                assert request_hashes[provider] == call['request_hash']
                cur.execute('INSERT INTO plan507_templates VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s)',
                    (provider, settled, row['entry_kind'], row['amount_microusd'],
                     json.dumps(row['entry_doc']),
                     None if row['terminal_response'] is None else json.dumps(row['terminal_response']),
                     row['call_identity'], call['request_hash']))
            identity = contracts.canonical_json(dict(call=contracts.PROVIDER_CALL_SCHEMA_VERSION,
                assignment_id='__assignment__', attempt=run['attempt'], icp_position=run['icp_position'],
                action_sequence='__sequence__', operation_id='__operation__', request_hash='__request__'))
            cur.execute("""CREATE TEMP TABLE plan507_calls AS
                SELECT g, 1+g%%40000 AS run_number,
                  CASE WHEN g%%3=0 THEN 'openrouter' ELSE 'deepline' END AS provider,
                  g%%2=1 AS settled,
                  'sha256:'||encode(extensions.digest(
                    replace(replace(replace(replace(%s,
                      '__assignment__',repeat('r',48)||'plan507-'||(1+g%%40000)),
                      '"__sequence__"',(g/40000)::text),
                      '__operation__',CASE WHEN g%%3=0 THEN 'openrouter.chat' ELSE 'scrapingdog.scrape' END),
                      '__request__',CASE WHEN g%%3=0 THEN %s ELSE %s END), 'sha256'),'hex') AS cid
                FROM generate_series(0,166666) g""",
                (identity, request_hashes['openrouter'], request_hashes['deepline']))
            # Rebind both JSON and encoded response bodies to the cloned call.
            cur.execute("""CREATE FUNCTION pg_temp.plan507_rebind(doc jsonb, oldcid text, newcid text, g integer)
                RETURNS jsonb LANGUAGE sql IMMUTABLE AS $rebind$
                SELECT CASE WHEN rewritten ? 'body_b64' THEN jsonb_set(rewritten,'{body_b64}',
                    to_jsonb(replace(encode(convert_to(replace(replace(replace(
                      convert_from(decode(doc->>'body_b64','base64'),'UTF8'),
                      substr(oldcid,8),substr(newcid,8)),substr(oldcid,8,32),substr(newcid,8,32)),
                      'gen-template507','gen-plan507-'||g),'UTF8'),'base64'),chr(10),'')))
                  ELSE rewritten END
                FROM (SELECT replace(replace(replace(doc::text,
                    substr(oldcid,8),substr(newcid,8)),substr(oldcid,8,32),substr(newcid,8,32)),
                    'gen-template507','gen-plan507-'||g)::jsonb AS rewritten) q
                $rebind$""")
            repl = {'entry_id':'%s+s', 'run_id':"repeat('r',48)||'plan507-'||c.run_number",
                    'round_id':"CASE WHEN c.run_number%%60=0 THEN l.round_id ELSE "
                               "'arena-2026-10-20-plan507'||(c.run_number%%60) END",
                    'call_identity':'c.cid', 'entry_kind':'t.entry_kind', 'provider':'c.provider',
                    'operation_id':"CASE WHEN c.provider='openrouter' THEN 'openrouter.chat' ELSE 'scrapingdog.scrape' END",
                    'amount_microusd':'t.amount',
                    'entry_doc':"CASE WHEN t.entry_kind='reservation' THEN jsonb_set("
                        "pg_temp.plan507_rebind(t.doc,t.cid,c.cid,c.g),'{action_sequence}',to_jsonb(c.g/40000)) "
                        "ELSE pg_temp.plan507_rebind(t.doc,t.cid,c.cid,c.g) END",
                    'terminal_response':'pg_temp.plan507_rebind(t.terminal,t.cid,c.cid,c.g)'}
            # Rebinding native JSON is setup work. Keep every insert bounded
            # by 20s and retain exactly 500k rows for all measured queries.
            for first in range(1, 500001, 25000):
                cur.execute('INSERT INTO lab_arena_ledger SELECT '+','.join(repl.get(k,'l.'+k) for k in cols)+
                            " FROM lab_arena_ledger l CROSS JOIN generate_series(%s,%s) s "
                            "JOIN plan507_calls c ON c.g=(s-1)/3 JOIN plan507_templates t "
                            "ON t.provider=c.provider AND t.settled=c.settled AND t.entry_kind="
                            "CASE WHEN s%%3=1 THEN 'reservation' WHEN s%%3=2 THEN 'dispatch' "
                            "WHEN c.settled THEN 'settlement' ELSE 'uncertain' END "
                            "WHERE l.call_identity=%s AND l.entry_kind='uncertain'",
                            (maximum, first, first + 24999, cid))
                assert cur.rowcount == 25000
            cur.execute("SELECT count(*) FROM lab_arena_runs WHERE run_id LIKE repeat('r',48)||'plan507-%%'")
            assert cur.fetchone() == (40000,)
            cur.execute("SELECT count(*) FROM lab_arena_rounds WHERE configuration_doc->>'network_name'='plan507'")
            assert cur.fetchone() == (60,)
            cur.execute("SELECT count(*),count(*) FILTER(WHERE entry_kind='uncertain'),"
                        "count(*) FILTER(WHERE provider='deepline' AND operation_id='openrouter.chat') "
                        "FROM lab_arena_ledger WHERE entry_id>%s", (maximum,))
            assert cur.fetchone() == (500000, 83333, 0)
            sequences = set()
            for order in ('ASC', 'DESC'):
                cur.execute(f"""SELECT u.entry_id, u.call_identity, r.entry_doc, u.entry_doc,
                        runs.assignment_id, runs.attempt, runs.icp_position,
                        public.lab_arena__deepline_cost_binding_v1(u.entry_doc,r.entry_doc,u.call_identity)
                    FROM lab_arena_ledger u JOIN lab_arena_ledger r USING(call_identity)
                    JOIN lab_arena_runs runs ON runs.run_id=u.run_id
                    WHERE u.entry_id>%s AND u.entry_kind='uncertain' AND r.entry_kind='reservation'
                      AND u.provider='deepline' ORDER BY u.entry_id {order} LIMIT 3""", (maximum,))
                for _, identity, reservation, uncertainty, assignment, attempt, position, bound in cur.fetchall():
                    sequences.add(reservation['action_sequence'])
                    assert bound and uncertainty['call']['call_succeeded'] is False
                    assert reservation['deepline_execution_key'] == 'arena:'+identity[7:]
                    assert reservation['deepline_request_id'] == 'ctx-tool-'+identity[7:39]
                    assert identity == contracts.provider_call_identity(assignment_id=assignment,
                        attempt=attempt, icp_position=position, action_sequence=reservation['action_sequence'],
                        operation_id='scrapingdog.scrape', request_hash=reservation['request_hash'])
                    for field in ('deepline_request_id', 'credential_fingerprint'):
                        malformed = dict(uncertainty, call=dict(uncertainty['call'], **{field: 'wrong'}))
                        cur.execute('SELECT public.lab_arena__deepline_cost_binding_v1(%s::jsonb,%s::jsonb,%s)',
                                    (json.dumps(malformed), json.dumps(reservation), identity))
                        assert cur.fetchone() == (False,)
                cur.execute(f"""SELECT call_identity,terminal_response
                    FROM lab_arena_ledger WHERE entry_id>%s AND provider='deepline'
                      AND entry_kind='settlement' ORDER BY entry_id {order} LIMIT 3""", (maximum,))
                for identity, terminal in cur.fetchall():
                    assert base64.b64decode(terminal['body_b64'], validate=True) == b'<html>synthetic</html>'
                    assert terminal['provider_cost']['request_id'] == 'ctx-tool-'+identity[7:39]
                cur.execute(f"""SELECT c.g,l.terminal_response FROM lab_arena_ledger l
                    JOIN plan507_calls c ON c.cid=l.call_identity
                    WHERE l.entry_id>%s AND l.provider='openrouter' AND l.entry_kind='settlement'
                    ORDER BY l.entry_id {order} LIMIT 3""", (maximum,))
                for sequence, terminal in cur.fetchall():
                    body = json.loads(base64.b64decode(terminal['body_b64'], validate=True))
                    assert body['id'] == 'gen-plan507-'+str(sequence)
            assert sequences == {0, 4}
            for table in ('lab_arena_ledger','lab_arena_runs','lab_arena_rounds'):
                cur.execute('ANALYZE '+table)
            cur.execute('SET LOCAL plan_cache_mode=force_generic_plan')
            cur.execute("SELECT current_setting('server_version'),current_setting('jit'),"
                        "current_setting('work_mem'),current_setting('random_page_cost')")
            settings = cur.fetchone()
            assert settings[0].startswith('17.9') and settings[1:] == ('off',work_mem,random_page_cost)
            results = []; plans = {'settings': settings}
            for name in ('test_before_'+prior.NAMES[0], prior.NAMES[0]):
                # The obsolete control deliberately rescans spilled history 60
                # times. Give only that reference a bounded allowance; every
                # optimized507 execution retains the deployed 20s budget.
                cur.execute('SET LOCAL statement_timeout=%s',
                            ('60s' if name.startswith('test_before_') else '20s',))
                cur.execute('SELECT public.'+name+"('live','plan507',71,'',0)")
                results.append(cur.fetchone()[0])
                cur.execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON, TIMING FALSE) SELECT public.'+
                            name+"('live','plan507',71,'',0)")
                full_plan = cur.fetchone()[0][0]
                plans[name]=full_plan
                plan = full_plan['Plan']
                if name.startswith('test_before_'):
                    before_reads = plan['Temp Read Blocks']
                else:
                    after_reads = plan['Temp Read Blocks']
            generic_reads = {}; generic_loops = {}
            def cte_loops(node):
                values = [node.get('Actual Loops',0)] if node.get('CTE Name')=='uncertainties' else []
                for child in node.get('Plans',[]):
                    values.extend(cte_loops(child))
                return values
            for index,name in enumerate(('test_before_'+prior.NAMES[0],prior.NAMES[0])):
                cur.execute('SET LOCAL statement_timeout=%s',
                            ('60s' if index == 0 else '20s',))
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',('public.'+name+prior.SUFFIX,))
                body = cur.fetchone()[0].split('AS $function$',1)[1].split('$function$',1)[0].strip().removesuffix(';')
                literal = body
                for key,value in {'p_mode':"'live'",'p_network_name':"'plan507'",'p_netuid':'71',
                                  'p_round_id':"''",'p_after_entry_id':'0'}.items():
                    literal = literal.replace(key,value)
                cur.execute('EXPLAIN (FORMAT JSON, COSTS TRUE) '+literal)
                plans['literal_'+name]=cur.fetchone()[0][0]
                for number,key in enumerate(('p_mode','p_network_name','p_netuid','p_round_id','p_after_entry_id'),1):
                    body = body.replace(key,'$'+str(number))
                prepared = 'plan507_'+str(index)
                cur.execute('PREPARE '+prepared+'(text,text,integer,text,bigint) AS '+body)
                cur.execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON, TIMING FALSE) EXECUTE '+
                            prepared+"('live','plan507',71,'',0)")
                full_plan=cur.fetchone()[0][0]
                plans['generic_'+name]=full_plan
                generic_reads[index]=full_plan['Plan']['Temp Read Blocks']
                generic_loops[index]=max(cte_loops(full_plan['Plan']))
            (tmp_path/'plans.json').write_text(json.dumps(plans,indent=2))
            assert generic_loops=={0:60,1:1}
            assert generic_reads[0]>0 and generic_reads[1]<generic_reads[0]/10
            assert results == [{'status':'none'}]*2
            assert before_reads > 0 and after_reads < before_reads / 10
            # Keep the history isolated from later callers of this module fixture.
            conn.rollback()
    finally:
        store.close()
