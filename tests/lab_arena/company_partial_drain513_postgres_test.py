"""Durable company partials close captured leases without weakening finality."""
import hashlib
import json
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.service import ArenaService, ServiceConfig, S3ObjectStore
from tests.lab_arena import partial_company_judgments_postgres_test as partial
from tests.lab_arena.test_lab_arena_service_round import keypair, wallet_verify, FakeChain
from tests.lab_arena.test_lab_arena_object_store import _ConditionalS3
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena import signal_judgments503_postgres_test as schema
from tests.lab_arena import legacy_company_cache501_postgres_test as company
from tests.lab_arena import owner_stopped_score_drain454_postgres_test as stopped

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/513-lab-arena-company-partial-drain-receipt.sql'
SIG = 'public.lab_arena__restart_drain_state_v1()'
PRE = '3ed11092880c10bad608420b33c076773557a308288d39ca0a45df9a95990e5a'
POST = '49f2d7713419d20c182bf1548f9e731cb369611b5121ab6b116239ba2a8f77f2'


def definition(cur):
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIG,))
    return cur.fetchone()[0]


def metadata(cur):
    cur.execute("SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (SIG,))
    return cur.fetchone()[0]


def drain(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena__restart_drain_state_v1()')
        return cur.fetchone()[0]


def seed(db, *, cause='judge_error', empty=False, accepted=False, quality=False):
    store = ArenaStore(PsycopgTransport(lambda: db[0].connect(**db[1])))
    store._test_database = db
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]' WHERE singleton")
        rid = 'arena-2099-10-20-drain513'
        _, items = company.open_new(store, round_id=rid, markers_by_submission=[['a','b','c']], quality=quality)
        # The inherited SQL-only fixture carries intent-details configuration
        # while its integrity scorer policy does not. Match this fixture's
        # scorer before creating a lease for the real signed service path.
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc-'intent_details_policy' WHERE round_id=%s", (rid,))
            cur.execute('SET LOCAL session_replication_role=origin')
        key = keypair('drain513-validator')
        runner = key.ss58_address
        lease, token, _, _ = company.original.claim(store, rid, runner, excluded=[runner])
        assert lease['status'] == 'leased'
        with db[0].connect(**db[1]) as conn:
            stopped._hold_capture(conn)
        if empty:
            result = store.complete_attempt(run_id=lease['run_id'], lease_token_hash=company.original.hash_lease_token(token),
                result={'terminal_status': cause}, terminal_cause=cause, output_ref='')
        elif not accepted:
            service = ArenaService(ServiceConfig(mode='live', store=store,
                object_store=S3ObjectStore('test-arena', client=_ConditionalS3()),
                signer=None, chain=FakeChain([runner]), verify_signature=wallet_verify,
                daily_icp_source=lambda **_: {}, banned_hotkeys_source=lambda: (),
                broker_factory=lambda *_: None))
            envelope = partial._request(rid, key, lease, token, terminal=cause)
            from tests.test_arena_score_integrity import _positive_breakdown
            from qualification.scoring.competition import raw_company_judgment
            rows = envelope['body']['output']['company_judgments']
            for index, row in enumerate(rows):
                marker = 'company'+str(index)
                if quality:
                    from tests.test_arena_company_quality_policy import _positive_breakdown as quality_breakdown
                    raw = raw_company_judgment(quality_breakdown(marker, marker+'.example', marker))
                else:
                    raw = raw_company_judgment(_positive_breakdown(marker, marker+'.example'))
                raw.update(company_identity_key='domain:'+marker+'.example|name:'+marker,
                    company_identity_alias_keys=['domain:'+marker+'.example|name:'+marker])
                if not quality:
                    raw['company_judgment_policy'] = 'company_judgment_v1'
                rows[index] = {**{k: row[k] for k in ('company_index','cache_key','company_input_hash','authority_slot')},
                               'raw_judgment': raw}
            envelope = partial._sign(rid, key, envelope['body'])
            result = service.handle_complete(envelope)
            assert store.get_run(lease['run_id'])['result_doc']['company_judgment_completion_request_hash'] == contracts.request_bytes_hash(envelope)
        else:
            # The ordinary v3 RPC validates every frozen miss, reservation,
            # original request, source, token and generation before caching.
            result = company.complete_subset(store, lease, token, items[0]['company_judgment_refs'],
                limit=None if accepted else 2, terminal='accepted' if accepted else cause)
        assert result['status'] == ('accepted' if accepted else 'failed')
        stored = store.get_run(lease['run_id'])
        lease.update(round_id=rid, miner_hotkey=stored['miner_hotkey'])
        return lease, stored
    finally:
        store.close()


@pytest.fixture(scope='module')
def database():
    gen = schema.database.__wrapped__()
    db = next(gen)
    try:
        lease, before_row = seed(db)
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            assert hashlib.sha256(definition(cur).encode()).hexdigest() == PRE
            before = metadata(cur)
            assert drain(conn)['lost_or_mutated_count'] == 1
            cur.execute(SQL.read_text())
            assert metadata(cur) == before
            assert hashlib.sha256(definition(cur).encode()).hexdigest() == POST
            assert drain(conn)['reported_terminal_receipt_count'] == 1
            assert drain(conn)['lost_or_mutated_count'] == 0
            assert drain(conn)['preserved'] is True
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s', (lease['run_id'],))
            assert cur.fetchone()[0] == before_row
            cur.execute(SQL.read_text())
            assert metadata(cur) == before
        yield db
    finally:
        gen.close()


@pytest.mark.parametrize('cause', ['judge_error', 'judge_timeout', 'credential_error'])
def test_company_partial_is_reported_and_never_accepted(database, cause):
    lease, before = seed(database, cause=cause)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        state = drain(conn)
        assert state['captured_count'] == state['reported_terminal_receipt_count'] == 1
        assert state['accepted_receipt_count'] == state['lost_or_mutated_count'] == 0
        assert state['preserved'] is True
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s', (lease['run_id'],))
        assert cur.fetchone()[0] == before
        cur.execute('SELECT count(*) FROM public.lab_arena_company_judgments WHERE source_score_run_id=%s', (lease['run_id'],))
        assert cur.fetchone()[0] == 2


@pytest.mark.parametrize('mode', ['empty', 'accepted'])
def test_existing_receipts_unchanged(database, mode):
    seed(database, empty=mode=='empty', accepted=mode=='accepted')
    with database[0].connect(**database[1]) as conn:
        state = drain(conn)
        assert state['preserved'] is True and state['lost_or_mutated_count'] == 0
        assert state['accepted_receipt_count'] == (mode == 'accepted')
        assert state['reported_terminal_receipt_count'] == (mode == 'empty')


@pytest.mark.parametrize('fault', [
    'missing_cache','cache_hit_only','completion_hash','output_ref','claim_id','claim_hash',
    'runner','scored_run','generation','captured_generation','authority',
    'misses','refs','scope','terminal','kind','policy','open_dispatch',
    'evidence_output','evidence_request','evidence_generation','evidence_slot',
])
def test_unbound_partial_remains_lost(database, fault):
    lease, _ = seed(database)
    run_id = lease['run_id']
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        if fault == 'missing_cache':
            cur.execute('DELETE FROM public.lab_arena_company_judgments WHERE source_score_run_id=%s', (run_id,))
        elif fault == 'cache_hit_only':
            cur.execute("UPDATE public.lab_arena_company_judgments SET source_score_run_id=source_scored_run_id,evidence_doc=jsonb_set(evidence_doc,'{source_score_run_id}',to_jsonb(source_scored_run_id)) WHERE source_score_run_id=%s", (run_id,))
        elif fault == 'completion_hash':
            cur.execute("UPDATE public.lab_arena_runs SET result_doc=result_doc-'company_judgment_completion_request_hash' WHERE run_id=%s", (run_id,))
        elif fault == 'output_ref':
            cur.execute("UPDATE public.lab_arena_runs SET output_ref='other-output' WHERE run_id=%s", (run_id,))
        elif fault in ('claim_id','claim_hash','runner','scored_run'):
            field = {'claim_id':'claim_request_id','claim_hash':'claim_request_hash','runner':'runner_hotkey','scored_run':'scored_run_id'}[fault]
            value = 'sha256:'+'f'*64 if fault=='claim_hash' else 'other'
            cur.execute(f'UPDATE public.lab_arena_runs SET {field}=%s WHERE run_id=%s', (value,run_id))
        elif fault == 'generation':
            cur.execute('UPDATE public.lab_arena_runs SET lease_generation=lease_generation+1 WHERE run_id=%s', (run_id,))
        elif fault == 'captured_generation':
            cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases=jsonb_set(captured_leases,'{0,lease_generation}','2')")
        elif fault == 'authority':
            cur.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{runner_authority_exclusions}','[]') WHERE run_id=%s", (run_id,))
        elif fault == 'misses':
            cur.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{company_judgment_cache,misses}','[]') WHERE run_id=%s", (run_id,))
        elif fault == 'refs':
            cur.execute("UPDATE public.lab_arena_runs SET company_judgment_refs='[]' WHERE run_id=%s", (run_id,))
        elif fault == 'scope':
            cur.execute("UPDATE public.lab_arena_company_judgments SET scope_doc=jsonb_set(scope_doc,'{evaluation_date}','\"2099-10-19\"') WHERE source_score_run_id=%s", (run_id,))
        elif fault == 'terminal':
            cur.execute("UPDATE public.lab_arena_runs SET result_doc=jsonb_set(result_doc,'{terminal_status}','\"accepted\"') WHERE run_id=%s", (run_id,))
        elif fault == 'kind':
            cur.execute("UPDATE public.lab_arena_runs SET kind='execute',company_judgment_refs=NULL WHERE run_id=%s", (run_id,))
        elif fault == 'policy':
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc-'company_judgment_policy' WHERE round_id=%s", (lease['round_id'],))
        elif fault == 'open_dispatch':
            cur.execute("""INSERT INTO public.lab_arena_ledger
              (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
               call_identity,provider,operation_id,funding_source,amount_microusd)
              VALUES ('dispatch',%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat','host',100)""",
              (lease['miner_hotkey'],lease['round_id'],lease['submission_id'],run_id,'sha256:'+'e'*64))
        elif fault == 'evidence_slot':
            cur.execute("UPDATE public.lab_arena_company_judgments SET authority_slot=1,evidence_doc=jsonb_set(evidence_doc,'{authority_slot}','1') WHERE source_score_run_id=%s", (run_id,))
        else:
            field, value = {'evidence_output':('source_output_ref','"other"'),
                'evidence_request':('source_completion_request_hash','"sha256:'+'f'*64+'"'),
                'evidence_generation':('source_lease_generation','2'),
                'evidence_slot':('authority_slot','1')}[fault]
            cur.execute('UPDATE public.lab_arena_company_judgments SET evidence_doc=jsonb_set(evidence_doc,%s,%s::jsonb) WHERE source_score_run_id=%s', ('{'+field+'}',value,run_id))
        cur.execute('SET LOCAL session_replication_role=origin')
        state = drain(conn)
        assert state['lost_or_mutated_count'] == 1, (fault,state)
        assert state['reported_terminal_receipt_count'] == 0 and state['preserved'] is False


@pytest.mark.parametrize('fault', ['body', 'acl', 'owner', 'path', 'volatile'])
def test_migration_fails_closed_on_drift(database, fault):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        source = definition(cur)
        if fault == 'body':
            cur.execute(source.replace("RETURN pg_catalog.jsonb_build_object(", "RETURN pg_catalog.jsonb_build_object(\n",1))
        elif fault == 'acl':
            cur.execute('GRANT EXECUTE ON FUNCTION '+SIG+' TO anon')
        elif fault == 'owner':
            cur.execute('ALTER FUNCTION '+SIG+' OWNER TO postgres')
        elif fault == 'path':
            cur.execute('ALTER FUNCTION '+SIG+' SET search_path=public')
        else:
            cur.execute('ALTER FUNCTION '+SIG+' VOLATILE')
        with pytest.raises(database[0].Error, match='differs'):
            cur.execute(SQL.read_text())
        conn.rollback()
        assert definition(cur) == source


def test_company_quality_partial_uses_same_bound_protocol(database):
    seed(database, quality=True)
    with database[0].connect(**database[1]) as conn:
        state = drain(conn)
        assert state['reported_terminal_receipt_count'] == 1
        assert state['lost_or_mutated_count'] == 0 and state['preserved'] is True


def test_legacy_partial_remains_reported(database):
    from tests.lab_arena import bound_partial_score_drain479_postgres_test as legacy
    with database[0].connect(**database[1]) as conn:
        lease, output_ref, receipt, _, _ = legacy._seed(conn)
        stopped._hold_capture(conn)
        assert legacy._complete(conn, lease, output_ref=output_ref, receipt=receipt)['status'] == 'failed'
        state = drain(conn)
        assert state['reported_terminal_receipt_count'] == 1
        assert state['lost_or_mutated_count'] == 0 and state['preserved'] is True


def test_non_superuser_apply_replay_keep_catalog_and_schema_acl(database):
    import re
    source = SQL.read_text()
    old = re.search(r'old_gate CONSTANT TEXT := \$old\$(.*?)\$old\$;', source, re.S)[1]
    new = re.search(r'new_gate CONSTANT TEXT := \$new\$(.*?)\$new\$;', source, re.S)[1]
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        post = definition(cur)
        assert post.count(new) == 1
        pre = post.replace(new, old)
        assert hashlib.sha256(pre.encode()).hexdigest() == PRE
        cur.execute(pre)
        before = metadata(cur)
        cur.execute('CREATE ROLE company513_migrator NOLOGIN NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO company513_migrator')
        cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO company513_migrator')
        cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE company513_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
        cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        acl = cur.fetchone()[0]
        cur.execute('SET ROLE company513_migrator')
        cur.execute("SELECT rolsuper,has_schema_privilege('lab_arena_owner','public','CREATE') FROM pg_roles WHERE rolname=current_user")
        assert cur.fetchone() == (False, False)
        for _ in range(2):
            cur.execute(source)
            assert metadata(cur) == before
            assert hashlib.sha256(definition(cur).encode()).hexdigest() == POST
            cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            assert cur.fetchone()[0] == acl
            for role in ('anon','authenticated','service_role','lab_arena_service'):
                cur.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role,SIG))
                assert cur.fetchone() == (False,)
        cur.execute('RESET ROLE')


@pytest.mark.parametrize('valid', [True, False])
def test_canonical_quiescence_authorize_and_release_require_bound_receipt(database, valid):
    lease, _ = seed(database)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        if not valid:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('DELETE FROM public.lab_arena_company_judgments WHERE source_score_run_id=%s', (lease['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
        generation = cur.fetchone()[0]
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r ORDER BY run_id')
        runs = cur.fetchall()
        cur.execute('SELECT to_jsonb(c) FROM public.lab_arena_company_judgments c ORDER BY cache_key,authority_slot')
        cache = cur.fetchall()
        state = stopped.guard._quiescence(conn, generation=generation)
        assert state['preserved'] is valid
        args = (stopped.guard.GUARD, stopped.guard.OWNER, generation, 'gateway_destructive')
        if valid:
            phase = stopped.guard._rpc(conn, 'lab_arena_authorize_restart_phase_v1', *args)
            assert phase['restart_phase'] == 'gateway_destructive'
            ready = stopped.guard._rpc(conn, 'lab_arena_mark_restart_ready_v1',
                stopped.guard.GUARD, stopped.guard.OWNER, generation, 'gateway_ready')
            assert ready['restart_phase'] == 'gateway_ready'
            released = stopped.guard._rpc(conn, 'lab_arena_release_restart_guard_v1',
                stopped.guard.GUARD, stopped.guard.OWNER, generation, 'company-partial-test')
            assert released['guard_present'] is False
        else:
            cur.execute('SAVEPOINT rejected_authority')
            with pytest.raises(database[0].Error, match='lab_arena_restart_drain_not_preserved'):
                stopped.guard._rpc(conn, 'lab_arena_authorize_restart_phase_v1', *args)
            cur.execute('ROLLBACK TO SAVEPOINT rejected_authority')
            cur.execute('SELECT restart_phase FROM public.lab_arena_restart_claim_control WHERE singleton')
            assert cur.fetchone() == ('draining',)
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r ORDER BY run_id')
        assert cur.fetchall() == runs  # Includes accepted source and frozen successors.
        cur.execute('SELECT to_jsonb(c) FROM public.lab_arena_company_judgments c ORDER BY cache_key,authority_slot')
        assert cur.fetchall() == cache
