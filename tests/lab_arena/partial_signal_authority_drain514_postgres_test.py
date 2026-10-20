"""A partial company receipt inherits the canonical reused-signal authority."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lab_arena import contracts
from lab_arena.store import hash_lease_token
from tests.lab_arena import company_partial_drain513_postgres_test as prior

SQL = Path(__file__).parents[2] / 'scripts/514-lab-arena-partial-signal-authority-drain.sql'
PRE = prior.POST
POST = 'a38782b4b6555c455fae64a2434d0bbd394f2f970bc42326475f187a4a1756b8'
HELPER = 'public.lab_arena_signal_authority_exclusions_v1(text)'
HELPER_SHA = 'a2dcb05a80bf6b9d45b047eb6b9bbb1d3db88c09e587cc476d9d6b8190352a47'


def expanded_partial(db):
    """Real signal miss/store/hit, then real signed company v3 failure."""
    open_new = prior.company.open_new
    claim = prior.company.original.claim
    observed = {}
    inherited = prior.company.original.hotkey('signal514-inherited-authority')

    def opened(store, **kwargs):
        kwargs['markers_by_submission'] = [['source','source-tail'], ['a','b','c']]
        value = open_new(store, **kwargs)
        prior.schema._enable_round(db, kwargs['round_id'])
        return value

    def claimed(store, round_id, runner, **kwargs):
        source, token, _, _ = claim(store, round_id, runner,
                                   **{**kwargs, 'excluded': [runner, inherited]})
        assert source['status'] == 'leased'
        cfg = store.get_round(round_id)['configuration_doc']
        scope = prior.schema._scope(round_id, 'shared514')
        scope.update(scorer_image_digest=cfg['scorer_image_digest'],
                     scorer_image_reference=cfg['scorer_image_reference'])
        scope.pop('cache_key')
        scope['cache_key'] = contracts.document_hash(scope)
        args = dict(company_index=0, signal_index=0, scope=scope)
        miss = store.claim_signal_judgment(run_id=source['run_id'],
            lease_token_hash=hash_lease_token(token), **args)
        assert miss['status'] == 'miss'
        judgment = {'raw': 1, 'decision': 'terminal'}
        assert store.complete_signal_judgment(run_id=source['run_id'],
            lease_token_hash=hash_lease_token(token), reference=miss['reference'],
            judgment=judgment, judgment_hash=contracts.document_hash(judgment),
            **args)['status'] == 'stored'
        assert store.complete_attempt(run_id=source['run_id'],
            lease_token_hash=hash_lease_token(token), result={'terminal_status':'judge_error'},
            terminal_cause='judge_error', output_ref='', company_judgment_evidence=[],
            output_hash='', completion_request_hash='sha256:'+'e'*64)['status'] == 'failed'
        result = claim(store, round_id, runner, **kwargs)
        recipient, recipient_token, _, _ = result
        assert recipient['status'] == 'leased' and recipient['run_id'] != source['run_id']
        hit = store.claim_signal_judgment(run_id=recipient['run_id'],
            lease_token_hash=hash_lease_token(recipient_token), **args)
        assert hit['status'] == 'hit' and hit['judgment'] == judgment
        original = store.get_run(recipient['run_id'])['claim_response']['runner_authority_exclusions']
        canonical = store.signal_authority_exclusions(recipient['run_id'])
        assert inherited not in original and inherited in canonical
        observed.update(source_run_id=source['run_id'], original=original,
                        canonical=canonical, inherited=inherited)
        return result

    with patch.object(prior.company, 'open_new', opened), patch.object(prior.company.original, 'claim', claimed):
        lease, row = prior.seed(db)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SELECT evidence_doc FROM public.lab_arena_company_judgments WHERE source_score_run_id=%s', (lease['run_id'],))
        evidence = [r[0] for r in cur.fetchall()]
        assert evidence and all(e['runner_authority_exclusions'] == observed['canonical'] for e in evidence)
        assert all(e['source_completion_request_hash'] == row['result_doc']['company_judgment_completion_request_hash'] for e in evidence)
    return lease, row, observed


def snapshots(cur):
    result = {}
    for table in ('lab_arena_runs','lab_arena_company_judgments',
                  'lab_arena_signal_judgments','lab_arena_signal_judgment_uses',
                  'lab_arena_ledger','lab_arena_restart_claim_control'):
        cur.execute('SELECT to_jsonb(r) FROM public.'+table+' r ORDER BY to_jsonb(r)::text')
        result[table] = cur.fetchall()
    return result


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        expanded_partial(db)
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            assert hashlib.sha256(prior.definition(cur).encode()).hexdigest() == PRE
            assert prior.drain(conn)['lost_or_mutated_count'] == 1
            rows = snapshots(cur)
            before = prior.metadata(cur)
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure),to_jsonb(p) FROM pg_proc p WHERE oid=%s::regprocedure', (HELPER,HELPER))
            helper = cur.fetchone()
            assert hashlib.sha256(helper[0].encode()).hexdigest() == HELPER_SHA
            cur.execute(SQL.read_text())
            assert prior.metadata(cur) == before
            assert hashlib.sha256(prior.definition(cur).encode()).hexdigest() == POST
            assert prior.drain(conn)['reported_terminal_receipt_count'] == 1
            assert prior.drain(conn)['preserved'] is True
            assert snapshots(cur) == rows
            cur.execute(SQL.read_text())
            assert prior.metadata(cur) == before and snapshots(cur) == rows
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure),to_jsonb(p) FROM pg_proc p WHERE oid=%s::regprocedure', (HELPER,HELPER))
            assert cur.fetchone() == helper
        yield db
    finally:
        gen.close()


def test_expanded_signal_partial_authorizes_canonical_drain_without_writes(database):
    lease, _, _ = expanded_partial(database)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        before = snapshots(cur)
        cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
        generation = cur.fetchone()[0]
        state = prior.stopped.guard._quiescence(conn, generation=generation)
        assert state['reported_terminal_receipt_count'] == 1 and state['preserved'] is True
        phase = prior.stopped.guard._rpc(conn, 'lab_arena_authorize_restart_phase_v1',
            prior.stopped.guard.GUARD, prior.stopped.guard.OWNER, generation, 'gateway_destructive')
        assert phase['restart_phase'] == 'gateway_destructive'
        after = snapshots(cur)
        for table in before:
            if table != 'lab_arena_restart_claim_control':
                assert after[table] == before[table]


@pytest.mark.parametrize('fault', ['original_only','extra','missing','reordered','removed_use','open_dispatch'])
def test_mismatched_signal_authority_or_unsettled_ledger_remains_lost(database, fault):
    lease, _, observed = expanded_partial(database)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        if fault in ('original_only','extra','reordered'):
            value = {'original_only':observed['original'],
                'extra':observed['canonical']+[prior.company.original.hotkey('signal514-forged')],
                'reordered':list(reversed(observed['canonical']))}[fault]
            cur.execute("UPDATE public.lab_arena_company_judgments SET evidence_doc=jsonb_set(evidence_doc,'{runner_authority_exclusions}',%s::jsonb) WHERE source_score_run_id=%s", (json.dumps(value),lease['run_id']))
        elif fault == 'missing':
            cur.execute("UPDATE public.lab_arena_company_judgments SET evidence_doc=evidence_doc-'runner_authority_exclusions' WHERE source_score_run_id=%s", (lease['run_id'],))
        elif fault == 'removed_use':
            cur.execute('DELETE FROM public.lab_arena_signal_judgment_uses WHERE run_id=%s', (lease['run_id'],))
        else:
            cur.execute("""INSERT INTO public.lab_arena_ledger
              (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
               call_identity,provider,operation_id,funding_source,amount_microusd)
              VALUES ('dispatch',%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat','host',100)""",
              (lease['miner_hotkey'],lease['round_id'],lease['submission_id'],lease['run_id'],'sha256:'+'e'*64))
        cur.execute('SET LOCAL session_replication_role=origin')
        state = prior.drain(conn)
        assert state['lost_or_mutated_count'] == 1 and state['preserved'] is False
        cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
        generation = cur.fetchone()[0]
        cur.execute('SAVEPOINT denied')
        with pytest.raises(database[0].Error, match='lab_arena_restart_drain_not_preserved'):
            prior.stopped.guard._rpc(conn, 'lab_arena_authorize_restart_phase_v1',
                prior.stopped.guard.GUARD, prior.stopped.guard.OWNER, generation, 'gateway_destructive')
        cur.execute('ROLLBACK TO SAVEPOINT denied')


@pytest.mark.parametrize('mode', ['company','quality','empty','accepted'])
def test_normal_non_signal_receipts_are_unchanged(database, mode):
    prior.seed(database, quality=mode=='quality', empty=mode=='empty', accepted=mode=='accepted')
    with database[0].connect(**database[1]) as conn:
        assert prior.drain(conn)['preserved'] is True


@pytest.mark.parametrize('fault', ['missing_cache','claim_hash','runner','generation','scope','authority'])
def test_original_negative_bindings_remain_closed(database, fault):
    prior.test_unbound_partial_remains_lost(database, fault)


@pytest.mark.parametrize('target,fault', [
    (target,fault) for target in ('drain','helper')
    for fault in ('body','acl','owner','path','volatile')
])
def test_migration_replay_rejects_security_or_source_drift(database, target, fault):
    sig = prior.SIG if target=='drain' else HELPER
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (sig,))
        before = cur.fetchone()[0]
        if fault=='body':
            cur.execute(before.replace('AS $function$','AS $function$\n',1))
        elif fault=='acl':
            cur.execute('GRANT EXECUTE ON FUNCTION '+sig+' TO anon')
        elif fault=='owner':
            cur.execute('ALTER FUNCTION '+sig+' OWNER TO postgres')
        elif fault=='path':
            cur.execute('ALTER FUNCTION '+sig+' SET search_path=public')
        else:
            cur.execute('ALTER FUNCTION '+sig+' VOLATILE')
        with pytest.raises(database[0].Error, match='differs'):
            cur.execute(SQL.read_text())
        conn.rollback()
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (sig,))
        assert cur.fetchone()[0] == before


def test_non_superuser_apply_and_replay_preserve_authority_and_rows(database):
    import re
    source = SQL.read_text()
    old = re.search(r'old_gate CONSTANT TEXT := \$old\$(.*?)\$old\$;', source, re.S)[1]
    new = re.search(r'new_gate CONSTANT TEXT := \$new\$(.*?)\$new\$;', source, re.S)[1]
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        post = prior.definition(cur)
        assert post.count(new) == 1
        pre = post.replace(new, old)
        assert hashlib.sha256(pre.encode()).hexdigest() == PRE
        cur.execute(pre)
        metadata = prior.metadata(cur)
        cur.execute('SELECT to_jsonb(p) FROM pg_proc p WHERE oid=%s::regprocedure', (HELPER,))
        helper = cur.fetchone()[0]
        rows = snapshots(cur)
        cur.execute('CREATE ROLE company514_migrator NOLOGIN NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO company514_migrator')
        cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO company514_migrator')
        cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE company514_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
        cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        acl = cur.fetchone()[0]
        cur.execute('SET ROLE company514_migrator')
        cur.execute("SELECT rolsuper,has_schema_privilege('lab_arena_owner','public','CREATE') FROM pg_roles WHERE rolname=current_user")
        assert cur.fetchone() == (False, False)
        for _ in range(2):
            cur.execute(source)
            assert prior.metadata(cur) == metadata
            assert hashlib.sha256(prior.definition(cur).encode()).hexdigest() == POST
            assert snapshots(cur) == rows
            cur.execute('SELECT to_jsonb(p) FROM pg_proc p WHERE oid=%s::regprocedure', (HELPER,))
            assert cur.fetchone()[0] == helper
            cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            assert cur.fetchone()[0] == acl
            for role in ('anon','authenticated','service_role','lab_arena_service'):
                cur.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role,prior.SIG))
                assert cur.fetchone() == (False,)
        cur.execute('RESET ROLE')
