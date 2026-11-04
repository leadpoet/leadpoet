"""Real PostgreSQL authority for one full-source alternate admission review."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from lab_arena import code_review, code_review_policy
from lab_arena.store import ArenaStoreError, new_lease_token
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    _age_review_error, recovery_connect,
    recovery_store, recovery_superuser,
)
from tests.lab_arena.refusal_review_fallback_test import (
    reviewer_for, router_receipt,
)
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    ReviewTransport, _archive, _review_body, _source,
)
from tests.postgres_migration_harness import SCRIPTS
from tests.lab_arena.test_lab_arena_migration_postgres import (
    encrypted_runtime_credentials, hotkey, round_config, source_submission_doc,
)

MIGRATION = SCRIPTS / '599-lab-arena-refusal-review-fallback.sql'
PRIMARY = code_review.DEFAULT_REVIEW_MODEL
ALTERNATE = code_review.FALLBACK_REVIEW_MODEL
SIGNATURES = (
    'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)',
    'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)',
    'public.lab_arena_code_review_schema_v1()',
)


@pytest.fixture(scope='module')
def recovery_database():
    database = current_service_database()
    try:
        psycopg2, dsn = next(database)
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute((SCRIPTS / '594-lab-arena-private-review-diagnostics.sql').read_text())
            cursor.execute(MIGRATION.read_text())
        yield psycopg2, dsn
    finally:
        database.close()


def begin(store, submission, miner, model=PRIMARY, token=None):
    return store.begin_submission_review(submission, miner, token or new_lease_token(), 50_000, model, 2, 1000)


def accepted_recovery_submission(store, suffix):
    now = datetime.now(timezone.utc)
    round_id = 'arena-2099-03-01-' + hashlib.sha256(suffix.encode()).hexdigest()[:12]
    submission = suffix + '-recovery'
    miner = hotkey('refusal599-' + suffix)
    config = round_config(round_id, [hotkey('runner599-' + suffix)])
    config['schedule'].update({
        'submission_open': (now-timedelta(hours=2)).isoformat(),
        'submission_cutoff': (now+timedelta(minutes=30)).isoformat(),
        'benchmark_deadline': (now+timedelta(hours=1)).isoformat(),
    })
    assert store.create_round(round_id, config)['status'] == 'created'
    assert store.register_submission(round_id, submission, miner, source_submission_doc(round_id, submission))['status'] == 'registered'
    assert store.accept_submission_with_credentials(round_id, submission, miner, encrypted_runtime_credentials(submission))['status'] == 'ok'
    return round_id, submission, miner


def failure(model=PRIMARY, error='code_review_provider_refused', retryable=True):
    return {'model': model, 'file_count': 2, 'source_bytes': 1000,
            'error_code': error, 'retryable': retryable}


def verdict(model=ALTERNATE):
    return {'model': model, 'file_count': 2, 'source_bytes': 1000,
            'passed': True, 'verdict': 'pass', 'categories': []}


@pytest.mark.parametrize('code', sorted(code_review_policy.ERROR_CODES))
def test_current_error_codes_persist_and_clear_lease(recovery_store, recovery_superuser, code):
    if code == 'code_review_research_unverified':
        from tests.lab_arena.research_review_result566_postgres_test import _claim, _document
        submission, miner, token = _claim(recovery_store, recovery_superuser, 'a599research01')
        document = _document('unverified', status='error')
    else:
        _, submission, miner = accepted_recovery_submission(recovery_store, 'error599-' + code)
        token = new_lease_token()
        assert begin(recovery_store, submission, miner, token=token)['status'] == 'claimed'
        document = {**code_review_policy.diagnostic(code), 'model': PRIMARY,
                    'file_count': 2, 'source_bytes': 1000}
    assert recovery_store.finish_submission_review(
        submission, miner, token, 'error', document, 0,
    )['status'] == 'error'
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['error_code'] == code
    assert row['code_review_expires_at'] is None
    assert row['code_review_doc']['cost_microusd'] == 0


def refused(store, connection, suffix, actual=1234):
    _, submission, miner = accepted_recovery_submission(store, suffix)
    token = new_lease_token()
    assert begin(store, submission, miner, token=token)['status'] == 'claimed'
    result = store.finish_submission_review(submission, miner, token, 'error', failure(), actual)
    assert result['status'] == 'error'
    _age_review_error(connection, submission)
    return submission, miner



def finish_error(store, submission, miner, token, error='code_review_provider_refused', model=PRIMARY, actual=1234):
    return store.finish_submission_review(submission, miner, token, 'error', failure(model, error), actual)


def next_error(store, connection, submission, miner, model, error='code_review_provider_refused', actual=1234):
    _age_review_error(connection, submission)
    token = new_lease_token()
    claimed = begin(store, submission, miner, model, token)
    assert claimed['status'] == 'claimed'
    before = store.list_ledger(submission_id=submission)
    replay = begin(store, submission, miner, model, token)
    assert replay['idempotent'] is True
    assert store.list_ledger(submission_id=submission) == before
    assert finish_error(store, submission, miner, token, error, model, actual)['status'] == 'error'
    return token


def test_two_primary_two_alternate_chain_with_mixed_failures(recovery_store, recovery_superuser):
    submission, miner = refused(recovery_store, recovery_superuser, 'sequence4')
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'route_required'
    next_error(recovery_store, recovery_superuser, submission, miner, PRIMARY, 'code_review_transport_error', None)
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner, PRIMARY)['review_model'] == ALTERNATE
    next_error(recovery_store, recovery_superuser, submission, miner, ALTERNATE, 'code_review_provider_timeout')
    assert code_review_policy.review_retry_available(recovery_store.get_submission(submission))
    next_error(recovery_store, recovery_superuser, submission, miner, ALTERNATE)
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['review_primary_attempts'] == 2
    assert row['code_review_doc']['review_alternate_attempts'] == 2
    assert row['code_review_doc']['retryable'] is False
    assert not code_review_policy.review_retry_available(row)
    before = recovery_store.list_ledger(submission_id=submission)
    assert len(before) == 12
    assert [e['entry_doc']['review_model'] for e in before if e['entry_kind']=='reservation'] == [PRIMARY,PRIMARY,ALTERNATE,ALTERNATE]
    assert [e['amount_microusd'] for e in before if e['entry_kind'] in ('settlement','uncertain')] == [1234,50000,1234,1234]
    for model in (PRIMARY,ALTERNATE):
        assert begin(recovery_store, submission, miner, model)['status'] == 'exhausted'
    assert recovery_store.list_ledger(submission_id=submission) == before


@pytest.mark.parametrize('result', ['passed','rejected'])
def test_valid_primary_retry_never_calls_alternate(recovery_store,recovery_superuser,result):
    submission,miner=refused(recovery_store,recovery_superuser,'valid'+result)
    _age_review_error(recovery_superuser,submission)
    token=new_lease_token()
    assert begin(recovery_store,submission,miner,PRIMARY,token)['status']=='claimed'
    doc=verdict(PRIMARY)
    if result=='rejected': doc.update(passed=False,verdict='reject')
    assert recovery_store.finish_submission_review(submission,miner,token,result,doc,100)['status']==result
    before=recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store,submission,miner,ALTERNATE)['status']=='existing'
    assert recovery_store.list_ledger(submission_id=submission)==before


@pytest.mark.parametrize('crash_position',[2,3,4])
def test_each_chain_crash_consumes_attempt_and_keeps_route(recovery_store,recovery_superuser,crash_position):
    submission,miner=refused(recovery_store,recovery_superuser,'crash'+str(crash_position))
    for position in range(2,crash_position):
        next_error(recovery_store,recovery_superuser,submission,miner,PRIMARY if position==2 else ALTERNATE)
    _age_review_error(recovery_superuser,submission)
    token=new_lease_token();model=PRIMARY if crash_position==2 else ALTERNATE
    assert begin(recovery_store,submission,miner,model,token)['status']=='claimed'
    with recovery_superuser.cursor() as c:
        c.execute("UPDATE public.lab_arena_submissions SET code_review_started_at=clock_timestamp()-interval '602 seconds',code_review_expires_at=clock_timestamp()-interval '1 second' WHERE submission_id=%s",(submission,))
    assert begin(recovery_store,submission,miner,model)['status']=='recovered'
    row=recovery_store.get_submission(submission)
    assert row['code_review_doc']['refusal_recovery'] is True
    assert row['code_review_doc']['review_alternate_attempts']==max(0,crash_position-2)
    assert row['code_review_doc']['retryable'] is (crash_position<4)
    assert code_review_policy.review_retry_available(row) is (crash_position<4)
    assert code_review_policy.review_model_for(row)==ALTERNATE
    before=recovery_store.list_ledger(submission_id=submission)
    assert before[-1]['entry_kind']=='uncertain'
    _age_review_error(recovery_superuser,submission)
    assert begin(recovery_store,submission,miner,ALTERNATE)['status']==('claimed' if crash_position<4 else 'exhausted')


DIAGNOSTIC={'finish_reason':'content_filter','native_finish_reason':'refusal','max_output_tokens':16384,
    'context_integrity':'verified','response_model':PRIMARY,
    'selected_model':'anthropic/claude-sonnet-5.5-20260928','selected_provider':'Google'}


@pytest.mark.parametrize('damage',[None,'untyped','no_diagnostic','http403','wrong_model','older','closed','deadline'])
def test_legacy_recovery_requires_latest_bound_authoritative_pair(recovery_store,recovery_superuser,damage):
    _,submission,miner=accepted_recovery_submission(recovery_store,'legacy'+str(damage))
    seed_model='anthropic/claude-sonnet-5' if damage=='wrong_model' else PRIMARY
    token=new_lease_token();assert begin(recovery_store,submission,miner,seed_model,token)['status']=='claimed'
    diagnostic=dict(DIAGNOSTIC)
    if damage=='untyped': diagnostic.update(finish_reason='stop',native_finish_reason='end_turn')
    if damage=='wrong_model': diagnostic['response_model']='anthropic/claude-sonnet-5'
    doc={**failure(seed_model,'code_review_provider_request_rejected',False),'provider_http_status':403 if damage=='http403' else 200,'provider_diagnostic':diagnostic}
    if damage=='no_diagnostic':doc.pop('provider_diagnostic')
    recovery_store.finish_submission_review(submission,miner,token,'error',doc,1226326)
    if damage=='older':
        # A real newer generic error must not inherit older refusal proof.
        _age_review_error(recovery_superuser,submission)
        token=new_lease_token();assert begin(recovery_store,submission,miner,PRIMARY,token)['status']=='claimed'
        recovery_store.finish_submission_review(submission,miner,token,'error',{**failure(PRIMARY,'code_review_provider_request_rejected',False),'provider_http_status':400},0)
    if damage in ('closed','deadline'):
        with recovery_superuser.cursor() as c:
            c.execute("SET session_replication_role='replica'")
            try:
                if damage=='closed':c.execute("UPDATE public.lab_arena_rounds SET status='committed' WHERE round_id=(SELECT round_id FROM public.lab_arena_submissions WHERE submission_id=%s)",(submission,))
                else:c.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,benchmark_deadline}',to_jsonb((clock_timestamp()-interval '1 second')::text)) WHERE round_id=(SELECT round_id FROM public.lab_arena_submissions WHERE submission_id=%s)",(submission,))
            finally:c.execute("SET session_replication_role='origin'")
    _age_review_error(recovery_superuser,submission)
    row=recovery_store.get_submission(submission)
    assert code_review_policy.review_retry_available(row) is (damage in (None,'untyped','no_diagnostic'))
    listed=recovery_store.list_submissions(row['round_id'],status='accepted')[0]
    assert code_review_policy.review_retry_available(listed) is (damage in (None,'untyped','no_diagnostic'))
    before=recovery_store.list_ledger(submission_id=submission)
    result=begin(recovery_store,submission,miner,PRIMARY)
    assert result['status']==({None:'claimed','untyped':'claimed','no_diagnostic':'claimed','closed':'round_closed','deadline':'deadline'}.get(damage,'exhausted'))
    if damage not in (None,'untyped','no_diagnostic'): assert recovery_store.list_ledger(submission_id=submission)==before


@pytest.mark.parametrize('failure_code',['source_archive_invalid','code_review_provider_authentication','code_review_provider_credit'])
def test_genuine_terminal_failures_stop_chain(recovery_store,recovery_superuser,failure_code):
    submission,miner=refused(recovery_store,recovery_superuser,'terminal'+str(sorted(code_review_policy.ERROR_CODES).index(failure_code)))
    _age_review_error(recovery_superuser,submission)
    token=new_lease_token();assert begin(recovery_store,submission,miner,PRIMARY,token)['status']=='claimed'
    recovery_store.finish_submission_review(submission,miner,token,'error',failure(PRIMARY,failure_code,False),0)
    row=recovery_store.get_submission(submission)
    assert not code_review_policy.review_retry_available(row)
    assert begin(recovery_store,submission,miner,PRIMARY)['status']=='exhausted'
    assert begin(recovery_store,submission,miner,ALTERNATE)['status']=='fallback_not_authorized'


def test_runtime_sequence_and_other_rows_remain_unchanged(recovery_store,recovery_superuser):
    _,other,_=accepted_recovery_submission(recovery_store,'other-control')
    original=recovery_store.get_submission(other)
    _,submission,_=accepted_recovery_submission(recovery_store,'runtime-sequence')
    payload=_archive(_source())
    def respond(parameters):
        body=_review_body(parameters)
        if parameters['model']==PRIMARY:body['choices'][0]['finish_reason']='content_filter'
        else:router_receipt(body,parameters)
        return {'body':body}
    transport=ReviewTransport(respond);reviewer,_=reviewer_for(payload,transport,recovery_store)
    for expected in ('error','error','passed'):
        row={**recovery_store.get_submission(submission),'source_size_bytes':len(payload)}
        assert reviewer.review(row)['status']==expected
        if expected=='error':_age_review_error(recovery_superuser,submission)
    assert [__import__('json').loads(r['body'])['model'] for r in transport.sent]==[PRIMARY,PRIMARY,ALTERNATE]
    before=recovery_store.list_ledger(submission_id=submission)
    assert reviewer.review(recovery_store.get_submission(submission))['status']=='existing'
    assert len(transport.sent)==3
    assert recovery_store.get_submission(other)==original
    assert recovery_store.list_ledger(submission_id=submission)==before


def test_current_capability_and_reapply_preserve_rows(recovery_store,recovery_superuser):
    marker=recovery_store.code_review_schema(require_refusal_fallback=True)['refusal_fallback']
    assert marker['max_primary_refusal_claims']==marker['max_alternate_claims']==2
    with recovery_superuser.cursor() as c:
        c.execute('SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s');before=c.fetchone()[0]
        c.execute(MIGRATION.read_text())
        c.execute('SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s');assert c.fetchone()[0]==before


def test_migration_reapply_preserves_rows_security_and_fails_closed_on_unknown_body(recovery_store, recovery_superuser):
    assert recovery_store.code_review_schema(require_refusal_fallback=True)['refusal_fallback']['max_alternate_claims'] == 2
    with recovery_superuser.cursor() as cursor:
        cursor.execute('SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s')
        rows_before = cursor.fetchone()[0]
        cursor.execute('SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')
        ledger_before = cursor.fetchone()[0]
        before = []
        for sig in SIGNATURES:
            cursor.execute('SELECT to_jsonb(p) FROM pg_proc p WHERE oid=%s::regprocedure', (sig,))
            before.append(cursor.fetchone()[0])
        cursor.execute(MIGRATION.read_text())
        for sig, expected in zip(SIGNATURES, before):
            cursor.execute('SELECT to_jsonb(p) FROM pg_proc p WHERE oid=%s::regprocedure', (sig,))
            assert cursor.fetchone()[0] == expected
            cursor.execute('SELECT has_function_privilege(\'lab_arena_service\',%s,\'EXECUTE\'),has_function_privilege(\'service_role\',%s,\'EXECUTE\'),has_function_privilege(\'anon\',%s,\'EXECUTE\'),has_function_privilege(\'authenticated\',%s,\'EXECUTE\')', (sig,)*4)
            assert cursor.fetchone() == (True, False, False, False)
        cursor.execute('SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s')
        assert cursor.fetchone()[0] == rows_before
        cursor.execute('SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')
        assert cursor.fetchone()[0] == ledger_before
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[0],))
        original = cursor.fetchone()[0]
        changed = original.replace('BEGIN\n', 'BEGIN\n  -- unknown body\n', 1)
        try:
            cursor.execute(changed)
            with pytest.raises(Exception, match='Arena refusal599 preimage differs'):
                cursor.execute(MIGRATION.read_text())
            cursor.execute('ROLLBACK')
        finally:
            cursor.execute('ROLLBACK')
            cursor.execute(original)



def test_exact_upgrade_enables_refusal_without_replaying_or_changing_existing_rows():
    from lab_arena.store import ArenaStore, PsycopgTransport
    database = current_service_database()
    store = None
    try:
        psycopg2, dsn = next(database)
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
        with connection.cursor() as cursor:
            cursor.execute((SCRIPTS / '594-lab-arena-private-review-diagnostics.sql').read_text())
        with pytest.raises(ArenaStoreError, match='fallback schema mismatch'):
            store.code_review_schema(require_refusal_fallback=True)
        _, submission, miner = accepted_recovery_submission(store, 'a599upgrade')
        token = new_lease_token()
        assert begin(store, submission, miner, token=token)['status'] == 'claimed'
        with pytest.raises(ArenaStoreError, match='code_review_input_invalid'):
            store.finish_submission_review(submission, miner, token, 'error', failure(), 1234)
        assert store.get_submission(submission)['code_review_status'] == 'reviewing'
        before = store.list_ledger(submission_id=submission)
        assert len(before) == 2
        with connection.cursor() as cursor:
            cursor.execute("CREATE ROLE refusal599_migrator NOLOGIN NOSUPERUSER INHERIT")
            cursor.execute("GRANT lab_arena_owner TO refusal599_migrator")
            cursor.execute("ALTER SCHEMA public OWNER TO refusal599_migrator")
            cursor.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC,lab_arena_owner")
            cursor.execute("SET ROLE refusal599_migrator")
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            acl = cursor.fetchone()[0]
            failed = MIGRATION.read_text().replace('  EXECUTE v_updated;', '  EXECUTE v_updated;\n  PERFORM 1/0;', 1)
            with pytest.raises(psycopg2.errors.DivisionByZero):
                cursor.execute(failed)
            cursor.execute('ROLLBACK')
            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
            cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            assert cursor.fetchone()[0] == acl
            cursor.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")
            assert cursor.fetchone()[0] is False
        assert store.list_ledger(submission_id=submission) == before
        assert store.code_review_schema(require_refusal_fallback=True)['refusal_fallback']['max_alternate_claims'] == 2
        assert store.finish_submission_review(submission, miner, token, 'error', failure(), 1234)['status'] == 'error'
        _age_review_error(connection, submission)
        assert begin(store, submission, miner, PRIMARY)['status'] == 'claimed'
        assert store.list_ledger(submission_id=submission)[:2] == before
        connection.close()
    finally:
        if store:
            store.close()
        database.close()


def test_primary_allowance_starts_at_first_refusal_after_seven_transients(recovery_store,recovery_superuser):
    _,submission,miner=accepted_recovery_submission(recovery_store,'episode-after7')
    for attempt in range(1,8):
        token=new_lease_token();assert begin(recovery_store,submission,miner,PRIMARY,token)['attempt']==attempt
        finish_error(recovery_store,submission,miner,token,'review_response_invalid')
        _age_review_error(recovery_superuser,submission)
    token=new_lease_token();assert begin(recovery_store,submission,miner,PRIMARY,token)['attempt']==8
    finish_error(recovery_store,submission,miner,token)
    row=recovery_store.get_submission(submission)
    assert row['code_review_doc']['refusal_episode_start']==8
    assert row['code_review_doc']['review_primary_attempts']==1
    assert code_review_policy.review_model_for(row)==PRIMARY
    next_error(recovery_store,recovery_superuser,submission,miner,PRIMARY,'code_review_transport_error',None)
    row=recovery_store.get_submission(submission)
    assert row['code_review_doc']['refusal_episode_start']==8
    assert row['code_review_doc']['review_primary_attempts']==2
    assert code_review_policy.review_model_for(row)==ALTERNATE


@pytest.mark.parametrize('unknown_at',[1,2,3])
def test_valid_unknown_cost_clears_lease_and_stops_recovery(recovery_store,recovery_superuser,unknown_at):
    _,submission,_=accepted_recovery_submission(recovery_store,'unknown-cost'+str(unknown_at))
    payload=_archive(_source())
    def respond(parameters):
        body=_review_body(parameters,cost=None if len(transport.sent)==unknown_at else '0.001234')
        if len(transport.sent)<unknown_at:body['choices'][0]['finish_reason']='content_filter'
        elif parameters['model']==ALTERNATE:router_receipt(body,parameters)
        return {'body':body}
    transport=ReviewTransport(respond);reviewer,_=reviewer_for(payload,transport,recovery_store)
    for _ in range(unknown_at):
        row={**recovery_store.get_submission(submission),'source_size_bytes':len(payload)}
        assert reviewer.review(row)['status']=='error'
        _age_review_error(recovery_superuser,submission)
    row=recovery_store.get_submission(submission)
    assert row['code_review_status']=='error'
    assert row['code_review_expires_at'] is None
    assert row['code_review_doc']['error_code']=='code_review_cost_unavailable'
    assert row['code_review_doc']['retryable'] is False
    before=recovery_store.list_ledger(submission_id=submission)
    assert before[-1]['entry_kind']=='uncertain'
    assert reviewer.review(row)['status']=='exhausted'
    assert len(transport.sent)==unknown_at
    assert recovery_store.list_ledger(submission_id=submission)==before


@pytest.mark.parametrize('gate', ['guard', 'deadline', 'closed', 'freeze'])
def test_alternate_retains_existing_admission_gates(recovery_store, recovery_superuser, gate):
    from tests.lab_arena.atomic_review_restart_guard587_postgres_test import acquire
    submission, miner = refused(recovery_store, recovery_superuser, 'a599gate' + gate)
    before = recovery_store.list_ledger(submission_id=submission)
    try:
        if gate == 'guard':
            acquire(recovery_superuser)
        else:
            with recovery_superuser.cursor() as cursor:
                cursor.execute("SET session_replication_role='replica'")
                try:
                    if gate == 'deadline':
                        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,benchmark_deadline}',to_jsonb((clock_timestamp()-interval '1 second')::text)) WHERE round_id=(SELECT round_id FROM public.lab_arena_submissions WHERE submission_id=%s)", (submission,))
                    elif gate == 'freeze':
                        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(jsonb_set(configuration_doc,'{schedule,submission_cutoff}',to_jsonb((clock_timestamp()+interval '2 hours')::text)),'{schedule,benchmark_deadline}',to_jsonb((clock_timestamp()+interval '3 hours')::text)) WHERE round_id=(SELECT round_id FROM public.lab_arena_submissions WHERE submission_id=%s)", (submission,))
                    else:
                        cursor.execute("UPDATE public.lab_arena_rounds SET status='committed' WHERE round_id=(SELECT round_id FROM public.lab_arena_submissions WHERE submission_id=%s)", (submission,))
                finally:
                    cursor.execute("SET session_replication_role='origin'")
        expected = {'guard': 'paused', 'deadline': 'deadline', 'closed': 'round_closed', 'freeze': 'deferred'}[gate]
        assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == expected
        payload = _archive(_source())
        transport = ReviewTransport(lambda _parameters: pytest.fail('admission gate must prevent provider dispatch'))
        reviewer, _ = reviewer_for(payload, transport, recovery_store)
        row = {**recovery_store.get_submission(submission), 'source_size_bytes': len(payload)}
        assert reviewer.review(row)['status'] == expected
        assert transport.sent == []
        assert recovery_store.list_ledger(submission_id=submission) == before
    finally:
        if gate == 'guard':
            with recovery_superuser.cursor() as cursor:
                cursor.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'::jsonb WHERE singleton")


def test_concurrent_workers_share_last_alternate_slot(recovery_store,recovery_superuser):
    submission,miner=refused(recovery_store,recovery_superuser,'last-alternate-race')
    next_error(recovery_store,recovery_superuser,submission,miner,PRIMARY)
    next_error(recovery_store,recovery_superuser,submission,miner,ALTERNATE)
    _age_review_error(recovery_superuser,submission)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:begin(recovery_store,submission,miner,ALTERNATE),range(2)))
    assert sorted(result['status'] for result in results)==['busy','claimed']
    ledger=recovery_store.list_ledger(submission_id=submission)
    assert sum(entry['entry_kind']=='reservation' and entry['entry_doc'].get('review_model')==ALTERNATE for entry in ledger)==2
