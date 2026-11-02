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


def refused(store, connection, suffix, actual=1234):
    _, submission, miner = accepted_recovery_submission(store, suffix)
    token = new_lease_token()
    assert begin(store, submission, miner, token=token)['status'] == 'claimed'
    result = store.finish_submission_review(submission, miner, token, 'error', failure(), actual)
    assert result['status'] == 'error'
    _age_review_error(connection, submission)
    return submission, miner


@pytest.mark.parametrize('actual', [1234, None])
def test_primary_cost_and_one_new_identity_settle_without_replay(recovery_store, recovery_superuser, actual):
    submission, miner = refused(recovery_store, recovery_superuser, 'a599settle' + str(actual), actual)
    first = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner)['status'] == 'fallback_required'
    token = new_lease_token()
    alternate = begin(recovery_store, submission, miner, ALTERNATE, token)
    assert alternate['status'] == 'claimed' and alternate['attempt'] == 2
    assert alternate['call_identity'] != first[0]['call_identity']
    replay = begin(recovery_store, submission, miner, ALTERNATE, token)
    assert replay['idempotent'] is True
    assert replay['call_identity'] == alternate['call_identity']
    assert recovery_store.finish_submission_review(submission, miner, token, 'passed', verdict(), 2000)['status'] == 'passed'
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert [item['entry_kind'] for item in ledger] == [
        'reservation', 'dispatch', 'settlement' if actual is not None else 'uncertain',
        'reservation', 'dispatch', 'settlement',
    ]
    assert ledger[:3] == first
    doc = recovery_store.get_submission(submission)['code_review_doc']
    assert doc['model'] == ALTERNATE
    assert doc['review_cost_microusd'] == (actual if actual is not None else 50_000) + 2000
    assert recovery_store.finish_submission_review(submission, miner, token, 'passed', verdict(), 2000)['status'] == 'existing'
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'existing'
    assert recovery_store.list_ledger(submission_id=submission) == ledger


def test_concurrent_stale_workers_make_one_alternate_reservation(recovery_store, recovery_superuser):
    submission, miner = refused(recovery_store, recovery_superuser, 'a599race')
    # Both workers prepared the alternate from the same committed refusal row.
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: begin(recovery_store, submission, miner, ALTERNATE), range(2)))
    assert sorted(item['status'] for item in results) == ['busy', 'claimed']
    assert begin(recovery_store, submission, miner, PRIMARY)['status'] == 'busy'
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert len(ledger) == 5
    assert sum(item['entry_kind'] == 'reservation' and item['entry_doc'].get('review_model') == ALTERNATE for item in ledger) == 1


@pytest.mark.parametrize('error', [
    'code_review_provider_refused', 'review_response_invalid', 'code_review_transport_error',
    'code_review_provider_timeout', 'code_review_provider_rate_limited',
    'code_review_preparation_unavailable', 'code_review_cost_unavailable',
])
def test_every_alternate_failure_is_terminal_in_sql_and_public_retry_policy(recovery_store, recovery_superuser, error):
    suffix = 'a599fail' + str(sorted(code_review_policy.ERROR_CODES).index(error))
    submission, miner = refused(recovery_store, recovery_superuser, suffix)
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, ALTERNATE, token)['status'] == 'claimed'
    assert recovery_store.finish_submission_review(submission, miner, token, 'error', failure(ALTERNATE, error), None)['status'] == 'error'
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['retryable'] is False
    assert not code_review_policy.review_retry_available(row)
    _age_review_error(recovery_superuser, submission)
    before = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'exhausted'
    assert begin(recovery_store, submission, miner, PRIMARY)['status'] == 'exhausted'
    assert recovery_store.list_ledger(submission_id=submission) == before


def test_expired_alternate_is_uncertain_and_cannot_dispatch_again(recovery_store, recovery_superuser):
    submission, miner = refused(recovery_store, recovery_superuser, 'a599crash')
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, ALTERNATE, token)['status'] == 'claimed'
    with recovery_superuser.cursor() as cursor:
        cursor.execute("UPDATE public.lab_arena_submissions SET code_review_started_at=clock_timestamp()-interval '602 seconds',code_review_expires_at=clock_timestamp()-interval '1 second' WHERE submission_id=%s", (submission,))
    assert begin(recovery_store, submission, miner, PRIMARY)['status'] == 'recovered'
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['model'] == ALTERNATE
    assert row['code_review_doc']['retryable'] is False
    assert not code_review_policy.review_retry_available(row)
    assert code_review_policy.review_model_for(row) == ALTERNATE
    before = recovery_store.list_ledger(submission_id=submission)
    assert before[-1]['entry_kind'] == 'uncertain'
    assert before[-1]['amount_microusd'] == 50_000
    assert before[-1]['entry_doc']['retryable'] is False
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'exhausted'
    assert begin(recovery_store, submission, miner, PRIMARY)['status'] == 'exhausted'
    assert recovery_store.finish_submission_review(submission, miner, token, 'passed', verdict(), 1234)['status'] == 'stale'
    assert recovery_store.list_ledger(submission_id=submission) == before


@pytest.mark.parametrize('error', ['code_review_provider_request_rejected', 'code_review_provider_authentication', 'code_review_provider_credit', 'review_response_invalid'])
def test_generic_failures_cannot_select_alternate(recovery_store, recovery_superuser, error):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'a599generic' + str(sorted(code_review_policy.ERROR_CODES).index(error)))
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, token=token)['status'] == 'claimed'
    recovery_store.finish_submission_review(submission, miner, token, 'error', failure(PRIMARY, error, retryable=False), 0)
    _age_review_error(recovery_superuser, submission)
    before = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'fallback_not_authorized'
    assert recovery_store.list_ledger(submission_id=submission) == before


def test_runtime_uses_real_claim_and_ledger_for_refusal_then_complete_review(recovery_store, recovery_superuser):
    _, submission, _miner = accepted_recovery_submission(recovery_store, 'a599runtime')
    payload = _archive(_source())
    def response(parameters):
        body = _review_body(parameters)
        if parameters['model'] == PRIMARY:
            body['choices'][0]['finish_reason'] = 'content_filter'
        else:
            router_receipt(body, parameters)
        return {'body': body}
    transport = ReviewTransport(response)
    reviewer, _ = reviewer_for(payload, transport, recovery_store)
    row = {**recovery_store.get_submission(submission), 'source_size_bytes': len(payload)}
    assert reviewer.review(row)['status'] == 'error'
    assert len(transport.sent) == 1
    assert reviewer.review(recovery_store.get_submission(submission))['status'] == 'backoff'
    _age_review_error(recovery_superuser, submission)
    row = {**recovery_store.get_submission(submission), 'source_size_bytes': len(payload)}
    assert reviewer.review(row)['status'] == 'passed'
    assert len(transport.sent) == 2
    before = recovery_store.list_ledger(submission_id=submission)
    assert reviewer.review(recovery_store.get_submission(submission))['status'] == 'existing'
    assert len(transport.sent) == 2
    assert recovery_store.list_ledger(submission_id=submission) == before
    assert before[0]['entry_doc']['review_model'] == PRIMARY
    assert before[3]['entry_doc']['review_model'] == ALTERNATE
    assert before[-1]['entry_doc']['provider_diagnostic']['selected_provider'] == 'Azure'


def test_migration_reapply_preserves_rows_security_and_fails_closed_on_unknown_body(recovery_store, recovery_superuser):
    assert recovery_store.code_review_schema(require_refusal_fallback=True)['refusal_fallback']['max_alternate_claims'] == 1
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


def test_primary_transient_recovery_remains_deadline_bounded(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'a599primarytransient')
    for attempt in range(1, 8):
        token = new_lease_token()
        assert begin(recovery_store, submission, miner, token=token)['attempt'] == attempt
        doc = failure(PRIMARY, 'review_response_invalid', retryable=False)
        assert recovery_store.finish_submission_review(submission, miner, token, 'error', doc, 10)['status'] == 'error'
        _age_review_error(recovery_superuser, submission)
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, token=token)['attempt'] == 8
    recovery_store.finish_submission_review(submission, miner, token, 'error', failure(), 10)
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['attempt'] == 9


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
        assert store.code_review_schema(require_refusal_fallback=True)['refusal_fallback']['max_alternate_claims'] == 1
        assert store.finish_submission_review(submission, miner, token, 'error', failure(), 1234)['status'] == 'error'
        _age_review_error(connection, submission)
        assert begin(store, submission, miner, ALTERNATE)['status'] == 'claimed'
        assert store.list_ledger(submission_id=submission)[:2] == before
        connection.close()
    finally:
        if store:
            store.close()
        database.close()
