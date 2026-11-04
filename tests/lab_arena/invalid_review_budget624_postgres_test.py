"""Real ledger/lease proof for finite unusable-review reservations."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from lab_arena import code_review_policy
from lab_arena.store import ArenaStoreError, new_lease_token
from tests.lab_arena.refusal_review_fallback599_postgres_test import (
    PRIMARY, ALTERNATE, SIGNATURES, accepted_recovery_submission, begin, failure,
    recovery_database as predecessor_database,
)
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    _age_review_error, recovery_connect, recovery_store, recovery_superuser,
)
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '624-lab-arena-invalid-review-reservation-budget.sql'


@pytest.fixture(scope='module')
def recovery_database():
    database = predecessor_database.__wrapped__()
    try:
        psycopg2, dsn = next(database)
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
        yield psycopg2, dsn
    finally:
        database.close()


def finish(store, submission, miner, token, error='review_response_invalid', *, model=PRIMARY, actual=1234, **extra):
    return store.finish_submission_review(submission, miner, token, 'error',
        {**failure(model, error), **extra}, actual)


def claim_error(store, connection, submission, miner, error='review_response_invalid', *, model=PRIMARY, **extra):
    if store.get_submission(submission)['code_review_status'] == 'error':
        _age_review_error(connection, submission)
    token = new_lease_token()
    assert begin(store, submission, miner, model, token)['status'] == 'claimed'
    assert finish(store, submission, miner, token, error, model=model, **extra)['status'] == 'error'


def test_bound_fails_before_migration_then_same_claim_settles_and_exhausts(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'invalid624-beforeafter')
    # Reinstall the exact predecessor only in this disposable database.
    with recovery_superuser.cursor() as cursor:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[1],))
        fixed = cursor.fetchone()[0]
        predecessor = fixed.replace("'not_finished', 'truncated', 'truncated_at_limit',", "'not_finished',")
        cursor.execute(predecessor)
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, token=token)['status'] == 'claimed'
    with pytest.raises(ArenaStoreError, match='lab_arena_code_review_input_invalid'):
        finish(recovery_store, submission, miner, token, error_reason='truncated')
    assert recovery_store.get_submission(submission)['code_review_status'] == 'reviewing'
    with recovery_superuser.cursor() as cursor:
        cursor.execute(fixed)
    assert finish(recovery_store, submission, miner, token, error_reason='truncated')['status'] == 'error'
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['error_reason'] == 'truncated'
    assert row['code_review_expires_at'] is None
    claim_error(recovery_store, recovery_superuser, submission, miner, 'code_review_transport_error', actual=None)
    claim_error(recovery_store, recovery_superuser, submission, miner)
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['review_invalid_attempts'] == 3
    assert row['code_review_doc']['invalid_episode_start'] == 1
    assert row['code_review_doc']['retryable'] is False
    assert not code_review_policy.review_retry_available(row)
    before = recovery_store.list_ledger(submission_id=submission)
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner)['status'] == 'exhausted'
    assert recovery_store.list_ledger(submission_id=submission) == before
    assert [entry['amount_microusd'] for entry in before if entry['entry_kind'] in {'settlement', 'uncertain'}] == [1234, 50000, 1234]


def test_budget_counter_fails_on_exact_predecessor_then_migration_bounds_it(recovery_connect):
    # Separate fixture proves the cap itself changes behavior; canonical624 is
    # still tested against exact599 hashes rather than a fabricated preimage.
    database = predecessor_database.__wrapped__()
    try:
        psycopg2, dsn = next(database)
        from lab_arena.store import ArenaStore, PsycopgTransport
        store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
        connection = psycopg2.connect(**dsn); connection.autocommit = True
        _, submission, miner = accepted_recovery_submission(store, 'invalid624-cap-before')
        for _ in range(3):
            claim_error(store, connection, submission, miner)
        _age_review_error(connection, submission)
        token = new_lease_token()
        assert begin(store, submission, miner, token=token)['status'] == 'claimed'
        # Old599 allows a fourth paid reservation. Its existing lease settles
        # before applying624; all prior ledger records must remain unchanged.
        assert finish(store, submission, miner, token)['status'] == 'error'
        before = store.list_ledger(submission_id=submission)
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
        assert store.list_ledger(submission_id=submission) == before
        # Legacy unfinished state starts at the latest bound failure, not a
        # retroactive global count. Two additional reservations are available.
        claim_error(store, connection, submission, miner)
        claim_error(store, connection, submission, miner)
        _age_review_error(connection, submission)
        assert begin(store, submission, miner)['status'] == 'exhausted'
        assert store.get_submission(submission)['code_review_doc']['invalid_episode_start'] == 4
        store.close(); connection.close()
    finally:
        database.close()


def test_prior_transients_do_not_consume_invalid_episode(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'invalid624-after-seven')
    for _ in range(7):
        claim_error(recovery_store, recovery_superuser, submission, miner, 'code_review_transport_error')
    claim_error(recovery_store, recovery_superuser, submission, miner)
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['review_invalid_attempts'] == 1
    assert row['code_review_doc']['invalid_episode_start'] == 8
    assert code_review_policy.review_retry_available(row)
    claim_error(recovery_store, recovery_superuser, submission, miner)
    claim_error(recovery_store, recovery_superuser, submission, miner)
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner)['status'] == 'exhausted'


@pytest.mark.parametrize('position', [2, 3])
def test_expired_reservation_consumes_budget_and_always_clears_lease(recovery_store, recovery_superuser, position):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'invalid624-expiry' + str(position))
    claim_error(recovery_store, recovery_superuser, submission, miner)
    if position == 3:
        claim_error(recovery_store, recovery_superuser, submission, miner, 'code_review_transport_error')
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner)['status'] == 'claimed'
    with recovery_superuser.cursor() as cursor:
        cursor.execute("UPDATE public.lab_arena_submissions SET code_review_started_at=clock_timestamp()-interval '602 seconds',code_review_expires_at=clock_timestamp()-interval '1 second' WHERE submission_id=%s", (submission,))
    assert begin(recovery_store, submission, miner)['status'] == 'recovered'
    row = recovery_store.get_submission(submission)
    assert row['code_review_expires_at'] is None
    assert row['code_review_doc']['review_invalid_attempts'] == position
    assert row['code_review_doc']['retryable'] is (position < 3)
    assert code_review_policy.review_retry_available(row) is (position < 3)
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner)['status'] == ('claimed' if position < 3 else 'exhausted')


def test_concurrent_final_reservation_is_single_and_replay_is_read_only(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'invalid624-concurrent')
    claim_error(recovery_store, recovery_superuser, submission, miner)
    claim_error(recovery_store, recovery_superuser, submission, miner)
    _age_review_error(recovery_superuser, submission)
    tokens = [new_lease_token() for _ in range(6)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda token: begin(recovery_store, submission, miner, token=token), tokens))
    assert sorted(result['status'] for result in results) == ['busy'] * 5 + ['claimed']
    winner = tokens[next(index for index, result in enumerate(results) if result['status'] == 'claimed')]
    before = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner, token=winner)['idempotent'] is True
    assert recovery_store.list_ledger(submission_id=submission) == before
    assert finish(recovery_store, submission, miner, winner)['status'] == 'error'
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner)['status'] == 'exhausted'


def test_error_classes_and_refusal_cannot_reset_budget(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'invalid624-no-reset')
    claim_error(recovery_store, recovery_superuser, submission, miner)
    claim_error(recovery_store, recovery_superuser, submission, miner, 'code_review_provider_refused')
    claim_error(recovery_store, recovery_superuser, submission, miner, 'code_review_transport_error')
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['invalid_episode_start'] == 1
    assert row['code_review_doc']['review_invalid_attempts'] == 3
    assert row['code_review_doc']['refusal_episode_start'] == 2
    assert row['code_review_doc']['review_primary_attempts'] == 2
    assert not code_review_policy.review_retry_available(row)
    _age_review_error(recovery_superuser, submission)
    for model in (PRIMARY, ALTERNATE):
        assert begin(recovery_store, submission, miner, model)['status'] == 'exhausted'


def test_diagnostics_and_input_counters_have_no_retry_authority(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'invalid624-no-input-authority')
    for _ in range(3):
        claim_error(recovery_store, recovery_superuser, submission, miner,
            review_invalid_attempts=0, invalid_episode_start=999,
            provider_diagnostic={'finish_reason': 'stop', 'max_output_tokens': 32768,
                                 'native_finish_reason': 'stop', 'context_integrity': 'verified'})
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['review_invalid_attempts'] == 3
    assert row['code_review_doc']['invalid_episode_start'] == 1
    assert not code_review_policy.review_retry_available(row)


def test_migration_reapply_is_noop_and_preserves_function_security(recovery_superuser):
    def read():
        with recovery_superuser.cursor() as cursor:
            cursor.execute("SELECT p.oid,pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=ANY(%s::regprocedure[])", (list(SIGNATURES),))
            return cursor.fetchall()
    before = read()
    with recovery_superuser.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    assert read() == before


def test_adaptive_output_recommendation_survives_transport_and_expiry(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'invalid624-capacity-expiry')
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, token=token)['review_output_token_recommendation'] == 32768
    assert finish(recovery_store, submission, miner, token, error_reason='truncated')['status'] == 'error'
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['review_output_token_recommendation'] == 65536
    _age_review_error(recovery_superuser, submission)
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, token=token)['review_output_token_recommendation'] == 65536
    with recovery_superuser.cursor() as cursor:
        cursor.execute("UPDATE public.lab_arena_submissions SET code_review_started_at=clock_timestamp()-interval '602 seconds',code_review_expires_at=clock_timestamp()-interval '1 second' WHERE submission_id=%s", (submission,))
    assert begin(recovery_store, submission, miner)['status'] == 'recovered'
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['review_output_token_recommendation'] == 65536
    assert row['code_review_doc']['error_code'] == 'code_review_transport_error'
    _age_review_error(recovery_superuser, submission)
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, token=token)['review_output_token_recommendation'] == 65536
    assert finish(recovery_store, submission, miner, token, error_reason='truncated_at_limit')['status'] == 'error'
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['error_reason'] == 'truncated_at_limit'
    assert row['code_review_doc']['retryable'] is False
    assert row['code_review_expires_at'] is None


def test_repeated_truncation_stops_at_second_claim_with_ledger_accounting(recovery_store, recovery_superuser):
    from tests.lab_arena.refusal_review_fallback_test import reviewer_for
    from tests.lab_arena.test_lab_arena_code_review_runtime import ReviewTransport, _archive, _review_body, _source
    import json
    _, submission, miner = accepted_recovery_submission(recovery_store, 'invalid624-runtime-truncated')
    payload = _archive(_source())
    def respond(parameters):
        body = _review_body(parameters)
        body['choices'][0]['finish_reason'] = 'length'
        return {'body': body}
    transport = ReviewTransport(respond)
    reviewer, _ = reviewer_for(payload, transport, recovery_store)
    row = {**recovery_store.get_submission(submission), 'source_size_bytes': len(payload)}
    assert reviewer.review(row)['status'] == 'error'
    _age_review_error(recovery_superuser, submission)
    row = {**recovery_store.get_submission(submission), 'source_size_bytes': len(payload)}
    assert reviewer.review(row)['status'] == 'error'
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['error_reason'] == 'truncated_at_limit'
    assert row['code_review_doc']['retryable'] is False
    assert row['code_review_expires_at'] is None
    assert reviewer.review(row)['status'] == 'exhausted'
    requests = [json.loads(call['body']) for call in transport.sent]
    assert [request['max_tokens'] for request in requests] == [32768, 65536]
    assert requests[0]['messages'] == requests[1]['messages']
    assert requests[0]['provider'] == requests[1]['provider']
    assert requests[0]['response_format'] == requests[1]['response_format']
    ledger = recovery_store.list_ledger(submission_id=submission)
    reservations = [entry for entry in ledger if entry['entry_kind'] == 'reservation']
    assert reservations[1]['amount_microusd'] > reservations[0]['amount_microusd']
    assert [entry['amount_microusd'] for entry in ledger if entry['entry_kind'] == 'settlement'] == [1234, 1234]
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner)['status'] == 'exhausted'


def test_wrong_preimage_rolls_back_without_changing_other_rpcs(recovery_superuser):
    with recovery_superuser.cursor() as cursor:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[0],))
        original = cursor.fetchone()[0]
        modified = original.replace('DECLARE\n', 'DECLARE\n  -- unrelated concurrent change\n', 1)
        cursor.execute(modified)
        try:
            with pytest.raises(Exception, match='preimage differs'):
                cursor.execute(MIGRATION.read_text())
            cursor.execute("ROLLBACK")
            cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[0],))
            assert cursor.fetchone()[0] == modified
        finally:
            cursor.execute(original)


def test_refusal_only_still_has_two_primary_and_two_alternate(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'invalid624-refusal-four')
    for model in (PRIMARY, PRIMARY, ALTERNATE, ALTERNATE):
        claim_error(recovery_store, recovery_superuser, submission, miner, 'code_review_provider_refused', model=model)
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['review_primary_attempts'] == 2
    assert row['code_review_doc']['review_alternate_attempts'] == 2
    assert 'review_invalid_attempts' not in row['code_review_doc']
    assert not code_review_policy.review_retry_available(row)
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'exhausted'


@pytest.mark.parametrize('status', ['passed', 'rejected'])
def test_completed_verdict_is_never_replayed_after_invalid_episode(recovery_store, recovery_superuser, status):
    from tests.lab_arena.refusal_review_fallback599_postgres_test import verdict
    _, submission, miner = accepted_recovery_submission(recovery_store, 'invalid624-final-' + status)
    claim_error(recovery_store, recovery_superuser, submission, miner)
    _age_review_error(recovery_superuser, submission)
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, token=token)['status'] == 'claimed'
    doc = verdict(PRIMARY)
    if status == 'rejected':
        doc.update(passed=False, verdict='reject')
    assert recovery_store.finish_submission_review(submission, miner, token, status, doc, 1234)['status'] == status
    before = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'existing'
    assert recovery_store.list_ledger(submission_id=submission) == before


def test_latest_runtime_requires_budget_capability(recovery_store, recovery_superuser):
    assert recovery_store.code_review_schema(require_invalid_review_budget=True)['invalid_review_budget']['max_reservations'] == 3
    signature = SIGNATURES[2]
    with recovery_superuser.cursor() as cursor:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        original = cursor.fetchone()[0]
        cursor.execute(original.replace("'invalid_review_budget'", "'missing_invalid_review_budget'"))
        try:
            with pytest.raises(ArenaStoreError, match='invalid budget schema mismatch'):
                recovery_store.code_review_schema(require_invalid_review_budget=True)
            # Older clients do not require the new marker during migration-first.
            assert recovery_store.code_review_schema()['version'] == 207
        finally:
            cursor.execute(original)
