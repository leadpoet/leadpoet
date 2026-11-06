"""Native ledger authority for the final existing invalid-review route."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from lab_arena import code_review_policy
from lab_arena.store import new_lease_token
from tests.lab_arena.invalid_review_budget624_postgres_test import (
    recovery_database as predecessor_database, claim_error, finish,
)
from tests.lab_arena.refusal_review_fallback599_postgres_test import (
    PRIMARY, ALTERNATE, SIGNATURES, accepted_recovery_submission, begin, verdict,
)
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    _age_review_error, recovery_connect, recovery_store, recovery_superuser,
)
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '634-lab-arena-invalid-review-alternate.sql'


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


def test_exact_upgrade_changes_only_route_and_preserves_rows(recovery_store, recovery_superuser):
    with recovery_superuser.cursor() as cursor:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[0],))
        fixed = cursor.fetchone()[0]
        predecessor = fixed.replace(MIGRATION.read_text().split('$new634$')[1],
                                    MIGRATION.read_text().split('$old634$')[1])
        assert predecessor != fixed
        cursor.execute(predecessor)
    _, submission, miner = accepted_recovery_submission(recovery_store, 'alternate634-upgrade')
    claim_error(recovery_store, recovery_superuser, submission, miner)
    claim_error(recovery_store, recovery_superuser, submission, miner)
    _age_review_error(recovery_superuser, submission)
    before = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'fallback_not_authorized'
    with recovery_superuser.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        cursor.execute(MIGRATION.read_text())
    assert recovery_store.list_ledger(submission_id=submission) == before
    assert begin(recovery_store, submission, miner, PRIMARY) == {'status': 'route_required', 'review_model': ALTERNATE}
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, ALTERNATE, token)['attempt'] == 3
    assert begin(recovery_store, submission, miner, ALTERNATE, token)['idempotent'] is True
    assert finish(recovery_store, submission, miner, token, model=ALTERNATE)['status'] == 'error'
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['review_invalid_attempts'] == 3
    assert row['code_review_doc']['retryable'] is False
    assert not code_review_policy.review_retry_available(row)
    _age_review_error(recovery_superuser, submission)
    after = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'exhausted'
    assert recovery_store.list_ledger(submission_id=submission) == after
    assert after[:len(before)] == before
    assert [x['amount_microusd'] for x in after if x['entry_kind'] == 'settlement'] == [1234] * 3


def test_last_slot_atomic_and_pass_is_final(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'alternate634-race')
    claim_error(recovery_store, recovery_superuser, submission, miner)
    claim_error(recovery_store, recovery_superuser, submission, miner, 'code_review_transport_error')
    _age_review_error(recovery_superuser, submission)
    tokens = [new_lease_token(), new_lease_token()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda token: begin(recovery_store, submission, miner, ALTERNATE, token), tokens))
    assert sorted(x['status'] for x in results) == ['busy', 'claimed']
    token = tokens[next(i for i, result in enumerate(results) if result['status'] == 'claimed')]
    assert recovery_store.finish_submission_review(submission, miner, token, 'passed', verdict(), 4321)['status'] == 'passed'
    before = recovery_store.list_ledger(submission_id=submission)
    assert sum(x['entry_kind'] == 'reservation' for x in before) == 3
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'existing'
    assert recovery_store.list_ledger(submission_id=submission) == before


def test_refusal_route_dominates_invalid_projection(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'alternate634-refusal')
    claim_error(recovery_store, recovery_superuser, submission, miner)
    claim_error(recovery_store, recovery_superuser, submission, miner, 'code_review_provider_refused')
    _age_review_error(recovery_superuser, submission)
    row = recovery_store.get_submission(submission)
    assert row['code_review_doc']['review_invalid_attempts'] == 2
    assert row['code_review_doc']['review_primary_attempts'] == 1
    assert code_review_policy.review_model_for(row) == PRIMARY
    assert begin(recovery_store, submission, miner, ALTERNATE) == {'status': 'route_required', 'review_model': PRIMARY}
    assert begin(recovery_store, submission, miner, PRIMARY)['status'] == 'claimed'


def test_no_early_alternate_for_ordinary_transients(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'alternate634-transient')
    for _ in range(3):
        claim_error(recovery_store, recovery_superuser, submission, miner, 'code_review_transport_error')
    _age_review_error(recovery_superuser, submission)
    before = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'fallback_not_authorized'
    assert recovery_store.list_ledger(submission_id=submission) == before
    assert begin(recovery_store, submission, miner, PRIMARY)['status'] == 'claimed'
