"""Native SQL proof for bounded owner excerpts without admission or ACL drift."""
from copy import deepcopy
from datetime import datetime, timedelta

import pytest

from lab_arena import code_review, owner_review
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, new_lease_token
from tests.lab_arena import prospective_admission669_postgres_test as alternate
from tests.lab_arena.prospective_admission660_postgres_test import migration_state_snapshot
from tests.lab_arena.refusal_review_fallback599_postgres_test import verdict
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '687-lab-arena-private-owner-review-findings.sql'
SIGNATURE = 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)'


@pytest.fixture(scope='module')
def database():
    generator = alternate.recovery_database.__wrapped__()
    try:
        pg, dsn = next(generator)
        with pg.connect(**dsn) as connection:
            connection.autocommit = True
            before = migration_state_snapshot(connection)
            with connection.cursor() as cursor:
                cursor.execute(MIGRATION.read_text())
            after = migration_state_snapshot(connection)
            assert after[1] == before[1]  # Includes historical submissions and ledger.
            changed = {name for name in before[0] if before[0][name] != after[0][name]}
            assert changed == {SIGNATURE.removeprefix('public.')}
            assert all(before[0][name][1] == after[0][name][1] for name in before[0])
            with connection.cursor() as cursor:
                cursor.execute(MIGRATION.read_text())
            assert migration_state_snapshot(connection) == after
        yield pg, dsn
    finally:
        generator.close()


@pytest.fixture
def context(database):
    pg, dsn = database
    connection = pg.connect(**dsn)
    connection.autocommit = True
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    try:
        yield store, connection
    finally:
        store.close()
        connection.close()


def start(store, suffix):
    rid, sid, miner = alternate.new(store, 'owner687-' + suffix)
    token = new_lease_token()
    assert alternate.begin(store, sid, miner, token=token)['status'] == 'claimed'
    return rid, sid, miner, token


def rejected():
    return {**verdict(code_review.PROSPECTIVE_REVIEW_MODEL), 'passed': False,
            'verdict': 'reject', 'categories': ['fabricated_evidence'],
            'owner_findings': owner_review.from_validated_findings([{
                'category': 'fabricated_evidence', 'file': 'src/agent.py',
                'evidence': 'own_source_call()', 'explanation': 'never retained'}])}


def test_native_rejected_findings_legacy_and_unknown_cost(context):
    store, connection = context
    for suffix, cost in [('accounted',1234),('unknown',None),('legacy',1234)]:
        _, sid, miner, token = start(store, suffix)
        document = rejected()
        if suffix == 'legacy': document.pop('owner_findings')
        receipt = store.finish_submission_review(sid, miner, token, 'rejected', document, cost)
        assert receipt['code_review_status'] == 'rejected'
        saved = store.get_submission(sid)['code_review_doc']
        assert saved.get('owner_findings') == document.get('owner_findings')
        assert saved['passed'] is False and saved['categories'] == ['fabricated_evidence']
        assert 'explanation' not in repr(saved)
        before = migration_state_snapshot(connection)[1]
        replay = store.finish_submission_review(sid, miner, token, 'rejected', document, cost)
        assert replay['status'] == 'existing'
        assert migration_state_snapshot(connection)[1] == before


def test_shape_size_category_and_status_gates(context):
    store, connection = context
    _, sid, miner, token = start(store, 'invalid')
    base = rejected()
    invalid = []
    for value in [None, [], {}, {**base['owner_findings'],'extra':'secret'},
                  {**base['owner_findings'],'truncated':'false'},
                  {**base['owner_findings'],'findings':[None]},
                  {**base['owner_findings'],'findings':base['owner_findings']['findings'] * 9}]:
        invalid.append({**base, 'owner_findings':value})
    for change in [{'extra':'private explanation'}, {'category':'malicious_behavior'},
                   {'file':''}, {'file':'p' * 513}, {'evidence':''},
                   {'evidence':'é' * 501}, {'evidence':None}]:
        document = deepcopy(base)
        document['owner_findings']['findings'][0].update(change)
        invalid.append(document)
    for document in invalid:
        with pytest.raises(ArenaStoreError, match='lab_arena_code_review_input_invalid'):
            store.finish_submission_review(sid, miner, token, 'rejected', document, 1234)
        assert store.get_submission(sid)['code_review_status'] == 'reviewing'
    with pytest.raises(ArenaStoreError, match='lab_arena_code_review_input_invalid'):
        store.finish_submission_review(sid, miner, token, 'passed',
            {**verdict(code_review.PROSPECTIVE_REVIEW_MODEL), 'owner_findings':base['owner_findings']},1234)
    assert store.finish_submission_review(sid, miner, token, 'passed',
        verdict(code_review.PROSPECTIVE_REVIEW_MODEL),1234)['code_review_status'] == 'passed'
    assert 'owner_findings' not in store.get_submission(sid)['code_review_doc']


def test_late_result_and_wrong_claim_cannot_publish_findings(context):
    store, connection = context
    rid, sid, miner, token = start(store, 'late')
    stale = store.finish_submission_review(sid, miner, new_lease_token(), 'rejected', rejected(),1234)
    assert stale['status'] == 'stale'
    assert store.get_submission(sid)['code_review_doc'] is None
    round_row = store.get_round(rid)
    deadline = datetime.fromisoformat(round_row['configuration_doc']['schedule']['benchmark_deadline'])
    with connection.cursor() as cursor:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIGNATURE,))
        definition = cursor.fetchone()[0]
        cursor.execute(definition.replace('pg_catalog.clock_timestamp()',
            "TIMESTAMPTZ '%s'" % (deadline + timedelta(seconds=1)).isoformat()))
    try:
        result = store.finish_submission_review(sid, miner, token, 'rejected', rejected(),1234)
    finally:
        with connection.cursor() as cursor:
            cursor.execute(definition)
    assert result['status'] == 'deadline' and result['code_review_status'] == 'error'
    assert 'owner_findings' not in store.get_submission(sid)['code_review_doc']


def test_exact_preimage_rejects_function_drift_and_no_public_grants(context):
    store, connection = context
    with connection.cursor() as cursor:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIGNATURE,))
        definition = cursor.fetchone()[0]
        for role, allowed in [('lab_arena_service',True),('anon',False),('authenticated',False),('service_role',False)]:
            cursor.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')',(role,SIGNATURE))
            assert cursor.fetchone()[0] is allowed
        try:
            cursor.execute(definition.replace('-- Optional diagnostics', '-- Changed optional diagnostics'))
            with pytest.raises(Exception, match='preimage differs'):
                cursor.execute(MIGRATION.read_text())
            cursor.execute('ROLLBACK')
        finally:
            cursor.execute(definition)
