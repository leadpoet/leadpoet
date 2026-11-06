"""Real durable admission claims on the exact635 recovery function bodies."""
import re

import pytest

from lab_arena import code_review, code_review_policy
from lab_arena.store import ArenaStoreError, new_lease_token
from tests.lab_arena import nov06_late_review635_postgres_test as late
from tests.lab_arena.refusal_review_fallback599_postgres_test import (
    PRIMARY, ALTERNATE, SIGNATURES, accepted_recovery_submission, begin, failure, verdict,
)
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    _age_review_error, recovery_connect, recovery_store, recovery_superuser,
)
from tests.lab_arena.prospective_admission_primary_test import DEEPSEEK, receipt, reviewer
from tests.lab_arena.test_lab_arena_code_review_runtime import _archive, _source
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '641-lab-arena-prospective-admission-primary.sql'


def restore_predecessor(cursor):
    text = MIGRATION.read_text()
    for i, signature in enumerate(SIGNATURES):
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        fixed = cursor.fetchone()[0]
        replacements = re.findall(r'v_old:=\$old641_'+str(i)+r'_(\d+)\$(.*?)\$old641_'+str(i)+r'_\1\$;\n  v_new:=\$new641_'+str(i)+r'_\1\$(.*?)\$new641_'+str(i)+r'_\1\$;', text, re.S)
        assert replacements
        for _, old, new in reversed(replacements):
            assert new in fixed
            fixed = fixed.replace(new, old)
        cursor.execute(fixed)


def function_snapshot(cursor):
    cursor.execute("SELECT oid::regprocedure::TEXT,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' "
                   "FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
    return {name: (definition, metadata) for name, definition, metadata in cursor.fetchall()}


@pytest.fixture(scope='module')
def recovery_database():
    generator = late.database.__wrapped__()
    try:
        pg, dsn, _ = next(generator)
        with pg.connect(**dsn) as connection, connection.cursor() as cursor:
            sql = late.M635.read_text()
            # Install only committed function definitions; no dated receipt,
            # recovery membership, row changes, or additional claims are made.
            cursor.execute(sql.split('DO $recover635$')[0]
                           + 'DO $restore635$' + sql.split('DO $restore635$', 1)[1])
            cursor.execute(MIGRATION.read_text())
        yield pg, dsn
    finally:
        generator.close()


def finish(store, submission, miner, token, model, *, error='code_review_provider_refused', actual=1234, **extra):
    return store.finish_submission_review(submission, miner, token, 'error',
        {**failure(model, error), **extra}, actual)


def test_fail_before_then_upgrade_preserves_rows_and_uses_actual_new_primary(recovery_store, recovery_superuser):
    text = MIGRATION.read_text()
    with recovery_superuser.cursor() as cursor:
        restore_predecessor(cursor)
        functions = function_snapshot(cursor)
    _, submission, miner = accepted_recovery_submission(recovery_store, 'primary641-before')
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, DEEPSEEK, token)['status'] == 'claimed'
    finish(recovery_store, submission, miner, token, DEEPSEEK)
    _age_review_error(recovery_superuser, submission)
    before = recovery_store.get_submission(submission)
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner, DEEPSEEK) == {'status': 'route_required', 'review_model': PRIMARY}
    with pytest.raises(ArenaStoreError, match='primary schema mismatch'):
        recovery_store.code_review_schema(require_prospective_primary=True)
    with recovery_superuser.cursor() as cursor:
        cursor.execute(text)
        after = function_snapshot(cursor)
        assert functions.keys() == after.keys()
        for name, previous in functions.items():
            assert after[name][1] == previous[1]
            if name not in {signature.removeprefix('public.') for signature in SIGNATURES}:
                assert after[name] == previous
    assert recovery_store.get_submission(submission) == before
    assert recovery_store.list_ledger(submission_id=submission) == ledger
    assert begin(recovery_store, submission, miner, PRIMARY) == {'status': 'route_required', 'review_model': DEEPSEEK}
    assert begin(recovery_store, submission, miner, DEEPSEEK)['attempt'] == 2


def test_migration_replay_and_unknown_final_body_preserve_all_functions(recovery_store, recovery_superuser):
    with recovery_superuser.cursor() as cursor:
        fixed = function_snapshot(cursor)
        cursor.execute(MIGRATION.read_text())
        assert function_snapshot(cursor) == fixed
        restore_predecessor(cursor)
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURES[-1],))
        original_schema = cursor.fetchone()[0]
        cursor.execute(original_schema.replace('SELECT', '-- unknown641\n  SELECT', 1))
        before = function_snapshot(cursor)
        with pytest.raises(Exception, match='primary641 preimage differs'):
            cursor.execute(MIGRATION.read_text())
        cursor.execute('ROLLBACK')
        assert function_snapshot(cursor) == before
        cursor.execute(original_schema)
        cursor.execute(MIGRATION.read_text())
        assert function_snapshot(cursor) == fixed


@pytest.mark.parametrize('primary', [PRIMARY, DEEPSEEK])
def test_refusal_episode_keeps_two_real_primary_and_two_alternate_claims(recovery_store, recovery_superuser, primary):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'primary641-'+('legacy' if primary == PRIMARY else 'new'))
    for index, model in enumerate([primary, primary, ALTERNATE, ALTERNATE], 1):
        if index > 1:
            _age_review_error(recovery_superuser, submission)
        token = new_lease_token()
        assert begin(recovery_store, submission, miner, model, token)['attempt'] == index
        assert begin(recovery_store, submission, miner, model, token)['idempotent'] is True
        finish(recovery_store, submission, miner, token, model, actual=None if index == 1 else 1234)
        row = recovery_store.get_submission(submission)
        assert row['code_review_doc']['review_primary_model'] == primary
        assert row['code_review_doc']['review_primary_attempts'] == min(index, 2)
        assert row['code_review_doc']['review_alternate_attempts'] == max(0, index-2)
        if index < 4:
            assert code_review_policy.review_model_for(row, prospective_primary_model=DEEPSEEK) == [primary, ALTERNATE, ALTERNATE][index-1]
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert [row['entry_doc']['review_model'] for row in ledger if row['entry_kind'] == 'reservation'] == [primary, primary, ALTERNATE, ALTERNATE]
    assert sum(row['entry_kind'] == 'uncertain' for row in ledger) == 1
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'exhausted'
    assert recovery_store.list_ledger(submission_id=submission) == ledger


def test_deepseek_truncation_uses_next_existing_slot_and_preserves_unknown_cost(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'primary641-truncated')
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, DEEPSEEK, token)['status'] == 'claimed'
    finish(recovery_store, submission, miner, token, DEEPSEEK,
           error='review_response_invalid', error_reason='truncated', actual=None)
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner, DEEPSEEK) == {'status': 'route_required', 'review_model': ALTERNATE}
    token = new_lease_token()
    claim = begin(recovery_store, submission, miner, ALTERNATE, token)
    assert claim['attempt'] == 2 and claim['review_output_token_recommendation'] == 65536
    finish(recovery_store, submission, miner, token, ALTERNATE, error='code_review_transport_error')
    _age_review_error(recovery_superuser, submission)
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, ALTERNATE, token)['attempt'] == 3
    finish(recovery_store, submission, miner, token, ALTERNATE, error='review_response_invalid')
    _age_review_error(recovery_superuser, submission)
    assert begin(recovery_store, submission, miner, ALTERNATE)['status'] == 'exhausted'
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert len([row for row in ledger if row['entry_kind'] == 'reservation']) == 3
    assert any(row['entry_kind'] == 'uncertain' for row in ledger)


@pytest.mark.parametrize('primary,other', [(PRIMARY, DEEPSEEK), (DEEPSEEK, PRIMARY)])
def test_transient_episode_cannot_change_primary_model(recovery_store, recovery_superuser, primary, other):
    _, submission, miner = accepted_recovery_submission(recovery_store,
        'primary641-transient-'+('old' if primary == PRIMARY else 'new'))
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, primary, token)['status'] == 'claimed'
    finish(recovery_store, submission, miner, token, primary, error='code_review_transport_error', actual=None)
    _age_review_error(recovery_superuser, submission)
    before = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner, other) == {'status': 'route_required', 'review_model': primary}
    assert recovery_store.list_ledger(submission_id=submission) == before
    assert begin(recovery_store, submission, miner, primary)['attempt'] == 2


def test_native_runtime_primary_pass_is_bound_to_real_ledger_and_final(recovery_store, recovery_superuser):
    _, submission, _ = accepted_recovery_submission(recovery_store, 'primary641-runtime')
    instance, _, _, transport = reviewer(lambda parameters: {'body': receipt(parameters)}, store=recovery_store)
    row = {**recovery_store.get_submission(submission), 'source_size_bytes': len(_archive(_source()))}
    assert instance.review(row)['status'] == 'passed'
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert len(transport.sent) == 1
    assert [item['entry_doc']['review_model'] for item in ledger if item['entry_kind'] == 'reservation'] == [DEEPSEEK]
    assert [item['amount_microusd'] for item in ledger if item['entry_kind'] == 'settlement'] == [1234]
    assert instance.review(recovery_store.get_submission(submission))['status'] == 'existing'
    assert recovery_store.list_ledger(submission_id=submission) == ledger


def test_actual_model_binding_and_db_first_schema_remain_fail_closed(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'primary641-binding')
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, DEEPSEEK, token)['status'] == 'claimed'
    with pytest.raises(ArenaStoreError, match='input_invalid'):
        recovery_store.finish_submission_review(submission, miner, token, 'passed', verdict(PRIMARY), 1234)
    assert recovery_store.get_submission(submission)['code_review_status'] == 'reviewing'
    assert recovery_store.finish_submission_review(submission, miner, token, 'passed', verdict(DEEPSEEK), 1234)['status'] == 'passed'
    marker = recovery_store.code_review_schema(require_refusal_fallback=True, require_invalid_review_budget=True,
                                               require_prospective_primary=True)
    assert marker['refusal_fallback']['primary_model'] == PRIMARY
    assert marker['invalid_review_budget']['max_reservations'] == 3
    assert marker['prospective_primary']['model'] == DEEPSEEK
