"""Real durable admission claims on the exact637 recovery function bodies."""
import re
import time

import pytest

from lab_arena import code_review, code_review_policy
from lab_arena.store import ArenaStoreError, new_lease_token
from tests.lab_arena import nov06_late_review635_postgres_test as late
from tests.lab_arena import nov06_ninth_review637_postgres_test as ninth
from tests.lab_arena.refusal_review_fallback599_postgres_test import (
    PRIMARY, ALTERNATE, SIGNATURES, accepted_recovery_submission, begin, failure, verdict,
)
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    _age_review_error, recovery_connect, recovery_store, recovery_superuser,
)
from tests.lab_arena.prospective_admission_primary_test import DEEPSEEK, receipt, reviewer
from tests.lab_arena.test_lab_arena_code_review_runtime import _archive, _source
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '643-lab-arena-prospective-admission-primary.sql'
parent_database = ninth.database


@pytest.fixture
def ninth_context(parent_database):
    yield from ninth.context.__wrapped__(parent_database)


def restore_predecessor(cursor):
    text = MIGRATION.read_text()
    for i, signature in enumerate(SIGNATURES):
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        fixed = cursor.fetchone()[0]
        replacements = re.findall(r'v_old:=\$old643_'+str(i)+r'_(\d+)\$(.*?)\$old643_'+str(i)+r'_\1\$;\n  v_new:=\$new643_'+str(i)+r'_\1\$(.*?)\$new643_'+str(i)+r'_\1\$;', text, re.S)
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
            ninth = (SCRIPTS / '637-arena-2026-11-06-ninth-review-recovery.sql').read_text()
            cursor.execute(ninth.split('DO $recover637$')[0]
                           + 'DO $restore637$' + ninth.split('DO $restore637$', 1)[1])
            cursor.execute(MIGRATION.read_text())
        yield pg, dsn
    finally:
        generator.close()


def finish(store, submission, miner, token, model, *, error='code_review_provider_refused', actual=1234, **extra):
    return store.finish_submission_review(submission, miner, token, 'error',
        {**failure(model, error), **extra}, actual)


@pytest.mark.parametrize('locked_table,key', [
    ('lab_arena_rounds', 'round_id'),
    ('lab_arena_submissions', 'submission_id'),
])
def test_busy_native_rows_refuse_migration_immediately_without_changes(
    recovery_connect, recovery_store, recovery_superuser, locked_table, key,
):
    with recovery_superuser.cursor() as cursor:
        restore_predecessor(cursor)
    round_id, submission, miner = accepted_recovery_submission(
        recovery_store, 'primary643-nowait-' + key)
    assert begin(recovery_store, submission, miner, PRIMARY)['status'] == 'claimed'

    def snapshot():
        with recovery_superuser.cursor() as cursor:
            functions = function_snapshot(cursor)
            rows = {}
            for table in ('lab_arena_rounds', 'lab_arena_submissions',
                          'lab_arena_ledger', 'lab_arena_restart_claim_control'):
                cursor.execute(f'SELECT to_jsonb(t) FROM public.{table} t ORDER BY to_jsonb(t)::TEXT')
                rows[table] = cursor.fetchall()
            return functions, rows

    before_functions, before_rows = snapshot()
    blocker = recovery_connect()
    try:
        with blocker.cursor() as cursor:
            cursor.execute(f'SELECT 1 FROM public.{locked_table} WHERE {key}=%s FOR UPDATE',
                           (round_id if key == 'round_id' else submission,))
            assert cursor.fetchone() == (1,)
        started = time.monotonic()
        with recovery_superuser.cursor() as cursor:
            try:
                with pytest.raises(Exception) as failure:
                    cursor.execute(MIGRATION.read_text())
                elapsed = time.monotonic() - started
                assert failure.value.pgcode == '55P03'
                assert elapsed < 1.0, f'busy migration waited {elapsed:.3f}s'
            finally:
                cursor.execute('ROLLBACK')
        assert snapshot() == (before_functions, before_rows)
    finally:
        blocker.rollback()
        blocker.close()

    with recovery_superuser.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    after_functions, after_rows = snapshot()
    assert after_rows == before_rows
    changed = {name for name in before_functions if after_functions[name] != before_functions[name]}
    assert changed == {signature.removeprefix('public.') for signature in SIGNATURES}
    for name in before_functions:
        assert after_functions[name][1] == before_functions[name][1]
    assert recovery_store.code_review_schema(require_prospective_primary=True)['prospective_primary']['model'] == DEEPSEEK


def test_fail_before_then_upgrade_preserves_rows_and_uses_actual_new_primary(recovery_store, recovery_superuser):
    text = MIGRATION.read_text()
    with recovery_superuser.cursor() as cursor:
        restore_predecessor(cursor)
        functions = function_snapshot(cursor)
    _, submission, miner = accepted_recovery_submission(recovery_store, 'primary643-before')
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
        cursor.execute(original_schema.replace('SELECT', '-- unknown643\n  SELECT', 1))
        before = function_snapshot(cursor)
        with pytest.raises(Exception, match='primary643 preimage differs'):
            cursor.execute(MIGRATION.read_text())
        cursor.execute('ROLLBACK')
        assert function_snapshot(cursor) == before
        cursor.execute(original_schema)
        cursor.execute(MIGRATION.read_text())
        assert function_snapshot(cursor) == fixed


def test_unknown_parent_recovery_helper_is_rejected_without_function_changes(recovery_store, recovery_superuser):
    signature = 'public.lab_arena__nov06_late_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)'
    with recovery_superuser.cursor() as cursor:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        original = cursor.fetchone()[0]
        cursor.execute(original.replace('SELECT', '-- unknown643parent\n SELECT', 1))
        before = function_snapshot(cursor)
        with pytest.raises(Exception, match='primary643 parent637 differs'):
            cursor.execute(MIGRATION.read_text())
        cursor.execute('ROLLBACK')
        assert function_snapshot(cursor) == before
        cursor.execute(original)


def test_ninth_receipt_keeps_exact_sol_attempt_and_prior_history(ninth_context):
    store, connection, sql, target, *_ = ninth_context
    ninth.apply(connection, sql)
    before_row = store.get_submission(ninth.SID)
    before_ledger = store.list_ledger(submission_id=ninth.SID)
    with connection.cursor() as cursor:
        functions = function_snapshot(cursor)
    ninth.apply(connection, MIGRATION.read_text())
    assert store.get_submission(ninth.SID) == before_row
    assert store.list_ledger(submission_id=ninth.SID) == before_ledger
    with connection.cursor() as cursor:
        after = function_snapshot(cursor)
    for signature in ('lab_arena__nov06_late_review635_v1(lab_arena_rounds,lab_arena_submissions)',
                      'lab_arena_submissions_frozen_v1()'):
        assert after[signature] == functions[signature]
    assert begin(store, ninth.SID, before_row['miner_hotkey'], DEEPSEEK) == {
        'status': 'route_required', 'review_model': ALTERNATE}
    assert ninth.finish_ninth(store, target)['status'] == 'error'
    row = store.get_submission(ninth.SID)
    assert row['code_review_attempts'] == 4
    assert row['code_review_doc']['model'] == ALTERNATE
    ledger = store.list_ledger(submission_id=ninth.SID)
    assert ledger[:len(before_ledger)] == before_ledger
    assert [entry['entry_doc']['review_model'] for entry in ledger if entry['entry_kind'] == 'reservation'] == [
        PRIMARY, PRIMARY, PRIMARY, ALTERNATE]
    _age_review_error(connection, ninth.SID)
    assert begin(store, ninth.SID, row['miner_hotkey'], ALTERNATE)['status'] == 'exhausted'
    assert store.list_ledger(submission_id=ninth.SID) == ledger


@pytest.mark.parametrize('primary', [PRIMARY, DEEPSEEK])
def test_refusal_episode_keeps_two_real_primary_and_two_alternate_claims(recovery_store, recovery_superuser, primary):
    _, submission, miner = accepted_recovery_submission(recovery_store, 'primary643-'+('legacy' if primary == PRIMARY else 'new'))
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
    _, submission, miner = accepted_recovery_submission(recovery_store, 'primary643-truncated')
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
        'primary643-transient-'+('old' if primary == PRIMARY else 'new'))
    token = new_lease_token()
    assert begin(recovery_store, submission, miner, primary, token)['status'] == 'claimed'
    finish(recovery_store, submission, miner, token, primary, error='code_review_transport_error', actual=None)
    _age_review_error(recovery_superuser, submission)
    before = recovery_store.list_ledger(submission_id=submission)
    assert begin(recovery_store, submission, miner, other) == {'status': 'route_required', 'review_model': primary}
    assert recovery_store.list_ledger(submission_id=submission) == before
    assert begin(recovery_store, submission, miner, primary)['attempt'] == 2


def test_native_runtime_primary_pass_is_bound_to_real_ledger_and_final(recovery_store, recovery_superuser):
    _, submission, _ = accepted_recovery_submission(recovery_store, 'primary643-runtime')
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
    _, submission, miner = accepted_recovery_submission(recovery_store, 'primary643-binding')
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
