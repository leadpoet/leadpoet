"""Native PostgreSQL proof for prospective alternate binding and legacy recovery."""
import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from lab_arena import code_review
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, new_lease_token
from tests.lab_arena import prospective_admission660_postgres_test as primary
from tests.lab_arena.refusal_review_fallback599_postgres_test import (
    begin as legacy_begin, failure, verdict, SIGNATURES,
)
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    _age_review_error, recovery_connect, recovery_store, recovery_superuser,
)
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS / '669-lab-arena-prospective-admission-alternate.sql'
KIMI = 'moonshotai/kimi-k3.1-pro'
DS = code_review.PROSPECTIVE_REVIEW_MODEL
SOL = code_review.FALLBACK_REVIEW_MODEL


@pytest.fixture(scope='module')
def recovery_database():
    generator = primary.recovery_database.__wrapped__()
    try:
        pg, dsn = next(generator)
        with pg.connect(**dsn) as connection:
            connection.autocommit = True
            store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
            try:
                for model, suffix in ((DS, 'archived-ds'), (code_review.DEFAULT_REVIEW_MODEL, 'archived-sonnet')):
                    _, sid, miner = new(store, suffix)
                    for _ in range(2):
                        token = new_lease_token()
                        assert legacy_begin(store, sid, miner, model, token)['status'] == 'claimed'
                        finish_error(store, connection, sid, miner, token, model)
                before = primary.migration_state_snapshot(connection)
                with pytest.raises(ArenaStoreError, match='alternate schema mismatch'):
                    store.code_review_schema(require_prospective_alternate=True)
                with connection.cursor() as cursor:
                    cursor.execute(MIGRATION.read_text())
                after = primary.migration_state_snapshot(connection)
                assert after[1] == before[1]
                changed = {key for key in before[0] if before[0][key] != after[0][key]}
                assert changed == {signature.removeprefix('public.') for signature in SIGNATURES}
                assert set(after[0]) - set(before[0]) == {'lab_arena_begin_submission_review_v2(text,text,text,bigint,text,integer,bigint,text)'}
                assert all(before[0][key][1] == after[0][key][1] for key in changed)
            finally:
                store.close()
        yield pg, dsn
    finally:
        generator.close()


def begin(store, submission, miner, model=DS, token=None, *, alternate=KIMI):
    return store.begin_submission_review(submission, miner, token or new_lease_token(),
        50000, model, 2, 1000, alternate)


def new(store, suffix):
    return primary.accepted_recovery_submission(store, 'alternate669-' + suffix)


def finish_error(store, connection, sid, miner, token, model=DS, error='code_review_provider_refused', **extra):
    result = store.finish_submission_review(sid, miner, token, 'error',
        {**failure(model, error), **extra}, 1234)
    _age_review_error(connection, sid)
    connection.commit()
    return result


def bound_episode(store, connection, suffix, error='code_review_provider_refused'):
    _, sid, miner = new(store, suffix)
    for index in range(2 if error == 'code_review_provider_refused' else 1):
        token = new_lease_token()
        claim = begin(store, sid, miner, DS, token)
        assert claim['review_alternate_model'] == KIMI
        assert claim['status'] == 'claimed'
        extra = {'error_reason': 'truncated'} if error == 'review_response_invalid' else {}
        finish_error(store, connection, sid, miner, token, error=error, **extra)
    return sid, miner


def diagnostic(**extra):
    return {'finish_reason': 'stop', 'native_finish_reason': 'stop',
        'context_integrity': 'verified', 'max_output_tokens': 65536,
        'prompt_tokens': 1000, 'selected_model': 'moonshotai/kimi-k3.1-pro-20261023',
        'selected_provider': 'Moonshot AI', **extra}


def test_new_primary_uses_first_immutable_marker_and_existing_refusal_slots(recovery_store, recovery_superuser):
    sid, miner = bound_episode(recovery_store, recovery_superuser, 'refusals')
    assert recovery_store.get_submission(sid)['code_review_doc']['review_alternate_model'] == KIMI
    before = recovery_store.list_ledger(submission_id=sid)
    assert begin(recovery_store, sid, miner, SOL) == {'status': 'route_required', 'review_model': KIMI}
    assert recovery_store.list_ledger(submission_id=sid) == before
    for count in (1, 2):
        token = new_lease_token()
        claim = begin(recovery_store, sid, miner, KIMI, token)
        assert claim['status'] == 'claimed'
        result = finish_error(recovery_store, recovery_superuser, sid, miner, token, KIMI,
            error='code_review_transport_error', review_alternate_model=SOL)
        assert result['review_alternate_model'] == KIMI
        assert result['review_alternate_attempts'] == count
    assert begin(recovery_store, sid, miner, KIMI)['status'] == 'exhausted'
    assert len([row for row in recovery_store.list_ledger(submission_id=sid) if row['entry_kind']=='reservation']) == 4


def test_ds_truncation_consumes_next_slot_then_kimi_65536_truncation_is_terminal(recovery_store, recovery_superuser):
    sid, miner = bound_episode(recovery_store, recovery_superuser, 'truncated', 'review_response_invalid')
    assert begin(recovery_store, sid, miner, SOL)['review_model'] == KIMI
    token = new_lease_token()
    claim = begin(recovery_store, sid, miner, KIMI, token)
    assert claim['status'] == 'claimed' and claim['review_output_token_recommendation'] == 65536
    result = finish_error(recovery_store, recovery_superuser, sid, miner, token, KIMI,
        error='review_response_invalid', error_reason='truncated_at_limit')
    assert result['review_invalid_attempts'] == 2 and result['retryable'] is False
    assert begin(recovery_store, sid, miner, KIMI)['status'] == 'exhausted'


@pytest.mark.parametrize('damage', [None, 'provider', 'canonical', 'integrity', 'budget'])
def test_kimi_verdict_requires_exact_native_route_evidence(recovery_store, recovery_superuser, damage):
    sid, miner = bound_episode(recovery_store, recovery_superuser, 'proof-' + str(damage), 'review_response_invalid')
    token = new_lease_token()
    assert begin(recovery_store, sid, miner, KIMI, token)['status'] == 'claimed'
    evidence = diagnostic()
    if damage == 'provider': evidence['selected_provider'] = 'Azure'
    elif damage == 'canonical': evidence['selected_model'] += '-other'
    elif damage == 'integrity': evidence['context_integrity'] = 'context_budget'
    elif damage == 'budget': evidence['max_output_tokens'] = 131072
    document = {**verdict(KIMI), 'provider_diagnostic': evidence, 'review_alternate_model': SOL}
    if damage:
        with pytest.raises(ArenaStoreError):
            recovery_store.finish_submission_review(sid, miner, token, 'passed', document, 1234)
        assert recovery_store.get_submission(sid)['code_review_status'] == 'reviewing'
        finish_error(recovery_store, recovery_superuser, sid, miner, token, KIMI, error='code_review_transport_error')
    else:
        assert recovery_store.finish_submission_review(sid, miner, token, 'passed', document, 1234)['status'] == 'passed'
        assert recovery_store.get_submission(sid)['code_review_doc']['review_alternate_model'] == KIMI


def test_concurrent_first_claim_writes_one_bound_reservation(recovery_store, recovery_connect):
    _, sid, miner = new(recovery_store, 'race')
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: begin(recovery_store, sid, miner, DS), range(2)))
    assert sorted(row['status'] for row in results) == ['busy', 'claimed']
    reservations = [row for row in recovery_store.list_ledger(submission_id=sid) if row['entry_kind']=='reservation']
    assert len(reservations) == 1 and reservations[0]['entry_doc']['review_alternate_model'] == KIMI


def test_migration_idempotent_and_function_only(recovery_superuser):
    before = primary.migration_state_snapshot(recovery_superuser)
    with recovery_superuser.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    assert primary.migration_state_snapshot(recovery_superuser) == before


def test_old_seven_arg_ds_still_binds_sol_and_new_caller_cannot_change_it(recovery_store, recovery_superuser):
    _, sid, miner = new(recovery_store, 'legacy7')
    for index in range(2):
        token = new_lease_token()
        claim = legacy_begin(recovery_store, sid, miner, DS, token)
        assert claim['status'] == 'claimed' and claim['review_alternate_model'] == SOL
        finish_error(recovery_store, recovery_superuser, sid, miner, token)
    assert begin(recovery_store, sid, miner, KIMI) == {'status': 'route_required', 'review_model': SOL}
    token = new_lease_token()
    assert legacy_begin(recovery_store, sid, miner, SOL, token)['status'] == 'claimed'
    assert recovery_store.finish_submission_review(sid, miner, token, 'passed', verdict(SOL), 1234)['status'] == 'passed'


def test_old_client_rollback_cannot_spend_sol_on_new_kimi_episode(recovery_store, recovery_superuser):
    sid, miner = bound_episode(recovery_store, recovery_superuser, 'rollback')
    before = recovery_store.list_ledger(submission_id=sid)
    assert legacy_begin(recovery_store, sid, miner, SOL) == {'status': 'route_required', 'review_model': KIMI}
    assert recovery_store.list_ledger(submission_id=sid) == before


@pytest.mark.parametrize('alternate', ['moonshotai/kimi-k3.1-pro:highspeed', '', 'openai/gpt-6.1-sol-20260929'])
def test_opt_in_rejects_unapproved_alternates(recovery_store, alternate):
    _, sid, miner = new(recovery_store, 'forged-' + hashlib.sha256(alternate.encode()).hexdigest()[:6])
    with pytest.raises(ArenaStoreError):
        begin(recovery_store, sid, miner, alternate=alternate)
    assert recovery_store.list_ledger(submission_id=sid) == []


def test_old_new_concurrent_first_claims_choose_one_immutable_authority(recovery_store, recovery_superuser):
    _, sid, miner = new(recovery_store, 'mixed-race')
    tokens = [new_lease_token(), new_lease_token()]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(legacy_begin, recovery_store, sid, miner, DS, tokens[0]),
                   executor.submit(begin, recovery_store, sid, miner, DS, tokens[1])]
        results = [future.result() for future in futures]
    assert sorted(row['status'] for row in results) == ['busy', 'claimed']
    winner = next(index for index, row in enumerate(results) if row['status']=='claimed')
    alternate = SOL if winner == 0 else KIMI
    assert results[winner]['review_alternate_model'] == alternate
    finish_error(recovery_store, recovery_superuser, sid, miner, tokens[winner])
    token = new_lease_token()
    assert begin(recovery_store, sid, miner, DS, token)['review_alternate_model'] == alternate
    finish_error(recovery_store, recovery_superuser, sid, miner, token)
    wrong = KIMI if alternate == SOL else SOL
    assert begin(recovery_store, sid, miner, wrong) == {'status': 'route_required', 'review_model': alternate}
    reservations = [row for row in recovery_store.list_ledger(submission_id=sid) if row['entry_kind']=='reservation']
    assert len(reservations) == 2 and all(row['entry_doc']['review_alternate_model']==alternate for row in reservations)


def test_existing_markerless_ds_and_sonnet_episodes_keep_sol(recovery_store, recovery_superuser):
    from tests.lab_arena.test_lab_arena_migration_postgres import hotkey
    for suffix in ('archived-ds', 'archived-sonnet'):
        sid = 'alternate669-' + suffix + '-recovery'
        miner = hotkey('primary660-alternate669-' + suffix)
        row = recovery_store.get_submission(sid)
        assert 'review_alternate_model' not in row['code_review_doc']
        before = recovery_store.list_ledger(submission_id=sid)
        assert begin(recovery_store, sid, miner, KIMI) == {'status':'route_required', 'review_model':SOL}
        assert recovery_store.list_ledger(submission_id=sid) == before
        token = new_lease_token()
        assert begin(recovery_store, sid, miner, SOL, token)['status'] == 'claimed'
        assert recovery_store.finish_submission_review(sid, miner, token, 'passed', verdict(SOL), 1234)['status'] == 'passed'
        assert recovery_store.list_ledger(submission_id=sid)[:len(before)] == before


def test_additive_schema_supports_old_and_new_startup_contracts(recovery_store):
    schema = recovery_store.code_review_schema(require_refusal_fallback=True,
        require_invalid_review_budget=True, require_prospective_primary=True)
    assert schema['prospective_primary']['truncation_alternate'] == SOL
    assert recovery_store.code_review_schema(require_prospective_alternate=True)['prospective_alternate']['model'] == KIMI


def test_native_runtime_and_postgres_commit_same_kimi_route_and_cost(recovery_store, recovery_superuser):
    from tests.lab_arena.prospective_admission_alternate_test import kimi_receipt
    from tests.lab_arena.prospective_admission_primary_test import reviewer
    from tests.lab_arena.test_lab_arena_code_review_runtime import _archive, _source
    instance, _, _, transport = reviewer(lambda parameters: {'body':kimi_receipt(parameters)}, store=recovery_store)
    instance._prices['models'][KIMI] = dict(instance._prices['models'][DS])
    sid, miner = bound_episode(recovery_store, recovery_superuser, 'native-runtime', 'review_response_invalid')
    row = {**recovery_store.get_submission(sid), 'source_size_bytes':len(_archive(_source()))}
    result = instance.review(row)
    assert result['status'] == 'passed'
    assert result['actual_microusd'] == 1234
    saved = recovery_store.get_submission(sid)['code_review_doc']
    assert saved['model'] == saved['review_alternate_model'] == KIMI
    settlement = [entry for entry in recovery_store.list_ledger(submission_id=sid)
                  if entry['call_identity']==result['call_identity'] and entry['entry_kind']=='settlement'][0]
    assert settlement['entry_doc']['provider_diagnostic']['selected_provider'] == 'Moonshot AI'
    assert settlement['entry_doc']['provider_diagnostic']['max_output_tokens'] == 65536
    assert len(transport.sent) == 1


@pytest.mark.parametrize('model', ['moonshotai/kimi-k3.1-pro-20261023', 'moonshotai/kimi-k3.1-pro:highspeed', 'openai/gpt-unapproved'])
def test_requested_model_must_be_an_exact_admission_id(recovery_store, model):
    _, sid, miner = new(recovery_store, 'requested-' + hashlib.sha256(model.encode()).hexdigest()[:6])
    for caller in (begin, legacy_begin):
        with pytest.raises(ArenaStoreError):
            caller(recovery_store, sid, miner, model)
    assert recovery_store.list_ledger(submission_id=sid) == []


def test_runtime_explicit_optin_freezes_first_primary_reservation(recovery_store):
    from tests.lab_arena.prospective_admission_primary_test import receipt, reviewer
    from tests.lab_arena.test_lab_arena_code_review_runtime import _archive, _source
    _, sid, _ = new(recovery_store, 'explicit-runtime-optin')
    instance, _, _, transport = reviewer(lambda parameters: {'body': receipt(parameters)},
        store=recovery_store, prospective_alternate_model=KIMI)
    row = {**recovery_store.get_submission(sid), 'source_size_bytes': len(_archive(_source()))}
    assert instance.review(row)['status'] == 'passed'
    assert len(transport.sent) == 1
    saved = recovery_store.get_submission(sid)['code_review_doc']
    assert saved['model'] == DS and saved['review_alternate_model'] == KIMI
    reservations = [entry for entry in recovery_store.list_ledger(submission_id=sid)
                    if entry['entry_kind'] == 'reservation']
    assert len(reservations) == 1 and reservations[0]['entry_doc']['review_alternate_model'] == KIMI
