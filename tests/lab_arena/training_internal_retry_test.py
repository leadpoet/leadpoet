"""Broker attempts share one action; only the final reply reaches the host."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_export, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import submission
from tests.lab_arena.training_hidden_retry_test import recovered_example
from tests.lab_arena.training_steps_test import seal


def internal_retry_example(*, accounting='uncertain', amount=None, api='openrouter.responses'):
    value = recovered_example(api=api)
    failed, success = value['provider_calls'][:2]
    base = contracts.hash_bytes(b'broker internal retry request')
    old_identity = success['call_identity']
    success_identity = contracts.document_hash({'base_call_identity': base, 'provider_attempt': 2})
    failed.update(call_identity=base, base_call_identity=base, provider_attempt=1,
                  terminal_entry_id=3,
                  action_sequence=success['action_sequence'], accounting_state=accounting,
                  actual_microusd=amount, response_status=402,
                  evidence_gaps=['provider_cost_unresolved'] if accounting == 'uncertain' else [],
                  response={'error': {'code': 'miner_credentials_unavailable'}})
    failed['response_sha256'] = contracts.document_hash(failed['response'])
    evidence = {'error_class': 'account_credential_failure', 'provider_status': 429,
                'action_sequence': failed['action_sequence'], 'provider_attempt': 1,
                'base_call_identity': base}
    receipt = failed['response_receipts'][0]
    receipt.update(entry_id=3, response_ref={'entry_id': 3, 'field': 'response'},
                   entry_kind=accounting, response_sha256=failed['response_sha256'],
                   wire_body_sha256=failed['response_sha256'],
                   wire_body_bytes=len(contracts.canonical_json(failed['response']).encode()),
                   metadata={'status': 402, 'call_succeeded': False, 'account_failure_evidence': evidence})
    failed['lifecycle'][0].update(entry_id=3, entry_kind=accounting, amount_microusd=0 if amount is None else amount)
    failed['upstream_response_receipts'][0]['entry_id'] = 3
    success.update(call_identity=success_identity, base_call_identity=base, provider_attempt=2,
                   funding_source='miner_key', reservation_entry_id=4, terminal_entry_id=6)
    for row in success['lifecycle']:
        row['entry_id'] += 3
        row['funding_source'] = 'miner_key'
    success['response_receipts'][0].update(entry_id=6, response_ref={'entry_id': 6, 'field': 'response'})
    success['response_provenance']['worker_received_receipt_entry_ids'] = [6]
    artifact = value['worker_observation_artifacts']['run']
    for row in artifact['payload']['response_receipts']:
        if row['call_identity'] == old_identity:
            row['call_identity'] = success_identity
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    for call in value['provider_calls'][1:]:
        call['response_provenance']['worker_receipt_ref']['payload_sha256'] = artifact['payload_sha256']
    if accounting == 'uncertain':
        value['completeness']['complete'] = False
        value['completeness']['incomplete_reasons'] = ['provider_cost_unresolved']
    return seal(value)


@pytest.mark.parametrize('accounting,amount', [('uncertain', None), ('settlement', 0), ('settlement', 7)])
@pytest.mark.parametrize('api', ['openrouter.responses', 'openrouter.chat'])
def test_internal_retry_keeps_cost_uncertainty_and_every_successful_target(accounting, amount, api):
    value = internal_retry_example(accounting=accounting, amount=amount, api=api)
    before = deepcopy(value)
    failed, success = value['provider_calls'][:2]
    assert training_thinking.verified_failed_attempt(value, failed)
    training_steps._candidate(value, success)
    with pytest.raises(ValueError, match='worker_write_unverified'):
        training_steps._candidate(value, failed)
    result = training_eligibility.assess_capture(submission(), {'run': value}, ['run'], enabled=True)
    assert result['eligible'], result
    assert result['counts']['reasoning_by_run'][0]['verified_failed_model_attempts'] == 1
    linked = deepcopy(value)
    errors = training_export._link_worker_receipts(
        {'run_id': 'run', 'kind': 'execute'}, linked['provider_calls'],
        linked['worker_observation_artifacts']['run'], require_client_ack=False,
        configuration=value['configuration'])
    assert not errors
    splits, manifest = training_thinking.build([value], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    original = deepcopy(value)
    original['provider_calls'].pop(0)
    seal(original)
    expected, _ = training_thinking.build([original], training_steps.Rules(production_rounds_only=False))
    fields = ('candidate_id', 'input', 'target', 'generated_thinking')
    assert [{key: row[key] for key in fields} for row in records] == [
        {key: row[key] for key in fields} for row in expected['train'] + expected['eval']], manifest
    assert len(records) == 3
    assert all(row['source_refs'][0]['call_identity'] != failed['call_identity'] for row in records)
    assert value == before


@pytest.mark.parametrize('damage', [
    'request', 'body', 'base', 'run', 'attempt', 'identity', 'future_attempt', 'action',
    'terminal_order', 'reservation_order', 'failed_reservation_order', 'success_reservation_order',
    'missing_success', 'missing_worker', 'write',
    'missing_upstream', 'generated', 'private_upstream', 'private_request', 'private_response',
    'private_success', 'private_success_request', 'generated_terminal',
    'missing_failure_evidence', 'failure_evidence', 'uncertain_cost', 'missing_cost',
    'terminal_cost', 'terminal_kind', 'terminal_status', 'terminal_success',
    'missing_cost_gap', 'known_cost_gap', 'legacy',
])
@pytest.mark.parametrize('api', ['openrouter.responses', 'openrouter.chat'])
def test_unproved_internal_retry_remains_hard(damage, api):
    value = internal_retry_example(api=api)
    failed, success = value['provider_calls'][:2]
    if damage == 'request': success['request_hash'] = contracts.hash_bytes(b'different')
    elif damage == 'body': success['request']['outbound']['body']['input'] = 'different'
    elif damage == 'base': success['base_call_identity'] = contracts.hash_bytes(b'different')
    elif damage == 'run': success['run_id'] = 'different'
    elif damage == 'attempt': success['provider_attempt'] = 1
    elif damage == 'identity': success['call_identity'] = contracts.hash_bytes(b'different')
    elif damage == 'future_attempt': failed['provider_attempt'] = 3
    elif damage == 'action': success['action_sequence'] += 1
    elif damage == 'terminal_order': success['terminal_entry_id'] = failed['terminal_entry_id']
    elif damage == 'reservation_order': success['reservation_entry_id'] = failed['terminal_entry_id']
    elif damage == 'failed_reservation_order': failed['reservation_entry_id'] = failed['terminal_entry_id']
    elif damage == 'success_reservation_order': success['reservation_entry_id'] = success['terminal_entry_id']
    elif damage == 'missing_success': value['provider_calls'].pop(1)
    elif damage == 'missing_worker': value['worker_observation_artifacts'] = {}
    elif damage == 'write': success['response_provenance']['delivery_to_worker'] = 'write_failed'
    elif damage == 'missing_upstream': failed['upstream_response_receipts'] = []
    elif damage == 'generated': failed['upstream_response_receipts'][0]['response']['output'] = [{'type': 'message'}]
    elif damage == 'private_upstream': failed['upstream_response_receipts'][0]['privacy_redactions'] = ['credential']
    elif damage == 'private_request': failed['request_capture']['redacted'] = True
    elif damage == 'private_response': failed['response_privacy_redactions'] = ['credential']
    elif damage == 'private_success': success['response_receipts'][0]['privacy_redactions'] = ['credential']
    elif damage == 'private_success_request': success['request_capture']['redacted'] = True
    elif damage == 'generated_terminal': failed['response']['output'] = [{'type': 'new_generated_output', 'text': 'answer'}]
    elif damage == 'missing_failure_evidence': failed['response_receipts'][0]['metadata'].pop('account_failure_evidence')
    elif damage == 'failure_evidence': failed['response_receipts'][0]['metadata']['account_failure_evidence']['provider_attempt'] = 2
    elif damage == 'uncertain_cost': failed['actual_microusd'] = 0
    elif damage == 'missing_cost': failed.pop('actual_microusd')
    elif damage == 'terminal_cost': failed['lifecycle'][0]['amount_microusd'] = 5
    elif damage == 'terminal_kind': failed['lifecycle'][0]['entry_kind'] = 'settlement'
    elif damage == 'terminal_status': failed['response_receipts'][0]['metadata']['status'] = 502
    elif damage == 'terminal_success': failed['response_receipts'][0]['metadata']['call_succeeded'] = True
    elif damage == 'missing_cost_gap': failed['evidence_gaps'] = []
    elif damage == 'known_cost_gap':
        failed.update(accounting_state='settlement', actual_microusd=7)
        failed['lifecycle'][0].update(entry_kind='settlement', amount_microusd=7)
        failed['response_receipts'][0]['entry_kind'] = 'settlement'
    else: value['configuration'].pop('training_maximum_missing_reasoning_percent')
    seal(value)
    assert not training_thinking.verified_failed_attempt(value, failed), damage


@pytest.mark.parametrize('api', ['openrouter.responses', 'openrouter.chat'])
def test_failure_status_claim_cannot_omit_a_proved_generated_target(api):
    value = internal_retry_example(api=api)
    success = value['provider_calls'][1]
    original = training_steps._candidate(value, success)
    success.update(response_status=402, accounting_state='uncertain', actual_microusd=None)
    seal(value)
    assert not training_thinking.verified_failed_attempt(value, success)
    assert training_steps._candidate(value, success)['target'] == original['target']


def test_later_action_free_retry_does_not_extend_to_unproved_chat_shape():
    value = recovered_example(api='openrouter.chat')
    assert not training_thinking.verified_failed_attempt(value, value['provider_calls'][0])


@pytest.mark.parametrize('api', ['openrouter.responses', 'openrouter.chat'])
@pytest.mark.parametrize('damage', [None, 'link', 'prior_amount', 'head_amount', 'extra_terminal'])
def test_later_billing_keeps_the_same_error_and_retry_proof(api, damage):
    value = internal_retry_example(api=api)
    failed = value['provider_calls'][0]
    failed.update(accounting_state='settlement', actual_microusd=7, terminal_entry_id=50,
                  response_entry_id=3, evidence_gaps=[])
    failed['response_receipts'][0]['terminal_source'] = 'entry_doc.call.openrouter_terminal_response'
    failed['lifecycle'].append({'entry_id': 50, 'entry_kind': 'settlement', 'amount_microusd': 7,
        'accounting': {'openrouter_delayed_reconciliation': True, 'reconciled_uncertainty_entry_id': 3}})
    value['completeness'].update(complete=True, incomplete_reasons=[])
    if damage == 'link': failed['lifecycle'][-1]['accounting']['reconciled_uncertainty_entry_id'] = 2
    elif damage == 'prior_amount': failed['lifecycle'][0]['amount_microusd'] = 1
    elif damage == 'head_amount': failed['lifecycle'][-1]['amount_microusd'] = 8
    elif damage == 'extra_terminal': failed['lifecycle'].append({'entry_id': 49, 'entry_kind': 'uncertain', 'amount_microusd': 0})
    seal(value)
    original = deepcopy(value)
    assert training_thinking.verified_failed_attempt(value, failed) is (damage is None)
    if damage is None:
        assert training_eligibility.assess_capture(submission(), {'run': value}, ['run'], enabled=True)['eligible']
    assert value == original


def unsettled_success_example(api):
    value = internal_retry_example(api=api)
    success = value['provider_calls'][1]
    success.update(accounting_state='uncertain', actual_microusd=None,
                   evidence_gaps=['provider_cost_unresolved'])
    success['lifecycle'][-1].update(entry_kind='uncertain', amount_microusd=0)
    success['response_receipts'][0]['entry_kind'] = 'uncertain'
    return seal(value)


@pytest.mark.parametrize('api', ['openrouter.responses', 'openrouter.chat'])
def test_uncertain_success_cost_does_not_veto_a_proved_internal_recovery(api):
    known = internal_retry_example(api=api)
    value = unsettled_success_example(api)
    original = deepcopy(value)
    failed, success = value['provider_calls'][:2]
    candidate = training_steps._candidate(value, success)
    assert candidate['target'] == training_steps._candidate(known, known['provider_calls'][1])['target']
    assert training_thinking.verified_failed_attempt(value, failed)
    result = training_eligibility.assess_capture(submission(), {'run': value}, ['run'], enabled=True)
    assert result['eligible'], result
    assert result['counts']['reasoning_by_run'][0]['verified_failed_model_attempts'] == 1
    linked = deepcopy(value)
    assert not training_export._link_worker_receipts(
        {'run_id': 'run', 'kind': 'execute'}, linked['provider_calls'],
        linked['worker_observation_artifacts']['run'], require_client_ack=False,
        configuration=value['configuration'])
    records, _ = training_thinking.build([value], training_steps.Rules(production_rounds_only=False))
    expected, _ = training_thinking.build([known], training_steps.Rules(production_rounds_only=False))
    fields = ('candidate_id', 'input', 'target', 'generated_thinking')
    assert [{key: row[key] for key in fields} for row in records['train'] + records['eval']] == [
        {key: row[key] for key in fields} for row in expected['train'] + expected['eval']]
    assert success['accounting_state'] == 'uncertain' and success['actual_microusd'] is None
    assert value == original


@pytest.mark.parametrize('api', ['openrouter.responses', 'openrouter.chat'])
@pytest.mark.parametrize('damage', ['missing_cost', 'known_cost', 'terminal_kind', 'terminal_amount',
    'missing_terminal', 'duplicate_terminal', 'missing_cost_gap', 'worker', 'privacy', 'request', 'route'])
def test_uncertain_success_still_requires_coherent_accounting_and_exact_delivery(api, damage):
    value = unsettled_success_example(api)
    failed, success = value['provider_calls'][:2]
    if damage == 'missing_cost': success.pop('actual_microusd')
    elif damage == 'known_cost': success['actual_microusd'] = 0
    elif damage == 'terminal_kind': success['lifecycle'][-1]['entry_kind'] = 'settlement'
    elif damage == 'terminal_amount': success['lifecycle'][-1]['amount_microusd'] = 7
    elif damage == 'missing_terminal': success['lifecycle'].pop()
    elif damage == 'duplicate_terminal': success['lifecycle'].append(deepcopy(success['lifecycle'][-1]))
    elif damage == 'missing_cost_gap': success['evidence_gaps'] = []
    elif damage == 'worker': success['response_provenance']['delivery_to_worker'] = 'write_failed'
    elif damage == 'privacy': success['response_receipts'][0]['privacy_redactions'] = ['credential']
    elif damage == 'request': success['request_hash'] = contracts.hash_bytes(b'different')
    else: success['reasoning_capture']['provider_route'] = 'other'
    seal(value)
    assert not training_thinking.verified_failed_attempt(value, failed)


def test_legacy_later_action_recovery_keeps_its_settled_success_requirement():
    value = recovered_example()
    failed, success = value['provider_calls'][:2]
    success.update(accounting_state='uncertain', actual_microusd=None,
                   evidence_gaps=['provider_cost_unresolved'])
    success['lifecycle'][-1].update(entry_kind='uncertain', amount_microusd=0)
    seal(value)
    training_steps._candidate(value, success)
    assert not training_thinking.verified_failed_attempt(value, failed)


@pytest.mark.parametrize('api', ['openrouter.responses', 'openrouter.chat'])
@pytest.mark.parametrize('damage', ['amount', 'kind', 'cost_gap'])
def test_settled_internal_success_cannot_claim_a_contradictory_accounting_head(api, damage):
    value = internal_retry_example(api=api)
    failed, success = value['provider_calls'][:2]
    if damage == 'amount': success['lifecycle'][-1]['amount_microusd'] += 1
    elif damage == 'kind': success['lifecycle'][-1]['entry_kind'] = 'uncertain'
    else: success['evidence_gaps'] = ['provider_cost_unresolved']
    seal(value)
    assert not training_thinking.verified_failed_attempt(value, failed)
