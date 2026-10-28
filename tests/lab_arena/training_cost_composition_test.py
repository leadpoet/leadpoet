"""An ancillary judge gap must not turn one bounded cost gap into a hard failure."""
from copy import deepcopy

import pytest

from lab_arena import training_eligibility, training_export, training_thinking
from tests.lab_arena.training_eligibility_test import (
    _unsettled_selected_call, assess_ten, ten_selected_examples,
)
from tests.lab_arena.training_steps_test import seal
from tests.lab_arena.training_optional_undispatched_composition_test import optional_undispatched, relink, assert_no_targets
from tests.lab_arena.training_export_test import per_icp_cost_reader
from tests.lab_arena.training_undispatched_coverage_test import assess_bank, undispatched_values


def _judge_privacy(example):
    example['provider_calls'].append({
        'run_id': example['score_run_id'], 'operation_id': 'scrapingdog.scrape',
        'provider': 'scrapingdog', 'evidence_gaps': ['response_privacy_nonexact'],
        'response_privacy_redactions': ['credential'],
    })
    proof = example['completeness']
    proof['incomplete_reasons'].append('response_privacy_nonexact')
    proof['incomplete_reasons'].sort()
    proof['reasons_by_dimension']['research_process'].append('response_privacy_nonexact')
    proof['reasons_by_dimension']['research_process'].sort()
    return seal(example)


def test_bound_judge_privacy_and_selected_unknown_bill_consume_one_icp():
    examples = ten_selected_examples()
    damaged = _judge_privacy(_unsettled_selected_call(examples['run-9']))
    before = deepcopy(damaged)
    result = assess_ten(examples)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9
    assert result['counts']['incomplete_runs'] == 1
    assert damaged['provider_calls'][0]['actual_microusd'] is None
    assert training_thinking.selected_research_evidence(damaged) is None
    assert not assess_ten(examples, percent=100)['eligible']
    assert damaged == before


@pytest.mark.parametrize('damage', ['selected', 'foreign', 'unbound', 'unsubstantiated'])
def test_cost_and_judge_privacy_cannot_hide_unbound_or_selected_gaps(damage):
    examples = ten_selected_examples()
    damaged = _judge_privacy(_unsettled_selected_call(examples['run-9']))
    judge = damaged['provider_calls'][-1]
    if damage == 'selected':
        judge['run_id'] = damaged['run_id']
    elif damage == 'foreign':
        judge['run_id'] = 'foreign'
    elif damage == 'unbound':
        damaged['input_snapshots'][damaged['score_run_id']]['input']['companies'] = ['wrong']
    else:
        judge['response_privacy_redactions'] = []
    seal(damaged)
    result = assess_ten(examples)
    assert not result['eligible'], result
    assert 'trajectory_evidence_incomplete' in result['reasons']


@pytest.mark.parametrize('damage', [
    'response_credential', 'receipt_credential', 'both_credential',
    'response_unknown', 'receipt_unknown',
])
def test_judge_privacy_cannot_authorize_unreported_selected_redaction(damage):
    examples = ten_selected_examples()
    damaged = _judge_privacy(_unsettled_selected_call(examples['run-9']))
    selected = damaged['provider_calls'][0]
    if damage in {'response_credential', 'both_credential', 'response_unknown'}:
        selected['response_privacy_redactions'] = ['unknown' if damage.endswith('unknown') else 'credential']
    if damage in {'receipt_credential', 'both_credential', 'receipt_unknown'}:
        selected['response_receipts'][0]['privacy_redactions'] = [
            'unknown' if damage.endswith('unknown') else 'credential']
    seal(damaged)
    result = assess_ten(examples)
    assert not result['eligible'], result
    assert 'trajectory_evidence_incomplete' in result['reasons']


@pytest.mark.parametrize('operation', ['deepline.execute', 'openrouter.chat', 'openrouter.responses'])
def test_proven_nondispatch_composes_with_unknown_bill(operation):
    values = undispatched_values(operation=operation, complete=True)
    example = training_export.assemble_example(**values)
    nondispatched = example['provider_calls'][-1]
    assert training_thinking.verified_undispatched_transport(example, nondispatched)
    _unsettled_selected_call(example)
    result = assess_bank(values, example)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9
    assert result['counts']['incomplete_runs'] == 1
    assert nondispatched['actual_microusd'] == 0 and 'response' not in nondispatched


def test_unproven_nondispatch_stays_hard_in_cost_composition():
    values = undispatched_values(operation='openrouter.responses', complete=True)
    example = training_export.assemble_example(**values)
    _unsettled_selected_call(example)
    example['provider_calls'][-1]['lifecycle'][-1]['accounting']['reason'] = 'unknown'
    seal(example)
    assert not assess_bank(values, example)['eligible']


def test_nondispatch_unknown_bill_and_bound_judge_privacy_share_one_icp():
    values = undispatched_values(operation='openrouter.responses', complete=True)
    example = training_export.assemble_example(**values)
    _judge_privacy(_unsettled_selected_call(example))
    before = deepcopy(example)
    result = assess_bank(values, example)
    assert result['eligible'], result
    assert result['counts']['complete_runs'] == 9
    assert result['counts']['incomplete_runs'] == 1
    assert not assess_bank(values, example, percent=100)['eligible']
    assert example == before


@pytest.mark.parametrize('category', ['web_fetch', 'response_received'])
@pytest.mark.parametrize('damage', [None, 'unproved_nondispatch', 'dispatch', 'unknown_nondispatch_bill',
                                  'call_privacy', 'receipt_privacy', 'unknown_privacy', 'missing_receipt'])
def test_optional_nondispatch_and_separate_unknown_bill_keep_full_proof(category, damage):
    values, _ = optional_undispatched(category)
    # A separate dispatched call has a retained response but an unknown bill.
    terminal = next(row for row in values['ledger_rows'] if row['entry_kind'] == 'settlement')
    terminal.update(entry_kind='uncertain', amount_microusd=0)
    values['cost_eligibility_reader'] = per_icp_cost_reader(spend=43, reason='provider_cost_uncertain')
    example = training_export.assemble_example(**values)
    selected, nondispatched = example['provider_calls'][0], example['provider_calls'][-1]
    if damage == 'unproved_nondispatch':
        nondispatched['lifecycle'][-1]['accounting']['reason'] = 'unknown'
    elif damage == 'dispatch':
        nondispatched['lifecycle'].insert(1, {**deepcopy(nondispatched['lifecycle'][0]), 'entry_kind': 'dispatch'})
    elif damage == 'unknown_nondispatch_bill':
        nondispatched['actual_microusd'] = None
    elif damage == 'call_privacy':
        selected['response_privacy_redactions'] = ['credential']
    elif damage in {'receipt_privacy', 'unknown_privacy'}:
        selected['response_receipts'][0]['privacy_redactions'] = [
            'unknown' if damage == 'unknown_privacy' else 'credential']
    elif damage == 'missing_receipt':
        payload = example['worker_observation_artifacts'][example['run_id']]['payload']
        payload['response_receipts'].pop(0)
        payload['response_receipt_count'] -= 1
    relink(example)
    before = deepcopy(example)
    result = assess_bank(values, example)
    assert result['eligible'] is (damage is None), result
    assert result['counts']['complete_runs'] == 9
    assert result['counts']['incomplete_runs'] == 1
    assert not assess_bank(values, example, percent=100)['eligible']
    assert selected['actual_microusd'] is None
    assert_no_targets(example)
    assert example == before
