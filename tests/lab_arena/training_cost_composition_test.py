"""Billing and judge diagnostics cannot waive or invalidate exact selected data."""
from copy import deepcopy

import pytest

from lab_arena import training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import (
    _unsettled_selected_call, assess_ten, ten_selected_examples,
)
from tests.lab_arena.training_steps_test import seal


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


def test_exact_selected_data_remains_usable_with_unknown_bill_and_private_judge():
    examples = ten_selected_examples()
    example = _judge_privacy(_unsettled_selected_call(examples['run-9']))
    before = deepcopy(example)
    result = assess_ten(examples, percent=100)
    assert result['eligible'], result
    assert example['provider_calls'][0]['actual_microusd'] is None
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(splits['train']) + len(splits['eval']) == 2
    assert example == before


@pytest.mark.parametrize('damage', ['response_privacy', 'receipt_privacy', 'request_hash', 'score_input'])
def test_billing_and_judge_gaps_cannot_hide_invalid_selected_data(damage):
    examples = ten_selected_examples()
    example = _judge_privacy(_unsettled_selected_call(examples['run-9']))
    call = example['provider_calls'][1]
    if damage == 'response_privacy':
        call['response_privacy_redactions'] = ['credential']
    elif damage == 'receipt_privacy':
        call['response_receipts'][0]['privacy_redactions'] = ['credential']
    elif damage == 'request_hash':
        call['request']['request_hash'] = 'sha256:' + '0' * 64
    else:
        example['input_snapshots'][example['score_run_id']]['input']['companies'] = ['wrong']
    seal(example)
    result = assess_ten(examples)
    assert result['eligible'] and result['counts']['complete_runs'] == 9
    assert result['counts']['rejected_runs'][0]['run_id'] == 'run-9'
    assert not assess_ten(examples, percent=100)['eligible']
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    assert len(records) == (0 if damage == 'score_input' else 1)
    assert all(ref['call_identity'] != call['call_identity'] for row in records for ref in row['source_refs'])
