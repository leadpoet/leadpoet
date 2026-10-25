"""Saved route-unavailable responses remain failures, never model targets."""

import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lab_arena import contracts, training_eligibility, training_thinking
from tests.lab_arena.training_call_semantics_test import _failed_example
from tests.lab_arena.training_eligibility_test import (
    captured_example, refresh_response_links, submission, ten_selected_examples,
)
from tests.lab_arena.training_steps_test import seal

FIXTURES = Path(__file__).parents[1] / 'fixtures/full_thinking_provider_unavailable'


def saved(size):
    return json.loads((FIXTURES / ('gpt-oss-' + size + '.json')).read_text())


def selected_context(probe):
    """Keep real failed call proof; supply the existing scored-context fixture."""
    original = captured_example()
    run_id = probe['run_id']

    def rename(value):
        if isinstance(value, dict):
            return {run_id if key == 'run' else key: rename(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rename(child) for child in value]
        return run_id if value == 'run' else value

    example = rename(original)
    example.update(provider_calls=deepcopy(probe['provider_calls']),
                   worker_observation_artifacts=deepcopy(probe['worker_observation_artifacts']),
                   configuration={**deepcopy(probe['configuration']),
                                  'training_maximum_missing_reasoning_percent': 5})
    return seal(example)


@pytest.mark.parametrize('size', ['120b', '20b'])
def test_saved_ignored_provider_404_is_verified_but_never_complete(size):
    fixture = saved(size)
    probe = fixture['example']
    call = probe['provider_calls'][0]
    raw = base64.b64decode(fixture['wire_body_b64'], validate=True)
    assert json.loads(raw) == call['response']
    receipt = call['response_receipts'][0]
    assert receipt['wire_body_sha256'] == contracts.hash_bytes(raw)
    assert receipt['wire_body_bytes'] == len(raw)
    assert training_thinking.verified_failed_attempt(probe, call) is True

    example = selected_context(probe)
    examples = ten_selected_examples()
    for other in examples.values():
        other['configuration']['training_maximum_missing_reasoning_percent'] = 5
        seal(other)
    examples.pop('run-9')
    examples[example['run_id']] = example
    result = training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=5)
    assert result['eligible'] is True, result
    assert result['counts']['complete_runs'] == 9
    assert result['counts']['incomplete_runs'] == 1
    failed_run = next(row for row in result['counts']['reasoning_by_run']
                      if row['run_id'] == example['run_id'])
    assert failed_run['verified_failed_model_attempts'] == 1
    assert failed_run['completed_untruncated_verified_reasoning_tokens'] == 0
    strict = training_eligibility.assess_capture(
        submission(), {example['run_id']: example}, [example['run_id']], enabled=True,
        maximum_missing_reasoning_percent=5)
    assert strict['eligible'] is False
    assert 'full_thinking_call_missing' in strict['reasons']
    from lab_arena import training_steps
    splits, _ = training_thinking.build(
        [example], training_steps.Rules(production_rounds_only=False))
    assert not splits['train'] and not splits['eval']


@pytest.mark.parametrize('damage', ['old404', 'wrong_model', 'request_hash', 'wire',
                                   'ack', 'provider_pin', 'funnel', 'endpoint', 'reported_model', 'generation'])
def test_404_allowance_keeps_other_errors_and_tamper_hard(damage):
    example = selected_context(saved('120b')['example'])
    call = example['provider_calls'][0]
    if damage == 'old404':
        call['response']['error']['message'] = 'No endpoints found for the model'
        refresh_response_links(example, 0)
    elif damage == 'wrong_model':
        call['request']['outbound']['body']['model'] = 'openai/not-reviewed'
    elif damage == 'request_hash':
        call['request']['request_hash'] = contracts.hash_bytes(b'wrong')
    elif damage == 'wire':
        call['response_receipts'][0]['wire_body_bytes'] += 1
    elif damage == 'ack':
        call['response_provenance']['agent_receipt'] = 'not_reported'
    elif damage == 'provider_pin':
        call['request']['outbound']['body']['provider']['only'] = ['wrong-provider']
        call['request']['outbound']['body_sha256'] = contracts.document_hash(
            call['request']['outbound']['body'])
    elif damage == 'funnel':
        call['response']['error']['metadata']['routing_funnel'][-1]['endpoint_count'] = 0
        refresh_response_links(example, 0)
    elif damage == 'endpoint':
        for endpoint in call['response']['openrouter_metadata']['endpoints']['available']:
            if endpoint['provider'] == 'DeepInfra':
                endpoint['model'] = 'openai/wrong-model'
        refresh_response_links(example, 0)
    elif damage == 'reported_model':
        call['response']['openrouter_metadata']['requested'] = 'openai/wrong-model'
        refresh_response_links(example, 0)
    else:
        call['response']['choices'] = [{'message': {'role': 'assistant', 'content': 'A target'}}]
        refresh_response_links(example, 0)
    seal(example)
    assert training_thinking.verified_failed_attempt(example, call) is False
    result = training_eligibility.assess_capture(
        submission(), {example['run_id']: example}, [example['run_id']], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=5)
    assert result['eligible'] is False


def test_generic_404_is_not_a_verified_retryable_failure():
    example = _failed_example(404)
    assert training_thinking.verified_failed_attempt(example, example['provider_calls'][0]) is False
