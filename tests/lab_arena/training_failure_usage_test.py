"""An error cannot hide generated work from capture completeness checks."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_thinking
from tests.lab_arena.training_call_semantics_test import _failed_example
from tests.lab_arena.training_eligibility_test import refresh_response_links
from tests.lab_arena.training_hidden_retry_test import recovered_example
from tests.lab_arena.training_host_result_test import without_client_callbacks
from tests.lab_arena.training_internal_retry_test import internal_retry_example
from tests.lab_arena.training_steps_test import seal


def failure_with_usage(path, usage):
    if path.startswith('internal'):
        api = 'openrouter.chat' if path.endswith('chat') else 'openrouter.responses'
        value = internal_retry_example(api=api)
    elif path == 'hidden':
        value = recovered_example()
    else:
        value = _failed_example(503)
    call = value['provider_calls'][0]
    if path in {'host', 'legacy'}:
        call['response']['usage'] = deepcopy(usage)
        refresh_response_links(value, 0)
        if path == 'host':
            value = without_client_callbacks(value)
    else:
        receipt = call['upstream_response_receipts'][0]
        receipt['response']['usage'] = deepcopy(usage)
        digest = contracts.document_hash(receipt['response'])
        size = len(contracts.canonical_json(receipt['response']).encode())
        receipt.update(response_sha256=digest, body_sha256=digest,
                       wire_body_sha256=digest, body_bytes=size, wire_body_bytes=size)
    return seal(value)


@pytest.mark.parametrize('path', ['internal_chat', 'internal_responses', 'hidden', 'host', 'legacy'])
@pytest.mark.parametrize('usage', [
    {'reasoning_tokens': 1}, {'output_tokens': 1}, {'completion_tokens': 1},
    {'completion_tokens': 0, 'completion_tokens_details': {'reasoning_tokens': 1}},
    {'output_tokens': 0, 'output_tokens_details': {'reasoning_tokens': 1}},
    {'completion_tokens_details': {'audio_tokens': 1}},
    {'output_tokens': -1}, {'output_tokens': '0'}, {'output_tokens': False},
    {'output_tokens': None}, {'output_tokens_details': []}, [],
])
def test_generated_or_ambiguous_usage_cannot_be_dropped_as_empty_failure(path, usage):
    value = failure_with_usage(path, usage)
    before = deepcopy(value)
    assert not training_thinking.verified_failed_attempt(value, value['provider_calls'][0])
    assert value == before


@pytest.mark.parametrize('path', ['internal_chat', 'internal_responses', 'hidden', 'host', 'legacy'])
@pytest.mark.parametrize('usage', [None, {}, {'input_tokens': 17, 'prompt_tokens': 17},
    {'output_tokens': 0, 'completion_tokens': 0, 'reasoning_tokens': 0},
    {'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0},
     'output_tokens_details': {'reasoning_tokens': 0}},
])
def test_missing_input_only_and_zero_generation_usage_preserve_failure_and_costs(path, usage):
    value = failure_with_usage(path, usage)
    before = deepcopy(value)
    assert training_thinking.verified_failed_attempt(value, value['provider_calls'][0])
    assert value == before
