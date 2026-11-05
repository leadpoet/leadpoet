"""Stored JSON key order must not replace the hash of the actual wire bytes."""
from copy import deepcopy
import json

import pytest

from lab_arena import contracts, operations, training_capture, training_dataset, training_steps, training_thinking
from tests.lab_arena.test_industry_schema_order import request, wires
from tests.lab_arena.training_steps_test import fixture, seal


def industry_wire(body):
    # Reference the established protocol: canonical JSON except for the
    # industry schema's top-level properties, ordered by its required list.
    ordered = json.loads(contracts.canonical_json(body))
    schema = ordered['response_format']['json_schema']['schema']
    schema['properties'] = {key: schema['properties'][key] for key in schema['required']}
    return json.dumps(ordered, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def captured_industry_example(model='mistralai/mistral-large-4-0', *, canonical=False):
    value = fixture()
    call = value['provider_calls'][0]
    parameters = request()
    body = json.loads(wires(parameters)[1])
    body['model'] = parameters['model'] = model
    parameters['reasoning'] = body['reasoning'] = {'effort': 'low'}
    digest = contracts.document_hash(parameters)
    captured = {**call['request'], 'request_hash': digest, 'hashed_parameters': parameters,
                'effective_parameters': deepcopy(parameters),
                'outbound': {**call['request']['outbound'], 'body': body,
                             'body_sha256': contracts.document_hash(body) if canonical else contracts.hash_bytes(industry_wire(body))}}
    cleaned, redactions = training_capture.sanitize_training_value(captured)
    assert redactions == []
    # Private capture storage canonicalizes object keys while retaining the
    # original wire digest, as in both saved native industry probes.
    call['request'] = json.loads(contracts.canonical_json(cleaned))
    call['request_hash'] = digest
    return seal(value)


@pytest.mark.parametrize('model', ['mistralai/mistral-large-4-0', 'xiaomi/mimo-v2.6-flash', 'openai/gpt-6.1-sol'])
@pytest.mark.parametrize('canonical', [False, True])
def test_captured_industry_wire_is_exact_without_changing_saved_body(model, canonical):
    example = captured_industry_example(model, canonical=canonical)
    before = deepcopy(example)
    call = example['provider_calls'][0]
    outbound = call['request']['outbound']
    assert (contracts.document_hash(outbound['body']) == outbound['body_sha256']) is canonical
    assert training_steps._outbound_request_exact(call)
    candidate = training_steps._candidate(example, call)
    assert candidate['input'] == outbound['body']
    assert training_dataset._split_model_request_links(example) == ['model-input:openrouter.chat:' + outbound['body_sha256']]
    assert example == before


@pytest.mark.parametrize('damage', ['hash', 'message', 'required', 'schema_name', 'strict', 'arbitrary_order', 'route'])
def test_industry_wire_still_rejects_changed_evidence(damage):
    example = captured_industry_example()
    call = example['provider_calls'][0]
    outbound = call['request']['outbound']
    body = outbound['body']
    schema = body['response_format']['json_schema']['schema']
    if damage == 'hash': outbound['body_sha256'] = contracts.hash_bytes(b'wrong')
    elif damage == 'message': body['messages'][0]['content'] += ' changed'
    elif damage == 'required': schema['required'].reverse()
    elif damage == 'schema_name': body['response_format']['json_schema']['name'] = 'another_verifier'
    elif damage == 'strict': body['response_format']['json_schema']['strict'] = False
    elif damage == 'arbitrary_order':
        shuffled = dict(reversed(list(body.items())))
        outbound['body_sha256'] = contracts.hash_bytes(json.dumps(shuffled, separators=(',', ':'), ensure_ascii=False).encode())
    else: outbound['path'] = '/wrong'
    assert not training_steps._outbound_request_exact(call)
    assert training_dataset._split_model_request_links(example) == []
    with pytest.raises(ValueError, match='outbound_(body_hash|identity)_mismatch'):
        training_steps._candidate(seal(example), call)


def bind_industry_request(call):
    schema = request()['response_format']
    for field in ('hashed_parameters', 'effective_parameters', 'admitted_parameters'):
        call['request'][field]['response_format'] = deepcopy(schema)
    body = call['request']['outbound']['body']
    body['response_format'] = deepcopy(schema)
    call['request_hash'] = call['request']['request_hash'] = contracts.document_hash(call['request']['hashed_parameters'])
    call['request']['outbound']['body_sha256'] = contracts.hash_bytes(industry_wire(body))


def test_industry_wire_preserves_recovered_retry_proof_and_rejects_tampering():
    from tests.lab_arena.training_internal_retry_test import internal_retry_example
    example = internal_retry_example(api='openrouter.chat')
    failed, success = example['provider_calls'][:2]
    for call in (failed, success): bind_industry_request(call)
    seal(example)
    assert training_thinking.verified_recovered_provider_retry(example, failed)
    assert training_thinking.verified_failed_attempt(example, failed)
    failed['request']['outbound']['body']['messages'][0]['content'] += ' changed'
    assert not training_thinking.verified_recovered_provider_retry(example, failed)
    assert not training_thinking.verified_failed_attempt(example, failed)


def test_industry_wire_preserves_failed_attempt_proof_and_rejects_tampering():
    from tests.lab_arena.training_credit_refusal_test import credit_example
    from tests.lab_arena.training_failed_delivery_coverage_test import refresh
    example = credit_example()
    failed = example['provider_calls'][0]
    bind_industry_request(failed)
    outbound = failed['request']['outbound']
    wire_hash = outbound['body_sha256']
    outbound['body_sha256'] = contracts.document_hash(outbound['body'])
    refresh(example)
    assert training_thinking.verified_failed_attempt(example, failed)
    outbound['body_sha256'] = wire_hash
    seal(example)
    assert training_thinking.verified_failed_attempt(example, failed)
    failed['request']['outbound']['body_sha256'] = contracts.hash_bytes(b'wrong')
    assert not training_thinking.verified_failed_attempt(example, failed)


def test_ordinary_canonical_capture_remains_exact():
    example = fixture()
    assert training_steps._outbound_request_exact(example['provider_calls'][0])
    assert training_steps._candidate(example, example['provider_calls'][0])
    assert len(training_dataset._split_model_request_links(example)) == 1


def test_industry_serialization_is_not_another_api_or_arbitrary_json_order():
    call = captured_industry_example()['provider_calls'][0]
    outbound = call['request']['outbound']
    assert not operations.outbound_body_hash_matches('openrouter.responses', outbound['body'], outbound['body_sha256'])
    assert operations.outbound_body_hash_matches('exa.search', None, contracts.hash_bytes(b''))
    assert not operations.outbound_body_hash_matches('exa.search', None, contracts.document_hash(None))
