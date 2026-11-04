"""Empty credential query values contain no private bytes to remove."""
import base64
import copy
import json
from urllib.parse import quote

import pytest

from lab_arena import contracts, training_capture as capture, training_export, training_steps
from lab_arena.service import LocalObjectStore
from tests.lab_arena.test_lab_arena_broker import CONTEXT, HOST_KEYS, FakeTransport, make_broker
from tests.lab_arena.training_steps_test import seal
from tests.lab_arena.training_usable_evidence_test import usable_example, check, exported


@pytest.mark.parametrize('query', [
    'auth=', 'auth', 'token=', 'api_key=', 'password=', 'X-Amz-Signature=',
    'key=&token=&auth=&keep=a%20b', '%61uth=&keep=a+b', 'auth=&auth=&keep=1',
])
@pytest.mark.parametrize('embedded', [False, True])
def test_empty_sensitive_query_keeps_exact_public_text(query, embedded):
    url = 'https://example.test/source?' + query + '#public'
    value = json.dumps({'source': url}) if embedded else url
    assert capture.sanitize_training_value(value) == (value, [])
    assert capture.sanitize_training_value({'provider_calls': [{'response': value}]}, aggregate_parts=True) == (
        {'provider_calls': [{'response': value}]}, [])
    assert training_steps._exact(value)


def test_empty_query_request_capture_matches_unchanged_broker_dispatch(tmp_path):
    objects = LocalObjectStore(tmp_path)
    provider, store, transport = make_broker(
        capture_request=capture.RequestCapture(objects).prepare,
        transport=FakeTransport([(200, {'usage': {'cost': '0.000001'}})]))
    parameters = {'model': 'openai/gpt-4o-mini', 'messages': [{
        'role': 'user', 'content': 'Read https://example.test/source?auth=&token=&keep=a%20b'}]}
    original = copy.deepcopy(parameters)
    result = provider.execute(CONTEXT, operation_id='openrouter.chat', parameters=parameters,
                              action_sequence=1, timeout_ms=5000)
    assert result.status == 200
    assert parameters == original
    receipt = next(iter(store.calls.values()))
    pointer = receipt['call_doc']['request_capture']
    saved = capture.read_request(objects, pointer)
    wire = transport.sent[0]['body']
    assert saved['outbound']['body'] == json.loads(wire)
    assert saved['outbound']['body']['messages'] == parameters['messages']
    assert saved['outbound']['body_sha256'] == contracts.hash_bytes(wire)
    assert contracts.document_hash(saved['hashed_parameters']) == saved['request_hash']
    assert pointer['redacted'] is False and pointer['redactions'] == []
    assert saved['privacy'] == {'redacted': False, 'redactions': []}
    assert training_steps._exact(saved)
    assert not any(secret in json.dumps(saved) for secret in HOST_KEYS.values())


def test_empty_query_provider_response_remains_exact_at_export():
    response = {'choices': [{'message': {'role': 'assistant', 'content':
        'Read https://example.test/source?token=&auth='}, 'finish_reason': 'stop'}]}
    wire = contracts.canonical_json(response).encode()
    projected, redactions = training_export._response_value({'body_b64': base64.b64encode(wire).decode()}, 'openrouter')
    assert projected == response and redactions == []
    assert contracts.document_hash(projected) == contracts.document_hash(response)
    assert training_steps._exact(projected, response=True)


def test_empty_query_capture_keeps_native_icp_and_individual_pairs_usable(tmp_path):
    example = usable_example()
    call = example['provider_calls'][1]
    request = call['request']
    body = request['outbound']['body']
    body['messages'][0]['content'] = 'Read https://example.test/source?auth=&keep=a%20b'
    request['outbound']['body_sha256'] = contracts.document_hash(body)
    original_request = copy.deepcopy(request)
    objects = LocalObjectStore(tmp_path)
    pointer = capture.RequestCapture(objects)(request)
    call.update(request=capture.read_request(objects, pointer), request_capture=pointer)
    if pointer['redacted']:
        call['evidence_gaps'] = ['request_privacy_nonexact']
        proof = example['completeness']
        proof['complete'] = False
        proof['incomplete_reasons'].append('request_privacy_nonexact')
        proof['dimensions']['research_process'] = False
        proof['reasons_by_dimension']['research_process'].append('request_privacy_nonexact')
    seal(example)
    assert pointer['redacted'] is False
    assert call['request']['outbound'] == original_request['outbound']
    assert training_steps._candidate(example, call)['input'] == body
    assert check(example)['eligible']
    assert len(exported(example)) == 2


@pytest.mark.parametrize('query', [
    'auth=actual-secret', 'token=actual-secret', 'api_key=actual-secret',
    'X-Amz-Signature=actual-secret', 'auth=%20', 'auth=+', 'auth=%00',
    'auth=&token=actual-secret', 'auth=actual-secret&auth=',
])
def test_nonempty_credential_query_stays_private(query):
    value = 'https://example.test/source?' + query
    cleaned, reasons = capture.sanitize_training_value(value)
    assert reasons == ['credential'] and cleaned != value
    assert 'actual-secret' not in cleaned
    assert capture.sanitize_training_value(cleaned)[0] == cleaned


@pytest.mark.parametrize('private_context', [
    'Authorization: Bearer actual-secret-token ',
    'password=',
    'Authorization: Basic dXNlcjpwYXNz ',
])
def test_empty_query_does_not_bypass_surrounding_credentials(private_context):
    value = private_context + 'https://example.test/source?auth='
    cleaned, reasons = capture.sanitize_training_value(value)
    assert reasons == ['credential']
    assert 'actual-secret-token' not in cleaned and 'dXNlcjpwYXNz' not in cleaned
    if private_context == 'password=':
        assert 'https://' not in cleaned


def test_empty_query_does_not_bypass_known_secret_or_userinfo():
    secret = 'known-private-value'
    for value in ['https://example.test/' + secret + '?auth=',
                  'https://example.test/source?auth=&keep=' + quote(secret),
                  'https://private-user:private-password@example.test/source?auth=']:
        cleaned, reasons = capture.sanitize_training_value(value, secrets=(secret,))
        assert reasons == ['credential']
        assert secret not in cleaned and 'private-user' not in cleaned and 'private-password' not in cleaned
