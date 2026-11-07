"""Trusted judge citations keep their exact target and all existing proof gates."""
from copy import deepcopy

import pytest

from lab_arena import contracts, internal_capture, training_steps, training_thinking
from tests.lab_arena.judge_thinking_export_test import judge_example


def example():
    value = judge_example()
    call = value['provider_calls'][-1]
    call['response']['choices'][0]['message']['annotations'] = [{
        'type': 'url_citation', 'url_citation': {
            'url': 'https://example.com/source', 'title': 'Public source',
            'start_index': 0, 'end_index': 0}}]
    return bind_response(value)


def bind_response(value):
    call = value['provider_calls'][-1]
    wire = contracts.canonical_json(call['response']).encode()
    digest = contracts.hash_bytes(wire)
    call['response_sha256'] = contracts.document_hash(call['response'])
    receipt = call['response_receipts'][0]
    receipt.update(response_sha256=call['response_sha256'], wire_body_sha256=digest,
                   wire_body_bytes=len(wire))
    artifact = value['worker_observation_artifacts'][call['run_id']]
    artifact['payload']['response_receipts'][0].update(body_sha256=digest, body_bytes=len(wire))
    artifact['payload']['agent_observations'][0]['data'].update(body_sha256=digest, body_bytes=len(wire))
    artifact['payload_sha256'] = contracts.document_hash(artifact['payload'])
    for ref in call['response_provenance']['agent_receipt_refs'] + [call['response_provenance']['worker_receipt_ref']]:
        ref['payload_sha256'] = artifact['payload_sha256']
    value['example_hash'] = contracts.document_hash({k: v for k, v in value.items() if k != 'example_hash'})
    return value


def candidate(value):
    return training_steps._candidate(value, value['provider_calls'][-1], host_observed=True, role='judge')


def test_judge_without_declared_hosted_tools_keeps_exact_nested_citations():
    value = example()
    before = deepcopy(value)
    call = value['provider_calls'][-1]
    assert not call['request']['outbound']['body'].get('tools')
    record = candidate(value)
    assert record['target']['annotations'] == call['response']['choices'][0]['message']['annotations']
    assert record['source_refs'][0]['delivered_response_sha256'] == call['response_sha256']
    assert record['source_refs'][0]['judge_context']['scored_run_id'] == value['run_id']
    assert record['quality'] == {'status': 'unreviewed', 'correct_action': 'not_established'}
    assert value == before


def test_citations_do_not_grant_full_thinking_or_miner_or_internal_search_authority():
    value = example()
    call = value['provider_calls'][-1]
    call.pop('reasoning_capture')
    call['response']['choices'][0]['message'].pop('reasoning_content')
    bind_response(value)
    assert candidate(value)['target']['annotations']
    registry = value['configuration']['reasoning_route_registry']
    assert training_thinking.full_text_segments(call, 'judge', registry)[0] == []
    with pytest.raises(ValueError, match='assistant_shape_unsupported'):
        training_steps._target(call['request']['outbound']['body'], call['response'], 'openrouter.chat')
    with pytest.raises(ValueError, match='internal_chat_search_undeclared'):
        internal_capture._internal_chat_annotations(
            call['request']['outbound']['body'], call['response']['choices'][0]['message'])


def test_invalid_citation_is_a_dataset_exclusion_instead_of_an_export_failure():
    value = example()
    value['provider_calls'][-1]['response']['choices'][0]['message']['annotations'][0]['url_citation'].pop('title')
    bind_response(value)
    splits, manifest = training_steps._build([value], host_observed=True, role='judge')
    assert not splits['train'] and not splits['eval']
    assert manifest['counts']['exclusion_reasons']['chat_annotations_unsupported'] == 1


@pytest.mark.parametrize('damage', [
    'unknown_annotation', 'unknown_citation', 'annotation_type', 'annotation_count',
    'missing_title', 'title_type', 'url_type', 'non_https', 'private_url', 'url_userinfo',
    'offset_type', 'negative_offset', 'content_type', 'oversized_url', 'oversized_title',
    'oversized_content', 'tool_calls', 'top_level_citations',
])
def test_judge_invalid_citation_shapes_are_excluded(damage):
    value = example()
    message = value['provider_calls'][-1]['response']['choices'][0]['message']
    annotation = message['annotations'][0]
    citation = annotation['url_citation']
    if damage == 'unknown_annotation': annotation['extra'] = True
    elif damage == 'unknown_citation': citation['extra'] = True
    elif damage == 'annotation_type': annotation['type'] = 'reasoning.encrypted'
    elif damage == 'annotation_count': message['annotations'] *= 129
    elif damage == 'missing_title': citation.pop('title')
    elif damage == 'title_type': citation['title'] = 1
    elif damage == 'url_type': citation['url'] = 1
    elif damage == 'non_https': citation['url'] = 'http://example.com/'
    elif damage == 'private_url': citation['url'] = 'https://127.0.0.1/'
    elif damage == 'url_userinfo': citation['url'] = 'https://user:password@example.com/'
    elif damage == 'offset_type': citation['start_index'] = '0'
    elif damage == 'negative_offset': citation['end_index'] = -1
    elif damage == 'content_type': citation['content'] = {'text': 'untyped'}
    elif damage == 'oversized_url': citation['url'] = 'https://example.com/' + 'a' * 8192
    elif damage == 'oversized_title': citation['title'] = 'a' * 32001
    elif damage == 'oversized_content': citation['content'] = 'a' * 32001
    elif damage == 'top_level_citations': value['provider_calls'][-1]['response']['citations'] = ['https://example.com/']
    else: message['tool_calls'] = [{'id': 'undeclared'}]
    bind_response(value)
    with pytest.raises(ValueError):
        candidate(value)


@pytest.mark.parametrize('damage', [
    'request_privacy', 'response_privacy', 'model_policy', 'score_status',
    'score_identity', 'request_hash', 'response_hash', 'receipt_hash',
])
def test_judge_annotations_do_not_bypass_pair_proof(damage):
    value = example()
    call = value['provider_calls'][-1]
    if damage == 'request_privacy': call['request_capture'] = {'redacted': True}
    elif damage == 'response_privacy': call['response_privacy_redactions'] = ['credential']
    elif damage == 'model_policy': value['configuration']['scorer_policy']['judge_models'] = {}
    elif damage == 'score_status': value['attempts'][-1]['status'] = 'failed'
    elif damage == 'score_identity': value['score_run_id'] = 'foreign-score'
    elif damage == 'request_hash': call['request_hash'] = 'sha256:' + 'a' * 64
    elif damage == 'response_hash': call['response_receipts'][0]['response_sha256'] = 'sha256:' + 'b' * 64
    else: call['response_receipts'][0]['wire_body_sha256'] = 'sha256:' + 'c' * 64
    with pytest.raises(ValueError):
        candidate(value)
