"""Retained Responses citation evidence must not be lost at internal export."""
from copy import deepcopy
import json

import pytest

from lab_arena import internal_capture, operations, training_steps
from lab_arena.service import LocalObjectStore
from tests.lab_arena.internal_capture_responses_test import export, request, response


def cited_response(uncited=True):
    value = response()
    citation = value['output'][-1]['content'][0]['annotations'][0]
    citation.update(uncited=uncited, content='Original provider source excerpt.')
    return value


@pytest.mark.parametrize('uncited', [True, False])
def test_internal_export_keeps_exact_uncited_metadata_and_source_content(tmp_path, uncited):
    provider = cited_response(uncited)
    before = deepcopy(provider)
    raw, candidate = export(tmp_path, provider=provider)
    assert candidate is not None
    assert candidate['target'] == provider['output'] == before['output']
    assert raw['response'] == before
    assert raw['sft_exclusions'] == []
    assert len(candidate['available_reasoning']) == 3
    assert candidate['quality'] == {'status': 'unreviewed', 'correct_action': 'not_established'}
    evidence = raw['research_evidence']['responses_url_citations']
    assert evidence == candidate['source']['research_evidence']['responses_url_citations']
    assert evidence['references'] == evidence['annotations_with_content'] == 1
    assert evidence['content_characters'] == len('Original provider source excerpt.')
    assert evidence['items'][0]['content_path'] == '/output/5/content/0/annotations/0/content'
    assert evidence['items'][0]['uncited'] is uncited
    assert raw['research_evidence']['underlying_page_contents_verified'] is False
    assert raw['research_evidence']['all_internal_provider_turns_verified'] is False


@pytest.mark.parametrize('damage', ['uncited_string', 'uncited_integer', 'uncited_null',
    'unknown_field', 'missing_url', 'bad_content', 'unsafe_url', 'annotation_count'])
def test_uncited_support_does_not_accept_malformed_evidence(tmp_path, damage):
    provider = cited_response()
    annotations = provider['output'][-1]['content'][0]['annotations']
    citation = annotations[0]
    if damage == 'uncited_string': citation['uncited'] = 'true'
    elif damage == 'uncited_integer': citation['uncited'] = 1
    elif damage == 'uncited_null': citation['uncited'] = None
    elif damage == 'unknown_field': citation['unexpected'] = 'value'
    elif damage == 'missing_url': citation.pop('url')
    elif damage == 'bad_content': citation['content'] = ['not text']
    elif damage == 'unsafe_url': citation['url'] = 'file:///private/file'
    else: annotations *= 129
    raw, candidate = export(tmp_path, provider=provider)
    assert candidate is None
    assert raw['sft_exclusions'] == ['target_shape_or_completion_unsupported']


def test_provider_output_support_does_not_change_miner_input_contract():
    body, provider = request(), cited_response()
    body['max_tool_calls'] = 1
    body['tools'][0]['parameters'] = {'engine': 'exa', 'max_uses': 1, 'max_total_results': 5}
    target, _ = training_steps._target(body, provider, 'openrouter.responses', allow_reasoning_history=True)
    assert target == provider['output']
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request('openrouter.responses',
            {'model': body['model'], 'input': [provider['output'][-1]]}, allow_full_thinking=True)


def test_internal_uncited_metadata_cannot_authorize_undeclared_search(tmp_path):
    body, provider = request(), cited_response()
    body.pop('tools')
    body.pop('max_tool_calls')
    raw, candidate = export(tmp_path, body=body, provider=provider)
    assert candidate is None
    assert raw['sft_exclusions'] == ['target_shape_or_completion_unsupported']


def test_multiple_interleaved_messages_keep_original_annotation_paths(tmp_path):
    provider = cited_response()
    intermediate = deepcopy(provider['output'][-1])
    intermediate['id'] = 'intermediate'
    intermediate['phase'] = 'commentary'
    intermediate['content'][0]['annotations'][0].pop('uncited')
    provider['output'].insert(2, intermediate)
    provider['output'][-1]['content'][0]['annotations'].append({
        'type': 'url_citation', 'url': 'https://example.com/second', 'title': 'URL only'})
    before = deepcopy(provider)
    raw, candidate = export(tmp_path, provider=provider)
    assert candidate['target'] == before['output']
    citations = raw['research_evidence']['responses_url_citations']
    assert citations['references'] == 3 and citations['annotations_with_content'] == 2
    assert [r['annotation_path'] for r in citations['items']] == [
        '/output/2/content/0/annotations/0', '/output/6/content/0/annotations/0',
        '/output/6/content/0/annotations/1']
    assert [r['uncited'] for r in citations['items']] == [None, True, None]
    assert provider == before


def test_private_citation_content_stays_redacted_and_excluded(tmp_path):
    objects = LocalObjectStore(tmp_path)
    provider = cited_response()
    provider['output'][-1]['content'][0]['annotations'][0]['content'] = 'fixture-private-token'
    call = internal_capture.InternalCapture(objects).start(
        purpose='icp_exclusion', context={}, request=request(), api='openrouter.responses',
        secrets=('fixture-private-token',))
    call.response(body=json.dumps(provider).encode(), status=200)
    call.finish(outcome={'status': 'fixture'}, cost_microusd=1)
    raw, candidate = internal_capture.read_call(objects, call.prefix + 'request.json')
    assert raw['response']['output'][-1]['content'][0]['annotations'][0]['content'] == '[REDACTED]'
    assert raw['research_evidence']['responses_url_citations']['annotations_with_content'] == 0
    assert candidate is None and raw['sft_exclusions'] == ['privacy_nonexact']


@pytest.mark.parametrize('text,expected', [('', 0), ('[REDACTED]', 0),
    (' [REDACTED]\n', 0), ('Retained excerpt.', 1)])
def test_responses_citation_summary_counts_only_retained_text(text, expected):
    provider = cited_response()
    provider['output'][-1]['content'][0]['annotations'][0]['content'] = text
    before = deepcopy(provider)
    evidence = internal_capture._search_evidence(provider)['responses_url_citations']
    assert evidence['references'] == 1
    assert evidence['annotations_with_content'] == expected
    assert evidence['content_characters'] == (len(text) if expected else 0)
    assert (evidence['items'][0]['content_path'] is not None) is bool(expected)
    assert provider == before
