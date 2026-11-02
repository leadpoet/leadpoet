"""One complete alternate review after a typed primary refusal; no paid calls."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from lab_arena import broker, code_review, code_review_policy
from lab_arena.code_review_runtime import SubmissionCodeReviewer
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    MINER_KEY, Objects, ReviewStore, ReviewTransport, SUBMISSION,
    _archive, _price_table, _review_body, _source,
)


def fallback_prices():
    prices = _price_table()
    prices['models'][code_review.FALLBACK_REVIEW_MODEL] = dict(prices['models'][code_review.DEFAULT_REVIEW_MODEL])
    return prices


def fallback_row(payload):
    return {
        **SUBMISSION, 'source_size_bytes': len(payload), 'code_review_status': 'error',
        'code_review_attempts': 7,
        'code_review_started_at': (datetime.now(timezone.utc)-timedelta(hours=1)).isoformat(),
        'code_review_doc': {'model': code_review.DEFAULT_REVIEW_MODEL,
                            'error_code': 'code_review_provider_refused', 'retryable': True},
    }


def router_receipt(body, parameters, *, prompt_tokens=585_000):
    body['usage'] = {**body.get('usage', {}), 'prompt_tokens': prompt_tokens}
    body['openrouter_metadata'] = {
        'requested': parameters['model'], 'strategy': 'direct', 'attempt': 1, 'pipeline': [],
        'endpoints': {'total': 1, 'available': [{
            'provider': 'Azure', 'model': code_review._REVIEW_MODEL_CANONICAL_SLUGS[parameters['model']],
            'selected': True,
        }]},
    }
    return body


def reviewer_for(payload, transport, store=None):
    store = store or ReviewStore()
    return SubmissionCodeReviewer(
        store=store, objects=Objects(payload), credential_for=lambda _row: MINER_KEY,
        price_table=fallback_prices(), transport=transport,
    ), store


@pytest.mark.parametrize('research', [False, True])
def test_alternate_preserves_complete_request_and_policy(research):
    payload = _archive({**_source(), 'large.txt': b'public source line\n' * 80_000})
    options = {'research_control': research, 'similarity_contexts': ({'comparison_id': 'a'},)}
    primary = code_review.prepare_request(payload, **options)
    alternate = code_review.prepare_request(
        payload, model=code_review.FALLBACK_REVIEW_MODEL,
        context_window_tokens=code_review.FALLBACK_CONTEXT_WINDOW_TOKENS, **options,
    )
    assert alternate.parameters['messages'] == primary.parameters['messages']
    assert alternate.files == primary.files
    assert alternate.research_control == primary.research_control
    assert alternate.source_manifest_sha256 == primary.source_manifest_sha256
    assert alternate.parameters['max_tokens'] == primary.parameters['max_tokens']
    assert alternate.parameters['response_format'] == primary.parameters['response_format']
    assert alternate.parameters['provider'] == {
        'only': ['azure'], 'allow_fallbacks': False,
        'require_parameters': True, 'data_collection': 'deny', 'zdr': True,
    }
    assert alternate.parameters['transforms'] == []
    assert alternate.parameters['plugins'] == [{'id': 'context-compression', 'enabled': False}]
    assert alternate.requires_context_integrity is True


@pytest.mark.parametrize('unknown_primary_cost', [False, True])
def test_typed_refusal_selects_next_claim_without_inline_dispatch(unknown_primary_cost):
    payload = _archive(_source())
    def response(parameters):
        body = _review_body(parameters, cost=None if unknown_primary_cost and parameters['model'] == code_review.DEFAULT_REVIEW_MODEL else '0.001234')
        if parameters['model'] == code_review.DEFAULT_REVIEW_MODEL:
            body['choices'][0]['finish_reason'] = 'content_filter'
        else:
            router_receipt(body, parameters)
        return {'body': body}
    transport = ReviewTransport(response)
    reviewer, store = reviewer_for(payload, transport)
    row = {**SUBMISSION, 'source_size_bytes': len(payload)}
    assert reviewer.review(row)['status'] == 'error'
    assert len(transport.sent) == 1
    original_doc = store.finish_calls[-1][4]
    assert original_doc['error_code'] == 'code_review_provider_refused'
    store.terminal = None
    assert reviewer.review({**fallback_row(payload), 'code_review_doc': original_doc})['status'] == 'passed'
    assert len(transport.sent) == 2
    assert [call[4] for call in store.begin_calls] == [code_review.DEFAULT_REVIEW_MODEL, code_review.FALLBACK_REVIEW_MODEL]
    assert store.finish_calls[0][5] == (None if unknown_primary_cost else 1234)
    assert store.finish_calls[1][5] == 1234
    first, second = [json.loads(call['body']) for call in transport.sent]
    assert first['messages'] == second['messages']
    assert code_review_policy.review_model_for({'code_review_doc': {'error_code': 'code_review_provider_request_rejected'}}) == code_review.DEFAULT_REVIEW_MODEL


@pytest.mark.parametrize('failure', [
    'refusal', 'malformed', 'truncated', 'coverage', 'unknown_cost', 'wrong_model',
    'missing_receipt', 'wrong_provider', 'hidden_attempt', 'compressed', 'too_large',
    'authentication', 'credit', 'transient', 'transport',
])
def test_failed_alternate_is_terminal_without_fabricated_verdict(failure):
    payload = _archive(_source())
    def response(parameters):
        if failure == 'transport':
            raise broker.ProviderTransportError('private transport detail')
        body = router_receipt(_review_body(parameters), parameters)
        status = 200
        if failure == 'refusal':
            body['choices'][0]['finish_reason'] = 'content_filter'
        elif failure == 'malformed':
            body['choices'][0]['message']['content'] = 'not json'
        elif failure == 'truncated':
            body['choices'][0]['finish_reason'] = 'length'
        elif failure == 'coverage':
            body['choices'][0]['message']['content'] = json.dumps({'verdict': 'pass', 'reviewed_files': [], 'findings': []})
        elif failure == 'unknown_cost':
            body['usage'].pop('cost')
        elif failure == 'wrong_model':
            body['model'] = code_review.DEFAULT_REVIEW_MODEL
        elif failure == 'missing_receipt':
            body.pop('openrouter_metadata')
        elif failure == 'wrong_provider':
            body['openrouter_metadata']['endpoints']['available'][0]['provider'] = 'OpenAI'
        elif failure == 'hidden_attempt':
            body['openrouter_metadata']['attempt'] = 2
        elif failure == 'compressed':
            body['openrouter_metadata']['pipeline'] = ['compression']
        elif failure == 'too_large':
            body['usage']['prompt_tokens'] = 922001
        else:
            status = {'authentication': 401, 'credit': 402, 'transient': 503}[failure]
            body = {'error': {'code': status}}
        return {'status': status, 'body': body}
    transport = ReviewTransport(response)
    reviewer, store = reviewer_for(payload, transport)
    assert reviewer.review(fallback_row(payload))['status'] == 'error'
    doc = store.finish_calls[-1][4]
    assert doc['model'] == code_review.FALLBACK_REVIEW_MODEL
    assert doc['retryable'] is False
    assert doc.get('passed') is not True
    terminal = {**fallback_row(payload), 'code_review_doc': doc}
    store.terminal = None
    assert reviewer.review(terminal)['status'] == 'exhausted'
    assert len(transport.sent) == 1
    assert not code_review_policy.review_retry_available(terminal)


def test_large_alternate_retains_every_file_and_conservative_cost():
    members = {**_source(), 'large.txt': b'public source line\n' * 80_000}
    payload = _archive(members)
    transport = ReviewTransport(lambda parameters: {'body': router_receipt(_review_body(parameters), parameters)})
    reviewer, store = reviewer_for(payload, transport)
    assert reviewer.review(fallback_row(payload))['status'] == 'passed'
    request = json.loads(transport.sent[0]['body'])
    supplied = json.loads(request['messages'][1]['content'])
    assert {item['path']: item['content'].encode() for item in supplied['submission_files']} == members
    reservation = store.begin_calls[0][3]
    catalog_only = broker.max_openrouter_cost_microusd(fallback_prices(), code_review.FALLBACK_REVIEW_MODEL, request, max_output_tokens=request['max_tokens'])
    assert reservation > catalog_only
    assert store.finish_calls[0][5] == 1234


def test_runtime_preserves_research_and_similarity_contexts_on_alternate():
    from tests.lab_arena.research_control_code_review_test import SOURCE, _RoundStore, _assessment
    payload = _archive(SOURCE)
    reference = _archive({**SOURCE, 'prompt.txt': b'Check observations using another source before acting.\n'})
    store = _RoundStore({'reasoning_capture_policy': 'full_generated_thinking_v1'})
    def response(parameters):
        supplied = json.loads(parameters['messages'][1]['content'])
        contexts = supplied['similarity_contexts']
        assert contexts
        body = _review_body(parameters)
        document = json.loads(body['choices'][0]['message']['content'])
        document['reviewed_comparisons'] = [item['comparison_id'] for item in contexts]
        prepared = code_review.prepare_request(
            payload, model=parameters['model'], research_control=True,
            similarity_contexts=tuple(contexts),
            context_window_tokens=code_review.FALLBACK_CONTEXT_WINDOW_TOKENS,
        )
        document['research_control'] = _assessment(prepared)
        body['choices'][0]['message']['content'] = json.dumps(document)
        if parameters['model'] == code_review.DEFAULT_REVIEW_MODEL:
            body['choices'][0]['finish_reason'] = 'content_filter'
        else:
            router_receipt(body, parameters)
        return {'body': body}
    transport = ReviewTransport(response)
    reviewer, _ = reviewer_for(payload, transport, store)
    reviewer._similarity_references_for = lambda _row: [{
        'row': {'submission_id': 'public-reference', 'source_ref': 'unused',
                'source_size_bytes': len(reference)},
        'payload': reference, 'source_public': True,
    }]
    row = {**SUBMISSION, 'source_size_bytes': len(payload)}
    assert reviewer.review(row)['status'] == 'error'
    primary_doc = store.finish_calls[0][4]
    store.terminal = None
    assert reviewer.review({**fallback_row(payload), 'code_review_doc': primary_doc})['status'] == 'passed'
    first, alternate = [json.loads(call['body']) for call in transport.sent]
    assert first['messages'] == alternate['messages']
    submitted = json.loads(alternate['messages'][1]['content'])
    stored = store.finish_calls[1][4]['research_control']
    assert stored['source_archive_sha256'] == submitted['research_control']['source_archive_sha256']
    assert stored['source_manifest_sha256'] == submitted['research_control']['source_manifest_sha256']
    assert stored['policy_hash'] == code_review.RESEARCH_POLICY_HASH
    assert stored['classification'] == 'model_directed'
