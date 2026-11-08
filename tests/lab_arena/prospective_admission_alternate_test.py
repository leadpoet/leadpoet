"""Prospective alternate authority never changes historical admission episodes."""
import json
import hashlib
import os
from pathlib import Path

import pytest

from lab_arena import broker, code_review, code_review_policy
from tests.lab_arena.prospective_admission_primary_test import reviewer, receipt, DEEPSEEK
from tests.lab_arena.test_lab_arena_code_review_runtime import ReviewStore, _archive, _source

KIMI = 'moonshotai/kimi-k3.1-pro'
CANONICAL = 'moonshotai/kimi-k3.1-pro-20261023'


def truncated(model=DEEPSEEK, *, alternate=None, reason='truncated'):
    document = {'model': model, 'error_code': 'review_response_invalid',
                'error_reason': reason, 'review_invalid_attempts': 1}
    if alternate is not None:
        document['review_alternate_model'] = alternate
    return {'code_review_attempts': 1, 'code_review_status': 'error', 'code_review_doc': document}


def test_only_saved_exact_alternate_changes_future_truncation_route():
    assert code_review_policy.review_model_for(truncated(alternate=KIMI)) == KIMI
    for model in (DEEPSEEK, code_review.DEFAULT_REVIEW_MODEL):
        expected = code_review.FALLBACK_REVIEW_MODEL if model == DEEPSEEK else model
        assert code_review_policy.review_model_for(truncated(model)) == expected
    assert code_review_policy.review_model_for(truncated(alternate='moonshotai/kimi-k3.1-pro:highspeed')) == code_review.FALLBACK_REVIEW_MODEL
    terminal = truncated(KIMI, alternate=KIMI, reason='truncated_at_limit')
    assert code_review_policy.review_retry_available(terminal) is False


def kimi_receipt(parameters):
    response = receipt(parameters, provider='Moonshot AI')
    response['openrouter_metadata']['endpoints']['available'][0]['model'] = CANONICAL
    return response


def test_exact_canonical_endpoint_is_supported_without_output_rewriting():
    prepared = code_review.prepare_request(_archive(_source()), model=KIMI,
        max_output_tokens=65536, context_window_tokens=1048576)
    response = kimi_receipt(prepared.parameters)
    raw = json.dumps(response)
    assert code_review.parse_response(response, prepared).passed
    assert json.dumps(response) == raw
    assert prepared.parameters['provider']['only'] == ['moonshotai/mxfp4']
    assert prepared.parameters['reasoning'] == {'effort': 'high', 'exclude': False}


@pytest.mark.parametrize('damage', ['model', 'provider', 'compression', 'attempt', 'window', 'prompt_ceiling'])
def test_exact_route_and_full_input_contract_fail_closed(damage):
    prepared = code_review.prepare_request(_archive(_source()), model=KIMI,
        max_output_tokens=65536, context_window_tokens=1048576)
    response = kimi_receipt(prepared.parameters)
    metadata = response['openrouter_metadata']
    if damage == 'model': metadata['endpoints']['available'][0]['model'] += '-other'
    elif damage == 'provider': metadata['endpoints']['available'][0]['provider'] = 'Moonshot AI Highspeed'
    elif damage == 'compression': metadata['pipeline'] = ['context-compression']
    elif damage == 'attempt': metadata['attempt'] = 2
    elif damage == 'window': response['usage']['prompt_tokens'] = 1048576 - 65536 + 1
    else: response['usage']['prompt_tokens'] = 922001
    with pytest.raises(code_review.CodeReviewError) as error:
        code_review.parse_response(response, prepared)
    assert error.value.response_reason == 'context_integrity'


class BoundStore(ReviewStore):
    alternate = KIMI
    def begin_submission_review(self, *args):
        return {**super().begin_submission_review(*args), 'review_alternate_model': self.alternate}


def test_native_runtime_uses_bound_alternate_and_endpoint_price_ceiling():
    store = BoundStore()
    store.output_token_recommendation = 65536
    instance, row, store, transport = reviewer(lambda p: {'body': kimi_receipt(p)}, store=store)
    instance._prices['models'][KIMI] = {'prompt': '0.00000001', 'completion': '0.00000001', 'internal_reasoning': '0', 'request': '0', 'web_search': '0'}
    row.update(truncated(alternate=KIMI))
    assert instance.review(row)['status'] == 'passed'
    parameters = json.loads(transport.sent[0]['body'])
    assert parameters['model'] == KIMI
    assert parameters['max_tokens'] == 65536
    assert parameters['provider']['max_price'] == {'prompt': 2.0, 'completion': 10.0, 'request': 0.0}
    expected = broker._max_openrouter_cost_for_pricing(
        {'prompt': '0.000002', 'completion': '0.00001', 'internal_reasoning': '0', 'request': '0', 'web_search': '0'}, parameters,
        model=KIMI, max_output_tokens=65536)
    assert store.begin_calls[0][3] == expected
    assert store.finish_calls[0][4]['review_alternate_model'] == KIMI


def test_sql_route_mismatch_settles_undispatched_claim_at_zero():
    store = BoundStore()
    store.alternate = code_review.FALLBACK_REVIEW_MODEL
    store.output_token_recommendation = 65536
    instance, row, store, transport = reviewer(lambda p: pytest.fail('must not dispatch'), store=store)
    instance._prices['models'][KIMI] = {'prompt': '0.000002', 'completion': '0.00001', 'internal_reasoning': '0', 'request': '0', 'web_search': '0'}
    row.update(truncated(alternate=KIMI))
    assert instance.review(row)['status'] == 'error'
    assert transport.sent == []
    assert store.finish_calls[0][5] == 0


def test_retained_sixteen_native_replies_without_response_normalization():
    """Opt-in offline replay keeps unreviewed private source/replies out of Git."""
    location = os.environ.get('ARENA_KIMI_REGRESSION_DIR')
    if not location:
        pytest.skip('set ARENA_KIMI_REGRESSION_DIR to the private admission-alternate evidence directory')
    directory = Path(location)
    frozen = json.loads((Path(__file__).parents[1] / 'fixtures/admission_kimi_regression_20261108.json').read_text())
    manifest = json.loads((directory / 'kimi65536/manifest.json').read_text())
    assert manifest['manifest_sha256'] == frozen['manifest_sha256']
    attacks = benign = matches = 0
    for case in frozen['cases']:
        request_raw = (directory / 'kimi65536/requests' / (case['case'] + '.json')).read_bytes()
        response_raw = (directory / 'kimi65536-live-v1' / (case['case'] + '.response.json')).read_bytes()
        assert hashlib.sha256(request_raw).hexdigest() == case['request_sha256']
        assert hashlib.sha256(response_raw).hexdigest() == case['response_sha256']
        parameters = json.loads(request_raw)
        submitted = json.loads(parameters['messages'][1]['content'])
        files = tuple(code_review.ReviewFile(item['path'], item['size_bytes'], item['content'])
                      for item in submitted['submission_files'])
        prepared = code_review.PreparedCodeReview(parameters, files,
            code_review.REQUEST_TOKEN_OVERHEAD + len(request_raw), (), True, 1048576,
            True, case['source_archive_sha256'], case['source_manifest_sha256'])
        result = code_review.parse_response(json.loads(response_raw), prepared)
        actual = {'verdict': result.verdict, 'categories': sorted({x['category'] for x in result.findings}),
                  'research_classification': result.research_control['classification']}
        for field, expected in case['expected'].items():
            if expected is not None:
                assert actual[field] == expected, (case['case'], field)
        matches += 1
        attacks += case['expected']['verdict'] == 'reject'
        benign += case['expected']['verdict'] == 'pass'
    assert (matches, attacks, benign) == (16, 8, 7)


def test_postgrest_transport_selects_explicit_opt_in_without_changing_legacy_rpc():
    import httpx
    from lab_arena.store import ArenaStore, PostgrestTransport, new_lease_token
    requests = []
    def respond(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={'status': 'paused'})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        store = ArenaStore(PostgrestTransport('https://arena.example/rest/v1',
            service_key='sb_secret_fixture_not_a_real_credential', http_client=client))
        args = ('fixture-submission', 'fixture-hotkey', new_lease_token(), 1234, DEEPSEEK, 2, 1000)
        store.begin_submission_review(*args)
        store.begin_submission_review(*args, KIMI)
    assert requests[0][0].endswith('/rpc/lab_arena_begin_submission_review')
    assert 'p_review_alternate_model' not in requests[0][1]
    assert requests[1][0].endswith('/rpc/lab_arena_begin_submission_review_v2')
    assert requests[1][1] == {**requests[0][1], 'p_review_alternate_model': KIMI}


def test_primary_selection_alone_keeps_standalone_legacy_rpc_and_document():
    instance, row, store, _ = reviewer(lambda parameters: {'body':receipt(parameters)})
    assert instance.review(row)['status'] == 'passed'
    assert len(store.begin_calls[0]) == 7
    assert 'review_alternate_model' not in store.finish_calls[0][4]


def test_explicit_alternate_optin_uses_v2_and_retains_authoritative_marker():
    instance, row, store, _ = reviewer(lambda parameters: {'body':receipt(parameters)},
        store=BoundStore(), prospective_alternate_model=KIMI)
    assert instance.review(row)['status'] == 'passed'
    assert len(store.begin_calls[0]) == 8
    assert store.begin_calls[0][-1] == KIMI
    assert store.finish_calls[0][4]['review_alternate_model'] == KIMI
