"""Prospective admission routing preserves historical requests and claim limits."""
import json

import pytest

from lab_arena import broker, code_review, code_review_policy, contracts
from lab_arena.code_review_runtime import SubmissionCodeReviewer
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    MINER_KEY, Objects, ReviewStore, ReviewTransport, SUBMISSION,
    _archive, _price_table, _review_body, _source,
)

DEEPSEEK = 'deepseek/deepseek-v4.1-flash-1017'
CANONICAL = 'deepseek/deepseek-v4.1-flash-20261017'


def receipt(parameters, *, verdict='pass', cost='0.001234', provider='Nebius'):
    body = _review_body(parameters, verdict=verdict, findings=[] if verdict == 'pass' else [{
        'category': 'fabricated_evidence', 'file': 'src/agent.py',
        'evidence': 'def run_icp(icp, tools): return tools.search(icp.industry)',
        'explanation': 'Fixture finding, not an external source judgment.',
    }], cost=cost)
    body.update(provider=provider, openrouter_metadata={
        'requested': parameters['model'], 'strategy': 'direct', 'attempt': 1, 'pipeline': [],
        'endpoints': {'total': 1, 'available': [{
            'provider': provider, 'model': CANONICAL, 'selected': True}]},
    })
    if cost is not None:
        body['usage']['prompt_tokens'] = 1000
    return body


def reviewer(response, *, store=None, capture=None,
             prospective_alternate_model=code_review.FALLBACK_REVIEW_MODEL):
    payload = _archive(_source())
    prices = _price_table()
    prices['models'][DEEPSEEK] = dict(prices['models'][code_review.DEFAULT_REVIEW_MODEL])
    prices['models'][code_review.FALLBACK_REVIEW_MODEL] = dict(prices['models'][code_review.DEFAULT_REVIEW_MODEL])
    store = store or ReviewStore()
    transport = ReviewTransport(response)
    instance = SubmissionCodeReviewer(store=store, objects=Objects(payload),
        credential_for=lambda row: MINER_KEY, price_table=prices, transport=transport,
        prospective_primary_model=DEEPSEEK, capture=capture,
        prospective_alternate_model=prospective_alternate_model)
    row = {**SUBMISSION, 'source_size_bytes': len(payload), 'code_review_status': 'pending',
           'code_review_attempts': 0}
    return instance, row, store, transport


@pytest.mark.parametrize('verdict,status', [('pass', 'passed'), ('reject', 'rejected')])
def test_new_primary_uses_exact_native_route_and_actual_ledger_identity(verdict, status):
    instance, row, store, transport = reviewer(lambda p: {'body': receipt(p, verdict=verdict)})
    assert instance.review(row)['status'] == status
    body = json.loads(transport.sent[0]['body'])
    assert body['model'] == DEEPSEEK
    assert body['provider']['only'] == ['nebius/fp8']
    assert body['provider']['allow_fallbacks'] is False
    assert body['provider']['max_price'] == {'prompt': 0.3, 'completion': 1.2, 'request': 0.0}
    assert body['provider']['enforce_distillable_text'] is True
    assert body['reasoning'] == {'effort': 'low', 'exclude': False}
    assert body['max_tokens'] == 32768
    assert body['response_format']['type'] == 'json_schema'
    assert body['response_format']['json_schema']['strict'] is True
    assert store.begin_calls[0][4] == store.finish_calls[0][4]['model'] == DEEPSEEK
    assert store.finish_calls[0][5] == 1234


def test_prospective_truncation_uses_existing_alternate_slot_not_65536_deepseek():
    row = {'code_review_status': 'error', 'code_review_attempts': 1,
           'code_review_doc': {'model': DEEPSEEK, 'error_code': 'review_response_invalid',
                               'error_reason': 'truncated', 'review_invalid_attempts': 1}}
    assert code_review_policy.review_model_for(row, prospective_primary_model=DEEPSEEK) == code_review.FALLBACK_REVIEW_MODEL
    assert code_review_policy.review_output_tokens_for(row) == 65536


@pytest.mark.parametrize('document,expected', [
    ({'model': code_review.DEFAULT_REVIEW_MODEL, 'error_code': 'code_review_transport_error'}, code_review.DEFAULT_REVIEW_MODEL),
    ({'model': code_review.DEFAULT_REVIEW_MODEL, 'error_code': 'code_review_provider_refused',
      'refusal_recovery': True, 'review_primary_attempts': 1}, code_review.DEFAULT_REVIEW_MODEL),
    ({'model': DEEPSEEK, 'error_code': 'code_review_provider_refused',
      'refusal_recovery': True, 'review_primary_attempts': 1}, DEEPSEEK),
    ({'model': DEEPSEEK, 'error_code': 'code_review_provider_refused',
      'refusal_recovery': True, 'review_primary_attempts': 2}, code_review.FALLBACK_REVIEW_MODEL),
    ({'model': code_review.FALLBACK_REVIEW_MODEL, 'error_code': 'code_review_transport_error',
      'refusal_recovery': True, 'review_alternate_attempts': 1}, code_review.FALLBACK_REVIEW_MODEL),
])
def test_saved_episodes_do_not_follow_new_global_primary(document, expected):
    assert code_review_policy.review_model_for({'code_review_attempts': 7, 'code_review_doc': document},
                                               prospective_primary_model=DEEPSEEK) == expected


@pytest.mark.parametrize('damage', ['wrong_provider', 'wrong_version', 'fallback_attempt', 'compression'])
def test_wrong_actual_route_or_context_stays_accounted_error_without_retry(damage):
    def response(parameters):
        body = receipt(parameters)
        if damage == 'wrong_provider':
            body['openrouter_metadata']['endpoints']['available'][0]['provider'] = 'Fireworks'
        elif damage == 'wrong_version':
            body['openrouter_metadata']['endpoints']['available'][0]['model'] = 'deepseek/deepseek-v4.1-flash-20261018'
        elif damage == 'fallback_attempt':
            body['openrouter_metadata']['attempt'] = 2
        else:
            body['openrouter_metadata']['pipeline'] = ['context-compression']
        return {'body': body}
    instance, row, store, transport = reviewer(response)
    assert instance.review(row)['status'] == 'error'
    assert store.finish_calls[0][4]['error_reason'] == 'context_integrity'
    assert store.finish_calls[0][5] == 1234
    assert len(transport.sent) == 1


def test_missing_actual_cost_remains_unknown_and_cannot_commit_a_pass():
    def response(parameters):
        body = receipt(parameters, cost=None)
        body['usage'] = {'prompt_tokens': 1000}
        return {'body': body}
    instance, row, store, transport = reviewer(response)
    assert instance.review(row)['status'] == 'error'
    assert store.finish_calls[0][4]['error_code'] == 'code_review_cost_unavailable'
    assert store.finish_calls[0][5] is None
    assert len(transport.sent) == 1


def test_idempotent_claim_never_dispatches_or_finishes_again():
    class Claimed(ReviewStore):
        def begin_submission_review(self, *args):
            return {'status': 'claimed', 'idempotent': True}
    instance, row, store, transport = reviewer(lambda parameters: pytest.fail('HTTP not allowed'), store=Claimed())
    assert instance.review(row)['idempotent'] is True
    assert transport.sent == store.finish_calls == []


def test_real_wiring_requires_db_capability_and_selects_only_prospective_admission(monkeypatch):
    from tests.lab_arena.broker_round_lease_wiring_test import _wired_factory
    service, _, _ = _wired_factory(monkeypatch)
    assert service.config.code_reviewer.prospective_primary_model == DEEPSEEK
    assert service.config.code_reviewer.prospective_alternate_model == code_review.PROSPECTIVE_ALTERNATE_MODEL


def test_reservation_uses_the_endpoint_ceiling_when_catalog_is_cheaper():
    instance, row, store, transport = reviewer(lambda parameters: {'body': receipt(parameters)})
    instance._prices['models'][DEEPSEEK]['prompt'] = '0.00000001'
    instance._prices['models'][DEEPSEEK]['completion'] = '0.00000001'
    instance.review(row)
    parameters = json.loads(transport.sent[0]['body'])
    pricing = {**instance._prices['models'][DEEPSEEK], **code_review.PROSPECTIVE_PRICE_CEILING}
    expected = broker._max_openrouter_cost_for_pricing(pricing, parameters, model=DEEPSEEK, max_output_tokens=32768)
    assert store.begin_calls[0][3] == expected


def test_deepseek_research_classification_and_exact_excerpts_keep_native_security_checks():
    from tests.lab_arena.research_control_code_review_test import SOURCE, _assessment, _response
    payload = _archive(SOURCE)
    prepared = code_review.prepare_request(payload, model=DEEPSEEK, research_control=True)
    response = _response(prepared, _assessment(prepared))
    response.update({key: receipt(prepared.parameters)[key] for key in ('provider', 'openrouter_metadata', 'usage')})
    result = code_review.parse_response(response, prepared)
    assert result.passed is True
    assert result.research_control['classification'] == 'model_directed'
    assert result.research_control['source_archive_sha256'] == contracts.hash_bytes(payload)
    document = json.loads(response['choices'][0]['message']['content'])
    document['research_control']['evidence'][0]['excerpt'] = 'this source claim was never present'
    response['choices'][0]['message']['content'] = json.dumps(document)
    with pytest.raises(code_review.CodeReviewError) as error:
        code_review.parse_response(response, prepared)
    assert error.value.response_reason == 'research_evidence_mismatch'


def test_deepseek_research_format_suffix_repeats_only_existing_native_contract():
    from tests.lab_arena.research_control_code_review_test import SOURCE
    payload = _archive(SOURCE)
    prepared = code_review.prepare_request(payload, model=DEEPSEEK, research_control=True)
    native_prefix = code_review._SYSTEM_PROMPT + code_review._RESEARCH_PROMPT + code_review._RESEARCH_RESPONSE_GUIDANCE
    suffix = code_review._PROSPECTIVE_RESEARCH_RESPONSE_GUIDANCE
    assert prepared.parameters['messages'][0]['content'] == native_prefix + suffix
    assert contracts.document_hash(prepared.parameters) == 'sha256:b011ba9a73b760cb8efc9bd825258a0800c08137c889a4d333cd3d7fbee30d34'
    assert '12 non-whitespace characters' in suffix
    assert 'exact contiguous substring' in suffix
    assert 'adjacent source lines' in suffix
    for classification, claims in {
        'model_directed': 'observation_to_action, finish_continue',
        'workflow': 'fixed_control', 'unverified': 'uncertainty',
    }.items():
        assert classification + ': ' + claims in suffix
    assert 'explanation or limitations' in suffix
    for model in (code_review.DEFAULT_REVIEW_MODEL, code_review.FALLBACK_REVIEW_MODEL):
        historical = code_review.prepare_request(payload, model=model, research_control=True)
        assert historical.parameters['messages'][0]['content'] == native_prefix
    plain = code_review.prepare_request(payload, model=DEEPSEEK)
    assert plain.parameters['messages'][0]['content'] == code_review._SYSTEM_PROMPT


@pytest.mark.parametrize('damage,reason', [
    ('short_exact_excerpt', 'research_evidence_mismatch'),
    ('extra_scaffolding_claim', 'research_classification'),
])
def test_format_guidance_never_weakens_excerpt_or_classification_validator(damage, reason):
    from tests.lab_arena.research_control_code_review_test import SOURCE, _assessment, _response
    prepared = code_review.prepare_request(_archive(SOURCE), model=DEEPSEEK, research_control=True)
    assessment = _assessment(prepared)
    if damage == 'short_exact_excerpt':
        assessment['evidence'][0]['excerpt'] = 'while True:'
    else:
        assessment['evidence'].append({**assessment['evidence'][0], 'claim': 'fixed_control'})
    response = _response(prepared, assessment)
    response.update({key: receipt(prepared.parameters)[key] for key in ('provider', 'openrouter_metadata', 'usage')})
    with pytest.raises(code_review.CodeReviewError) as error:
        code_review.parse_response(response, prepared)
    assert error.value.response_reason == reason


@pytest.mark.parametrize('summary_only', [False, True])
def test_actual_low_thinking_is_captured_without_changing_the_probe_mode(tmp_path, summary_only):
    from lab_arena import internal_capture, reasoning_capture
    from lab_arena.service import LocalObjectStore
    class Capture(internal_capture.InternalCapture):
        def start(self, **kwargs):
            self.call = super().start(**kwargs)
            return self.call
    objects = LocalObjectStore(tmp_path)
    capture = Capture(objects)
    def response(parameters):
        body = receipt(parameters)
        body['choices'][0]['message']['reasoning_details'] = [{
            'type': 'reasoning.summary' if summary_only else 'reasoning.text',
            'summary' if summary_only else 'text': 'Compare the reachable source with the policy.'}]
        body['usage']['completion_tokens_details'] = {'reasoning_tokens': 12}
        return {'body': body}
    instance, row, _, transport = reviewer(response, capture=capture)
    assert instance.review(row)['status'] == 'passed'
    registry = reasoning_capture.load_registry()
    raw, candidate = internal_capture.read_call(objects, capture.call.prefix + 'request.json', registry=registry)
    assert raw['request'] == json.loads(transport.sent[0]['body'])
    assert raw['request']['reasoning']['effort'] == 'low'
    assert raw['reasoning']['full_generated_thinking'] == ([] if summary_only else [{
        'field': 'reasoning_details.text', 'text': 'Compare the reachable source with the policy.',
        'source_paths': [['choices', 0, 'message', 'reasoning_details', 0, 'text']]}])
    assert candidate['training_permission']['status'] == 'confirmed'
    entry = next(r for r in registry['approved_routes'] if r['model_id'] == DEEPSEEK and
                 r['provider'] == 'nebius/fp8' and r['api'] == 'openrouter.chat')
    assert entry['mode'] == {'effort': 'high'}


# Exact future request bindings for the configured routes and output plans.
# The evidence-reasoning prompt applies when a new review is prepared;
# completed historical rows return before preparation or provider dispatch.
@pytest.mark.parametrize('model,research,cap,digest', [
    ('anthropic/claude-sonnet-5.5', False, 32768, '77d5fa07e1574088935a3414fe096103ce5c02813326e47aa4adf73b0362c6e7'),
    ('anthropic/claude-sonnet-5.5', False, 65536, '17225be7ed2a0e9755a0fd9a323099bc292920c8b0722007089fa415d3a79ae8'),
    ('anthropic/claude-sonnet-5.5', True, 32768, '1637c92b10484d17525576971bbe521a602a8ebc4958729726ef75e12c0aaab4'),
    ('anthropic/claude-sonnet-5.5', True, 65536, '4d23157f2402ccee9a3a5bdea5c9bf0f5c4f81ab303eeac0303a23775728402a'),
    ('openai/gpt-6.1-sol', False, 32768, 'edfa909cd8d9d8463090fd61c9c1b3a11ec4a2d7eeeda399366d2e644126bfd6'),
    ('openai/gpt-6.1-sol', False, 65536, '63aa63e65275a9f18ec07b9d9b462153c773f066bc2ce47d4078bc62302546fc'),
    ('openai/gpt-6.1-sol', True, 32768, 'ca464b6adb0b841baef635b601b2a9ec9da9f0ac3a2cc800437769b2330fef1e'),
    ('openai/gpt-6.1-sol', True, 65536, 'c7138ec393b589ca84635171986e8ad5d638636d97d70267bdf5ee47a02f7c7d'),
])
def test_configured_route_request_bytes_are_bound(model, research, cap, digest):
    prepared = code_review.prepare_request(_archive(_source()), model=model,
        research_control=research, max_output_tokens=cap,
        context_window_tokens=1050000 if model == code_review.FALLBACK_REVIEW_MODEL else 1000000)
    assert contracts.document_hash(prepared.parameters) == 'sha256:' + digest
