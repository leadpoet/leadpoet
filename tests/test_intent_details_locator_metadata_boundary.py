"""Unread locator metadata must not become paragraph citation evidence."""
import asyncio
from copy import deepcopy
import json

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import intent_details as review, verification_helpers
from tests.test_intent_details_industry_support_handoff import _inputs


MARKER = inv._IDENTITY_LINK_CONTEXT_MARKER
UNREAD = 'https://caci.com/press/unread-product-claim'


def response(document, index, quote):
    return {**{key: True for key in review._CHECKS},
            'signal_coverage': [{'matched_icp_signal': 0, 'covered': True}],
            'unit_grounding': [
                {'unit_id': unit['unit_id'], 'contains_factual_claim': i == 0,
                 'status': 'VERIFIED',
                 'evidence': [{'source_index': index, 'quote': quote}] if i == 0 else []}
                for i, unit in enumerate(document['intent_details_units'])]}


def company_document(monkeypatch, suffix=UNREAD):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'reviewer')
    inputs = _inputs()
    web = inputs[3]['dimension_evidence']['industry']['web_evidence']
    url = web['url'].rstrip('/') + '/press/'
    web['url'] = url
    text = web['quote'] + ' ' + MARKER + ' ' + suffix
    document = review.review_evidence(*inputs, company_source_contexts=[{
        'dimension': 'industry', 'url': url, 'text': text}])
    index = next(s['source_index'] for s in document['admitted_evidence']
                 if s.get('evidence_kind') == 'verified_company_source_context')
    return inputs, document, index, url, web['quote'], text


@pytest.mark.parametrize('suffix', [UNREAD, 'https://caci.com/papers/unread.pdf',
                                   'https://www.linkedin.com/company/unread'])
def test_full_company_projection_strips_all_existing_locator_metadata(monkeypatch, suffix):
    _, doc, index, url, quote, _ = company_document(monkeypatch, suffix)
    source = doc['admitted_evidence'][index]
    assert source['source_url'] == url
    assert quote in source['admitted_text'][0]
    assert all(MARKER not in text and suffix not in text
               for source in doc['admitted_evidence'] for text in source.get('admitted_text', []))
    assert review._validate_review_response(json.dumps(response(doc, index, quote)), doc,
                                          initial_review=False)['facts_supported'] is True
    with pytest.raises(ValueError):
        review._validate_review_response(json.dumps(response(doc, index, suffix)), doc,
                                         initial_review=False)


@pytest.mark.parametrize('bad_quote', [UNREAD, MARKER + ' ' + UNREAD,
                                      'Real body evidence. ' + MARKER + ' ' + UNREAD])
def test_saved_or_manual_document_cannot_bind_suffix_or_spanning_quote(bad_quote):
    doc = review.review_evidence(*_inputs())
    index = 0
    doc['admitted_evidence'][index]['admitted_text'] = ['Real body evidence. ' + MARKER + ' ' + UNREAD]
    with pytest.raises(ValueError):
        review._validate_review_response(json.dumps(response(doc, index, bad_quote)), doc,
                                         initial_review=False)
    assert review._validate_review_response(json.dumps(response(doc, index, 'Real body evidence.')),
                                           doc, initial_review=False)['facts_supported'] is True


def test_positive_signal_context_is_clean_before_window_and_body_dates(monkeypatch):
    inputs = _inputs()
    trace = inputs[2][0]['judge_verdict']['verification_trace']
    fact = 'The company announced a contract.'
    trace['verified_source_context'] = [{'url': 'https://source.example/contract',
        'text': fact + ' ' + MARKER + '\nSeattle, October 20, 2026 - ' + UNREAD}]
    doc = review.review_evidence(*inputs)
    assert all(UNREAD not in text and MARKER not in text
               for source in doc['admitted_evidence'] for text in source.get('admitted_text', []))
    assert not any(date.get('basis') == 'source_body_dateline_date'
                   for source in doc['admitted_evidence'] for date in source.get('observed_dates', []))
    assert any(fact in text for source in doc['admitted_evidence'] for text in source.get('admitted_text', []))


@pytest.mark.parametrize('kind', ['signal', 'nonqualifying', 'company'])
def test_central_projection_never_admits_suffix_in_body_title_or_quote(kind):
    text = 'Real body evidence. ' + MARKER + ' ' + UNREAD
    doc = {'verified_signals': [], 'verified_company_evidence': {}}
    if kind == 'signal':
        doc['verified_signals'] = [{'matched_icp_signal': 0, 'supporting_quotes': [text],
            'source_urls': ['https://source.example/'], 'source_context': [{
                'url': 'https://source.example/', 'text': text, 'source_title': text}]}]
    elif kind == 'nonqualifying':
        doc['non_qualifying_signals'] = [{'matched_icp_signal': 0,
            'source_urls': ['https://source.example/'], 'supporting_quotes': [text],
            'quote_sources': {'supporting_quotes': ['https://source.example/']}}]
    else:
        doc['verified_company_evidence'] = {'industry': {'url': 'https://source.example/',
            'quote': text, 'source_context': {'url': 'https://source.example/', 'text': text}}}
    review._project_admitted_evidence(doc)
    assert all(MARKER not in value and UNREAD not in value
               for source in doc['admitted_evidence'] for value in source.get('admitted_text', []))
    assert any('Real body evidence.' in value for source in doc['admitted_evidence']
               for value in source.get('admitted_text', []))


def test_continuous_window_and_date_helpers_cannot_select_metadata():
    text = 'Real source body. ' + MARKER + '\nSeattle, October 20, 2026 - ' + UNREAD
    assert review._continuous_source_window(text, 2000, [UNREAD], paragraph=UNREAD) == 'Real source body. '
    assert review._body_dateline_dates(text) == []


def test_native_review_does_not_accept_metadata_citation(monkeypatch):
    inputs, doc, index, url, _, text = company_document(monkeypatch)
    calls = []
    async def model(prompt, **kwargs):
        calls.append(kwargs)
        supplied = json.loads(prompt)
        if 'review_document' in supplied:
            # A failed first citation gets only the existing bounded repair.
            return json.dumps({'repairs': [{'unit_id': 0, 'evidence': [
                {'source_index': index, 'quote': UNREAD}]}]})
        return json.dumps(response(supplied, index, UNREAD))
    monkeypatch.setattr(verification_helpers, 'openrouter_chat', model)
    result = asyncio.run(review.review_intent_details(*inputs, company_source_contexts=[{
        'dimension': 'industry', 'url': url, 'text': text}]))
    assert result['decision'] == 'unavailable'
    assert result['failure_reason_code'] == 'malformed_response'
    assert len(calls) <= 2


def test_ordinary_body_with_real_url_is_not_removed():
    value = 'API documentation is available at https://caci.com/api.'
    assert review._continuous_source_window(value, 2000, []) == value
    assert review._quote_is_bound(value, [value])


@pytest.mark.parametrize('dimension', ['industry', 'required_attribute', 'employee_size',
                                       'geography', 'stage', 'first_party_company',
                                       'first_party_context'])
def test_all_company_context_routes_keep_body_but_not_metadata(monkeypatch, dimension):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'reviewer')
    inputs = _inputs()
    company, _, _, receipt = inputs
    web = receipt['dimension_evidence']['industry']['web_evidence']
    url, quote = web['url'], web['quote']
    if dimension not in {'first_party_company', 'first_party_context'}:
        receipt['dimension_evidence'][dimension] = {'decision': 'match',
                                                    'web_evidence': {'url': url, 'quote': quote}}
    if dimension == 'first_party_company':
        from types import SimpleNamespace
        company.required_attribute = SimpleNamespace(evidence_url=url, evidence_quote=quote)
    text = company.company_name + '. ' + quote
    context = {'dimension': dimension, 'url': url, 'text': text + ' ' + MARKER + ' ' + UNREAD}
    if dimension == 'first_party_context':
        context['final_url'] = url
    doc = review.review_evidence(*inputs, company_source_contexts=[context])
    sources = [s for s in doc['admitted_evidence']
               if s.get('evidence_kind') == 'verified_company_source_context'
               and s.get('company_dimension') == dimension]
    assert len(sources) == 1
    assert quote in sources[0]['admitted_text'][0]
    assert MARKER not in sources[0]['admitted_text'][0]
    assert UNREAD not in sources[0]['admitted_text'][0]
    assert sources[0]['source_url'] == url


def test_context_cleaning_does_not_mutate_retained_trace():
    trace = {'verified_source_context': [{'url': 'https://source.example/',
                                        'text': 'Real body. ' + MARKER + ' ' + UNREAD}]}
    before = deepcopy(trace)
    cleaned = review._source_context_trace(trace)
    assert trace == before
    assert cleaned['verified_source_context'][0]['text'] == 'Real body. '


def test_empty_metadata_only_context_does_not_create_fact_or_break_indices():
    doc = {'verified_signals': [], 'verified_company_evidence': {
        'industry': {'source_context': {'url': 'https://source.example/',
                                       'text': MARKER + ' ' + UNREAD}},
        'stage': {'url': 'https://source.example/stage', 'quote': 'Acme is public.'}}}
    review._project_admitted_evidence(doc)
    assert len(doc['admitted_evidence']) == 1
    assert doc['admitted_evidence'][0]['source_index'] == 0
    assert doc['admitted_evidence'][0]['admitted_text'] == ['Acme is public.']


def test_marker_like_body_text_and_real_source_urls_are_preserved():
    body = 'Real text mentioning SERVER_VISIBLE_LINK_DESTINATIONS without the delimiter.'
    doc = {'verified_signals': [], 'verified_company_evidence': {
        'industry': {'url': UNREAD, 'quote': body}}}
    review._project_admitted_evidence(doc)
    assert doc['admitted_evidence'][0]['source_url'] == UNREAD
    assert doc['admitted_evidence'][0]['admitted_text'] == [body]


@pytest.mark.parametrize('field', ['supporting_quotes', 'contradicting_quotes'])
def test_known_metadata_quote_cannot_be_re_admitted_as_mandatory_observation(field):
    inputs = _inputs()
    trace = inputs[2][0]['judge_verdict']['verification_trace']
    url = 'https://source.example/contract'
    trace['verified_source_context'] = [{'url': url, 'text': 'Real body. ' + MARKER + ' ' + UNREAD}]
    evaluation = trace['intent_verdict']['signal_evaluations'][0]
    evaluation[field] = [UNREAD]
    cleaned = review._source_context_trace(trace)
    assert cleaned['intent_verdict']['signal_evaluations'][0][field] == []
    if field == 'supporting_quotes':
        with pytest.raises(ValueError, match='verified signal lacks supporting source quotes'):
            review.review_evidence(*inputs)
    else:
        doc = review.review_evidence(*inputs)
        assert not any(UNREAD in value for source in doc['admitted_evidence']
                       for value in source.get('admitted_text', []))


def test_unrelated_source_body_cannot_rescue_metadata_attributed_elsewhere():
    a, b = 'https://source.example/contract', 'https://unrelated.example/document'
    trace = {'verified_source_context': [
        {'url': a, 'text': 'Real body. ' + MARKER + ' ' + UNREAD},
        {'url': b, 'text': 'A different document mentions ' + UNREAD}],
        'intent_verdict': {'signal_evaluations': [{'evidence_urls_used': [a],
                                                 'supporting_quotes': [UNREAD]}]}}
    cleaned = review._source_context_trace(trace)
    assert cleaned['intent_verdict']['signal_evaluations'][0]['supporting_quotes'] == []
    # If that exact body is explicitly cited, the native model may bind to B,
    # never to A's unread locator. Its quote text is kept unchanged.
    trace['intent_verdict']['signal_evaluations'][0]['evidence_urls_used'] = [a, b]
    cleaned = review._source_context_trace(trace)
    assert cleaned['intent_verdict']['signal_evaluations'][0]['supporting_quotes'] == [UNREAD]


def test_valid_body_and_unobserved_mandatory_quotes_stay_exact():
    body_quote = 'Real body evidence.'
    mandatory = 'A verified observation outside the optional retained body window.'
    trace = {'verified_source_context': [{'url': 'https://source.example/',
              'text': body_quote + ' ' + MARKER + ' ' + UNREAD}],
        'intent_verdict': {'signal_evaluations': [{'evidence_urls_used': ['https://source.example/'],
             'supporting_quotes': [body_quote, mandatory, UNREAD]}]}}
    before = deepcopy(trace)
    cleaned = review._source_context_trace(trace)
    assert cleaned['intent_verdict']['signal_evaluations'][0]['supporting_quotes'] == [body_quote, mandatory]
    assert trace == before


def test_unrelated_metadata_does_not_disprove_a_different_cited_source_quote():
    a, b = 'https://source.example/contract', 'https://other.example/document'
    trace = {'verified_source_context': [
        {'url': a, 'text': 'Real body. ' + MARKER + ' ' + UNREAD},
        {'url': b, 'text': 'Other real body.'}],
        'intent_verdict': {'signal_evaluations': [{'evidence_urls_used': [b],
                                                 'supporting_quotes': [UNREAD]}]}}
    cleaned = review._source_context_trace(trace)
    # B's retained window need not contain every verified mandatory quote.
    # An unread locator on A cannot establish that a quote attributed to B
    # came only from metadata, even when their strings happen to match.
    assert cleaned['intent_verdict']['signal_evaluations'][0]['supporting_quotes'] == [UNREAD]


def test_ingress_uses_only_the_existing_five_context_allowance():
    contexts = [{'url': 'https://source.example/' + str(i),
                 'text': 'Real body.' + MARKER + ' ' + UNREAD} for i in range(20)]
    cleaned = review._source_context_trace({'verified_source_context': contexts})
    assert len(cleaned['verified_source_context']) == 5
    assert [c['url'] for c in cleaned['verified_source_context']] == [c['url'] for c in contexts[:5]]


@pytest.mark.parametrize('quote,suffix', [
    ('https://source.example/unread/one https://source.example/unread/two',
     'https://source.example/unread/one https://source.example/unread/two'),
    ('https://source.example/unread/one https://source.example/unread/two',
     'https://source.example/unread/one\n\t https://source.example/unread/two'),
    ('Unread document discovery context', 'Unread document discovery context'),
])
def test_contiguous_multitoken_metadata_quote_cannot_use_mandatory_fallback(quote, suffix):
    url = 'https://source.example/contract'
    trace = {'verified_source_context': [{'url': url, 'text': 'Actual body. ' + MARKER + ' ' + suffix}],
             'intent_verdict': {'signal_evaluations': [{'evidence_urls_used': [url],
                'supporting_quotes': [quote, 'An unrelated real mandatory observation.']} ]}}
    cleaned = review._source_context_trace(trace)
    assert cleaned['intent_verdict']['signal_evaluations'][0]['supporting_quotes'] == [
        'An unrelated real mandatory observation.']


def test_public_ingress_removes_multilink_fallback_but_keeps_valid_signal_quote():
    inputs = _inputs()
    trace = inputs[2][0]['judge_verdict']['verification_trace']
    fact = 'The company announced a contract.'
    metadata = 'https://source.example/unread/one\n https://source.example/unread/two'
    quote = 'https://source.example/unread/one https://source.example/unread/two'
    trace['verified_source_context'] = [{'url': 'https://source.example/contract',
                                       'text': fact + ' ' + MARKER + ' ' + metadata}]
    trace['intent_verdict']['signal_evaluations'][0]['supporting_quotes'] = [fact, quote]
    doc = review.review_evidence(*inputs)
    assert all('source.example/unread' not in value for source in doc['admitted_evidence']
               for value in source.get('admitted_text', []))
    assert any(fact in value for source in doc['admitted_evidence']
               for value in source.get('admitted_text', []))
