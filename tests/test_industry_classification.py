import json

import pytest

from qualification.scoring import industry_classification as industry


CRITERION = {'industry': 'Software', 'subindustry': 'Customer service',
             'product_service': 'Supplies ticket and account workflow software.'}
COMPANY = {'name': 'Example', 'website': 'https://example.com', 'linkedin': ''}


def packet():
    return industry.prepare(CRITERION, COMPANY, [
        {'source_id': 's0', 'source_url': 'https://example.com/tickets',
         'body': 'Example supplies ticket management software.'},
        {'source_id': 's1', 'source_url': 'https://example.com/accounts',
         'body': 'Account workflows connect to the same product through its supported integration.'},
    ], evaluation_date='2026-10-24')


def proposal(value, **updates):
    return {'observed_activity': 'Supplies ticket and account workflow software.',
            'reason': 'The two supplied pages jointly support the connected offering.',
            'missing_requirements': [], 'role': 'supplier_operator',
            'evidence_span_ids': list(value.spans()), 'decision': 'match', **updates}


def test_all_exact_source_spans_survive_json_roundtrip():
    value = packet()
    result = industry.resolve(value, json.dumps(proposal(value)))
    assert len(result['resolved_evidence']) == 2
    assert json.loads(json.dumps(result)) == result
    assert [item['source_url'] for item in result['resolved_evidence']] == [
        'https://example.com/tickets', 'https://example.com/accounts']


@pytest.mark.parametrize('updates', [
    {'missing_requirements': ['Account workflows']},
    {'role': 'customer_user'},
    {'evidence_span_ids': []},
    {'evidence_span_ids': ['another-company:s0:0:50']},
])
def test_partial_or_unbound_positive_cannot_pass(updates):
    value = packet()
    with pytest.raises(ValueError):
        industry.resolve(value, proposal(value, **updates))


def test_source_reference_is_bound_to_criterion_and_company():
    value = packet()
    other = industry.prepare({**CRITERION, 'product_service': 'Only billing software'},
                             COMPANY, [{'source_id': sid, 'body': body, 'source_url': url}
                                       for sid, body, url in value.pages], evaluation_date='2026-10-24')
    with pytest.raises(ValueError):
        industry.resolve(other, proposal(value))


def test_request_has_no_prior_verdict_and_uses_frozen_role():
    request = industry.request_payload(packet(), model='openai/gpt-6.1-sol')
    assert request['model'] == 'openai/gpt-6.1-sol'
    assert request['max_tokens'] == 2000
    assert request['reasoning']['effort'] == 'medium'
    data = json.loads(request['messages'][1]['content'])
    assert set(data) == {'criterion', 'company_locator', 'evaluation_date'}
    assert data['evaluation_date'] == '2026-10-24'
    assert data['criterion'] == CRITERION
    assert len(request['messages']) == 4


def test_decoder_citations_are_limited_to_this_packet():
    value = packet()
    request = industry.request_payload(value, model=industry.DEFAULT_MODEL)
    schema = request['response_format']['json_schema']['schema']
    allowed = schema['properties']['evidence_span_ids']['items']['enum']
    assert allowed == list(value.spans())
    malformed = allowed[0] + ' full?'
    assert malformed not in allowed
    other = industry.prepare({**CRITERION, 'industry': 'Other'}, COMPANY, [
        {'source_id': sid, 'body': body, 'source_url': url}
        for sid, body, url in value.pages], evaluation_date='2026-10-24')
    assert not set(other.spans()).intersection(allowed)
    # Provider constraints do not replace host validation.
    with pytest.raises(ValueError, match='unknown or foreign'):
        industry.resolve(value, proposal(value, evidence_span_ids=[malformed]))
    assert 'enum' not in industry.SCHEMA['properties']['evidence_span_ids']['items']


def test_no_sources_permit_only_an_empty_citation_list():
    value = industry.prepare(CRITERION, COMPANY, [], evaluation_date='2026-10-24')
    request = industry.request_payload(value, model=industry.DEFAULT_MODEL)
    citations = request['response_format']['json_schema']['schema']['properties']['evidence_span_ids']
    assert citations['maxItems'] == 0
    assert 'enum' not in citations['items']
    result = industry.resolve(value, proposal(
        value, decision='unavailable', role='unresolved',
        missing_requirements=['No source evidence']))
    assert result['resolved_evidence'] == []


@pytest.mark.parametrize('body', ['Product evidence. ' * 1500, '"' * 24000],
                         ids=['ordinary', 'json-escaping'])
def test_full_native_corpus_passes_arena_request_bounds_without_losing_text(body):
    from lab_arena.operations import validate_operation_request
    pages = [{'source_id': f's{i}', 'source_url': f'https://example.com/p{i}',
              'body': body[:24000]} for i in range(7)]
    value = industry.prepare(CRITERION, COMPANY, pages, evaluation_date='2026-10-24')
    request = industry.request_payload(value, model='openai/gpt-6.1-sol')
    validate_operation_request('openrouter.chat', request)
    restored = [json.loads(m['content'])['untrusted_source_pages'][0]
                for m in request['messages'][2:]]
    assert {page['source_id']: ''.join(span['text'] for part in restored
                                      if part['source_id'] == page['source_id']
                                      for span in part['spans']) for page in pages} == {
        page['source_id']: page['body'] for page in pages}


def test_native_source_bounds_reject_extra_pages_and_oversized_bodies():
    with pytest.raises(ValueError):
        industry.prepare(CRITERION, COMPANY, [
            {'source_id': f's{i}', 'body': 'Source', 'source_url': 'https://example.com'}
            for i in range(industry.MAX_PAGES + 1)], evaluation_date='2026-10-24')
    with pytest.raises(ValueError):
        industry.prepare(CRITERION, COMPANY, [
            {'source_id': 's0', 'body': 'x' * 24001, 'source_url': 'https://example.com'}], evaluation_date='2026-10-24')


@pytest.mark.parametrize('character', ['X', '"', '\U0001f30d'])
def test_retry_corpus_fits_citation_schema_without_losing_source_bytes(character):
    from lab_arena.operations import validate_operation_request
    # Uneven page lengths can exceed 128 spans even below the text ceiling.
    pages = [{'source_id': f's{i}', 'source_url': f'https://example.com/{i}',
              'body': character * (6401 if i < 14 else 4801)} for i in range(30)]
    value = industry.prepare(CRITERION, COMPANY, pages, evaluation_date='2026-10-24')
    request = industry.request_payload(value, model=industry.DEFAULT_MODEL)
    validate_operation_request('openrouter.chat', request)
    assert len(value.spans()) <= 128
    for source_id, body, _ in value.pages:
        spans = [s for s in value.spans().values() if s['source_id'] == source_id]
        assert ''.join(s['text'] for s in spans) == body
        assert all(len(s['text']) <= 2000 for s in spans)
        assert spans[0]['start_byte'] == 0
        assert spans[-1]['end_byte'] == len(body.encode('utf-8'))


def test_unavailable_needs_a_missing_requirement():
    value = packet()
    with pytest.raises(ValueError):
        industry.resolve(value, proposal(value, decision='unavailable'))


@pytest.mark.parametrize('decision', ['unavailable', 'mismatch'])
def test_initial_research_query_does_not_change_final_evidence_contract(decision):
    from lab_arena.operations import validate_operation_request
    value = packet()
    request = industry.request_payload(value, model=industry.DEFAULT_MODEL, research=True)
    validate_operation_request('openrouter.chat', request)
    assert 'research_query' in request['response_format']['json_schema']['schema']['required']
    response = proposal(value, decision=decision, missing_requirements=['Account workflows'],
                        research_query='account workflow integration')
    result = industry.resolve(value, response, research=True)
    assert result['decision'] == decision
    assert result['packet_sha256'] == value.digest
    assert len(result['resolved_evidence']) == 2
    with pytest.raises(ValueError, match='invalid result shape'):
        industry.resolve(value, response)
    final = industry.request_payload(value, model=industry.DEFAULT_MODEL)
    assert 'research_query' not in final['response_format']['json_schema']['schema']['properties']
    assert 'research_query' not in industry.SCHEMA['properties']


@pytest.mark.parametrize('decision', ['unavailable', 'mismatch'])
@pytest.mark.parametrize('query', ['', ' ', None, 42, 'x' * 301, 'account\nworkflow'])
def test_initial_discovery_query_is_bounded_and_required_for_a_gap(query, decision):
    value = packet()
    with pytest.raises(ValueError, match='invalid research query'):
        industry.resolve(value, proposal(value, decision=decision,
            missing_requirements=['Account workflows'], research_query=query), research=True)


def test_research_query_cannot_supply_evidence_or_change_supplier_role():
    value = packet()
    for update in [{'evidence_span_ids': []}, {'role': 'customer_user'}]:
        with pytest.raises(ValueError):
            industry.resolve(value, proposal(value, research_query='', **update), research=True)
    with pytest.raises(ValueError, match='invalid research query'):
        industry.resolve(value, proposal(value, research_query='unnecessary query'), research=True)
