"""The host's frozen as-of date binds selection, judgment and verdict reuse."""
import asyncio
from datetime import date
import json
import time

import pytest

from qualification.scoring import industry_classification as industry
from qualification.scoring import company_evidence_investigator as investigator

CRITERIA = {'industry': 'Software', 'subindustry': '',
            'product_service': 'Supplies reporting software.'}
COMPANY = {'name': 'Acme', 'website': 'https://acme.example'}
URL = 'https://acme.example/product'
PAGES = [{'source_id': 's0', 'source_url': URL,
          'body': 'Acme supplies reporting software through June 30, 2026.'}]


def test_packet_requires_explicit_frozen_evaluation_date():
    with pytest.raises(TypeError):
        industry.prepare(CRITERIA, COMPANY, PAGES)


@pytest.mark.parametrize('value', [None, '', 20261024, '2026-02-30', '2026-1-2',
                                   '20261024', '2026-10-24T00:00:00Z'])
def test_packet_rejects_invalid_or_noncanonical_dates(value):
    with pytest.raises(ValueError, match='evaluation date'):
        industry.prepare(CRITERIA, COMPANY, PAGES, evaluation_date=value)


@pytest.mark.parametrize('research', [False, True])
def test_date_binds_requests_digest_and_citations_without_changing_sources(research):
    before = industry.prepare(CRITERIA, COMPANY, PAGES, evaluation_date='2026-06-30')
    after = industry.prepare(CRITERIA, COMPANY, PAGES, evaluation_date='2026-07-01')
    assert before.pages == after.pages
    assert before.criteria == after.criteria
    assert before.digest != after.digest
    assert not set(before.spans()).intersection(after.spans())
    for packet, expected in [(before, '2026-06-30'), (after, '2026-07-01')]:
        request = industry.request_payload(packet, model=industry.DEFAULT_MODEL, research=research)
        metadata = json.loads(request['messages'][1]['content'])
        assert metadata['evaluation_date'] == expected
        assert metadata['criterion'] == CRITERIA
        assert industry.CRITERION_RULES in request['messages'][0]['content']
        assert 'as of the supplied evaluation_date' in request['messages'][0]['content']
        assert request['messages'][2:] == industry.source_messages(packet)


def test_completed_verdict_can_only_be_reused_on_the_same_evaluation_date(monkeypatch):
    requests = []

    async def post(_session, _url, *, payload, **_kwargs):
        requests.append(payload)
        spans = [span['span_id'] for msg in payload['messages'][2:]
                 for page in json.loads(msg['content'])['untrusted_source_pages']
                 for span in page['spans']]
        result = {'observed_activity': 'Reporting software', 'reason': 'Own supplied service.',
                  'missing_requirements': [], 'role': 'supplier_operator',
                  'evidence_span_ids': spans, 'decision': 'match'}
        return 200, {'choices': [{'finish_reason': 'stop',
                                'message': {'content': json.dumps(result)}}]}

    monkeypatch.setattr(investigator, '_post_openrouter_json', post)

    async def review(as_of, previous=None):
        return await investigator._review_industry_sources(
            None, model=industry.DEFAULT_MODEL, openrouter_key='offline-key',
            started=time.monotonic(), company_locator=COMPANY, criteria=CRITERIA,
            evaluation_date_value=as_of,
            fetched_pages={URL: PAGES[0]['body']}, fetched_final_urls={URL: URL},
            first_party_domains={'acme.example'}, identity_names={'acme'},
            identity_anchor={'submitted_domain': 'acme.example', 'observed_domain': 'acme.example'},
            completed_review=previous)

    first, failure, count = asyncio.run(review('2026-06-30'))
    assert first['status'] == 'VERIFIED' and not failure and count == 1
    same, failure, count = asyncio.run(review('2026-06-30', first))
    assert same['_industry_review']['reused_completed_packet'] is True
    assert not failure and count == 0 and len(requests) == 1
    later, failure, count = asyncio.run(review('2026-07-01', first))
    assert not failure and count == 1 and len(requests) == 2
    assert not later['_industry_review'].get('reused_completed_packet')
    assert later['_industry_review']['packet_sha256'] != first['_industry_review']['packet_sha256']
    # A prior date's returned citation IDs cannot be replayed into the new packet.
    packet = industry.prepare(CRITERIA, COMPANY, PAGES, evaluation_date='2026-07-01')
    previous_packet = industry.prepare(CRITERIA, COMPANY, PAGES,
                                       evaluation_date='2026-06-30')
    with pytest.raises(ValueError, match='unknown or foreign'):
        industry.resolve(packet, {
            'observed_activity': 'Reporting software', 'reason': 'Replayed old citations.',
            'missing_requirements': [], 'role': 'supplier_operator',
            'evidence_span_ids': list(previous_packet.spans()), 'decision': 'match'})


def test_advisory_final_and_source_sharing_use_one_frozen_investigation_date(monkeypatch):
    from tests.test_native_industry_review import install
    from tests.test_company_evidence_refusal_fallback import FEATURES

    def unavailable(result):
        result.update(decision='unavailable', missing_requirements=['Account workflow'])

    run, calls, _, reviews = install(monkeypatch, mutation=unavailable)
    dates = iter([date(2026, 10, 24), date(2026, 10, 25)])
    monkeypatch.setattr(investigator, 'evaluation_date', lambda: next(dates))
    prepared = []
    original = industry.prepare

    def prepare(*args, **kwargs):
        packet = original(*args, **kwargs)
        prepared.append((kwargs, packet))
        return packet

    monkeypatch.setattr(industry, 'prepare', prepare)
    url = 'https://www.coreworks.ai/'
    result = asyncio.run(run(
        prefetched_pages={url: {'final_url': url, 'text': FEATURES}},
        prior_observations={'submitted_source_urls': [url]}))
    assert result['_completed_submit'] is True
    assert len(reviews) == 2
    assert len(prepared) >= 3  # Includes the researcher source-sharing packet.
    assert all(options.get('evaluation_date') == '2026-10-24'
               for options, _packet in prepared)
    assert all(json.loads(request['messages'][1]['content'])['evaluation_date'] == '2026-10-24'
               for request in reviews)
    assert calls['model']
    document = next(json.loads(message['content'].split('\n', 1)[1])
                    for message in calls['model'][0]['messages']
                    if message['content'].startswith('Investigate this bounded request.'))
    assert document['evaluation_date'] == '2026-10-24'
    # The getter was used once; the next host date was never substituted later.
    assert next(dates) == date(2026, 10, 25)
