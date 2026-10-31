"""Multi-source verdicts retain the existing company attribution requirement."""
import asyncio
import json
import time

import pytest

from qualification.scoring import company_evidence_investigator as c, lead_scorer


@pytest.mark.parametrize('named_support', [True, False])
@pytest.mark.parametrize('separate_identity_source', [True, False])
def test_primary_citation_uses_bound_company_evidence(monkeypatch, named_support, separate_identity_source):
    first = 'https://coreworks.ai/services'
    second = 'https://coreworks.ai/projects'
    pages = {
        first: 'We operate customer infrastructure and migrate workloads to the cloud.',
        second: ('Coreworks' if named_support else 'We') +
                ' modernized a customer mainframe and now operate its cloud environment.',
    }
    identity_url = 'https://coreworks.ai/about'
    if separate_identity_source:
        pages[identity_url] = 'Coreworks provides technology services.'
    requests = []

    async def post(session, url, *, headers, payload, **kwargs):
        requests.append(payload)
        spans = [span for message in payload['messages'][2:]
                 for page in json.loads(message['content'])['untrusted_source_pages']
                 for span in page['spans']]
        result = {'observed_activity': 'Managed infrastructure and modernization services',
                  'reason': 'The two services jointly cover the supplied capabilities.',
                  'missing_requirements': [], 'role': 'supplier_operator',
                  'evidence_span_ids': [span['span_id'] for span in spans[:2]], 'decision': 'match'}
        return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}

    monkeypatch.setattr(c, '_post_openrouter_json', post)
    finding, failure, calls = asyncio.run(c._review_industry_sources(
        None, model='test-model', openrouter_key='offline-key', started=time.monotonic(),
        company_locator={'name': 'Coreworks', 'website': first},
        criteria={'industry': 'IT services', 'subindustry': '',
                  'product_service': 'Managed infrastructure, cloud migration and mainframe modernization'},
        fetched_pages=pages, fetched_final_urls={url: url for url in pages},
        first_party_domains={'coreworks.ai'}, identity_names={'coreworks'},
        identity_anchor={'submitted_domain': 'coreworks.ai', 'observed_domain': 'coreworks.ai'}, evaluation_date_value='2026-10-24'))
    assert calls == len(requests) == 1 and not failure
    assert finding['status'] == ('VERIFIED' if named_support or separate_identity_source else 'UNPROVEN')
    if named_support:
        assert finding['evidence_url'] == second
        assert finding['evidence_quote'] == pages[second]
        assert finding['supporting_evidence'] == [{'url': first, 'quote': pages[first]}]
        projected = lead_scorer._project_investigator_industry({}, finding)
        stored = json.loads(json.dumps(lead_scorer._dimension_web_evidence(projected, 'industry')))
        assert stored == {'url': second, 'quote': pages[second],
                          'supporting_evidence': [{'url': first, 'quote': pages[first]}]}
    elif separate_identity_source:
        assert finding['evidence_url'] == identity_url
        assert finding['evidence_quote'] == pages[identity_url]
        assert finding['supporting_evidence'] == [
            {'url': url, 'quote': pages[url]} for url in (first, second)]
    else:
        assert finding['reason'] == 'source quote did not identify the investigated company'
