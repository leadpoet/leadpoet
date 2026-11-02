"""Retain source bodies without spending another judgment before selection."""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_company_evidence_investigator import _finding

SOURCE = 'https://www.coreworks.ai/workflows'
BODY = 'Coreworks supplies account workflow management with its reporting software.'


@pytest.mark.parametrize('source_kind', ['new', 'same', 'foreign'])
def test_full_source_discovery_flows_to_research_and_final_review_once(monkeypatch, source_kind):
    def assess(value):
        if len(value['evidence_span_ids']) < 2:
            value.update(decision='unavailable', missing_requirements=['Account workflows'])
    run, _, _, reviews = install(monkeypatch, mutation=assess)
    original = c._post_json
    turns, searches = [], []
    source = URL if source_kind == 'same' else (
        'https://other.example/workflows' if source_kind == 'foreign' else SOURCE)

    async def search(session, query, **kwargs):
        searches.append(kwargs)
        return {'results': [{'url': source, 'title': 'Product documentation'}],
                '_source_pages': {source: BODY}, 'notice': 'discovery_only_not_evidence'}

    async def post(session, url, *, headers, payload):
        if 'response_format' in payload:
            return await original(session, url, headers=headers, payload=payload)
        turns.append(payload)
        return 200, _response('submit_findings', {'findings': [_finding(
            'industry', status='UNPROVEN', observed_value=None, evidence_url='', evidence_quote='')]})

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run(
        prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
        prior_observations={'submitted_source_urls': [URL]}))
    assert len(searches) == 1 and searches[0].get('read_source_text') is True
    assert (BODY in json.dumps(turns[0])) is (source_kind == 'new')
    assert 'usage' in result, result
    assert len(reviews) == result['usage']['industry_review_calls'] == 2
    assert result['claims']['industry']['status'] == ('VERIFIED' if source_kind == 'new' else 'UNPROVEN')
    assert result['usage']['fetch_calls'] == 0
    assert result['usage']['search_calls'] == 1
