"""An old citation is a locator, not a reason to spend the shared read budget."""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_company_evidence_investigator import _finding


@pytest.mark.parametrize('choose_old_source', [False, True])
def test_researcher_selects_old_citation_only_when_useful(monkeypatch, choose_old_source):
    old = 'https://www.coreworks.ai/old-profile'
    selected = ([old] if choose_old_source else [
        'https://www.coreworks.ai/analytics',
        'https://www.coreworks.ai/reporting',
        'https://www.coreworks.ai/workflows',
    ])
    fetched = []
    def assess(value):
        if not set(selected) <= set(fetched):
            value.update(decision='unavailable', missing_requirements=['Connected offering'])
    run, _, _, reviews = install(monkeypatch, mutation=assess)
    post = c._post_json
    pending = list(selected)
    turns = []

    async def search(*args, **kwargs):
        return {'results': [{'url': u, 'title': 'Company product'} for u in selected],
                'notice': 'discovery_only_not_evidence'}

    async def fetch(session, url, **kwargs):
        fetched.append(url)
        if url == old and not choose_old_source:
            return {'ok': False, 'error': 'empty_page'}
        return {'ok': True, 'url': url, 'final_url': url, 'text': FEATURES}

    async def no_document(*args, **kwargs):
        return {'ok': False, 'error': 'source_not_found'}

    async def respond(session, url, *, headers, payload):
        if 'response_format' in payload:
            return await post(session, url, headers=headers, payload=payload)
        turns.append(payload)
        if pending:
            return 200, _response('fetch_page', {'url': pending.pop(0)})
        return 200, _response('submit_findings', {'findings': [_finding('industry')]})

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_fetch_page', fetch)
    monkeypatch.setattr(c, '_fetch_exa_exact_page', no_document)
    monkeypatch.setattr(c, '_post_json', respond)
    result = asyncio.run(run(
        prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
        prior_observations={'submitted_source_urls': [URL],
                            'industry_evidence_url': old,
                            'industry_evidence_quote': 'UNTRUSTED PRIOR CLAIM'},
    ))
    assert fetched == selected
    assert old in json.dumps(turns[0])  # Still available for selection.
    assert 'UNTRUSTED PRIOR CLAIM' not in json.dumps(reviews)
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert result['usage']['fetch_calls'] == len(selected)
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
