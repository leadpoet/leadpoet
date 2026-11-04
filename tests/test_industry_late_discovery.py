"""Near the research deadline, discovery must bring readable source text."""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_company_evidence_investigator import _finding

SOURCE = 'https://www.coreworks.ai/customers/workflows'
BODY = 'Coreworks supplies account workflow management to its customers.'


@pytest.mark.parametrize('elapsed, source_kind', [
    (10, 'bound'), (50, 'bound'), (60, 'bound'), (79, 'bound'),
    (80, 'bound'), (60, 'foreign'), (60, 'empty'),
])
def test_late_exchange_uses_full_text_without_extending_admission(monkeypatch, elapsed, source_kind):
    supplied = {'new': False}

    def assess(result):
        if not supplied['new']:
            result.update(decision='unavailable', missing_requirements=['Account workflows'])

    run, _, clock, reviews = install(monkeypatch, mutation=assess, locator_transport=True)
    original = c._post_json
    searches, alternatives, turns = [], [], []

    async def search(session, query, **options):
        searches.append(options)
        if len(searches) < 3:
            return {'results': []}
        # An admitted full-text request may settle after the research window.
        clock.now = c.ADMISSION_DEADLINE_SECONDS - c.JUDGMENT_ADMISSION_RESERVE_SECONDS
        supplied['new'] = source_kind == 'bound'
        source = SOURCE if source_kind == 'bound' else 'https://other.example/workflows'
        pages = {} if source_kind == 'empty' else {source: BODY}
        return {'results': [{'url': source}], '_source_pages': pages}

    async def post(session, url, *, headers, payload):
        if 'response_format' in payload:
            return await original(session, url, headers=headers, payload=payload)
        if payload['model'] == 'perplexity/sonar':
            alternatives.append(payload)
            return 200, {'choices': [{'message': {'content': SOURCE + ' — ' + BODY}}]}
        turns.append(payload)
        if len(turns) <= 2:
            if len(turns) == 2:
                clock.now = elapsed
            return 200, _response('search_web', {'query': 'Coreworks account workflows'})
        return 200, _response('submit_findings', {'findings': [_finding('industry')]})

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run(
        prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
        prior_observations={'submitted_source_urls': [URL]}))
    full_text = 50 <= elapsed < 80
    assert len(alternatives) == int(elapsed < 50)
    assert len(searches) == 2 + int(full_text)
    if full_text:
        assert searches[-1]['read_source_text'] is True
        assert searches[-1]['include_domains'] == ('coreworks.ai',)
    accepted = full_text and source_kind == 'bound'
    assert result['claims']['industry']['status'] == ('VERIFIED' if accepted else 'UNPROVEN')
    assert (BODY in json.dumps(reviews[-1])) is accepted
    assert result['_completed_submit'] is True
    assert result['usage']['search_calls'] == (3 if elapsed < 80 else 2)
    assert result['usage']['fetch_calls'] == 0
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
