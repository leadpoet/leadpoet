"""Discovery failures stay operational failures, never company non-fit proof."""
import asyncio

from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL


def test_discovery_provider_failure_does_not_become_a_negative_verdict(monkeypatch):
    run, _, _, reviews = install(monkeypatch, decision='unavailable',
        missing=['Account workflows'], locator_transport=True)
    original_post = c._post_json
    async def post(session, url, *, headers, payload):
        if payload['model'] == 'perplexity/sonar':
            return 503, {'error': 'provider unavailable'}
        return await original_post(session, url, headers=headers, payload=payload)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert len(reviews) == 1
    assert result['failure_reason'] == 'provider_error'
    assert not result['claims'] and not result.get('_completed_submit')
