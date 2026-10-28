"""Discovery failures stay operational failures, never company non-fit proof."""
import asyncio

from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL


def test_discovery_provider_failure_does_not_become_a_negative_verdict(monkeypatch):
    run, _, _, reviews = install(monkeypatch, decision='unavailable',
        missing=['Account workflows'], locator_transport=True)
    async def search(*args, **kwargs):
        raise RuntimeError('search_provider_unavailable')
    monkeypatch.setattr(c, '_search_web', search)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert len(reviews) == 1
    assert result['failure_reason'] == 'provider_error'
    assert not result['claims'] and not result.get('_completed_submit')
