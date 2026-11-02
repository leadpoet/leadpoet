"""Do not reroll a completed verdict on identical company-bound evidence."""
import asyncio
import pytest

from qualification.scoring import company_evidence_investigator as c

from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL


def test_same_packet_completed_match_is_not_sampled_again(monkeypatch):
    seen = []

    def second_call_disagrees(result):
        seen.append(result['decision'])
        if len(seen) > 1:
            result.update(decision='unavailable', missing_requirements=['Account workflows'])

    run, calls, _, reviews = install(monkeypatch, mutation=second_call_disagrees)
    result = asyncio.run(run(
        prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
        prior_observations={'submitted_source_urls': [URL]},
    ))
    assert result['_completed_submit'] is True
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert len(reviews) == result['usage']['industry_review_calls'] == 1
    assert reviews[0]['reasoning']['effort'] == 'medium'
    assert 'research_query' in reviews[0]['response_format']['json_schema']['schema']['properties']
    assert result['claims']['industry']['_industry_review']['reused_completed_packet'] is True
    assert not calls['search'] and not calls['primary']


@pytest.mark.parametrize('change', ['body', 'criterion', 'company', 'model', 'redirect'])
def test_changed_packet_or_authority_cannot_reuse_a_match(monkeypatch, change):
    seen = []

    def second_call_rejects(result):
        seen.append(result['decision'])
        if len(seen) > 1:
            result.update(decision='unavailable', missing_requirements=['Account workflows'])

    run, _, _, reviews = install(monkeypatch, mutation=second_call_rejects)
    original_review = c._review_industry_sources

    async def review_changed_evidence(*args, **kwargs):
        if not kwargs.get('research'):
            if change == 'body':
                kwargs['fetched_pages'][URL] += ' Coreworks no longer offers account workflows.'
            elif change == 'criterion':
                kwargs['criteria'] = {**kwargs['criteria'], 'product_service': 'Also supplies voice support'}
            elif change == 'company':
                kwargs['company_locator'] = {**kwargs['company_locator'], 'name': 'Another company'}
            elif change == 'model':
                kwargs['model'] = 'another/model'
            else:
                kwargs['fetched_final_urls'][URL] = 'https://another.example/'
        return await original_review(*args, **kwargs)

    monkeypatch.setattr(c, '_review_industry_sources', review_changed_evidence)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert result['claims']['industry']['status'] == 'UNPROVEN'
    assert not result['claims']['industry']['_industry_review'].get('reused_completed_packet')
    assert len(reviews) == (1 if change == 'redirect' else 2)


@pytest.mark.parametrize('decision', ['unavailable', 'mismatch'])
def test_unproven_initial_assessment_is_not_a_reusable_verdict(monkeypatch, decision):
    # A supplier mismatch remains UNPROVEN under the existing native rules.
    run, _, _, reviews = install(monkeypatch, decision=decision, missing=['Account workflows'])
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert result['claims']['industry']['status'] == 'UNPROVEN'
    assert not result['claims']['industry']['_industry_review'].get('reused_completed_packet')
    assert len(reviews) >= 2


def test_same_packet_completed_mismatch_is_preserved_without_mutating_initial(monkeypatch):
    run, _, _, reviews = install(monkeypatch, decision='mismatch',
                                mutation=lambda result: result.update(role='customer_user'))
    original_review = c._review_industry_sources
    reused = []

    async def retain_initial(*args, **kwargs):
        result = await original_review(*args, **kwargs)
        if kwargs.get('research'):
            before = len(reviews)
            assert reviews[-1]['reasoning']['effort'] == 'medium'
            assert 'research_query' in reviews[-1]['response_format']['json_schema']['schema']['properties']
            final, error, calls = await original_review(*args, **{
                **kwargs, 'research': False, 'admitted_final_judgment': True,
                'completed_review': result[0],
            })
            assert final['status'] == 'CONTRADICTED'
            assert final['_industry_review']['reused_completed_packet'] is True
            assert not result[0]['_industry_review'].get('reused_completed_packet')
            assert not error and calls == 0 and len(reviews) == before
            reused.append(final)
        return result

    monkeypatch.setattr(c, '_review_industry_sources', retain_initial)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert result['claims']['industry']['status'] == 'CONTRADICTED'
    assert len(reused) == 1
