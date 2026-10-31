"""The reserved industry judgment must settle an admitted shared submission."""
import asyncio

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install


@pytest.mark.parametrize('decision', ['match', 'mismatch', 'unavailable'])
@pytest.mark.parametrize('crossing', ['fetch', 'submit'])
def test_admitted_submission_finishes_reserved_industry_review(monkeypatch, decision, crossing):
    run, calls, clock, reviews = install(
        monkeypatch, decision=decision,
        missing=['Account workflow integration'] if decision == 'unavailable' else [])
    original_post = c._post_json
    original_fetch = c._fetch_page

    async def fetch(*args, **kwargs):
        result = await original_fetch(*args, **kwargs)
        if len(calls['primary']) == 2:
            clock.now = c.ADMISSION_DEADLINE_SECONDS + (5 if crossing == 'fetch' else -1)
        return result

    async def post(session, url, **kwargs):
        result = await original_post(session, url, **kwargs)
        tool_calls = result[1].get('choices', [{}])[0].get('message', {}).get('tool_calls', [])
        if crossing == 'submit' and any(
                call.get('function', {}).get('name') == 'submit_findings'
                for call in tool_calls):
            clock.now = c.ADMISSION_DEADLINE_SECONDS + 5
        return result

    monkeypatch.setattr(c, '_fetch_page', fetch)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run())
    assert result['_completed_submit'] is True
    assert result['failure_reason'] == ''
    assert result['claims']['industry']['_industry_review']['decision'] == decision
    assert result['claims']['industry']['status'] == {
        'match': 'VERIFIED', 'mismatch': 'UNPROVEN', 'unavailable': 'UNPROVEN'}[decision]
    # The existing relationship gate keeps a supplier's different activity
    # unproven. The deadline fix must not turn that negative into an admission.
    assert len(reviews) == result['usage']['industry_review_calls'] == 1
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    assert len(calls['primary']) == 2
    assert len(calls['search']) == 1  # Initial empty-corpus discovery.


def test_unadmitted_industry_review_still_stops_at_research_deadline(monkeypatch):
    _, _, clock, reviews = install(monkeypatch)
    clock.now = c.ADMISSION_DEADLINE_SECONDS + 5
    finding, failure, count = asyncio.run(c._review_industry_sources(
        None, model='openai/gpt-6.1-sol', openrouter_key='offline-key', started=0,
        company_locator={}, criteria={}, fetched_pages={}, fetched_final_urls={},
        first_party_domains=set(), identity_anchor={}, identity_names=[], evaluation_date_value='2026-10-24'))
    assert finding['status'] == 'UNPROVEN'
    assert failure == c.ADMISSION_BUDGET_INTERRUPTED_FAILURE_REASON
    assert count == 0 and not reviews
