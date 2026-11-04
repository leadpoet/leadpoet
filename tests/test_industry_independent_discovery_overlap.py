"""Independent stage discovery must not consume the industry read window."""
import asyncio

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_native_industry_review import install
from tests.test_company_evidence_refusal_fallback import FEATURES, URL
from tests.test_company_evidence_investigator import _finding
from tests.test_company_evidence_judgment_reserve import _response


def _case(monkeypatch):
    findings = [
        _finding('industry', evidence_url=URL, evidence_quote=FEATURES),
        _finding('stage', status='UNPROVEN', observed_value=None,
                 evidence_url='', evidence_quote=''),
    ]
    run, calls, clock, reviews = install(monkeypatch, finding=findings)
    investigate = c.investigate_company_evidence

    async def invoke(**kwargs):
        kwargs['targets'] = ('stage', 'industry')
        return await investigate(**kwargs)

    monkeypatch.setattr(c, 'investigate_company_evidence', invoke)
    original_post = c._post_json

    async def post(session, url, *, headers, payload):
        if 'response_format' in payload:
            return await original_post(session, url, headers=headers, payload=payload)
        calls['model'].append(payload)
        return 200, _response('submit_findings', {'findings': findings})

    monkeypatch.setattr(c, '_post_json', post)

    async def perform():
        return await run(requested_stage='Private Equity',
                         prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                         prior_observations={'submitted_source_urls': [URL]})

    return perform, calls, clock, reviews


def test_independent_advisory_and_stage_search_overlap_without_more_calls(monkeypatch):
    perform, calls, _, reviews = _case(monkeypatch)
    post = c._post_json
    review_started = asyncio.Event()
    overlapped = []

    async def search(session, query, **kwargs):
        calls['search'].append(query)
        try:
            await asyncio.wait_for(review_started.wait(), timeout=0.1)
            overlapped.append(True)
        except asyncio.TimeoutError:
            overlapped.append(False)
        return {'results': [], 'notice': 'discovery_only_not_evidence'}

    async def review(session, url, *, headers, payload):
        if 'response_format' in payload:
            review_started.set()
        return await post(session, url, headers=headers, payload=payload)

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', review)
    result = asyncio.run(perform())
    assert overlapped == [True]
    assert result['_completed_submit'] is True
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert result['claims']['stage']['status'] == 'UNPROVEN'
    assert len(reviews) == result['usage']['industry_review_calls'] == 1
    assert result['usage']['search_calls'] == 1
    assert result['usage']['fetch_calls'] == 0
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS


@pytest.mark.parametrize('failure', ['provider', 'contract'])
def test_failed_advisory_still_waits_for_admitted_stage_call(monkeypatch, failure):
    perform, _, _, _ = _case(monkeypatch)
    stage_started, stage_settled = asyncio.Event(), asyncio.Event()
    review_saw_pending_stage = []

    async def search(session, query, **kwargs):
        stage_started.set()
        await asyncio.sleep(0.02)
        stage_settled.set()
        return {'results': [], 'notice': 'discovery_only_not_evidence'}

    async def post(session, url, *, headers, payload):
        assert 'response_format' in payload
        await stage_started.wait()
        review_saw_pending_stage.append(not stage_settled.is_set())
        if failure == 'contract':
            raise ValueError('offline contract failure')
        return 401, {'error': 'offline provider failure'}

    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(perform())
    assert review_saw_pending_stage == [True]
    assert stage_settled.is_set()
    assert result.get('_completed_submit') is not True
    assert result['failure_reason'] == (
        c.MALFORMED_RESPONSE_FAILURE_REASON if failure == 'contract'
        else c.PROVIDER_ERROR_FAILURE_REASON)


def test_cancellation_does_not_abandon_the_two_admitted_provider_calls(monkeypatch):
    perform, _, _, _ = _case(monkeypatch)

    async def scenario():
        started = {key: asyncio.Event() for key in ('stage', 'review')}
        release = asyncio.Event()
        settled = set()
        post = c._post_json

        async def search(session, query, **kwargs):
            started['stage'].set()
            await release.wait()
            settled.add('stage')
            return {'results': [], 'notice': 'discovery_only_not_evidence'}

        async def review(session, url, *, headers, payload):
            if 'response_format' in payload:
                started['review'].set()
                await release.wait()
                settled.add('review')
            return await post(session, url, headers=headers, payload=payload)

        monkeypatch.setattr(c, '_search_web', search)
        monkeypatch.setattr(c, '_post_json', review)
        task = asyncio.create_task(perform())
        try:
            await asyncio.wait_for(asyncio.gather(*(e.wait() for e in started.values())), 0.1)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert settled == {'stage', 'review'}
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
