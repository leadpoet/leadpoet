"""Only an exact employer-bound native API posting supplies its first date."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import intent_verification_three_stage as verifier
from qualification.scoring import lead_scorer
from qualification.scoring.evaluation_clock import use_evaluation_date
from tests.test_arena_score_integrity import _company_model, _icp_model


ASHBY = 'https://jobs.ashbyhq.com/fluency/c6aceb26-9658-48c2-ac07-6cdf8bbb38df'
GREENHOUSE = 'https://job-boards.greenhouse.io/monzo/jobs/6635837'


def native_result(url=ASHBY, *, notes=(), published='2025-07-21', decision='approve'):
    provider = 'greenhouse' if url == GREENHOUSE else 'ashby'
    return {
        'client_ready': decision == 'approve', 'decision': decision,
        'rejection_reason': '' if decision == 'approve' else 'stage3_contradicted',
        'stage1': {'status': 'supported'},
        'stage3': {'status': 'supported', 'decision': decision,
                   'claim_matches_miner_date': 'no_date_in_content'},
        'company_check': True, 'job_publisher_relationship': 'verified',
        'verified_job_source_urls': [url], 'source_publication_dates': [published],
        'scrape': {'result_count': 1, 'statuses': [{
            'url': url, 'source': f'scrapingdog_{provider}_api',
            'stage': f'sd:{provider}_api:1',
        }]},
        'verdict': {'signal_evaluations': [{
            'signal_status': 'supported', 'verification_mode': 'source_grounded',
            'same_entity_check': 'pass', 'confidence': 'high',
            'risk_notes': list(notes), 'unsupported_parts': [],
            'evidence_urls_used': [url], 'supporting_quotes': ['We are hiring engineers.'],
        }]},
    }


def score(monkeypatch, result, *, url=ASHBY, integrity=True, evidence_type='HIRING', compound=False):
    before = deepcopy(result)
    monkeypatch.setattr(verifier, 'verify_three_stage', AsyncMock(return_value=result))
    signal = _company_model([{
        'source': 'job_board', 'url': url, 'description': 'Acme is hiring an infrastructure engineer.',
        'date': None, 'snippet': 'Engineering role with an application form.', 'matched_icp_signal': 0,
    }]).intent_signals[0]
    icp = _icp_model().model_copy(update={
        'intent_signals': ['Actively hiring engineers per current job postings.'],
        'intent_signal_evidence_types': [evidence_type], 'intent_max_age_days': 365,
    })
    evidence = [signal, signal.model_copy(update={'url': ASHBY.replace('c6aceb26', 'a6aceb26')})] if compound else None
    receipts = []
    with use_evaluation_date('2026-11-07'):
        actual = asyncio.run(lead_scorer._score_single_intent_signal(
            signal, icp, None, 'Acme', 'https://acme.com',
            stage1_soft_reject=True, llm_only_intent_gate=True,
            trust_signal_date=True, integrity_policy=integrity,
            verdict_out=receipts, evidence_signals=evidence,
        ))
    assert result == before
    return actual, receipts


@pytest.mark.parametrize('url', [ASHBY, GREENHOUSE])
@pytest.mark.parametrize('published', ['2025-02-14', '2025-07-21', '2025-08-29', '2025-11-05'])
@pytest.mark.parametrize('notes', [(), ('source_event_date:2026-11-07',),
                                  ('source_publication_date:2026-11-07',)])
def test_old_native_first_date_rejects_missing_or_invented_model_date(monkeypatch, url, published, notes):
    result = json.loads(json.dumps(native_result(url, notes=notes, published=published)))
    actual, receipts = score(monkeypatch, result, url=url)
    assert actual[0] == 0 and actual[2] == 'out_of_window'
    assert receipts[-1]['authoritative_date'] == published
    assert receipts[-1]['rejection_reason'] == 'source_grounded_event_out_of_window'


@pytest.mark.parametrize('published', ['2025-11-07', '2026-05-27', '2026-10-01'])
def test_valid_current_native_posting_remains_eligible(monkeypatch, published):
    actual, _ = score(monkeypatch, native_result(published=published))
    assert actual[0] > 0 and actual[2] == 'uncertain'


def test_newer_api_date_does_not_clear_an_existing_model_date_rejection(monkeypatch):
    actual, receipts = score(monkeypatch, native_result(
        published='2026-10-01', notes=['source_event_date:2025-01-01']))
    assert actual[0] == 0 and actual[2] == 'out_of_window'
    assert receipts[-1]['authoritative_date'] == '2025-01-01'


@pytest.mark.parametrize('fault', [
    'wrong_path', 'query', 'fragment', 'userinfo', 'origin_only', 'wrong_provider',
    'fallback', 'wrong_stage', 'unverified', 'wrong_verified_url', 'multiple_dates',
    'invalid_date', 'multiple_results', 'boolean_result_count', 'multiple_statuses', 'generic_source',
])
def test_unbound_metadata_keeps_the_existing_date_path(monkeypatch, fault):
    result = native_result()
    status = result['scrape']['statuses'][0]
    if fault == 'wrong_path': status['url'] = ASHBY.replace('c6aceb26', 'a6aceb26')
    elif fault == 'query': status['url'] += '?secret=untrusted'
    elif fault == 'fragment': status['url'] += '#untrusted'
    elif fault == 'userinfo': status['url'] = ASHBY.replace('https://', 'https://user@')
    elif fault == 'origin_only': status['url'] = 'https://jobs.ashbyhq.com'
    elif fault == 'wrong_provider': status['source'] = 'scrapingdog_greenhouse_api'
    elif fault == 'fallback': status['source'] += '_fallback'
    elif fault == 'wrong_stage': status['stage'] = 'ashby_api_exhausted'
    elif fault == 'unverified': result['job_publisher_relationship'] = 'unverified'
    elif fault == 'wrong_verified_url': result['verified_job_source_urls'] = [ASHBY.replace('c6aceb26', 'a6aceb26')]
    elif fault == 'multiple_dates': result['source_publication_dates'].append('2026-11-07')
    elif fault == 'invalid_date': result['source_publication_dates'] = ['2025-02-30']
    elif fault == 'multiple_results': result['scrape']['result_count'] = 2
    elif fault == 'boolean_result_count': result['scrape']['result_count'] = True
    elif fault == 'multiple_statuses': result['scrape']['statuses'].append(dict(status))
    elif fault == 'generic_source': status['source'] = 'scrapingdog'
    actual, _ = score(monkeypatch, result)
    assert actual[2] == 'uncertain' and actual[0] > 0


@pytest.mark.parametrize('option', ['no_policy', 'non_hiring', 'compound'])
def test_other_scopes_keep_existing_rules(monkeypatch, option):
    actual, _ = score(monkeypatch, native_result(), integrity=option != 'no_policy',
                      evidence_type='PRODUCT_LAUNCH' if option == 'non_hiring' else 'HIRING',
                      compound=option == 'compound')
    assert actual[2] != 'out_of_window'


def test_classifier_rejection_is_not_promoted(monkeypatch):
    actual, _ = score(monkeypatch, native_result(published='2026-10-01', decision='reject'))
    assert actual[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['ashby', 'greenhouse'])
@pytest.mark.parametrize('status_fault', [None, 'query', 'fragment', 'wrong_path', 'userinfo'])
@pytest.mark.parametrize('kind', ['ashby_job', 'greenhouse_job', ''])
async def test_native_api_status_preserves_exact_private_url_but_not_prompt_status(monkeypatch, provider, status_fault, kind):
    posting_url = ASHBY if provider == 'ashby' else 'https://job-boards.greenhouse.io/fluency/jobs/6635837'
    status_url = posting_url
    if status_fault == 'query': status_url += '?untrusted=1'
    elif status_fault == 'fragment': status_url += '#untrusted'
    elif status_fault == 'wrong_path': status_url = posting_url.replace('/fluency/', '/other/')
    elif status_fault == 'userinfo': status_url = posting_url.replace('https://', 'https://user@')
    body = (Path(__file__).parent/'fixtures/fluency_ashby_ai_platform.txt').read_text()
    data = {'results': [{'url': posting_url, 'text': body, 'meta': {'kind': kind},
                         'source_publication_date': '2025-07-21'}],
            'statuses': [{'url': status_url, 'source': f'scrapingdog_{provider}_api', 'stage': f'sd:{provider}_api:1'}]}
    projected_status_url = verifier._project_contents_for_prompt(data)['statuses'][0]['url']
    assert projected_status_url != posting_url

    async def fetch(urls, **_kwargs):
        if urls == [posting_url]: return deepcopy(data)
        return {'results': [{'url': urls[0], 'text': 'Official company careers', 'meta': {
            'observed_ownership_links': [{'url': ('https://jobs.ashbyhq.com/fluency' if provider == 'ashby'
                else 'https://job-boards.greenhouse.io/fluency'), 'label': 'Careers'}],
        }}], 'statuses': []}

    answer = native_result()['verdict']
    answer['signal_evaluations'][0].update(signal_id='signal-1', claim='Fluency is hiring engineers.',
        source_accessibility='accessible', source_urls_supplied=[posting_url], contradicting_quotes=[],
        evidence_urls_used=[posting_url], claim_matches_miner_date='no_date_in_content')
    calls = AsyncMock(return_value={'answer': answer, 'model': 'test', 'usage': {}})
    monkeypatch.setattr(verifier, '_call_openrouter', calls)
    monkeypatch.setattr(verifier, '_fetch_sd_then_exa', fetch)
    actual = await verifier.verify_three_stage(
        None, company_name='Fluency', company_website='https://usefluency.com',
        company_linkedin='https://www.linkedin.com/company/usefluency', source_url=posting_url,
        miner_claim='Fluency is hiring engineers.', target_signal_text='Actively hiring engineers.',
        evidence_type='HIRING', declared_source='job_board', stage1_soft_reject=True,
        integrity_policy=True, buyer_max_age_days=365,
        verified_company_identity={'decision': 'match', 'observed_name': 'fluency',
            'observed_domain': 'usefluency.com', 'observed_linkedin_slug': 'usefluency',
            'evidence_source': 'company_web_reverification'},
    )
    assert actual['decision'] == 'approve'
    assert posting_url in actual['verified_job_source_urls']
    assert actual['scrape']['statuses'][0]['url'] == (posting_url if status_fault is None and kind == provider+'_job' else projected_status_url)
    assert data['statuses'][0]['url'] == status_url


@pytest.mark.parametrize('url', [ASHBY+'?untrusted=1', ASHBY+'#untrusted',
    ASHBY.replace('https://', 'https://user@'), ASHBY.replace('.com/', '.com:444/'),
    ASHBY.replace('https://', 'http://'), '\n'+ASHBY,
    ASHBY.replace('jobs.ashbyhq.com', 'unrelated.example')])
def test_raw_status_cannot_introduce_a_noncanonical_posting(url):
    status = native_result()['scrape']['statuses'][0]
    status['url'] = url
    assert verifier._exact_native_ats_api_status(url, [status]) is None


def test_private_api_status_retains_only_its_three_bounded_fields():
    status = native_result()['scrape']['statuses'][0]
    expected = deepcopy(status)
    status['untrusted_extra'] = 'Do not put this in a trace.'
    assert verifier._exact_native_ats_api_status(ASHBY, [status]) == expected


@pytest.mark.asyncio
async def test_concurrent_postings_do_not_share_their_dates(monkeypatch):
    results = {ASHBY: native_result(), GREENHOUSE: native_result(GREENHOUSE, published='2026-10-01')}
    before = deepcopy(results)

    async def judge(*_args, **kwargs):
        await asyncio.sleep(0)
        return json.loads(json.dumps(results[kwargs['source_url']]))

    monkeypatch.setattr(verifier, 'verify_three_stage', judge)
    icp = _icp_model().model_copy(update={'intent_signals': ['Actively hiring engineers.'],
        'intent_signal_evidence_types': ['HIRING'], 'intent_max_age_days': 365})
    signals = [_company_model([{'source': 'job_board', 'url': url, 'date': None,
        'description': 'Acme is hiring an infrastructure engineer.',
        'snippet': 'Engineering role with an application form.',
        'matched_icp_signal': 0}]).intent_signals[0] for url in results]
    with use_evaluation_date('2026-11-07'):
        old, current = await asyncio.gather(*[
            lead_scorer._score_single_intent_signal(signal, icp, None, 'Acme', 'https://acme.com',
                stage1_soft_reject=True, llm_only_intent_gate=True,
                trust_signal_date=True, integrity_policy=True)
            for signal in signals])
    assert old[0] == 0 and old[2] == 'out_of_window'
    assert current[0] > 0 and current[2] == 'uncertain'
    assert results == before
