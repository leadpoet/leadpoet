"""An explicit page publication date survives extraction without borrowing cards."""

import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.qualification.models import ICPPrompt, IntentSignal
from qualification.scoring import intent_details, verification_helpers
from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring.arena_integrity import (
    source_dates_from_verdict,
    source_grounded_date_verdict,
)
from qualification.scoring.lead_scorer import _score_single_intent_signal


URL = 'https://business.adobe.com/blog/announcing-adobe-marketing-campaign-analytics'
TITLE = 'Announcing Adobe Marketing Campaign Analytics'
QUOTE = 'Adobe introduces Marketing Campaign Analytics, a new campaign measurement solution.'
PUBLICATION = '2026-04-20'


def _html(meta='<meta name="publication-date" content="2026-04-20">'):
    # Minimal shape from captured SD settlement 6355327, SHA256
    # 4f67ca1014a755e4b4db9292c3330c9ac2b431cc45d2d356aac25c7d94eb962b.
    return (
        '<html><head><title>' + TITLE + '</title>'
        '<link rel="canonical" href="' + URL + '">' + meta + '</head><body>'
        '<main><div class="article-header"><h1>' + TITLE + '</h1>'
        '<div class="article-byline"><p class="article-date">04-20-2026</p></div>'
        '</div><article>' + ('<p>' + QUOTE + '</p>') * 35 + '</article></main>'
        '<aside><a href="https://other.test/unrelated">Other company</a>'
        '<meta name="publication-date" content="2026-10-24">'
        '<time datetime="2026-10-24">October 24, 2026</time></aside>'
        '<div><div>cardDate</div><div>2026-10-24</div></div></body></html>'
    )


@pytest.mark.parametrize('alias', ['publication-date', 'datePublished', 'article:published_time'])
@pytest.mark.parametrize('meta', [
    '<meta name="{alias}" content="2026-04-20">',
    "<meta content='2026-04-20' name='{alias}'>",
    '<meta property="{alias}" content="2026-04-20T13:00:00Z">',
])
def test_explicit_head_publication_alias_ignores_related_cards(alias, meta):
    assert intent._published_date_from_html(_html(meta.format(alias=alias)), URL) == PUBLICATION


@pytest.mark.parametrize('meta', [
    '<meta name="publication-date" content="2026-02-30">',
    '<meta name="publication-date" content="2026-04-20; untrusted suffix">',
    '<meta name="dateModified" content="2026-04-20">',
    '<meta name="article:modified_time" content="2026-04-20">',
    '',
])
def test_invalid_modified_or_body_only_dates_never_supply_publication(meta):
    assert intent._published_date_from_html(_html(meta), URL) == ''


@pytest.mark.parametrize('other', ['publication-date', 'datePublished', 'article:published_time'])
def test_conflicting_explicit_head_dates_remain_unknown(other):
    meta = ('<meta name="publication-date" content="2026-04-20">'
            f'<meta name="{other}" content="2026-04-21">')
    assert intent._published_date_from_html(_html(meta), URL) == ''


def test_modified_date_is_not_a_conflict_and_unclosed_head_is_not_metadata():
    meta = ('<meta name="publication-date" content="2026-04-20">'
            '<meta name="dateModified" content="2026-10-24">')
    assert intent._published_date_from_html(_html(meta), URL) == PUBLICATION
    assert intent._published_date_from_html(_html(meta).replace('</head>', ''), URL) == ''


@pytest.mark.parametrize('alias', ['publication-date', 'datePublished', 'article:published_time'])
@pytest.mark.parametrize('fragment', [
    '<!-- <meta name="{alias}" content="2026-04-20"> -->',
    '<script>const example = \'<meta name="{alias}" content="2026-04-20">\';</script>',
    '<meta data-name="{alias}" content="2026-04-20">',
])
def test_only_actual_head_meta_nodes_and_exact_attributes_can_supply_dates(alias, fragment):
    assert intent._published_date_from_html(_html(fragment.format(alias=alias)), URL) == ''


@pytest.mark.parametrize('meta', [
    '<meta name="publication-date" content="2026-04-20" content="2026-04-21">',
    '<meta name="publication-date" name="dateModified" content="2026-04-20">',
])
def test_duplicate_metadata_attributes_are_not_accepted(meta):
    assert intent._published_date_from_html(_html(meta), URL) == ''


def _install_source(monkeypatch):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _endpoint, **kwargs):
            assert kwargs['params']['url'] == URL
            return httpx.Response(200, text=_html())

    monkeypatch.setenv('SCRAPINGDOG_API_KEY', 'test-only')
    monkeypatch.setattr(intent.httpx, 'AsyncClient', Client)
    # Exercise the production failure mode: article extraction omits the head
    # and visible byline, while preserving independently visible body text.
    monkeypatch.setattr(verification_helpers, 'extract_article_body', lambda _: QUOTE * 35)


def _row():
    return dict(id='signal-1', company='Adobe', website='https://adobe.com',
                company_linkedin='', contact_linkedin='', claim=QUOTE,
                signal_date=PUBLICATION, signal_type='intent',
                claimed_source_urls=[URL], _target_signal_text='Launched a new advertising product',
                _evidence_type='PRODUCT_LAUNCH')


def test_source_extraction_prompt_and_date_grounding_preserve_actual_date(monkeypatch):
    _install_source(monkeypatch)
    scraped = asyncio.run(intent._scrape_sd_hardened(URL))
    assert scraped['ok'] is True
    assert scraped['source_publication_date'] == PUBLICATION
    assert '04-20-2026' not in scraped['content']
    contents = intent._project_contents_for_prompt({'results': [{
        'url': URL, 'text': scraped['content'],
        'source_publication_date': scraped['source_publication_date'],
    }]})
    prompt = intent._build_final_judge_prompt(_row(), contents)
    assert 'SOURCE PAGE PUBLICATION METADATA: 2026-04-20' in prompt
    assert 'SOURCE PAGE PUBLICATION METADATA: 2026-10-24' not in prompt
    event, pubs = source_dates_from_verdict(
        {'risk_notes': ['source_publication_date:2026-04-20']},
        [scraped['source_publication_date']],
    )
    verdict = source_grounded_date_verdict(event_date=event, publication_dates=pubs,
        buyer_cap_days=365, evaluated_on=date(2026, 10, 24))
    assert verdict.authoritative_date == PUBLICATION
    assert verdict.basis == 'publication_date'
    # A source-bound older event still controls; publication cannot refresh it.
    event, pubs = source_dates_from_verdict(
        {'risk_notes': ['source_event_date:2024-01-01', 'source_publication_date:2026-04-20']},
        [scraped['source_publication_date']],
    )
    assert source_grounded_date_verdict(event_date=event, publication_dates=pubs,
        buyer_cap_days=365, evaluated_on=date(2026, 10, 24)).verdict == 'out_of_window'


def _provider(status='supported'):
    return {'answer': {'signal_evaluations': [{
        'signal_id': 'signal-1', 'signal_status': status, 'confidence': 'high',
        'same_entity_check': 'pass', 'verification_mode': 'source_grounded',
        'evidence_urls_used': [URL], 'claim_matches_miner_date': 'consistent',
        'supporting_quotes': [QUOTE], 'unsupported_parts': [],
        'risk_notes': ['source_publication_date:2026-04-20'],
    }], 'overall_verdict': 'qualified' if status == 'supported' else 'needs_review',
        'overall_confidence': 'high', 'summary': 'Offline held judge response.',
        'missing_or_risks': []}, 'model': 'test-only', 'usage': {}}


def _score(monkeypatch, fetched_url=URL, status='supported', claimed_date=PUBLICATION):
    _install_source(monkeypatch)
    scraped = asyncio.run(intent._scrape_sd_hardened(URL))
    monkeypatch.setattr(intent, '_fetch_sd_then_exa', AsyncMock(return_value={
        'results': [{'url': fetched_url, 'text': scraped['content'],
                     'source_title': TITLE,
                     'source_publication_date': scraped['source_publication_date']}],
        'statuses': [],
    }))
    judge = AsyncMock(return_value=_provider(status))
    monkeypatch.setattr(intent, '_call_openrouter', judge)
    monkeypatch.setenv('LEADPOET_COMPETITION_EVALUATION_DATE', '2026-10-24')
    signal = IntentSignal(source='news', description=QUOTE, url=URL,
                          date=claimed_date, snippet=QUOTE, matched_icp_signal=0)
    icp = ICPPrompt(icp_id='head-publication-alias', prompt='Advertising vendors with new products',
        industry='Advertising', sub_industry='Adtech', employee_count='1001-5000',
        company_stage='Public', geography='United States', country='United States',
        product_service='Programmatic advertising platform',
        intent_signals=['Launched a new advertising product or major platform capability in the last year'],
        intent_signal_evidence_types=['PRODUCT_LAUNCH'])
    receipts = []
    scored = asyncio.run(_score_single_intent_signal(signal, icp, None, 'Adobe',
        'https://adobe.com', company_linkedin='https://www.linkedin.com/company/adobe',
        stage1_soft_reject=True, llm_only_intent_gate=True, integrity_policy=True,
        verdict_out=receipts, evidence_signals=[signal]))
    return scored, receipts, judge, signal, icp


def test_held_supported_response_reaches_score_and_bound_paragraph_evidence(monkeypatch):
    scored, receipts, judge, signal, icp = _score(monkeypatch, claimed_date='2026-10-24')
    assert scored[0] > 0
    # Submitted date cannot replace the source's older actual publication.
    assert receipts[-1]['authoritative_date'] == PUBLICATION
    assert receipts[-1]['authoritative_date_basis'] == 'publication_date'
    assert judge.await_count == 1
    company = SimpleNamespace(company_name='Adobe', company_website='https://adobe.com',
        intent_details='Adobe introduced Marketing Campaign Analytics on April 20, 2026.',
        intent_signals=[signal])
    doc = intent_details.review_evidence(company, icp, [{
        'after_decay': scored[0], 'matched_icp_signal': 0,
        'evidence_urls': [URL], 'judge_verdict': receipts[-1],
    }], {'decision': 'match', 'dimension_evidence': {}})
    assert intent_details._quote_is_bound(PUBLICATION, intent_details._bound_evidence_sources(doc)[0])


@pytest.mark.parametrize('fetched_url', [
    'https://business.adobe.com/blog/a-different-page',
    'https://other.test/blog/a-different-company',
])
def test_other_page_or_company_date_cannot_cross_score_or_review_boundary(monkeypatch, fetched_url):
    scored, receipts, judge, *_ = _score(monkeypatch, fetched_url=fetched_url)
    assert scored[0] == 0
    assert judge.await_count == 0
    assert 'verified_source_context' not in receipts[-1].get('verification_trace', {})


def test_metadata_does_not_override_held_unsupported_claim_verdict(monkeypatch):
    scored, _, _, *_ = _score(monkeypatch, status='partially_supported')
    assert scored[0] == 0
