"""The complete-criterion classifier can reject a genuine supplier's fit."""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as c, lead_scorer
from tests.test_native_industry_review import install
from tests.test_company_evidence_investigator import _finding


@pytest.mark.parametrize('missing', [['Marketplace selling'], ['Requested account workflow integration']])
def test_complete_supplier_mismatch_survives_native_projection(monkeypatch, missing):
    run, _, _, reviews = install(monkeypatch, decision='mismatch', missing=missing)
    result = asyncio.run(run(requested_product_service='Branded store and marketplace selling'))
    finding = result['claims']['industry']
    assert finding['_industry_review']['decision'] == 'mismatch'
    assert finding['status'] == 'CONTRADICTED'
    assert finding['evidence_url'] and finding['evidence_quote']
    assert finding['supporting_evidence']
    projected = lead_scorer._project_investigator_industry({}, finding)
    assert projected['industry_matches'] is False
    stored = json.loads(json.dumps(lead_scorer._dimension_web_evidence(projected, 'industry')))
    assert stored['quote'] == finding['evidence_quote']
    assert stored['supporting_evidence'] == finding['supporting_evidence']
    assert len(reviews) == 1  # Completed initial packet is reused, never resampled.


def test_native_submit_cannot_claim_authoritative_mismatch():
    url = 'https://www.saatva.com/'
    quote = 'Saatva sells luxury mattresses through its own branded online store.'
    raw = _finding('industry', status='CONTRADICTED', activity_role='supplier_operator',
                   observed_industry='Luxury mattresses', evidence_url=url, evidence_quote=quote)
    raw['_industry_review'] = {'schema': 'industry_evidence_review_v1', 'decision': 'mismatch'}
    result = c._validated_findings({'findings': [raw]}, targets=('industry',),
        fetched_pages={url: quote}, fetched_final_urls={url: url},
        first_party_domains={'saatva.com'}, identity_names={'saatva'},
        identity_anchor={'submitted_name': 'Saatva', 'submitted_domain': 'saatva.com',
                         'observed_domain': 'saatva.com'})
    assert result['industry']['status'] == 'UNPROVEN'


@pytest.mark.parametrize('fault', [None, 'quote', 'name', 'source', 'role', 'observation'])
def test_classifier_mismatch_keeps_native_attribution_gates(fault):
    url = 'https://www.saatva.com/'
    quote = 'Saatva sells luxury mattresses through its own branded online store.'
    raw = _finding('industry', status='CONTRADICTED', activity_role='supplier_operator',
                   observed_industry='Luxury mattresses', evidence_url=url, evidence_quote=quote)
    pages, finals = {url: quote}, {url: url}
    if fault == 'quote':
        raw['evidence_quote'] = 'Saatva sells products on third-party marketplaces.'
    elif fault == 'name':
        pages[url] = raw['evidence_quote'] = quote.replace('Saatva', 'OtherCo')
    elif fault == 'source':
        pages.clear()
    elif fault == 'role':
        raw['activity_role'] = 'unresolved'
    elif fault == 'observation':
        raw['observed_industry'] = ''
    result = c._validated_findings({'findings': [raw]}, targets=('industry',),
        fetched_pages=pages, fetched_final_urls=finals,
        first_party_domains={'saatva.com'}, identity_names={'saatva'},
        identity_anchor={'submitted_name': 'Saatva', 'submitted_domain': 'saatva.com',
                         'observed_domain': 'saatva.com'}, classified_industry_mismatch=True)
    finding = result['industry']
    if fault is None:
        assert finding['status'] == 'CONTRADICTED'
        assert finding['evidence_quote'] == quote
    else:
        assert finding['status'] == 'UNPROVEN'
        assert finding['evidence_url'] == finding['evidence_quote'] == ''


def test_classifier_rejects_unbound_redirect_before_review():
    import time
    url = 'https://www.saatva.com/'
    finding, failure, calls = asyncio.run(c._review_industry_sources(
        None, model='offline', openrouter_key='', started=time.monotonic(),
        company_locator={'name': 'Saatva', 'website': url},
        criteria={'industry': 'Retail', 'subindustry': '', 'product_service': 'Marketplace selling'},
        evaluation_date_value='2026-11-09',
        fetched_pages={url: 'Saatva sells its own mattresses.'},
        fetched_final_urls={url: 'https://unrelated.example/'},
        first_party_domains={'saatva.com'}, identity_names={'saatva'},
        identity_anchor={'submitted_name': 'Saatva', 'submitted_domain': 'saatva.com',
                         'observed_domain': 'saatva.com'}))
    assert finding['status'] == 'UNPROVEN'
    assert finding['_industry_review']['decision'] == 'unavailable'
    assert not failure and calls == 0
