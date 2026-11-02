"""Identity uses the document title, not SVG accessibility or template titles."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_verification as verification


@pytest.mark.parametrize('extra', [
    '<svg><title>Loading</title></svg>',
    '<svg><g><title>Acme Logo</title></g></svg>',
    '<math><title>Equation</title></math>',
    '<template><title>Unrendered preview</title></template>',
    '<title>Another document title</title>',
])
def test_document_title_is_not_joined_to_unrelated_titles(extra):
    page = '<html><head><title>Acme</title></head><body>' + extra + '</body></html>'
    assert verification._homepage_company_names(page) == ['Acme']


@pytest.mark.parametrize('prefix', [
    '<svg><title>Loading</title></svg>',
    '<template><title>Unrendered</title></template>',
    '<svg><svg><title>Nested</title></svg></svg>',
])
def test_foreign_or_hidden_title_cannot_claim_the_document_title(prefix):
    assert verification._homepage_company_names(prefix + '<title>Acme</title>') == ['Acme']


def _verify(monkeypatch, *, title='Acme', extra='', linkedin='acme', final_url='https://acme.example/'):
    page = ('<html><head><title>' + title + '</title></head><body>' + extra
            + '<a href="https://www.linkedin.com/company/' + linkedin + '">LinkedIn</a>'
            + '<p>We provide commercial infrastructure products and support services for enterprise customers.</p>'
            + '</body></html>')
    fetch = AsyncMock(return_value=(200, final_url, page))
    monkeypatch.setattr(verification, '_fetch_bounded_html', fetch)
    result = asyncio.run(verification.verify_company_exists(
        'Acme', 'https://acme.example/', company_linkedin='https://www.linkedin.com/company/acme',
        company_quality=True))
    return result


def test_native_homepage_identity_accepts_name_domain_linkedin_with_svg_label(monkeypatch):
    result = _verify(monkeypatch, extra='<svg><title>Loading</title></svg>')
    assert result.decision == 'match'
    assert result.details['identity']['observed_name'] == 'acme'


def test_accessibility_title_does_not_establish_company_identity(monkeypatch):
    result = _verify(monkeypatch, title='Other Business', extra='<svg><title>Acme</title></svg>')
    assert result.decision != 'match'


@pytest.mark.parametrize('kwargs', [
    {'linkedin': 'different-company'},
    {'final_url': 'https://different.example/'},
])
def test_document_title_does_not_bypass_other_identity_guards(monkeypatch, kwargs):
    result = _verify(monkeypatch, extra='<svg><title>Loading</title></svg>', **kwargs)
    assert result.decision != 'match'


def test_metadata_still_supplies_name_and_title_fragments_remain_supported():
    assert verification._homepage_company_names(
        '<meta property="og:site_name" content="Acme"><title>Acme: Products</title>'
        '<svg><title>Decorative icon</title></svg>') == ['Acme', 'Acme: Products', 'Products']


@pytest.mark.parametrize('source,quote_present,expected', [
    ('https://acme.example/pricing', True, 'VERIFIED'),
    ('https://other.example/pricing', True, 'UNPROVEN'),
    ('https://acme.example/pricing', False, 'UNPROVEN'),
])
def test_document_identity_binds_only_fetched_first_party_quote(
    monkeypatch, source, quote_present, expected,
):
    from qualification.scoring import company_evidence_investigator as investigator
    from qualification.scoring.lead_scorer import _verified_homepage_identity_anchor

    identity = _verify(monkeypatch, extra='<svg><title>Privacy opt-out icon</title></svg>')
    verified = _verified_homepage_identity_anchor(identity)
    quote = 'Business password manager. $9 per user per month, billed yearly.'
    anchor = {
        'submitted_name': 'Acme', 'observed_name': 'Acme',
        'verified_name': verified['normalized_name'],
        'submitted_domain': 'acme.example', 'observed_domain': 'acme.example',
        'verified_domain': verified['registrable_dns_domain'],
        'submitted_linkedin_slug': 'acme', 'observed_linkedin_slug': 'acme',
        'verified_linkedin_slug': verified['linkedin_company_slug'],
    }
    findings = investigator._validated_findings(
        {'findings': [{
            'target': 'required_attribute', 'status': 'VERIFIED',
            'activity_role': 'supplier_operator',
            'evidence_url': source, 'evidence_quote': quote,
        }]},
        targets=('required_attribute',),
        fetched_pages={source: quote if quote_present else 'Contact sales.'},
        fetched_final_urls={source: source}, first_party_domains={'acme.example'},
        identity_names={'acme'}, identity_anchor=anchor,
    )
    assert findings['required_attribute']['status'] == expected
