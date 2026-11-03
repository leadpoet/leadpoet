"""Visible inline markup must not turn the same quotation into missing proof."""
import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_company_evidence_investigator import _finding


@pytest.mark.parametrize('opening,closing', [('(', ')'), ('[', ']')])
def test_inline_element_spacing_preserves_the_same_visible_quote(opening, closing):
    source = c._plain_text(
        f'<p>Acme {opening}<a href="https://exchange.example/acme">Nasdaq: ACME</a>'
        f'{closing} supplies reporting software.</p>')
    quote = f'Acme {opening}Nasdaq: ACME{closing} supplies reporting software.'
    assert c._quote_occurs(quote, source)
    # The existing retained surface may contain either rendering convention.
    assert c._quote_occurs(c._visible_quote_surface(source).strip(), quote)


@pytest.mark.parametrize('target', ['stage', 'required_attribute'])
def test_formatting_only_quote_difference_passes_normal_evidence_validation(target):
    url = 'https://acme.example/news'
    source = 'Acme ( Nasdaq: ACME ) supplies reporting software on a recurring subscription.'
    quote = 'Acme (Nasdaq: ACME) supplies reporting software on a recurring subscription.'
    finding = _finding(target, status='VERIFIED', evidence_url=url, evidence_quote=quote,
                       activity_role='supplier_operator', observed_value='public' if target == 'stage' else True)
    result = c._validated_findings(
        {'findings': [finding]}, targets=(target,), fetched_pages={url: source},
        fetched_final_urls={url: url}, first_party_domains={'acme.example'},
        identity_names={'acme'}, identity_anchor={'submitted_name': 'Acme'},
        requested_stage='Public')
    assert result[target]['status'] == 'VERIFIED'
    assert result[target]['evidence_quote'] == quote


@pytest.mark.parametrize('quote,source', [
    ('Acme supplies software.', 'Acme does not supply software.'),
    ('Acme sells ($20) software.', 'Acme sells ( $200 ) software.'),
    ('Acme sells (20) licenses.', 'Acme sells [ 20 ] licenses.'),
    ('Acme supplies cloudsoftware.', 'Acme supplies cloud software.'),
    ('Acme supplies software for banks.', 'Acme supplies software. Separate Company supplies software for banks.'),
    ('Acme supplies software.', 'Acme ( previously ) supplies software.'),
])
def test_typography_does_not_change_facts_or_join_separate_source_spans(quote, source):
    assert not c._quote_occurs(quote, source)
