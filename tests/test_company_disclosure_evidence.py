"""Collapsed public disclosures are evidence; arbitrary hidden text is not."""

import pytest

from qualification.scoring.company_evidence_investigator import _plain_text, _quote_occurs
from qualification.scoring.verification_helpers import visible_html_links, visible_html_text


DISCLOSURE = ('<article><div role="button" aria-expanded="false">Engineering</div>'
              '<div inert aria-hidden="true"><p>We design server and accelerator boards.</p>'
              '<a href="https://example.test/product">Product</a></div></article>')


def test_opt_in_retains_adjacent_public_disclosure_and_quote():
    assert 'accelerator' not in visible_html_text(DISCLOSURE)
    text = _plain_text(DISCLOSURE, include_disclosures=True)
    assert _quote_occurs('We design server and accelerator boards.', text)
    assert visible_html_links(DISCLOSURE) == ()  # identity-link scope is unchanged


@pytest.mark.parametrize('control_first', [True, False])
def test_explicit_control_can_precede_or_follow_unique_panel(control_first):
    control = '<button aria-expanded="false" aria-controls="details">Expand</button>'
    panel = '<section id="details" aria-hidden="true">Public product facts</section>'
    raw = control + panel if control_first else panel + control
    assert 'Public product facts' in visible_html_text(raw, include_disclosures=True)


@pytest.mark.parametrize('wrapper', [
    '<div hidden>{}</div>', '<div aria-hidden="true">{}</div>',
    '<script>{}</script>', '<template>{}</template>', '<nav>{}</nav>',
])
def test_hidden_ancestor_or_executable_markup_cannot_be_reopened(wrapper):
    assert 'accelerator' not in visible_html_text(wrapper.format(DISCLOSURE), include_disclosures=True)


@pytest.mark.parametrize('change', [
    lambda s: s.replace('aria-expanded="false"', ''),
    lambda s: s.replace('role="button"', 'role="presentation"'),
    lambda s: s.replace('role="button"', 'role="button" aria-disabled="true"'),
    lambda s: s.replace('<div inert', '<p>Intervening sibling</p><div inert'),
    lambda s: s.replace('<article>', '<section>').replace('</article>', '</section>'),
    lambda s: s.replace('inert ', ''),
    lambda s: s.replace('<div inert', '<div hidden inert'),
    lambda s: s.replace('<div inert', '<div style="display:none" inert'),
])
def test_unbound_or_independently_hidden_panel_stays_hidden(change):
    assert 'accelerator' not in visible_html_text(change(DISCLOSURE), include_disclosures=True)


def test_control_does_not_reopen_second_sibling_or_duplicate_id():
    raw = DISCLOSURE.replace('</article>', '<div inert aria-hidden="true">Secret sibling</div></article>')
    assert 'Secret sibling' not in visible_html_text(raw, include_disclosures=True)
    raw = ('<button aria-expanded="false" aria-controls="dup">Expand</button>'
           '<div id="dup" aria-hidden="true">First secret</div>'
           '<div id="dup" aria-hidden="true">Second secret</div>')
    assert 'secret' not in visible_html_text(raw, include_disclosures=True)


def test_disclosure_does_not_reopen_hidden_descendants():
    raw = DISCLOSURE.replace('<p>', '<span aria-hidden="true">Secret child</span><p>')
    text = visible_html_text(raw, include_disclosures=True)
    assert 'accelerator' in text and 'Secret child' not in text
