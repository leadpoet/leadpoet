"""A bounded HTML fetch must not turn unfinished markup into page evidence."""

import pytest

from qualification.scoring.company_evidence_investigator import _plain_text
from qualification.scoring.verification_helpers import visible_html_text


@pytest.mark.parametrize('tail', [
    '<image xlink:href="data:image/png;base64,' + 'AAAA' * 10000,
    '<img src="data:image/jpeg;base64,' + 'BBBB' * 10000,
    '<div title="not visible company proof',
    '<!-- not visible company proof',
    '<script src="not visible company proof',
], ids=['svg_image', 'html_image', 'attribute', 'comment', 'script_attribute'])
def test_truncated_markup_does_not_become_evidence(tail):
    raw = '<main><p>Our company supplies payment software.</p>' + tail
    assert visible_html_text(raw) == 'Our company supplies payment software.'
    assert _plain_text(raw) == 'Our company supplies payment software.'


@pytest.mark.parametrize('text', ['Revenue < cost', 'The price is < 5', 'A &amp; B', 'One &lt; two'])
def test_literal_and_escaped_text_is_preserved(text):
    expected = text.replace('&amp;', '&').replace('&lt;', '<')
    assert visible_html_text('<p>' + text) == expected
