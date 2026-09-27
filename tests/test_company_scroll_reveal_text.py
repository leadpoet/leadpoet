"""Rendered pages must retain ordinary below-fold entrance-animation copy."""

import pytest

from qualification.scoring.company_evidence_investigator import _plain_text
from qualification.scoring.verification_helpers import visible_html_text


REVEAL = 'class="wp-block-paragraph wow fadeInUp animated" style="visibility: hidden; animation-name: none;"'
CLAIM = "The Echosystem is deployed across more than 10 commercial sites."


def test_exact_scroll_reveal_markup_preserves_company_evidence():
    raw = f"<main><h1>Company financing</h1><p {REVEAL}>{CLAIM}</p></main>"
    assert CLAIM in _plain_text(raw)
    # Article/intent extraction does not acquire a new visibility rule.
    assert CLAIM not in visible_html_text(raw)
    assert CLAIM in visible_html_text(raw, include_scroll_reveal=True)


@pytest.mark.parametrize("animation", ["fadeIn", "slideInLeft", "fadeInDownBig"])
def test_same_known_entrance_animation_on_other_company_pages(animation):
    raw = f'<p class="wow {animation}" style="animation-name:none;visibility:hidden">Ordinary product evidence.</p>'
    assert "Ordinary product evidence." in _plain_text(raw)


@pytest.mark.parametrize("attributes", [
    REVEAL + " hidden",
    REVEAL + ' aria-hidden="true"',
    REVEAL.replace("visibility: hidden;", "display:none;visibility:hidden;"),
    REVEAL.replace("animation-name: none;", "animation-name: none; opacity:0;"),
    REVEAL.replace("fadeInUp", "fadeOut"),
    REVEAL.replace("wow", "arbitrary"),
    'style="visibility:hidden"',
    'class="wow fadeInUp" style="visibility:hidden!important;animation-name:none"',
])
def test_reveal_does_not_admit_other_hidden_content(attributes):
    assert CLAIM not in _plain_text(f"<p {attributes}>{CLAIM}</p>")


@pytest.mark.parametrize("wrapper", [
    '<div hidden>{}</div>',
    '<div aria-hidden="true">{}</div>',
    '<div style="display:none">{}</div>',
    '<div style="visibility:hidden">{}</div>',
    '<aside>{}</aside>', '<nav>{}</nav>', '<template>{}</template>',
    '<div class="related-posts">{}</div>',
])
def test_hidden_ancestor_still_wins(wrapper):
    assert CLAIM not in _plain_text(wrapper.format(f"<p {REVEAL}>{CLAIM}</p>"))


def test_explicit_hidden_css_and_nonparagraph_elements_still_fail():
    for raw in (
        f'<style>.secret {{visibility:hidden}}</style><p {REVEAL.replace("wp-block-paragraph", "secret")}>{CLAIM}</p>',
        f'<style>#secret {{display:none}}</style><p id="secret" {REVEAL}>{CLAIM}</p>',
        f'<div {REVEAL}>{CLAIM}</div>',
        f'<script {REVEAL}>{CLAIM}</script>',
    ):
        assert CLAIM not in _plain_text(raw)


def test_revealed_parent_cannot_unhide_child_or_change_claim():
    raw = f'<p {REVEAL}>{CLAIM}<span hidden>Guaranteed unrelated claim.</span></p>'
    assert _plain_text(raw) == CLAIM
