"""Core product facts must survive a long visible site menu at the same cap."""
import pytest

from qualification.scoring import company_evidence_investigator as c
from qualification.scoring import verification_helpers as v


@pytest.mark.parametrize('opening,closing', [('<main>', '</main>'), ('<div role="main">', '</div>')])
def test_large_visible_menu_cannot_displace_core_company_facts(monkeypatch, opening, closing):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    # Many real pages implement their menu as visible divs rather than nav.
    menu = '<header><div>' + 'Services Industries Regions ' * 1500 + '</div></header>'
    facts = 'We manage customer systems day to day and modernize their mainframes.'
    raw = menu + opening + '<h1>Cloud services</h1><p>' + facts + '</p>' + closing
    assert facts in c._plain_text(raw)
    assert len(c._plain_text(raw)) <= c.MAX_PAGE_CHARACTERS


def test_main_window_preserves_tables_and_excludes_hidden_claims(monkeypatch):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    raw = '<div>' + 'Menu ' * 6000 + '</div><main><h1>Plans</h1><table><tr><td>Voice</td><td>Included</td></tr></table><p hidden>Runs a bank.</p><script>Owns a bank.</script></main>'
    text = c._plain_text(raw)
    assert 'Voice Included' in text
    assert 'bank' not in text


def test_small_or_unstructured_pages_keep_the_existing_surface(monkeypatch):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    for raw in ['Before <main>Product facts</main> After', 'Plain text facts', '<div>Only ordinary divs.</div>']:
        expected = c._plain_text(raw)
        monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', '')
        assert c._plain_text(raw) == expected
        monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')


def test_old_frozen_policy_keeps_its_source_window(monkeypatch):
    raw = '<div>' + 'Menu ' * 6000 + '</div><main>Product facts.</main>'
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', '')
    assert 'Product facts.' not in c._plain_text(raw)
    assert 'Product facts.' in v.visible_html_text(raw)
