"""Split visible copyright text must retain exact legal identity, not infer an alias."""
import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_verification as verification
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer

URL = 'https://www.renaissance.com/'
PROFILE = 'https://www.linkedin.com/company/renaissance-learning/'
SUPPORT = 'https://support.renaissance.com/s/article/Classroom-license-what-is-it-and-how-to-get-it-1752678293150'
QUOTE = 'A classroom license is purchased for 365 days.'
# Minimal public fragment retained in the Nov7 homepage provider response.
COPYRIGHT = '<p><span>©</span><span>2026</span><span>Renaissance Learning, Inc. All rights reserved.</span></p>'


def html(footer=COPYRIGHT, *, profile=PROFILE):
    return ('<meta property="og:site_name" content="Renaissance">'
            '<title>Renaissance | Personalized teaching to help you See Every Student</title>'
            f'<footer>{footer}<a href="{profile}">LinkedIn</a></footer>')


def homepage(monkeypatch, body=None, *, final=URL):
    monkeypatch.setattr(verification, '_fetch_bounded_html', AsyncMock(return_value=(200, final, body or html())))
    return asyncio.run(verification.verify_company_exists(
        'Renaissance Learning', URL, company_linkedin=PROFILE,
        require_https_transport=True, company_quality=False,
    ))


def anchor(receipt):
    verified = lead_scorer._verified_homepage_identity_anchor(receipt)
    return {'submitted_name': 'Renaissance Learning', 'observed_name': 'Renaissance Learning',
            'verified_name': verified['normalized_name'],
            'submitted_domain': 'renaissance.com', 'observed_domain': 'renaissance.com',
            'verified_domain': verified['registrable_dns_domain'],
            'submitted_linkedin_slug': 'renaissance-learning',
            'observed_linkedin_slug': 'renaissance-learning',
            'verified_linkedin_slug': verified['linkedin_company_slug']}


def test_native_homepage_recovers_exact_submitted_name_from_inline_copyright(monkeypatch):
    result = homepage(monkeypatch)
    assert result.decision == 'match'
    assert result.details['identity']['observed_name'] == 'renaissancelearning'


def test_existing_strict_attribute_binding_accepts_actual_unnamed_annual_quote(monkeypatch):
    identity = anchor(homepage(monkeypatch))
    assert investigator._complete_verified_first_party_identity(SUPPORT, {'renaissance.com'}, identity)
    assert investigator._required_attribute_support_rejection(
        SUPPORT, QUOTE, fetched_pages={SUPPORT: QUOTE}, fetched_final_urls={SUPPORT: SUPPORT},
        first_party_domains={'renaissance.com'}, identity_names={'renaissancelearning'},
        identity_anchor=identity,
    ) is None


@pytest.mark.parametrize('footer', [
    '<p><span>©</span><span>2026</span></p><p>Renaissance Learning, Inc. All rights reserved.</p>',
    '<p><span>©</span><span hidden>2026</span><span>Renaissance Learning, Inc. All rights reserved.</span></p>',
    '<p hidden>'+COPYRIGHT+'</p>',
    '<div aria-hidden="true">'+COPYRIGHT+'</div>',
    '<div style="display:none">'+COPYRIGHT+'</div>',
    '<div inert>'+COPYRIGHT+'</div>',
    '<svg>'+COPYRIGHT+'</svg>',
    '<math>'+COPYRIGHT+'</math>',
    '<p><span>©</span><span style="visibility:hidden">2026</span><span>Renaissance Learning, Inc. All rights reserved.</span></p>',
    '<script>'+COPYRIGHT+'</script>',
    '<template>'+COPYRIGHT+'</template>',
    '<p>Our customer <span>Renaissance Learning, Inc. All rights reserved.</span></p>',
    COPYRIGHT+'<p><span>© 2026</span><span>Different Holdings, Inc. All rights reserved.</span></p>',
    COPYRIGHT+'<p>© 2026 Different Holdings, Inc. All rights reserved.</p>',
    COPYRIGHT+'<p><span>© 2026</span><span>Renaissance Learning Holdings, Inc. All rights reserved.</span></p>',
    '<p><span>© 2026</span><a href="/customers">Customer</a><span>Renaissance Learning, Inc. All rights reserved.</span></p>',
    '<p><span>©</span><span>'+(' '*501)+'</span><span>Renaissance Learning, Inc. All rights reserved.</span></p>',
])
def test_inline_copyright_does_not_join_unrelated_or_hidden_text(monkeypatch, footer):
    names = verification._homepage_company_names(html(footer))
    assert 'Renaissance Learning, Inc.' not in names
    result = homepage(monkeypatch, html(footer))
    if result.decision == 'match':
        assert not investigator._complete_verified_first_party_identity(SUPPORT, {'renaissance.com'}, anchor(result))


@pytest.mark.parametrize('changed', [
    {'observed_domain': 'other.example'}, {'verified_domain': 'other.example'},
    {'observed_linkedin_slug': 'different-company'}, {'verified_linkedin_slug': 'different-company'},
    {'observed_name': 'Renaissance Holdings'},
])
def test_footer_does_not_relax_existing_attribute_identity_conflict_checks(monkeypatch, changed):
    identity = anchor(homepage(monkeypatch));identity.update(deepcopy(changed))
    assert not investigator._complete_verified_first_party_identity(SUPPORT, {'renaissance.com'}, identity)


def test_wrong_homepage_profile_does_not_gain_exact_company_binding(monkeypatch):
    result = homepage(monkeypatch, html(profile='https://www.linkedin.com/company/unrelated-company/'))
    assert result.decision != 'match'


def test_wrong_final_domain_does_not_gain_exact_company_binding(monkeypatch):
    assert homepage(monkeypatch, final='https://other.example/').decision != 'match'


@pytest.mark.parametrize('body', [
    '<style>.invisible {display:none}</style>'+html(COPYRIGHT.replace('<p>', '<p class="invisible">')),
    '<style>#legal {visibility:hidden}</style>'+html('<div id="legal">'+COPYRIGHT+'</div>'),
    '<style>.invisible {display:none}</style><div class="invisible">'+html()+'</div>',
    html('<p><span>©</span><span>2026 Renaissance Learning, Inc.</span><span hidden>hidden</span><span> Other Holdings, Inc. All rights reserved.</span></p>'),
])
def test_stylesheet_hidden_and_interrupted_copyright_cannot_supply_a_name(body):
    assert 'Renaissance Learning, Inc.' not in verification._homepage_company_names(body)
