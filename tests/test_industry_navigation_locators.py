"""Known service locators must survive corporate-news-heavy site navigation."""
from qualification.scoring import company_verification as verification
from qualification.scoring import company_evidence_investigator as investigator


def menu():
    news = ''.join(f'<a href="/about/newsroom/releases/{i}">Read more</a>' for i in range(50))
    return (
        news + '<a href="/pricing">Pricing</a><a href="/about">About</a>'
        '<a href="/contact">Contact</a><a href="/about/newsroom">Newsroom</a>'
        '<a href="/terms-of-service">Terms of service</a>'
        '<a href="/services/cloud">Cloud</a><a href="/services/modernization">Modernization</a>'
        '<a href="/services/hidden" hidden>Hidden service</a>'
        '<a href="https://another.example/services">Services</a>'
    )


def test_industry_policy_preserves_service_and_company_locators(monkeypatch):
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', 'openai/gpt-6.1-sol')
    locators = verification._homepage_navigation_locators(
        menu(), final_url='https://example.com/', verified_domain='example.com',
    )
    urls = {row['url'] for row in locators}
    assert 'https://example.com/services/cloud' in urls
    assert 'https://example.com/services/modernization' in urls
    assert {'https://example.com/' + suffix for suffix in (
        'pricing', 'about', 'contact', 'about/newsroom', 'terms-of-service',
    )} <= urls
    assert len(locators) == verification.MAX_HOMEPAGE_NAVIGATION_LOCATORS
    assert 'https://example.com/services/hidden' not in urls
    assert not any('another.example' in url for url in urls)


def test_old_policy_keeps_original_navigation_selection(monkeypatch):
    monkeypatch.delenv('ARENA_INDUSTRY_EVIDENCE_MODEL', raising=False)
    locators = verification._homepage_navigation_locators(
        menu(), final_url='https://example.com/', verified_domain='example.com',
    )
    assert locators[0]['url'] == 'https://example.com/pricing'
    assert 'https://example.com/services/cloud' not in {row['url'] for row in locators}


def test_industry_only_navigation_requires_independent_identity():
    locators = [{'url': 'https://example.com/services/cloud', 'label': 'Cloud'}]
    identity = {'normalized_name': 'Example', 'registrable_dns_domain': 'example.com',
                'linkedin_company_slug': 'example'}
    assert investigator._validated_homepage_navigation_locators(
        locators, positive_semantic_review=False, industry_review=True,
        verified_homepage_identity=identity,
    ) == locators
    assert investigator._validated_homepage_navigation_locators(
        locators, positive_semantic_review=False, industry_review=True,
        verified_homepage_identity={**identity, 'linkedin_company_slug': ''},
    ) == []
    assert investigator._validated_homepage_navigation_locators(
        locators, positive_semantic_review=False,
        verified_homepage_identity=identity,
    ) == []
