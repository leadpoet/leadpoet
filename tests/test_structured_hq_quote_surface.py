"""Keep bound HQ projections quotable without admitting identity link targets."""
from copy import deepcopy

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import lead_scorer as scorer
from qualification.scoring.linkedin_company_size import (
    STRUCTURED_PROFILE_HEADQUARTERS_SOURCE_FIELD,
    STRUCTURED_PROFILE_PROVIDER,
)

URL = "https://www.linkedin.com/company/example-consulting"
QUOTE = "Example Consulting headquarters: London, England, United Kingdom."
MARKER = inv._IDENTITY_LINK_CONTEXT_MARKER
SOURCE = "Example Consulting | LinkedIn About Management consulting. "
LINKS = " https://example.com/identity-only-does-not-prove-headquarters"
IDENTITY = {"normalized_name": "Example Consulting", "registrable_dns_domain": "example.com",
            "linkedin_company_slug": "example-consulting"}
HQ = {"name": "Example Consulting", "text": QUOTE, "provider": STRUCTURED_PROFILE_PROVIDER,
      "source_field": STRUCTURED_PROFILE_HEADQUARTERS_SOURCE_FIELD, "url": URL,
      "website": "https://example.com/", "country": "United Kingdom", "state": "England"}


def compose(text, *, evidence=None, identity=None, final_url=URL):
    return scorer._investigator_prefetched_pages(
        {}, [URL], verified_homepage_pages={URL: {"final_url": final_url, "text": text}},
        structured_profile_headquarters_evidence=HQ if evidence is None else evidence,
        verified_identity=IDENTITY if identity is None else identity,
        include_structured_headquarters=True,
    )


@pytest.mark.parametrize("marker", [False, True])
def test_projection_remains_a_literal_quote_and_identity_links_remain_excluded(marker):
    original = SOURCE + (MARKER + LINKS if marker else "")
    pages = compose(original)
    text = pages[URL]["text"]
    assert SOURCE in text
    assert QUOTE in text
    assert inv._quote_occurs(QUOTE, text)
    assert inv._quote_occurs("Example Consulting | LinkedIn", text)
    if marker:
        assert text.endswith(MARKER + LINKS)
        assert not inv._quote_occurs(LINKS.strip(), text)
    finding = inv._validated_findings(
        {"findings": [{"target": "geography", "status": "VERIFIED",
                       "observed_country": "United Kingdom", "observed_state": "England",
                       "evidence_url": URL, "evidence_quote": QUOTE}]},
        targets=("geography",), fetched_pages={URL: text},
        fetched_final_urls={URL: URL}, first_party_domains={"example.com"},
        identity_names={"exampleconsulting"},
        identity_anchor={"verified_name": "Example Consulting", "verified_domain": "example.com",
                         "verified_linkedin_slug": "example-consulting"},
        structured_headquarters_evidence=HQ,
    )["geography"]
    assert finding["status"] == "VERIFIED"
    assert compose(text) == pages


def test_identity_suffix_cannot_suppress_the_real_server_projection():
    pages = compose(SOURCE + MARKER + " " + QUOTE)
    assert inv._quote_occurs(QUOTE, pages[URL]["text"])
    assert pages[URL]["text"].endswith(MARKER + " " + QUOTE)


@pytest.mark.parametrize("marker", [False, True])
@pytest.mark.parametrize("extra", [0, 1])
def test_combined_page_bound_does_not_truncate_source_or_projection(marker, extra):
    suffix = MARKER + LINKS if marker else ""
    added = len("\n\n" + QUOTE) + (2 if marker else 0)
    original = "x" * (inv.MAX_PAGE_CHARACTERS - len(suffix) - added + extra) + suffix
    text = compose(original)[URL]["text"]
    if extra:
        assert text == original
        assert not inv._quote_occurs(QUOTE, text)
    else:
        assert len(text) == inv.MAX_PAGE_CHARACTERS
        assert inv._quote_occurs(QUOTE, text)
        assert text.startswith(original.partition(MARKER)[0])


@pytest.mark.parametrize("fault", ["name", "domain", "slug", "final_url"])
def test_unbound_identity_or_final_url_cannot_gain_a_projection(fault):
    identity = deepcopy(IDENTITY)
    final = URL
    if fault == "name": identity["normalized_name"] = "Another Company"
    elif fault == "domain": identity["registrable_dns_domain"] = "another.example"
    elif fault == "slug": identity["linkedin_company_slug"] = "another-company"
    else: final = "https://www.linkedin.com/company/another-company"
    original = SOURCE + MARKER + LINKS
    pages = compose(original, identity=identity, final_url=final)
    assert pages[URL]["text"] == original
    assert not inv._quote_occurs(QUOTE, pages[URL]["text"])


def validate_hq(*, observed_state="England", observed_country="United Kingdom",
                evidence=None, quote=QUOTE, status="VERIFIED"):
    source = HQ if evidence is None else evidence
    return inv._validated_findings(
        {"findings": [{"target": "geography", "status": status,
                       "observed_country": observed_country, "observed_state": observed_state,
                       "evidence_url": URL, "evidence_quote": quote}]},
        targets=("geography",), fetched_pages={URL: source["text"]},
        fetched_final_urls={URL: URL}, first_party_domains={"example.com"},
        identity_names={"exampleconsulting"},
        identity_anchor={"verified_name": "Example Consulting", "verified_domain": "example.com",
                         "verified_linkedin_slug": "example-consulting"},
        structured_headquarters_evidence=source,
    )["geography"]


@pytest.mark.parametrize("structured_state", ["London", "England", ""])
@pytest.mark.parametrize("observed_state", ["London", "England", ""])
def test_non_us_hq_uses_country_and_grounded_location_not_state_field_equality(
    structured_state, observed_state,
):
    evidence = {**HQ, "state": structured_state}
    finding = validate_hq(observed_state=observed_state, evidence=evidence)
    assert finding["status"] == "VERIFIED"
    assert finding["evidence_quote"] == QUOTE
    assert finding["observed_state"] == observed_state


def test_non_us_locality_absent_from_exact_hq_quote_still_fails():
    finding = validate_hq(observed_state="Manchester")
    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == "source quote did not prove the submitted headquarters"


def test_non_us_country_must_match_the_private_source():
    assert validate_hq(observed_country="France")["status"] == "UNPROVEN"


@pytest.mark.parametrize("field,value", [
    ("provider", "untrusted_provider"),
    ("source_field", "locations"),
    ("name", "Another Company"),
    ("website", "https://another.example/"),
    ("url", "https://www.linkedin.com/company/another-company"),
    ("unexpected_authority", True),
])
def test_non_us_state_tolerance_does_not_admit_forged_private_sources(field, value):
    source = {**HQ, field: value}
    assert validate_hq(observed_state="London", evidence=source)["status"] == "UNPROVEN"


def test_us_state_conflict_still_fails_with_literal_headquarters_proof():
    quote = "Example Consulting headquarters: Ohio, United States."
    source = {**HQ, "country": "United States", "state": "Ohio", "text": quote}
    assert validate_hq(observed_country="United States", observed_state="Ohio",
                       evidence=source, quote=quote)["status"] == "VERIFIED"
    assert validate_hq(observed_country="United States", observed_state="Indiana",
                       evidence=source, quote=quote)["status"] == "UNPROVEN"


def test_non_us_country_match_does_not_override_contradicted_requested_city():
    from types import SimpleNamespace

    finding = validate_hq(observed_state="London", status="CONTRADICTED")
    assert finding["status"] == "CONTRADICTED"
    request = SimpleNamespace(country="United Kingdom", geography="Paris")
    company = SimpleNamespace(country="United Kingdom", state="London")
    projected = scorer._project_investigator_geography(
        {}, finding, icp=request, company=company, company_quality=True,
    )
    assert scorer._geography_needs_semantic_resolution(request)
    assert scorer._decision_from_observed_geography(
        projected, request, company=company, company_quality=True,
        validated_geography_finding=finding,
    ) == "mismatch"


def test_non_us_hq_does_not_invent_london_from_londonderry():
    quote = "Example Consulting headquarters: Londonderry, Northern Ireland, United Kingdom."
    source = {**HQ, "state": "Northern Ireland", "text": quote}
    finding = validate_hq(observed_state="London", evidence=source, quote=quote)
    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == "source quote did not prove the submitted headquarters"


@pytest.mark.parametrize("place,country,quote,expected", [
    ("London", "United Kingdom", QUOTE, True),
    ("England", "United Kingdom", QUOTE, True),
    ("", "United Kingdom", QUOTE, True),
    ("Manchester", "United Kingdom", QUOTE, False),
    ("London", "United Kingdom", "Example Consulting headquarters: Londonderry, United Kingdom.", False),
    ("São Paulo", "Brazil", "Example Consulting headquarters: São Paulo, Brazil.", True),
    ("São Paulo", "Brazil", "Example Consulting headquarters: São Pauloland, Brazil.", False),
    ("Łódź", "Poland", "Example Consulting headquarters: Łódź, Poland.", True),
    ("Łódź", "Poland", "Example Consulting headquarters: Łódźshire, Poland.", False),
    ("St. Helier", "Jersey", "Example Consulting headquarters: St. Helier, Jersey.", True),
    ("London", "U.K.", "Example Consulting headquarters: London, U.K.", True),
    ("London", "UK", "Example Consulting headquarters: London, UK.", True),
    ("London", "UK", "Example Consulting headquarters: London, UKraine.", False),
    ("London", "", QUOTE, False),
])
def test_non_us_headquarters_location_matches_complete_literal_tokens(place, country, quote, expected):
    assert inv._quote_supports_headquarters(
        quote, observed_country=country, observed_state=place,
    ) is expected
