"""A bound explicit HQ field can establish country through a full subdivision."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from qualification.scoring import lead_scorer
from qualification.scoring import company_evidence_investigator as investigator
from tests.test_company_evidence_investigator import _company, _complete_verdict, _icp
from tests.test_fetched_profile_headquarters import (
    ANCHOR, BODY, HQ, IDENTITY, URL, _profile_finding,
)


def profile(location):
    field = "Headquarters\n" + location
    return BODY.replace(HQ, field), field


@pytest.mark.parametrize("location,country", [
    ("Brisbane City, Queensland", "Australia"),
    ("Montreal, Quebec", "Canada"),
    ("Montréal, Québec", "Canada"),
    ("Toronto, Ontario", "Canada"),
    ("London, England", "United Kingdom"),
])
def test_bound_hq_subdivision_reaches_final_validation_and_native_projection(location, country):
    body, quote = profile(location)
    finding = _profile_finding(body=body, quote=quote, country=country, state="")
    assert finding["status"] == "VERIFIED"
    assert finding["evidence_quote"] == quote

    observation = _complete_verdict(observed_hq_country="", observed_hq_state="")
    observation.update(geography_matches=None, geography_evidence_url="", geography_evidence_quote="")
    company = _company().model_copy(update={"country": country, "state": ""})
    cache = {}
    projected = lead_scorer._project_verified_linkedin_headquarters(
        observation, verified_identity=IDENTITY, invocation_cache=cache,
        linkedin_profile_source={"url": URL, "text": body},
        icp=_icp(geography=country), company=company, company_quality=True,
    )
    assert projected["observed_hq_country"] == country
    assert projected["geography_evidence_quote"] == quote
    assert cache["linkedin_headquarters_verified"] is True


@pytest.mark.parametrize("location,country", [
    ("London, London", "United Kingdom"),
    ("Seattle", "United States"),
    ("Melbourne, Victoria", "Australia"),
    ("Darwin, Northern Territory", "Australia"),
    ("Brisbane, QLD", "Australia"),
    ("Montreal, QC", "Canada"),
    ("Quebec", "Canada"),
    ("Brisbane, Queensland, Canada", "Australia"),
    ("Brisbane, Queensland, Canada", "Canada"),
    ("Montreal, Quebec, Australia", "Canada"),
    ("Brisbane, Queensland / Canada", "Australia"),
    ("Brisbane, Queensland", "Canada"),
])
def test_ambiguous_abbreviated_or_conflicting_hq_does_not_supply_country(location, country):
    body, quote = profile(location)
    finding = _profile_finding(body=body, quote=quote, country=country, state="")
    assert finding["status"] == "UNPROVEN"


@pytest.mark.parametrize("location", [
    "London, London", "Seattle", "Melbourne, Victoria", "Darwin, Northern Territory",
    "Brisbane, QLD", "Montreal, QC", "Quebec", "Brisbane, Queensland, Canada",
    "Montreal, Quebec, Australia", "Montreal, Quebec, Unknown",
])
def test_native_projection_does_not_select_a_country_from_ambiguous_or_conflicting_hq(location):
    body, _quote = profile(location)
    observation = _complete_verdict(observed_hq_country="", observed_hq_state="")
    cache = {}
    projected = lead_scorer._project_verified_linkedin_headquarters(
        observation, verified_identity=IDENTITY, invocation_cache=cache,
        linkedin_profile_source={"url": URL, "text": body},
        icp=_icp(geography="United Kingdom"), company=_company(), company_quality=True,
    )
    assert projected == observation
    assert not cache.get("linkedin_headquarters_verified")


@pytest.mark.parametrize("fault", ["identity", "redirect", "office", "locations", "borrowed_country"])
def test_subdivision_preserves_profile_identity_and_exact_hq_scope(fault):
    body, quote = profile("Brisbane City, Queensland")
    anchor, final = deepcopy(ANCHOR), URL
    if fault == "identity":
        anchor["verified_linkedin_slug"] = "other-company"
    elif fault == "redirect":
        final = "https://www.linkedin.com/company/other-company"
    elif fault == "office":
        body = body.replace(quote, "Office\nBrisbane City, Queensland")
        quote = "Office\nBrisbane City, Queensland"
    elif fault == "locations":
        body = body.replace(quote, "") + "\n## Locations\nHeadquarters\nBrisbane City, Queensland"
    else:
        body, quote = profile("London, London")
        body += "\n## Locations\nBrisbane City, Queensland, Australia"
    result = _profile_finding(body=body, quote=quote, country="Australia", state="", anchor=anchor, final_url=final)
    assert result["status"] == "UNPROVEN"


def test_subdivision_hq_keeps_opposite_country_mismatch_and_custom_locality_review():
    body, quote = profile("Montreal, Quebec")
    observation = _complete_verdict(observed_hq_country="", observed_hq_state="")
    company = _company().model_copy(update={"country": "Canada", "state": ""})
    for geography, expected in [("United States", False), ("Canada, Toronto", None)]:
        projected = lead_scorer._project_verified_linkedin_headquarters(
            observation, verified_identity=IDENTITY, invocation_cache={},
            linkedin_profile_source={"url": URL, "text": body},
            icp=_icp(geography=geography), company=company, company_quality=True,
        )
        assert projected["observed_hq_country"] == "Canada"
        assert projected["geography_evidence_quote"] == quote
        assert projected["geography_matches"] is expected


@pytest.mark.parametrize("name,country", [("stanwell", "Australia"), ("nectari", "Canada")])
def test_captured_quotes_and_bound_profile_reach_the_same_country(name, country):
    capture = json.loads((Path(__file__).parent / f"fixtures/company_evidence/{name}_headquarters_subdivision.json").read_text())
    source, identity = capture["source"], capture["identity"]
    url, text = source["url"], source["text"]
    anchor = {
        "verified_name": identity["normalized_name"],
        "verified_domain": identity["registrable_dns_domain"],
        "verified_linkedin_slug": identity["linkedin_company_slug"],
    }
    assert investigator._bound_linkedin_headquarters_source(url, url, text, anchor)
    for finding in capture["findings"]:
        result = investigator._validated_findings(
            {"findings": [finding]}, targets=("geography",),
            fetched_pages={url: text}, fetched_final_urls={url: url},
            first_party_domains={identity["registrable_dns_domain"]}, identity_names={identity["normalized_name"]},
            identity_anchor=anchor,
        )["geography"]
        assert result["status"] == "VERIFIED"
        assert result["evidence_quote"] == finding["evidence_quote"]
        assert result["observed_country"] == country
    company = _company().model_copy(update={
        "company_name": identity["normalized_name"],
        "company_website": "https://www." + identity["registrable_dns_domain"] + "/",
        "company_linkedin": url, "country": country, "state": "",
    })
    projected = lead_scorer._project_verified_linkedin_headquarters(
        _complete_verdict(observed_hq_country="", observed_hq_state=""),
        verified_identity=identity, invocation_cache={}, linkedin_profile_source=source,
        icp=_icp(geography=country), company=company, company_quality=True,
    )
    assert projected["observed_hq_country"] == country
    assert projected["geography_matches"] is True
