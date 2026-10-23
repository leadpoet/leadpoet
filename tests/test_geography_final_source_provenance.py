"""HQ overrides require the fetched destination to retain company provenance."""
from copy import deepcopy

import pytest

from qualification.scoring import company_evidence_investigator as inv
from tests.test_geography_source_conflict import ANCHOR, FIXTURE, project


def validate(final_urls, *, url=None, quote=None, **kwargs):
    url = url or FIXTURE["source_url"]
    quote = quote or FIXTURE["quote"]
    return inv._validated_findings(
        {"findings": [{"target": "geography", "status": "VERIFIED",
                       "observed_country": "Singapore", "observed_state": "",
                       "evidence_url": url, "evidence_quote": quote}]},
        targets=("geography",), fetched_pages={url: quote},
        fetched_final_urls=final_urls, first_party_domains={"airwallex.com"},
        identity_names={"airwallex"}, identity_anchor=ANCHOR, **kwargs,
    )["geography"]


@pytest.mark.parametrize("final_urls", [
    None, {}, {FIXTURE["source_url"]: ""},
    {FIXTURE["source_url"]: "https://unrelated.example/airwallex-story"},
    {FIXTURE["source_url"]: "https://airwallex.com.unrelated.example/story"},
])
def test_missing_or_unrelated_final_source_cannot_override_linkedin(final_urls):
    finding = validate(final_urls)
    assert finding["status"] == "UNPROVEN"
    assert "fetched final" in finding["reason"]
    assert finding["evidence_url"] == finding["evidence_quote"] == ""
    assert project(finding)["observed_hq_country"] == "Hong Kong"


@pytest.mark.parametrize("final_url", [
    FIXTURE["source_url"], "https://airwallex.com/about", "https://news.airwallex.com/hq",
])
def test_same_firstparty_destination_keeps_validated_coheadquarters(final_url):
    finding = validate({FIXTURE["source_url"]: final_url})
    assert finding["status"] == "VERIFIED"
    assert project(finding)["observed_hq_country"] == "Singapore"


def test_thirdparty_request_cannot_borrow_a_firstparty_final_destination():
    url = "https://unrelated.example/company"
    assert validate({url: FIXTURE["source_url"]}, url=url)["status"] == "UNPROVEN"


def test_bound_structured_linkedin_projection_keeps_its_existing_provenance():
    source = deepcopy(FIXTURE["headquarters"])
    raw = {"target": "geography", "status": "CONTRADICTED",
           "observed_country": source["country"], "observed_state": source["state"],
           "evidence_url": source["url"], "evidence_quote": source["text"]}
    finding = inv._validated_findings(
        {"findings": [raw]}, targets=("geography",),
        fetched_pages={source["url"]: source["text"]}, fetched_final_urls={},
        first_party_domains={"airwallex.com"}, identity_names={"airwallex"},
        identity_anchor=ANCHOR, structured_headquarters_evidence=source,
    )["geography"]
    assert finding["status"] == "CONTRADICTED"


def test_exact_linkedin_profile_keeps_its_existing_final_profile_check():
    url = "https://www.linkedin.com/company/airwallex"
    text = "Airwallex | LinkedIn\nWebsite\nhttps://airwallex.com\nAbout\nHeadquarters\nSingapore"
    finding = validate({url: url}, url=url, quote=text)
    assert finding["status"] == "VERIFIED"
    assert validate({url: "https://www.linkedin.com/company/other"},
                    url=url, quote=text)["status"] == "UNPROVEN"


def test_verified_rebrand_can_bind_old_request_and_new_firstparty_destination():
    old_url, new_url = "https://quizizz.com/about", "https://wayground.com/about"
    rebrand_url = "https://help.wayground.com/rebrand"
    rebrand_quote = "Quizizz is now Wayground following our rebrand."
    rebrand = {"target": "rebrand", "status": "VERIFIED", "evidence_url": rebrand_url,
               "evidence_quote": rebrand_quote, "old_name": "Quizizz", "new_name": "Wayground",
               "old_domain": "quizizz.com", "new_domain": "wayground.com",
               "shared_linkedin_slug": "quizizz"}
    hq_quote = "Wayground headquarters: Singapore."
    hq = {"target": "geography", "status": "VERIFIED", "observed_country": "Singapore",
          "observed_state": "", "evidence_url": old_url, "evidence_quote": hq_quote}
    finding = inv._validated_findings(
        {"findings": [hq, rebrand]}, targets=("geography", "rebrand"),
        fetched_pages={old_url: hq_quote,
                       rebrand_url: rebrand_quote + " quizizz.com wayground.com"},
        fetched_final_urls={old_url: new_url, rebrand_url: rebrand_url},
        first_party_domains={"quizizz.com", "wayground.com"},
        identity_names={"quizizz", "wayground"},
        identity_anchor={"submitted_domain": "quizizz.com", "observed_domain": "wayground.com"},
    )
    assert finding["rebrand"]["status"] == finding["geography"]["status"] == "VERIFIED"
