"""ITW's retained root-to-careers redirect; provider replies are controlled fixtures."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_verification as verification
from qualification.scoring import lead_scorer as scorer
from qualification.scoring import linkedin_company_size as profiles


REQUEST = "https://www.itw.com/careers/working-at-itw/"
FINAL = "https://careers.itw.com/us/en"
PROFILE = "https://www.linkedin.com/company/illinoistoolworks"
DOMAIN = "itw.com"
COMPANY = SimpleNamespace(company_name="ITW", company_website=REQUEST, company_linkedin=PROFILE)
# Minimal exact identity elements from native scrape entry 10260313. Its
# Firecrawl metadata reports FINAL; the worker received the complete HTML.
HTML = ('<title>Careers at ITW | ITW jobs</title><meta property="og:site_name" content="ITW">'
        '<a href="https://www.linkedin.com/company/illinoistoolworks/">LinkedIn</a>')
PAYLOAD = {"status": "completed", "result": {"data": {"status": 200, "element": {
    "name": "ITW", "website": "https://www.itw.com/", "linkedinUrl": PROFILE,
    "employeeCountRange": {"start": 10001, "end": None},
}}}}


def homepage(monkeypatch):
    async def fetch(_session, url):
        assert url == REQUEST
        return 200, FINAL, HTML
    monkeypatch.setattr(verification, "_fetch_bounded_html", fetch)
    return asyncio.run(verification.verify_company_exists(
        COMPANY.company_name, REQUEST, company_linkedin=PROFILE,
        require_https_transport=True, company_quality=True,
    ))


def proof(payload=PAYLOAD):
    return profiles.project_structured_linkedin_company_identity(
        DOMAIN, PROFILE, deepcopy(payload), preserve_identity_website=True,
        expected_company_name="itw",
    )


def lookup(receipt, transport=DOMAIN):
    return scorer._alias_unresolved_structured_profile_lookup(
        receipt, transport, server_verified_homepage_receipt=True,
    )


def bind(receipt, evidence, transport=DOMAIN):
    return scorer._structured_profile_alias_identity_receipt(
        COMPANY, receipt, evidence, transport, company_quality=True,
        server_verified_homepage_receipt=True,
    )


def test_retained_child_homepage_locates_profile_without_approving_identity(monkeypatch):
    result = homepage(monkeypatch)
    receipt = result.details["identity"]
    before = deepcopy(receipt)
    assert result.decision == "unavailable"
    assert receipt["observed_domain"] == "careers.itw.com"
    assert result.details["actual_final_url"] == FINAL
    assert lookup(receipt) == {
        "normalized_name": "itw", "registrable_dns_domain": DOMAIN,
        "linkedin_company_slug": "illinoistoolworks",
    }
    assert not bind(receipt, None)
    resolved = bind(receipt, proof())
    assert resolved["decision"] == "match"
    assert resolved["observed_domain"] == DOMAIN
    assert resolved["raw_observed_domain"] == "careers.itw.com"
    assert resolved["structured_profile_identity"]["provider_website_url"] == "https://www.itw.com/"
    assert bind(json.loads(json.dumps(receipt)), json.loads(json.dumps(proof()))) == resolved
    assert receipt == before


@pytest.mark.parametrize("update,transport", [
    ({"observed_name": "differentitw"}, DOMAIN),
    ({"observed_linkedin_slug": "other-company"}, DOMAIN),
    ({"observed_linkedin_slug": ""}, DOMAIN),
    ({"submitted_linkedin_slug": ""}, DOMAIN),
    ({"observed_domain": "careers.itw.com.attacker.example"}, DOMAIN),
    ({"submitted_domain": "investors.itw.com"}, DOMAIN),
    ({"submitted_domain": "github.io", "observed_domain": "itw.github.io"}, "github.io"),
    ({"submitted_domain": "itw.github.io", "observed_domain": "careers.itw.github.io"}, "itw.github.io"),
    ({"submitted_domain": "co.uk", "observed_domain": "itw.co.uk"}, "co.uk"),
    ({"evidence_source": "company_web_reverification"}, DOMAIN),
    ({"decision": "mismatch"}, DOMAIN),
    ({"reason_code": "identity_name_alias_unresolved"}, DOMAIN),
    ({}, ""), ({}, "other.example"),
])
def test_child_lookup_never_uses_conflicting_or_shared_host_identity(monkeypatch, update, transport):
    receipt = {**homepage(monkeypatch).details["identity"], **update}
    assert not lookup(receipt, transport)
    assert not bind(receipt, proof(), transport)


@pytest.mark.parametrize("fault", [
    "name", "slug", "website", "raw_child", "raw_sibling", "raw_foreign",
    "raw_credentials", "raw_port", "no_raw_website", "provider", "source_field", "extra",
])
def test_returned_profile_must_independently_prove_exact_root_company(monkeypatch, fault):
    receipt = homepage(monkeypatch).details["identity"]
    evidence = proof()
    if fault == "no_raw_website":
        evidence.pop("provider_website_url")
    else:
        field, value = {
            "name": ("name", "ITW Customer"),
            "slug": ("url", "https://www.linkedin.com/company/other"),
            "website": ("website", "https://other.example/"),
            "raw_child": ("provider_website_url", FINAL),
            "raw_sibling": ("provider_website_url", "https://investors.itw.com/"),
            "raw_foreign": ("provider_website_url", "https://other.example/"),
            "raw_credentials": ("provider_website_url", "https://user@itw.com/"),
            "raw_port": ("provider_website_url", "https://itw.com:444/"),
            "provider": ("provider", "model"),
            "source_field": ("source_field", "description"),
            "extra": ("extra", "untrusted"),
        }[fault]
        evidence[field] = value
    assert not bind(receipt, evidence)


@pytest.mark.parametrize("fault", [None, "wrong_name", "wrong_profile", "child_website", "failed"])
def test_actual_fit_caller_uses_one_profile_lookup_and_keeps_unresolved_alias(monkeypatch, fault):
    from tests.test_company_headcount_evidence import company, icp, verdict, install_no_network
    prior = homepage(monkeypatch)
    before = deepcopy(prior.details)
    candidate = company().model_copy(update=vars(COMPANY))
    buyer = icp().model_copy(update={"employee_count": "10,001+"})
    observed = {**verdict(), "observed_company_name": "Illinois Tool Works Inc.",
        "observed_company_website": "https://www.itw.com", "observed_company_linkedin": "",
        "observed_employee_count": None, "employee_size_matches": None,
        "employee_size_evidence_url": "", "employee_size_evidence_quote": ""}
    install_no_network(monkeypatch, observed)
    calls = []
    async def structured(domain, url, **kwargs):
        calls.append((domain, url))
        assert kwargs["preserve_identity_website"] is True
        assert kwargs["expected_company_name"] == "itw"
        if fault == "failed":
            return None
        payload = deepcopy(PAYLOAD)
        payload["result"]["data"]["element"].update({
            "wrong_name": {"name": "ITW Customer"},
            "wrong_profile": {"linkedinUrl": PROFILE + "-customer"},
            "child_website": {"website": FINAL},
        }.get(fault, {}))
        evidence = profiles.project_structured_linkedin_company_identity(
            domain, url, payload, preserve_identity_website=True,
            expected_company_name=kwargs["expected_company_name"],
        )
        if evidence:
            kwargs["company_identity_evidence"].update(evidence)
        return profiles.project_structured_linkedin_company_size(domain, url, payload)
    async def no_page_proof(*args, **kwargs):
        return None
    monkeypatch.setattr(scorer, "fetch_structured_linkedin_company_size", structured)
    monkeypatch.setattr(scorer, "fetch_current_linkedin_company_size", no_page_proof)
    result = asyncio.run(scorer._llm_reverify_company(candidate, buyer,
        require_company_fit_dimensions=True, company_quality=True,
        verified_homepage_identity=prior))
    assert calls == [(DOMAIN, PROFILE)]
    # The typed profile may prove size. It cannot invent ITW/legal-name alias
    # continuity that the web observation omitted, or approve the whole company.
    assert result.details["dimension_decisions"]["employee_size"] == ("match" if fault is None else "unavailable")
    assert result.details["identity_decision"] != "match"
    assert prior.details == before
