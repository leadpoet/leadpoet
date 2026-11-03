"""A changed web name cannot make an unproved LinkedIn alias terminal."""

import asyncio
from copy import deepcopy
import inspect
from itertools import product
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import lead_scorer as scorer
from scripts import arena_reviewer_homepage_linkedin_conflict_seam as seam
from tests.test_fit_evidence_url_compat import (
    _company, _complete_verdict as _base_verdict, _icp, _verified_homepage_identity,
)


def _complete_verdict(**changes):
    verdict = _base_verdict(**changes)
    verdict.update(observed_hq_country="Canada", geography_evidence_quote="Acme headquarters are in Canada.",
                   observed_employee_count="100", employee_size_evidence_quote="Acme has 100 employees.")
    return verdict


async def _preserve_employee_observation(verdict, **_kwargs):
    return dict(verdict)


@pytest.mark.parametrize("submitted_linkedin", ["https://linkedin.com/company/acme", ""])
def test_changed_name_and_slug_get_one_independent_identity_repair(monkeypatch, submitted_linkedin):
    prompts = []
    observed = _complete_verdict(name="Acme Banking Limited")
    observed["observed_company_linkedin"] = "https://linkedin.com/company/acme-new"
    original = deepcopy(observed)

    async def request(**kwargs):
        prompts.append(kwargs["prompt"])
        return (deepcopy(observed) if len(prompts) == 1 else _complete_verdict()), ""

    monkeypatch.setattr(scorer, "_ground_employee_size_observation", _preserve_employee_observation)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(scorer, "_request_company_reverify_json", request)
    # No alias transport proof. A repair must observe a complete identity.
    monkeypatch.setattr(scorer, "_fetch_bounded_html", AsyncMock(side_effect=ValueError))
    result = asyncio.run(scorer._llm_reverify_company(
        _company(company_linkedin=submitted_linkedin, country="Canada"),
        _icp().model_copy(update={"geography": "Canada", "country": "Canada"}),
        require_company_fit_dimensions=True,
        verified_homepage_identity=_verified_homepage_identity(),
        company_quality=True,
    ))
    assert result.decision == "match"
    assert len(prompts) == 2
    assert "IDENTITY CONFLICT REPAIR" in prompts[1]
    assert result.details["identity_receipt"]["observed_name"] == "acme"
    assert result.details["identity_receipt"]["observed_linkedin_slug"] == "acme"
    assert observed == original


@pytest.mark.parametrize("submitted_linkedin", ["https://linkedin.com/company/acme", ""])
def test_repeated_changed_identity_conflict_stays_unavailable(monkeypatch, submitted_linkedin):
    prompts = []

    async def request(**kwargs):
        prompts.append(kwargs["prompt"])
        verdict = _complete_verdict(name="Acme Subsidiary Limited")
        verdict["observed_company_linkedin"] = "https://linkedin.com/company/acme-subsidiary"
        return verdict, ""

    monkeypatch.setattr(scorer, "_ground_employee_size_observation", _preserve_employee_observation)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(scorer, "_request_company_reverify_json", request)
    monkeypatch.setattr(scorer, "_fetch_bounded_html", AsyncMock(side_effect=ValueError))
    result = asyncio.run(scorer._llm_reverify_company(
        _company(company_linkedin=submitted_linkedin, country="Canada"),
        _icp().model_copy(update={"geography": "Canada", "country": "Canada"}),
        require_company_fit_dimensions=True,
        verified_homepage_identity=_verified_homepage_identity(),
        company_quality=True,
    ))
    assert result.decision == "unavailable"
    assert len(prompts) == 2
    assert result.details["identity_receipt"]["observed_name"] == "acmesubsidiary"
    assert result.details["identity_receipt"]["observed_linkedin_slug"] == "acme-subsidiary"


@pytest.mark.parametrize("changes,anchor", [
    ({"observed_company_website": "https://other.example/"}, _verified_homepage_identity()),
    ({}, None),
])
def test_unbound_or_different_domain_identity_stays_mismatch(changes, anchor):
    verdict = _complete_verdict(name="Different Company")
    verdict.update(changes)
    receipt = scorer._web_identity_receipt(
        _company(), verdict,
        verified_homepage_identity=(scorer._verified_homepage_identity_anchor(anchor)),
        company_quality=True,
    )
    assert receipt["decision"] == "mismatch"


def test_wrong_submitted_linkedin_cannot_use_homepage_conflict_repair():
    receipt = scorer._web_identity_receipt(
        _company(company_linkedin="https://linkedin.com/company/wrong-submitted"),
        _complete_verdict(name="Different Company"),
        verified_homepage_identity=scorer._verified_homepage_identity_anchor(_verified_homepage_identity()),
        company_quality=True,
    )
    assert receipt["decision"] == "mismatch"


@pytest.mark.parametrize("homepage_slug", ["1234", "acme-old"])
def test_existing_matched_optional_profile_identity_is_preserved(homepage_slug):
    anchor = scorer._verified_homepage_identity_anchor(
        _verified_homepage_identity(observed_linkedin_slug=homepage_slug))
    receipt = scorer._web_identity_receipt(
        _company(company_linkedin=""), _complete_verdict(),
        verified_homepage_identity=anchor, company_quality=True,
    )
    assert receipt["decision"] == "match"
    assert receipt["observed_linkedin_slug"] == "acme"


@pytest.mark.parametrize("structured_changes,verified", [
    ({}, True),
    ({"company_id": "9999"}, False),
    ({"website": "https://wrong.example/"}, False),
    ({"url": "https://www.linkedin.com/company/other"}, False),
])
def test_existing_numeric_alias_keeps_its_structured_identity_route(structured_changes, verified):
    numeric_url = "https://www.linkedin.com/company/1234"
    vanity_url = "https://www.linkedin.com/company/acme"
    company = _company(company_linkedin=numeric_url, company_website="https://acme.com")
    anchor = scorer._verified_homepage_identity_anchor(
        _verified_homepage_identity(observed_domain="acme.com", observed_linkedin_slug="1234"))
    verdict = _complete_verdict(name="Acme Inc", website="https://acme.com")
    verdict["observed_company_linkedin"] = vanity_url
    receipt = scorer._web_identity_receipt(
        company, verdict, verified_homepage_identity=anchor, company_quality=True,
    )
    assert receipt["decision"] == "unavailable"
    assert receipt["reason_code"] == "identity_linkedin_alias_unresolved"
    lookup = scorer._alias_unresolved_structured_profile_lookup(
        receipt, anchor["registrable_dns_domain"],
    )
    assert lookup["requested_profile_url"] == numeric_url
    assert lookup["observed_profile_url"] == vanity_url
    evidence = {
        "name": "Acme Inc", "provider": "harvestapi_get_company",
        "source_field": "name", "url": vanity_url,
        "website": "https://acme.com", "company_id": "1234",
        "requested_url": numeric_url, **structured_changes,
    }
    resolved = scorer._structured_profile_alias_identity_receipt(
        company, receipt, evidence, anchor["registrable_dns_domain"],
        company_quality=True,
    )
    if verified:
        assert resolved["decision"] == "match"
        assert resolved["reason_code"] == "structured_numeric_linkedin_alias_verified"
    else:
        assert resolved == {}


@pytest.mark.parametrize("company_quality", [False, True])
def test_existing_match_and_unavailable_routes_are_preserved_across_identity_matrix(company_quality):
    source = inspect.getsource(scorer._web_identity_receipt)
    assert source.count(seam.NEW_DISPATCH) == 1
    original_source = source.replace(seam.NEW_DISPATCH, seam.OLD_DISPATCH, 1)
    namespace = dict(scorer.__dict__)
    exec(compile(original_source, "original_web_identity_receipt.py", "exec"), namespace)
    original = namespace["_web_identity_receipt"]
    changed = preserved = 0
    for anchor_slug, submitted_url, name, observed_slug in product(
        ("acme", "1234"),
        ("", "  ", "not-a-profile", "https://www.linkedin.com/company/acme",
         "https://www.linkedin.com/company/1234", "https://www.linkedin.com/company/other"),
        ("Acme", "Acme Inc", "Acme Banking Limited", ""),
        ("acme", "other", "1234", "5678", ""),
    ):
        # Exercise raw helper inputs too; public model validation already
        # rejects malformed nonempty URLs before this boundary.
        company = _company(company_website="https://acme.com").model_copy(
            update={"company_linkedin": submitted_url})
        anchor = scorer._verified_homepage_identity_anchor(_verified_homepage_identity(
            observed_domain="acme.com", observed_linkedin_slug=anchor_slug))
        verdict = _complete_verdict(name=name, website="https://acme.com")
        verdict["observed_company_linkedin"] = (
            "https://www.linkedin.com/company/" + observed_slug if observed_slug else ""
        )
        kwargs = {"verified_homepage_identity": anchor, "company_quality": company_quality}
        before = original(company, verdict, **kwargs)
        after = scorer._web_identity_receipt(company, verdict, **kwargs)
        case = (anchor_slug, submitted_url, name, observed_slug, company_quality)
        if before["decision"] in {"match", "unavailable"}:
            assert after == before, case
            preserved += 1
        if after != before:
            assert before["decision"] == "mismatch", case
            assert before["reason_code"] == "identity_mismatch", case
            assert after["decision"] == "unavailable", case
            assert after["reason_code"] == "web_linkedin_conflicts_with_verified_homepage", case
            changed += 1
    assert preserved > 0 and changed > 0


@pytest.mark.parametrize("repair_changes", [
    {"observed_company_name": ""},
    {"observed_company_website": "https://wrong.example/"},
    {"observed_company_linkedin": "https://linkedin.com/company/wrong"},
    {"industry_evidence_quote": ""},
    {"industry_evidence_url": "not-a-source"},
])
def test_repair_needs_complete_identity_and_each_dimension_proof(monkeypatch, repair_changes):
    prompts = []

    async def request(**kwargs):
        prompts.append(kwargs["prompt"])
        verdict = _complete_verdict()
        if len(prompts) == 1:
            verdict.update(observed_company_name="Acme Banking Limited",
                           observed_company_linkedin="https://linkedin.com/company/acme-new")
        else:
            verdict.update(repair_changes)
        return verdict, ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(scorer, "_request_company_reverify_json", request)
    monkeypatch.setattr(scorer, "_ground_employee_size_observation", _preserve_employee_observation)
    monkeypatch.setattr(scorer, "_fetch_bounded_html", AsyncMock(side_effect=ValueError))
    result = asyncio.run(scorer._llm_reverify_company(
        _company(country="Canada"),
        _icp().model_copy(update={"geography": "Canada", "country": "Canada"}),
        require_company_fit_dimensions=True,
        verified_homepage_identity=_verified_homepage_identity(),
        company_quality=True,
    ))
    assert result.decision != "match"
    assert len(prompts) == 2


def test_repair_network_failure_does_not_create_a_third_model_call(monkeypatch):
    purposes = []

    async def request(**kwargs):
        purposes.append(kwargs["telemetry_purpose"])
        if len(purposes) > 1:
            return None, "provider unavailable"
        verdict = _complete_verdict(name="Acme Banking Limited")
        verdict["observed_company_linkedin"] = "https://linkedin.com/company/acme-new"
        return verdict, ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(scorer, "_request_company_reverify_json", request)
    monkeypatch.setattr(scorer, "_ground_employee_size_observation", _preserve_employee_observation)
    monkeypatch.setattr(scorer, "_fetch_bounded_html", AsyncMock(side_effect=ValueError))
    result = asyncio.run(scorer._llm_reverify_company(
        _company(country="Canada"),
        _icp().model_copy(update={"geography": "Canada", "country": "Canada"}),
        require_company_fit_dimensions=True,
        verified_homepage_identity=_verified_homepage_identity(),
        company_quality=True,
    ))
    assert result.decision == "unavailable"
    assert len(purposes) == 2
