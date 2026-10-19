"""Exact transport redirects bind vanity aliases without relaxing identity."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from gateway.qualification.models import CompanyOutput
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    company_quality_receipt_matches_claim,
)


OLD = "https://www.linkedin.com/company/brim-financial-inc"
CURRENT = "https://www.linkedin.com/company/brimfinancial"
HOMEPAGE_IDENTITY = {
    "normalized_name": "brim",
    "registrable_dns_domain": "brimfinancial.com",
    "linkedin_company_slug": "brim-financial-inc",
    # Explicit official-site brand/legal-name evidence, not a name-prefix rule.
    "verified_legal_name_aliases": ["Brim Financial"],
}


def _company(**changes):
    return CompanyOutput.model_construct(**{
        "company_name": "Brim",
        "company_website": "https://brimfinancial.com/",
        "company_linkedin": OLD,
        **changes,
    })


def _verdict(**changes):
    return {
        "observed_company_name": "Brim Financial",
        "observed_company_website": "https://brimfinancial.com/",
        "observed_company_linkedin": CURRENT,
        **changes,
    }


def _resolve(company, verdict, *, responses=None, error=None):
    fetch = AsyncMock(side_effect=error if error is not None else responses)
    with patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        resolved = asyncio.run(
            lead_scorer._resolve_observed_linkedin_redirect_alias(company, verdict)
        )
    return resolved, fetch


def _receipt(company, resolved, *, anchor=HOMEPAGE_IDENTITY):
    return lead_scorer._web_identity_receipt(
        company, resolved,
        verified_homepage_identity=anchor,
        verified_homepage_transport_domain="brimfinancial.com",
        company_quality=True,
    )


def test_old_homepage_anchor_redirects_to_current_observation_with_real_identity():
    company = _company()
    verdict = _verdict()
    # The current page does not redirect. Only the old anchor proves the pair.
    resolved, fetch = _resolve(company, verdict, responses=[
        (200, CURRENT, ""), (200, CURRENT, ""),
    ])
    receipt = _receipt(company, resolved)

    assert [call.args[1] for call in fetch.await_args_list] == [CURRENT, OLD]
    assert receipt["decision"] == "match"
    assert receipt["submitted_name"] == "brim"
    assert receipt["observed_name"] == "brimfinancial"
    assert receipt["submitted_domain"] == receipt["observed_domain"] == "brimfinancial.com"
    assert receipt["submitted_linkedin_slug"] == "brim-financial-inc"
    assert receipt["observed_linkedin_requested_url"] == OLD
    assert receipt["observed_linkedin_final_url"] == CURRENT
    assert receipt["verified_legal_name_aliases"] == ["Brim Financial"]
    assert company_quality_receipt_matches_claim(receipt, company.model_dump())
    # Neither the submitted company nor the provider observation is rewritten.
    assert company.company_linkedin == OLD
    assert verdict == _verdict()


def test_current_anchor_observed_old_redirect_preserves_existing_one_request_path():
    company = _company(company_name="Brim Financial", company_linkedin=CURRENT)
    resolved, fetch = _resolve(company, _verdict(observed_company_linkedin=OLD),
                               responses=[(999, CURRENT, "")])
    receipt = _receipt(company, resolved, anchor=None)
    assert fetch.await_count == 1
    assert receipt["decision"] == "match"
    assert receipt["reason_code"] == "verified_linkedin_redirect_alias"
    assert company_quality_receipt_matches_claim(receipt, company.model_dump())


@pytest.mark.parametrize("name,website,anchor", [
    ("Unrelated Corporation", "https://brimfinancial.com/", HOMEPAGE_IDENTITY),
    ("Brim Financial", "https://unrelated.example/", HOMEPAGE_IDENTITY),
    ("Brim Financial", "https://brimfinancial.com/", None),
    ("Brim Financial", "https://brimfinancial.com/", {
        **HOMEPAGE_IDENTITY, "verified_legal_name_aliases": ["Other Company"],
    }),
])
def test_redirect_does_not_supply_missing_or_conflicting_name_domain_evidence(name, website, anchor):
    company = _company()
    resolved, fetch = _resolve(company, _verdict(
        observed_company_name=name, observed_company_website=website,
    ), responses=[(200, CURRENT, ""), (200, CURRENT, "")])
    assert fetch.await_count == 2
    assert _receipt(company, resolved, anchor=anchor)["decision"] != "match"


@pytest.mark.parametrize("final_url", [
    OLD,  # No redirect from the anchor; a shared name/domain is insufficient.
    "https://www.linkedin.com/company/unrelated",
    "https://example.com/company/brimfinancial",
    "https://linkedin.com.evil.example/company/brimfinancial",
    "https://www.linkedin.com/login?next=/company/brimfinancial",
    "http://www.linkedin.com/company/brimfinancial",
    "https://user:pass@www.linkedin.com/company/brimfinancial",
    "https://www.linkedin.com:invalid/company/brimfinancial",
    "https://www.linkedin.com/company/brimfinancial/posts",
    None,
    {},
])
def test_forward_alias_rejects_missing_unrelated_unsafe_or_malformed_redirect(final_url):
    company = _company()
    resolved, fetch = _resolve(company, _verdict(), responses=[
        (200, CURRENT, ""), (200, final_url, ""),
    ])
    assert fetch.await_count == 2
    assert resolved["observed_company_linkedin"] == CURRENT
    assert lead_scorer._VERIFIED_LINKEDIN_REDIRECT_FINAL not in resolved
    assert _receipt(company, resolved)["decision"] == "mismatch"


@pytest.mark.parametrize("error", [asyncio.TimeoutError, aiohttp.ClientError, ValueError])
def test_provider_errors_and_forged_transport_fields_cannot_prove_alias(error):
    verdict = _verdict(**{
        lead_scorer._VERIFIED_LINKEDIN_REDIRECT_REQUESTED: OLD,
        lead_scorer._VERIFIED_LINKEDIN_REDIRECT_FINAL: CURRENT,
    })
    resolved, fetch = _resolve(_company(), verdict, error=error)
    assert fetch.await_count == 2
    assert lead_scorer._VERIFIED_LINKEDIN_REDIRECT_REQUESTED not in resolved
    assert lead_scorer._VERIFIED_LINKEDIN_REDIRECT_FINAL not in resolved
    assert _receipt(_company(), resolved)["decision"] == "mismatch"


@pytest.mark.parametrize("phase", ["__aenter__", "__aexit__"])
@pytest.mark.parametrize("error", [asyncio.TimeoutError, aiohttp.ClientError, ValueError])
def test_session_context_failure_never_publishes_redirect_proof(phase, error):
    context = MagicMock()
    getattr(context, phase).side_effect = error
    fetch = AsyncMock(side_effect=[(200, CURRENT, ""), (200, CURRENT, "")])
    verdict = _verdict(**{
        lead_scorer._VERIFIED_LINKEDIN_REDIRECT_REQUESTED: OLD,
        lead_scorer._VERIFIED_LINKEDIN_REDIRECT_FINAL: CURRENT,
    })
    with patch.object(lead_scorer.aiohttp, "ClientSession", return_value=context), \
            patch.object(lead_scorer, "_fetch_bounded_html", fetch):
        resolved = asyncio.run(
            lead_scorer._resolve_observed_linkedin_redirect_alias(_company(), verdict)
        )
    assert fetch.await_count == (0 if phase == "__aenter__" else 2)
    assert resolved["observed_company_linkedin"] == CURRENT
    assert lead_scorer._VERIFIED_LINKEDIN_REDIRECT_REQUESTED not in resolved
    assert lead_scorer._VERIFIED_LINKEDIN_REDIRECT_FINAL not in resolved
    assert _receipt(_company(), resolved)["decision"] == "mismatch"


@pytest.mark.parametrize("submitted,observed", [
    (OLD, OLD),
    ("", CURRENT),
    ("https://example.com/company/brim-financial-inc", CURRENT),
    ("http://www.linkedin.com/company/brim-financial-inc", CURRENT),
    (OLD, "https://www.linkedin.com/login"),
    (OLD, [CURRENT]),
    ("https://www.linkedin.com/company/1234", CURRENT),
    (OLD, "https://www.linkedin.com/company/1234"),
])
def test_equal_missing_unsafe_and_numeric_inputs_do_not_fetch(submitted, observed):
    resolved, fetch = _resolve(_company(company_linkedin=submitted),
                               _verdict(observed_company_linkedin=observed))
    assert fetch.await_count == 0
    assert lead_scorer._VERIFIED_LINKEDIN_REDIRECT_FINAL not in resolved
