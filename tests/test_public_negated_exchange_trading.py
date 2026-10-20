"""A negated current market statement is not Public-stage proof."""

import pytest

from qualification.scoring import company_evidence_investigator as investigator


DALLAS_URL = "https://investor.dallasnewscorp.com/overview"
DALLAS_QUOTE = (
    "On September 24, 2025, DallasNews Corporation (Nasdaq: DALN) announced "
    "the successful completion of its acquisition by Hearst, one of the "
    "nation's largest global, diversified information, services and media "
    "companies. As a result of the acquisition, DallasNews stock is no "
    "longer traded on Nasdaq and was purchased by Hearst for $16.50 per share."
)


def _validated(quote, *, name="Acme", domain="acme.example", url=None):
    url = url or f"https://{domain}/investors"
    return investigator._validated_findings(
        {"findings": [{
            "target": "stage", "status": "VERIFIED", "observed_value": "Public",
            "evidence_url": url, "evidence_quote": quote,
            "reason": "current issuer market status",
        }]},
        targets=("stage",),
        fetched_pages={url: quote},
        fetched_final_urls={url: url},
        first_party_domains={domain},
        identity_names={"".join(ch for ch in name.casefold() if ch.isalnum())},
        identity_anchor={"submitted_name": name, "submitted_domain": domain},
    )["stage"]["status"]


def test_exact_dallasnews_overview_does_not_prove_current_public_stage():
    assert _validated(
        DALLAS_QUOTE, name="DallasNews Corporation", domain="dallasnewscorp.com",
        url=DALLAS_URL,
    ) == "UNPROVEN"


@pytest.mark.parametrize("quote", [
    "Acme common stock is no longer traded on Nasdaq.",
    "Acme common stock is not currently traded on Nasdaq.",
    "Acme common stock is no longer listed on Nasdaq.",
    "Acme is no longer publicly traded.",
    "Acme (Nasdaq: ACM) announced an update. Acme was delisted yesterday.",
    "Acme common stock trades on Nasdaq. Acme was taken private yesterday.",
    "Acme bonds trade on Nasdaq.",
])
def test_negated_completed_and_nonequity_proof_is_unproven(quote):
    assert _validated(quote) == "UNPROVEN"


@pytest.mark.parametrize("quote", [
    "Acme common stock is currently traded on Nasdaq.",
    "Acme common stock is listed on Nasdaq.",
    "Acme (Nasdaq: ACM) announced current results. "
    "OtherCo stock is no longer traded on Nasdaq.",
    "OtherCo stock is no longer traded on Nasdaq. "
    "Acme (Nasdaq: ACM) announced current results.",
    "Acme (Nasdaq: ACM) announced current results. "
    "Its subsidiary OtherCo stock is no longer traded on Nasdaq.",
    "Acme common stock trades on Nasdaq. It plans to delist next year.",
    "Acme common stock is no longer traded on Nasdaq. "
    "Acme common stock is currently traded on NYSE.",
])
def test_independent_current_issuer_proof_survives(quote):
    assert _validated(quote) == "VERIFIED"


def test_other_issuer_ticker_does_not_prove_investigated_company_public():
    assert _validated(
        "Acme stock is no longer traded on Nasdaq. "
        "OtherCo (Nasdaq: OTHR) announced current results."
    ) == "UNPROVEN"


def test_name_prefix_sponsor_exit_does_not_negate_listed_issuer():
    assert _validated(
        "NextEra Energy Partners (NYSE: NEP) announced current earnings. "
        "Its separate sponsor NextEra Energy is no longer publicly traded.",
        name="NextEra Energy Partners", domain="nexteraenergypartners.com",
    ) == "VERIFIED"
