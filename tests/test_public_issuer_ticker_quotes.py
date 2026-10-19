import pytest

from qualification.scoring.lead_scorer import _stage_quote_supports_observation


@pytest.mark.parametrize(
    "quote",
    [
        (
            "Rapid7, Inc. (“Rapid7”) (NASDAQ: RPD), a global leader in "
            "AI-powered managed cybersecurity operations, announced that the "
            "company granted inducement awards, effective as of September 15, "
            "2026, to employees as a material inducement to commence employment "
            "with Rapid7 and its subsidiaries."
        ),
        (
            "BOSTON, Aug. 10, 2026 (GLOBE NEWSWIRE) -- Rapid7, Inc. "
            "(Nasdaq: RPD), a global leader in AI-powered managed cybersecurity "
            "operations, today announced its financial results for the second "
            "quarter 2026."
        ),
        (
            'Tenable Holdings, Inc. ("Tenable") (Nasdaq: TENB), the exposure '
            "management company, today announced financial results for the "
            "quarter ended June 30, 2026."
        ),
        (
            "Tenable® Holdings, Inc. (NASDAQ: TENB), the exposure management "
            "company, today announced the appointment of Dino DiMarino as "
            "Chief Revenue Officer."
        ),
        (
            "Rapid7, Inc. (“Rapid7” or the “Company”) (NASDAQ: RPD) announced "
            "current quarterly results."
        ),
    ],
)
def test_current_issuer_ticker_quotes_are_public_stage_proof(quote):
    assert _stage_quote_supports_observation("public", quote) is True


@pytest.mark.parametrize(
    "separator",
    ["--", "–", "—"],
)
def test_newswire_separator_starts_company_bound_ticker_proof(separator):
    quote = (
        f"AUSTIN, Texas{separator}(BUSINESS WIRE){separator}CrowdStrike "
        "(NASDAQ: CRWD) today announced an executive appointment."
    )

    assert _stage_quote_supports_observation("public", quote) is True


@pytest.mark.parametrize(
    "quote",
    [
        'Tenable Holdings, Inc. ("Tenable") announced results (Nasdaq: TENB).',
        '("Tenable") (Nasdaq: TENB)',
        'Tenable Holdings bonds ("Tenable") (Nasdaq: TENB)',
        'Formerly Tenable Holdings, Inc. ("Tenable") (Nasdaq: TENB)',
        (
            'Tenable Holdings, Inc. ("Tenable") (Nasdaq: TENB), then was '
            "taken private."
        ),
        (
            'Tenable Holdings, Inc. ("Tenable") (Nasdaq: TENB) listing is '
            "planned for next year."
        ),
        'Tenable Holdings, Inc. ("formerly listed") (Nasdaq: TENB)',
        'Tenable Holdings, Inc. (“not listed”) (Nasdaq: TENB)',
        'Tenable Holdings, Inc. ("Tenable not listed") (Nasdaq: TENB)',
        'Tenable Holdings, Inc. (“Tenable formerly listed”) (Nasdaq: TENB)',
        'Tenable Holdings, Inc. ("Tenable") (OTCQX: TENB)',
        'Tenable Holdings, Inc. ("Tenable") (Nasdaq:)',
    ],
)
def test_alias_and_newswire_grammar_preserves_public_stage_rejections(quote):
    assert _stage_quote_supports_observation("public", quote) is False


def _validated_public(quote, *, company="Acme", domain="acme.example", text=None):
    from qualification.scoring import company_evidence_investigator as investigator

    url = f"https://{domain}/investors"
    return investigator._validated_findings(
        {"findings": [{"target": "stage", "status": "VERIFIED",
                       "observed_value": "Public", "evidence_url": url,
                       "evidence_quote": quote, "reason": "Current listing"}]},
        targets=("stage",), fetched_pages={url: quote if text is None else text},
        fetched_final_urls={url: url}, first_party_domains={domain},
        identity_names={company.casefold()},
        identity_anchor={"submitted_name": company, "submitted_domain": domain,
                         "observed_name": company, "observed_domain": domain},
    )["stage"]


@pytest.mark.parametrize("company,domain,quote", [
    ("Change Financial", "changefinancial.com",
     "Change Financial (Change) is a global fintech, listed on the Australian Securities Exchange (ASX: CCA)."),
    ("Change Financial", "changefinancial.com", "Change Financial Limited (ASX: CCA) is a fintech."),
    ("Huron", "huronconsultinggroup.com",
     "Global professional services firm Huron (NASDAQ: HURN) will announce its financial results for the third quarter of 2026 after the market closes."),
    ("Acme", "acme.example", "Global technology firm Acme (ASX: ACM) announced results."),
    ("Acme", "acme.example", "Acme is listed on the Australian Securities Exchange."),
    ("Acme", "acme.example", "Acme is the parent company of Example and is listed on Nasdaq."),
    ("Acme", "acme.example", "Acme is listed on Nasdaq, but its planned merger remains subject to approval."),
    ("Acme", "acme.example", "Acme is listed on Nasdaq. A merger remains subject to approval."),
    ("Acme", "acme.example", "Acme is listed on Nasdaq. Its platform processes orders if customers pay."),
    ("Acme", "acme.example", "Acme is a sponsor of sporting events, listed on Nasdaq."),
    ("Acme", "acme.example", "Our common shares are listed on Nasdaq."),
    ("Acme", "acme.example", "Our shares are listed on the Australian Securities Exchange."),
    ("Acme", "acme.example", "Acme is a manager of investment funds, listed on Nasdaq."),
    ("Acme", "acme.example", "Acme is a manager of investment funds, listed on the Australian Securities Exchange."),
    ("Acme", "acme.example", "Acme is a fintech with outstanding debt, listed on the Australian Securities Exchange."),
    ("A. O. Smith", "aosmith.example", "A. O. Smith (NYSE: AOS) announced results."),
    ("J.M. Smucker", "smucker.example", "J.M. Smucker (NYSE: SJM) announced results."),
    ("L.B. Foster", "foster.example", "L.B. Foster (NASDAQ: FSTR) announced results."),
    ("Dr. Reddy's Laboratories", "reddy.example", "Dr. Reddy's Laboratories (NYSE: RDY) announced results."),
])
def test_issuer_bound_public_forms_pass_actual_finding_validation(company, domain, quote):
    assert _validated_public(quote, company=company, domain=domain)["status"] == "VERIFIED"


@pytest.mark.parametrize("quote", [
    "Acme partners with Example Corp (ASX: EXM).",
    "Acme partners with Example Corp, listed on the Australian Securities Exchange.",
    "Acme is a partner of Example Corp, listed on the Australian Securities Exchange.",
    "Acme is partnering with Example Corp, listed on the Australian Securities Exchange.",
    "Acme partners with Example Corp, listed on Nasdaq.",
    "Acme's parent Example Corp (ASX: EXM) announced results.",
    "Acme's owner Example Corp (ASX: EXM) announced results.",
    "Acme's subsidiary Example Corp (ASX: EXM) announced results.",
    "Acme is a subsidiary of Example Corp, listed on the Australian Securities Exchange.",
    "Acme formerly listed on the Australian Securities Exchange.",
    "Acme (ASX: ACM) was delisted.",
    "Acme (ASX: ACM) was taken private.",
    "Acme plans an initial public offering on the Australian Securities Exchange.",
    "Acme (ASX: ACM) listing is planned for next year.",
    "Acme will be listed on the Australian Securities Exchange.",
    "Acme is conditionally listed on the Australian Securities Exchange subject to approval.",
    "Acme bonds are listed on the Australian Securities Exchange.",
    "Acme debt is listed on the Australian Securities Exchange.",
    "Acme notes are listed on the Australian Securities Exchange.",
    "Acme (UNKNOWN: ACM) announced results.",
    "Acme completed its IPO in 2020.",
])
def test_expanded_public_forms_preserve_entity_and_current_equity_guards(quote):
    assert _validated_public(quote)["status"] == "UNPROVEN"


def test_expanded_public_proof_must_be_in_fetched_source():
    quote = "Acme (ASX: ACM) announced results."
    assert _validated_public(quote, text="Acme provides software.")["status"] == "UNPROVEN"


def test_anonymous_equity_listing_cannot_bind_third_party_source():
    from qualification.scoring import lead_scorer

    assert not lead_scorer._stage_evidence_supports_observation(
        "Public", "Our shares are listed on the Australian Securities Exchange.",
        evidence_url="https://other.example/investors",
        first_party_domains=("acme.example",), identity_names=("Acme",),
    )


@pytest.mark.parametrize("quote,expected", [
    ("Acme common stock is listed on NASDAQ under ticker ACME.", True),
    ("Acme is listed on the Australian Securities Exchange.", False),
    ("Acme (NASDAQ: ACM) announced results.", False),
])
def test_legacy_no_name_first_party_prose_does_not_expand_asx_or_ticker_trust(quote, expected):
    from qualification.scoring import lead_scorer

    assert lead_scorer._stage_evidence_supports_observation(
        "Public", quote, evidence_url="https://acme.example/investors",
        first_party_domains=("acme.example",),
    ) is expected
