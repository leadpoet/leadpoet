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
