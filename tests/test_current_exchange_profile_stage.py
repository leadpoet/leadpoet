"""Current exchange quote cards can prove Public without weakening history guards."""

import pytest

from qualification.scoring.lead_scorer import _stage_evidence_supports_observation


JLL = (
    "NYSE / JLL JONES LANG LASALLE INC 321.91 Stock price decreased by "
    "-2.49 dollars, -0.77 percent -2.49 (-0.77%)"
)
PROLOGIS = (
    "Quote & Chart Chart Prologis Inc. New York Stock Exchange: PLD Events "
    "Price Market Line 10m Volume. Today September 25, 2026 4:03 PM ET "
    "Last 133.05 Volume 2.54m"
)


@pytest.mark.parametrize(("quote", "url", "domains", "names"), [
    (JLL, "https://www.nyse.com/quote/XNYS:JLL", (), ("JLL",)),
    (PROLOGIS, "https://ir.prologis.com/stock/quote-chart",
     ("prologis.com",), ("Prologis", "Prologis, Inc.")),
])
def test_current_exchange_profile_is_bound_public_evidence(
    quote, url, domains, names,
):
    assert _stage_evidence_supports_observation(
        "Public", quote, evidence_url=url,
        first_party_domains=domains, identity_names=names,
    )


@pytest.mark.parametrize("quote", [
    "NYSE / JLL ACME HOLDINGS INC 321.91 Stock price decreased by 2 dollars",
    "JLL completed its IPO on July 22, 1997 on the New York Stock Exchange.",
    "Prologis Inc. New York Stock Exchange: PLD",
    "Prologis Inc. New York Stock Exchange: PLD ceased trading today.",
    "Prologis Inc. New York Stock Exchange: PLD became private today.",
])
def test_exchange_profile_rejects_wrong_issuer_history_static_and_supersession(quote):
    assert not _stage_evidence_supports_observation(
        "Public", quote,
        evidence_url="https://www.nyse.com/quote/XNYS:JLL",
        identity_names=("JLL",),
    )


def test_exchange_profile_rejects_non_exchange_third_party_source():
    assert not _stage_evidence_supports_observation(
        "Public", JLL,
        evidence_url="https://example.com/quote/JLL",
        identity_names=("JLL",),
    )
