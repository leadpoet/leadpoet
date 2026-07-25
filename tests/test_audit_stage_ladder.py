"""ICP-gen #1: the company-stage hard gate used exact string equality, so a
real company whose honestly-reported stage differed lexically from the ICP's
range stage ("Series C+", "Public", "Private Equity") was zeroed. The ladder
helper must accept genuine matches without opening false positives."""

from qualification.scoring.lead_scorer import (
    _normalize_company_stage as norm,
    _company_stage_satisfies_icp as sat,
)


def _match(company, icp):
    return sat(norm(company), norm(icp))


def test_series_c_plus_accepts_c_and_later():
    for company in ("Series C", "Series C+", "Series D", "Series E", "Growth", "late stage"):
        assert _match(company, "Series C+") is True, company


def test_series_c_plus_rejects_earlier_and_other_categories():
    for company in ("Series A", "Series B", "Seed", "Public", "Private Equity"):
        assert _match(company, "Series C+") is False, company


def test_private_equity_synonyms():
    assert _match("PE-backed", "Private Equity") is True
    assert _match("Private Equity", "Private Equity") is True
    assert _match("Buyout", "Private Equity") is True
    assert _match("Series D", "Private Equity") is False


def test_public_synonyms():
    assert _match("Publicly traded", "Public") is True
    assert _match("Listed on NASDAQ", "Public") is True
    assert _match("Public", "Public") is True
    assert _match("Series E", "Public") is False


def test_exact_series_rounds_still_exact():
    assert _match("Series A", "Series A") is True
    assert _match("Series B round", "Series B") is True   # was a lexical FN
    assert _match("Series B", "Series A") is False         # B does not satisfy A
    assert _match("Seed", "Seed") is True
