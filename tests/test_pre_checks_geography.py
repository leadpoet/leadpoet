"""Country pre-check: recognized-country matching with LLM deferral.

The deterministic country gate must hard-fail only on an unambiguous
mismatch between recognized countries, resolve the shapes production and
client ICPs actually use (country+region, city-first, multi-country,
continents, US states), and defer unrecognizable geography strings to the
ICP-fit scorer instead of zeroing every company (the false-negative class
reported from the shared validation code).
"""

import pytest

from qualification.scoring.pre_checks import (
    _allowed_countries_from_icp_geography,
    _normalize_country,
    _resolve_country,
    check_country_match,
)


# ---------------------------------------------------------------------------
# Live production shapes (qualification_private_icp_sets carries plain
# "United States" / "United Arab Emirates" country values today).
# ---------------------------------------------------------------------------

def test_plain_country_match_passes():
    assert check_country_match("United States", "United States").passed
    assert check_country_match("United Arab Emirates", "United Arab Emirates").passed


def test_plain_country_mismatch_fails():
    assert not check_country_match("United Arab Emirates", "United States").passed
    assert not check_country_match("Canada", "United States").passed


def test_country_plus_region_suffix_passes():
    # Research Lab style: country first, regional nuance after the comma.
    assert check_country_match("United States", "United States, West Coast").passed
    assert not check_country_match("Canada", "United States, West Coast").passed


def test_empty_icp_geography_accepts_any_country():
    assert check_country_match("Germany", "").passed
    assert check_country_match("Germany", "   ").passed


def test_missing_lead_country_fails():
    assert not check_country_match("", "United States").passed
    assert not check_country_match("   ", "United States").passed


# ---------------------------------------------------------------------------
# Provider spellings: ISO codes and colloquial names must not create
# false negatives (reported from the site using this same validation code).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("lead", "icp"),
    (
        ("US", "United States"),
        ("USA", "United States"),
        ("U.S.A.", "United States"),
        ("America", "United States"),
        ("GB", "United Kingdom"),
        ("GBR", "United Kingdom"),
        ("UK", "United Kingdom"),
        ("England", "United Kingdom"),
        ("Scotland", "United Kingdom"),
        ("DE", "Germany"),
        ("Deutschland", "Germany"),
        ("Netherlands", "The Netherlands"),
        ("Holland", "Netherlands"),
        ("South Korea", "Korea, Republic of"),
        ("KR", "South Korea"),
        ("Russia", "Russian Federation"),
        ("Viet Nam", "Vietnam"),
        ("Türkiye", "Turkey"),
        ("AE", "United Arab Emirates"),
        ("UAE", "United Arab Emirates"),
    ),
)
def test_provider_spellings_match(lead, icp):
    assert check_country_match(lead, icp).passed, (lead, icp)


def test_iso_codes_still_reject_true_mismatches():
    assert not check_country_match("GB", "United States").passed
    assert not check_country_match("DE", "France").passed


def test_lowercase_prose_words_are_not_iso_codes():
    # "in" (India), "it" (Italy), "no" (Norway) as prose must not resolve.
    assert _resolve_country("in") is None
    assert _resolve_country("it") is None
    assert _resolve_country("no") is None
    assert _resolve_country("IN") == "india"
    # A prose geography must not accidentally permit India via "in".
    allowed = _allowed_countries_from_icp_geography("United States, in the region")
    assert allowed == frozenset({"united states"})


# ---------------------------------------------------------------------------
# City-first, multi-country, continent, and US-state geography shapes.
# ---------------------------------------------------------------------------

def test_city_first_geography_resolves_country():
    assert check_country_match("United Kingdom", "London, United Kingdom").passed
    assert not check_country_match("United States", "London, United Kingdom").passed


def test_multi_country_geography_permits_each_listed_country():
    geo = "United States or Canada"
    assert check_country_match("United States", geo).passed
    assert check_country_match("Canada", geo).passed
    assert not check_country_match("Mexico", geo).passed
    assert check_country_match("UK", "US/UK").passed


def test_continent_geography_enforces_membership():
    assert check_country_match("Germany", "Europe").passed
    assert check_country_match("France", "Europe").passed
    assert not check_country_match("United States", "Europe").passed
    assert check_country_match("United States", "North America").passed
    assert check_country_match("Japan", "Asia").passed
    assert not check_country_match("Japan", "Europe").passed


def test_city_state_country_geography_shapes():
    # Full "city, state, country" strings: unknown city segments are ignored,
    # the state and/or country segments carry the requirement.
    assert check_country_match(
        "United States", "San Francisco, California, United States"
    ).passed
    assert check_country_match("United States", "Austin, TX, US").passed
    assert not check_country_match(
        "Canada", "San Francisco, California, United States"
    ).passed
    assert check_country_match(
        "United Kingdom", "London, Greater London, United Kingdom"
    ).passed
    assert check_country_match("Canada", "Toronto, Ontario, Canada").passed
    # "city, state" with no country still implies the United States.
    assert check_country_match("United States", "New York City, New York").passed


def test_us_state_geography_permits_united_states():
    # "Georgia" is both a US state and a country: a US company must not be
    # rejected when the ICP names the state.
    assert check_country_match("United States", "Georgia, United States").passed
    assert check_country_match("United States", "Atlanta, Georgia").passed
    assert check_country_match("United States", "California").passed
    # The country Georgia also stays permitted for the ambiguous bare token.
    assert check_country_match("Georgia", "Georgia").passed


def test_unrecognized_geography_defers_to_fit_scorer():
    # Business regions with no ISO decomposition must not zero every company.
    for geo in ("EMEA", "DACH region", "Nordics", "Worldwide", "Remote-first"):
        assert check_country_match("Germany", geo).passed, geo
        assert check_country_match("United States", geo).passed, geo


def test_unrecognized_lead_country_fails_against_recognized_requirement():
    # Junk lead countries must not slip through a recognized requirement.
    assert not check_country_match("Atlantis", "United States").passed
    assert not check_country_match("Earth", "United States").passed


# ---------------------------------------------------------------------------
# Helper-level behavior pins.
# ---------------------------------------------------------------------------

def test_normalize_country_canonicalizes():
    assert _normalize_country("USA") == "united states"
    assert _normalize_country("Korea, Republic of") == "south korea"
    assert _normalize_country("Czech Republic") == "czechia"


def test_allowed_countries_from_country_with_comma_in_name():
    # A country name that itself contains a comma must resolve via alias.
    allowed = _allowed_countries_from_icp_geography("Korea, Republic of")
    assert "south korea" in allowed
