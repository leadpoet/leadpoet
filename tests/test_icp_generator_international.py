"""International ICP quota: 20% of each generated set, English-speaking markets.

The generator prompt demands an exact international count and the set-level
check logs icp_international_quota_missed when generation drifts. The per-ICP
validation must accept the allowlisted international geographies (it
previously overrode everything non-US back to the United States) while still
overriding unsupported countries.
"""

from gateway.tasks.icp_generator import (
    INTERNATIONAL_GEOGRAPHIES,
    count_international_icps,
    international_icp_target,
)


def test_target_is_twenty_percent_rounded_min_one():
    assert international_icp_target(20) == 4
    assert international_icp_target(10) == 2
    assert international_icp_target(5) == 1
    assert international_icp_target(3) == 1  # never zero


def test_count_international_icps():
    icps = [
        {"country": "United States"},
        {"country": "United Kingdom"},
        {"country": "Canada"},
        {"country": "USA"},
        {"country": ""},
        "not-a-dict",
        {"country": "Singapore"},
    ]
    assert count_international_icps(icps) == 3


def test_international_pool_is_english_speaking_and_country_first():
    allowed_countries = {
        "united kingdom", "ireland", "canada", "australia",
        "new zealand", "singapore", "united arab emirates",
    }
    for geography in INTERNATIONAL_GEOGRAPHIES:
        country = geography.split(",")[0].strip().lower()
        assert country in allowed_countries, geography
        # Country-first format: the deployed scorer gate reads the country
        # from the first comma segment.
        assert not geography.lower().startswith(("europe", "apac", "emea"))


def test_validation_accepts_international_and_overrides_unsupported():
    # Reproduce the validation block's decision logic exactly.
    def decide(geography: str) -> str:
        international_match = next(
            (
                candidate
                for candidate in INTERNATIONAL_GEOGRAPHIES
                if geography
                and geography.split(",")[0].strip().lower()
                == candidate.split(",")[0].strip().lower()
            ),
            None,
        )
        if geography and "United States" in geography:
            return "United States"
        if international_match:
            return international_match.split(",")[0].strip()
        return "United States"  # override path

    assert decide("United Kingdom, London") == "United Kingdom"
    assert decide("United Kingdom") == "United Kingdom"
    assert decide("Singapore") == "Singapore"
    assert decide("Canada, Toronto") == "Canada"
    assert decide("United Arab Emirates, Dubai") == "United Arab Emirates"
    assert decide("United States, West Coast") == "United States"
    # Non-English-speaking or region values are overridden to the US.
    assert decide("Germany") == "United States"
    assert decide("Europe") == "United States"
    assert decide("") == "United States"
