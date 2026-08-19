from gateway.utils.geo_normalize import (
    CITIES_BY_COUNTRY,
    US_CITIES_BY_STATE,
    validate_location,
)


def test_city_lookup_keys_match_lowercase_validation_normalization() -> None:
    assert all(
        city == city.lower()
        for cities in US_CITIES_BY_STATE.values()
        for city in cities
    )
    assert all(
        city == city.lower()
        for cities in CITIES_BY_COUNTRY.values()
        for city in cities
    )


def test_mixed_case_lookup_cities_validate() -> None:
    assert validate_location(
        city="Holland",
        state="Ohio",
        country="United States",
    ) == (True, None)
    assert validate_location(
        city="Lebanon",
        state="Ohio",
        country="United States",
    ) == (True, None)
