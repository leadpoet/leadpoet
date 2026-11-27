"""Documented direct vendor contracts and uncertain usage."""
from urllib.parse import parse_qs, urlsplit

import pytest

from lab_arena.tool_market.catalog import operation


@pytest.mark.parametrize("name,slug", [("linkedin_profile", "in/lee"), ("linkedin_company", "company/acme")])
def test_harvest_maps_market_url_to_vendor_query_and_requires_a_result(name, slug):
    op = operation("harvestapi", name)
    profile = "https://www.linkedin.com/" + slug
    parts = urlsplit(op.build({"url": profile})[0])
    assert parse_qs(parts.query) == {"query": [profile]}
    assert op.units(200, {"status": 200, "element": {"id": "1234"}}) == 1
    for invalid in ({}, {"error": "temporary"}, {"status": 200, "element": None}, []):
        assert op.units(200, invalid) is None


def test_unrecognized_zero_bounce_verdict_does_not_create_a_charge():
    op = operation("zerobounce", "validate")
    assert op.units(200, {"status": "processing"}) is None
