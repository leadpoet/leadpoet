"""The higher-cost Harvest operations keep distinct names and native request contracts."""

from urllib.parse import parse_qs, urlsplit

import pytest

from lab_arena.tool_market import catalog, platform
from lead_order_vault.tools import TOOLS as VAULT_TOOLS


def wire(alias, payload):
    tool, name, params, price_key = platform.platform_request(alias, payload)
    op = catalog.operation(tool, name)
    url, _, body = op.build(params)
    parts = urlsplit(url)
    assert VAULT_TOOLS[tool].route(op.method, parts.path, parts.query)
    assert body == b""
    return name, price_key, parts.path, parse_qs(parts.query)


@pytest.mark.parametrize("main", [True, False])
def test_email_profile_uses_only_its_distinctly_priced_operation(main):
    name, key, path, query = wire(
        "harvestapi_get_profile", {"url": "https://linkedin.com/in/example", "findEmail": "true", "main": main}
    )
    assert (name, key, path) == (
        "linkedin_profile_email",
        "harvestapi_get_profile.findEmail",
        "/linkedin/profile",
    )
    assert query == {
        "query": ["https://linkedin.com/in/example"],
        "findEmail": ["true"],
        **({"main": ["true"]} if main else {}),
    }
    with pytest.raises(catalog.CatalogError):
        catalog.operation("harvestapi", "linkedin_profile").build({"url": "example", "findEmail": "true"})


def test_standalone_people_search_preserves_filters_and_requests_one_page():
    params = {
        "currentCompanies": "https://www.linkedin.com/company/example",
        "currentJobTitles": "VP Sales,Sales Director",
        "locations": "Seattle,London",
        "companyHeadcount": "D,E",
        "recentlyChangedJobs": True,
        "excludeCurrentJobTitles": "Consultant",
        "page": 1,
    }
    name, key, path, query = wire("harvestapi_search_leads", params)
    assert (name, key, path) == ("search_leads", "harvestapi_search_leads", "/linkedin/lead-search")
    assert query == {k: ["true" if v is True else str(v)] for k, v in params.items()}


@pytest.mark.parametrize(
    "extra",
    [
        {"sessionId": "session"},
        {"page": 2},
        {"salesNavUrl": "search"},
        {"requiredAccountId": "id"},
        {"usePrivatePool": "true"},
        {"lastNames": "Jones,,Doe"},
        {"seniorityLevelIds": ",".join(["1"] * 21)},
    ],
)
def test_standalone_search_refuses_unsupported_session_or_provider_controls(extra):
    with pytest.raises(catalog.CatalogError):
        wire("harvestapi_search_leads", {"currentJobTitles": "Sales", **extra})


def test_search_usage_and_filter_required():
    op = catalog.operation("harvestapi", "search_leads")
    with pytest.raises(catalog.CatalogError):
        op.build({"page": 1})
    assert op.units(200, {"status": 200, "elements": [{"id": "123"}]}) == 1
    assert op.units(200, {"status": 200, "elements": []}) == 1
    assert op.units(200, {"status": 503, "error": "unavailable"}) is None
    assert op.timeout_seconds >= 65
