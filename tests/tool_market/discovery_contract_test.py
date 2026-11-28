from lab_arena.tool_market.config import ToolMarketConfig
from lab_arena.tool_market.discovery import services_document, openapi


def test_api_endpoint_is_the_same_origin_used_by_the_operation_paths():
    config = ToolMarketConfig(public_base_url="https://gateway.example", tools=frozenset({"harvestapi"}))
    doc = services_document(config)["services"][0]
    spec = openapi(config)
    assert doc["endpoint"] == spec["servers"][0]["url"]
    assert doc["spec"] == doc["endpoint"] + "/tool-market/openapi.json"
    assert "/tool-market/catalog" in spec["paths"]
