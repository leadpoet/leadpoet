"""Exercise normal gateway construction with the external dependencies replaced."""
from unittest.mock import Mock

from lab_arena.tool_market import wiring
from lab_arena.tool_market.config import ToolMarketConfig


def test_live_factory_uses_the_same_gamma_dispatcher_as_the_purchase_tests(monkeypatch):
    buyer = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    config = ToolMarketConfig(mode="live", tools=frozenset({"zerobounce"}), gamma_service=buyer,
                              public_base_url="https://market.example", platform_max_asks={"zerobounce_validate": 5_000})
    store = Mock()
    monkeypatch.setattr(wiring, "ToolMarketStore", lambda _: store)
    monkeypatch.setattr(wiring, "PostgrestTransport", lambda *a, **k: None)
    monkeypatch.setattr(wiring.chain_module, "connect_substrate", lambda *a, **k: Mock())
    monkeypatch.setattr(wiring.chain_module, "load_arena_cutover", lambda: Mock())
    monkeypatch.setattr(wiring.vault_trust, "load_deployment", lambda: Mock())
    monkeypatch.setattr(wiring, "VaultClient", lambda *a, **k: Mock())
    monkeypatch.setattr(wiring, "VaultTransport", lambda _: Mock())
    monkeypatch.setattr(wiring, "MarketGammaReader", lambda *a, **k: Mock())
    for name, value in (("LAB_ARENA_CHAIN_ENDPOINT", "wss://example.test"),
                        ("LAB_ARENA_SUPABASE_URL", "https://example.test"), ("LAB_ARENA_SERVICE_KEY", "test"),
                        ("LEAD_ORDER_VAULT_CALLER_TOKEN", "test")):
        monkeypatch.setenv(name, value)
    instance = wiring.build_from_environment(config)
    assert instance.services()["payments"][0]["type"] == "gamma"
    assert "/tool-market/call" in instance.openapi()["paths"]
