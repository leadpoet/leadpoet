"""Exercise normal gateway construction with the external dependencies replaced."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lab_arena.tool_market import wiring
from lab_arena.tool_market.config import ToolMarketConfig
from lab_arena.tool_market.market import ToolMarketError


def test_real_factory_cannot_sell_or_solicit_spend_without_supplier_settlement(monkeypatch):
    buyer = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    config = ToolMarketConfig(mode="live", tools=frozenset({"zerobounce"}), gamma_service=buyer,
                              public_base_url="https://market.example", platform_max_asks={"zerobounce_validate": 5_000})
    store = Mock()
    store.open_tab.return_value = {"status": "opened", "tab_id": "existing"}
    store.tab.return_value = {"open": True}
    store.close_tab.return_value = {"open": False}
    store.book.return_value = []
    store.usage.return_value = {"calls": [], "watermark": 0}
    transport = Mock()
    monkeypatch.setattr(wiring, "ToolMarketStore", lambda _: store)
    monkeypatch.setattr(wiring, "PostgrestTransport", lambda *a, **k: None)
    monkeypatch.setattr(wiring.chain_module, "connect_substrate", lambda *a, **k: Mock())
    monkeypatch.setattr(wiring.chain_module, "load_arena_cutover", lambda: Mock())
    monkeypatch.setattr(wiring.vault_trust, "load_deployment", lambda: Mock())
    monkeypatch.setattr(wiring, "VaultClient", lambda *a, **k: Mock())
    monkeypatch.setattr(wiring, "VaultTransport", lambda _: transport)
    monkeypatch.setattr(wiring, "MarketGammaReader", lambda *a, **k: Mock())
    for name, value in (("LAB_ARENA_CHAIN_ENDPOINT", "wss://example.test"),
                        ("LAB_ARENA_SUPABASE_URL", "https://example.test"), ("LAB_ARENA_SERVICE_KEY", "test"),
                        ("LEAD_ORDER_VAULT_CALLER_TOKEN", "test")):
        monkeypatch.setenv(name, value)
    instance = wiring.build_from_environment(config)
    identity = SimpleNamespace(hotkey=buyer)
    call = {"tool": "zerobounce", "operation": "validate", "params": {"email": "a@example.test"}}
    for request in (lambda: instance.open_tab(identity, {"budget_microusd": 1000, "seconds": 90}),
                    lambda: instance.call("tmt_existing", call), lambda: instance.call("", call, identity),
                    lambda: instance.platform_call("tmt_existing", "zerobounce_validate", {"payload": call["params"]})):
        with pytest.raises(ToolMarketError, match="miner_settlement_unavailable") as refused:
            request()
        assert refused.value.status == 503
    store.reserve_call.assert_not_called()
    store.open_tab.assert_not_called()
    transport.send.assert_not_called()
    services, spec = instance.services(), instance.openapi()
    assert services["payments"] == [] and "/tool-market/call" not in spec["paths"]
    assert "spend gamma" not in json.dumps(spec).lower()
    assert instance.tool_catalog()["tools"][0]["tool"] == "zerobounce"
    assert instance.tab("existing")["open"]
    assert not instance.close_tab("existing")["open"]
    assert instance.usage(identity, 0, 100)["calls"] == []
