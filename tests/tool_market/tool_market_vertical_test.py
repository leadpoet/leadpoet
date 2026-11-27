"""Public HTTP contracts through signature verification, PostgreSQL, vault and vendor metering."""
import json
import time
from urllib.parse import parse_qs, urlsplit

from bittensor_wallet import Keypair
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api import tool_market as routes
from lab_arena.chain import wallet_signature_verifier
from lab_arena.lead_orders.auth import signed_message
from lab_arena.lead_orders.vault_client import VaultClient, VaultTransport
from lab_arena.tool_market.config import ToolMarketConfig, PUBLISHED_TERMS
from lab_arena.tool_market.market import ATTESTATION, ToolMarket
from lead_order_vault.vault import UpstreamResponse
from tests.lead_order_vault.vault_test import _vault, _seal
from tests.tool_market.tool_market_pg_test import FakeGamma, database, store


def test_private_platform_lookup_from_funded_tab_to_miner_statement(store, monkeypatch):
    buyer, miner = (Keypair.create_from_uri(uri) for uri in ("//Alice", "//Bob"))
    version, digest = next(iter(PUBLISHED_TERMS.items()))
    cfg = ToolMarketConfig(mode="live", tools=frozenset({"zerobounce"}), gamma_service=miner.ss58_address,
                           public_base_url="https://gw.example", terms_version=version, terms_sha256=digest,
                           approved_suppliers=frozenset({("zerobounce", miner.ss58_address)}),
                           lead_search_verify_max_ask=5000, private_suppliers=frozenset({miner.ss58_address}))
    secret = "fixture-zb-key"
    statuses = iter(["valid", "unknown"])
    def provider(method, host, target, headers, body, timeout, limit):
        parsed = urlsplit(target)
        assert host == "api.zerobounce.net" and method == "GET"
        assert parse_qs(parsed.query)["api_key"] == [secret]
        if parsed.path == "/v2/getcredits":
            value = {"Credits": "100"}
        else:
            assert parsed.path == "/v2/validate"
            assert parse_qs(parsed.query)["email"] == ["buyer@example.test"]
            value = {"status": next(statuses)}
        return UpstreamResponse(200, [("content-type", "application/json")], json.dumps(value).encode())

    vault = _vault(provider)
    transport = VaultClient({"url": "https://vault.example"}, "fixture-caller-token")
    # Replace the attested TLS hop and external vendor network with deterministic local transports.
    handlers = {"/v1/check": vault.check, "/v1/call": vault.call}
    monkeypatch.setattr(transport, "_post", lambda path, doc, timeout: handlers[path](doc))
    gamma = FakeGamma()
    gamma.units[buyer.ss58_address] = 10**9
    market = ToolMarket(config=cfg, store=store, vault=transport, transport=VaultTransport(transport),
                        gamma_reader=gamma, current_epoch=lambda: 71, is_registered=lambda hk: hk == miner.ss58_address)
    app = FastAPI()
    app.include_router(routes.router)
    routes.configure(builder=lambda: (cfg, market), verifier=wallet_signature_verifier)
    try:
        with TestClient(app) as http:
            def signed(key, path, payload=None):
                body = json.dumps(payload).encode() if payload is not None else b""
                method, ts = ("POST" if payload is not None else "GET"), str(int(time.time()))
                message = signed_message(method, cfg.public_host, path, ts, body)
                response = http.request(method, path, content=body, headers={"x-hotkey": key.ss58_address,
                    "x-timestamp": ts, "x-signature": key.sign(message.encode()).hex()})
                assert response.status_code == 200, response.text
                return response.json()
            for role, key in (("buyer", buyer), ("miner", miner)):
                signed(key, "/tool-market/terms", dict(role=role, terms_version=version, terms_sha256=digest))
            envelope = _seal(secret, provider="zerobounce", hotkey=miner.ss58_address)
            listing = signed(miner, "/tool-market/keys", dict(tool="zerobounce", envelope=envelope,
                             attestation=ATTESTATION, daily_cap_microusd=1_000_000))
            signed(miner, "/tool-market/asks", dict(key_id=listing["key_id"], ask_microusd=5_000))
            tab = signed(buyer, "/tool-market/tabs", dict(budget_microusd=50_000, seconds=600))
            headers = {"Authorization": "Bearer " + tab["token"], "x-deepline-execute-response-contract": "raw-v2"}
            for verdict, cost in (("valid", 0.0055), ("unknown", 0)):
                response = http.post("/tool-market/api/v2/integrations/zerobounce_validate/execute",
                                     headers=headers, json={"payload": {"email": "buyer@example.test"}})
                result = response.json()
                assert result["status"] == "completed" and result["toolResponse"]["raw"]["status"] == verdict
                assert result["billing"]["cost_usd"] == cost and secret not in response.text
            signed(miner, "/tool-market/asks", dict(key_id=listing["key_id"], ask_microusd=5001))
            refused = http.post("/tool-market/api/v2/integrations/zerobounce_validate/execute",
                                headers=headers, json={"payload": {"email": "buyer@example.test"}})
            assert refused.status_code == 503 and refused.json()["error"] == "no_supply"
            keys = signed(miner, "/tool-market/statement")["keys"]
            assert keys[0]["calls"] == 2 and keys[0]["owed_microusd"] == 5000
    finally:
        routes.configure()
