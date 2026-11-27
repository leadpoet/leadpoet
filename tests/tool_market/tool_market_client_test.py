"""The buyer client signs what the gateway verifies, and its MCP server forwards calls."""

from __future__ import annotations

import io
import json

from bittensor_wallet import Keypair

from lab_arena.chain import wallet_signature_verifier
from lab_arena.lead_orders.auth import verify_buyer_request
from lab_arena.tool_market import client as client_module


class Recorder:
    def __init__(self, answer):
        self.answer, self.sent = answer, []

    def __call__(self, request, timeout):
        self.sent.append(request)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.answer).encode()


def test_a_signed_request_passes_the_gateways_check(monkeypatch):
    sent = Recorder({"status": "opened", "token": "tmt_x"})
    monkeypatch.setattr(client_module.urllib.request, "urlopen", sent)
    alice = Keypair.create_from_uri("//Alice")
    client = client_module.Client("https://gw.example:443/", keypair=alice)
    client.request("POST", "/tool-market/tabs", {"budget_microusd": 1, "seconds": 60}, signed=True)
    request = sent.sent[0]
    identity = verify_buyer_request(
        method="POST", host="gw.example", path="/tool-market/tabs", body=request.data,
        hotkey=request.get_header("X-hotkey"), timestamp=request.get_header("X-timestamp"),
        signature=request.get_header("X-signature"), verifier=wallet_signature_verifier,
        use_nonce=lambda *a: True)
    assert identity.hotkey == alice.ss58_address


def test_the_mcp_server_lists_two_tools_and_calls_with_the_api_key(monkeypatch):
    sent = Recorder({"status": 200, "charge_microusd": 11})
    monkeypatch.setattr(client_module.urllib.request, "urlopen", sent)
    lines = [json.dumps(m) for m in (
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "call_tool", "arguments": {
            "tool": "hunter", "operation": "email_finder", "params": {"domain": "acme.com"}}}},
    )]
    out = io.StringIO()
    client_module.serve_mcp(client_module.Client("https://gw.example", api_key="tmt_key"), lines, out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3]  # no reply to the notification
    assert [t["name"] for t in replies[1]["result"]["tools"]] == ["find_tool", "call_tool"]
    assert replies[2]["result"]["isError"] is False
    request = sent.sent[0]
    assert request.get_header("Authorization") == "Bearer tmt_key"
    assert json.loads(request.data) == {"tool": "hunter", "operation": "email_finder",
                                        "params": {"domain": "acme.com"}}


def test_a_refused_call_reaches_the_agent_as_an_error(monkeypatch):
    def refuse(request, timeout):
        raise client_module.urllib.error.HTTPError(request.full_url, 402, "", {},
                                                   io.BytesIO(b'{"error": "tab_budget"}'))

    monkeypatch.setattr(client_module.urllib.request, "urlopen", refuse)
    reply = client_module.mcp_answer(client_module.Client("https://gw.example", api_key="k"), {
        "jsonrpc": "2.0", "id": 9, "method": "tools/call",
        "params": {"name": "call_tool", "arguments": {"tool": "hunter", "operation": "email_finder"}}})
    assert reply["result"]["isError"] is True
    assert json.loads(reply["result"]["content"][0]["text"]) == {"error": "tab_budget", "status": 402}


def test_an_unreachable_gateway_fails_the_call_not_the_mcp_server(monkeypatch):
    def down(request, timeout):
        raise client_module.urllib.error.URLError("connection refused")

    monkeypatch.setattr(client_module.urllib.request, "urlopen", down)
    lines = [json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                         "params": {"name": "find_tool", "arguments": {}}}) for i in (1, 2)]
    out = io.StringIO()
    client_module.serve_mcp(client_module.Client("https://gw.example", api_key="k"), lines, out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [(r["id"], r["result"]["isError"]) for r in replies] == [(1, True), (2, True)]
