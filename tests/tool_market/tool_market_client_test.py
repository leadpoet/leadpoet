"""The buyer client signs what the gateway verifies, and its MCP server forwards calls."""

from __future__ import annotations

import io
import json
from urllib.parse import urlsplit

import pytest

from bittensor_wallet import Keypair

from lab_arena.chain import wallet_signature_verifier
from lab_arena.lead_orders.auth import verify_buyer_request
from lab_arena.tool_market import btauth, client as client_module


class Recorder:
    status = 200

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


def test_a_call_with_no_api_key_is_signed_with_btauth_to_the_published_service_hotkey(monkeypatch):
    bob = Keypair.create_from_uri("//Bob").ss58_address
    sent = Recorder({"publishers": [{"netuid": 71, "hotkey": bob}], "status": 200})
    monkeypatch.setattr(client_module, "_open", sent)
    alice = Keypair.create_from_uri("//Alice")
    client = client_module.Client("https://gw.example:443/", keypair=alice)
    client.call("hunter", "email_finder", {"domain": "acme.com"})
    client.call("hunter", "email_finder", {"domain": "acme.com"})
    paths = [urlsplit(r.full_url).path for r in sent.sent]
    assert paths == ["/tool-market/services.json", "/tool-market/call", "/tool-market/call"]  # read once
    request = sent.sent[1]
    headers = {k.lower(): v for k, v in request.header_items()}
    identity = btauth.verify(headers, request.data, method="POST", target="/tool-market/call", receiver=bob,
                             verifier=wallet_signature_verifier, use_nonce=lambda *a: True)
    assert identity.hotkey == alice.ss58_address and "authorization" not in headers


def test_the_terms_are_signed_in_the_scheme_whose_message_is_stored(monkeypatch):
    sent = Recorder({"status": "accepted"})
    monkeypatch.setattr(client_module, "_open", sent)
    alice = Keypair.create_from_uri("//Alice")
    client = client_module.Client("https://gw.example:443/", keypair=alice)
    client.request("POST", "/tool-market/terms", {"role": "buyer"}, proof=True)
    request = sent.sent[0]
    identity = verify_buyer_request(
        method="POST", host="gw.example", path="/tool-market/terms", body=request.data,
        hotkey=request.get_header("X-hotkey"), timestamp=request.get_header("X-timestamp"),
        signature=request.get_header("X-signature"), verifier=wallet_signature_verifier,
        use_nonce=lambda *a: True)
    assert identity.hotkey == alice.ss58_address


def test_the_mcp_server_lists_two_tools_and_calls_with_the_api_key(monkeypatch):
    sent = Recorder({"status": 200, "charge_microusd": 11})
    monkeypatch.setattr(client_module, "_open", sent)
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
                                        "params": {"domain": "acme.com"}, "private": False}


def test_a_refused_call_reaches_the_agent_as_an_error(monkeypatch):
    def refuse(request, timeout):
        raise client_module.urllib.error.HTTPError(request.full_url, 402, "", {},
                                                   io.BytesIO(b'{"error": "tab_budget"}'))

    monkeypatch.setattr(client_module, "_open", refuse)
    reply = client_module.mcp_answer(client_module.Client("https://gw.example", api_key="k"), {
        "jsonrpc": "2.0", "id": 9, "method": "tools/call",
        "params": {"name": "call_tool", "arguments": {"tool": "hunter", "operation": "email_finder"}}})
    assert reply["result"]["isError"] is True
    assert json.loads(reply["result"]["content"][0]["text"]) == {"error": "tab_budget", "status": 402}


def test_an_unreachable_gateway_fails_the_call_not_the_mcp_server(monkeypatch):
    def down(request, timeout):
        raise client_module.urllib.error.URLError("connection refused")

    monkeypatch.setattr(client_module, "_open", down)
    lines = [json.dumps({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                         "params": {"name": "find_tool", "arguments": {}}}) for i in (1, 2)]
    out = io.StringIO()
    client_module.serve_mcp(client_module.Client("https://gw.example", api_key="k"), lines, out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [(r["id"], r["result"]["isError"]) for r in replies] == [(1, True), (2, True)]


def test_the_client_names_its_gateway_and_follows_no_redirect(monkeypatch):
    for url in ("http://gw.example", "ftp://gw.example", "gw.example"):
        with pytest.raises(ValueError):
            client_module.Client(url)
    client_module.Client("http://localhost:8080")  # this machine only

    def moved(request, timeout):  # what the no-redirect opener raises for a 302
        raise client_module.urllib.error.HTTPError(request.full_url, 302, "", {},
                                                   io.BytesIO(b'{"status": "found"}'))

    monkeypatch.setattr(client_module, "_open", moved)
    answer = client_module.Client("https://gw.example", api_key="k").tools()
    assert answer == {"error": "http_302", "status": 302}  # an error body's own status cannot clash
    assert client_module._NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example/") is None


def test_mcp_returns_vendor_failure(monkeypatch):
    sent = Recorder({"status": 503, "body": {}, "charge_microusd": 0})
    monkeypatch.setattr(client_module, "_open", sent)
    client = client_module.Client("https://gw.example", api_key="tmt_key")
    msg = {"id": 1, "method": "tools/call", "params": {"name": "call_tool", "arguments": {
        "tool": "exa", "operation": "search", "private": True}}}
    assert client_module.mcp_answer(client, msg)["result"]["isError"]
    assert len(sent.sent) == 1


def test_a_private_request_is_never_dropped_on_the_way(monkeypatch):
    sent = Recorder({"status": 200})
    monkeypatch.setattr(client_module, "_open", sent)
    client = client_module.Client("https://gw.example", api_key="k")
    for private in (True, "true", 1):  # the gateway refuses all but True; none may become a public call
        client_module.mcp_answer(client, {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "call_tool", "arguments": {"tool": "hunter", "operation": "email_finder", "private": private}}})
        assert json.loads(sent.sent[-1].data)["private"] == private


def test_an_answer_that_is_not_a_json_object_is_an_error(monkeypatch):
    for body in (b"<html>proxy</html>", b"[1]"):
        page = Recorder(None)
        page.read = lambda body=body: body
        monkeypatch.setattr(client_module, "_open", page)
        assert client_module.Client("https://gw.example", api_key="k").tools() == {"error": "http_200", "status": 200}
