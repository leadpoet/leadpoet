"""Buyer client for the Subnet 71 tool market: a CLI and a small MCP server.

After spending Gamma at the market, the hotkey signs each call (btauth/1)
and the call is charged to its balance. An agent that should not hold the
hotkey uses an API key instead: a tab the hotkey opens once, capped by its
budget, with an expiry. Usage is in "For buyers" in
docs/gamma-tool-market.md; ``mcp`` runs a stdio MCP server for agents.

``SUBNET71_URL`` (or ``--url``) is the gateway's public origin, with no path:
https, or http only to this machine. ``name=value`` sends text and
``name:=value`` sends JSON (a number, a boolean or a list).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from lab_arena.lead_orders.auth import signed_message
from lab_arena.tool_market import btauth
from lab_arena.tool_market.config import PUBLISHED_TERMS

TIMEOUT_SECONDS = 150  # above the 120 s a call can take (market.FAILOVER_SECONDS)
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is an error: following one would send the API key or a
    signed request to a host the buyer never named."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_open = urllib.request.build_opener(_NoRedirect).open


class Client:
    def __init__(self, url: str, api_key: str = "", keypair: Any = None) -> None:
        parts = urlsplit(url)
        if not (parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in _LOCAL_HOSTS)):
            raise ValueError("the gateway URL must be https")
        if not parts.hostname or parts.username is not None or parts.password is not None or (
            parts.path not in ("", "/") or parts.query or parts.fragment
        ):
            raise ValueError("the gateway URL must be an origin without credentials, query or path")
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.keypair = keypair
        self._receiver = ""
        self._nonce, self._nonce_lock = 0, threading.Lock()

    def _next_nonce(self) -> int:
        """Strictly increasing, so two requests within one clock tick stay distinct."""

        with self._nonce_lock:
            self._nonce = max(time.time_ns(), self._nonce + 1)
            return self._nonce

    def receiver(self) -> str:
        """The market's Gamma service hotkey, from its services document."""

        if not self._receiver:
            publishers = self.request("GET", "/tool-market/services.json").get("publishers") or [{}]
            self._receiver = str(publishers[0].get("hotkey") or "")
        return self._receiver

    def request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None, *,
                signed: bool = False, proof: bool = False) -> Dict[str, Any]:
        """The gateway's JSON answer; an error answer comes back as ``{"error": ..., "status": ...}``.

        ``signed``: btauth/1 by the hotkey. ``proof``: the lead-order scheme,
        which the terms need, since their signature is stored."""

        data = json.dumps(body).encode() if body is not None else b""
        headers = {"content-type": "application/json", "accept": "application/json"}
        if signed:
            receiver = self.receiver()
            if not receiver:
                return {"error": "services_document_unreadable"}
            headers.update(btauth.sign(self.keypair, method=method, target=path, body=data, receiver=receiver,
                                       now_ns=self._next_nonce))
        elif proof:
            timestamp = str(int(time.time()))
            message = signed_message(method, urlsplit(self.url).netloc, path, timestamp, data)
            headers.update({"x-hotkey": self.keypair.ss58_address, "x-timestamp": timestamp,
                            "x-signature": self.keypair.sign(message.encode()).hex()})
        elif self.api_key:
            headers["authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.url + path, data=data if body is not None else None,
                                         headers=headers, method=method)
        try:
            with _open(request, timeout=TIMEOUT_SECONDS) as answer:
                status, raw = answer.status, answer.read()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read()
        except TimeoutError:  # the answer is lost, not the request: a call may still be charged
            return {"error": "outcome_unknown", "status": 0,
                    "hint": "usage shows a call after 11 minutes; a retry before then can pay twice"}
        try:
            document = json.loads(raw or b"{}")
        except ValueError:
            document = None
        ok = isinstance(document, dict)
        if ok and 200 <= status < 300:
            return document
        return {"error": "http_%d" % status, **(document if ok else {}), "status": status}

    def tools(self) -> Dict[str, Any]:
        return self.request("GET", "/tool-market/catalog")

    def call(self, tool: str, operation: str, params: Dict[str, Any],
             max_ask_microusd: Optional[int] = None, private: Any = False) -> Dict[str, Any]:
        """On the API key when one is set, else signed by the hotkey."""

        body: Dict[str, Any] = {"tool": tool, "operation": operation, "params": params, "private": private}
        if max_ask_microusd is not None:
            body["max_ask_microusd"] = max_ask_microusd
        return self.request("POST", "/tool-market/call", body, signed=not self.api_key)


# -- MCP (stdio, JSON-RPC 2.0, one message per line) ----------------------------

MCP_TOOLS = [
    {"name": "find_tool",
     "description": "List the GTM data tools you can call: each operation, what it does, its parameters "
                    "and the current price in micro-USD. Read this before call_tool.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "call_tool",
     "description": "Call one tool operation from find_tool. It is charged to the Gamma balance "
                    "(or the API key's budget, when one is set).",
     "inputSchema": {"type": "object", "required": ["tool", "operation"], "properties": {
         "tool": {"type": "string"}, "operation": {"type": "string"},
         "params": {"type": "object", "description": "The operation's parameters, as find_tool lists them."},
         "max_ask_microusd": {"type": "integer", "description": "Optional: skip keys asking more per unit."},
         "private": {"type": "boolean", "description": "Optional: only suppliers that keep the request private."}}}},
]


def mcp_answer(client: Client, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The reply to one MCP message, or ``None`` for a notification."""

    if "id" not in message:
        return None
    method, params = message.get("method"), message.get("params") or {}
    if method == "initialize":
        result: Any = {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                       "capabilities": {"tools": {}},
                       "serverInfo": {"name": "subnet71-tool-market", "version": "1"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": MCP_TOOLS}
    elif method == "tools/call":
        name, arguments = params.get("name"), params.get("arguments") or {}
        if name == "find_tool":
            answer = client.tools()
        elif name == "call_tool":
            answer = client.call(str(arguments.get("tool", "")), str(arguments.get("operation", "")),
                                 arguments.get("params") or {}, arguments.get("max_ask_microusd"),
                                 arguments.get("private", False))
        else:
            return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32602, "message": "unknown tool"}}
        result = {"content": [{"type": "text", "text": json.dumps(answer)}], "isError": "error" in answer or (isinstance(answer.get("status"), int) and answer["status"] >= 400)}
    else:
        return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "method not found"}}
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


def serve_mcp(client: Client, lines=sys.stdin, out=sys.stdout) -> None:
    for line in lines:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if not isinstance(message, dict):
            continue
        try:
            reply = mcp_answer(client, message)
        except Exception as error:  # a network failure fails this call, not the server
            reply = {"jsonrpc": "2.0", "id": message.get("id"), "result": {
                "content": [{"type": "text", "text": "gateway unreachable: %s" % error}], "isError": True}} \
                if "id" in message else None
        if reply is not None:
            out.write(json.dumps(reply) + "\n")
            out.flush()


# -- CLI ------------------------------------------------------------------------

def _keypair(args: argparse.Namespace) -> Any:
    if args.hotkey_uri:
        from bittensor_wallet import Keypair

        return Keypair.create_from_uri(args.hotkey_uri)
    from bittensor_wallet import Wallet

    return Wallet(name=args.wallet_name, hotkey=args.hotkey_name).hotkey


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="subnet71", description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=os.environ.get("SUBNET71_URL", ""))
    parser.add_argument("--wallet-name", default="default")
    parser.add_argument("--hotkey-name", default="default")
    parser.add_argument("--hotkey-uri", default="", help="for tests, such as //Alice")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("terms", help="accept the buyer terms (once; needed for an API key)")
    commands.add_parser("balance", help="your Gamma balance at the market, in USD")
    key = commands.add_parser("key", help="create an API key that spends up to --budget USD of your balance")
    key.add_argument("--budget", type=float, required=True, help="USD")
    key.add_argument("--days", type=float, default=30)
    commands.add_parser("key-status", help="the API key's budget, spend and expiry")
    commands.add_parser("key-close", help="close the API key; its unspent budget returns to your balance")
    commands.add_parser("tools", help="the catalog")
    call = commands.add_parser("call", help="call TOOL OPERATION name=value ...")
    call.add_argument("tool")
    call.add_argument("operation")
    call.add_argument("params", nargs="*")
    call.add_argument("--max-ask-microusd", type=int, help="maximum miner ask per unit before the fee")
    call.add_argument("--private", action="store_true", help="only suppliers that keep the request private")
    commands.add_parser("mcp", help="run the MCP server on stdio")
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("set SUBNET71_URL or --url")
    try:
        client = Client(args.url, os.environ.get("SUBNET71_API_KEY", ""))
    except ValueError as error:
        parser.error(str(error))

    # Calls sign with the hotkey unless an API key is set.
    if args.command in ("terms", "balance", "key") or (args.command in ("call", "mcp") and not client.api_key):
        client.keypair = _keypair(args)
    if args.command == "terms":
        version, sha256 = list(PUBLISHED_TERMS.items())[-1]
        answer = client.request("POST", "/tool-market/terms", {
            "role": "buyer", "terms_version": version, "terms_sha256": sha256}, proof=True)
        wanted = answer.get("terms_version")
        if answer.get("error") == "terms_outdated" and isinstance(wanted, str) and wanted in PUBLISHED_TERMS:
            # The gateway trades on another published version: sign that one.
            answer = client.request("POST", "/tool-market/terms", {
                "role": "buyer", "terms_version": wanted, "terms_sha256": PUBLISHED_TERMS[wanted]}, proof=True)
    elif args.command == "balance":
        answer = client.request("GET", "/tool-market/gamma/balance?hotkey=" + client.keypair.ss58_address)
    elif args.command == "key":
        answer = client.request("POST", "/tool-market/tabs", {
            "budget_microusd": round(args.budget * 1_000_000), "seconds": round(args.days * 86400)}, signed=True)
        if "token" in answer:
            if "error" in answer:  # the store did not confirm: an open in flight ends within 30 s
                print("# This key may still open. Keep it; after a minute, key-status shows it, "
                      "and key-close frees its budget.", file=sys.stderr)
            print("export SUBNET71_API_KEY=" + answer["token"], file=sys.stderr)
    elif args.command == "key-status":
        answer = client.request("GET", "/tool-market/tab")
    elif args.command == "key-close":
        answer = client.request("POST", "/tool-market/tab/close", {})
    elif args.command == "tools":
        answer = client.tools()
    elif args.command == "call":
        params: Dict[str, Any] = {}
        for item in args.params:
            name, sep, text = item.partition("=")
            if not sep:
                parser.error("parameters are name=value (text) or name:=value (JSON)")
            try:
                params[name.removesuffix(":")] = json.loads(text) if name.endswith(":") else text
            except ValueError:
                parser.error("%s is not JSON" % name)
        answer = client.call(args.tool, args.operation, params, max_ask_microusd=args.max_ask_microusd, private=args.private)
    else:
        serve_mcp(client)
        return 0
    print(json.dumps(answer, indent=2))
    return 1 if "error" in answer else 0


if __name__ == "__main__":
    sys.exit(main())
