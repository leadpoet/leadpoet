"""Buyer client for the Subnet 71 tool market: a CLI and a small MCP server.

The hotkey signs three things once: the terms, a balance read and an API
key (a tab). Everything after that uses only the API key, which is capped
by its budget and expires, so an agent never holds the hotkey.

    python -m lab_arena.tool_market.client terms
    python -m lab_arena.tool_market.client balance
    python -m lab_arena.tool_market.client key --budget 20 --days 30
    export SUBNET71_API_KEY=...
    python -m lab_arena.tool_market.client tools
    python -m lab_arena.tool_market.client call hunter email_finder domain=acme.com full_name="Ada Lee"
    python -m lab_arena.tool_market.client mcp        # stdio MCP server for agents

``SUBNET71_URL`` (or ``--url``) is the gateway's public origin, with no path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from lab_arena.lead_orders.auth import signed_message
from lab_arena.tool_market.config import PUBLISHED_TERMS

TIMEOUT_SECONDS = 120


class Client:
    def __init__(self, url: str, api_key: str = "", keypair: Any = None) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.keypair = keypair

    def request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None, *,
                signed: bool = False) -> Dict[str, Any]:
        """The gateway's JSON answer; an error answer comes back as ``{"error": ..., "status": ...}``."""

        data = json.dumps(body).encode() if body is not None else b""
        headers = {"content-type": "application/json", "accept": "application/json"}
        if signed:
            timestamp = str(int(time.time()))
            message = signed_message(method, urlsplit(self.url).netloc, path, timestamp, data)
            headers.update({"x-hotkey": self.keypair.ss58_address, "x-timestamp": timestamp,
                            "x-signature": self.keypair.sign(message.encode()).hex()})
        elif self.api_key:
            headers["authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.url + path, data=data if body is not None else None,
                                         headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as answer:
                return json.loads(answer.read() or b"{}")
        except urllib.error.HTTPError as error:
            try:
                document = json.loads(error.read() or b"{}")
            except ValueError:
                document = {}
            return dict({"error": "http_%d" % error.code}, **(document if isinstance(document, dict) else {}),
                        status=error.code)

    def tools(self) -> Dict[str, Any]:
        return self.request("GET", "/tool-market/catalog")

    def call(self, tool: str, operation: str, params: Dict[str, Any],
             max_ask_microusd: Optional[int] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"tool": tool, "operation": operation, "params": params}
        if max_ask_microusd is not None:
            body["max_ask_microusd"] = max_ask_microusd
        return self.request("POST", "/tool-market/call", body)


# -- MCP (stdio, JSON-RPC 2.0, one message per line) ----------------------------

MCP_TOOLS = [
    {"name": "find_tool",
     "description": "List the GTM data tools you can call: each operation, what it does, its parameters "
                    "and the current price in micro-USD. Read this before call_tool.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "call_tool",
     "description": "Call one tool operation from find_tool. It is charged to the API key's budget.",
     "inputSchema": {"type": "object", "required": ["tool", "operation"], "properties": {
         "tool": {"type": "string"}, "operation": {"type": "string"},
         "params": {"type": "object", "description": "The operation's parameters, as find_tool lists them."},
         "max_ask_microusd": {"type": "integer", "description": "Optional: skip keys asking more per unit."}}}},
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
                                 arguments.get("params") or {}, arguments.get("max_ask_microusd"))
        else:
            return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32602, "message": "unknown tool"}}
        result = {"content": [{"type": "text", "text": json.dumps(answer)}], "isError": "error" in answer}
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


def _value(text: str) -> Any:
    try:
        return json.loads(text)  # numbers, booleans and lists; anything else stays text
    except ValueError:
        return text


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="subnet71", description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=os.environ.get("SUBNET71_URL", ""))
    parser.add_argument("--wallet-name", default="default")
    parser.add_argument("--hotkey-name", default="default")
    parser.add_argument("--hotkey-uri", default="", help="for tests, such as //Alice")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("terms", help="accept the buyer terms (once)")
    commands.add_parser("balance", help="your Gamma balance: what you spent to the market, less what is held")
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
    commands.add_parser("mcp", help="run the MCP server on stdio")
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("set SUBNET71_URL or --url")
    client = Client(args.url, os.environ.get("SUBNET71_API_KEY", ""))

    if args.command in ("terms", "balance", "key"):
        client.keypair = _keypair(args)
    if args.command == "terms":
        version, sha256 = list(PUBLISHED_TERMS.items())[-1]
        answer = client.request("POST", "/tool-market/terms", {
            "role": "buyer", "terms_version": version, "terms_sha256": sha256}, signed=True)
    elif args.command == "balance":
        answer = client.request("GET", "/tool-market/balance", signed=True)
    elif args.command == "key":
        answer = client.request("POST", "/tool-market/tabs", {
            "budget_microusd": round(args.budget * 1_000_000), "seconds": round(args.days * 86400)}, signed=True)
        if "token" in answer:
            print("export SUBNET71_API_KEY=" + answer["token"], file=sys.stderr)
    elif args.command == "key-status":
        answer = client.request("GET", "/tool-market/tab")
    elif args.command == "key-close":
        answer = client.request("POST", "/tool-market/tab/close", {})
    elif args.command == "tools":
        answer = client.tools()
    elif args.command == "call":
        if not all("=" in item for item in args.params):
            parser.error("parameters are name=value")
        params = dict(item.split("=", 1) for item in args.params)
        answer = client.call(args.tool, args.operation, {k: _value(v) for k, v in params.items()})
    else:
        serve_mcp(client)
        return 0
    print(json.dumps(answer, indent=2))
    return 1 if "error" in answer else 0


if __name__ == "__main__":
    sys.exit(main())
