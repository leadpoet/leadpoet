"""One entry per tool: where its key goes, what it may call, how it is checked.

This table is the vault's own allowlist. The vault is a separately attested
image, so it does not trust the gateway's catalog: a request reaches a host
only through the entry for that host, and only on that entry's routes.

Adding a tool is one ``Tool`` here (and its operations in the gateway's
catalog, ``lab_arena/tool_market/catalog.py``). The envelope's provider and
host lists are derived from this table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple
from urllib.parse import quote

# Request shapes a miner's key may take. Anything else, such as billing
# top-ups, plays, monitors or key management, is refused.
OPENROUTER_ROUTES = frozenset({
    ("POST", "/api/v1/chat/completions"),
    ("POST", "/api/alpha/decisions"),
    ("GET", "/api/v1/generation"),  # free cost read-back by generation id
})
# The broker URL-encodes the id, which may hold "." and ":" (as %3A).
_GENERATION_QUERY = re.compile(r"id=[A-Za-z0-9._~%-]{1,600}")
# A URL-encoded query string: names and values the caller encoded itself.
_PLAIN_QUERY = re.compile(r"(?:[A-Za-z0-9_]{1,32}=[A-Za-z0-9._~%+-]{0,2000})(?:&[A-Za-z0-9_]{1,32}=[A-Za-z0-9._~%+-]{0,2000}){0,15}")

Route = Callable[[str, str, str], bool]


def _openrouter_route(method: str, path: str, query: str) -> bool:
    if (method, path) not in OPENROUTER_ROUTES:
        return False
    return _GENERATION_QUERY.fullmatch(query) is not None if method == "GET" else query == ""


def shape(method: str, pattern: str, *, query: bool) -> Route:
    """A route rule: one method, a path pattern, and a plain query or none."""

    compiled = re.compile(pattern)

    def allowed(asked: str, path: str, text: str) -> bool:
        return (asked == method and compiled.fullmatch(path) is not None
                and (_PLAIN_QUERY.fullmatch(text) is not None if query and text else text == ""))
    return allowed


@dataclass(frozen=True)
class Tool:
    """One vendor: its host, where its key goes, its routes, and its free check.

    The key goes in ``header`` after ``scheme`` (``"Bearer "`` or ``""``), or,
    when ``query_param`` is set, in that query parameter instead. A tool
    without ``balance_url`` has no free check: its key is opened and named,
    and its first calls prove it. ``balance_field`` is the path to the
    remaining balance in the check's answer.
    """

    host: str
    route: Route
    header: str = ""
    scheme: str = ""
    query_param: Optional[str] = None
    balance_url: Optional[str] = None
    balance_field: Tuple = ()
    # HTTP refusal codes that unambiguously reject this credential.
    invalid_statuses: Tuple[int, ...] = (401, 403, 404)

    def place(self, secret: str, headers: Mapping[str, str], target: str) -> Tuple[dict, str]:
        """``(headers, target)`` with the key where the vendor reads it."""

        if self.query_param is None:
            return dict(headers, **{self.header: self.scheme + secret}), target
        joiner = "&" if "?" in target else "?"
        return dict(headers), target + joiner + self.query_param + "=" + quote(secret, safe="")

    def caller_sets_key(self, query: str) -> bool:
        """The caller's own query already names the key parameter."""

        return self.query_param is not None and any(
            part.split("=", 1)[0].lower() == self.query_param.lower() for part in query.split("&") if part)


TOOLS: Mapping[str, Tool] = {
    # Lead orders.
    "openrouter": Tool("openrouter.ai", _openrouter_route, "authorization", "Bearer "),
    "deepline": Tool("code.deepline.com", shape("POST", r"/api/v2/integrations/[a-z0-9_]{1,64}/execute", query=False),
                     "authorization", "Bearer ",
                     balance_url="https://code.deepline.com/api/v2/billing/balance"),
    # Tool market (docs/gamma-tool-market.md): data endpoints only, never
    # account, billing or key management.
    # Its credits come as text, and "-1" for a key it does not know.
    "zerobounce": Tool("api.zerobounce.net", shape("GET", r"/v2/validate", query=True), query_param="api_key",
                       balance_url="https://api.zerobounce.net/v2/getcredits", balance_field=("Credits",)),
    "prospeo": Tool("api.prospeo.io", shape("POST", r"/enrich-person", query=False), "x-key",
                    balance_url="https://api.prospeo.io/account-information",
                    balance_field=("response", "remaining_credits")),
    "findymail": Tool("app.findymail.com", shape("POST", r"/api/search/(?:business-profile|name)", query=False),
                      "authorization", "Bearer ", balance_url="https://app.findymail.com/api/credits",
                      balance_field=("credits",)),
    "hunter": Tool("api.hunter.io", shape("GET", r"/v2/email-finder", query=True), "x-api-key",
                   # Its account answer gives a plan's searches, not what is left, so
                   # the check proves the key only.
                   balance_url="https://api.hunter.io/v2/account", invalid_statuses=(401,)),
    "harvestapi": Tool("api.harvestapi.io", shape("GET", r"/linkedin/(?:profile|company)", query=True), "x-api-key"),
    "exa": Tool("api.exa.ai", shape("POST", r"/(?:search|contents)", query=False), "x-api-key"),
}
HOSTS: Mapping[str, str] = {tool.host: name for name, tool in TOOLS.items()}
assert len(HOSTS) == len(TOOLS)

__all__ = ["HOSTS", "TOOLS", "Tool"]
