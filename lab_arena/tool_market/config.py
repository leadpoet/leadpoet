"""Tool-market configuration from the environment. Off unless set otherwise.

When live, every catalog tool trades. ``LEADPOET_TOOL_MARKET_TOOLS_OFF``
turns single tools off, e.g. ``exa,hunter``; a new catalog tool is on
unless it is named there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import FrozenSet, Mapping, Optional
from urllib.parse import urlsplit

from lab_arena.lead_orders.auth import normalize_host
from lab_arena.lead_orders.config import is_valid_hotkey
from lab_arena.lead_orders.config import GAMMA_SERVICE_ENV as LEAD_ORDERS_GAMMA_SERVICE_ENV
from lab_arena.tool_market.catalog import TOOLS

MODE_ENV = "LEADPOET_TOOL_MARKET_MODE"
TOOLS_OFF_ENV = "LEADPOET_TOOL_MARKET_TOOLS_OFF"
BUYERS_ENV = "LEADPOET_TOOL_MARKET_BUYER_HOTKEYS"
PRIVATE_SUPPLIERS_ENV = "LEADPOET_TOOL_MARKET_PRIVATE_SUPPLIERS"
GAMMA_SERVICE_ENV = "LEADPOET_TOOL_MARKET_GAMMA_SERVICE"
TERMS_VERSION_ENV = "LEADPOET_TOOL_MARKET_TERMS_VERSION"
TERMS_SHA256_ENV = "LEADPOET_TOOL_MARKET_TERMS_SHA256"
PUBLIC_BASE_URL_ENV = "LEADPOET_TOOL_MARKET_PUBLIC_BASE_URL"

MODE_OFF = "off"
MODE_LIVE = "live"

MAX_TAB_SECONDS = 24 * 3600
MAX_TAB_MICROUSD = 1_000_000_000  # $1,000
# Low at launch: what one key may earn in 24 h. Raising it is a code change.
MAX_KEY_DAILY_CAP_MICROUSD = 20_000_000  # $20

# Each published terms version (docs/tool-market-terms.md) and the SHA-256 of
# that file's exact bytes. Live mode trades only on one of these.
PUBLISHED_TERMS: Mapping[str, str] = {
    "tool-market-terms-v1": "1c76cd8317b7fc68be0655ff24fd36a31ae1745e6b0c8e4ca878d60a67457988",
}


class ToolMarketConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolMarketConfig:
    mode: str = MODE_OFF
    tools: FrozenSet[str] = frozenset()
    # Empty: any hotkey may buy. Otherwise only these (a staff rollout).
    buyer_hotkeys: FrozenSet[str] = frozenset()
    # Miners who signed a private-supply agreement: they do not read buyers'
    # requests in their vendor accounts. Serves calls marked private.
    private_suppliers: FrozenSet[str] = frozenset()
    gamma_service: str = ""
    terms_version: str = ""
    terms_sha256: str = ""
    public_base_url: str = ""

    @property
    def enabled(self) -> bool:
        return self.mode == MODE_LIVE

    @property
    def public_host(self) -> str:
        return normalize_host(urlsplit(self.public_base_url).netloc)

    def may_buy(self, hotkey: str) -> bool:
        return self.enabled and (not self.buyer_hotkeys or hotkey in self.buyer_hotkeys)


def load_config(environ: Optional[Mapping[str, str]] = None) -> ToolMarketConfig:
    env = os.environ if environ is None else environ
    mode = (env.get(MODE_ENV) or MODE_OFF).strip().lower()
    if mode not in (MODE_OFF, MODE_LIVE):
        raise ToolMarketConfigError("%s must be off or live" % MODE_ENV)
    if mode == MODE_OFF:
        return ToolMarketConfig()
    off = frozenset(t.strip() for t in env.get(TOOLS_OFF_ENV, "").split(",") if t.strip())
    tools = frozenset(TOOLS) - off
    buyers = frozenset(h.strip() for h in env.get(BUYERS_ENV, "").split(",") if h.strip())
    private = frozenset(h.strip() for h in env.get(PRIVATE_SUPPLIERS_ENV, "").split(",") if h.strip())
    config = ToolMarketConfig(
        mode=mode, tools=tools, buyer_hotkeys=buyers, private_suppliers=private,
        gamma_service=env.get(GAMMA_SERVICE_ENV, "").strip(),
        terms_version=env.get(TERMS_VERSION_ENV, "").strip(),
        terms_sha256=env.get(TERMS_SHA256_ENV, "").strip().lower(),
        public_base_url=env.get(PUBLIC_BASE_URL_ENV, "").strip().rstrip("/"),
    )
    if not off <= set(TOOLS):
        raise ToolMarketConfigError("%s names a tool the catalog does not have" % TOOLS_OFF_ENV)
    for name, hotkeys in ((BUYERS_ENV, buyers), (PRIVATE_SUPPLIERS_ENV, private)):
        if not all(is_valid_hotkey(h) for h in hotkeys):
            raise ToolMarketConfigError("%s has an invalid hotkey" % name)
    if not is_valid_hotkey(config.gamma_service):
        raise ToolMarketConfigError("%s must be the tool market's Gamma service SS58 hotkey" % GAMMA_SERVICE_ENV)
    # One Gamma deposit must not fund two products.
    if config.gamma_service == env.get(LEAD_ORDERS_GAMMA_SERVICE_ENV, "").strip():
        raise ToolMarketConfigError("%s must differ from %s" % (GAMMA_SERVICE_ENV, LEAD_ORDERS_GAMMA_SERVICE_ENV))
    if PUBLISHED_TERMS.get(config.terms_version) != config.terms_sha256:
        raise ToolMarketConfigError("%s and %s must name published terms" % (TERMS_VERSION_ENV, TERMS_SHA256_ENV))
    if not config.public_base_url.startswith("https://"):
        raise ToolMarketConfigError("%s must be an https URL" % PUBLIC_BASE_URL_ENV)
    return config


__all__ = [
    "MAX_KEY_DAILY_CAP_MICROUSD",
    "MAX_TAB_MICROUSD",
    "MAX_TAB_SECONDS",
    "ToolMarketConfig",
    "ToolMarketConfigError",
    "load_config",
]
