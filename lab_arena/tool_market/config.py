"""Tool-market configuration from the environment. Off unless set otherwise.

``LEADPOET_TOOL_MARKET_TOOLS`` lists the tools that may trade. It is empty
by default: a tool is listed only once its vendor allows its keys to be
resold this way (docs/gamma-tool-market.md).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import FrozenSet, Mapping, Optional
from urllib.parse import urlsplit

from lab_arena.lead_orders.auth import normalize_host
from lab_arena.lead_orders.config import DEFAULT_GAMMA_PALLET, DEFAULT_GAMMA_STORAGE, is_valid_hotkey
from lab_arena.tool_market.catalog import TOOLS

MODE_ENV = "LEADPOET_TOOL_MARKET_MODE"
TOOLS_ENV = "LEADPOET_TOOL_MARKET_TOOLS"
BUYERS_ENV = "LEADPOET_TOOL_MARKET_BUYER_HOTKEYS"
GAMMA_SERVICE_ENV = "LEADPOET_TOOL_MARKET_GAMMA_SERVICE"
TERMS_VERSION_ENV = "LEADPOET_TOOL_MARKET_TERMS_VERSION"
TERMS_SHA256_ENV = "LEADPOET_TOOL_MARKET_TERMS_SHA256"
PUBLIC_BASE_URL_ENV = "LEADPOET_TOOL_MARKET_PUBLIC_BASE_URL"

MODE_OFF = "off"
MODE_LIVE = "live"

# A key with a free balance check is re-checked before use once its last
# check is older than this.
CHECK_MAX_AGE_SECONDS = 6 * 3600
MAX_TAB_SECONDS = 24 * 3600
MAX_TAB_MICROUSD = 1_000_000_000  # $1,000
MAX_KEY_DAILY_CAP_MICROUSD = 1_000_000_000

_TERMS_RE = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ToolMarketConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolMarketConfig:
    mode: str = MODE_OFF
    tools: FrozenSet[str] = frozenset()
    # Empty: any hotkey may buy. Otherwise only these (a staff rollout).
    buyer_hotkeys: FrozenSet[str] = frozenset()
    gamma_service: str = ""
    gamma_pallet: str = DEFAULT_GAMMA_PALLET
    gamma_storage: str = DEFAULT_GAMMA_STORAGE
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
    tools = frozenset(t.strip() for t in env.get(TOOLS_ENV, "").split(",") if t.strip())
    buyers = frozenset(h.strip() for h in env.get(BUYERS_ENV, "").split(",") if h.strip())
    config = ToolMarketConfig(
        mode=mode, tools=tools, buyer_hotkeys=buyers,
        gamma_service=env.get(GAMMA_SERVICE_ENV, "").strip(),
        terms_version=env.get(TERMS_VERSION_ENV, "").strip(),
        terms_sha256=env.get(TERMS_SHA256_ENV, "").strip().lower(),
        public_base_url=env.get(PUBLIC_BASE_URL_ENV, "").strip().rstrip("/"),
    )
    if not tools <= set(TOOLS):
        raise ToolMarketConfigError("%s names a tool the catalog does not have" % TOOLS_ENV)
    if not all(is_valid_hotkey(h) for h in buyers):
        raise ToolMarketConfigError("%s has an invalid hotkey" % BUYERS_ENV)
    if not is_valid_hotkey(config.gamma_service):
        raise ToolMarketConfigError("%s must be the tool market's Gamma service SS58 hotkey" % GAMMA_SERVICE_ENV)
    if not _TERMS_RE.fullmatch(config.terms_version) or not _SHA_RE.fullmatch(config.terms_sha256):
        raise ToolMarketConfigError("%s and %s are required" % (TERMS_VERSION_ENV, TERMS_SHA256_ENV))
    if not config.public_base_url.startswith("https://"):
        raise ToolMarketConfigError("%s must be an https URL" % PUBLIC_BASE_URL_ENV)
    return config


__all__ = [
    "CHECK_MAX_AGE_SECONDS",
    "MAX_KEY_DAILY_CAP_MICROUSD",
    "MAX_TAB_MICROUSD",
    "MAX_TAB_SECONDS",
    "ToolMarketConfig",
    "ToolMarketConfigError",
    "load_config",
]
