"""Lead-order configuration from the environment. Off unless set otherwise.

No secret has a default. A mode other than ``off`` with missing required
settings refuses to start rather than running half-configured.
"""

from __future__ import annotations

import hmac
import os
import re
from dataclasses import dataclass, field
from typing import FrozenSet, Mapping, Optional

MODE_ENV = "LEADPOET_LEAD_ORDERS_MODE"
STAFF_HOTKEYS_ENV = "LEADPOET_LEAD_ORDERS_STAFF_HOTKEYS"
BLOCKED_HOTKEYS_ENV = "LEADPOET_LEAD_ORDERS_BLOCKED_HOTKEYS"
RUNNER_SECRET_ENV = "LEADPOET_LEAD_ORDERS_RUNNER_SECRET"
POLICY_PATH_ENV = "LEADPOET_LEAD_ORDERS_POLICY_PATH"
GAMMA_SERVICE_ENV = "LEADPOET_LEAD_ORDERS_GAMMA_SERVICE"
GAMMA_PALLET_ENV = "LEADPOET_LEAD_ORDERS_GAMMA_PALLET"
GAMMA_STORAGE_ENV = "LEADPOET_LEAD_ORDERS_GAMMA_STORAGE"
TERMS_VERSION_ENV = "LEADPOET_LEAD_ORDERS_TERMS_VERSION"
PUBLIC_BASE_URL_ENV = "LEADPOET_LEAD_ORDERS_PUBLIC_BASE_URL"
LEADPOET_OPENROUTER_KEY_ENV = "LEADPOET_LEAD_ORDERS_OPENROUTER_API_KEY"
LEADPOET_DEEPLINE_KEY_ENV = "LEADPOET_LEAD_ORDERS_DEEPLINE_API_KEY"

MODE_OFF = "off"
MODE_STAFF = "staff"
MODE_SHADOW = "shadow"
MODE_LIVE = "live"
MODES = (MODE_OFF, MODE_STAFF, MODE_SHADOW, MODE_LIVE)

DEFAULT_GAMMA_PALLET = "SubtensorModule"
DEFAULT_GAMMA_STORAGE = "GammaSpent"
MIN_RUNNER_SECRET_BYTES = 32

_HOTKEY_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,48}$")
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SERVICE_RE = re.compile(r"^(?:[1-9A-HJ-NP-Za-km-z]{46,48}|[0-9]{1,10})$")
_TERMS_RE = re.compile(r"^[A-Za-z0-9._-]{1,32}$")


class LeadOrdersConfigError(RuntimeError):
    pass


def _hotkeys(raw: str, name: str) -> FrozenSet[str]:
    values = frozenset(item.strip() for item in raw.split(",") if item.strip())
    for value in values:
        if not _HOTKEY_RE.fullmatch(value):
            raise LeadOrdersConfigError("%s has an invalid hotkey" % name)
    return values


@dataclass(frozen=True)
class LeadOrdersConfig:
    mode: str = MODE_OFF
    staff_hotkeys: FrozenSet[str] = frozenset()
    blocked_hotkeys: FrozenSet[str] = frozenset()
    runner_secret: str = field(default="", repr=False)
    policy_path: str = ""
    gamma_service: str = ""
    gamma_pallet: str = DEFAULT_GAMMA_PALLET
    gamma_storage: str = DEFAULT_GAMMA_STORAGE
    terms_version: str = ""
    public_base_url: str = ""
    leadpoet_openrouter_key: str = field(default="", repr=False)
    leadpoet_deepline_key: str = field(default="", repr=False)

    @property
    def enabled(self) -> bool:
        return self.mode != MODE_OFF

    @property
    def uses_gamma(self) -> bool:
        return self.mode == MODE_LIVE

    @property
    def uses_miner_keys(self) -> bool:
        return self.mode in (MODE_SHADOW, MODE_LIVE)

    @property
    def computes_rewards(self) -> bool:
        return self.mode in (MODE_SHADOW, MODE_LIVE)

    @property
    def weights_rewards(self) -> bool:
        return self.mode == MODE_LIVE

    def may_order(self, hotkey: str) -> bool:
        """Mode gate only; live orders also need a confirmed buyer and balance."""

        if hotkey in self.blocked_hotkeys or not self.enabled:
            return False
        if self.mode in (MODE_STAFF, MODE_SHADOW):
            return hotkey in self.staff_hotkeys
        return True

    def runner_secret_matches(self, presented: Optional[str]) -> bool:
        if not self.runner_secret or not isinstance(presented, str):
            return False
        return hmac.compare_digest(presented.encode("utf-8"), self.runner_secret.encode("utf-8"))


def load_config(environ: Optional[Mapping[str, str]] = None) -> LeadOrdersConfig:
    env = os.environ if environ is None else environ
    mode = (env.get(MODE_ENV) or MODE_OFF).strip().lower()
    if mode not in MODES:
        raise LeadOrdersConfigError("%s must be one of %s" % (MODE_ENV, ", ".join(MODES)))
    if mode == MODE_OFF:
        return LeadOrdersConfig()
    config = LeadOrdersConfig(
        mode=mode,
        staff_hotkeys=_hotkeys(env.get(STAFF_HOTKEYS_ENV, ""), STAFF_HOTKEYS_ENV),
        blocked_hotkeys=_hotkeys(env.get(BLOCKED_HOTKEYS_ENV, ""), BLOCKED_HOTKEYS_ENV),
        runner_secret=env.get(RUNNER_SECRET_ENV, ""),
        policy_path=env.get(POLICY_PATH_ENV, "").strip(),
        gamma_service=env.get(GAMMA_SERVICE_ENV, "").strip(),
        gamma_pallet=(env.get(GAMMA_PALLET_ENV) or DEFAULT_GAMMA_PALLET).strip(),
        gamma_storage=(env.get(GAMMA_STORAGE_ENV) or DEFAULT_GAMMA_STORAGE).strip(),
        terms_version=env.get(TERMS_VERSION_ENV, "").strip(),
        public_base_url=env.get(PUBLIC_BASE_URL_ENV, "").strip().rstrip("/"),
        leadpoet_openrouter_key=env.get(LEADPOET_OPENROUTER_KEY_ENV, ""),
        leadpoet_deepline_key=env.get(LEADPOET_DEEPLINE_KEY_ENV, ""),
    )
    if len(config.runner_secret.encode("utf-8")) < MIN_RUNNER_SECRET_BYTES:
        raise LeadOrdersConfigError("%s must be at least %d bytes" % (RUNNER_SECRET_ENV, MIN_RUNNER_SECRET_BYTES))
    if not config.policy_path:
        raise LeadOrdersConfigError("%s is required" % POLICY_PATH_ENV)
    if not _TERMS_RE.fullmatch(config.terms_version):
        raise LeadOrdersConfigError("%s is required" % TERMS_VERSION_ENV)
    if not config.public_base_url.startswith("https://"):
        raise LeadOrdersConfigError("%s must be an https URL" % PUBLIC_BASE_URL_ENV)
    if not _NAME_RE.fullmatch(config.gamma_pallet) or not _NAME_RE.fullmatch(config.gamma_storage):
        raise LeadOrdersConfigError("Gamma pallet and storage names are invalid")
    if mode == MODE_LIVE and not _SERVICE_RE.fullmatch(config.gamma_service):
        # The key type of the service id is not settled (subtensor PR 3343):
        # an SS58 account or a numeric service id are both accepted.
        raise LeadOrdersConfigError("%s must be the Leadpoet Gamma service id in live mode" % GAMMA_SERVICE_ENV)
    if mode in (MODE_STAFF, MODE_SHADOW) and not config.staff_hotkeys:
        raise LeadOrdersConfigError("%s is required in %s mode" % (STAFF_HOTKEYS_ENV, mode))
    return config


__all__ = [
    "LeadOrdersConfig",
    "LeadOrdersConfigError",
    "MODES",
    "MODE_LIVE",
    "MODE_OFF",
    "MODE_SHADOW",
    "MODE_STAFF",
    "load_config",
]
