"""Build the tool market from the environment (only when ``live``).

Reuses the Arena's production pieces and their environment names: the
PostgREST store credential (``LAB_ARENA_SUPABASE_URL``,
``LAB_ARENA_SERVICE_KEY``) and the chain endpoint. Keys go through the key
vault (``lead_order_vault/deployment.json``, ``LEAD_ORDER_VAULT_CALLER_TOKEN``).
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

from lab_arena import chain as chain_module
from lab_arena.lead_orders.gamma import GammaBalanceReader
from lab_arena.lead_orders.vault_client import VaultClient, VaultTransport
from lab_arena.store import PostgrestTransport
from lab_arena.tool_market.config import ToolMarketConfigError, load_config
from lab_arena.tool_market.market import ToolMarket
from lab_arena.tool_market.store import ToolMarketStore
from lead_order_vault import trust as vault_trust


def _required(name: str, env: Mapping[str, str]) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise ToolMarketConfigError("environment %s is required" % name)
    return value


def build_from_environment(environ: Optional[Mapping[str, str]] = None):
    """Return ``(config, market)`` for live mode (``gateway_support`` checks the mode)."""

    env = os.environ if environ is None else environ
    config = load_config(env)
    store = ToolMarketStore(PostgrestTransport(
        _required("LAB_ARENA_SUPABASE_URL", env), service_key=_required("LAB_ARENA_SERVICE_KEY", env),
    ))
    chain_config = chain_module.ArenaChainConfig(
        endpoint=_required("LAB_ARENA_CHAIN_ENDPOINT", env), netuid=int(env.get("LAB_ARENA_NETUID", "71")),
        network_name=env.get("LAB_ARENA_NETWORK", "finney"),
        request_timeout_seconds=int(env.get("LAB_ARENA_CHAIN_TIMEOUT_SECONDS", "30")),
    )
    arena_chain = chain_module.ArenaChain(chain_config, chain_module.connect_substrate(chain_config))
    cutover = chain_module.load_arena_cutover()
    try:
        deployment = vault_trust.load_deployment()
    except vault_trust.TrustError as error:
        raise ToolMarketConfigError(str(error)) from None
    vault = VaultClient(deployment, _required("LEAD_ORDER_VAULT_CALLER_TOKEN", env))
    market = ToolMarket(
        config=config, store=store, vault=vault, transport=VaultTransport(vault),
        gamma_reader=GammaBalanceReader(arena_chain, service_id=config.gamma_service, pallet=config.gamma_pallet,
                                        storage=config.gamma_storage),
        current_epoch=lambda: chain_module.current_settlement_epoch(arena_chain, cutover),
    )
    return config, market


__all__ = ["build_from_environment"]
