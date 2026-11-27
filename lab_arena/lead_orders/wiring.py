"""Build the lead-order service from the environment (only when not ``off``).

Reuses the Arena's production pieces and their environment names: the
PostgREST store credential (``LAB_ARENA_SUPABASE_URL``,
``LAB_ARENA_SERVICE_KEY``), the chain endpoint, the KMS signing key that
signs Arena documents. Miner keys go through the lead-order key vault
(``lead_order_vault/deployment.json``, ``LEAD_ORDER_VAULT_CALLER_TOKEN``). Nothing is
built, and no network is touched, while the mode is ``off``.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from lab_arena import chain as chain_module
from lab_arena import signing
from lab_arena.broker import HttpxProviderTransport, fetch_openrouter_price_table
from lab_arena.lead_orders import keys as lead_keys
from lab_arena.lead_orders.broker import LeadOrderBroker
from lab_arena.lead_orders.config import LeadOrdersConfig, load_config
from lab_arena.lead_orders.gamma import GammaBalanceReader
from lab_arena.lead_orders.policy import load_policy
from lab_arena.lead_orders.service import LeadOrderService
from lab_arena.lead_orders.store import LeadOrderStore
from lab_arena.lead_orders.vault_client import VaultClient, VaultTransport
from lead_order_vault import trust as vault_trust
from lab_arena.store import PostgrestTransport

PRICE_TABLE_TTL_SECONDS = 3600


class LeadOrdersUnavailable(RuntimeError):
    pass


def _required(name: str, env: Mapping[str, str]) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise LeadOrdersUnavailable("environment %s is required" % name)
    return value


class _CachedPriceTable:
    def __init__(self) -> None:
        import time

        self._time = time.monotonic
        self._lock = threading.Lock()
        self._table: Optional[Mapping[str, Any]] = None
        self._at = 0.0

    def __call__(self) -> Mapping[str, Any]:
        with self._lock:
            if self._table is None or self._time() - self._at > PRICE_TABLE_TTL_SECONDS:
                self._table = fetch_openrouter_price_table()
                self._at = self._time()
            return self._table


def build_from_environment(environ: Optional[Mapping[str, str]] = None):
    """Return ``(config, service, broker)``; ``(config, None, None)`` when off."""

    env = os.environ if environ is None else environ
    config: LeadOrdersConfig = load_config(env)
    if not config.enabled:
        return config, None, None
    signer = signing.KmsSigner(_required("LAB_ARENA_SIGNING_KEY_ID", env), region_name=env.get("AWS_REGION"))
    document = json.loads(Path(config.policy_path).read_text(encoding="utf-8"))
    policy = load_policy(document, public_key_der=signer.public_key_der, expected_public_key_hash=signer.public_key_hash)
    store = LeadOrderStore(PostgrestTransport(
        _required("LAB_ARENA_SUPABASE_URL", env), service_key=_required("LAB_ARENA_SERVICE_KEY", env),
    ))
    chain_config = chain_module.ArenaChainConfig(
        endpoint=_required("LAB_ARENA_CHAIN_ENDPOINT", env), netuid=int(env.get("LAB_ARENA_NETUID", "71")),
        network_name=env.get("LAB_ARENA_NETWORK", "finney"),
        request_timeout_seconds=int(env.get("LAB_ARENA_CHAIN_TIMEOUT_SECONDS", "30")),
    )
    arena_chain = chain_module.ArenaChain(chain_config, chain_module.connect_substrate(chain_config))
    cutover = chain_module.load_arena_cutover()
    current_epoch = lambda: chain_module.current_settlement_epoch(arena_chain, cutover)
    gamma_reader = None
    if config.uses_gamma:
        gamma_reader = GammaBalanceReader(
            arena_chain, service_id=config.gamma_service, pallet=config.gamma_pallet, storage=config.gamma_storage,
        )
    registrar = None
    miner_transport = None
    if config.uses_miner_keys:
        try:
            deployment = vault_trust.load_deployment()
        except vault_trust.TrustError as error:
            raise LeadOrdersUnavailable(str(error)) from None
        vault = VaultClient(deployment, _required("LEAD_ORDER_VAULT_CALLER_TOKEN", env))
        registrar = lead_keys.MinerKeyRegistrar(vault)
        miner_transport = VaultTransport(vault)
    broker = LeadOrderBroker(
        store=store, policy=policy, transport=HttpxProviderTransport(), registrar=registrar,
        miner_transport=miner_transport,
        owner_secrets={"openrouter": config.owner_openrouter_key, "deepline": config.owner_deepline_key},
        price_table=_CachedPriceTable(), current_epoch=current_epoch,
    )
    service = LeadOrderService(
        config=config, policy=policy, policy_document=document, store=store, signer=signer,
        current_epoch=current_epoch, gamma_reader=gamma_reader, registrar=registrar,
        resolver=broker.resolve_uncertain,
    )
    return config, service, broker


__all__ = ["LeadOrdersUnavailable", "build_from_environment"]
