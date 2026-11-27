"""Build the tool market from the environment (only when ``live``).

Reuses the Arena's production pieces and their environment names: the
PostgREST store credential (``LAB_ARENA_SUPABASE_URL``,
``LAB_ARENA_SERVICE_KEY``) and the chain endpoint. Keys go through the key
vault (``lead_order_vault/deployment.json``, ``LEAD_ORDER_VAULT_CALLER_TOKEN``).
"""

from __future__ import annotations

import os
from typing import Mapping

from Leadpoet.utils.subnet_epoch import SubnetEpochError
from lab_arena import chain as chain_module
from lab_arena.lead_orders.gamma import GammaBalanceReader
from lab_arena.lead_orders.vault_client import VaultClient, VaultTransport
from lab_arena.store import PostgrestTransport
from lab_arena.tool_market.config import ToolMarketConfig, ToolMarketConfigError
from lab_arena.tool_market.market import ToolMarket
from lab_arena.tool_market.store import ToolMarketStore
from lead_order_vault import trust as vault_trust


def _required(name: str, env: Mapping[str, str]) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise ToolMarketConfigError("environment %s is required" % name)
    return value


class MarketGammaReader(GammaBalanceReader):
    """Bind credit reads and purchases to the same finalized subnet incarnation."""

    def __init__(self, chain, service, cutover, store):
        super().__init__(chain, service_id=service)
        self.cutover = cutover
        head = chain.finalized_head()
        self.genesis = chain_module.normalize_block_hash(cutover.network_genesis_hash)
        if (chain.config.netuid != 71 or chain.config.netuid != cutover.netuid
                or chain.finalized_block_hash(0, head=head) != self.genesis):
            raise chain_module.ArenaChainError("Gamma network differs from SN71 cutover")
        self.registered_at = self._registration(head)
        store.bind_gamma_source(genesis=self.genesis, netuid=chain.config.netuid,
                                service=service, registered_at=self.registered_at)

    def _registration(self, head):
        try:
            observed = self._chain.client.query(
                "SubtensorModule", "NetworkRegisteredAt", [self._chain.config.netuid], block_hash=head.hash)
        except Exception as exc:
            raise chain_module.ArenaChainError("Gamma subnet registration unavailable") from exc
        value = getattr(observed, "value", observed)
        # A missing ValueQuery entry can decode as zero.
        if type(value) is not int or not 0 < value <= head.number:
            raise chain_module.ArenaChainError("Gamma subnet registration unavailable")
        return value

    def read(self, hotkey, *, head=None):
        head = head if head is not None else self._chain.finalized_head()
        if (self._chain.finalized_block_hash(0, head=head) != self.genesis
                or self._registration(head) != self.registered_at):
            raise chain_module.ArenaChainError("Gamma credit source changed")
        return super().read(hotkey, head=head)

    def current_epoch(self):
        snapshot = chain_module.finalized_epoch_snapshot(self._chain)
        # settlement_epoch_id checks the genesis and netuid against the manifest.
        try:
            epoch = snapshot.settlement_epoch_id(self.cutover)
        except SubnetEpochError as exc:
            raise chain_module.ArenaChainError("Gamma settlement epoch unavailable") from exc
        if self._registration(chain_module.BlockRef(snapshot.current_block, snapshot.block_hash)) != self.registered_at:
            raise chain_module.ArenaChainError("Gamma credit source changed")
        return epoch


def build_from_environment(config: ToolMarketConfig):
    """The live market for ``config``, which ``gateway_support`` loads and checks."""

    env = os.environ
    store = ToolMarketStore(PostgrestTransport(
        _required("LAB_ARENA_SUPABASE_URL", env), service_key=_required("LAB_ARENA_SERVICE_KEY", env),
    ))
    chain_config = chain_module.ArenaChainConfig(
        endpoint=_required("LAB_ARENA_CHAIN_ENDPOINT", env), netuid=config.netuid,
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
    gamma = MarketGammaReader(arena_chain, config.gamma_service, cutover, store)
    return ToolMarket(
        config=config, store=store, vault=vault, transport=VaultTransport(vault),
        gamma_reader=gamma, current_epoch=gamma.current_epoch,
        is_registered=lambda hotkey: chain_module.is_registered(arena_chain.metagraph(), hotkey),
    )


__all__ = ["build_from_environment"]
