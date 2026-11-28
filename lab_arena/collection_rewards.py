"""Fixed run quotes settle once in the next epoch. No balance carries forward."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import time
import urllib.request

from lab_arena import weight_state
from leadpoet_canonical.arena_weights import (
    COLLECTION_BASIS_SCHEMA,
    TOOL_MARKET_SCHEMA,
    validate_tool_market,
    validate_accepted_weight_state,
    verify_accepted_weight_state_signature,
)


def tao_usd(now: int) -> tuple[int, int]:
    """Read fresh price and provider timestamp. Never authorize from an old static price."""
    url = (
        "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd"
        "&include_last_updated_at=true&precision=full"
    )
    with urllib.request.urlopen(url, timeout=6) as response:
        data = response.read(4097)
    if len(data) > 4096:
        raise ValueError("oversized TAO/USD quote")
    quote = json.loads(data, parse_float=Decimal)["bittensor"]
    price, updated = quote["usd"], quote["last_updated_at"]
    if type(price) not in (int, Decimal) or not 0 < price < 10**8:
        raise ValueError("invalid TAO/USD price")
    if type(updated) is not int or not 0 <= now - updated <= 900:
        raise ValueError("stale TAO/USD price")
    return int(price * 10**6), updated


def _epoch_alpha_values(chain, block_hash):
    events = chain.client.get_events(block_hash)
    distributions = [
        event["event"]["attributes"]["emissions"]
        for event in events
        if event["event"]["module_id"] == "SubtensorModule"
        and event["event"]["event_id"] == "IncentiveAlphaEmittedToMiners"
        and event["event"]["attributes"]["netuid"] == chain.config.netuid
    ]
    if (
        len(distributions) != 1
        or not isinstance(distributions[0], list)
        or not distributions[0]
    ):
        raise ValueError("finalized miner emission is unavailable")
    if not all(type(x) is int and x >= 0 for x in distributions[0]):
        raise ValueError("invalid miner emission")
    alpha = sum(distributions[0])
    price = chain.client.runtime_call(
        api="SwapRuntimeApi",
        method="current_alpha_price",
        params=[chain.config.netuid],
        block_hash=block_hash,
    )
    price = getattr(price, "value", price)
    if type(price) is not int or price <= 0:
        raise ValueError("invalid finalized alpha/TAO price")
    return alpha, price


def emission_valuation(
    chain, start_block: int, *, now=time.time, price_source=tao_usd
) -> dict:
    """Value the exact finalized SN71 epoch boundary, including on delayed issuance."""
    from async_substrate_interface.errors import SubstrateRequestException

    block_hash = chain.finalized_block_hash(start_block)
    try:
        alpha, price = _epoch_alpha_values(chain, block_hash)
    except SubstrateRequestException as exc:
        if "state already discarded" not in str(exc).lower() or (chain.config.network_name, chain.config.netuid) != ("finney", 71):
            raise
        from dataclasses import replace
        from Leadpoet.utils.subnet_epoch import OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT
        from lab_arena import chain as chain_module

        config = replace(chain.config, endpoint=OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT)
        archive = chain_module.ArenaChain(config, chain_module.connect_substrate(config))
        try:
            if archive.finalized_block_hash(start_block) != block_hash:
                raise ValueError("archive finalized block differs from primary")
            alpha, price = _epoch_alpha_values(archive, block_hash)
        finally:
            archive.close()
    observed = int(now())
    usd, updated = price_source(observed)
    return {
        "block": start_block,
        "block_hash": block_hash,
        "miner_alpha_rao": alpha,
        "tao_rao_per_alpha": price,
        "tao_price_microusd": usd,
        "price_updated_at": updated,
        "observed_at": observed,
        "emission_microusd": alpha * price * usd // 10**18,
    }


def existing_collection_state(row, *, epoch, genesis_hash, signer):
    state = validate_accepted_weight_state(row["state_doc"])
    verify_accepted_weight_state_signature(
        state,
        public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    )
    if (
        state["network"],
        state["netuid"],
        state["epoch"],
        state["genesis_hash"],
        state["reward_basis"]["schema_version"],
    ) != ("finney", 71, epoch, genesis_hash, COLLECTION_BASIS_SCHEMA) or row[
        "state_hash"
    ] != state[
        "state_hash"
    ]:
        raise ValueError("accepted collection state differs from requested scope")
    return state


class EpochNotCurrent(ValueError):
    """A collection state was asked for an epoch the chain is no longer settling.

    This is the ordinary epoch-boundary race a validator hits once per tempo, not
    a fault: the legacy reward path answers it with
    ``accepted_weight_epoch_not_current`` (409), and the collection path must too.
    """


def _supplier_targets(market_store, chain, scope, basis, burn_hotkey, valuation_source):
    """Use the same finalized epoch valuation after reserving every collection quote."""
    market = market_store()
    epoch = scope["epoch"]
    if not market.has_unallocated(epoch=epoch):
        return None
    valuation = basis["valuation"] or valuation_source(scope["valid_from_block"])
    emission = valuation["emission_microusd"]
    residual = emission - sum(basis["quotes_microusd"].values())
    if residual <= 0:
        return None
    graph = chain.metagraph(finalized=True)
    if graph.netuid != 71 or not scope["valid_from_block"] <= graph.block_number <= scope["valid_until_block"]:
        raise ValueError("supplier eligibility snapshot outside finalized epoch")
    hotkeys = [key for key in graph.hotkeys if key != burn_hotkey]
    targets = market.epoch_allocations(epoch=epoch, emission_microusd=emission,
                                       cap_microusd=residual, eligible_hotkeys=hotkeys)
    if not targets:
        return None
    if not set(targets).issubset(hotkeys):
        raise ValueError("unregistered supplier in tool market targets")
    doc = dict(schema_version=TOOL_MARKET_SCHEMA, epoch=epoch, emission_microusd=emission,
               allocations_microusd=targets)
    if not basis["job_count"]:
        doc["valuation"] = valuation
    validate_tool_market(doc, basis, epoch=epoch, start_block=scope["valid_from_block"], burn=burn_hotkey)
    return doc


def epoch_state(store, chain, signer, *, epoch, burn_hotkey, valuation_source, market_store=None):
    """Read the accepted state, or freeze completed work and issue its first signed state."""
    scope = chain.accepted_weight_epoch_scope()
    genesis = scope["genesis_hash"].removeprefix("0x").lower()
    existing = store.get_weight_state("finney", 71, epoch)
    if existing:
        return existing_collection_state(
            existing, epoch=epoch, genesis_hash=genesis, signer=signer
        )
    if scope["epoch"] != epoch:
        raise EpochNotCurrent("cannot issue a collection state outside its epoch")
    basis = store.collection_reward_quotes(epoch)
    if basis["job_count"]:
        basis["valuation"] = valuation_source(scope["valid_from_block"])
    else:
        basis["valuation"] = None
    # The DB serializes publication with acceptance. A raced snapshot is retried
    # using the immutable first winner, never by creating a second obligation.
    overlay = None
    if market_store is not None:
        try:
            overlay = _supplier_targets(market_store, chain, scope, basis, burn_hotkey, valuation_source)
        except Exception as exc:
            # The lab can still sign its normal state. Unscheduled debt stays owed.
            logging.getLogger(__name__).warning("Tool market allocation deferred for epoch %s (%s)",
                                                epoch, type(exc).__name__)

    def build(targets):
        return weight_state.build_accepted_weight_state(
            signer, network="finney", genesis_hash=genesis, netuid=71, epoch=epoch,
            valid_from_block=scope["valid_from_block"], valid_until_block=scope["valid_until_block"],
            reward_basis=basis, burn_hotkey=burn_hotkey, tool_market=targets,
            issued_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    from lab_arena.store import ArenaStoreError

    state = build(overlay)
    try:
        stored = store.publish_weight_state("finney", 71, epoch, state["state_hash"], state)
    except ArenaStoreError:
        previous = store.get_weight_state("finney", 71, epoch)
        if previous:
            return existing_collection_state(previous, epoch=epoch, genesis_hash=genesis, signer=signer)
        if overlay is None:
            raise
        # A raced supplier settlement can invalidate the quoted market debt.
        # No state exists. The ordinary published-once DB function rechecks the
        # collection rows, and pays no supplier twice even if a writer raced us.
        logging.getLogger(__name__).warning("Publishing collection-only state after supplier snapshot conflict")
        state = build(None)
        stored = store.publish_weight_state("finney", 71, epoch, state["state_hash"], state)
    return existing_collection_state(
        {"state_doc": stored["state"], "state_hash": stored["state_hash"]},
        epoch=epoch,
        genesis_hash=genesis,
        signer=signer,
    )
