"""Fixed run quotes settle once in the next epoch. No balance carries forward."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import time
import urllib.request

from lab_arena import weight_state
from leadpoet_canonical.arena_weights import (
    COLLECTION_BASIS_SCHEMA,
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


def emission_valuation(
    chain, start_block: int, *, now=time.time, price_source=tao_usd
) -> dict:
    """Use the finalized epoch-boundary miner distribution and alpha spot price.

    This values the weight target. Final emitted balances depend on Subtensor consensus.
    """
    block_hash = chain.finalized_block_hash(start_block)
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


def epoch_state(store, chain, signer, *, epoch, burn_hotkey, valuation_source):
    """Read the accepted state, or freeze completed work and issue its first signed state."""
    scope = chain.accepted_weight_epoch_scope()
    genesis = scope["genesis_hash"].removeprefix("0x").lower()
    existing = store.get_weight_state("finney", 71, epoch)
    if existing:
        return existing_collection_state(
            existing, epoch=epoch, genesis_hash=genesis, signer=signer
        )
    if scope["epoch"] != epoch:
        raise ValueError("cannot issue a collection state outside its epoch")
    basis = store.collection_reward_quotes(epoch)
    if basis["job_count"]:
        basis["valuation"] = valuation_source(scope["valid_from_block"])
    else:
        basis["valuation"] = None
    # The DB serializes publication with acceptance. A raced snapshot is retried
    # using the immutable first winner, never by creating a second obligation.
    state = weight_state.build_accepted_weight_state(
        signer,
        network="finney",
        genesis_hash=genesis,
        netuid=71,
        epoch=epoch,
        valid_from_block=scope["valid_from_block"],
        valid_until_block=scope["valid_until_block"],
        reward_basis=basis,
        burn_hotkey=burn_hotkey,
        issued_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    from lab_arena.store import ArenaStoreError

    try:
        stored = store.publish_weight_state(
            "finney", 71, epoch, state["state_hash"], state
        )
    except ArenaStoreError:
        previous = store.get_weight_state("finney", 71, epoch)
        if not previous:
            raise
        return existing_collection_state(
            previous, epoch=epoch, genesis_hash=genesis, signer=signer
        )
    return existing_collection_state(
        {"state_doc": stored["state"], "state_hash": stored["state_hash"]},
        epoch=epoch,
        genesis_hash=genesis,
        signer=signer,
    )
