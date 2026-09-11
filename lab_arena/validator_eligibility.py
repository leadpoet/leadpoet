"""Arena validator identity and minimum stake for new benchmark leases.

All inputs come from one finalized Arena metagraph snapshot. Runner lists and
the gateway's unrelated active/500K role policy do not grant benchmark access.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


MIN_BENCHMARK_STAKE_WEIGHT = 75_000


class ValidatorIneligible(ValueError):
    """Valid chain data proves that this hotkey cannot perform the operation."""


def validator_uid(
    snapshot: Any, hotkey: str, *, netuid: int, allow_active_testnet: bool = False
) -> int:
    """Require registration and a permit, without imposing a stake minimum.

    The active testnet exception is only for identity checks on existing work.
    New benchmark leases never use that exception.
    """

    if isinstance(snapshot.netuid, bool) or snapshot.netuid != netuid:
        raise ValueError("metagraph subnet differs")
    hotkeys = tuple(snapshot.hotkeys)
    if len(set(hotkeys)) != len(hotkeys):
        raise ValueError("metagraph hotkeys are not unique")
    if hotkey not in hotkeys:
        raise ValidatorIneligible("runner_hotkey_unregistered")
    uid = hotkeys.index(hotkey)
    permits = tuple(snapshot.validator_permit)
    if len(permits) != len(hotkeys) or any(type(permit) is not bool for permit in permits):
        raise ValueError("metagraph validator permits are invalid")
    if permits[uid]:
        return uid
    if allow_active_testnet:
        active = tuple(snapshot.active)
        if len(active) != len(hotkeys) or any(type(item) is not bool for item in active):
            raise ValueError("metagraph activity is invalid")
        if active[uid]:
            return uid
    raise ValidatorIneligible("runner_validator_required")


def benchmark_validator_uid(snapshot: Any, hotkey: str, *, netuid: int) -> int:
    """Require a permitted validator with at least 75,000 effective stake.

    ``stake`` is decoded chain SubnetState.total_stake (Bittensor Metagraph.S),
    not a wallet balance or a separately calculated alpha/TAO approximation.
    The returned UID also binds the caller's coldkey exclusions to this snapshot.
    """

    uid = validator_uid(snapshot, hotkey, netuid=netuid)
    stakes = tuple(snapshot.stake)
    owners = tuple(snapshot.coldkeys)
    if len(stakes) != len(snapshot.hotkeys) or len(owners) != len(stakes):
        raise ValueError("metagraph stake/ownership lengths differ")
    if any(not isinstance(owner, str) or not owner for owner in owners):
        raise ValueError("metagraph coldkey is invalid")
    stake = stakes[uid]
    if isinstance(stake, bool) or not isinstance(stake, Real) or not math.isfinite(stake) or stake < 0:
        raise ValueError("metagraph stake is invalid")
    if stake < MIN_BENCHMARK_STAKE_WEIGHT:
        raise ValidatorIneligible("runner_stake_below_minimum")
    return uid
