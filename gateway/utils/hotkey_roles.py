"""Pure subnet hotkey role classification shared by gateway processes."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, List, Optional, Tuple


MIN_VALIDATOR_STAKE_WEIGHT = 75_000
# Backward-compatible name for registry imports and external callers.
STAKE_THRESHOLD = MIN_VALIDATOR_STAKE_WEIGHT


class ValidatorIneligible(ValueError):
    """Valid chain data proves that a hotkey is not an eligible validator."""


def _scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _snapshot_hotkeys(metagraph: Any) -> Tuple[str, ...]:
    try:
        hotkeys = tuple(metagraph.hotkeys)
    except (AttributeError, TypeError) as exc:
        raise ValueError("metagraph hotkeys are invalid") from exc
    if any(not isinstance(hotkey, str) or not hotkey for hotkey in hotkeys):
        raise ValueError("metagraph hotkeys are invalid")
    if len(set(hotkeys)) != len(hotkeys):
        raise ValueError("metagraph hotkeys are not unique")
    return hotkeys


def _snapshot_vector(metagraph: Any, field: str, size: int) -> Tuple[Any, ...]:
    try:
        values = tuple(getattr(metagraph, field))
    except (AttributeError, TypeError) as exc:
        raise ValueError("metagraph %s is invalid" % field) from exc
    if len(values) != size:
        raise ValueError("metagraph %s length differs" % field)
    return tuple(_scalar(value) for value in values)


def _stake_vector(metagraph: Any, size: int) -> Tuple[Any, ...]:
    values = getattr(metagraph, "S", None)
    if values is None:
        values = getattr(metagraph, "stake", None)
    if values is None:
        raise ValueError("metagraph stake is unavailable")
    try:
        decoded = tuple(_scalar(value) for value in values)
    except TypeError as exc:
        raise ValueError("metagraph stake is invalid") from exc
    if len(decoded) != size:
        raise ValueError("metagraph stake length differs")
    return decoded


def _require_bool(value: Any, field: str) -> bool:
    # Bittensor 10.5 uses an int64 vector for activity, but bool for permits.
    if field == "active" and type(value) is int and value in (0, 1):
        return bool(value)
    if type(value) is not bool:
        raise ValueError("metagraph %s is not boolean" % field)
    return value


def _require_stake(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("metagraph stake is not numeric")
    stake = float(value)
    if not math.isfinite(stake) or stake < 0:
        raise ValueError("metagraph stake is invalid")
    return stake


def _role_at_uid(
    metagraph: Any,
    uid: int,
    *,
    network_name: str,
    require_stake: bool,
    hotkey_count: int,
) -> Tuple[str, str]:
    permits = _snapshot_vector(metagraph, "validator_permit", hotkey_count)
    decoded_permits = tuple(
        _require_bool(value, "validator_permit") for value in permits
    )
    permit = decoded_permits[uid]
    if network_name == "test":
        active_values = _snapshot_vector(metagraph, "active", hotkey_count)
        decoded_active = tuple(
            _require_bool(value, "active") for value in active_values
        )
        active = decoded_active[uid]
        return classify_hotkey_role(
            active, permit, 0, network_name=network_name
        )

    if not require_stake:
        if permit:
            return "validator", "permit=True"
        return "miner", "permit=False"

    stakes = _stake_vector(metagraph, hotkey_count)
    decoded_stakes = tuple(_require_stake(value) for value in stakes)
    return classify_hotkey_role(
        False, permit, decoded_stakes[uid], network_name=network_name
    )


def classify_hotkey_role(
    active: bool,
    validator_permit: bool,
    stake: float,
    *,
    network_name: str,
) -> Tuple[str, str]:
    """Apply the shared network-specific validator policy to one neuron."""

    permit = _require_bool(_scalar(validator_permit), "validator_permit")
    if network_name == "test":
        is_active = _require_bool(_scalar(active), "active")
        if permit:
            return "validator", "testnet, permit=True"
        if is_active:
            return "validator", "testnet, active=True"
        return "miner", "testnet, active=False, permit=False"

    stake_weight = _require_stake(_scalar(stake))
    if permit and stake_weight >= STAKE_THRESHOLD:
        return (
            "validator",
            "stake_weight=%.0f >= %d, permit=True"
            % (stake_weight, STAKE_THRESHOLD),
        )
    return "miner", "stake_weight=%.0f, permit=%s" % (stake_weight, permit)


def classify_hotkey_from_metagraph(
    hotkey: str,
    metagraph: Any,
    *,
    network_name: str,
) -> Tuple[bool, Optional[str]]:
    """Look up and classify one hotkey with the shared validator policy."""

    hotkeys = _snapshot_hotkeys(metagraph)
    if hotkey not in hotkeys:
        return False, None
    role, _reason = _role_at_uid(
        metagraph,
        hotkeys.index(hotkey),
        network_name=network_name,
        require_stake=True,
        hotkey_count=len(hotkeys),
    )
    return True, role


def validator_uid(
    metagraph: Any,
    hotkey: str,
    *,
    netuid: int,
    network_name: str,
    require_stake: bool = True,
) -> int:
    """Return an eligible validator UID or raise a stable eligibility code."""

    observed_netuid = _scalar(getattr(metagraph, "netuid", None))
    if (
        isinstance(observed_netuid, bool)
        or not isinstance(observed_netuid, int)
        or observed_netuid != netuid
    ):
        raise ValueError("metagraph subnet differs")
    hotkeys = _snapshot_hotkeys(metagraph)
    if hotkey not in hotkeys:
        raise ValidatorIneligible("runner_hotkey_unregistered")
    uid = hotkeys.index(hotkey)
    role, _reason = _role_at_uid(
        metagraph,
        uid,
        network_name=network_name,
        require_stake=require_stake,
        hotkey_count=len(hotkeys),
    )
    if role == "validator":
        return uid
    permits = _snapshot_vector(metagraph, "validator_permit", len(hotkeys))
    permit = _require_bool(permits[uid], "validator_permit")
    if network_name != "test" and permit and require_stake:
        raise ValidatorIneligible("runner_stake_below_minimum")
    raise ValidatorIneligible("runner_validator_required")


def validator_hotkeys_from_metagraph(
    metagraph: Any, *, network_name: str
) -> List[str]:
    """Return all validator hotkeys using the same policy as single lookups."""

    hotkeys = _snapshot_hotkeys(metagraph)
    return [
        hotkey
        for uid, hotkey in enumerate(hotkeys)
        if _role_at_uid(
            metagraph,
            uid,
            network_name=network_name,
            require_stake=True,
            hotkey_count=len(hotkeys),
        )[0]
        == "validator"
    ]


__all__ = [
    "MIN_VALIDATOR_STAKE_WEIGHT",
    "STAKE_THRESHOLD",
    "ValidatorIneligible",
    "classify_hotkey_from_metagraph",
    "classify_hotkey_role",
    "validator_hotkeys_from_metagraph",
    "validator_uid",
]
