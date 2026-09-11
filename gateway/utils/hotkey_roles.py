"""Pure subnet hotkey role classification shared by gateway processes."""

from __future__ import annotations

from typing import Any, Optional, Tuple


STAKE_THRESHOLD = 500000  # 500K TAO for mainnet stake-based classification


def _scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def classify_hotkey_role(
    active: bool,
    validator_permit: bool,
    stake: float,
    *,
    network_name: str,
) -> Tuple[str, str]:
    """Apply the gateway's validator/miner policy to one registered neuron."""

    if network_name == "test":
        # A registered testnet validator can retain its permit while inactive.
        # Keep existing active-neuron behavior, without imposing Finney stake.
        if validator_permit:
            return "validator", "testnet, permit=True"
        if active:
            return "validator", "testnet, active=True"
        return (
            "miner",
            "testnet, active=%s, permit=%s, stake=%.0f"
            % (active, validator_permit, stake),
        )

    is_validator = (
        (active and validator_permit)
        or (stake > STAKE_THRESHOLD and validator_permit)
    )
    if is_validator:
        if active and validator_permit:
            return "validator", "active=True, permit=True"
        return (
            "validator",
            "stake=%.0f τ > %d, permit=True" % (stake, STAKE_THRESHOLD),
        )
    return (
        "miner",
        "active=%s, permit=%s, stake=%.0f"
        % (active, validator_permit, stake),
    )


def classify_hotkey_from_metagraph(
    hotkey: str,
    metagraph: Any,
    *,
    network_name: str,
) -> Tuple[bool, Optional[str]]:
    """Look up and classify one hotkey from a metagraph-shaped snapshot.

    Bittensor metagraphs expose total stake as ``S``. Arena's immutable
    finalized snapshot calls the same field ``stake``. Missing role fields for
    a registered hotkey are an unavailable authority source, not an
    unregistered result, so malformed inputs raise and callers fail closed.
    """

    hotkeys = list(metagraph.hotkeys)
    if hotkey not in hotkeys:
        return False, None
    uid = hotkeys.index(hotkey)
    active = bool(_scalar(metagraph.active[uid]))
    validator_permit = bool(_scalar(metagraph.validator_permit[uid]))
    stakes = getattr(metagraph, "S", None)
    if stakes is None:
        stakes = getattr(metagraph, "stake", None)
    if stakes is None:
        raise ValueError("metagraph stake is unavailable")
    stake = float(_scalar(stakes[uid]))
    if stake != stake or stake in (float("inf"), float("-inf")):
        raise ValueError("metagraph stake is not finite")
    role, _reason = classify_hotkey_role(
        active,
        validator_permit,
        stake,
        network_name=network_name,
    )
    return True, role


__all__ = [
    "STAKE_THRESHOLD",
    "classify_hotkey_from_metagraph",
    "classify_hotkey_role",
]
