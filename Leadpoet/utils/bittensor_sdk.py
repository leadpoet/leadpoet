"""Compatibility helpers for Bittensor extrinsic responses."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterator


@dataclass(frozen=True)
class ExtrinsicOutcome:
    """Normalized result from a Bittensor extrinsic submission."""

    success: bool
    message: str

    @classmethod
    def from_sdk(cls, response: Any) -> "ExtrinsicOutcome":
        """Prefer the v10 response API while accepting the supported v9 shapes."""

        success = getattr(response, "success", None)
        if isinstance(success, bool):
            return cls(success=success, message=_message(getattr(response, "message", "")))

        if (
            isinstance(response, (tuple, list))
            and len(response) == 2
            and isinstance(response[0], bool)
        ):
            return cls(success=response[0], message=_message(response[1]))

        if isinstance(response, bool):
            return cls(success=response, message="")

        raise TypeError(
            "unsupported Bittensor extrinsic response: "
            f"{type(response).__name__}"
        )


@contextmanager
def weight_hyperparameters_compat(
    subtensor: Any,
    *,
    netuid: int,
    sdk_version: str,
) -> Iterator[None]:
    """Supply exact chain storage values to Bittensor 9 weight encoding.

    Finney's composite netuid runtime type cannot be encoded by Bittensor 9's
    subnet-hyperparameter runtime API. The two values used by its timelocked
    weight extrinsic remain available from canonical Subtensor storage.
    """

    try:
        sdk_major = int(str(sdk_version).split(".", 1)[0])
    except (TypeError, ValueError):
        sdk_major = 10
    if sdk_major >= 10:
        yield
        return

    original = subtensor.get_subnet_hyperparameters
    expected_netuid = int(netuid)

    def direct_hyperparameters(requested_netuid: int, block=None):
        if int(requested_netuid) != expected_netuid:
            return original(requested_netuid, block=block)
        block_hash = (
            subtensor.substrate.get_block_hash(int(block))
            if block is not None
            else None
        )

        def storage_value(name: str) -> int:
            value = subtensor.substrate.query(
                module="SubtensorModule",
                storage_function=name,
                params=[expected_netuid],
                block_hash=block_hash,
            )
            return int(getattr(value, "value", value))

        tempo = storage_value("Tempo")
        reveal_period = storage_value("RevealPeriodEpochs")
        if tempo <= 0 or reveal_period <= 0:
            raise RuntimeError(
                "weight hyperparameters from chain storage are invalid"
            )
        return SimpleNamespace(
            tempo=tempo,
            commit_reveal_period=reveal_period,
        )

    subtensor.get_subnet_hyperparameters = direct_hyperparameters
    try:
        yield
    finally:
        subtensor.get_subnet_hyperparameters = original


def _message(value: Any) -> str:
    return "" if value is None else str(value)
