"""Compatibility helpers for Bittensor extrinsic responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


def _message(value: Any) -> str:
    return "" if value is None else str(value)
