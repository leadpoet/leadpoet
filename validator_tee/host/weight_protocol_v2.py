"""Authoritative validator weight protocol selection."""

from __future__ import annotations

from typing import Optional


AUTHORITATIVE_V2_PROTOCOL = "authoritative_v2"


def normalize_weight_protocol(value: Optional[str]) -> str:
    protocol = str(value or AUTHORITATIVE_V2_PROTOCOL).strip().lower()
    if protocol != AUTHORITATIVE_V2_PROTOCOL:
        raise RuntimeError(
            "VALIDATOR_WEIGHT_PROTOCOL must be authoritative_v2; "
            "V1 authority is retired"
        )
    return protocol
