"""Order-book money and ranking rules, as pure functions.

All amounts are integer micro-USD. A miner's ask is what they earn per unit;
the buyer pays the ask plus ``MARKUP_PPM`` (10%), rounded up. For a call
asked at $0.01 the buyer pays $0.011 and the miner is owed $0.01.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional

from lab_arena.tool_market.catalog import Operation

PPM = 1_000_000
MARKUP_PPM = 100_000
# Highest ask: $10 per call, or 10x the vendor's own price.
MAX_ASK_MICROUSD = 10_000_000

FAILURE_UNAUTHORIZED = "unauthorized"
FAILURE_NO_CREDIT = "no_credit"
FAILURE_FORBIDDEN = "forbidden"


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def owed_microusd(operation: Operation, units: int, ask_microusd: int) -> int:
    """What the miner earns for ``units`` (capped at the operation's maximum)."""

    counted = max(0, min(int(units), operation.max_units))
    return counted * int(ask_microusd) // operation.unit_scale


def charge_microusd(owed: int) -> int:
    """What the buyer pays for ``owed``: 10% more, rounded up."""

    return _ceil_div(int(owed) * (PPM + MARKUP_PPM), PPM)


def reserve_microusd(operation: Operation, ask_microusd: int) -> int:
    """The most one call can charge, held against the tab until it settles."""

    return charge_microusd(_ceil_div(operation.max_units * int(ask_microusd), operation.unit_scale))


def valid_ask(ask_microusd: object) -> bool:
    return isinstance(ask_microusd, int) and not isinstance(ask_microusd, bool) \
        and 0 < ask_microusd <= MAX_ASK_MICROUSD


@dataclass(frozen=True)
class Ask:
    """One eligible key's standing price, as ``tool_market_book_v1`` returns it."""

    key_id: str
    miner_hotkey: str
    envelope: str
    ask_microusd: int
    ask_sequence: int
    # A key whose vendor has a free balance check and whose last check is old.
    needs_check: bool = False

    @classmethod
    def from_row(cls, row: Mapping) -> "Ask":
        return cls(key_id=str(row["key_id"]), miner_hotkey=str(row["miner_hotkey"]), envelope=str(row["envelope"]),
                   ask_microusd=int(row["ask_microusd"]), ask_sequence=int(row["ask_sequence"]),
                   needs_check=bool(row.get("needs_check")))


def rank(asks: Iterable[Ask]) -> List[Ask]:
    """Cheapest first; an equal price goes to the earlier ask."""

    return sorted(asks, key=lambda a: (a.ask_microusd, a.ask_sequence))


def key_failure(status: int) -> Optional[str]:
    """A vendor answer that says the key itself is unusable, not the request."""

    return {401: FAILURE_UNAUTHORIZED, 402: FAILURE_NO_CREDIT, 403: FAILURE_FORBIDDEN}.get(int(status))


def rate_limited(status: int, headers: Mapping[str, str]) -> bool:
    """A 429 with no Retry-After: this key is spent for now, try the next one.

    With Retry-After the buyer gets the 429 and decides when to retry.
    """

    lowered = {str(k).lower(): str(v) for k, v in headers.items()}
    return int(status) == 429 and not lowered.get("retry-after", "").strip()


__all__ = [
    "Ask",
    "MARKUP_PPM",
    "MAX_ASK_MICROUSD",
    "charge_microusd",
    "key_failure",
    "owed_microusd",
    "rank",
    "rate_limited",
    "reserve_microusd",
    "valid_ask",
]
