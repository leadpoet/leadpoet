"""Order-book money and ranking rules, as pure functions.

All amounts are integer micro-USD. A miner's ask is what they earn per unit;
the buyer pays the ask plus ``MARKUP_PPM`` (10%), rounded up. For a call
asked at $0.01 the buyer pays $0.011 and the miner is owed $0.01.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping

from lab_arena.tool_market.catalog import Operation

PPM = 1_000_000
MARKUP_PPM = 100_000


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def owed_microusd(operation: Operation, units: int, ask_microusd: int) -> int:
    """What the miner earns for ``units`` (capped at the operation's maximum)."""

    counted = max(0, min(int(units), operation.max_units))
    return counted * int(ask_microusd) // operation.unit_scale


def charge_microusd(owed: int) -> int:
    """What the buyer pays for ``owed``: 10% more, rounded up."""

    return _ceil_div(int(owed) * (PPM + MARKUP_PPM), PPM)


@dataclass(frozen=True)
class Ask:
    """One eligible key's standing price, as ``tool_market_book_v1`` returns it."""

    key_id: str
    miner_hotkey: str
    envelope: str
    ask_microusd: int
    ask_sequence: int
    # What the key may still earn today (its daily cap less its last 24 h).
    remaining_microusd: int
    # Rate-limited in the last minute, or unreliable in the last hour.
    demoted: bool = False

    @classmethod
    def from_row(cls, row: Mapping) -> "Ask":
        return cls(key_id=str(row["key_id"]), miner_hotkey=str(row["miner_hotkey"]), envelope=str(row["envelope"]),
                   ask_microusd=int(row["ask_microusd"]), ask_sequence=int(row["ask_sequence"]),
                   remaining_microusd=int(row["remaining_microusd"]), demoted=bool(row.get("demoted")))


def rank(asks: Iterable[Ask]) -> List[Ask]:
    """Cheapest first, demoted keys last; an equal price goes to the earlier ask."""

    return sorted(asks, key=lambda a: (a.demoted, a.ask_microusd, a.ask_sequence))


__all__ = [
    "Ask",
    "charge_microusd",
    "owed_microusd",
    "rank",
]
