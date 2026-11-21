"""Read ``GammaSpent(netuid, service, hotkey)`` at a finalized block.

Gamma is a dollar-valued, non-transferable credit (10^9 units per USD). The
chain keeps a running total of what each hotkey spent to each service.
Runtime 493 identifies the service by its AccountId32 hotkey. Older runtimes
can lack the storage item, and Gamma may not be enabled on netuid 71 yet.
An absent item reads as ``available=False`` with zero spent: no balance, and
live orders are refused. Any other read failure raises, so the caller fails
closed instead of crediting a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from lab_arena.chain import ArenaChain, ArenaChainError, BlockRef

_SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,48}$")
MAX_U128 = (1 << 128) - 1


class GammaReadError(ArenaChainError):
    pass


@dataclass(frozen=True)
class GammaReading:
    hotkey: str
    block_number: int
    block_hash: str
    spent_units: int
    available: bool


def parse_service_id(value: str) -> str:
    """The service id is the service hotkey's SS58 account address."""

    text = str(value or "").strip()
    if _SS58_RE.fullmatch(text):
        return text
    raise GammaReadError("Gamma service id must be an SS58 hotkey")


def _storage_missing(exc: BaseException) -> bool:
    # async_substrate_interface raises StorageFunctionNotFound (or a ValueError
    # naming the missing item) when the runtime metadata has no such pallet or
    # storage function. Walk the cause chain: ArenaChain wraps client errors.
    current: Optional[BaseException] = exc
    while current is not None:
        name = type(current).__name__
        text = str(current).lower()
        if name in ("StorageFunctionNotFound", "PalletNotFound") or (
            "not found" in text and ("storage" in text or "pallet" in text)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _units(value: Any) -> int:
    raw = getattr(value, "value", value)
    if isinstance(raw, dict) and len(raw) == 1:
        raw = next(iter(raw.values()))
    if raw is None:
        return 0  # ValueQuery default for a hotkey that never spent
    if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw <= MAX_U128:
        raise GammaReadError("GammaSpent value is not a u128")
    return int(raw)


class GammaBalanceReader:
    """Finalized, fail-closed reads of the Gamma spent counter."""

    def __init__(
        self,
        chain: ArenaChain,
        *,
        service_id: str,
        pallet: str = "SubtensorModule",
        storage: str = "GammaSpent",
    ) -> None:
        self._chain = chain
        self._service = parse_service_id(service_id)
        self._pallet = pallet
        self._storage = storage

    def read(self, hotkey: str, *, head: Optional[BlockRef] = None) -> GammaReading:
        if not _SS58_RE.fullmatch(str(hotkey or "")):
            raise GammaReadError("hotkey is invalid")
        block = head if head is not None else self._chain.finalized_head()
        try:
            value = self._chain.client.query(
                self._pallet,
                self._storage,
                [self._chain.config.netuid, self._service, hotkey],
                block_hash=block.hash,
            )
        except Exception as exc:
            if _storage_missing(exc):
                return GammaReading(hotkey, block.number, block.hash, 0, False)
            raise GammaReadError("GammaSpent read failed") from exc
        return GammaReading(hotkey, block.number, block.hash, _units(value), True)


__all__ = ["GammaBalanceReader", "GammaReadError", "GammaReading", "parse_service_id"]
