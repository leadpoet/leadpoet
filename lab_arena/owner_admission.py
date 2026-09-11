"""Resolve a submitting hotkey's owner from one finalized metagraph snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lab_arena import chain as chain_module
from lab_arena.contracts import require_hotkey


class OwnerAdmissionError(RuntimeError):
    """Finalized owner admission could not be established safely."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class OwnerAdmission:
    coldkey: str
    block_number: int
    block_hash: str


def resolve_finalized_owner(chain: Any, hotkey: str) -> OwnerAdmission:
    """Return the owner and exact finalized block used for the decision."""

    try:
        snapshot = chain.metagraph(finalized=True)
        uid = chain_module.uid_for_hotkey(snapshot, hotkey)
    except Exception as exc:
        raise OwnerAdmissionError("owner_resolution_unavailable") from exc
    if uid is None:
        raise OwnerAdmissionError("hotkey_unregistered")
    try:
        coldkey = require_hotkey(snapshot.coldkeys[uid], "owner_coldkey")
        block_number = int(snapshot.block_number)
        block_hash = chain_module.normalize_block_hash(snapshot.block_hash)
    except Exception as exc:
        raise OwnerAdmissionError("owner_resolution_unavailable") from exc
    if isinstance(snapshot.block_number, bool) or block_number < 0:
        raise OwnerAdmissionError("owner_resolution_unavailable")
    return OwnerAdmission(
        coldkey=coldkey,
        block_number=block_number,
        block_hash=block_hash,
    )
