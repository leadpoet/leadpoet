"""Pay lead-order rewards through the collection epoch's signed weight state.

Validators derive weights only from the accepted weight state
(``leadpoet_canonical.arena_weights.derive_arena_weights``). In an epoch that
pays lead orders, the state carries ``lead_orders``: the public payout
document's hash and the micro-USD paid per miner. The validator code splits
the epoch's miner emission: collection is guaranteed 80%, lead orders 20%,
and each may use what the other leaves unused.

``LeadOrderLane.settle`` is the gateway half. It caps the payout at the lead
orders' allowance before recording it, so what the ledger marks paid is what
validators weight. ``scripts/730`` makes the database refuse a state that
disagrees with the recorded payout, and closes the epoch to new collection
jobs once the payout is recorded, so the total it was capped against stands.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from leadpoet_canonical.arena_weights import LEAD_ORDER_EMISSION_SHARE
from lab_arena.lead_orders.money import OwedEntry

_VALUATION_FIELDS = (
    "block", "block_hash", "miner_alpha_rao", "tao_rao_per_alpha", "tao_price_microusd",
    "price_updated_at", "observed_at", "emission_microusd",
)
TAO_PRICE_SOURCE = "coingecko:bittensor/usd"


def lead_order_allowance_microusd(emission_microusd: int, collection_microusd: int) -> int:
    """Most the lead orders may be paid: their share, plus what collection leaves."""

    emission, collection = int(emission_microusd), int(collection_microusd)
    if emission < 0 or collection < 0:
        raise ValueError("amounts must not be negative")
    guaranteed = emission * LEAD_ORDER_EMISSION_SHARE.numerator // LEAD_ORDER_EMISSION_SHARE.denominator
    return max(guaranteed, emission - collection)


def registered_entries(entries: Iterable[OwedEntry], hotkeys: Sequence[str]) -> List[OwedEntry]:
    """Owed entries whose miner can receive weight now; the rest wait."""

    registered = set(hotkeys)
    return [entry for entry in entries if entry.miner_hotkey in registered]


class LeadOrderLane:
    """One epoch's lead-order payout, for the collection weight state."""

    def __init__(self, service: Any, *, registered_hotkeys: Callable[[], Sequence[str]]) -> None:
        self._service = service
        self._registered_hotkeys = registered_hotkeys

    def settle(self, epoch: int, *, collection_microusd: int, burn_hotkey: str,
               valuation: Callable[[], Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return ``{"valuation", "lead_orders"}`` for a paying epoch, else ``None``.

        A payout already recorded for the epoch is returned with the valuation
        it was capped by, so a retry signs the same amounts. Shadow mode
        records its what-if document and returns ``None``. The burn hotkey
        is never paid: a state that pays it is invalid.
        """

        config = self._service.config
        if not config.computes_rewards:
            return None
        if not config.weights_rewards:
            try:
                self._service.run_epoch(epoch)
            except Exception:  # shadow pays nothing and must never hold up weights
                logging.getLogger(__name__).warning("shadow lead order epoch %s failed", epoch, exc_info=True)
            return None

        def priced() -> Mapping[str, Any]:
            return dict(valuation(), tao_price_source=TAO_PRICE_SOURCE)

        document = self._service.run_epoch(
            epoch, valuation=priced, collection_microusd=int(collection_microusd),
            # Read only when a payout is worked out, never on a retry of a recorded one.
            registered_hotkeys=lambda: [key for key in self._registered_hotkeys() if key != burn_hotkey],
            payable_cap_microusd=lambda emission: lead_order_allowance_microusd(emission, collection_microusd),
        )
        if not document or document.get("mode") != "live" or not document.get("paid_microusd"):
            return None
        paid: Dict[str, int] = {}
        for payment in document["payments"]:
            paid[payment["miner_hotkey"]] = paid.get(payment["miner_hotkey"], 0) + int(payment["paid_microusd"])
        return {
            "valuation": {field: document["emission_valuation"][field] for field in _VALUATION_FIELDS},
            "lead_orders": {"payout_hash": document["payout_hash"], "paid_microusd": paid},
        }


__all__ = ["LeadOrderLane", "TAO_PRICE_SOURCE", "lead_order_allowance_microusd", "registered_entries"]
