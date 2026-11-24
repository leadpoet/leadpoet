"""Per-order miner rewards and per-epoch lane payouts.

* When a live order finishes, ``order_rewards`` computes what each miner
  earned from its settled calls (``money.order_miner_rewards``, each call's
  cost capped at its reservation) and the store keeps it once. A cost that
  becomes known later (a late answer or a generation read-back) is owed as a
  separate late reward per call (``lead_order_reward_late_v1``). Shadow
  orders run on Leadpoet's keys and publish ``what_if_reward`` only.
* Each epoch, ``epoch_payout_document`` pays owed USD oldest first under the
  lane cap and carries the rest forward. In shadow mode the document is
  published only and nothing is recorded as paid; in live mode the payments
  are recorded and ``weights.LeadOrderLane`` puts them in the epoch's signed weight state.

The lane cap is a share of the epoch's miner emission in USD.
``ChainEmissionPrice`` values it once, when the payout starts: the alpha
emitted to miners at the epoch's first block, times the chain's alpha price
in TAO at that block, times one fresh TAO/USD quote (Gavin, Nov 23 2026: one
read, no averaging). The inputs are published in the signed payout document.
When any read fails the source fails closed: a live payout is refused (the
Arena vector is unchanged, the rewards stay owed) and a shadow document
publishes what is owed with no payments.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

from lab_arena import signing
from lab_arena.contracts import document_hash
from lab_arena.lead_orders import money
from lab_arena.lead_orders.policy import LeadOrderPolicy

PAYOUT_SCHEMA_VERSION = "leadpoet.lead_orders.epoch_payout.v1"
PAYOUT_HASH_FIELD = "payout_hash"
WEIGHT_STEPS = 65_535  # leadpoet_canonical.arena_weights.U16_MAX


class EmissionPriceUnavailable(RuntimeError):
    pass


class EmissionPriceSource(Protocol):
    def valuation(self, epoch: int) -> Mapping[str, Any]:
        """The epoch's miner emission: ``emission_microusd`` plus its inputs."""


class UnavailableEmissionPrice:
    """Fail-closed default when no emission USD source is wired in."""

    def valuation(self, epoch: int) -> Mapping[str, Any]:
        raise EmissionPriceUnavailable("no emission USD price source is configured")


class ChainEmissionPrice:
    """One read at payout start, shared with the collection lane's valuation.

    ``scope`` returns the finalized epoch and its first block; ``valuation``
    values the miner emission at a block
    (``lab_arena.collection_rewards.emission_valuation``). Only the current
    finalized epoch is valued: a fresh TAO/USD quote says nothing about an
    older epoch.
    """

    def __init__(
        self,
        *,
        scope: Callable[[], Mapping[str, Any]],
        valuation: Callable[[int], Mapping[str, Any]],
        tao_price_source: str,
    ) -> None:
        self._scope = scope
        self._valuation = valuation
        self._tao_price_source = tao_price_source

    def valuation(self, epoch: int) -> Mapping[str, Any]:
        try:
            scope = self._scope()
            current = int(scope["epoch"])
            document = None if current != int(epoch) else dict(self._valuation(int(scope["valid_from_block"])))
        except Exception as exc:  # any chain or quote failure: no price, no payout
            raise EmissionPriceUnavailable("emission valuation is unavailable") from exc
        if document is None:
            raise EmissionPriceUnavailable("only the current finalized epoch can be valued")
        emission = document.get("emission_microusd")
        if type(emission) is not int or emission <= 0:
            raise EmissionPriceUnavailable("emission valuation is not a positive amount")
        return dict(document, tao_price_source=self._tao_price_source)


def _rewardable_calls(calls: Sequence[Mapping[str, Any]]) -> List[tuple]:
    """(miner hotkey, rewardable cost) per settled call. A cost above the
    call's reservation is recorded and charged, but rewards stop at the
    reservation: the excess is shown in the miner's statement."""

    return [
        (row.get("miner_hotkey"), min(int(row["cost_microusd"]), int(row["reserved_microusd"])))
        for row in calls
        if row.get("status") == "settled" and row.get("cost_microusd") is not None
    ]


def order_rewards(policy: LeadOrderPolicy, *, order: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """USD owed per miner for one finished order, from its settled calls."""

    return money.order_miner_rewards(policy, _rewardable_calls(calls),
                                     ceiling_microusd=int(order["spend_ceiling_microusd"]))


def what_if_reward(policy: LeadOrderPolicy, *, order: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]) -> int:
    """Shadow: what miners would have earned had their keys paid every call."""

    spend = money.rewardable_spend_by_hotkey(
        [("what-if", cost) for _, cost in _rewardable_calls(calls)],
        ceiling_microusd=int(order["spend_ceiling_microusd"]),
    ).get("what-if", 0)
    return money.miner_reward_microusd(policy, spend)


def owed_entries(rows: Sequence[Mapping[str, Any]]) -> List[money.OwedEntry]:
    return [
        money.OwedEntry(
            order_id=str(row["order_id"]), miner_hotkey=str(row["miner_hotkey"]),
            settled_epoch=int(row["settled_epoch"]), settled_sequence=int(row["settled_sequence"]),
            owed_microusd=int(row["owed_microusd"]), paid_microusd=int(row["paid_microusd"]),
            call_id=None if row.get("call_id") is None else str(row["call_id"]),
        )
        for row in rows
    ]


def epoch_payout_document(
    policy: LeadOrderPolicy,
    *,
    epoch: int,
    mode: str,
    entries: Sequence[money.OwedEntry],
    valuation: Optional[Mapping[str, Any]],
    payable_cap_microusd: Optional[int] = None,
    collection_microusd: Optional[int] = None,
) -> Dict[str, Any]:
    """The published epoch document (unsigned body plus its hash).

    ``valuation`` is the emission price source's answer, published whole so
    anyone can redo the arithmetic; ``None`` means no price was available.

    ``payable_cap_microusd`` lets the caller lower the cap further, for
    example to the burn residual's USD value in live mode.
    """

    if mode not in ("shadow", "live"):
        raise money.MoneyError("payouts exist only in shadow and live mode")
    owed_total = sum(e.remaining_microusd for e in entries if e.settled_epoch <= epoch)
    if valuation is None:
        if mode == "live":
            raise EmissionPriceUnavailable("live payouts need the emission USD price")
        body = {
            "schema_version": PAYOUT_SCHEMA_VERSION, "epoch": int(epoch), "mode": mode,
            "policy_version": policy.version, "emission_price_available": False,
            "cap_microusd": None, "paid_microusd": 0, "owed_microusd": owed_total,
            "carry_forward_microusd": owed_total, "payments": [], "weight_fractions": {},
        }
        return dict(body, **{PAYOUT_HASH_FIELD: document_hash(body)})
    miner_emission_microusd = int(valuation["emission_microusd"])
    cap = money.lane_cap_microusd(policy, miner_emission_microusd)
    if payable_cap_microusd is not None:
        cap = min(cap, max(0, int(payable_cap_microusd)))
    # Validators emit u16 weights: a share under 1/65535 of the epoch rounds to
    # nothing, so a live payout that small waits for a later epoch.
    minimum = miner_emission_microusd // WEIGHT_STEPS + 1 if mode == "live" else 0
    payout = money.epoch_payout(epoch, entries, cap_microusd=cap, min_hotkey_microusd=minimum)
    fractions = money.lane_weight_fractions(payout.paid_by_hotkey, miner_emission_microusd)
    body = {
        "schema_version": PAYOUT_SCHEMA_VERSION, "epoch": int(epoch), "mode": mode,
        "policy_version": policy.version, "emission_price_available": True,
        "miner_emission_microusd": miner_emission_microusd, "emission_valuation": dict(valuation),
        "cap_microusd": cap, "min_hotkey_microusd": minimum,
        "paid_microusd": payout.paid_microusd, "owed_microusd": owed_total,
        "carry_forward_microusd": payout.carry_forward_microusd,
        # The collection total the cap was worked out against (weight-state epochs only).
        **({} if collection_microusd is None else {"collection_microusd": int(collection_microusd)}),
        "payments": [
            # call_id names a late repayment (one call), None an order's reward.
            {"order_id": order_id, "miner_hotkey": hotkey, "paid_microusd": amount, "call_id": call_id}
            for order_id, hotkey, amount, call_id in payout.payments
        ],
        # Exact rationals as strings so every validator derives the same vector.
        "weight_fractions": {k: "%d/%d" % (v.numerator, v.denominator) for k, v in sorted(fractions.items())},
    }
    return dict(body, **{PAYOUT_HASH_FIELD: document_hash(body)})


def lane_fractions(document: Mapping[str, Any]) -> Dict[str, Fraction]:
    """Weight shares from a live payout document; empty for shadow."""

    if document.get("mode") != "live":
        return {}
    return {k: Fraction(v) for k, v in (document.get("weight_fractions") or {}).items()}


def sign_payout(signer: signing.ArenaSigner, document: Mapping[str, Any]) -> Dict[str, Any]:
    return signing.sign_document(signer, document, hash_field=PAYOUT_HASH_FIELD)


__all__ = [
    "ChainEmissionPrice",
    "EmissionPriceSource",
    "EmissionPriceUnavailable",
    "PAYOUT_HASH_FIELD",
    "UnavailableEmissionPrice",
    "epoch_payout_document",
    "lane_fractions",
    "order_rewards",
    "owed_entries",
    "sign_payout",
    "what_if_reward",
]
