"""Per-order miner rewards and per-epoch lane payouts.

* When a live order finishes, ``order_rewards`` computes what each miner
  earned from its settled calls (``money.order_miner_rewards``, each call's
  cost capped at its reservation) and the store keeps it once. A cost that
  becomes known later (a late answer or a generation read-back) is owed as a
  separate late reward per call (``lead_order_reward_late_v1``). Shadow
  orders run on the subnet's own keys and publish ``what_if_reward`` only.
* Each epoch, ``epoch_payout_document`` pays owed USD oldest first under the
  lane cap and carries the rest forward. In shadow mode the document is
  published only and nothing is recorded as paid; in live mode the payments
  are recorded and ``weights.merge_lane_weights`` turns them into weight.

The lane cap is a share of the epoch's miner emission in USD, so it needs a
USD price for emission. No such price is on chain and the source is not
decided, so the default ``EmissionPriceSource`` fails closed: a live payout
is refused (the Arena vector is unchanged) and a shadow document publishes
what is owed with no payments.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from lab_arena import signing
from lab_arena.contracts import document_hash
from lab_arena.lead_orders import money
from lab_arena.lead_orders.policy import LeadOrderPolicy

PAYOUT_SCHEMA_VERSION = "leadpoet.lead_orders.epoch_payout.v1"
PAYOUT_HASH_FIELD = "payout_hash"


class EmissionPriceUnavailable(RuntimeError):
    pass


class EmissionPriceSource(Protocol):
    def miner_emission_microusd(self, epoch: int) -> int: ...


class UnavailableEmissionPrice:
    """Fail-closed default until the emission USD source is decided."""

    def miner_emission_microusd(self, epoch: int) -> int:
        raise EmissionPriceUnavailable("no emission USD price source is configured")


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
    miner_emission_microusd: Optional[int],
    payable_cap_microusd: Optional[int] = None,
) -> Dict[str, Any]:
    """The published epoch document (unsigned body plus its hash).

    ``payable_cap_microusd`` lets the caller lower the cap further, for
    example to the burn residual's USD value in live mode.
    """

    if mode not in ("shadow", "live"):
        raise money.MoneyError("payouts exist only in shadow and live mode")
    owed_total = sum(e.remaining_microusd for e in entries if e.settled_epoch <= epoch)
    if miner_emission_microusd is None:
        if mode == "live":
            raise EmissionPriceUnavailable("live payouts need the emission USD price")
        body = {
            "schema_version": PAYOUT_SCHEMA_VERSION, "epoch": int(epoch), "mode": mode,
            "policy_version": policy.version, "emission_price_available": False,
            "cap_microusd": None, "paid_microusd": 0, "owed_microusd": owed_total,
            "carry_forward_microusd": owed_total, "payments": [], "weight_fractions": {},
        }
        return dict(body, **{PAYOUT_HASH_FIELD: document_hash(body)})
    cap = money.lane_cap_microusd(policy, miner_emission_microusd)
    if payable_cap_microusd is not None:
        cap = min(cap, max(0, int(payable_cap_microusd)))
    payout = money.epoch_payout(epoch, entries, cap_microusd=cap)
    fractions = money.lane_weight_fractions(payout.paid_by_hotkey, miner_emission_microusd)
    body = {
        "schema_version": PAYOUT_SCHEMA_VERSION, "epoch": int(epoch), "mode": mode,
        "policy_version": policy.version, "emission_price_available": True,
        "miner_emission_microusd": int(miner_emission_microusd),
        "cap_microusd": cap, "paid_microusd": payout.paid_microusd, "owed_microusd": owed_total,
        "carry_forward_microusd": payout.carry_forward_microusd,
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
