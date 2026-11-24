"""Pure lead-order money arithmetic in integer micro-USD.

No I/O and no floats. The SQL functions in ``scripts/707-lead-orders.sql``
apply the same rules atomically; ``tests/lead_orders`` checks they agree.

Rounding always favours the side that keeps the lane safe: the minimum charge
rounds up, balances and miner rewards round down.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from lab_arena.lead_orders.policy import LeadOrderPolicy

PPM = 1_000_000
# Gamma is 10^9 units per USD, so one micro-USD is 1000 units.
GAMMA_UNITS_PER_MICROUSD = 1_000

REFUSAL_BELOW_MINIMUM = "below_minimum"
REFUSAL_DAILY_CAP = "daily_cap"
REFUSAL_INSUFFICIENT_BALANCE = "insufficient_balance"


class MoneyError(ValueError):
    pass


def _non_negative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MoneyError("%s must be a non-negative integer" % name)
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


# -- balance and hold --------------------------------------------------------


def gamma_units_to_microusd(units: int) -> int:
    return _non_negative(units, "gamma units") // GAMMA_UNITS_PER_MICROUSD


def balance_microusd(gamma_spent_units: int, settled_charges_microusd: int, open_holds_microusd: int) -> int:
    """Gamma spent to the service, minus charges on settled orders and open holds.

    The result can be negative only if the chain counter went backwards, which
    the caller treats as no balance.
    """

    return (
        gamma_units_to_microusd(gamma_spent_units)
        - _non_negative(settled_charges_microusd, "charges")
        - _non_negative(open_holds_microusd, "holds")
    )


def order_hold_microusd(policy: LeadOrderPolicy, leads: int) -> int:
    return _non_negative(leads, "leads") * policy.price_per_lead_microusd


def spend_ceiling_microusd(policy: LeadOrderPolicy, leads: int) -> int:
    return _non_negative(leads, "leads") * policy.spend_ceiling_per_lead_microusd


def order_refusal(
    policy: LeadOrderPolicy,
    *,
    leads: int,
    leads_ordered_today: int,
    balance_microusd: Optional[int],
) -> Optional[str]:
    """Why an order must be refused, or ``None``.

    ``balance_microusd=None`` means the mode does not use Gamma (staff and
    shadow orders); live callers always pass a number.
    """

    _non_negative(leads, "leads")
    _non_negative(leads_ordered_today, "leads ordered today")
    if leads < policy.min_leads_per_order:
        return REFUSAL_BELOW_MINIMUM
    if leads_ordered_today + leads > policy.daily_leads_per_hotkey:
        return REFUSAL_DAILY_CAP
    if balance_microusd is not None and balance_microusd < order_hold_microusd(policy, leads):
        return REFUSAL_INSUFFICIENT_BALANCE
    return None


# -- spend ceiling -------------------------------------------------------------


def reservation_allowed(ceiling_microusd: int, committed_microusd: int, reserve_microusd: int) -> bool:
    """A call may be sent only if its worst case still fits under the ceiling.

    ``committed`` is settled cost plus every open or uncertain reservation.
    """

    return (
        _non_negative(committed_microusd, "committed")
        + _non_negative(reserve_microusd, "reserve")
        <= _non_negative(ceiling_microusd, "ceiling")
    )


# -- charge and miner reward ----------------------------------------------------


def minimum_charge_microusd(policy: LeadOrderPolicy, spend_microusd: int) -> int:
    return _ceil_div(_non_negative(spend_microusd, "spend") * policy.min_charge_ratio_ppm, PPM)


def charge_microusd(policy: LeadOrderPolicy, *, hold_microusd: int, delivered: int, spend_microusd: int) -> int:
    """charge = min(hold, max(delivered x price, min_charge_ratio x spend))."""

    delivered_value = _non_negative(delivered, "delivered") * policy.price_per_lead_microusd
    return min(
        _non_negative(hold_microusd, "hold"),
        max(delivered_value, minimum_charge_microusd(policy, spend_microusd)),
    )


def rewardable_spend_by_hotkey(
    calls: Iterable[Tuple[Optional[str], int]],
    *,
    ceiling_microusd: int,
) -> Dict[str, int]:
    """Settled spend per miner hotkey that earns a reward.

    ``calls`` holds ``(miner_hotkey or None for Leadpoet's key, cost)`` for
    every settled call on the order. A final call can settle above its
    reservation (Deepline is reserved at a flat estimate), so the order's total
    spend can pass its ceiling. Rewards never pay for spend above the ceiling:
    every key's spend is scaled down by ceiling / total in that case. That
    keeps the anti-farming bound (the charge is at least min_charge_ratio x
    rewarded spend) true for every order.
    """

    ceiling = _non_negative(ceiling_microusd, "ceiling")
    total = 0
    by_hotkey: Dict[str, int] = {}
    for hotkey, cost in calls:
        _non_negative(cost, "call cost")
        total += cost
        if hotkey is not None:
            by_hotkey[hotkey] = by_hotkey.get(hotkey, 0) + cost
    if total <= ceiling or total == 0:
        return by_hotkey
    return {hotkey: (spend * ceiling) // total for hotkey, spend in by_hotkey.items()}


def miner_reward_microusd(policy: LeadOrderPolicy, rewardable_spend_microusd: int) -> int:
    """reward = spend x (1 + premium), rounded down."""

    return (_non_negative(rewardable_spend_microusd, "spend") * (PPM + policy.miner_premium_ppm)) // PPM


def order_miner_rewards(
    policy: LeadOrderPolicy,
    calls: Iterable[Tuple[Optional[str], int]],
    *,
    ceiling_microusd: int,
) -> Dict[str, int]:
    """USD owed per miner hotkey for one settled order. Leadpoet earns nothing."""

    return {
        hotkey: reward
        for hotkey, spend in rewardable_spend_by_hotkey(calls, ceiling_microusd=ceiling_microusd).items()
        for reward in (miner_reward_microusd(policy, spend),)
        if reward > 0
    }


# -- epoch payout under the lane cap -----------------------------------------------


@dataclass(frozen=True)
class OwedEntry:
    """What one settled order owes one miner hotkey, and what is already paid."""

    order_id: str
    miner_hotkey: str
    settled_epoch: int
    settled_sequence: int
    owed_microusd: int
    paid_microusd: int = 0
    # A late repayment for one call; None for an order's own reward.
    call_id: Optional[str] = None

    @property
    def remaining_microusd(self) -> int:
        return self.owed_microusd - self.paid_microusd


@dataclass(frozen=True)
class EpochPayout:
    epoch: int
    cap_microusd: int
    payments: Tuple[Tuple[str, str, int, Optional[str]], ...]  # (order_id, miner_hotkey, amount, call_id)
    paid_by_hotkey: Mapping[str, int]
    paid_microusd: int
    carry_forward_microusd: int


def lane_cap_microusd(policy: LeadOrderPolicy, miner_emission_microusd: int) -> int:
    """Largest USD value the lane may pay in one epoch."""

    return (_non_negative(miner_emission_microusd, "miner emission") * policy.lane_emission_cap_ppm) // PPM


def epoch_payout(epoch: int, entries: Sequence[OwedEntry], *, cap_microusd: int,
                 min_hotkey_microusd: int = 0) -> EpochPayout:
    """Pay owed amounts oldest first up to the cap; the rest carries forward.

    A hotkey whose payments this epoch total less than ``min_hotkey_microusd``
    is not paid: a weight that small rounds to zero, so it waits and adds up.

    Amounts stay in USD until paid. The paying epoch converts its payout to
    weight with its own emission value (``lane_weight_fractions``). Entries
    settled after ``epoch`` are not payable yet.
    """

    _non_negative(epoch, "epoch")
    cap = _non_negative(cap_microusd, "cap")
    seen = set()
    payable: List[OwedEntry] = []
    for entry in entries:
        key = (entry.order_id, entry.miner_hotkey, entry.call_id)
        if key in seen:
            raise MoneyError("duplicate owed entry")
        seen.add(key)
        _non_negative(entry.owed_microusd, "owed")
        _non_negative(entry.paid_microusd, "paid")
        if entry.paid_microusd > entry.owed_microusd:
            raise MoneyError("an entry was paid more than it was owed")
        if entry.settled_epoch <= epoch and entry.remaining_microusd > 0:
            payable.append(entry)
    payable.sort(key=lambda e: (e.settled_epoch, e.settled_sequence, e.order_id, e.miner_hotkey, e.call_id or ""))
    left = cap
    payments: List[Tuple[str, str, int, Optional[str]]] = []
    by_hotkey: Dict[str, int] = {}
    owed_total = 0
    for entry in payable:
        owed_total += entry.remaining_microusd
        amount = min(left, entry.remaining_microusd)
        if amount <= 0:
            continue
        left -= amount
        payments.append((entry.order_id, entry.miner_hotkey, amount, entry.call_id))
        by_hotkey[entry.miner_hotkey] = by_hotkey.get(entry.miner_hotkey, 0) + amount
    small = {hotkey for hotkey, total in by_hotkey.items() if total < min_hotkey_microusd}
    payments = [payment for payment in payments if payment[1] not in small]
    by_hotkey = {hotkey: total for hotkey, total in by_hotkey.items() if hotkey not in small}
    paid = sum(by_hotkey.values())
    return EpochPayout(
        epoch=epoch,
        cap_microusd=cap,
        payments=tuple(payments),
        paid_by_hotkey=by_hotkey,
        paid_microusd=paid,
        carry_forward_microusd=owed_total - paid,
    )


def lane_weight_fractions(paid_by_hotkey: Mapping[str, int], miner_emission_microusd: int) -> Dict[str, Fraction]:
    """Share of the epoch's miner emission each hotkey's payout needs."""

    emission = _non_negative(miner_emission_microusd, "miner emission")
    if emission == 0:
        if any(paid_by_hotkey.values()):
            raise MoneyError("cannot pay from an epoch with no emission value")
        return {}
    return {hotkey: Fraction(amount, emission) for hotkey, amount in paid_by_hotkey.items() if amount > 0}


__all__ = [
    "EpochPayout",
    "GAMMA_UNITS_PER_MICROUSD",
    "MoneyError",
    "OwedEntry",
    "REFUSAL_BELOW_MINIMUM",
    "REFUSAL_DAILY_CAP",
    "REFUSAL_INSUFFICIENT_BALANCE",
    "balance_microusd",
    "charge_microusd",
    "epoch_payout",
    "gamma_units_to_microusd",
    "lane_cap_microusd",
    "lane_weight_fractions",
    "miner_reward_microusd",
    "minimum_charge_microusd",
    "order_hold_microusd",
    "order_miner_rewards",
    "order_refusal",
    "reservation_allowed",
    "rewardable_spend_by_hotkey",
    "spend_ceiling_microusd",
]
