"""Policy record and money arithmetic for Gamma lead orders."""

from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction

import pytest

from lab_arena import signing
from lab_arena.lead_orders import money
from lab_arena.lead_orders.policy import (
    DEFAULT_POLICY_BODY,
    PolicyError,
    build_policy_document,
    load_policy,
    load_policy_file,
    sign_policy,
    unsigned_policy,
)

USD = 1_000_000
POLICY = unsigned_policy()
MINER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
OTHER = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


def body(**changes):
    out = json.loads(json.dumps(DEFAULT_POLICY_BODY))
    out.update(changes)
    return out


# -- policy ------------------------------------------------------------------


def test_default_policy_matches_contract_start_values():
    assert POLICY.price_per_lead_microusd == 200_000
    assert POLICY.miner_premium == Decimal("0.25")
    assert POLICY.min_leads_per_order == 10
    assert POLICY.min_charge_ratio_ppm == 400_000
    assert POLICY.spend_ceiling_per_lead_microusd == 500_000
    assert POLICY.lane_emission_cap_ppm == 1_000_000
    assert POLICY.ranking_window_epochs == 140
    assert POLICY.daily_leads_per_hotkey == 500
    assert POLICY.order_deadline_minutes == 60


@pytest.mark.parametrize(
    "changes",
    [
        {"min_charge_ratio": "0.35"},  # == premium 0.25 + Deepline 0.10
        {"min_charge_ratio": "0.30"},
        {"miner_premium": "0.30"},
        {"max_provider_discounts": {"deepline": "0.20", "openrouter": "0"}},
        {"max_provider_discounts": {"deepline": "0.10"}},
        {"spend_ceiling_per_lead_usd": "0.60"},  # 0.4 x 0.6 > 0.20 hold per lead
        {"price_per_lead_usd": 0.2},  # float, not a decimal string
        {"price_per_lead_usd": "0.0000001"},
        {"price_per_lead_usd": "0"},
        {"lane_emission_cap": "1.5"},
        {"min_leads_per_order": 0},
        {"daily_leads_per_hotkey": 5},
        {"version": True},
        {"schema_version": "other"},
        {"miner_terms_sha256": "A" * 64},
        {"miner_terms_sha256": "ab"},
    ],
)
def test_unsafe_or_malformed_policy_is_refused(changes):
    with pytest.raises(PolicyError):
        build_policy_document(body(**changes))


def test_default_policy_names_the_published_miner_terms_text():
    # Editing the terms without a new version and hash would let a miner's
    # signature point at words they never saw.
    from lab_arena.lead_orders.service import MINER_TERMS_PATH
    import hashlib
    text = MINER_TERMS_PATH.read_bytes()
    assert hashlib.sha256(text).hexdigest() == DEFAULT_POLICY_BODY["miner_terms_sha256"]
    assert ("`%s`" % DEFAULT_POLICY_BODY["miner_terms_version"]).encode() in text


def test_unknown_field_is_refused():
    with pytest.raises(PolicyError):
        build_policy_document(body(extra=1))


def test_signed_policy_round_trips_and_detects_tampering(tmp_path):
    signer = signing.LocalSigner.generate()
    document = sign_policy(signer, body(version=7))
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(document))
    loaded = load_policy_file(path, public_key_der=signer.public_key_der, expected_public_key_hash=signer.public_key_hash)
    assert loaded.version == 7 and loaded.policy_hash == document["policy_hash"]

    tampered = dict(document, price_per_lead_usd="0.10")
    with pytest.raises(PolicyError):
        load_policy(tampered, public_key_der=signer.public_key_der, expected_public_key_hash=signer.public_key_hash)
    other = signing.LocalSigner.generate()
    with pytest.raises(PolicyError):
        load_policy(document, public_key_der=other.public_key_der, expected_public_key_hash=other.public_key_hash)
    unsigned = build_policy_document(body())
    with pytest.raises(PolicyError):
        load_policy(unsigned, public_key_der=signer.public_key_der, expected_public_key_hash=signer.public_key_hash)


def test_signed_unsafe_policy_is_still_refused():
    signer = signing.LocalSigner.generate()
    document = sign_policy(signer, body())
    # Re-sign an unsafe body by bypassing build validation: load must refuse it.
    unsafe = dict(document, min_charge_ratio="0.30")
    from lab_arena.contracts import document_hash

    unsafe_body = {k: v for k, v in unsafe.items() if k not in ("policy_hash", "signature")}
    unsafe["policy_hash"] = document_hash(unsafe_body)
    unsafe = signing.sign_document(signer, {k: v for k, v in unsafe.items() if k != "signature"}, hash_field="policy_hash")
    with pytest.raises(PolicyError):
        load_policy(unsafe, public_key_der=signer.public_key_der, expected_public_key_hash=signer.public_key_hash)


# -- hold, refusals, balance ----------------------------------------------------


def test_hold_ceiling_and_refusals():
    assert money.order_hold_microusd(POLICY, 10) == 2 * USD
    assert money.spend_ceiling_microusd(POLICY, 10) == 5 * USD
    assert money.order_refusal(POLICY, leads=9, leads_ordered_today=0, balance_microusd=10 * USD) == "below_minimum"
    assert money.order_refusal(POLICY, leads=10, leads_ordered_today=495, balance_microusd=10 * USD) == "daily_cap"
    assert money.order_refusal(POLICY, leads=10, leads_ordered_today=490, balance_microusd=10 * USD) is None
    assert money.order_refusal(POLICY, leads=10, leads_ordered_today=0, balance_microusd=2 * USD - 1) == "insufficient_balance"
    assert money.order_refusal(POLICY, leads=10, leads_ordered_today=0, balance_microusd=2 * USD) is None
    # Staff and shadow orders do not use Gamma.
    assert money.order_refusal(POLICY, leads=10, leads_ordered_today=0, balance_microusd=None) is None


def test_balance_converts_gamma_units_and_subtracts_charges_and_holds():
    # 3 USD of Gamma = 3 x 10^9 units.
    assert money.balance_microusd(3_000_000_000, 1 * USD, 500_000) == 1_500_000
    assert money.gamma_units_to_microusd(1_999) == 1  # rounds down


def test_reservation_ceiling():
    assert money.reservation_allowed(5 * USD, 4 * USD, 1 * USD)
    assert not money.reservation_allowed(5 * USD, 4 * USD, 1 * USD + 1)


# -- charge ---------------------------------------------------------------------


def test_charge_formula():
    hold = 2 * USD
    assert money.charge_microusd(POLICY, hold_microusd=hold, delivered=10, spend_microusd=1 * USD) == 2 * USD
    assert money.charge_microusd(POLICY, hold_microusd=hold, delivered=6, spend_microusd=1 * USD) == 1_200_000
    # Little delivered, real spend: the minimum charge applies.
    assert money.charge_microusd(POLICY, hold_microusd=hold, delivered=1, spend_microusd=3 * USD) == 1_200_000
    # Never above the hold.
    assert money.charge_microusd(POLICY, hold_microusd=hold, delivered=0, spend_microusd=9 * USD) == hold
    # Minimum charge rounds up.
    assert money.minimum_charge_microusd(POLICY, 1) == 1


def test_worked_example_lead_costs_ten_cents_ten_leads_ordered():
    """docs/gamma-lead-orders.md: a lead costs $0.10 to find, 10 leads ordered."""

    hold = money.order_hold_microusd(POLICY, 10)
    ceiling = money.spend_ceiling_microusd(POLICY, 10)
    assert (hold, ceiling) == (2 * USD, 5 * USD)
    # All 10 found for $1.00 of provider spend on one miner key.
    calls = [(MINER, 100_000)] * 10
    spend = sum(cost for _, cost in calls)
    charge = money.charge_microusd(POLICY, hold_microusd=hold, delivered=10, spend_microusd=spend)
    rewards = money.order_miner_rewards(POLICY, calls, ceiling_microusd=ceiling)
    assert charge == 2 * USD  # buyer pays $0.20 a lead
    assert rewards == {MINER: 1_250_000}  # miner repaid $1.00 x 1.25
    assert charge - rewards[MINER] == 750_000  # Gamma burned exceeds emissions paid
    # Only 6 found for the same spend: charged for 6.
    charge6 = money.charge_microusd(POLICY, hold_microusd=hold, delivered=6, spend_microusd=spend)
    assert charge6 == 1_200_000
    # Remaining hold is released.
    assert hold - charge6 == 800_000


@pytest.mark.parametrize("spend_usd_micro", [1, 10_000, 1 * USD, 4_999_999, 5 * USD, 7 * USD, 50 * USD])
@pytest.mark.parametrize("leads", [10, 37, 500])
def test_fake_order_on_own_key_always_loses_money(spend_usd_micro, leads):
    """A miner orders from their own key and delivers nothing.

    Net = reward - provider bill at the largest discount - charge. It must be
    negative for every spend, including spend past the ceiling.
    """

    hold = money.order_hold_microusd(POLICY, leads)
    ceiling = money.spend_ceiling_microusd(POLICY, leads)
    calls = [(MINER, spend_usd_micro)]
    reward = money.order_miner_rewards(POLICY, calls, ceiling_microusd=ceiling).get(MINER, 0)
    discount = POLICY.max_provider_discount
    provider_bill = Fraction(spend_usd_micro) * (1 - Fraction(str(discount)))
    charge = money.charge_microusd(POLICY, hold_microusd=hold, delivered=0, spend_microusd=spend_usd_micro)
    assert reward - provider_bill - charge < 0


def test_fake_order_with_mixed_keys_still_loses():
    hold = money.order_hold_microusd(POLICY, 10)
    ceiling = money.spend_ceiling_microusd(POLICY, 10)
    calls = [(MINER, 3 * USD), (None, 1 * USD), (MINER, 3 * USD)]  # overshoots ceiling
    reward = money.order_miner_rewards(POLICY, calls, ceiling_microusd=ceiling)[MINER]
    charge = money.charge_microusd(POLICY, hold_microusd=hold, delivered=0, spend_microusd=7 * USD)
    assert reward - Fraction(6 * USD) * Fraction(9, 10) - charge < 0


def test_rewards_skip_leadpoet_and_scale_past_ceiling():
    calls = [(MINER, 3 * USD), (OTHER, 1 * USD), (None, 2 * USD)]
    spend = money.rewardable_spend_by_hotkey(calls, ceiling_microusd=5 * USD)
    assert spend == {MINER: 2_500_000, OTHER: 833_333}
    rewards = money.order_miner_rewards(POLICY, calls, ceiling_microusd=5 * USD)
    assert rewards == {MINER: 3_125_000, OTHER: 1_041_666}
    assert money.order_miner_rewards(POLICY, [(None, USD)], ceiling_microusd=5 * USD) == {}


# -- epoch payout -------------------------------------------------------------------


def entry(order, hotkey, epoch, owed, paid=0, seq=0):
    return money.OwedEntry(order, hotkey, epoch, seq, owed, paid)


def test_epoch_payout_pays_oldest_first_and_carries_forward():
    entries = [
        entry("o3", MINER, 12, 4 * USD),
        entry("o1", OTHER, 10, 2 * USD, paid=1 * USD),
        entry("o2", MINER, 11, 3 * USD),
        entry("o4", OTHER, 14, 9 * USD),  # settled after the paying epoch
    ]
    payout = money.epoch_payout(12, entries, cap_microusd=3 * USD)
    assert payout.payments == (("o1", OTHER, 1 * USD, None), ("o2", MINER, 2 * USD, None))
    assert payout.paid_by_hotkey == {OTHER: 1 * USD, MINER: 2 * USD}
    assert payout.paid_microusd == 3 * USD
    assert payout.carry_forward_microusd == 1 * USD + 4 * USD


def test_epoch_payout_under_cap_pays_everything():
    payout = money.epoch_payout(5, [entry("o1", MINER, 5, 700)], cap_microusd=10 * USD)
    assert payout.paid_microusd == 700 and payout.carry_forward_microusd == 0


def test_epoch_payout_refuses_bad_entries():
    with pytest.raises(money.MoneyError):
        money.epoch_payout(5, [entry("o1", MINER, 5, 1), entry("o1", MINER, 5, 1)], cap_microusd=1)
    with pytest.raises(money.MoneyError):
        money.epoch_payout(5, [entry("o1", MINER, 5, 1, paid=2)], cap_microusd=1)


def test_lane_cap_and_weight_fractions():
    emission = 100 * USD
    assert money.lane_cap_microusd(POLICY, emission) == emission
    capped = unsigned_policy(dict(DEFAULT_POLICY_BODY, lane_emission_cap="0.30"))
    assert money.lane_cap_microusd(capped, emission) == 30 * USD
    fractions = money.lane_weight_fractions({MINER: 15 * USD, OTHER: 15 * USD}, emission)
    assert sum(fractions.values()) == Fraction(3, 10)
    assert money.lane_weight_fractions({}, 0) == {}
    with pytest.raises(money.MoneyError):
        money.lane_weight_fractions({MINER: 1}, 0)
