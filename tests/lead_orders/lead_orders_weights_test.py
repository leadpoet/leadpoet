"""Lead-order rewards ride the collection weight state: 80/20 with spill-over both ways."""

from __future__ import annotations

import random
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
from bittensor_wallet import Keypair

from lab_arena import collection_rewards, signing, weight_state
from lab_arena.lead_orders import money, weights
from lab_arena.store import ArenaStoreError
from leadpoet_canonical.arena_weights import (
    COLLECTION_BASIS_SCHEMA,
    PARTS_PER_BILLION,
    derive_arena_weights,
    verify_accepted_weight_state_signature,
)

A, B, C, BURN = [Keypair.create_from_uri("//LeadWeights" + x).ss58_address for x in ("A", "B", "C", "Burn")]
KEYS = [A, B, C, BURN]
SIGNER = signing.LocalSigner.generate()
EPOCH, BLOCK = 26600, 50
GENESIS = "2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
HASH = "sha256:" + "d" * 64
USD = 1_000_000


def valuation(block=BLOCK, emission=USD):
    return {"block": block, "block_hash": "0x" + "c" * 64, "miner_alpha_rao": 10**9, "tao_rao_per_alpha": 10**9,
            "tao_price_microusd": emission, "price_updated_at": 200, "observed_at": 201, "emission_microusd": emission}


def basis(quotes, *, valued):
    return {"schema_version": COLLECTION_BASIS_SCHEMA, "source_epoch": EPOCH - 1, "job_count": len(quotes),
            "job_set_sha256": "b" * 64, "quotes_microusd": dict(quotes), "valuation": valuation() if valued else None}


def state(quotes, lead=None, *, emission=USD, **overrides):
    reward_basis = basis(quotes, valued=bool(quotes) or lead is not None)
    if reward_basis["valuation"]:
        reward_basis["valuation"] = valuation(emission=emission)
    return weight_state.build_accepted_weight_state(
        SIGNER, network="finney", genesis_hash=GENESIS, netuid=71, epoch=EPOCH, valid_from_block=BLOCK,
        valid_until_block=409, burn_hotkey=BURN, issued_at="2026-11-23T00:01:00Z",
        lead_orders=None if lead is None else {"payout_hash": HASH, "paid_microusd": dict(lead)},
        **{"reward_basis": reward_basis, **overrides},
    )


def shares(document, keys=KEYS):
    got = derive_arena_weights(document, keys)
    return (got.get("collection_share_ppb"), got.get("lead_order_share_ppb"), got["burned_residual_ppb"])


def ppb(numerator, denominator=10):
    return PARTS_PER_BILLION * numerator // denominator


@pytest.mark.parametrize("quotes,lead,expected", [
    # Both inside their share: each is paid in full, the rest burns.
    ({A: 500_000}, {B: 100_000}, (ppb(5), ppb(1), ppb(4))),
    # Collection over 80% takes the 10% lead orders leave.
    ({A: 1_200_000}, {B: 100_000}, (ppb(9), ppb(1), 0)),
    # Lead orders over 20% take what collection leaves.
    ({A: 300_000}, {B: 600_000}, (ppb(3), ppb(6), ppb(1))),
    ({A: 300_000}, {B: 900_000}, (ppb(3), ppb(7), 0)),
    # Both over: exactly 80 / 20.
    ({A: 2_000_000}, {B: 1_000_000}, (ppb(8), ppb(2), 0)),
    # No collection work: lead orders may use the whole epoch.
    ({}, {B: 400_000}, (0, ppb(4), ppb(6))),
    ({}, {B: 1_500_000}, (0, ppb(10), 0)),
])
def test_collection_is_guaranteed_80_percent_lead_orders_20_and_each_uses_the_others_unused(quotes, lead, expected):
    document = state(quotes, lead)
    verify_accepted_weight_state_signature(
        document, public_key_der=SIGNER.public_key_der, expected_public_key_hash=SIGNER.public_key_hash)
    assert shares(document) == expected


def test_a_state_without_lead_orders_keeps_the_exact_collection_vector():
    for quotes in ({}, {A: 250_000, B: 250_000}, {A: 3_000_000, B: 1_000_000}):
        document = state(quotes)
        assert "lead_orders" not in document
        got = derive_arena_weights(document, KEYS)
        assert "lead_order_share_ppb" not in got
        budget = max(USD, sum(quotes.values()))
        assert got["collection_share_ppb"] == PARTS_PER_BILLION * sum(quotes.values()) // budget


def test_one_miner_paid_in_both_lanes_gets_the_sum_and_an_unregistered_one_burns():
    got = derive_arena_weights(state({A: 400_000}, {A: 100_000, B: 100_000}), KEYS)
    assert dict(zip(got["sparse_uids"], got["sparse_weights_u16"])) == {0: 65535, 1: 13107, 3: 52428}
    # B left the subnet: its share burns, nobody else gains.
    assert shares(state({A: 400_000}, {B: 100_000}), [A, C, BURN]) == (ppb(4), 0, ppb(6))


def test_lead_order_field_is_validated_and_signed():
    for lead in ({}, {BURN: 5}, {A: 0}, {A: True}, {"not-a-hotkey": 5}):
        with pytest.raises(ValueError):
            derive_arena_weights(state({A: 100}, lead), KEYS)
    good = state({A: 100}, {B: 5})
    for patch in ({"payout_hash": "d" * 64}, {"paid_microusd": {B: 6}}, {"paid_microusd": {B: 5}, "extra": 1}):
        altered = deepcopy(good)
        altered["lead_orders"] = dict(altered["lead_orders"], **patch)
        with pytest.raises(ValueError):
            derive_arena_weights(altered, KEYS)
    # Paying lead orders needs the epoch's valuation, even with no collection job.
    with pytest.raises(ValueError, match="valuation"):
        derive_arena_weights(state({}, {B: 5}, reward_basis=basis({}, valued=False)), KEYS)
    # And an empty epoch still may not carry one.
    with pytest.raises(ValueError, match="valuation"):
        derive_arena_weights(state({}, reward_basis=basis({}, valued=True)), KEYS)


def test_a_payout_capped_by_the_gateway_allowance_is_never_scaled_by_validators():
    rng = random.Random(71)
    for _ in range(300):
        emission = rng.choice([1, 7, 999_999, USD, 123_456_789])
        collection = rng.choice([0, emission // 3, emission * 4 // 5, emission, emission * 3])
        owed = rng.choice([1, emission // 7 + 1, emission // 5, emission, emission * 2])
        allowance = weights.lead_order_allowance_microusd(emission, collection)
        assert allowance >= emission // 5
        paid = min(owed, allowance)
        if not paid:
            continue
        document = state({A: collection} if collection else {}, {B: paid}, emission=emission)
        got = derive_arena_weights(document, KEYS)
        assert got["lead_order_share_ppb"] == PARTS_PER_BILLION * paid // emission
        # Collection keeps its full claim, or at least its 80%.
        asked = PARTS_PER_BILLION * collection // emission
        assert got["collection_share_ppb"] >= min(asked, ppb(8)) - 1


def test_registered_entries_keep_only_miners_that_can_be_weighted_now():
    entries = [money.OwedEntry("o1", A, 1, 1, 10), money.OwedEntry("o2", "5" + "Z" * 47, 1, 2, 10)]
    assert [e.order_id for e in weights.registered_entries(entries, KEYS)] == ["o1"]


# --- gateway half: one price read, lead orders never hold up collection -----------------------


class FakeStore:
    def __init__(self, quotes, *, refuse=False):
        self.quotes, self.refuse, self.rows = quotes, refuse, {}

    def get_weight_state(self, _network, _netuid, epoch):
        return self.rows.get(epoch)

    def collection_reward_quotes(self, _epoch):
        return {k: v for k, v in basis(self.quotes, valued=False).items() if k != "valuation"}

    def publish_weight_state(self, _network, _netuid, epoch, state_hash, document):
        if self.refuse:
            raise ArenaStoreError("weight state differs from the lead order payout")
        self.rows[epoch] = {"state_doc": document, "state_hash": state_hash}
        return {"state": document, "state_hash": state_hash}


class FakeLane:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def settle(self, epoch, *, collection_microusd, valuation):
        self.calls.append((epoch, collection_microusd))
        if self.error:
            raise self.error
        if self.result:
            valuation()  # a paying epoch reads the price
        return self.result


def issue(store, lane, reads):
    def source(block):
        reads.append(block)
        return valuation(block)

    chain = NS(accepted_weight_epoch_scope=lambda: {
        "epoch": EPOCH, "genesis_hash": "0x" + GENESIS, "valid_from_block": BLOCK, "valid_until_block": 409})
    return collection_rewards.epoch_state(store, chain, SIGNER, epoch=EPOCH, burn_hotkey=BURN,
                                          valuation_source=source, lead_orders=lane)


def test_epoch_state_signs_the_lead_payout_with_one_price_read():
    paid = {"valuation": valuation(), "lead_orders": {"payout_hash": HASH, "paid_microusd": {B: 100_000}}}
    reads, lane = [], FakeLane(paid)
    document = issue(FakeStore({A: 500_000}), lane, reads)
    assert lane.calls == [(EPOCH, 500_000)] and reads == [BLOCK]
    assert document["lead_orders"] == paid["lead_orders"]
    assert shares(document) == (ppb(5), ppb(1), ppb(4))
    # No collection job: the lead payout still gets a valuation and its weight.
    reads = []
    assert shares(issue(FakeStore({}), FakeLane(paid), reads)) == (0, ppb(1), ppb(9))
    # Nothing owed: the state is the collection state validators already accept.
    for quotes, expected_reads in (({A: 500_000}, [BLOCK]), ({}, [])):
        reads = []
        document = issue(FakeStore(quotes), FakeLane(None), reads)
        assert "lead_orders" not in document and reads == expected_reads


def test_a_failing_lead_payout_never_holds_up_collection_pay():
    reads = []
    document = issue(FakeStore({A: 500_000}), FakeLane(error=RuntimeError("provider down")), reads)
    assert "lead_orders" not in document and shares(document) == (ppb(5), None, ppb(5))
    # But when the database says a payout was recorded, no state is issued without it.
    with pytest.raises(ArenaStoreError):
        issue(FakeStore({A: 500_000}, refuse=True), FakeLane(error=RuntimeError("lost reply")), [])


class FakeService:
    def __init__(self, mode, document=None):
        self.config = NS(computes_rewards=mode in ("shadow", "live"), weights_rewards=mode == "live")
        self.document, self.calls = document, []

    def run_epoch(self, epoch, **kwargs):
        self.calls.append((epoch, kwargs))
        return self.document


def test_lane_pays_only_in_live_mode_and_caps_at_the_allowance():
    for mode, ran in (("off", 0), ("staff", 0), ("shadow", 1)):
        service = FakeService(mode, {"mode": "shadow", "paid_microusd": 0, "payments": []})
        lane = weights.LeadOrderLane(service, registered_hotkeys=lambda: KEYS)
        assert lane.settle(EPOCH, collection_microusd=0, valuation=valuation) is None and len(service.calls) == ran
    document = {
        "mode": "live", "paid_microusd": 300_000, "payout_hash": HASH,
        "emission_valuation": dict(valuation(), tao_price_source=weights.TAO_PRICE_SOURCE),
        "payments": [{"order_id": "o1", "miner_hotkey": B, "paid_microusd": 100_000, "call_id": None},
                     {"order_id": "o2", "miner_hotkey": B, "paid_microusd": 150_000, "call_id": "c1"},
                     {"order_id": "o3", "miner_hotkey": A, "paid_microusd": 50_000, "call_id": None}],
    }
    service = FakeService("live", document)
    lane = weights.LeadOrderLane(service, registered_hotkeys=lambda: KEYS)
    settled = lane.settle(EPOCH, collection_microusd=900_000, valuation=valuation)
    assert settled == {"valuation": valuation(),
                       "lead_orders": {"payout_hash": HASH, "paid_microusd": {B: 250_000, A: 50_000}}}
    (_, kwargs), = service.calls
    assert kwargs["registered_hotkeys"] == KEYS
    assert kwargs["payable_cap_microusd"](USD) == 200_000  # collection asks 90%: lead orders keep their 20%
    assert kwargs["valuation"]()["tao_price_source"] == weights.TAO_PRICE_SOURCE
    # A live epoch that paid nothing adds nothing to the state.
    idle = weights.LeadOrderLane(FakeService("live", dict(document, paid_microusd=0, payments=[])),
                                 registered_hotkeys=lambda: KEYS)
    assert idle.settle(EPOCH, collection_microusd=0, valuation=valuation) is None
