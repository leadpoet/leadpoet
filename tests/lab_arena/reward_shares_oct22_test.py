"""October 22 changes signed slot shares, with each round's date as authority."""

from copy import deepcopy
from datetime import date, datetime, timezone
from fractions import Fraction
import threading
from types import SimpleNamespace

import pytest

from lab_arena import contracts, rewards, signing
from lab_arena.service import ArenaService, ServiceError
from leadpoet_canonical import arena_weights, lab_arena_rewards as kernel
from tests.lab_arena.cumulative_reward_pot_test import (
    ALICE, BOB, CAROL, BURN, HOTKEYS, _basis, _vector, _win,
)
from tests.lab_arena.slot_decay_postgres_test import database, state, _winner
from tests.lab_arena.slot_decay_kernel_test import _state
from tests.lab_arena.reward_slots_postgres_test import _basis as _pg_basis


@pytest.mark.parametrize("day,pool,shares", [
    (date(2026, 10, 21), 30, [50, 30, 20]),
    (date(2026, 10, 22), 50, [40, 30, 30]),
    (date(2026, 11, 1), 50, [40, 30, 30]),
])
def test_policy_changes_at_evaluation_date(day, pool, shares):
    policy = rewards.reward_slot_policy_document(day)
    assert policy["pool_percent"] == pool
    assert [tier["allocation_percent"] for tier in policy["tiers"]] == shares
    assert policy["assignment_mode"] == "all_qualifying"
    assert policy["decay"] == {"epochs_per_halving": 140, "max_halvings": 4}


def _activation_service(day, *, activated=False):
    row = {
        "round_id": "arena-" + day, "evaluation_date": day,
        "status": "published", "reward_activated_at": "done" if activated else None,
        "effective_reward_epoch": 101, "baseline_promoted_at": "done",
        "configuration_doc": {
            "mode": "live", "rewards_enabled": True, "baseline_hotkey": BURN,
            "reward_constants": rewards.reward_constants_document(),
        },
        "publication_doc": {"published_at": "2026-10-25T12:00:00Z",
                            "king_decision": {"outcome": "crowned", "king_hotkey": ALICE}},
    }
    captured = {}
    def snapshot(round_id, policy):
        captured["snapshot"] = (round_id, deepcopy(policy))
        return [None, None, None]
    def activate(round_id, basis, key):
        captured.update(round_id=round_id, basis=basis, key=key)
        return {"status": "activated"}
    service = object.__new__(ArenaService)
    service._store = SimpleNamespace(
        published_reward_bases=lambda **_kwargs: [],
        reward_slot_snapshot=snapshot, activate_reward=activate,
    )
    service._signer = signing.LocalSigner.generate()
    service._signer_lock = threading.Lock()
    service._config = SimpleNamespace(
        network_name="finney", netuid=71, reward_signer_factory=None,
        chain=SimpleNamespace(current_settlement_epoch=lambda: 100),
        clock=lambda: datetime(2026, 10, 25, tzinfo=timezone.utc),
    )
    service._round = lambda _rid: row
    service._superseded_by_published_day = lambda _row: False
    service._baseline_final_ready = lambda _row: True
    service._pending_promotion_blocks = lambda: False
    return service, row, captured


@pytest.mark.parametrize("day,pool", [("2026-10-21", 30), ("2026-10-22", 50)])
def test_service_signs_same_dated_policy_as_snapshot_even_when_activated_late(day, pool):
    service, row, captured = _activation_service(day)
    assert service.activate_reward(row["round_id"]) == {"status": "activated"}
    basis = captured["basis"]
    assert captured["snapshot"] == (row["round_id"], basis["slot_policy"])
    assert basis["slot_policy"]["pool_percent"] == pool
    assert basis["effective_reward_epoch"] == 101
    assert contracts.validate_reward_basis(basis) == basis
    assert signing.verify_document_signature(
        basis, public_key_der=service._signer.public_key_der,
        hash_field="reward_basis_hash",
        expected_public_key_hash=service._signer.public_key_hash,
    ) == basis["reward_basis_hash"]


def test_existing_activation_returns_immutable_epoch_without_new_signing():
    service, row, captured = _activation_service("2026-10-21", activated=True)
    assert service.activate_reward(row["round_id"]) == {
        "status": "existing", "effective_reward_epoch": 101,
    }
    assert captured == {}


@pytest.mark.parametrize("bad", ["", "2026-10-2", "2026-13-22", None])
def test_service_rejects_invalid_evaluation_date(bad):
    service, row, captured = _activation_service("2026-10-22")
    row["evaluation_date"] = bad
    with pytest.raises(ServiceError, match="round_evaluation_date_invalid"):
        service.activate_reward(row["round_id"])
    assert captured == {}


def test_signed_weights_pay_twenty_fifteen_fifteen_and_preserve_old_basis():
    slots = [_win(ALICE, 10, 1, "high"), _win(BOB, 5, 2, "mid"),
             _win(CAROL, 1, 3, "low")]
    old = _basis(slots)
    old_copy = deepcopy(old)
    policy = rewards.reward_slot_policy_document(date(2026, 10, 22))
    new = _basis(slots, policy=policy)
    assert kernel.slot_allocations(new, 100, HOTKEYS) == {
        ALICE: Fraction(1, 5), BOB: Fraction(3, 20), CAROL: Fraction(3, 20),
    }
    vector = _vector(new)
    assert vector["champion_share_ppb"] == 500_000_000
    assert vector["burned_residual_ppb"] == 500_000_000
    assert vector["sparse_weights_u16"] == [65535, 26214, 19661, 19661]
    assert old == old_copy
    assert _vector(old)["champion_share_ppb"] == 300_000_000
    policy["tiers"][0]["allocation_percent"] = 1
    assert new["slot_policy"]["tiers"][0]["allocation_percent"] == 40


def test_old_slot_age_funding_factor_and_missing_uid_keep_their_effect():
    policy = rewards.reward_slot_policy_document(date(2026, 10, 22))
    slots = [_win(ALICE, 10, 1, "high"), _win(BOB, 5, 2, "mid"), None]
    for slot in slots[:2]:
        slot["start_epoch"] = 0
    # One completed halving applies independently from the funding penalty.
    basis = rewards.reward_basis_document(
        round_id="arena-2026-10-22", published_at="2026-10-23T00:00:00Z",
        finalized_epoch=139, king_outcome="crowned", king_hotkey=BOB,
        champion_reward_factor_ppm=500_000, slot_policy=policy, reward_slots=slots,
    )
    assert kernel.slot_allocations(basis, 140, HOTKEYS) == {
        ALICE: Fraction(1, 10), BOB: Fraction(3, 80),
    }
    assert kernel.slot_allocations(basis, 140, [BURN, ALICE]) == {ALICE: Fraction(1, 10)}
    assert [slot["start_epoch"] for slot in basis["reward_slots"][:2]] == [0, 0]
    accepted, _ = _state(basis, 140)
    vector = arena_weights.derive_arena_weights(accepted, [BURN, ALICE])
    assert vector["champion_share_ppb"] == 100_000_000
    assert vector["burned_residual_ppb"] == 900_000_000


@pytest.mark.parametrize("delta,slots,share", [
    (10, [True, True, True], Fraction(1, 2)),
    (5, [False, True, True], Fraction(3, 10)),
    (1, [False, False, True], Fraction(3, 20)),
])
def test_all_qualifying_achievements_keep_cumulative_shares(delta, slots, share):
    winner = _win(ALICE, delta, 1, "winner")
    basis = _basis([winner if occupied else None for occupied in slots],
                   policy=rewards.reward_slot_policy_document(date(2026, 10, 22)))
    assert kernel.slot_allocations(basis, 100, HOTKEYS) == {ALICE: share}


def test_postgres_accepts_new_policy_and_persists_exact_slots_and_weights(state):
    store, control, chain = state
    _winner(control, chain, "high", day="2026-10-20", delta=10, hotkey=ALICE, epoch=100)
    _winner(control, chain, "mid", day="2026-10-21", delta=5, hotkey=BOB, epoch=110)
    current = _winner(control, chain, "low", day="2026-10-22", delta=1,
                      hotkey=CAROL, activated=False)
    policy = rewards.reward_slot_policy_document(date(2026, 10, 22))
    slots = store.reward_slot_snapshot(current, policy)
    assert [slot["start_epoch"] for slot in slots] == [100, 110, None]
    basis, key = _pg_basis(current, slots, policy=policy, hotkey=CAROL, epoch=120)
    assert store.activate_reward(current, basis, key)["status"] == "activated"
    assert store.activate_reward(current, basis, key)["status"] == "existing"
    saved = store.get_round(current)["reward_basis_doc"]
    assert saved == basis
    assert kernel.slot_allocations(saved, 120, HOTKEYS) == {
        ALICE: Fraction(1, 5), BOB: Fraction(3, 20), CAROL: Fraction(3, 20),
    }
    accepted, _ = _state(saved, 120)
    vector = arena_weights.derive_arena_weights(accepted, HOTKEYS)
    assert vector["champion_share_ppb"] == 500_000_000
    assert vector["burned_residual_ppb"] == 500_000_000
