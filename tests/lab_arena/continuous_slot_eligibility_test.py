"""An accepted v2 state's signed rule governs slot age across idle epochs."""

from copy import deepcopy
from datetime import date, datetime, timezone
from fractions import Fraction
from types import SimpleNamespace

import pytest

from lab_arena import rewards, signing, weight_state
from leadpoet_canonical import arena_weights, lab_arena_rewards as kernel
from leadpoet_canonical.lab_arena_rewards import sha256_json
from lab_arena.service import ArenaService, ServiceError
from tests.lab_arena.slot_decay_kernel_test import ALICE, BOB, BURN, HOTKEYS, _basis, _state, _win
from tests.lab_arena.slot_decay_postgres_test import database, state
import json


def _held_basis(*, old_rates=False):
    day = date(2026, 10, 21) if old_rates else date(2026, 10, 22)
    policy = rewards.reward_slot_policy_document(day)
    winner = _win(ALICE, delta=11, start=25865)
    return _basis([deepcopy(winner) for _ in range(3)], epoch=25865, policy=policy)


@pytest.mark.parametrize("age,old_rates,share", [
    (45, False, 500_000_000),
    (46, False, 500_000_000),
    (48, False, 500_000_000),
    (48, True, 300_000_000),
    (560, False, 31_250_000),
    (700, True, 18_750_000),
])
def test_signed_continuity_keeps_registered_slots_and_decay_floor(age, old_rates, share):
    basis = _held_basis(old_rates=old_rates)
    epoch = 25865 + age
    state, signer = _state(basis, epoch)
    assert state["slot_eligibility"] == arena_weights.SLOT_ELIGIBILITY_UNTIL_REPLACED
    assert arena_weights.verify_accepted_weight_state_signature(
        state, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    ) == state["state_hash"]
    result = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert result["champion_share_ppb"] == share
    assert result["burned_residual_ppb"] == 1_000_000_000 - share
    assert result["sparse_uids"] == [0, 1]


def test_legacy_v2_state_keeps_exact_45_epoch_limit_and_hash_body():
    basis = _held_basis()
    at_limit, signer = _state(basis, 25910, slot_eligibility=None)
    assert "slot_eligibility" not in at_limit
    assert arena_weights.accepted_weight_state_body(at_limit) == {
        key: value for key, value in at_limit.items() if key not in ("state_hash", "signature")
    }
    assert arena_weights.derive_arena_weights(at_limit, HOTKEYS)["champion_share_ppb"] == 500_000_000
    expired, signer = _state(basis, 25911, slot_eligibility=None)
    assert arena_weights.verify_accepted_weight_state_signature(
        expired, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    ) == expired["state_hash"]
    result = arena_weights.derive_arena_weights(expired, HOTKEYS)
    assert result["champion_share_ppb"] == 0
    assert result["sparse_uids"] == [0]
    assert result["sparse_weights_u16"] == [65535]


def test_new_qualifying_basis_replaces_only_won_slot_after_long_gap():
    old = _held_basis()
    replacement = _win(BOB, delta=2, start=None, name="new-low", round_id="arena-2026-10-23")
    policy = rewards.reward_slot_policy_document(date(2026, 10, 22))
    later = _basis(
        [deepcopy(old["reward_slots"][0]), deepcopy(old["reward_slots"][1]), replacement],
        epoch=25914, policy=policy, king=BOB, round_id="arena-2026-10-23",
    )
    assert kernel.slot_allocations(later, 25914, HOTKEYS, until_replaced=True) == {
        ALICE: Fraction(7, 20), BOB: Fraction(3, 20),
    }
    state, _ = _state(later, 25914)
    result = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert result["champion_share_ppb"] == 500_000_000
    assert result["sparse_uids"] == [0, 1, 2]


def test_late_funding_factor_and_empty_slots_still_burn():
    basis = _held_basis()
    from lab_arena import contracts
    body = {key: value for key, value in basis.items() if key != "reward_basis_hash"}
    body["champion_reward_factor_ppm"] = 500_000
    basis = contracts.finalize_reward_basis(body)
    state, _ = _state(basis, 26425)
    result = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert result["champion_share_ppb"] == 15_625_000
    assert result["burned_residual_ppb"] == 984_375_000
    empty = _basis([None, None, None], epoch=25865,
                   policy=rewards.reward_slot_policy_document(date(2026, 10, 22)))
    state, _ = _state(empty, 25913)
    assert kernel.epoch_eligible(empty, 25913, until_replaced=True) is False
    assert arena_weights.derive_arena_weights(state, HOTKEYS)["champion_share_ppb"] == 0


def test_missing_holder_still_burns_and_policy_tamper_fails():
    basis = _held_basis()
    state, signer = _state(basis, 25913)
    assert arena_weights.derive_arena_weights(state, [BURN, BOB])["champion_share_ppb"] == 0
    tampered = dict(state, slot_eligibility="until_replaced ")
    with pytest.raises(arena_weights.ArenaWeightError, match="slot_eligibility"):
        arena_weights.verify_accepted_weight_state_signature(
            tampered, public_key_der=signer.public_key_der,
            expected_public_key_hash=signer.public_key_hash,
        )
    unsigned_change = dict(state, slot_eligibility="until_replaced")
    unsigned_change["state_hash"] = "sha256:" + "0" * 64
    with pytest.raises(arena_weights.ArenaWeightError, match="state_hash"):
        arena_weights.verify_accepted_weight_state_signature(
            unsigned_change, public_key_der=signer.public_key_der,
            expected_public_key_hash=signer.public_key_hash,
        )
    unknown = dict(state, slot_eligibility="unknown")
    with pytest.raises(arena_weights.ArenaWeightError, match="slot_eligibility"):
        arena_weights.validate_accepted_weight_state(unknown)
    for invalid in (None, True, 1, ["until_replaced"]):
        with pytest.raises(arena_weights.ArenaWeightError, match="slot_eligibility"):
            arena_weights.validate_accepted_weight_state(dict(state, slot_eligibility=invalid))
    with pytest.raises(arena_weights.ArenaWeightError, match="fields"):
        arena_weights.validate_accepted_weight_state(dict(state, extra_policy="yes"))


def test_marker_presence_is_covered_by_state_hash_and_signature():
    basis = _held_basis()
    marked, signer = _state(basis, 25913)
    removed = dict(marked)
    removed.pop("slot_eligibility")
    with pytest.raises(arena_weights.ArenaWeightError, match="state_hash"):
        arena_weights.verify_accepted_weight_state_signature(
            removed, public_key_der=signer.public_key_der,
            expected_public_key_hash=signer.public_key_hash,
        )
    legacy, signer = _state(basis, 25913, slot_eligibility=None)
    added = dict(legacy, slot_eligibility="until_replaced")
    added["state_hash"] = sha256_json(arena_weights.accepted_weight_state_body(added))
    with pytest.raises(arena_weights.ArenaWeightError, match="signature"):
        arena_weights.verify_accepted_weight_state_signature(
            added, public_key_der=signer.public_key_der,
            expected_public_key_hash=signer.public_key_hash,
        )


def _public_service(basis, epoch, existing, signer):
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(
        mode="live", chain=SimpleNamespace(current_settlement_epoch=lambda: epoch),
    )
    service._store = SimpleNamespace(get_weight_state=lambda *_args: existing)
    service._clock = lambda: datetime(2026, 10, 21, tzinfo=timezone.utc)
    service.active_rounds = lambda **_kwargs: []
    service.latest_published_round = lambda **_kwargs: None
    service.public_reward_basis = lambda *_args, **_kwargs: basis
    service._chain_scope = lambda: ("finney", 71)
    service._reward_signer = lambda: signer
    return service


def test_public_status_uses_existing_signed_rule_and_refuses_basis_conflict():
    basis = _held_basis()
    marked, signer = _state(basis, 25913)
    governing = marked["reward_basis"]
    service = _public_service(governing, 25913, {"state_doc": marked}, signer)
    service._reward_signer = lambda: pytest.fail("public status must not initialize the signer")
    assert service.public_current()["epoch_eligible"] is True
    legacy, signer = _state(basis, 25913, slot_eligibility=None)
    assert _public_service(governing, 25913, {"state_doc": legacy}, signer).public_current()["epoch_eligible"] is False
    service = _public_service(governing, 25913, {"state_doc": legacy}, signer)
    assert service.public_weight_state(25913) == {"state": legacy, "lookup_ok": True}
    assert _public_service(governing, 25913, None, signer).public_current()["epoch_eligible"] is None
    different = dict(governing, reward_basis_hash="sha256:" + "f" * 64)
    with pytest.raises(ServiceError, match="accepted_weight_state_reward_conflict"):
        _public_service(different, 25913, {"state_doc": legacy}, signer).public_current()


def test_v1_basis_rejects_slot_policy_marker():
    basis = rewards.reward_basis_document(
        round_id="legacy", published_at="2026-10-01T00:00:00Z",
        finalized_epoch=25864, king_hotkey=ALICE, king_outcome="crowned",
    )
    signer = signing.LocalSigner.generate()
    signed = signing.sign_document(signer, basis, hash_field="reward_basis_hash")
    state = weight_state.build_accepted_weight_state(
        signer, network="finney", genesis_hash="1" * 64, netuid=71,
        epoch=25865, valid_from_block=1000, valid_until_block=1100,
        reward_basis=signed, burn_hotkey=BURN, issued_at=basis["published_at"],
    )
    assert "slot_eligibility" not in state
    with pytest.raises(arena_weights.ArenaWeightError, match="slot_eligibility"):
        arena_weights.validate_accepted_weight_state(dict(state, slot_eligibility="until_replaced"))


def test_existing_database_accepts_signed_optional_field_without_migration(state):
    store, control, _chain = state
    basis = _held_basis()
    basis["reward_slots"][0]["baseline_score"] = 40.0
    basis["reward_slots"][0]["winner_score"] = 51.0
    # Finalize again because the signed reward basis must cover exact JSON numbers.
    from lab_arena import contracts
    basis = contracts.finalize_reward_basis({
        key: value for key, value in basis.items() if key != "reward_basis_hash"
    })
    accepted, signer = _state(basis, 25913)
    result = store.publish_weight_state("finney", 71, 25913, accepted["state_hash"], accepted)
    assert result["status"] == "accepted"
    with control.cursor() as cursor:
        cursor.execute(
            "SELECT state_doc::text FROM public.lab_arena_accepted_weight_states "
            "WHERE network=%s AND netuid=%s AND epoch=%s",
            ("finney", 71, 25913),
        )
        saved = json.loads(cursor.fetchone()[0])
    assert saved["slot_eligibility"] == "until_replaced"
    assert arena_weights.verify_accepted_weight_state_signature(
        saved, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    ) == accepted["state_hash"]
    assert arena_weights.derive_arena_weights(saved, HOTKEYS)["champion_share_ppb"] == 500_000_000
