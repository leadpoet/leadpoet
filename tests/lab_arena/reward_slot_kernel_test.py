"""Signed slot contracts and exact shared validator/signer weight arithmetic."""

import ast
import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from lab_arena import contracts, rewards, signing, weight_state
from lab_arena.weight_signer import ArenaWeightSigner
from leadpoet_canonical import arena_weights, lab_arena_rewards as kernel

BURN, ALICE, BOB, CAROL = ["5" + letter * 47 for letter in "ABCD"]
HOTKEYS = [BURN, ALICE, BOB, CAROL]
POLICY = {
    "assignment_mode": "highest_only",
    "tiers": [
        {"minimum_improvement": 10, "allocation_percent": 50},
        {"minimum_improvement": 5, "allocation_percent": 30},
        {"minimum_improvement": 1, "allocation_percent": 20},
    ],
}


def _achievement(hotkey, improvement, submission="winner", round_id="arena-2026-01-01"):
    return {
        "round_id": round_id, "submission_id": submission, "miner_hotkey": hotkey,
        "baseline_submission_id": "baseline", "baseline_score": 50.1,
        "winner_score": 50.1 + improvement,
    }


def _basis(slots=None, mode="highest_only", factor=1_000_000, outcome="crowned"):
    basis = rewards.reward_basis_document(
        round_id="arena-2026-10-08", published_at="2026-10-09T00:00:00Z",
        finalized_epoch=99, king_hotkey=ALICE if outcome != "no_king" else "",
        king_outcome=outcome, champion_reward_factor_ppm=factor,
    )
    basis.pop("reward_basis_hash")
    basis["schema_version"] = kernel.REWARD_BASIS_V2_SCHEMA_VERSION
    basis["slot_policy"] = deepcopy(POLICY)
    basis["slot_policy"]["assignment_mode"] = mode
    basis["reward_slots"] = slots if slots is not None else [
        _achievement(ALICE, 10, "alice"), _achievement(BOB, 5, "bob"),
        _achievement(CAROL, 1, "carol"),
    ]
    return contracts.finalize_reward_basis(basis)


def _state(basis, epoch=100):
    signer = signing.LocalSigner.generate()
    signed = signing.sign_document(signer, basis, hash_field="reward_basis_hash")
    state = weight_state.build_accepted_weight_state(
        signer, network="finney", genesis_hash="1" * 64, netuid=71,
        epoch=epoch, valid_from_block=1000, valid_until_block=1100,
        reward_basis=signed, burn_hotkey=BURN, issued_at=basis["published_at"],
    )
    return state, signer


def _body(basis):
    return {key: value for key, value in basis.items() if key not in ("reward_basis_hash", "signature")}


def test_three_slots_pay_total_emissions_and_keep_accepted_state_contract():
    basis = _basis()
    assert kernel.slot_allocations(basis, 100, HOTKEYS) == {
        ALICE: Fraction(1, 2), BOB: Fraction(3, 10), CAROL: Fraction(1, 5),
    }
    state, signer = _state(basis)
    assert state["schema_version"] == arena_weights.ACCEPTED_WEIGHT_STATE_SCHEMA_VERSION
    assert arena_weights.verify_accepted_weight_state_signature(
        state, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    ) == state["state_hash"]
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["sparse_uids"] == [1, 2, 3]
    assert vector["sparse_weights_u16"] == [65535, 39321, 26214]
    assert vector["champion_share_ppb"] == 1_000_000_000
    assert vector["burned_residual_ppb"] == 0
    assert set(vector) == {
        "state_hash", "netuid", "epoch", "sparse_uids", "sparse_weights_u16",
        "weights_hash", "champion_share_ppb", "burned_residual_ppb",
    }


def test_unoccupied_and_unregistered_slots_burn_only_their_own_share():
    basis = _basis([_achievement(ALICE, 10), None, _achievement(CAROL, 1)])
    assert kernel.slot_allocations(basis, 100, [BURN, ALICE]) == {ALICE: Fraction(1, 2)}
    state, _ = _state(basis)
    vector = arena_weights.derive_arena_weights(state, [BURN, ALICE])
    assert vector["sparse_uids"] == [0, 1]
    assert vector["sparse_weights_u16"] == [65535, 65535]
    assert vector["burned_residual_ppb"] == 500_000_000


def test_same_hotkey_aggregates_distinct_highest_only_achievements_before_quantization():
    basis = _basis([_achievement(ALICE, 10, "high"), _achievement(ALICE, 5, "mid"),
                    _achievement(ALICE, 1, "low")])
    assert kernel.slot_allocations(basis, 100, HOTKEYS) == {ALICE: Fraction(1)}
    state, _ = _state(basis)
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["sparse_uids"] == [1]
    assert vector["sparse_weights_u16"] == [65535]


def test_all_qualifying_mode_allows_same_achievement_in_all_slots():
    winner = _achievement(ALICE, 10)
    basis = _basis([deepcopy(winner) for _ in range(3)], mode="all_qualifying")
    assert kernel.slot_allocations(basis, 100, HOTKEYS) == {ALICE: Fraction(1)}


def test_holder_equal_to_burn_merges_with_residual_instead_of_duplicating_uid():
    basis = _basis([_achievement(BURN, 10), None, None])
    state, _ = _state(basis)
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["sparse_uids"] == [0]
    assert vector["sparse_weights_u16"] == [65535]
    assert vector["champion_share_ppb"] == 500_000_000
    assert vector["burned_residual_ppb"] == 500_000_000


def test_fallback_factor_applies_only_to_current_king_holder():
    basis = _basis(factor=500_000)
    assert kernel.slot_allocations(basis, 100, HOTKEYS) == {
        ALICE: Fraction(1, 4), BOB: Fraction(3, 10), CAROL: Fraction(1, 5),
    }
    state, _ = _state(basis)
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["champion_share_ppb"] == 750_000_000
    assert vector["burned_residual_ppb"] == 250_000_000
    assert vector["sparse_uids"] == [0, 1, 2, 3]
    assert vector["sparse_weights_u16"] == [54612, 54612, 65535, 43690]


def test_fallback_factor_follows_holder_hotkey_across_multiple_slots():
    basis = _basis([_achievement(ALICE, 10, "high"), _achievement(ALICE, 5, "mid"),
                    _achievement(CAROL, 1, "low")], factor=500_000)
    assert kernel.slot_allocations(basis, 100, HOTKEYS) == {
        ALICE: Fraction(2, 5), CAROL: Fraction(1, 5),
    }


@pytest.mark.parametrize("outcome", ["no_king", "retained_ineligible"])
def test_slot_eligibility_does_not_depend_on_current_king_outcome(outcome):
    basis = _basis(outcome=outcome)
    assert kernel.epoch_eligible(basis, 100)
    assert sum(kernel.slot_allocations(basis, 100, HOTKEYS).values()) == 1


def test_basis_freshness_boundary_and_empty_slots_fail_closed():
    basis = _basis()
    assert kernel.epoch_eligible(basis, 145)
    assert not kernel.epoch_eligible(basis, 146)
    assert kernel.slot_allocations(basis, 146, HOTKEYS) == {}
    with pytest.raises(kernel.LabArenaRewardError, match="not effective"):
        kernel.slot_allocations(basis, 99, HOTKEYS)
    empty = _basis([None, None, None], outcome="no_king")
    assert not kernel.epoch_eligible(empty, 100)
    state, _ = _state(empty)
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["sparse_uids"] == [0]
    assert vector["champion_share_ppb"] == 0
    assert vector["burned_residual_ppb"] == 1_000_000_000


def test_old_holder_and_king_decay_clock_do_not_reduce_fresh_slot_basis():
    basis = _body(_basis())
    basis["king_start_epoch"] = 0
    basis["effective_reward_epoch"] = 1000
    basis = contracts.finalize_reward_basis(basis)
    assert all(slot["round_id"] == "arena-2026-01-01" for slot in basis["reward_slots"])
    assert sum(kernel.slot_allocations(basis, 1000, HOTKEYS).values()) == 1


def test_policy_values_are_generic_and_are_signed_not_hidden_kernel_constants():
    basis = _body(_basis())
    basis["slot_policy"]["tiers"] = [
        {"minimum_improvement": 12, "allocation_percent": 40},
        {"minimum_improvement": 4, "allocation_percent": 35},
        {"minimum_improvement": 2, "allocation_percent": 25},
    ]
    basis["reward_slots"] = [_achievement(ALICE, 12), _achievement(BOB, 4), _achievement(CAROL, 2)]
    basis = contracts.finalize_reward_basis(basis)
    assert kernel.slot_allocations(basis, 100, HOTKEYS) == {
        ALICE: Fraction(2, 5), BOB: Fraction(7, 20), CAROL: Fraction(1, 4),
    }


@pytest.mark.parametrize("slot_index", [0, 1, 2])
@pytest.mark.parametrize("offset", [-0.0000000001, 0, 0.0000000001])
def test_slot_threshold_exact_decimal_boundary(slot_index, offset):
    threshold = POLICY["tiers"][slot_index]["minimum_improvement"]
    slots = [None, None, None]
    slots[slot_index] = _achievement(ALICE, threshold + offset)
    if offset < 0:
        with pytest.raises(contracts.ArenaContractError, match="threshold"):
            _basis(slots)
    else:
        basis = _basis(slots)
        assert kernel.slot_allocations(basis, 100, HOTKEYS) == {
            ALICE: Fraction(POLICY["tiers"][slot_index]["allocation_percent"], 100),
        }


@pytest.mark.parametrize("slot_index,higher_threshold", [(1, 10), (2, 5)])
def test_highest_only_excludes_next_higher_threshold(slot_index, higher_threshold):
    slots = [None, None, None]
    slots[slot_index] = _achievement(ALICE, higher_threshold)
    with pytest.raises(contracts.ArenaContractError, match="higher slot"):
        _basis(slots)
    assert _basis(slots, mode="all_qualifying")


@pytest.mark.parametrize("mutation", [
    "mode", "extra_policy", "two_tiers", "duplicate_threshold", "ascending_threshold",
    "bool_threshold", "zero_threshold", "large_threshold", "extra_tier", "wrong_total", "bool_share",
    "negative_share", "floating_share",
])
def test_slot_policy_rejects_malformed_rules(mutation):
    policy = deepcopy(POLICY)
    if mutation == "mode":
        policy["assignment_mode"] = "locked"
    elif mutation == "extra_policy":
        policy["lock_weeks"] = 4
    elif mutation == "two_tiers":
        policy["tiers"].pop()
    elif mutation == "duplicate_threshold":
        policy["tiers"][1]["minimum_improvement"] = 10
    elif mutation == "ascending_threshold":
        policy["tiers"].reverse()
    elif mutation == "bool_threshold":
        policy["tiers"][0]["minimum_improvement"] = True
    elif mutation == "zero_threshold":
        policy["tiers"][2]["minimum_improvement"] = 0
    elif mutation == "large_threshold":
        policy["tiers"][0]["minimum_improvement"] = 101
    elif mutation == "extra_tier":
        policy["tiers"][0]["lock_epochs"] = 10
    elif mutation == "wrong_total":
        policy["tiers"][0]["allocation_percent"] = 49
    elif mutation == "bool_share":
        policy["tiers"][0]["allocation_percent"] = True
    elif mutation == "negative_share":
        policy["tiers"][0]["allocation_percent"] = -1
    else:
        policy["tiers"][0]["allocation_percent"] = 50.0
    with pytest.raises(kernel.LabArenaRewardError):
        kernel.validate_slot_policy(policy)


@pytest.mark.parametrize("mode", [{}, [], None, True])
def test_slot_policy_mode_rejects_nonscalar_input_as_contract_error(mode):
    policy = deepcopy(POLICY)
    policy["assignment_mode"] = mode
    with pytest.raises(kernel.LabArenaRewardError, match="assignment_mode"):
        kernel.validate_slot_policy(policy)


@pytest.mark.parametrize("field,value", [
    ("round_id", None), ("submission_id", ""), ("miner_hotkey", "invalid"),
    ("baseline_submission_id", "winner"), ("baseline_score", True),
    ("baseline_score", "50"), ("baseline_score", float("nan")),
    ("baseline_score", float("inf")), ("baseline_score", -1),
    ("winner_score", 101), ("winner_score", 10**400),
])
def test_achievement_fields_fail_closed_in_both_contract_readers(field, value):
    basis = _body(_basis([_achievement(ALICE, 10), None, None]))
    basis["reward_slots"][0][field] = value
    with pytest.raises(kernel.LabArenaRewardError):
        kernel.validate_reward_basis(basis)
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_reward_basis(basis)


@pytest.mark.parametrize("mutation", ["two_slots", "dict_slots", "extra_slot_field", "swapped", "missing_policy", "missing_slots"])
def test_signed_slot_order_and_exact_shape_are_required(mutation):
    basis = _body(_basis())
    if mutation == "two_slots":
        basis["reward_slots"].pop()
    elif mutation == "dict_slots":
        basis["reward_slots"] = {}
    elif mutation == "extra_slot_field":
        basis["reward_slots"][0]["score_delta"] = 10
    elif mutation == "swapped":
        basis["reward_slots"].reverse()
    elif mutation == "missing_policy":
        basis.pop("slot_policy")
    else:
        basis.pop("reward_slots")
    with pytest.raises(kernel.LabArenaRewardError):
        kernel.validate_reward_basis(basis)
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_reward_basis(basis)


def test_slot_payload_and_policy_are_covered_by_basis_hash_and_signature():
    basis = _basis()
    state, signer = _state(basis)
    signed = deepcopy(state["reward_basis"])
    signed["reward_slots"][0]["submission_id"] = "tampered"
    with pytest.raises(kernel.LabArenaRewardError, match="hash"):
        kernel.verify_reward_basis_signature(
            signed, public_key_der=signer.public_key_der,
            expected_public_key_hash=signer.public_key_hash,
        )
    signed["reward_basis_hash"] = kernel.sha256_json(_body(signed))
    with pytest.raises(kernel.LabArenaRewardError, match="signature invalid"):
        kernel.verify_reward_basis_signature(
            signed, public_key_der=signer.public_key_der,
            expected_public_key_hash=signer.public_key_hash,
        )
    tampered = deepcopy(basis)
    tampered["slot_policy"]["assignment_mode"] = "all_qualifying"
    with pytest.raises(kernel.LabArenaRewardError, match="hash"):
        kernel.validate_reward_basis(tampered)


def test_v2_json_round_trip_preserves_hash_and_input():
    basis = _basis()
    before = deepcopy(basis)
    round_trip = json.loads(json.dumps(basis))
    assert contracts.validate_reward_basis(round_trip) == basis
    assert kernel.validate_reward_basis(round_trip) == basis
    assert basis == before


def test_v1_bytes_and_arithmetic_remain_unchanged_and_slots_cannot_leak_into_v1():
    basis = rewards.reward_basis_document(
        round_id="arena-2026-10-08", published_at="2026-10-09T00:00:00Z",
        finalized_epoch=99, king_outcome="crowned", king_hotkey=ALICE,
    )
    basis.pop("champion_reward_factor_ppm")
    basis.pop("reward_basis_hash")
    basis = contracts.finalize_reward_basis(basis)
    assert basis["reward_basis_hash"] == "sha256:decff19e260731bbe199149ef02be232440aed709155e455b2e4e9a59b7193f5"
    assert kernel.champion_values(basis, 100, HOTKEYS)["champion_share"] == 0.3
    assert kernel.champion_values(basis, 146, HOTKEYS)["champion_share"] == 0
    state, signer = _state(basis)
    assert arena_weights.verify_accepted_weight_state_signature(
        state, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    ) == state["state_hash"]
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["sparse_uids"] == [0, 1]
    assert vector["sparse_weights_u16"] == [65535, 28086]
    basis["reward_slots"] = [None, None, None]
    with pytest.raises(kernel.LabArenaRewardError, match="fields"):
        kernel.validate_reward_basis(basis)


def test_wrong_kernel_and_duplicate_uid_ownership_fail_closed():
    with pytest.raises(kernel.LabArenaRewardError, match="slot_allocations"):
        kernel.champion_values(_basis(), 100, HOTKEYS)
    with pytest.raises(kernel.LabArenaRewardError, match="duplicate"):
        kernel.slot_allocations(_basis(), 100, HOTKEYS + [ALICE])


def test_v2_host_and_signer_derive_same_vector_and_restart_recovers_exact_bytes():
    from tests.test_arena_weights import _ArenaChain, _Drand, _StateSource, _chain_profile

    state, authority = _state(_basis(factor=500_000))

    class Chain(_ArenaChain):
        def read_finalized_snapshot(self, **kwargs):
            snapshot = super().read_finalized_snapshot(**kwargs)
            snapshot["metagraph"]["hotkeys"] = HOTKEYS
            return snapshot

    options = {
        "validator_hotkey": BURN, "hotkey_public_key_hex": "3" * 64,
        "chain_profile": _chain_profile(), "chain_source": Chain(state),
        "drand_backend": _Drand(), "verify_sr25519": lambda sig, _payload: sig == b"x" * 64,
        "arena_public_key_der": authority.public_key_der,
        "arena_public_key_hash": authority.public_key_hash, "network": "finney",
        "netuid": 71, "burn_hotkey": BURN, "state_source": _StateSource(state),
    }
    signer = ArenaWeightSigner(**options, sign_sr25519=lambda _payload: b"x" * 64)
    signed = signer.prepare({
        "accepted_state": state, "nonce": 7, "era_current": 1000,
        "runtime_block_hash": "2" * 64, "block_hash": "2" * 64,
    })
    host = arena_weights.derive_arena_weights(state, HOTKEYS)
    for field in ("state_hash", "weights_hash", "sparse_uids", "sparse_weights_u16"):
        assert signed[field] == host[field]

    def forbid_resigning(_payload):
        raise AssertionError("restart must recover without signing again")

    restarted = ArenaWeightSigner(**options, sign_sr25519=forbid_resigning)
    recovered = restarted.recover({"accepted_state": state, "recovery_record": signed["recovery_record"]})
    assert recovered["extrinsic_hex"] == signed["extrinsic_hex"]
    assert recovered["weights_hash"] == host["weights_hash"]


def test_canonical_reward_module_keeps_python37_syntax():
    ast.parse(Path(kernel.__file__).read_text(), feature_version=(3, 7))
