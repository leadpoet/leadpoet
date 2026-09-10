from datetime import datetime, timezone

import pytest
from bittensor_wallet import Keypair

from lab_arena import contracts, rewards, signing, weight_state
from leadpoet_canonical.arena_weights import verify_accepted_weight_state_signature


KING = Keypair.create_from_uri("//ArenaWeightKing").ss58_address
BURN = Keypair.create_from_uri("//ArenaWeightBurn").ss58_address
FULFILLMENT = Keypair.create_from_uri("//ArenaWeightFulfillment").ss58_address


def _basis(signer):
    unsigned = rewards.reward_basis_document(
        round_id="arena-2026-09-10", published_at="2026-09-10T00:00:00Z",
        finalized_epoch=24999, king_outcome="crowned", king_hotkey=KING,
    )
    return signing.sign_document(signer, unsigned, hash_field="reward_basis_hash")


def test_accepted_weight_state_binds_scope_economics_and_nested_reward_basis():
    signer = signing.LocalSigner.generate()
    state = weight_state.build_accepted_weight_state(
        signer, network="finney", genesis_hash="11" * 32, netuid=71,
        epoch=25000, valid_from_block=100, valid_until_block=459,
        reward_basis=_basis(signer),
        fixed_allocations=[{"hotkey": BURN, "share_ppb": 300_000_000}],
        fulfillment_demands=[{"hotkey": FULFILLMENT, "share_ppb": 605_000_000}],
        burn_hotkey=BURN, issued_at="2026-09-10T00:00:00Z",
    )
    assert verify_accepted_weight_state_signature(
        state, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    ) == state["state_hash"]
    tampered = dict(state, epoch=25001)
    with pytest.raises(ValueError, match="state_hash"):
        verify_accepted_weight_state_signature(
            tampered, public_key_der=signer.public_key_der,
            expected_public_key_hash=signer.public_key_hash,
        )


def test_chain_outcome_is_fresh_and_request_id_is_content_addressed():
    now = datetime(2026, 9, 10, tzinfo=timezone.utc)
    core = {
        "schema_version": weight_state.CHAIN_OUTCOME_SCHEMA_VERSION,
        "network": "finney", "netuid": 71, "epoch": 25000,
        "validator_hotkey": KING, "state_hash": "sha256:" + "1" * 64,
        "weights_hash": "2" * 64,
        "extrinsic_hash": "0x" + "3" * 64,
        "finalized_block_hash": "0x" + "4" * 64,
        "finalized_block_number": 123, "observed_at": "2026-09-10T00:00:00Z",
    }
    document = {**core, "request_id": contracts.document_hash(core), "signature": "0x1234"}
    assert weight_state.validate_chain_outcome(document, now=now)["epoch"] == 25000
    assert weight_state.validate_chain_outcome(
        document, now=datetime(2026, 9, 11, tzinfo=timezone.utc)
    )["request_id"] == document["request_id"]
    future = {**document, "observed_at": "2026-09-10T00:06:00Z"}
    core = {key: future[key] for key in future if key not in {"request_id", "signature"}}
    future["request_id"] = contracts.document_hash(core)
    with pytest.raises(contracts.ArenaContractError, match="future"):
        weight_state.validate_chain_outcome(future, now=datetime(2026, 9, 10, tzinfo=timezone.utc))
