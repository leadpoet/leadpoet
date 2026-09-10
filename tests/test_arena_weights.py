import base64
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from leadpoet_canonical import arena_weights
from leadpoet_canonical.lab_arena_rewards import public_key_hash, sha256_json
from lab_arena.rewards import reward_basis_document, reward_constants_document
from lab_arena import contracts
from leadpoet_canonical.weights import normalize_to_u16_with_uids
from leadpoet_canonical.hotkey_authority_v2 import CHAIN_SIGNING_PROFILE_SCHEMA_VERSION
from validator_tee.enclave.arena_weight_signer import ArenaWeightSigner, ArenaWeightSignerError


HOTKEYS = ["5" + ("A" * 47), "5" + ("B" * 47), "5" + ("C" * 47)]


def _signed_state(epoch=100, winner=HOTKEYS[2], valid_until_block=1100):
    key = ec.generate_private_key(ec.SECP256R1())
    der = key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    key_hash = public_key_hash(der)
    basis = reward_basis_document(
        round_id="round-1", published_at="2026-09-10T00:00:00Z",
        finalized_epoch=epoch - 1, king_outcome="crowned", king_hotkey=winner,
        previous_king_start_epoch=None,
        reward_constants=reward_constants_document(pool_percent=5),
    )
    basis_hash = sha256_json({name: basis[name] for name in (
        "schema_version", "round_id", "published_at", "effective_reward_epoch",
        "king_hotkey", "king_outcome", "king_start_epoch", "reward_constants",
    )})
    basis["reward_basis_hash"] = basis_hash
    basis["signature"] = {
        "algorithm": "ECDSA_SHA_256", "public_key_hash": key_hash,
        "signature_b64": base64.b64encode(key.sign(("reward_basis_hash:" + basis_hash).encode(), ec.ECDSA(hashes.SHA256()))).decode(),
    }
    body = {
        "schema_version": arena_weights.ACCEPTED_WEIGHT_STATE_SCHEMA_VERSION,
        "network": "finney", "genesis_hash": "1" * 64, "netuid": 71,
        "epoch": epoch, "valid_from_block": 1000, "valid_until_block": valid_until_block,
        "reward_basis": basis,
        "burn_hotkey": HOTKEYS[0], "issued_at": "2026-09-10T00:00:00Z",
    }
    state_hash = sha256_json(body)
    state = dict(body)
    state["state_hash"] = state_hash
    state["signature"] = {
        "algorithm": "ECDSA_SHA_256", "public_key_hash": key_hash,
        "signature_b64": base64.b64encode(key.sign((arena_weights.WEIGHT_STATE_SIGNATURE_PREFIX + state_hash).encode(), ec.ECDSA(hashes.SHA256()))).decode(),
    }
    return state, der, key_hash


def test_signed_state_derives_exact_finalized_uid_vector():
    state, der, key_hash = _signed_state()
    assert arena_weights.verify_accepted_weight_state_signature(state, public_key_der=der, expected_public_key_hash=key_hash) == state["state_hash"]
    result = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert result["champion_share_ppb"] == 50_000_000
    assert result["burned_residual_ppb"] == 950_000_000
    assert result["sparse_uids"] == [0, 2]
    assert result["sparse_weights_u16"] == [65535, 3449]
    # The dependency-free enclave kernel must quantize exactly like Bittensor.
    float_weights = [0.95, 0.05]
    assert normalize_to_u16_with_uids([0, 2], float_weights) == (
        result["sparse_uids"], result["sparse_weights_u16"]
    )


@pytest.mark.parametrize("mutation,error", [
    (lambda state: state.update(epoch=101), "state_hash"),
    (lambda state: state.update(fixed_allocations=[]), "fields"),
    (lambda state: state.update(valid_until_block=999), "validity window"),
])
def test_state_tampering_and_bad_bounds_fail_closed(mutation, error):
    state, der, key_hash = _signed_state()
    mutation(state)
    with pytest.raises(arena_weights.ArenaWeightError, match=error):
        arena_weights.verify_accepted_weight_state_signature(state, public_key_der=der, expected_public_key_hash=key_hash)


def test_wrong_key_and_changed_finalized_uid_ownership_fail_safe():
    state, der, key_hash = _signed_state()
    other = ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    with pytest.raises(arena_weights.ArenaWeightError, match="pinned Arena key"):
        arena_weights.verify_accepted_weight_state_signature(state, public_key_der=other, expected_public_key_hash=key_hash)
    moved = arena_weights.derive_arena_weights(state, [HOTKEYS[2], HOTKEYS[0], HOTKEYS[1]])
    assert moved["sparse_uids"] == [0, 1]
    assert moved["sparse_weights_u16"] == [3449, 65535]


def test_unregistered_champion_returns_all_emissions_to_burn():
    state, _der, _key_hash = _signed_state(winner="5" + ("D" * 47))
    result = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert result["champion_share_ppb"] == 0
    assert result["burned_residual_ppb"] == 1_000_000_000
    assert result["sparse_uids"] == [0]
    assert result["sparse_weights_u16"] == [65535]


def test_burn_hotkey_must_be_registered():
    state, _der, _key_hash = _signed_state()
    state["burn_hotkey"] = "5" + ("D" * 47)
    body = arena_weights.accepted_weight_state_body(state)
    state["state_hash"] = sha256_json(body)  # derivation checks content hash; signature is tested separately
    with pytest.raises(arena_weights.ArenaWeightError, match="burn_hotkey is not registered"):
        arena_weights.derive_arena_weights(state, HOTKEYS)


def test_protected_application_signer_accepts_only_canonical_arena_claim():
    request = {
        "schema_version": contracts.SIGNED_REQUEST_SCHEMA_VERSION,
        "scope": contracts.SCOPE_CLAIM, "round_id": "round-1",
        "hotkey": HOTKEYS[0], "timestamp": 1_789_000_000,
        "request_id": "1" * 32, "body": {"declared_parallelism": 2},
    }
    message = contracts.signed_request_message(request).encode()
    assert arena_weights.classify_arena_signed_request_message(message, validator_hotkey=HOTKEYS[0]) == "validator.arena_claim.v1"
    altered = message.replace(b'"declared_parallelism":2', b'"declared_parallelism":0')
    with pytest.raises(arena_weights.ArenaWeightError):
        arena_weights.classify_arena_signed_request_message(altered, validator_hotkey=HOTKEYS[0])
    with pytest.raises(arena_weights.ArenaWeightError):
        arena_weights.classify_arena_signed_request_message(message, validator_hotkey=HOTKEYS[1])


def test_arena_enclave_mode_closes_legacy_rpc_surface(monkeypatch):
    from validator_tee.enclave import tee_service

    monkeypatch.setenv("LEADPOET_ENCLAVE_MODE", "arena")
    denied = tee_service.handle_request({"command": "configure_authoritative_v2"})
    assert denied == {"status": "error", "error": "RPC is outside Arena enclave mode"}
    health = tee_service.handle_request({"command": "health"})
    assert health["status"] == "ok"
    assert health["arena_weight_signer_v1_supported"] is True


def _chain_profile():
    return {
        "schema_version": CHAIN_SIGNING_PROFILE_SCHEMA_VERSION, "network": "finney",
        "chain_endpoint": "wss://entrypoint-finney.opentensor.ai:443", "genesis_hash": "1" * 64,
        "spec_version": 432, "transaction_version": 1, "version_key": 10005000,
        "commit_call_index": "0776", "serve_axon_call_index": "0704",
        "commit_reveal_version": 4, "mechid": 0, "tempo": 360,
        "subnet_reveal_period_epochs": 1, "block_time_millis": 12000,
        "max_snapshot_block_drift": 64, "extrinsic_period": 8,
        "signed_extensions": ["CheckMortality", "CheckNonce", "ChargeTransactionPayment", "CheckMetadataHash", "CheckSpecVersion", "CheckTxVersion", "CheckGenesis", "CheckMortalityAdditionalSigned", "CheckMetadataHashAdditionalSigned"],
    }


class _ArenaChain:
    def __init__(self, state): self.state, self.nonce = state, 7
    def read_finalized_snapshot(self, **_kwargs):
        return {"header": {"block": 1000}, "finalized_block_hash": "2" * 64,
                "metagraph": {"hotkeys": HOTKEYS},
                "epoch_authority": {"settlement_epoch_id": 100, "last_epoch_block": 900,
                    "pending_epoch_at": 0, "subnet_epoch_index": 100, "tempo": 360,
                    "blocks_since_last_step": 100, "current_block": 1000,
                    "next_epoch_block": 1260}}
    def read_chain_signing_runtime(self, **_kwargs):
        return {"runtime_block": 1000, "runtime_block_hash": "2" * 64,
                "finalized_block": 1000, "finalized_block_hash": "2" * 64,
                "spec_version": 432, "transaction_version": 1, "genesis_hash": "1" * 64}
    def read_canonical_block_hash(self, **_kwargs): return "2" * 64
    def read_finalized_account_nonce(self, **_kwargs): return self.nonce


class _StateSource:
    def __init__(self, state): self.state = state
    def read(self, **_kwargs): return deepcopy(self.state)


class _Drand:
    def generate_commit(self, **_kwargs): return b"abc", 123


def test_protected_signer_checks_gateway_state_finalized_nonce_and_recovers_exact_bytes():
    state, der, key_hash = _signed_state()
    chain = _ArenaChain(state)
    signer = ArenaWeightSigner(
        validator_hotkey=HOTKEYS[0], hotkey_public_key_hex="3" * 64,
        chain_profile=_chain_profile(), chain_source=chain, drand_backend=_Drand(),
        sign_sr25519=lambda _payload: b"x" * 64,
        verify_sr25519=lambda signature, _payload: signature == b"x" * 64,
        arena_public_key_der=der, arena_public_key_hash=key_hash,
        network="finney", netuid=71, burn_hotkey=HOTKEYS[0], state_source=_StateSource(state),
    )
    request = {"accepted_state": state, "nonce": 7, "era_current": 1000,
               "runtime_block_hash": "2" * 64, "block_hash": "2" * 64}
    signed = signer.prepare(request)
    assert signed["state_hash"] == state["state_hash"]
    assert signed["extrinsic_hash"].startswith("0x")
    restarted = ArenaWeightSigner(
        validator_hotkey=HOTKEYS[0], hotkey_public_key_hex="3" * 64,
        chain_profile=_chain_profile(), chain_source=chain, drand_backend=_Drand(),
        sign_sr25519=lambda _payload: (_ for _ in ()).throw(AssertionError("must not re-sign")),
        verify_sr25519=lambda signature, _payload: signature == b"x" * 64,
        arena_public_key_der=der, arena_public_key_hash=key_hash,
        network="finney", netuid=71, burn_hotkey=HOTKEYS[0], state_source=_StateSource(state),
    )
    recovered = restarted.recover({"accepted_state": state, "recovery_record": signed["recovery_record"]})
    assert recovered["extrinsic_hex"] == signed["extrinsic_hex"]
    tampered = deepcopy(signed["recovery_record"])
    tampered["maximum_block"] += 1
    another_restart = ArenaWeightSigner(
        validator_hotkey=HOTKEYS[0], hotkey_public_key_hex="3" * 64,
        chain_profile=_chain_profile(), chain_source=chain, drand_backend=_Drand(),
        sign_sr25519=lambda _payload: b"x" * 64,
        verify_sr25519=lambda signature, payload: signature == b"x" * 64 and payload.endswith(sha256_json({key: signed["recovery_record"][key] for key in signed["recovery_record"] if key != "record_signature_hex"}).encode()),
        arena_public_key_der=der, arena_public_key_hash=key_hash,
        network="finney", netuid=71, burn_hotkey=HOTKEYS[0], state_source=_StateSource(state),
    )
    with pytest.raises(ArenaWeightSignerError, match="record signature"):
        another_restart.recover({"accepted_state": state, "recovery_record": tampered})
    chain.nonce = 8
    fresh = ArenaWeightSigner(
        validator_hotkey=HOTKEYS[0], hotkey_public_key_hex="3" * 64,
        chain_profile=_chain_profile(), chain_source=chain, drand_backend=_Drand(),
        sign_sr25519=lambda _payload: b"x" * 64,
        verify_sr25519=lambda signature, _payload: signature == b"x" * 64,
        arena_public_key_der=der, arena_public_key_hash=key_hash,
        network="finney", netuid=71, burn_hotkey=HOTKEYS[0], state_source=_StateSource(state),
    )
    with pytest.raises(ArenaWeightSignerError, match="nonce"):
        fresh.prepare(dict(request, nonce=7, era_current=1000))


def test_protected_signer_rejects_state_burn_key_outside_sealed_policy():
    state, der, key_hash = _signed_state()
    signer = ArenaWeightSigner(
        validator_hotkey=HOTKEYS[0], hotkey_public_key_hex="3" * 64,
        chain_profile=_chain_profile(), chain_source=_ArenaChain(state), drand_backend=_Drand(),
        sign_sr25519=lambda _payload: b"x" * 64,
        verify_sr25519=lambda _signature, _payload: True,
        arena_public_key_der=der, arena_public_key_hash=key_hash,
        network="finney", netuid=71, burn_hotkey=HOTKEYS[1],
        state_source=_StateSource(state),
    )
    with pytest.raises(ArenaWeightSignerError, match="burn hotkey"):
        signer.prepare({"accepted_state": state, "nonce": 7, "era_current": 1000,
                        "runtime_block_hash": "2" * 64, "block_hash": "2" * 64})


def test_protected_signer_will_not_sign_an_era_past_state_expiry():
    state, der, key_hash = _signed_state(valid_until_block=1003)
    signer = ArenaWeightSigner(
        validator_hotkey=HOTKEYS[0], hotkey_public_key_hex="3" * 64,
        chain_profile=_chain_profile(), chain_source=_ArenaChain(state), drand_backend=_Drand(),
        sign_sr25519=lambda _payload: b"x" * 64,
        verify_sr25519=lambda _signature, _payload: True,
        arena_public_key_der=der, arena_public_key_hash=key_hash,
        network="finney", netuid=71, burn_hotkey=HOTKEYS[0],
        state_source=_StateSource(state),
    )
    with pytest.raises(ArenaWeightSignerError, match="mortal era extends"):
        signer.prepare({"accepted_state": state, "nonce": 7, "era_current": 1000,
                        "runtime_block_hash": "2" * 64, "block_hash": "2" * 64})


def test_recovered_retry_sequence_uses_highest_protected_attempt_number():
    state, der, key_hash = _signed_state()
    chain = _ArenaChain(state)
    kwargs = dict(
        validator_hotkey=HOTKEYS[0], hotkey_public_key_hex="3" * 64,
        chain_profile=_chain_profile(), chain_source=chain, drand_backend=_Drand(),
        sign_sr25519=lambda _payload: b"x" * 64,
        verify_sr25519=lambda signature, _payload: signature == b"x" * 64,
        arena_public_key_der=der, arena_public_key_hash=key_hash,
        network="finney", netuid=71, burn_hotkey=HOTKEYS[0],
        state_source=_StateSource(state),
    )
    request = {"accepted_state": state, "nonce": 7, "era_current": 1000,
               "runtime_block_hash": "2" * 64, "block_hash": "2" * 64}
    signer = ArenaWeightSigner(**kwargs)
    first = signer.prepare(request)
    assert first["attempt_sequence"] == 1
    # Model the protected expiry release. The old attempt remains recoverable.
    signer._records.pop(state["state_hash"])
    second = signer.prepare(request)
    assert second["attempt_sequence"] == 2

    restarted = ArenaWeightSigner(**kwargs)
    restarted.recover({"accepted_state": state,
                       "recovery_record": second["recovery_record"]})
    restarted._records.pop(state["state_hash"])
    third = restarted.prepare(request)
    assert third["attempt_sequence"] == 3


def test_signed_authorization_binds_rewarded_uids_to_finalized_hotkeys():
    state, der, key_hash = _signed_state()
    signer = ArenaWeightSigner(
        validator_hotkey=HOTKEYS[0], hotkey_public_key_hex="3" * 64,
        chain_profile=_chain_profile(), chain_source=_ArenaChain(state),
        drand_backend=_Drand(), sign_sr25519=lambda _payload: b"x" * 64,
        verify_sr25519=lambda signature, _payload: signature == b"x" * 64,
        arena_public_key_der=der, arena_public_key_hash=key_hash,
        network="finney", netuid=71, burn_hotkey=HOTKEYS[0],
        state_source=_StateSource(state),
    )
    signed = signer.prepare({"accepted_state": state, "nonce": 7,
                             "era_current": 1000,
                             "runtime_block_hash": "2" * 64,
                             "block_hash": "2" * 64})
    assert signed["recipient_uid_hotkeys"] == [
        {"uid": 0, "hotkey": HOTKEYS[0]},
        {"uid": 2, "hotkey": HOTKEYS[2]},
    ]
    tampered = deepcopy(signed["recovery_record"])
    tampered["authorization"]["recipient_uid_hotkeys"][1]["hotkey"] = HOTKEYS[1]
    restarted = ArenaWeightSigner(
        validator_hotkey=HOTKEYS[0], hotkey_public_key_hex="3" * 64,
        chain_profile=_chain_profile(), chain_source=_ArenaChain(state),
        drand_backend=_Drand(), sign_sr25519=lambda _payload: b"x" * 64,
        verify_sr25519=lambda signature, payload: signature == b"x" * 64 and payload.endswith(
            sha256_json({key: signed["recovery_record"][key]
                         for key in signed["recovery_record"]
                         if key != "record_signature_hex"}).encode()),
        arena_public_key_der=der, arena_public_key_hash=key_hash,
        network="finney", netuid=71, burn_hotkey=HOTKEYS[0],
        state_source=_StateSource(state),
    )
    with pytest.raises(ArenaWeightSignerError, match="record signature"):
        restarted.recover({"accepted_state": state, "recovery_record": tampered})


def test_reveal_proof_rejects_rewarded_uid_recycling():
    from validator_tee.enclave.chain_source_v2 import (
        ValidatorChainSourceV2Error, validate_rewarded_uid_ownership,
    )
    bindings = [{"uid": 0, "hotkey": HOTKEYS[0]},
                {"uid": 1, "hotkey": HOTKEYS[1]}]
    validate_rewarded_uid_ownership(HOTKEYS, bindings)
    recycled = [HOTKEYS[0], HOTKEYS[2], HOTKEYS[1]]
    with pytest.raises(ValidatorChainSourceV2Error, match="UID ownership changed"):
        validate_rewarded_uid_ownership(recycled, bindings)
