"""End-to-end weight-commit tests: primary submits, auditor verifies.

Every cryptographic step runs for real — the sr25519 hotkey binding
signature, the ed25519 enclave signature over the weights hash, and the
canonical weights-hash recomputation — through the exact production
handler and auditor code paths. The only stubbed boundary is the AWS
Nitro attestation document, which cannot be minted outside enclave
hardware, plus database/chain I/O.

Why this suite exists: epoch 23918 was lost to a runtime-evaluated
annotation crash inside the submit handler, and epoch 23929 to a missing
sr25519 dependency silently swallowed as "invalid hotkey binding". Both
would have failed these tests. If this file fails, weight submission is
broken in whatever runtime pytest is running under.
"""

import asyncio
import base64
import hashlib

import pytest

pytest.importorskip("substrateinterface")
pytest.importorskip("bittensor")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from substrateinterface import Keypair

from validator_tee.host.legacy_v1_compat import build_legacy_v1_submission


NETUID = 71
EPOCH_ID = 12345
BLOCK = EPOCH_ID * 360 + 346  # inside the submission window, matches epoch
CHAIN = "wss://entrypoint-finney.opentensor.ai:443"
TEST_PCR0 = "ab" * 48
TEST_PCR0_COMMIT = "c" * 40


class FakeEnclaveSigner:
    """Duck-typed LegacyV1EnclaveClient backed by a real ed25519 key."""

    def __init__(self):
        self._key = Ed25519PrivateKey.generate()
        raw_pub = self._key.public_key().public_bytes_raw()
        self._pubkey = raw_pub.hex()
        self._code_hash = hashlib.sha256(b"e2e-code").hexdigest()

    def get_public_key(self) -> str:
        return self._pubkey

    def get_code_hash(self) -> str:
        return self._code_hash

    def sign_weights_hash(self, weights_hash: str) -> str:
        return self._key.sign(bytes.fromhex(weights_hash)).hex()

    def get_attestation(self, epoch_id: int) -> str:
        return base64.b64encode(f"e2e-attestation-{epoch_id}".encode()).decode()


class _FakeQuery:
    def __getattr__(self, _name):
        return lambda *a, **k: self

    def execute(self):
        class _R:
            data = []

        return _R()


class _FakeReadClient:
    def table(self, _name):
        return _FakeQuery()


class _FakeSubtensor:
    def get_current_block(self):
        return BLOCK

    def metagraph(self, netuid):  # pragma: no cover - snapshot is best-effort
        raise RuntimeError("no chain in tests")


def _build_submission(hotkey_keypair: Keypair, enclave: FakeEnclaveSigner) -> dict:
    prepared = build_legacy_v1_submission(
        client=enclave,
        netuid=NETUID,
        epoch_id=EPOCH_ID,
        block=BLOCK,
        uids=[0, 5, 42],
        weights=[0.5, 0.3, 0.2],
        validator_hotkey=hotkey_keypair.ss58_address,
        sign_binding_message=lambda message: hotkey_keypair.sign(message),
        expected_chain=CHAIN,
        validator_version="e2etest",
    )
    return prepared


def _patch_gateway(monkeypatch, hotkey_ss58: str):
    import gateway.api.weights as weights_api

    monkeypatch.setattr(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {hotkey_ss58})
    monkeypatch.setattr(weights_api, "EXPECTED_CHAIN", CHAIN)
    monkeypatch.setattr(weights_api, "ALLOWED_NETUIDS", {NETUID})
    # "test" network skips the transparency-log and DB persistence branches
    # while every verification step still runs.
    monkeypatch.setattr(weights_api, "BITTENSOR_NETWORK", "test")
    # The handler imports the DB client lazily inside the function and reads
    # the subtensor through a module-level singleton.
    import gateway.db.client as db_client

    monkeypatch.setattr(db_client, "get_read_client", lambda: _FakeReadClient())
    monkeypatch.setattr(weights_api, "_subtensor", _FakeSubtensor())
    monkeypatch.setattr(
        weights_api,
        "verify_validator_attestation",
        lambda attestation_b64, expected_pubkey, expected_epoch_id: (
            True,
            {"pcr0": TEST_PCR0, "pcr0_commit": TEST_PCR0_COMMIT},
        ),
    )
    return weights_api


def test_primary_validator_weight_commit_accepted(monkeypatch):
    hotkey = Keypair.create_from_uri("//weights-e2e-primary")
    enclave = FakeEnclaveSigner()
    prepared = _build_submission(hotkey, enclave)

    weights_api = _patch_gateway(monkeypatch, hotkey.ss58_address)
    submission = weights_api.WeightSubmission(**prepared["payload"])
    response = asyncio.run(weights_api.submit_weights(submission))

    assert response.success is True
    assert response.epoch_id == EPOCH_ID
    assert response.weights_count == len(prepared["uids"])


def test_tampered_binding_signature_is_rejected(monkeypatch):
    """The sr25519 verification must be real, not vacuous."""
    hotkey = Keypair.create_from_uri("//weights-e2e-primary")
    enclave = FakeEnclaveSigner()
    prepared = _build_submission(hotkey, enclave)

    intruder = Keypair.create_from_uri("//weights-e2e-intruder")
    payload = dict(prepared["payload"])
    payload["validator_hotkey_signature"] = intruder.sign(
        payload["binding_message"].encode()
    ).hex()

    weights_api = _patch_gateway(monkeypatch, hotkey.ss58_address)
    submission = weights_api.WeightSubmission(**payload)
    with pytest.raises(Exception) as excinfo:
        asyncio.run(weights_api.submit_weights(submission))
    assert getattr(excinfo.value, "status_code", None) == 403


def test_tampered_weights_hash_is_rejected(monkeypatch):
    hotkey = Keypair.create_from_uri("//weights-e2e-primary")
    enclave = FakeEnclaveSigner()
    prepared = _build_submission(hotkey, enclave)

    payload = dict(prepared["payload"])
    payload["weights_u16"] = list(payload["weights_u16"])
    payload["weights_u16"][0] = (payload["weights_u16"][0] + 1) % 65536

    weights_api = _patch_gateway(monkeypatch, hotkey.ss58_address)
    submission = weights_api.WeightSubmission(**payload)
    with pytest.raises(Exception) as excinfo:
        asyncio.run(weights_api.submit_weights(submission))
    assert getattr(excinfo.value, "status_code", None) == 400


def _served_bundle(prepared: dict) -> dict:
    """The bundle exactly as the gateway serves it to auditors."""
    payload = prepared["payload"]
    return {
        "netuid": payload["netuid"],
        "epoch_id": payload["epoch_id"],
        "block": payload["block"],
        "uids": payload["uids"],
        "weights_u16": payload["weights_u16"],
        "weights_hash": payload["weights_hash"],
        "validator_hotkey": payload["validator_hotkey"],
        "validator_enclave_pubkey": payload["validator_enclave_pubkey"],
        "validator_signature": payload["validator_signature"],
        "validator_attestation_b64": payload["validator_attestation_b64"],
        "validator_code_hash": payload["validator_code_hash"],
        "validator_pcr0": TEST_PCR0,
        "pcr0_commit_hash": TEST_PCR0_COMMIT,
    }


def _bare_auditor(monkeypatch):
    from neurons import auditor_validator as auditor_module

    auditor = object.__new__(auditor_module.AuditorValidator)

    class _Cfg:
        netuid = NETUID

    auditor.config = _Cfg()
    monkeypatch.setattr(
        auditor_module.AuditorValidator,
        "_v1_pcr0_commit_is_allowed",
        staticmethod(lambda pcr0, commit_hash: True),
    )
    monkeypatch.setattr(
        auditor_module.AuditorValidator,
        "_verify_v1_nitro_attestation",
        lambda self, bundle, *, pcr0, pubkey, epoch_id: True,
    )
    return auditor


def test_audit_validator_verifies_served_bundle(monkeypatch):
    hotkey = Keypair.create_from_uri("//weights-e2e-primary")
    enclave = FakeEnclaveSigner()
    prepared = _build_submission(hotkey, enclave)
    bundle = _served_bundle(prepared)

    auditor = _bare_auditor(monkeypatch)
    verified = auditor.verify_attested_weights_v1(bundle, expected_epoch_id=EPOCH_ID)

    assert verified is not None
    assert verified["uids"] == prepared["uids"]
    assert verified["weights_u16"] == prepared["weights_u16"]


def test_audit_validator_rejects_tampered_bundle(monkeypatch):
    hotkey = Keypair.create_from_uri("//weights-e2e-primary")
    enclave = FakeEnclaveSigner()
    prepared = _build_submission(hotkey, enclave)
    bundle = _served_bundle(prepared)
    bundle["weights_u16"] = list(bundle["weights_u16"])
    bundle["weights_u16"][-1] = (bundle["weights_u16"][-1] + 1) % 65536

    auditor = _bare_auditor(monkeypatch)
    assert (
        auditor.verify_attested_weights_v1(bundle, expected_epoch_id=EPOCH_ID) is None
    )
