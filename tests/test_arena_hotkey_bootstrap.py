import base64
from copy import deepcopy
import json
import hashlib
import sys
import types
from pathlib import Path

import pytest
import sr25519
from bittensor_wallet import Keypair

from lab_arena import contracts, signing
from leadpoet_canonical.lab_arena_rewards import sha256_json
from leadpoet_canonical.chain_source_v2 import ss58_encode_account_id
from tests.test_kms_recipient_v2 import _kms_cms_encrypt
from validator_tee.enclave.arena_hotkey import (
    ArenaHotkeyAuthority, ArenaHotkeyError, POLICY_SCHEMA, sealed_payload,
    load_chain_signing_profile,
)
from validator_tee.host import arena_hotkey_bootstrap, arena_state_relay


def _policy(seed=b"a" * 32):
    public, _private = sr25519.pair_from_seed(seed)
    arena_signer = signing.LocalSigner.generate()
    profile = load_chain_signing_profile(Path("validator_tee/enclave/chain_signing_profile_v2.json"))
    cutover = {
        "schema_version": "leadpoet.subnet_epoch_cutover.v1",
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": "0x" + profile["genesis_hash"],
        "netuid": 71, "cutover_block": 8_637_156,
        "cutover_block_hash": "0x" + "2" * 64,
        "first_subnet_epoch_index": 23_927,
        "first_settlement_epoch_id": 23_992, "last_legacy_epoch_id": 23_991,
    }
    epoch = {"mode": "stateful_v1", "cutover_manifest": {
        **cutover, "mapping_hash": sha256_json(cutover),
    }}
    return {
        "schema_version": POLICY_SCHEMA, "network": profile["network"], "netuid": 71,
        "validator_hotkey": ss58_encode_account_id(public), "hotkey_public_key": public.hex(),
        "arena_api_base_url": "https://arena.example.com",
        "arena_signing_key": signing.signing_key_document(arena_signer.public_key_der),
        "arena_signing_key_hash": arena_signer.public_key_hash,
        "burn_hotkey": Keypair.create_from_uri("//ArenaBootstrapBurn").ss58_address,
        "chain_profile": profile, "chain_archive_host": "archive.chain.opentensor.ai",
        "epoch_authority": epoch, "drand_library_sha256": "1" * 64,
    }, arena_signer


class _Client:
    def __init__(self, authority): self.authority = authority; self.provisions = 0
    def get_arena_hotkey_state_v1(self): return self.authority.public_state()
    def get_arena_hotkey_recipient_v1(self): return self.authority.recipient_request()
    def provision_arena_hotkey_v1(self, value):
        self.provisions += 1
        return self.authority.provision(value)
    def provision_arena_legacy_hotkey_v1(self, value):
        self.provisions += 1
        return self.authority.provision_legacy_seed(value)


class _Kms:
    def __init__(self, captured): self.captured = captured; self.plaintext = None; self.decrypts = 0
    def encrypt(self, **kwargs):
        self.plaintext = bytes(kwargs["Plaintext"])
        return {"KeyId": "kms-key", "CiphertextBlob": b"opaque-kms-ciphertext"}
    def decrypt(self, **kwargs):
        self.decrypts += 1
        request = {"recipient_public_key_der_b64": base64.b64encode(self.captured["public_key"]).decode()}
        return {"KeyId": "kms-key", "CiphertextForRecipient": base64.b64decode(_kms_cms_encrypt(request, self.plaintext.decode()))}


def _legacy_envelope(seed, policy, key_id="kms-key"):
    ciphertext = b"legacy-kms-ciphertext"
    context = {"purpose": "leadpoet.validator_hotkey_seed.v2", "validator_hotkey": policy["validator_hotkey"]}
    return {
        "schema_version": arena_hotkey_bootstrap.LEGACY_ENVELOPE_SCHEMA,
        "ciphertext_blob_b64": base64.b64encode(ciphertext).decode(),
        # Exact wire format emitted by the retired v2 envelope writer.
        "ciphertext_blob_hash": "sha256:" + hashlib.sha256(ciphertext).hexdigest(),
        "encryption_context": context,
        "encryption_context_hash": sha256_json(context),
        "hotkey_public_key": policy["hotkey_public_key"],
        "kms_key_id_hash": arena_hotkey_bootstrap._kms_key_reference_hash(key_id),
        "validator_hotkey": policy["validator_hotkey"],
    }


class _LegacyKms:
    def __init__(self, captured, seed, key_id="kms-key"):
        self.captured = captured; self.seed = seed; self.key_id = key_id; self.decrypts = 0
    def decrypt(self, **kwargs):
        self.decrypts += 1
        request = {"recipient_public_key_der_b64": base64.b64encode(self.captured["public_key"]).decode()}
        return {"KeyId": self.key_id, "CiphertextForRecipient": base64.b64decode(_kms_cms_encrypt(request, self.seed.decode("latin1")))}


def test_legacy_raw_seed_migrates_only_to_measured_policy():
    seed = b"a" * 32
    policy, _ = _policy(seed)
    captured = {}
    authority = ArenaHotkeyAuthority(
        attestation_supplier=lambda **kwargs: captured.update(kwargs) or b"nsm",
        measured_policy=policy,
    )
    client = _Client(authority); kms = _LegacyKms(captured, seed)
    state = arena_hotkey_bootstrap.provision_legacy(
        _legacy_envelope(seed, policy), expected_policy=policy,
        kms_key_id="kms-key", client=client, kms_client=kms,
    )
    assert state["policy"] == policy and state["validator_hotkey"] == policy["validator_hotkey"]
    assert client.provisions == 1 and kms.decrypts == 1


def test_legacy_migration_cli_without_aws_region_uses_key_arn(tmp_path, monkeypatch):
    seed = b"a" * 32
    policy, _ = _policy(seed)
    captured = {}
    authority = ArenaHotkeyAuthority(
        attestation_supplier=lambda **kwargs: captured.update(kwargs) or b"nsm",
        measured_policy=policy,
    )
    key_id = "arn:aws:kms:us-east-1:123456789012:key/test-key"
    kms = _LegacyKms(captured, seed, key_id=key_id)
    calls = []
    def create(service, **kwargs):
        calls.append((service, kwargs))
        return kms
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=create))
    client = _Client(authority)
    monkeypatch.setattr("validator_tee.host.vsock_client.ValidatorEnclaveClient", lambda: client)
    envelope_path = tmp_path / "envelope.json"
    policy_path = tmp_path / "policy.json"
    for path, document in ((envelope_path, _legacy_envelope(seed, policy, key_id)), (policy_path, policy)):
        path.write_text(json.dumps(document)); path.chmod(0o600)
    assert arena_hotkey_bootstrap.main([
        "migrate-legacy", "--legacy-envelope", str(envelope_path),
        "--policy", str(policy_path), "--kms-key-id", key_id,
    ]) == 0
    assert calls == [("kms", {"region_name": "us-east-1"})]
    assert client.provisions == kms.decrypts == 1
    assert client.get_arena_hotkey_state_v1()["policy"] == policy


@pytest.mark.parametrize("key_id", ["arn:aws:s3:us-east-1:123456789012:key/x", "arn:aws:kms::123456789012:key/x", "arn:aws:kms:us-east-1:bad:key/x"])
def test_kms_region_rejects_malformed_arn(key_id):
    with pytest.raises(ArenaHotkeyError, match="ARN"):
        arena_hotkey_bootstrap.kms_region(key_id)


def test_legacy_migration_rejects_unprefixed_ciphertext_hash():
    policy, _ = _policy()
    envelope = _legacy_envelope(b"a" * 32, policy)
    envelope["ciphertext_blob_hash"] = envelope["ciphertext_blob_hash"].removeprefix("sha256:")
    with pytest.raises(ArenaHotkeyError, match="integrity"):
        arena_hotkey_bootstrap.provision_legacy(
            envelope, expected_policy=policy, kms_key_id="kms-key", client=None, kms_client=None,
        )


def test_legacy_migration_rejects_tampered_policy_identity_and_plaintext():
    seed = b"a" * 32; policy, _ = _policy(seed); captured = {}
    authority = ArenaHotkeyAuthority(
        attestation_supplier=lambda **kwargs: captured.update(kwargs) or b"nsm",
        measured_policy=policy,
    )
    envelope = _legacy_envelope(seed, policy)
    changed = deepcopy(policy); changed["arena_api_base_url"] = "https://changed.example.com"
    with pytest.raises(ArenaHotkeyError, match="measured policy"):
        arena_hotkey_bootstrap.provision_legacy(
            envelope, expected_policy=changed, kms_key_id="kms-key",
            client=_Client(authority), kms_client=_LegacyKms(captured, seed),
        )
    wrong_identity = deepcopy(envelope); wrong_identity["validator_hotkey"] = Keypair.create_from_uri("//Other").ss58_address
    with pytest.raises(ArenaHotkeyError, match="identity"):
        arena_hotkey_bootstrap.provision_legacy(
            wrong_identity, expected_policy=policy, kms_key_id="kms-key",
            client=_Client(authority), kms_client=_LegacyKms(captured, seed),
        )

    class PlaintextKms(_LegacyKms):
        def decrypt(self, **kwargs):
            return {"KeyId": self.key_id, "Plaintext": self.seed, "CiphertextForRecipient": b"forbidden"}
    with pytest.raises(ArenaHotkeyError, match="plaintext"):
        arena_hotkey_bootstrap.provision_legacy(
            envelope, expected_policy=policy, kms_key_id="kms-key",
            client=_Client(authority), kms_client=PlaintextKms(captured, seed),
        )


def test_real_cms_recipient_provisions_bound_sr25519_and_does_not_reunseal():
    seed = b"a" * 32
    policy, _arena_signer = _policy(seed)
    captured = {}
    authority = ArenaHotkeyAuthority(attestation_supplier=lambda **kwargs: captured.update(kwargs) or b"nsm-attestation", measured_policy=policy)
    client = _Client(authority)
    kms = _Kms(captured)
    envelope = arena_hotkey_bootstrap.seal(seed, policy, kms_key_id="kms-key", kms_client=kms)
    state = arena_hotkey_bootstrap.provision(envelope, client=client, kms_client=kms)
    assert state["validator_hotkey"] == policy["validator_hotkey"]
    message = contracts.build_signed_request(
        scope=contracts.SCOPE_CLAIM, round_id="arena-2026-09-10",
        hotkey=policy["validator_hotkey"], body={"declared_parallelism": 1},
        timestamp=1_789_000_000, sign_message=lambda _message: "00",
    )
    unsigned = {key: value for key, value in message.items() if key != "signature"}
    raw = contracts.signed_request_message(unsigned).encode()
    result = authority.sign_application(raw)
    assert Keypair(ss58_address=policy["validator_hotkey"]).verify(raw, bytes.fromhex(result["signature"]))
    assert arena_hotkey_bootstrap.provision(envelope, client=client, kms_client=kms) == state
    assert client.provisions == 1 and kms.decrypts == 1


def test_seed_policy_and_parent_plaintext_fail_closed():
    policy, _ = _policy()
    kms = _Kms({})
    with pytest.raises(ArenaHotkeyError, match="seed"):
        arena_hotkey_bootstrap.seal(b"b" * 32, policy, kms_key_id="kms-key", kms_client=kms)
    changed = deepcopy(policy); changed["netuid"] = 72
    with pytest.raises(ArenaHotkeyError, match="policy"):
        arena_hotkey_bootstrap.seal(b"a" * 32, changed, kms_key_id="kms-key", kms_client=kms)

    class PlaintextKms(_Kms):
        def decrypt(self, **kwargs): return {"KeyId": "kms-key", "Plaintext": b"forbidden"}
    captured = {}
    authority = ArenaHotkeyAuthority(attestation_supplier=lambda **kwargs: captured.update(kwargs) or b"nsm", measured_policy=policy)
    envelope = arena_hotkey_bootstrap.seal(b"a" * 32, policy, kms_key_id="kms-key", kms_client=_Kms(captured))
    with pytest.raises(ArenaHotkeyError, match="plaintext"):
        arena_hotkey_bootstrap.provision(envelope, client=_Client(authority), kms_client=PlaintextKms(captured))


def test_recipient_retry_refreshes_attestation_without_replacing_recipient_key():
    calls = []

    def attest(**kwargs):
        calls.append(kwargs)
        return ("hardware-attestation-%d" % len(calls)).encode()

    policy, _ = _policy()
    authority = ArenaHotkeyAuthority(attestation_supplier=attest, measured_policy=policy)
    first = authority.recipient_request()
    second = authority.recipient_request()
    assert len(calls) == 2
    assert first["attestation_document_b64"] != second["attestation_document_b64"]
    assert first["nonce"] != second["nonce"]
    assert calls[0]["public_key"] == calls[1]["public_key"]


@pytest.mark.parametrize("mutation", [
    lambda policy: policy.update(arena_api_base_url="http://arena.example.com"),
    lambda policy: policy.update(network="other-network"),
    lambda policy: policy.update(netuid=72),
])
def test_policy_rejects_api_chain_and_netuid_mismatches(mutation):
    policy, _ = _policy()
    mutation(policy)
    with pytest.raises(ArenaHotkeyError, match="policy"):
        arena_hotkey_bootstrap.seal(b"a" * 32, policy, kms_key_id="kms-key", kms_client=_Kms({}))


def test_recipient_rejects_policy_changed_after_seed_binding():
    policy, _ = _policy()
    document = json.loads(sealed_payload(b"a" * 32, policy))
    document["policy"]["arena_api_base_url"] = "https://changed.example.com"
    captured = {}
    authority = ArenaHotkeyAuthority(attestation_supplier=lambda **kwargs: captured.update(kwargs) or b"nsm", measured_policy=policy)
    authority.recipient_request()
    request = {"recipient_public_key_der_b64": base64.b64encode(captured["public_key"]).decode()}
    ciphertext = _kms_cms_encrypt(request, json.dumps(document, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ArenaHotkeyError, match="provisioning failed"):
        authority.provision(ciphertext)


def test_application_signer_rejects_other_validator_and_chain_payloads():
    seed = b"a" * 32; policy, _ = _policy(seed)
    captured = {}; authority = ArenaHotkeyAuthority(attestation_supplier=lambda **kwargs: captured.update(kwargs) or b"nsm", measured_policy=policy)
    client = _Client(authority); kms = _Kms(captured)
    arena_hotkey_bootstrap.provision(arena_hotkey_bootstrap.seal(seed, policy, kms_key_id="kms-key", kms_client=kms), client=client, kms_client=kms)
    with pytest.raises((ArenaHotkeyError, ValueError)):
        authority.sign_application(b"arbitrary application message")
    with pytest.raises((ArenaHotkeyError, ValueError)):
        authority.sign_application(bytes.fromhex("deadbeef"))


def test_relay_exact_policy_private_dns_rejection_and_cleanup(monkeypatch):
    policy, _ = _policy(); relay = arena_state_relay.ArenaStateRelay(policy)
    control = {"schema_version": arena_state_relay.RELAY_SCHEMA, "host": "arena.example.com", "port": 443, "policy_hash": sha256_json(policy)}
    relay.validate_control(control)
    with pytest.raises(ValueError, match="policy differs"):
        relay.validate_control({**control, "host": "other.example.com"})
    monkeypatch.setattr(arena_state_relay.socket, "getaddrinfo", lambda *_args, **_kwargs: [(2, 1, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(ValueError, match="public address"):
        relay._connect()

    class Connection:
        def __init__(self): self.closed = False; self.shutdown_called = False
        def shutdown(self, _how): self.shutdown_called = True
        def close(self): self.closed = True
    class Thread:
        def join(self, timeout): self.timeout = timeout
        def is_alive(self): return False
    listener = Connection(); active = Connection(); accept_thread = Thread(); worker = Thread()
    relay._listener = listener; relay._accept_thread = accept_thread
    relay._connections.add(active); relay._threads.add(worker)
    relay.stop()
    assert listener.closed and active.closed and active.shutdown_called


def test_tee_service_arena_boot_wire_does_not_require_legacy_configuration(monkeypatch):
    from validator_tee.enclave import arena_hotkey, tee_service
    from validator_tee.enclave import drand_v2

    seed = b"a" * 32
    policy, _ = _policy(seed)
    captured = {}
    monkeypatch.setenv("LEADPOET_ENCLAVE_MODE", "arena")
    monkeypatch.setattr(arena_hotkey, "_nsm_attest", lambda **kwargs: captured.update(kwargs) or b"measured-nsm")
    monkeypatch.setattr(arena_hotkey, "load_measured_policy", lambda: policy)

    class ExternalDrandBoundary:
        def __init__(self, *, library_path, expected_sha256):
            self.library_path = library_path
            self.library_sha256 = expected_sha256

    monkeypatch.setattr(drand_v2, "CtypesDrandCommitBackendV2", ExternalDrandBoundary)
    for name in (
        "validator_chain_source_v2",
        "validator_arena_weight_signer_v1", "validator_arena_hotkey_authority_v1",
    ):
        monkeypatch.setattr(tee_service, name, None)

    recipient_response = tee_service.handle_request({"command": "get_arena_hotkey_recipient_v1"})
    assert recipient_response["status"] == "ok"
    assert captured["public_key"]
    kms = _Kms(captured)
    envelope = arena_hotkey_bootstrap.seal(seed, policy, kms_key_id="kms-key", kms_client=kms)
    recipient = recipient_response["recipient_request"]
    decrypted = kms.decrypt(
        KeyId="kms-key", CiphertextBlob=base64.b64decode(envelope["ciphertext_blob_b64"]),
        EncryptionContext=envelope["encryption_context"],
        Recipient={"KeyEncryptionAlgorithm": recipient["key_encryption_algorithm"],
                   "AttestationDocument": base64.b64decode(recipient["attestation_document_b64"])},
    )
    provisioned = tee_service.handle_request({
        "command": "provision_arena_hotkey_v1",
        "ciphertext_for_recipient_b64": base64.b64encode(decrypted["CiphertextForRecipient"]).decode(),
    })
    assert provisioned["status"] == "ok"
    assert provisioned["arena_hotkey_state"]["validator_hotkey"] == policy["validator_hotkey"]
    state = tee_service.handle_request({"command": "get_arena_hotkey_state_v1"})
    assert state["status"] == "ok" and state["arena_hotkey_state"]["provisioned"] is True

    configuration = {
        "network": policy["network"], "netuid": policy["netuid"],
        "signing_key": policy["arena_signing_key"],
        "expected_public_key_hash": policy["arena_signing_key_hash"],
    }
    configured = tee_service.handle_request({
        "command": "configure_arena_weight_signer_v1", "configuration": configuration,
    })
    assert configured["status"] == "ok" and configured["arena_signer_state"]["configured"] is True
    mismatch = tee_service.handle_request({
        "command": "configure_arena_weight_signer_v1",
        "configuration": {**configuration, "netuid": 72},
    })
    assert mismatch["status"] == "error" and "sealed policy" in mismatch["error"]

    for command in ("configure_authoritative_v2", "configure_hotkey_authority_v2", "get_hotkey_recipient_v2", "arbitrary_rpc"):
        rejected = tee_service.handle_request({"command": command})
        assert rejected == {"status": "error", "error": "RPC is outside Arena enclave mode"}
