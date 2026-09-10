"""Arena hotkey and policy unsealed together inside Nitro.

The encrypted payload binds the seed to its allowed Arena and chain policy.
Parent RPC configuration cannot change that policy. KMS recipient attestation
still restricts decryption to an approved enclave image.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import threading
from pathlib import Path
from typing import Any, Dict, Mapping
from urllib.parse import urlsplit

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from leadpoet_canonical.arena_weights import classify_arena_signed_request_message
from leadpoet_canonical.chain_source_v2 import ss58_encode_account_id
from leadpoet_canonical.hotkey_authority_v2 import validate_chain_signing_profile
from leadpoet_canonical.kms_recipient import decrypt_kms_recipient_ciphertext
from leadpoet_canonical.lab_arena_rewards import sha256_json, signing_key_from_document

POLICY_SCHEMA = "leadpoet.arena.signer_policy.v1"
PAYLOAD_SCHEMA = "leadpoet.arena.sealed_hotkey.v1"
RECIPIENT_SCHEMA = "leadpoet.arena.hotkey_recipient.v1"
ENCRYPTION_ALGORITHM = "RSAES_OAEP_SHA_256"
MEASURED_DRAND_LIBRARY_PATH = "/app/validator_tee/enclave/libbittensor_drand_v2.so"
_STATEFUL_EPOCH_MODE = "stateful_v1"
_EPOCH_SCHEME = "bittensor.subnet_epoch_index.v1"
_CUTOVER_SCHEMA_VERSION = "leadpoet.subnet_epoch_cutover.v1"


class ArenaHotkeyError(RuntimeError):
    pass


def _validate_epoch_authority(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"mode", "cutover_manifest"}:
        raise ArenaHotkeyError("Arena epoch authority fields are invalid")
    if value.get("mode") != _STATEFUL_EPOCH_MODE:
        raise ArenaHotkeyError("Arena epoch authority mode is invalid")
    manifest = value.get("cutover_manifest")
    fields = {
        "schema_version", "epoch_scheme", "network_genesis_hash", "netuid",
        "cutover_block", "cutover_block_hash", "first_subnet_epoch_index",
        "first_settlement_epoch_id", "last_legacy_epoch_id", "mapping_hash",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != fields:
        raise ArenaHotkeyError("Arena epoch cutover fields are invalid")
    normalized = dict(manifest)
    if (normalized.get("schema_version") != _CUTOVER_SCHEMA_VERSION
            or normalized.get("epoch_scheme") != _EPOCH_SCHEME):
        raise ArenaHotkeyError("Arena epoch cutover identity is invalid")
    for field in (
        "netuid", "cutover_block", "first_subnet_epoch_index",
        "first_settlement_epoch_id", "last_legacy_epoch_id",
    ):
        raw = normalized.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ArenaHotkeyError("Arena epoch cutover %s is invalid" % field)
    if normalized["netuid"] <= 0:
        raise ArenaHotkeyError("Arena epoch cutover netuid is invalid")
    for field in ("network_genesis_hash", "cutover_block_hash"):
        raw_hash = str(normalized.get(field) or "").lower()
        if not re.fullmatch(r"0x[0-9a-f]{64}", raw_hash):
            raise ArenaHotkeyError("Arena epoch cutover %s is invalid" % field)
        normalized[field] = raw_hash
    if normalized["first_settlement_epoch_id"] != normalized["last_legacy_epoch_id"] + 1:
        raise ArenaHotkeyError("Arena epoch settlement mapping is not monotonic")
    body = {key: normalized[key] for key in fields if key != "mapping_hash"}
    if normalized.get("mapping_hash") != sha256_json(body):
        raise ArenaHotkeyError("Arena epoch cutover hash mismatch")
    return {"mode": _STATEFUL_EPOCH_MODE,
            "cutover_manifest": {**body, "mapping_hash": normalized["mapping_hash"]}}


def _nsm_attest(*, user_data: bytes, public_key: bytes) -> bytes:
    try:
        from validator_tee.enclave.nsm_lib import get_attestation_document
        document = get_attestation_document(
            user_data=bytes(user_data), public_key=bytes(public_key)
        )["Attestation"]["document"]
    except Exception as exc:
        raise ArenaHotkeyError("hardware Nitro attestation is unavailable") from exc
    if not isinstance(document, (bytes, bytearray)) or not document:
        raise ArenaHotkeyError("hardware Nitro attestation is empty")
    return bytes(document)


def load_chain_signing_profile(
    path: Path = Path("/app/validator_tee/enclave/chain_signing_profile_v2.json"),
) -> Dict[str, Any]:
    """Load the measured Arena chain profile from the image."""
    try:
        return validate_chain_signing_profile(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
    except (OSError, ValueError) as exc:
        raise ArenaHotkeyError("measured Arena chain profile is unavailable") from exc


def validate_policy(value: Any) -> Dict[str, Any]:
    fields = {
        "schema_version", "network", "netuid", "validator_hotkey",
        "hotkey_public_key", "arena_api_base_url", "arena_signing_key",
        "arena_signing_key_hash", "burn_hotkey", "chain_profile",
        "chain_archive_host", "epoch_authority", "drand_library_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields or value.get("schema_version") != POLICY_SCHEMA:
        raise ArenaHotkeyError("Arena signer policy fields are invalid")
    policy = json.loads(json.dumps(dict(value)))
    try:
        public = bytes.fromhex(policy["hotkey_public_key"])
        if len(public) != 32 or ss58_encode_account_id(public) != policy["validator_hotkey"]:
            raise ValueError("hotkey binding")
        profile = validate_chain_signing_profile(policy["chain_profile"])
        if policy["network"] != profile["network"]:
            raise ValueError("network binding")
        netuid = policy["netuid"]
        if isinstance(netuid, bool) or not isinstance(netuid, int) or not 1 <= netuid <= 65535:
            raise ValueError("netuid")
        epoch = _validate_epoch_authority(policy["epoch_authority"])
        cutover = epoch["cutover_manifest"]
        if cutover["netuid"] != netuid or cutover["network_genesis_hash"] != "0x" + profile["genesis_hash"]:
            raise ValueError("epoch binding")
        endpoint = urlsplit(policy["arena_api_base_url"])
        if (endpoint.scheme != "https" or not endpoint.hostname or endpoint.username
                or endpoint.password or endpoint.query or endpoint.fragment
                or endpoint.path not in ("", "/") or endpoint.port not in (None, 443)):
            raise ValueError("Arena endpoint")
        if not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]{0,251}[a-zA-Z0-9])?", policy["chain_archive_host"]):
            raise ValueError("archive host")
        if not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{40,64}", policy["burn_hotkey"]):
            raise ValueError("burn hotkey")
        if not re.fullmatch(r"[0-9a-f]{64}", policy["drand_library_sha256"]):
            raise ValueError("drand hash")
        signing_key_from_document(policy["arena_signing_key"], policy["arena_signing_key_hash"])
    except Exception as exc:
        raise ArenaHotkeyError("Arena signer policy is invalid") from exc
    return policy


def sealed_payload(seed: bytes, policy: Mapping[str, Any]) -> bytes:
    """Owner-side input for KMS Encrypt. Never use it in a validator runner."""

    normalized = validate_policy(policy)
    import sr25519

    if len(seed) != 32 or sr25519.pair_from_seed(bytes(seed))[0].hex() != normalized["hotkey_public_key"]:
        raise ArenaHotkeyError("Arena seed does not match the configured public key")
    # KMS recipient CMS uses CBC. Bind policy to the secret seed as well as
    # KMS's outer encryption context, so the parent cannot alter policy by
    # modifying the recipient ciphertext on its return trip to Nitro.
    policy_mac = hmac.new(bytes(seed), ("leadpoet.arena.policy.v1:" + sha256_json(normalized)).encode(), hashlib.sha256).hexdigest()
    payload = json.dumps({"schema_version": PAYLOAD_SCHEMA, "seed_hex": bytes(seed).hex(), "policy": normalized, "policy_mac": policy_mac}, sort_keys=True, separators=(",", ":")).encode()
    if len(payload) > 4096:
        raise ArenaHotkeyError("Arena sealed payload exceeds the KMS plaintext limit")
    return payload


class ArenaHotkeyAuthority:
    def __init__(self, *, attestation_supplier=None, decrypt_recipient=None) -> None:
        if attestation_supplier is None:
            attestation_supplier = _nsm_attest
        self._attest = attestation_supplier
        self._decrypt = decrypt_recipient or decrypt_kms_recipient_ciphertext
        self._recipient_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        self._recipient = None
        self._policy = None
        self._pair = None
        self._lock = threading.RLock()

    @property
    def policy(self) -> Dict[str, Any]:
        with self._lock:
            if self._policy is None:
                raise ArenaHotkeyError("Arena hotkey is not provisioned")
            return json.loads(json.dumps(self._policy))

    def recipient_request(self) -> Dict[str, Any]:
        with self._lock:
            # Refresh the hardware timestamp after a failed KMS attempt. Keep
            # the recipient key stable so an in-flight response remains valid.
            public = self._recipient_key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
            claim = {"schema_version": RECIPIENT_SCHEMA, "purpose": PAYLOAD_SCHEMA, "nonce": secrets.token_hex(32)}
            attestation = self._attest(user_data=json.dumps(claim, sort_keys=True, separators=(",", ":")).encode(), public_key=public)
            if not isinstance(attestation, bytes) or not attestation:
                raise ArenaHotkeyError("Arena hardware attestation is unavailable")
            self._recipient = {
                **claim, "attestation_document_b64": base64.b64encode(attestation).decode(),
                "key_encryption_algorithm": ENCRYPTION_ALGORITHM,
            }
            return dict(self._recipient)

    def provision(self, ciphertext_for_recipient_b64: str) -> Dict[str, Any]:
        with self._lock:
            if self._pair is not None or self._recipient is None:
                raise ArenaHotkeyError("Arena hotkey provisioning state is invalid")
            plaintext = bytearray()
            try:
                ciphertext = base64.b64decode(ciphertext_for_recipient_b64, validate=True)
                if not 1 <= len(ciphertext) <= 65536:
                    raise ValueError("recipient ciphertext size")
                plaintext = bytearray(self._decrypt(self._recipient_key, ciphertext))
                if not 1 <= len(plaintext) <= 4096:
                    raise ValueError("sealed payload size")
                document = json.loads(plaintext.decode("utf-8"))
                if not isinstance(document, dict) or set(document) != {"schema_version", "seed_hex", "policy", "policy_mac"} or document["schema_version"] != PAYLOAD_SCHEMA:
                    raise ValueError("sealed payload fields")
                policy = validate_policy(document["policy"])
                seed = bytearray.fromhex(document.pop("seed_hex"))
                try:
                    import sr25519
                    if len(seed) != 32:
                        raise ValueError("seed size")
                    expected_mac = hmac.new(bytes(seed), ("leadpoet.arena.policy.v1:" + sha256_json(policy)).encode(), hashlib.sha256).hexdigest()
                    if not isinstance(document["policy_mac"], str) or not hmac.compare_digest(document["policy_mac"], expected_mac):
                        raise ValueError("sealed policy integrity")
                    pair = sr25519.pair_from_seed(bytes(seed))
                    if pair[0].hex() != policy["hotkey_public_key"]:
                        raise ValueError("seed identity")
                    self._pair = (bytes(pair[0]), bytes(pair[1]))
                    self._policy = policy
                finally:
                    for index in range(len(seed)):
                        seed[index] = 0
            except Exception:
                raise ArenaHotkeyError("Arena recipient provisioning failed") from None
            finally:
                for index in range(len(plaintext)):
                    plaintext[index] = 0
            return self.public_state()

    def public_state(self) -> Dict[str, Any]:
        with self._lock:
            if self._policy is None:
                return {"provisioned": False}
            return {
                "provisioned": True,
                "validator_hotkey": self._policy["validator_hotkey"],
                "hotkey_public_key": self._policy["hotkey_public_key"],
                "policy_hash": sha256_json(self._policy), "policy": self.policy,
            }

    def sign_weight_payload(self, payload: bytes) -> bytes:
        """Internal callback for the exact-transaction verifier. No blind RPC."""
        with self._lock:
            if self._pair is None:
                raise ArenaHotkeyError("Arena hotkey is not provisioned")
            import sr25519
            return bytes(sr25519.sign(self._pair, bytes(payload)))

    def sign_application(self, message: bytes) -> Dict[str, Any]:
        if len(message) > 2 * 1024 * 1024:
            raise ArenaHotkeyError("Arena application message is too large")
        scope = classify_arena_signed_request_message(bytes(message), validator_hotkey=self.policy["validator_hotkey"])
        return {"signature": self.sign_weight_payload(bytes(message)).hex(), "scope": scope}
