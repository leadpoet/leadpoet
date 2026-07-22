"""KMS-sealed, purpose-constrained sr25519 authority inside validator Nitro."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_receipt_graph,
    build_execution_receipt_body,
    create_signed_execution_receipt,
    merkle_root,
    sha256_bytes,
    sha256_json,
    validate_transport_attempt,
    validate_receipt_graph,
    validate_signed_execution_receipt,
    verify_boot_identity_nitro,
)
from leadpoet_canonical.binding import parse_binding_message
from leadpoet_canonical.hotkey_authority_v2 import (
    build_application_signature_request_v2,
    build_serve_axon_extrinsic_authorization_v2,
    build_weight_extrinsic_authorization_v2,
    encode_signed_extrinsic_v2,
    mortal_era_bounds,
    signed_extrinsic_hash_v2,
    validate_chain_signing_profile,
    validate_weight_extrinsic_authorization_v2,
)
from leadpoet_canonical.kms_recipient import decrypt_kms_recipient_ciphertext
from leadpoet_canonical.weight_authority_v2 import (
    validate_published_weight_bundle_v2,
    validate_weight_snapshot_v2,
)


HOTKEY_RECIPIENT_SCHEMA_VERSION = "leadpoet.validator_hotkey_recipient.v2"
HOTKEY_RECIPIENT_PURPOSE = "leadpoet.validator_hotkey_unseal.v2"
HOTKEY_ENCRYPTION_ALGORITHM = "RSAES_OAEP_SHA_256"
HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION = "leadpoet.validator_hotkey_config.v2"
MEASURED_DRAND_LIBRARY_PATH = "/app/validator_tee/enclave/libbittensor_drand_v2.so"
MAX_RECIPIENT_CIPHERTEXT_BYTES = 64 * 1024
MAX_PENDING_WEIGHT_AUTHORIZATIONS = 8
MAX_SERVE_AXON_SIGNATURES_PER_BOOT = 8
SDK_SET_WEIGHTS_RETRIES_PER_CALL = 5

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAW_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_HOTKEY_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{40,64}$")


class ValidatorHotkeyAuthorityV2Error(RuntimeError):
    """The hotkey is unavailable or a requested signature is unauthorized."""


def validate_hotkey_authority_configuration(
    value: Mapping[str, Any]
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "validator_hotkey",
        "hotkey_public_key",
        "chain_signing_profile_hash",
        "drand_library_path",
        "drand_library_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValidatorHotkeyAuthorityV2Error(
            "validator hotkey configuration fields are invalid"
        )
    if value.get("schema_version") != HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION:
        raise ValidatorHotkeyAuthorityV2Error(
            "validator hotkey configuration schema is invalid"
        )
    hotkey = str(value.get("validator_hotkey") or "")
    public_key = str(value.get("hotkey_public_key") or "").lower()
    profile_hash = str(value.get("chain_signing_profile_hash") or "").lower()
    library_path = str(value.get("drand_library_path") or "")
    library_hash = str(value.get("drand_library_sha256") or "").lower()
    if (
        not _HOTKEY_RE.fullmatch(hotkey)
        or not _RAW_HASH_RE.fullmatch(public_key)
        or not _HASH_RE.fullmatch(profile_hash)
        or library_path != MEASURED_DRAND_LIBRARY_PATH
        or not _RAW_HASH_RE.fullmatch(library_hash)
    ):
        raise ValidatorHotkeyAuthorityV2Error(
            "validator hotkey configuration is invalid"
        )
    return {
        "schema_version": HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
        "validator_hotkey": hotkey,
        "hotkey_public_key": public_key,
        "chain_signing_profile_hash": profile_hash,
        "drand_library_path": library_path,
        "drand_library_sha256": library_hash,
    }


def hotkey_authority_configuration_hash(value: Mapping[str, Any]) -> str:
    return sha256_json(validate_hotkey_authority_configuration(value))


def _issued_at(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Sr25519Backend:
    def __init__(self) -> None:
        try:
            import sr25519
        except Exception as exc:
            raise ValidatorHotkeyAuthorityV2Error(
                "measured sr25519 runtime is unavailable"
            ) from exc
        self._module = sr25519

    def pair_from_seed(self, seed: bytes) -> Tuple[bytes, bytes]:
        return self._module.pair_from_seed(bytes(seed))

    def sign(self, keypair: Tuple[bytes, bytes], message: bytes) -> bytes:
        return self._module.sign(keypair, bytes(message))

    def verify(self, signature: bytes, message: bytes, public_key: bytes) -> bool:
        return bool(
            self._module.verify(
                bytes(signature), bytes(message), bytes(public_key)
            )
        )


def load_chain_signing_profile(
    path: Path = Path("/app/validator_tee/enclave/chain_signing_profile_v2.json"),
) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidatorHotkeyAuthorityV2Error(
            "measured chain signing profile is unavailable"
        ) from exc
    return validate_chain_signing_profile(value)


class ValidatorHotkeyAuthorityV2:
    """Own the production hotkey and expose only measured signing domains."""

    def __init__(
        self,
        *,
        boot_identity_supplier: Callable[[], Mapping[str, Any]],
        validator_hotkey: str,
        hotkey_public_key_hex: str,
        chain_profile: Mapping[str, Any],
        sign_receipt_digest: Callable[[bytes], bytes],
        attestation_supplier: Callable[..., bytes],
        drand_backend: Any,
        chain_source: Any = None,
        sr25519_backend: Any = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._boot_identity_supplier = boot_identity_supplier
        self.validator_hotkey = str(validator_hotkey or "")
        if not _HOTKEY_RE.fullmatch(self.validator_hotkey):
            raise ValidatorHotkeyAuthorityV2Error("validator hotkey is invalid")
        try:
            self.hotkey_public_key = bytes.fromhex(str(hotkey_public_key_hex or ""))
        except ValueError as exc:
            raise ValidatorHotkeyAuthorityV2Error(
                "hotkey public key is invalid"
            ) from exc
        if len(self.hotkey_public_key) != 32:
            raise ValidatorHotkeyAuthorityV2Error("hotkey public key is invalid")
        self.chain_profile = validate_chain_signing_profile(chain_profile)
        self._sign_receipt_digest = sign_receipt_digest
        self._attestation_supplier = attestation_supplier
        self._drand = drand_backend
        self._chain_source = chain_source
        self._sr25519 = sr25519_backend or _Sr25519Backend()
        self._clock = clock
        self._recipient_private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=3072
        )
        self._recipient_public_der = self._recipient_private_key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self._recipient_request = None  # type: Optional[Dict[str, Any]]
        self._keypair = None  # type: Optional[Tuple[bytes, bytes]]
        self._weights = {}  # type: Dict[str, Dict[str, Any]]
        self._commits = {}  # type: Dict[str, Dict[str, Any]]
        self._serve_axon_signature_count = 0
        self._lock = threading.Lock()

    def recipient_request(self) -> Dict[str, Any]:
        with self._lock:
            if self._recipient_request is not None:
                return dict(self._recipient_request)
            boot = dict(self._boot_identity_supplier())
            claim = {
                "schema_version": HOTKEY_RECIPIENT_SCHEMA_VERSION,
                "purpose": HOTKEY_RECIPIENT_PURPOSE,
                "boot_identity_hash": str(boot["boot_identity_hash"]),
                "validator_hotkey": self.validator_hotkey,
                "hotkey_public_key_hash": sha256_bytes(self.hotkey_public_key),
                "recipient_public_key_hash": sha256_bytes(
                    self._recipient_public_der
                ),
                "request_nonce": secrets.token_hex(16),
            }
            user_data = json.dumps(
                {
                    "schema_version": HOTKEY_RECIPIENT_SCHEMA_VERSION,
                    "purpose": HOTKEY_RECIPIENT_PURPOSE,
                    "claim_hash": sha256_json(claim),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            document = self._attestation_supplier(
                user_data=user_data,
                public_key=self._recipient_public_der,
            )
            if not isinstance(document, (bytes, bytearray)) or not document:
                raise ValidatorHotkeyAuthorityV2Error(
                    "hotkey recipient attestation is unavailable"
                )
            request = {
                **claim,
                "recipient_public_key_der_b64": base64.b64encode(
                    self._recipient_public_der
                ).decode("ascii"),
                "attestation_document_b64": base64.b64encode(
                    bytes(document)
                ).decode("ascii"),
                "key_encryption_algorithm": HOTKEY_ENCRYPTION_ALGORITHM,
            }
            self._recipient_request = request
            return dict(request)

    def provision_seed(self, *, ciphertext_for_recipient_b64: str) -> Dict[str, Any]:
        with self._lock:
            if self._keypair is not None:
                raise ValidatorHotkeyAuthorityV2Error(
                    "validator hotkey is already provisioned"
                )
            if self._recipient_request is None:
                raise ValidatorHotkeyAuthorityV2Error(
                    "hotkey recipient request was not created"
                )
        try:
            ciphertext = base64.b64decode(
                str(ciphertext_for_recipient_b64 or ""), validate=True
            )
        except Exception as exc:
            raise ValidatorHotkeyAuthorityV2Error(
                "hotkey recipient ciphertext is invalid"
            ) from exc
        if not ciphertext or len(ciphertext) > MAX_RECIPIENT_CIPHERTEXT_BYTES:
            raise ValidatorHotkeyAuthorityV2Error(
                "hotkey recipient ciphertext is outside limit"
            )
        try:
            seed = bytearray(
                decrypt_kms_recipient_ciphertext(
                    self._recipient_private_key,
                    ciphertext,
                )
            )
        except Exception as exc:
            raise ValidatorHotkeyAuthorityV2Error(
                "hotkey recipient unwrap failed"
            ) from exc
        try:
            if len(seed) != 32:
                raise ValidatorHotkeyAuthorityV2Error(
                    "validator hotkey seed must be 32 bytes"
                )
            public_key, secret_key = self._sr25519.pair_from_seed(bytes(seed))
            pair = (bytes(public_key), bytes(secret_key))
            if pair[0] != self.hotkey_public_key or len(pair[1]) != 64:
                raise ValidatorHotkeyAuthorityV2Error(
                    "unsealed seed does not match validator hotkey"
                )
            with self._lock:
                if self._keypair is not None:
                    raise ValidatorHotkeyAuthorityV2Error(
                        "validator hotkey provisioning raced"
                    )
                self._keypair = pair
        finally:
            for index in range(len(seed)):
                seed[index] = 0
        return self.public_state()

    def public_state(self) -> Dict[str, Any]:
        with self._lock:
            provisioned = self._keypair is not None
            pending_weights = len(self._weights)
            pending_commits = len(self._commits)
        return {
            "validator_hotkey": self.validator_hotkey,
            "hotkey_public_key": self.hotkey_public_key.hex(),
            "provisioned": provisioned,
            "pending_weight_authorizations": pending_weights,
            "pending_extrinsic_authorizations": pending_commits,
            "chain_signing_profile_hash": sha256_json(self.chain_profile),
        }

    def _require_keypair(self) -> Tuple[bytes, bytes]:
        with self._lock:
            keypair = self._keypair
        if keypair is None:
            raise ValidatorHotkeyAuthorityV2Error(
                "validator hotkey is not provisioned"
            )
        return keypair

    def register_weight_result(self, response: Mapping[str, Any]) -> str:
        expected_fields = {
            "weight_snapshot",
            "weight_result",
            "weights_signature",
            "receipt_graph",
            "boot_identity",
        }
        if not isinstance(response, Mapping) or set(response) != expected_fields:
            raise ValidatorHotkeyAuthorityV2Error(
                "authoritative weight result fields are invalid"
            )
        boot = dict(self._boot_identity_supplier())
        if dict(response["boot_identity"]) != boot:
            raise ValidatorHotkeyAuthorityV2Error(
                "authoritative weight result uses another boot"
            )
        expected_result = validate_weight_snapshot_v2(response["weight_snapshot"])
        if dict(response["weight_result"]) != expected_result:
            raise ValidatorHotkeyAuthorityV2Error(
                "authoritative weight result is not canonical"
            )
        graph = dict(response["receipt_graph"])
        validate_receipt_graph(graph)
        receipts = {
            str(item["receipt_hash"]): item for item in graph["receipts"]
        }
        root_hash = str(graph["root_receipt_hash"])
        root = receipts.get(root_hash)
        if (
            not isinstance(root, Mapping)
            or root.get("role") != "validator_weights"
            or root.get("purpose") != "validator.weights.computed.v2"
            or root.get("output_root") != sha256_json(expected_result)
            or root.get("boot_identity_hash") != boot["boot_identity_hash"]
        ):
            raise ValidatorHotkeyAuthorityV2Error(
                "authoritative weight root receipt is invalid"
            )
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(str(root["enclave_pubkey"]))
            ).verify(
                bytes.fromhex(str(response["weights_signature"])),
                bytes.fromhex(str(expected_result["weights_hash"])),
            )
        except Exception as exc:
            raise ValidatorHotkeyAuthorityV2Error(
                "authoritative weight signature is invalid"
            ) from exc
        if response["weight_snapshot"].get("validator_hotkey") != self.validator_hotkey:
            raise ValidatorHotkeyAuthorityV2Error(
                "authoritative weight snapshot hotkey differs"
            )
        authorization_id = sha256_json(
            {
                "boot_identity_hash": boot["boot_identity_hash"],
                "root_receipt_hash": root_hash,
                "snapshot_hash": response["weight_snapshot"]["snapshot_hash"],
                "nonce": secrets.token_hex(16),
            }
        )
        with self._lock:
            if len(self._weights) >= MAX_PENDING_WEIGHT_AUTHORIZATIONS:
                raise ValidatorHotkeyAuthorityV2Error(
                    "pending weight authorization capacity is full"
                )
            self._weights[authorization_id] = {
                "weight_snapshot": dict(response["weight_snapshot"]),
                "weight_result": expected_result,
                "root_receipt_hash": root_hash,
                "receipt_graph": graph,
                "attempts": 0,
                "finalization": None,
            }
        return authorization_id

    def recover_weight_publication(
        self,
        *,
        published_bundle: Mapping[str, Any],
        weight_submission_event_hash: str,
        extrinsic_signature_results: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Recover public signed authority after a parent or enclave restart.

        No secret state is imported.  The new enclave verifies the old Nitro
        identity, every receipt and sr25519 signature, then reconstructs only
        the minimum pending state needed to prove finalized chain inclusion.
        """

        self._require_keypair()
        event_hash = str(weight_submission_event_hash or "").lower()
        if not _HASH_RE.fullmatch(event_hash):
            raise ValidatorHotkeyAuthorityV2Error(
                "recovery publication event hash is invalid"
            )
        verified = validate_published_weight_bundle_v2(published_bundle)
        if verified["validator_hotkey"] != self.validator_hotkey:
            raise ValidatorHotkeyAuthorityV2Error(
                "recovery bundle uses another validator hotkey"
            )
        graph = dict(published_bundle["receipt_graph"])
        validate_receipt_graph(graph)
        validator_boots = [
            dict(identity)
            for identity in graph["boot_identities"]
            if identity.get("physical_role") == "validator_weights"
            and identity.get("boot_identity_hash")
            == verified["validator_boot_identity_hash"]
        ]
        if len(validator_boots) != 1:
            raise ValidatorHotkeyAuthorityV2Error(
                "recovery bundle needs exactly one computing validator boot"
            )
        old_boot = validator_boots[0]
        current_boot = dict(self._boot_identity_supplier())
        for field in (
            "commit_sha",
            "pcr0",
            "build_manifest_hash",
            "dependency_lock_hash",
            "config_hash",
        ):
            if old_boot.get(field) != current_boot.get(field):
                raise ValidatorHotkeyAuthorityV2Error(
                    "recovery validator release differs at %s" % field
                )
        try:
            verify_boot_identity_nitro(
                old_boot,
                expected_pcr0=old_boot["pcr0"],
                certificate_validity_at_attestation_time=True,
            )
        except Exception as exc:
            raise ValidatorHotkeyAuthorityV2Error(
                "recovery validator boot attestation is invalid"
            ) from exc
        try:
            binding_signature = bytes.fromhex(
                str(published_bundle["validator_hotkey_signature"])
            )
            binding_message = str(published_bundle["binding_message"]).encode(
                "utf-8"
            )
        except (KeyError, ValueError) as exc:
            raise ValidatorHotkeyAuthorityV2Error(
                "recovery gateway binding is invalid"
            ) from exc
        if not self._sr25519.verify(
            binding_signature, binding_message, self.hotkey_public_key
        ):
            raise ValidatorHotkeyAuthorityV2Error(
                "recovery gateway binding signature is invalid"
            )
        computed_receipts = [
            receipt
            for receipt in graph["receipts"]
            if receipt.get("receipt_hash") == verified["weight_receipt_hash"]
            and receipt.get("role") == "validator_weights"
            and receipt.get("purpose") == "validator.weights.computed.v2"
        ]
        if len(computed_receipts) != 1:
            raise ValidatorHotkeyAuthorityV2Error(
                "recovery computing receipt is missing or ambiguous"
            )
        computed_receipt = dict(computed_receipts[0])
        weight_result = dict(published_bundle["weight_result"])
        normalized_signed = []
        seen_authorizations = set()
        seen_extrinsics = set()
        for item in extrinsic_signature_results:
            expected_fields = {
                "schema_version",
                "authorization_hash",
                "validator_hotkey",
                "signature",
                "extrinsic_hash",
                "authorization",
                "receipt",
            }
            if not isinstance(item, Mapping) or set(item) != expected_fields:
                raise ValidatorHotkeyAuthorityV2Error(
                    "recovery extrinsic signature fields are invalid"
                )
            authorization = validate_weight_extrinsic_authorization_v2(
                item["authorization"], profile=self.chain_profile
            )
            output = {
                "schema_version": str(item["schema_version"]),
                "authorization_hash": str(item["authorization_hash"]),
                "validator_hotkey": str(item["validator_hotkey"]),
                "signature": str(item["signature"]).lower(),
                "extrinsic_hash": str(item["extrinsic_hash"]).lower(),
            }
            if (
                output["schema_version"]
                != "leadpoet.weight_extrinsic_signature.v2"
                or output["authorization_hash"]
                != authorization["authorization_hash"]
                or output["validator_hotkey"] != self.validator_hotkey
                or authorization["validator_hotkey"] != self.validator_hotkey
                or authorization["weight_receipt_hash"]
                != computed_receipt["receipt_hash"]
                or authorization["weight_submission_event_hash"] != event_hash
                or authorization["weights_hash"] != weight_result["weights_hash"]
                or authorization["sparse_uids"] != weight_result["sparse_uids"]
                or authorization["sparse_weights_u16"]
                != weight_result["sparse_weights_u16"]
            ):
                raise ValidatorHotkeyAuthorityV2Error(
                    "recovery extrinsic authorization differs from weights"
                )
            try:
                signature = bytes.fromhex(output["signature"])
                signed_message = bytes.fromhex(
                    authorization["signed_message_hex"]
                )
            except ValueError as exc:
                raise ValidatorHotkeyAuthorityV2Error(
                    "recovery extrinsic signature encoding is invalid"
                ) from exc
            if len(signature) != 64 or not self._sr25519.verify(
                signature, signed_message, self.hotkey_public_key
            ):
                raise ValidatorHotkeyAuthorityV2Error(
                    "recovery extrinsic signature is invalid"
                )
            signed_extrinsic = encode_signed_extrinsic_v2(
                hotkey_public_key_hex=self.hotkey_public_key.hex(),
                signature_hex=output["signature"],
                era_period=int(authorization["era_period"]),
                era_current=int(authorization["era_current"]),
                nonce=int(authorization["nonce"]),
                call_data_hex=str(authorization["call_data_hex"]),
            )
            if signed_extrinsic_hash_v2(signed_extrinsic) != output["extrinsic_hash"]:
                raise ValidatorHotkeyAuthorityV2Error(
                    "recovery signed extrinsic hash differs"
                )
            receipt = dict(item["receipt"])
            validate_signed_execution_receipt(receipt)
            if (
                receipt.get("role") != "validator_weights"
                or receipt.get("purpose") != "validator.set_weights_extrinsic.v2"
                or receipt.get("boot_identity_hash")
                != old_boot["boot_identity_hash"]
                or receipt.get("enclave_pubkey") != old_boot["signing_pubkey"]
                or receipt.get("input_root") != authorization["authorization_hash"]
                or receipt.get("output_root") != sha256_json(output)
                or receipt.get("parent_receipt_hashes")
                != [computed_receipt["receipt_hash"]]
                or int(receipt.get("epoch_id", -1))
                != int(weight_result["epoch_id"])
            ):
                raise ValidatorHotkeyAuthorityV2Error(
                    "recovery extrinsic receipt is invalid"
                )
            if (
                authorization["authorization_hash"] in seen_authorizations
                or output["extrinsic_hash"] in seen_extrinsics
            ):
                raise ValidatorHotkeyAuthorityV2Error(
                    "recovery contains duplicate signed extrinsics"
                )
            seen_authorizations.add(authorization["authorization_hash"])
            seen_extrinsics.add(output["extrinsic_hash"])
            normalized_signed.append(
                {
                    "authorization": authorization,
                    "output": output,
                    "receipt": receipt,
                    "extrinsic_hex": signed_extrinsic.hex(),
                }
            )
        recovery_id = sha256_json(
            {
                "schema_version": "leadpoet.validator_weight_recovery.v2",
                "current_boot_identity_hash": current_boot["boot_identity_hash"],
                "bundle_hash": verified["bundle_hash"],
                "weight_submission_event_hash": event_hash,
                "extrinsic_hashes": [
                    item["output"]["extrinsic_hash"] for item in normalized_signed
                ],
            }
        )
        with self._lock:
            existing = self._weights.get(recovery_id)
            if existing is None:
                if len(self._weights) >= MAX_PENDING_WEIGHT_AUTHORIZATIONS:
                    raise ValidatorHotkeyAuthorityV2Error(
                        "pending weight authorization capacity is full"
                    )
                self._weights[recovery_id] = {
                    "weight_snapshot": dict(published_bundle["weight_snapshot"]),
                    "weight_result": weight_result,
                    "root_receipt_hash": computed_receipt["receipt_hash"],
                    "receipt_graph": graph,
                    "attempts": len(normalized_signed),
                    "finalization": None,
                }
                for index, signed in enumerate(normalized_signed):
                    commit_id = sha256_json(
                        {
                            "recovery_id": recovery_id,
                            "authorization_hash": signed["authorization"][
                                "authorization_hash"
                            ],
                            "index": index,
                        }
                    )
                    self._commits[commit_id] = {
                        "weight_authorization_id": recovery_id,
                        "weight_submission_event_hash": event_hash,
                        "commitment": bytes.fromhex(
                            signed["authorization"]["commitment_hex"]
                        ),
                        "reveal_round": int(
                            signed["authorization"]["reveal_round"]
                        ),
                        "consumed": True,
                        "signed_result": signed,
                    }
            else:
                if (
                    existing["root_receipt_hash"]
                    != computed_receipt["receipt_hash"]
                    or existing["weight_result"] != weight_result
                ):
                    raise ValidatorHotkeyAuthorityV2Error(
                        "recovered weight authorization conflicts"
                    )
        return {
            "weight_authorization_id": recovery_id,
            "weight_submission_event_hash": event_hash,
            "signed_extrinsics": [
                {
                    "authorization_hash": item["authorization"][
                        "authorization_hash"
                    ],
                    "extrinsic_hash": item["output"]["extrinsic_hash"],
                    "extrinsic_hex": item["extrinsic_hex"],
                }
                for item in normalized_signed
            ],
        }

    def sign_application_message(
        self,
        *,
        message_hex: str,
        parent_receipt_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        keypair = self._require_keypair()
        try:
            message = bytes.fromhex(str(message_hex or ""))
        except ValueError as exc:
            raise ValidatorHotkeyAuthorityV2Error(
                "application message is invalid hex"
            ) from exc
        boot = dict(self._boot_identity_supplier())
        request = build_application_signature_request_v2(
            message=message,
            validator_hotkey=self.validator_hotkey,
            boot_identity_hash=boot["boot_identity_hash"],
        )
        parents = []
        if request["purpose"] == "validator.gateway_binding.v2":
            parent = str(parent_receipt_hash or "")
            with self._lock:
                records = [
                    item
                    for item in self._weights.values()
                    if item["root_receipt_hash"] == parent
                ]
            if len(records) != 1:
                raise ValidatorHotkeyAuthorityV2Error(
                    "gateway binding lacks an authoritative weight parent"
                )
            parsed, parts, _error = parse_binding_message(message.decode("utf-8"))
            record = records[0]
            if (
                not parsed
                or not isinstance(parts, Mapping)
                or int(parts.get("netuid", -1))
                != int(record["weight_result"]["netuid"])
                or parts.get("chain") != self.chain_profile["chain_endpoint"]
                or parts.get("enclave_pubkey") != boot["signing_pubkey"]
                or parts.get("validator_code_hash")
                != boot["build_manifest_hash"]
                or parts.get("version") != boot["commit_sha"]
            ):
                raise ValidatorHotkeyAuthorityV2Error(
                    "gateway binding differs from authoritative weight boot"
                )
            parents = [parent]
        elif parent_receipt_hash is not None:
            raise ValidatorHotkeyAuthorityV2Error(
                "application signature parent is not allowed"
            )
        signature = bytes(self._sr25519.sign(keypair, message))
        if len(signature) != 64:
            raise ValidatorHotkeyAuthorityV2Error(
                "sr25519 application signature is invalid"
            )
        output = {
            "schema_version": "leadpoet.application_signature_result.v2",
            "request_hash": request["request_hash"],
            "purpose": request["purpose"],
            "validator_hotkey": self.validator_hotkey,
            "signature": signature.hex(),
        }
        receipt = self._receipt(
            boot=boot,
            purpose="validator.hotkey_signature.v2",
            job_id="hotkey-signature:%s" % request["request_hash"].split(":", 1)[1],
            epoch_id=(
                int(records[0]["weight_result"]["epoch_id"])
                if request["purpose"] == "validator.gateway_binding.v2"
                else 0
            ),
            sequence=0,
            input_root=request["request_hash"],
            output_root=sha256_json(output),
            parent_receipt_hashes=parents,
        )
        return {**output, "receipt": receipt}

    def prepare_weight_commit(
        self,
        *,
        weight_authorization_id: str,
        weight_submission_event_hash: str,
        uids: Sequence[int],
        weights_u16: Sequence[int],
        version_key: int,
        last_epoch_block: int,
        pending_epoch_at: int,
        subnet_epoch_index: int,
        tempo: int,
        blocks_since_last_step: int,
        current_block: int,
        subnet_reveal_period_epochs: int,
        block_time: float,
        hotkey_public_key_hex: str,
    ) -> Dict[str, Any]:
        self._require_keypair()
        event_hash = str(weight_submission_event_hash or "").lower()
        if not _HASH_RE.fullmatch(event_hash):
            raise ValidatorHotkeyAuthorityV2Error(
                "weight submission event hash is invalid"
            )
        profile = self.chain_profile
        max_attempts = (
            int(profile["max_snapshot_block_drift"]) + 1
        ) * SDK_SET_WEIGHTS_RETRIES_PER_CALL
        with self._lock:
            record = self._weights.get(str(weight_authorization_id or ""))
            if record is None:
                raise ValidatorHotkeyAuthorityV2Error(
                    "weight authorization was not found"
                )
            if record["attempts"] >= max_attempts:
                raise ValidatorHotkeyAuthorityV2Error(
                    "weight extrinsic attempt limit is exhausted"
                )
            result = dict(record["weight_result"])
            attempt_number = int(record["attempts"])
        normalized_uids = [int(item) for item in uids]
        normalized_weights = [int(item) for item in weights_u16]
        if (
            normalized_uids != list(result["sparse_uids"])
            or normalized_weights != list(result["sparse_weights_u16"])
        ):
            raise ValidatorHotkeyAuthorityV2Error(
                "SDK weight vector differs from enclave computation"
            )
        schedule = {
            "last_epoch_block": int(last_epoch_block),
            "pending_epoch_at": int(pending_epoch_at),
            "subnet_epoch_index": int(subnet_epoch_index),
            "tempo": int(tempo),
            "blocks_since_last_step": int(blocks_since_last_step),
            "current_block": int(current_block),
        }
        if (
            int(version_key) != int(profile["version_key"])
            or schedule["tempo"] != int(profile["tempo"])
            or int(subnet_reveal_period_epochs)
            != int(profile["subnet_reveal_period_epochs"])
            or round(float(block_time) * 1000)
            != int(profile["block_time_millis"])
            or bytes.fromhex(str(hotkey_public_key_hex or ""))
            != self.hotkey_public_key
        ):
            raise ValidatorHotkeyAuthorityV2Error(
                "SDK chain parameters differ from measured profile"
            )
        if (
            any(value < 0 or value > 2**64 - 1 for value in schedule.values())
            or schedule["tempo"] <= 0
            or schedule["tempo"] > 65535
            or schedule["last_epoch_block"] > schedule["current_block"]
        ):
            raise ValidatorHotkeyAuthorityV2Error(
                "SDK epoch schedule state is invalid"
            )
        block = schedule["current_block"]
        snapshot_block = int(result["block"])
        if not (
            snapshot_block
            <= block
            <= snapshot_block + int(profile["max_snapshot_block_drift"])
        ):
            raise ValidatorHotkeyAuthorityV2Error(
                "SDK current block is outside weight authorization window"
            )
        commitment, reveal_round = self._drand.generate_commit(
            uids=normalized_uids,
            weights_u16=normalized_weights,
            version_key=int(version_key),
            last_epoch_block=schedule["last_epoch_block"],
            pending_epoch_at=schedule["pending_epoch_at"],
            subnet_epoch_index=schedule["subnet_epoch_index"],
            tempo=schedule["tempo"],
            blocks_since_last_step=schedule["blocks_since_last_step"],
            current_block=schedule["current_block"],
            subnet_reveal_period_epochs=int(subnet_reveal_period_epochs),
            block_time=float(block_time),
            hotkey_public_key=self.hotkey_public_key,
        )
        commit_id = sha256_json(
            {
                "weight_authorization_id": weight_authorization_id,
                "weight_submission_event_hash": event_hash,
                "commitment_hash": sha256_bytes(commitment),
                "reveal_round": int(reveal_round),
                "epoch_schedule": schedule,
                "attempt": attempt_number,
            }
        )
        with self._lock:
            current = self._weights.get(str(weight_authorization_id))
            if current is None or current["attempts"] != attempt_number:
                raise ValidatorHotkeyAuthorityV2Error(
                    "weight authorization changed while preparing commit"
                )
            current["attempts"] += 1
            self._commits[commit_id] = {
                "weight_authorization_id": str(weight_authorization_id),
                "weight_submission_event_hash": event_hash,
                "commitment": bytes(commitment),
                "reveal_round": int(reveal_round),
                "epoch_schedule": dict(schedule),
                "consumed": False,
            }
        return {
            "commit_authorization_id": commit_id,
            "commitment_hex": bytes(commitment).hex(),
            "reveal_round": int(reveal_round),
        }

    def sign_weight_extrinsic(
        self,
        *,
        commit_authorization_id: str,
        era_current: int,
        nonce: int,
        block_hash: str,
        signature_payload_hex: str,
    ) -> Dict[str, Any]:
        keypair = self._require_keypair()
        with self._lock:
            commit = self._commits.get(str(commit_authorization_id or ""))
            if commit is None:
                raise ValidatorHotkeyAuthorityV2Error(
                    "extrinsic commitment authorization was not found"
                )
            if commit["consumed"]:
                raise ValidatorHotkeyAuthorityV2Error(
                    "extrinsic commitment authorization was already used"
                )
            weight = self._weights.get(commit["weight_authorization_id"])
            if weight is None:
                raise ValidatorHotkeyAuthorityV2Error(
                    "parent weight authorization was not found"
                )
            result = dict(weight["weight_result"])
        authorization = build_weight_extrinsic_authorization_v2(
            profile=self.chain_profile,
            validator_hotkey=self.validator_hotkey,
            hotkey_public_key_hex=self.hotkey_public_key.hex(),
            epoch_id=int(result["epoch_id"]),
            netuid=int(result["netuid"]),
            subnet_epoch_index=int(
                commit["epoch_schedule"]["subnet_epoch_index"]
            ),
            weight_receipt_hash=weight["root_receipt_hash"],
            weight_submission_event_hash=commit["weight_submission_event_hash"],
            weights_hash=str(result["weights_hash"]),
            sparse_uids=result["sparse_uids"],
            sparse_weights_u16=result["sparse_weights_u16"],
            commitment=commit["commitment"],
            reveal_round=commit["reveal_round"],
            era_current=int(era_current),
            nonce=int(nonce),
            block_hash=str(block_hash),
        )
        try:
            payload = bytes.fromhex(str(signature_payload_hex or ""))
        except ValueError as exc:
            raise ValidatorHotkeyAuthorityV2Error(
                "SDK signature payload is invalid hex"
            ) from exc
        if payload.hex() != authorization["signed_message_hex"]:
            raise ValidatorHotkeyAuthorityV2Error(
                "SDK signature payload differs from authorized SCALE payload"
            )
        signature = bytes(self._sr25519.sign(keypair, payload))
        if len(signature) != 64:
            raise ValidatorHotkeyAuthorityV2Error(
                "sr25519 extrinsic signature is invalid"
            )
        signed_extrinsic = encode_signed_extrinsic_v2(
            hotkey_public_key_hex=self.hotkey_public_key.hex(),
            signature_hex=signature.hex(),
            era_period=int(authorization["era_period"]),
            era_current=int(authorization["era_current"]),
            nonce=int(authorization["nonce"]),
            call_data_hex=str(authorization["call_data_hex"]),
        )
        extrinsic_hash = signed_extrinsic_hash_v2(signed_extrinsic)
        output = {
            "schema_version": "leadpoet.weight_extrinsic_signature.v2",
            "authorization_hash": authorization["authorization_hash"],
            "validator_hotkey": self.validator_hotkey,
            "signature": signature.hex(),
            "extrinsic_hash": extrinsic_hash,
        }
        receipt = self._receipt(
            boot=dict(self._boot_identity_supplier()),
            purpose="validator.set_weights_extrinsic.v2",
            job_id="set-weights:%s:%s"
            % (result["epoch_id"], authorization["authorization_hash"].split(":", 1)[1]),
            epoch_id=int(result["epoch_id"]),
            sequence=int(weight["attempts"]),
            input_root=authorization["authorization_hash"],
            output_root=sha256_json(output),
            parent_receipt_hashes=[weight["root_receipt_hash"]],
        )
        with self._lock:
            current = self._commits.get(str(commit_authorization_id))
            if current is None or current["consumed"]:
                raise ValidatorHotkeyAuthorityV2Error(
                    "extrinsic commitment authorization changed"
                )
            current["consumed"] = True
            current["signed_result"] = {
                "authorization": dict(authorization),
                "output": dict(output),
                "receipt": dict(receipt),
                "extrinsic_hex": signed_extrinsic.hex(),
            }
        return {
            **output,
            "authorization": authorization,
            "receipt": receipt,
        }

    def confirm_weight_publication(
        self, *, weight_authorization_id: str
    ) -> Dict[str, Any]:
        """Prove one authorized extrinsic appeared in an authenticated finalized block."""

        if self._chain_source is None:
            raise ValidatorHotkeyAuthorityV2Error(
                "validator finalized chain source is unavailable"
            )
        authorization_id = str(weight_authorization_id or "")
        with self._lock:
            weight = self._weights.get(authorization_id)
            if weight is None:
                raise ValidatorHotkeyAuthorityV2Error(
                    "weight authorization was not found"
                )
            if weight.get("finalization") is not None:
                return dict(weight["finalization"])
            signed_results = [
                dict(commit["signed_result"])
                for commit in self._commits.values()
                if commit.get("weight_authorization_id") == authorization_id
                and isinstance(commit.get("signed_result"), Mapping)
            ]
            weight_result = dict(weight["weight_result"])
            original_graph = dict(weight["receipt_graph"])
        if not signed_results:
            raise ValidatorHotkeyAuthorityV2Error(
                "weight authorization has no signed extrinsic"
            )
        signed_results.sort(
            key=lambda item: item["authorization"]["authorization_hash"]
        )
        expected_extrinsics = {
            str(item["output"]["extrinsic_hash"]): str(item["extrinsic_hex"])
            for item in signed_results
        }
        expected_commitments = {
            str(item["output"]["extrinsic_hash"]): {
                "netuid": int(item["authorization"]["netuid"]),
                "subnet_epoch_index": int(
                    item["authorization"]["subnet_epoch_index"]
                ),
                "hotkey_public_key": self.hotkey_public_key.hex(),
                "commitment_hex": str(item["authorization"]["commitment_hex"]),
                "reveal_round": int(item["authorization"]["reveal_round"]),
            }
            for item in signed_results
        }
        bounds = [
            mortal_era_bounds(
                period=int(item["authorization"]["era_period"]),
                current=int(item["authorization"]["era_current"]),
            )
            for item in signed_results
        ]
        try:
            inclusion = self._chain_source.find_finalized_extrinsic_inclusion(
                expected_extrinsics=expected_extrinsics,
                expected_commitments=expected_commitments,
                minimum_block=min(item[0] for item in bounds),
                maximum_block=max(item[1] - 1 for item in bounds),
                epoch_id=int(weight_result["epoch_id"]),
            )
        except Exception as exc:
            raise ValidatorHotkeyAuthorityV2Error(
                "authorized weight extrinsic finalization was not proven"
            ) from exc
        included = [
            item
            for item in signed_results
            if item["output"]["extrinsic_hash"] == inclusion["extrinsic_hash"]
        ]
        if len(included) != 1:
            raise ValidatorHotkeyAuthorityV2Error(
                "finalized weight extrinsic is not uniquely authorized"
            )
        included_result = included[0]
        transport_attempts = [dict(item) for item in inclusion["attempts"]]
        for attempt in transport_attempts:
            validate_transport_attempt(attempt)
        source_artifacts = self._validate_source_artifacts(
            inclusion["artifacts"], transport_attempts
        )
        finalization = {
            "schema_version": "leadpoet.weight_finalization.v2",
            "validator_hotkey": self.validator_hotkey,
            "netuid": int(weight_result["netuid"]),
            "epoch_id": int(weight_result["epoch_id"]),
            "weights_hash": str(weight_result["weights_hash"]),
            "weight_receipt_hash": str(weight["root_receipt_hash"]),
            "weight_submission_event_hash": str(
                included_result["authorization"]["weight_submission_event_hash"]
            ),
            "extrinsic_authorization": dict(
                included_result["authorization"]
            ),
            "extrinsic_authorization_hash": str(
                included_result["authorization"]["authorization_hash"]
            ),
            "extrinsic_signature": str(
                included_result["output"]["signature"]
            ),
            "extrinsic_receipt_hash": str(
                included_result["receipt"]["receipt_hash"]
            ),
            "extrinsic_hash": str(inclusion["extrinsic_hash"]),
            "finalized_block": int(inclusion["finalized_block"]),
            "finalized_block_hash": str(inclusion["finalized_block_hash"]),
            "state_transition_hash": str(inclusion["state_transition_hash"]),
        }
        extrinsic_receipts = [dict(item["receipt"]) for item in signed_results]
        artifact_hashes = [
            sha256_bytes(bytes.fromhex(str(inclusion["extrinsic_hex"])))
        ] + [str(item["artifact_hash"]) for item in source_artifacts]
        final_receipt = self._receipt(
            boot=dict(self._boot_identity_supplier()),
            purpose="validator.weights.finalized.v2",
            job_id=str(inclusion["job_id"]),
            epoch_id=int(weight_result["epoch_id"]),
            sequence=len(extrinsic_receipts),
            input_root=sha256_json(
                {
                    "weight_submission_event_hash": finalization[
                        "weight_submission_event_hash"
                    ],
                    "extrinsic_receipt_hashes": sorted(
                        item["receipt_hash"] for item in extrinsic_receipts
                    ),
                }
            ),
            output_root=sha256_json(finalization),
            parent_receipt_hashes=sorted(
                item["receipt_hash"] for item in extrinsic_receipts
            ),
            transport_attempts=transport_attempts,
            artifact_hashes=artifact_hashes,
        )
        original_receipts = [dict(item) for item in original_graph["receipts"]]
        boot_identities = {
            str(identity["boot_identity_hash"]): dict(identity)
            for identity in original_graph["boot_identities"]
        }
        current_boot = dict(self._boot_identity_supplier())
        boot_identities[current_boot["boot_identity_hash"]] = current_boot
        graph = build_receipt_graph(
            root_receipt_hash=final_receipt["receipt_hash"],
            boot_identities=[
                boot_identities[key] for key in sorted(boot_identities)
            ],
            receipts=original_receipts + extrinsic_receipts + [final_receipt],
            transport_attempts=list(original_graph["transport_attempts"])
            + transport_attempts,
            host_operations=original_graph["host_operations"],
        )
        validate_receipt_graph(
            graph,
            required_purposes={
                "validator.weights.computed.v2",
                "validator.set_weights_extrinsic.v2",
                "validator.weights.finalized.v2",
            },
        )
        result = {
            "finalization": finalization,
            "receipt_graph": graph,
            "source_artifacts": source_artifacts,
        }
        with self._lock:
            current = self._weights.get(authorization_id)
            if current is None:
                raise ValidatorHotkeyAuthorityV2Error(
                    "weight authorization disappeared during finalization"
                )
            current["finalization"] = result
        return dict(result)

    @staticmethod
    def _validate_source_artifacts(value: Any, attempts: Any) -> list:
        if not isinstance(value, list):
            raise ValidatorHotkeyAuthorityV2Error(
                "chain finalization artifacts are invalid"
            )
        by_hash = {}
        for item in value:
            if not isinstance(item, Mapping) or set(item) != {
                "artifact_hash",
                "kind",
                "body_b64",
            }:
                raise ValidatorHotkeyAuthorityV2Error(
                    "chain finalization artifact fields are invalid"
                )
            try:
                body = base64.b64decode(str(item["body_b64"]), validate=True)
            except Exception as exc:
                raise ValidatorHotkeyAuthorityV2Error(
                    "chain finalization artifact body is invalid"
                ) from exc
            if sha256_bytes(body) != item["artifact_hash"]:
                raise ValidatorHotkeyAuthorityV2Error(
                    "chain finalization artifact hash differs"
                )
            by_hash[str(item["artifact_hash"])] = dict(item)
        required = set()
        for attempt in attempts:
            required.add(str(attempt["request_artifact_hash"]))
            if attempt.get("terminal_status") == "authenticated_response":
                required.add(str(attempt["response_artifact_hash"]))
        if set(by_hash) != required:
            raise ValidatorHotkeyAuthorityV2Error(
                "chain finalization artifact set is incomplete or contains extras"
            )
        return [by_hash[key] for key in sorted(by_hash)]

    def sign_serve_axon_extrinsic(
        self,
        *,
        netuid: int,
        version: int,
        ip: int,
        port: int,
        ip_type: int,
        protocol: int,
        placeholder1: int,
        placeholder2: int,
        era_current: int,
        nonce: int,
        block_hash: str,
        signature_payload_hex: str,
    ) -> Dict[str, Any]:
        """Sign only the exact measured ``serve_axon`` SCALE payload."""

        keypair = self._require_keypair()
        authorization = build_serve_axon_extrinsic_authorization_v2(
            profile=self.chain_profile,
            validator_hotkey=self.validator_hotkey,
            hotkey_public_key_hex=self.hotkey_public_key.hex(),
            netuid=int(netuid),
            version=int(version),
            ip=int(ip),
            port=int(port),
            ip_type=int(ip_type),
            protocol=int(protocol),
            placeholder1=int(placeholder1),
            placeholder2=int(placeholder2),
            era_current=int(era_current),
            nonce=int(nonce),
            block_hash=str(block_hash),
        )
        try:
            payload = bytes.fromhex(str(signature_payload_hex or ""))
        except ValueError as exc:
            raise ValidatorHotkeyAuthorityV2Error(
                "SDK serve axon signature payload is invalid hex"
            ) from exc
        if payload.hex() != authorization["signed_message_hex"]:
            raise ValidatorHotkeyAuthorityV2Error(
                "SDK serve axon payload differs from the measured call"
            )
        with self._lock:
            if self._serve_axon_signature_count >= MAX_SERVE_AXON_SIGNATURES_PER_BOOT:
                raise ValidatorHotkeyAuthorityV2Error(
                    "serve axon signature limit is exhausted"
                )
            sequence = self._serve_axon_signature_count
            self._serve_axon_signature_count += 1
        signature = bytes(self._sr25519.sign(keypair, payload))
        if len(signature) != 64:
            raise ValidatorHotkeyAuthorityV2Error(
                "sr25519 serve axon signature is invalid"
            )
        output = {
            "schema_version": "leadpoet.serve_axon_signature.v2",
            "authorization_hash": authorization["authorization_hash"],
            "validator_hotkey": self.validator_hotkey,
            "signature": signature.hex(),
        }
        receipt = self._receipt(
            boot=dict(self._boot_identity_supplier()),
            purpose="validator.serve_axon_extrinsic.v2",
            job_id="serve-axon:%s"
            % authorization["authorization_hash"].split(":", 1)[1],
            epoch_id=0,
            sequence=sequence,
            input_root=authorization["authorization_hash"],
            output_root=sha256_json(output),
            parent_receipt_hashes=(),
        )
        return {
            **output,
            "authorization": authorization,
            "receipt": receipt,
        }

    def _receipt(
        self,
        *,
        boot: Mapping[str, Any],
        purpose: str,
        job_id: str,
        epoch_id: int,
        sequence: int,
        input_root: str,
        output_root: str,
        parent_receipt_hashes: Sequence[str],
        transport_attempts: Sequence[Mapping[str, Any]] = (),
        artifact_hashes: Sequence[str] = (),
    ) -> Dict[str, Any]:
        attempt_hashes = []
        for attempt in transport_attempts:
            validate_transport_attempt(attempt)
            attempt_hashes.append(str(attempt["attempt_hash"]))
        body = build_execution_receipt_body(
            role="validator_weights",
            purpose=purpose,
            job_id=job_id,
            epoch_id=int(epoch_id),
            sequence=int(sequence),
            commit_sha=boot["commit_sha"],
            pcr0=boot["pcr0"],
            build_manifest_hash=boot["build_manifest_hash"],
            dependency_lock_hash=boot["dependency_lock_hash"],
            config_hash=boot["config_hash"],
            boot_identity_hash=boot["boot_identity_hash"],
            input_root=input_root,
            output_root=output_root,
            transport_root_hash=(
                merkle_root(attempt_hashes, domain="leadpoet-transport-v2")
                if attempt_hashes
                else EMPTY_TRANSPORT_ROOT
            ),
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=(
                merkle_root(artifact_hashes, domain="leadpoet-artifact-v2")
                if artifact_hashes
                else EMPTY_ARTIFACT_ROOT
            ),
            parent_receipt_hashes=parent_receipt_hashes,
            status="succeeded",
            failure_code=None,
            issued_at=_issued_at(self._clock),
        )
        return create_signed_execution_receipt(
            body=body,
            enclave_pubkey=boot["signing_pubkey"],
            sign_digest=self._sign_receipt_digest,
        )
