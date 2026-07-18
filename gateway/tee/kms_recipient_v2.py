"""Nitro KMS recipient envelopes for coordinator-only provider credentials."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading
from typing import Any, Callable, Dict, Mapping, Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from gateway.tee.provider_broker_v2 import credential_reference_hash
from gateway.tee.provider_broker_v2 import credential_value_hash
from gateway.tee.source_add_runtime_v2 import source_add_dynamic_job_slot
from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from leadpoet_canonical.kms_recipient import decrypt_kms_recipient_ciphertext


KMS_RECIPIENT_SCHEMA_VERSION = "leadpoet.kms_recipient.v2"
KMS_RECIPIENT_PURPOSE = "leadpoet.provider_credential_unseal.v2"
KMS_JOB_RECIPIENT_SCHEMA_VERSION = "leadpoet.kms_job_recipient.v2"
KMS_JOB_RECIPIENT_PURPOSE = "leadpoet.job_provider_credential_unseal.v2"
SOURCE_ADD_INGRESS_RECIPIENT_SCHEMA_VERSION = (
    "leadpoet.source_add_ingress_recipient.v2"
)
SOURCE_ADD_INGRESS_RECIPIENT_PURPOSE = (
    "leadpoet.source_add_credential_ingress.v2"
)
OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION = (
    "leadpoet.openrouter_ingress_recipient.v2"
)
OPENROUTER_INGRESS_RECIPIENT_PURPOSE = (
    "leadpoet.openrouter_credential_ingress.v2"
)
KMS_KEY_ENCRYPTION_ALGORITHM = "RSAES_OAEP_SHA_256"
MAX_CIPHERTEXT_FOR_RECIPIENT_BYTES = 64 * 1024
MAX_CREDENTIAL_BYTES = 64 * 1024
MAX_JOB_RECIPIENT_REQUESTS = 2048
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class KMSRecipientV2Error(RuntimeError):
    """A KMS recipient request or returned envelope is invalid."""


class KMSRecipientV2:
    """Generate one boot-local RSA recipient key and unwrap KMS responses."""

    def __init__(
        self,
        *,
        boot_identity_supplier: Callable[[], Mapping[str, Any]],
        expected_credential_ref_hashes: Mapping[str, str],
        expected_job_slot_ref_hashes: Optional[Mapping[str, str]] = None,
        expected_binary_ref_hashes: Optional[Mapping[str, str]] = None,
        attestation_supplier: Callable[..., bytes],
    ) -> None:
        self._boot_identity_supplier = boot_identity_supplier
        self._expected = {
            str(slot): str(digest or "").strip().lower()
            for slot, digest in expected_credential_ref_hashes.items()
        }
        self._binary_slots = {
            str(slot): str(digest or "").strip().lower()
            for slot, digest in (expected_binary_ref_hashes or {}).items()
        }
        self._job_slots = {
            str(slot): str(digest or "").strip().lower()
            for slot, digest in (expected_job_slot_ref_hashes or {}).items()
        }
        overlap = set(self._expected) & set(self._binary_slots)
        if overlap:
            raise KMSRecipientV2Error("KMS recipient slot types overlap")
        self._expected.update(self._binary_slots)
        if not self._expected or any(
            not slot or not _HASH_RE.fullmatch(digest)
            for slot, digest in self._expected.items()
        ):
            raise KMSRecipientV2Error("credential reference map is invalid")
        if any(
            not slot or not _HASH_RE.fullmatch(digest)
            for slot, digest in self._job_slots.items()
        ):
            raise KMSRecipientV2Error("job credential slot map is invalid")
        self._attestation_supplier = attestation_supplier
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=3072,
        )
        self._public_der = self._private_key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self._requests = {}  # type: Dict[str, Dict[str, Any]]
        self._job_requests = {}  # type: Dict[str, Dict[str, Any]]
        self._source_add_requests = {}  # type: Dict[str, Dict[str, Any]]
        self._openrouter_requests = {}  # type: Dict[str, Dict[str, Any]]
        self._provisioned = set()
        self._lock = threading.Lock()

    def recipient_request(self, slot: str) -> Dict[str, Any]:
        normalized_slot = str(slot or "")
        if normalized_slot not in self._expected:
            raise KMSRecipientV2Error("credential slot is not measured")
        with self._lock:
            boot = dict(self._boot_identity_supplier())
            public_hash = "sha256:" + hashlib.sha256(self._public_der).hexdigest()
            claim = {
                "schema_version": KMS_RECIPIENT_SCHEMA_VERSION,
                "purpose": KMS_RECIPIENT_PURPOSE,
                "boot_identity_hash": str(boot["boot_identity_hash"]),
                "credential_slot": normalized_slot,
                "credential_ref_hash": self._expected[normalized_slot],
                "recipient_public_key_hash": public_hash,
                "request_nonce": secrets.token_hex(16),
            }
            user_data = canonical_json(
                {
                    "schema_version": KMS_RECIPIENT_SCHEMA_VERSION,
                    "purpose": KMS_RECIPIENT_PURPOSE,
                    "claim_hash": sha256_json(claim),
                }
            ).encode("utf-8")
            document = self._attestation_supplier(
                user_data=user_data,
                signing_pubkey=self._public_der,
            )
            if not isinstance(document, (bytes, bytearray)) or not document:
                raise KMSRecipientV2Error("KMS recipient attestation is unavailable")
            request = {
                **claim,
                "recipient_public_key_der_b64": base64.b64encode(
                    self._public_der
                ).decode("ascii"),
                "attestation_document_b64": base64.b64encode(bytes(document)).decode(
                    "ascii"
                ),
                "key_encryption_algorithm": KMS_KEY_ENCRYPTION_ALGORITHM,
            }
            self._requests[normalized_slot] = request
            return dict(request)

    def _unwrap(
        self,
        *,
        slot: str,
        ciphertext_for_recipient_b64: str,
    ) -> bytes:
        normalized_slot = str(slot or "")
        if (
            normalized_slot not in self._expected
            and normalized_slot not in self._job_slots
            and not source_add_dynamic_job_slot(normalized_slot)
            and normalized_slot != "source_add_ingress"
            and normalized_slot != "openrouter_ingress"
        ):
            raise KMSRecipientV2Error("credential slot is not measured")
        try:
            ciphertext = base64.b64decode(
                str(ciphertext_for_recipient_b64 or ""),
                validate=True,
            )
        except Exception as exc:
            raise KMSRecipientV2Error("KMS recipient ciphertext is invalid") from exc
        if not ciphertext or len(ciphertext) > MAX_CIPHERTEXT_FOR_RECIPIENT_BYTES:
            raise KMSRecipientV2Error("KMS recipient ciphertext is outside limit")
        try:
            plaintext = decrypt_kms_recipient_ciphertext(
                self._private_key,
                ciphertext,
            )
        except Exception as exc:
            raise KMSRecipientV2Error("KMS recipient unwrap failed") from exc
        if not plaintext or len(plaintext) > MAX_CREDENTIAL_BYTES:
            raise KMSRecipientV2Error("unwrapped secret is invalid")
        return bytes(plaintext)

    def source_add_ingress_recipient_request(
        self,
        *,
        miner_hotkey: str,
        adapter_ref: str,
        credential_ref: str,
    ) -> Dict[str, Any]:
        """Create a one-use client recipient without exposing a secret hash."""

        normalized_miner = str(miner_hotkey or "")
        normalized_adapter = str(adapter_ref or "")
        normalized_ref = str(credential_ref or "")
        if (
            not normalized_miner
            or not re.fullmatch(r"source_add:[A-Za-z0-9_.:-]{1,200}", normalized_adapter)
            or not re.fullmatch(r"encrypted_ref:source_add:[0-9a-f]{32}", normalized_ref)
        ):
            raise KMSRecipientV2Error("SOURCE_ADD ingress scope is invalid")
        boot = dict(self._boot_identity_supplier())
        public_hash = "sha256:" + hashlib.sha256(self._public_der).hexdigest()
        claim = {
            "schema_version": SOURCE_ADD_INGRESS_RECIPIENT_SCHEMA_VERSION,
            "purpose": SOURCE_ADD_INGRESS_RECIPIENT_PURPOSE,
            "boot_identity_hash": str(boot["boot_identity_hash"]),
            "miner_hotkey_hash": "sha256:"
            + hashlib.sha256(normalized_miner.encode("utf-8")).hexdigest(),
            "adapter_ref_hash": "sha256:"
            + hashlib.sha256(normalized_adapter.encode("utf-8")).hexdigest(),
            "credential_ref": normalized_ref,
            "key_ref_hash": "sha256:"
            + hashlib.sha256(normalized_ref.encode("utf-8")).hexdigest(),
            "recipient_public_key_hash": public_hash,
            "request_nonce": secrets.token_hex(16),
        }
        request_id = sha256_json(claim)
        user_data = canonical_json(
            {
                "schema_version": SOURCE_ADD_INGRESS_RECIPIENT_SCHEMA_VERSION,
                "purpose": SOURCE_ADD_INGRESS_RECIPIENT_PURPOSE,
                "claim_hash": request_id,
            }
        ).encode("utf-8")
        document = self._attestation_supplier(
            user_data=user_data,
            signing_pubkey=self._public_der,
        )
        if not isinstance(document, (bytes, bytearray)) or not document:
            raise KMSRecipientV2Error(
                "SOURCE_ADD ingress recipient attestation is unavailable"
            )
        request = {
            **claim,
            "request_id": request_id,
            "recipient_public_key_der_b64": base64.b64encode(
                self._public_der
            ).decode("ascii"),
            "attestation_document_b64": base64.b64encode(bytes(document)).decode(
                "ascii"
            ),
            "key_encryption_algorithm": KMS_KEY_ENCRYPTION_ALGORITHM,
        }
        with self._lock:
            if len(self._source_add_requests) >= MAX_JOB_RECIPIENT_REQUESTS:
                raise KMSRecipientV2Error(
                    "SOURCE_ADD ingress recipient capacity is full"
                )
            self._source_add_requests[request_id] = {
                "request": dict(request),
                "miner_hotkey": normalized_miner,
                "adapter_ref": normalized_adapter,
                "used": False,
            }
        return request

    def unwrap_source_add_ingress_credential(
        self,
        *,
        request_id: str,
        ciphertext_b64: str,
    ) -> Dict[str, str]:
        normalized_request_id = str(request_id or "").lower()
        with self._lock:
            record = self._source_add_requests.get(normalized_request_id)
            if record is None:
                raise KMSRecipientV2Error(
                    "SOURCE_ADD ingress recipient request was not found"
                )
            if record["used"]:
                raise KMSRecipientV2Error(
                    "SOURCE_ADD ingress recipient request was already used"
                )
            request = dict(record["request"])
            miner_hotkey = str(record["miner_hotkey"])
            adapter_ref = str(record["adapter_ref"])
        plaintext = self._unwrap(
            slot="source_add_ingress",
            ciphertext_for_recipient_b64=str(ciphertext_b64 or ""),
        )
        try:
            credential = plaintext.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise KMSRecipientV2Error(
                "SOURCE_ADD ingress credential is not UTF-8"
            ) from exc
        if not credential or "\x00" in credential:
            raise KMSRecipientV2Error("SOURCE_ADD ingress credential is invalid")
        with self._lock:
            current = self._source_add_requests.get(normalized_request_id)
            if current is None or current["used"]:
                raise KMSRecipientV2Error("SOURCE_ADD ingress recipient changed")
            current["used"] = True
        return {
            "request_id": normalized_request_id,
            "miner_hotkey": miner_hotkey,
            "adapter_ref": adapter_ref,
            "credential_ref": str(request["credential_ref"]),
            "key_ref_hash": str(request["key_ref_hash"]),
            "credential_value_hash": credential_value_hash(credential),
            "credential": credential,
        }

    def openrouter_ingress_recipient_request(
        self,
        *,
        miner_hotkey: str,
        credential_kind: str,
    ) -> Dict[str, Any]:
        """Create a one-use recipient for one miner OpenRouter credential."""

        normalized_miner = str(miner_hotkey or "")
        normalized_kind = str(credential_kind or "")
        if not normalized_miner or normalized_kind not in {"runtime", "management"}:
            raise KMSRecipientV2Error("OpenRouter ingress scope is invalid")
        slot = (
            "openrouter"
            if normalized_kind == "runtime"
            else "openrouter_management"
        )
        boot = dict(self._boot_identity_supplier())
        public_hash = "sha256:" + hashlib.sha256(self._public_der).hexdigest()
        claim = {
            "schema_version": OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION,
            "purpose": OPENROUTER_INGRESS_RECIPIENT_PURPOSE,
            "boot_identity_hash": str(boot["boot_identity_hash"]),
            "miner_hotkey_hash": "sha256:"
            + hashlib.sha256(normalized_miner.encode("utf-8")).hexdigest(),
            "credential_kind": normalized_kind,
            "credential_slot": slot,
            "recipient_public_key_hash": public_hash,
            "request_nonce": secrets.token_hex(16),
        }
        request_id = sha256_json(claim)
        user_data = canonical_json(
            {
                "schema_version": OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION,
                "purpose": OPENROUTER_INGRESS_RECIPIENT_PURPOSE,
                "claim_hash": request_id,
            }
        ).encode("utf-8")
        document = self._attestation_supplier(
            user_data=user_data,
            signing_pubkey=self._public_der,
        )
        if not isinstance(document, (bytes, bytearray)) or not document:
            raise KMSRecipientV2Error(
                "OpenRouter ingress recipient attestation is unavailable"
            )
        request = {
            **claim,
            "request_id": request_id,
            "recipient_public_key_der_b64": base64.b64encode(
                self._public_der
            ).decode("ascii"),
            "attestation_document_b64": base64.b64encode(bytes(document)).decode(
                "ascii"
            ),
            "key_encryption_algorithm": KMS_KEY_ENCRYPTION_ALGORITHM,
        }
        with self._lock:
            if len(self._openrouter_requests) >= MAX_JOB_RECIPIENT_REQUESTS:
                raise KMSRecipientV2Error(
                    "OpenRouter ingress recipient capacity is full"
                )
            self._openrouter_requests[request_id] = {
                "request": dict(request),
                "miner_hotkey": normalized_miner,
                "used": False,
            }
        return request

    def unwrap_openrouter_ingress_credential(
        self,
        *,
        request_id: str,
        ciphertext_b64: str,
    ) -> Dict[str, str]:
        normalized_request_id = str(request_id or "").lower()
        with self._lock:
            record = self._openrouter_requests.get(normalized_request_id)
            if record is None:
                raise KMSRecipientV2Error(
                    "OpenRouter ingress recipient request was not found"
                )
            if record["used"]:
                raise KMSRecipientV2Error(
                    "OpenRouter ingress recipient request was already used"
                )
            request = dict(record["request"])
            miner_hotkey = str(record["miner_hotkey"])
        plaintext = self._unwrap(
            slot="openrouter_ingress",
            ciphertext_for_recipient_b64=str(ciphertext_b64 or ""),
        )
        try:
            raw_credential = plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise KMSRecipientV2Error(
                "OpenRouter ingress credential is not UTF-8"
            ) from exc
        from gateway.research_lab.key_vault import validate_openrouter_key_format

        try:
            credential = validate_openrouter_key_format(raw_credential)
        except Exception as exc:
            raise KMSRecipientV2Error(
                "OpenRouter ingress credential format is invalid"
            ) from exc
        with self._lock:
            current = self._openrouter_requests.get(normalized_request_id)
            if current is None or current["used"]:
                raise KMSRecipientV2Error(
                    "OpenRouter ingress recipient changed"
                )
            current["used"] = True
        return {
            "request_id": normalized_request_id,
            "miner_hotkey": miner_hotkey,
            "credential_kind": str(request["credential_kind"]),
            "credential_slot": str(request["credential_slot"]),
            "credential_value_hash": credential_value_hash(credential),
            "credential": credential,
        }

    def job_recipient_request(
        self,
        *,
        job_id: str,
        slot: str,
        credential_value_hash_expected: str,
        key_ref_hash: str,
    ) -> Dict[str, Any]:
        """Create a recipient bound to one miner-key/job commitment."""

        normalized_job_id = str(job_id or "")
        normalized_slot = str(slot or "")
        normalized_value_hash = str(credential_value_hash_expected or "").lower()
        normalized_key_ref_hash = str(key_ref_hash or "").lower()
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}", normalized_job_id
        ):
            raise KMSRecipientV2Error("job recipient id is invalid")
        if (
            normalized_slot not in self._expected
            and normalized_slot not in self._job_slots
            and not source_add_dynamic_job_slot(normalized_slot)
        ):
            raise KMSRecipientV2Error("job recipient slot is not measured")
        if (
            not _HASH_RE.fullmatch(normalized_value_hash)
            or not _HASH_RE.fullmatch(normalized_key_ref_hash)
        ):
            raise KMSRecipientV2Error("job recipient credential commitment is invalid")
        boot = dict(self._boot_identity_supplier())
        public_hash = "sha256:" + hashlib.sha256(self._public_der).hexdigest()
        claim = {
            "schema_version": KMS_JOB_RECIPIENT_SCHEMA_VERSION,
            "purpose": KMS_JOB_RECIPIENT_PURPOSE,
            "boot_identity_hash": str(boot["boot_identity_hash"]),
            "job_id": normalized_job_id,
            "credential_slot": normalized_slot,
            "credential_value_hash": normalized_value_hash,
            "key_ref_hash": normalized_key_ref_hash,
            "recipient_public_key_hash": public_hash,
            "request_nonce": secrets.token_hex(16),
        }
        request_id = sha256_json(claim)
        user_data = canonical_json(
            {
                "schema_version": KMS_JOB_RECIPIENT_SCHEMA_VERSION,
                "purpose": KMS_JOB_RECIPIENT_PURPOSE,
                "claim_hash": request_id,
            }
        ).encode("utf-8")
        document = self._attestation_supplier(
            user_data=user_data,
            signing_pubkey=self._public_der,
        )
        if not isinstance(document, (bytes, bytearray)) or not document:
            raise KMSRecipientV2Error("job recipient attestation is unavailable")
        request = {
            **claim,
            "request_id": request_id,
            "recipient_public_key_der_b64": base64.b64encode(
                self._public_der
            ).decode("ascii"),
            "attestation_document_b64": base64.b64encode(bytes(document)).decode(
                "ascii"
            ),
            "key_encryption_algorithm": KMS_KEY_ENCRYPTION_ALGORITHM,
        }
        with self._lock:
            if len(self._job_requests) >= MAX_JOB_RECIPIENT_REQUESTS:
                raise KMSRecipientV2Error("job recipient capacity is full")
            self._job_requests[request_id] = {
                "request": dict(request),
                "provisioned": False,
            }
        return request

    def unwrap_job_credential(
        self,
        *,
        request_id: str,
        ciphertext_for_recipient_b64: str,
    ) -> Dict[str, str]:
        normalized_request_id = str(request_id or "").lower()
        with self._lock:
            record = self._job_requests.get(normalized_request_id)
            if record is None:
                raise KMSRecipientV2Error("job recipient request was not found")
            if record["provisioned"]:
                raise KMSRecipientV2Error("job recipient request was already used")
            request = dict(record["request"])
        plaintext = self._unwrap(
            slot=str(request["credential_slot"]),
            ciphertext_for_recipient_b64=ciphertext_for_recipient_b64,
        )
        try:
            credential = plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise KMSRecipientV2Error("job credential is not UTF-8") from exc
        if credential_value_hash(credential) != request["credential_value_hash"]:
            raise KMSRecipientV2Error("job credential value hash mismatch")
        with self._lock:
            current = self._job_requests.get(normalized_request_id)
            if current is None or current["provisioned"]:
                raise KMSRecipientV2Error("job recipient request changed")
            current["provisioned"] = True
        return {
            "job_id": str(request["job_id"]),
            "credential_slot": str(request["credential_slot"]),
            "credential_value_hash": str(request["credential_value_hash"]),
            "key_ref_hash": str(request["key_ref_hash"]),
            "credential": credential,
        }

    def _mark_provisioned(self, slot: str) -> None:
        with self._lock:
            if slot in self._provisioned:
                raise KMSRecipientV2Error("credential slot is already provisioned")
            self._provisioned.add(slot)

    def unwrap_credential(
        self,
        *,
        slot: str,
        ciphertext_for_recipient_b64: str,
    ) -> str:
        normalized_slot = str(slot or "")
        if normalized_slot in self._binary_slots:
            raise KMSRecipientV2Error("credential slot expects binary key material")
        plaintext = self._unwrap(
            slot=normalized_slot,
            ciphertext_for_recipient_b64=ciphertext_for_recipient_b64,
        )
        try:
            credential = plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise KMSRecipientV2Error("unwrapped credential is not UTF-8") from exc
        if not credential or "\x00" in credential:
            raise KMSRecipientV2Error("unwrapped credential is invalid")
        if credential_reference_hash(credential) != self._expected[normalized_slot]:
            raise KMSRecipientV2Error("unwrapped credential reference mismatch")
        self._mark_provisioned(normalized_slot)
        return credential

    def unwrap_binary_secret(
        self,
        *,
        slot: str,
        ciphertext_for_recipient_b64: str,
        hash_domain: str,
    ) -> bytes:
        normalized_slot = str(slot or "")
        if normalized_slot not in self._binary_slots:
            raise KMSRecipientV2Error("binary secret slot is not measured")
        plaintext = self._unwrap(
            slot=normalized_slot,
            ciphertext_for_recipient_b64=ciphertext_for_recipient_b64,
        )
        domain = (
            bytes(hash_domain)
            if isinstance(hash_domain, (bytes, bytearray))
            else str(hash_domain).encode("ascii")
        )
        observed = "sha256:" + hashlib.sha256(domain + plaintext).hexdigest()
        if observed != self._binary_slots[normalized_slot]:
            raise KMSRecipientV2Error("unwrapped binary secret reference mismatch")
        self._mark_provisioned(normalized_slot)
        return plaintext

    def provisioned_slots(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._provisioned))
