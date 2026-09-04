"""Arena signing identity (labarena.md section 5.4).

One AWS KMS asymmetric sign-verify key signs every Arena authority document.
The algorithm is pinned; the public key hash is published with every round.
``LocalSigner`` exists for tests and for the public verifier's negative cases
only; production must construct ``KmsSigner``. Verification never needs AWS.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from lab_arena.contracts import (
    ArenaContractError,
    ArenaSignatureError,
    SIGNING_KEY_DOCUMENT_SCHEMA_VERSION,
    document_hash,
    require_sha256,
)

SIGNING_ALGORITHM = "ECDSA_SHA_256"
SIGNING_KEY_SPEC = "ECC_NIST_P256"
SIGNATURE_FIELD = "signature"


def public_key_hash(public_key_der: bytes) -> str:
    return "sha256:" + hashlib.sha256(bytes(public_key_der)).hexdigest()


def signing_key_document(public_key_der: bytes) -> Dict[str, Any]:
    """The public document served by ``GET /signing-key``."""

    return {
        "schema_version": SIGNING_KEY_DOCUMENT_SCHEMA_VERSION,
        "algorithm": SIGNING_ALGORITHM,
        "key_spec": SIGNING_KEY_SPEC,
        "public_key_der_b64": base64.b64encode(bytes(public_key_der)).decode("ascii"),
        "public_key_hash": public_key_hash(public_key_der),
    }


class ArenaSigner:
    """Signer interface: ``sign`` returns a DER-encoded ECDSA signature."""

    algorithm: str = SIGNING_ALGORITHM
    public_key_der: bytes = b""

    @property
    def public_key_hash(self) -> str:
        return public_key_hash(self.public_key_der)

    def sign(self, message: bytes) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class LocalSigner(ArenaSigner):
    """Test-only signer holding a P-256 private key in memory."""

    private_key: Any

    @classmethod
    def generate(cls) -> "LocalSigner":
        return cls(private_key=ec.generate_private_key(ec.SECP256R1()))

    @property
    def public_key_der(self) -> bytes:  # type: ignore[override]
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def sign(self, message: bytes) -> bytes:
        return self.private_key.sign(bytes(message), ec.ECDSA(hashes.SHA256()))


class KmsSigner(ArenaSigner):
    """Production signer: AWS KMS ``Sign`` with ``ECDSA_SHA_256`` on a P-256 key.

    ``boto3`` is imported lazily so no test or verifier path needs AWS.
    """

    def __init__(self, key_id: str, *, client: Any = None, region_name: Optional[str] = None) -> None:
        if client is None:
            import boto3  # noqa: WPS433 - lazy production import

            client = boto3.client("kms", region_name=region_name)
        self._client = client
        self.key_id = str(key_id)
        described = self._client.get_public_key(KeyId=self.key_id)
        if described.get("KeySpec") != SIGNING_KEY_SPEC:
            raise ArenaSignatureError("Arena signing key must be %s" % SIGNING_KEY_SPEC)
        usage = described.get("KeyUsage")
        if usage != "SIGN_VERIFY":
            raise ArenaSignatureError("Arena signing key must be a SIGN_VERIFY key")
        if SIGNING_ALGORITHM not in tuple(described.get("SigningAlgorithms") or ()):
            raise ArenaSignatureError("Arena signing key does not support %s" % SIGNING_ALGORITHM)
        self._public_key_der = bytes(described["PublicKey"])

    @property
    def public_key_der(self) -> bytes:  # type: ignore[override]
        return self._public_key_der

    def sign(self, message: bytes) -> bytes:
        response = self._client.sign(
            KeyId=self.key_id,
            Message=bytes(message),
            MessageType="RAW",
            SigningAlgorithm=SIGNING_ALGORITHM,
        )
        return bytes(response["Signature"])


def _signed_message(document: Mapping[str, Any], hash_field: str) -> bytes:
    digest = document.get(hash_field)
    require_sha256(digest, hash_field)
    body = {k: v for k, v in document.items() if k not in (hash_field, SIGNATURE_FIELD)}
    if document_hash(body) != digest:
        raise ArenaContractError("%s does not match document contents" % hash_field)
    # The signed bytes bind the field name so a hash of one document type can
    # never be replayed as another type's signature input.
    return ("%s:%s" % (hash_field, digest)).encode("utf-8")


def sign_document(signer: ArenaSigner, document: Mapping[str, Any], *, hash_field: str) -> Dict[str, Any]:
    """Attach an Arena signature to a hashed document."""

    message = _signed_message(document, hash_field)
    signature = signer.sign(message)
    out = dict(document)
    out[SIGNATURE_FIELD] = {
        "algorithm": signer.algorithm,
        "public_key_hash": signer.public_key_hash,
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }
    return out


def verify_document_signature(
    document: Mapping[str, Any],
    *,
    hash_field: str,
    public_key_der: bytes,
    expected_public_key_hash: str,
) -> str:
    """Verify an Arena signature against the pinned key; returns the hash.

    Raises ``ArenaSignatureError`` on any mismatch: wrong algorithm, wrong key
    hash, key bytes that do not hash to the pinned value, or a bad signature.
    """

    signature = document.get(SIGNATURE_FIELD)
    if not isinstance(signature, Mapping):
        raise ArenaSignatureError("document is unsigned")
    if set(signature) != {"algorithm", "public_key_hash", "signature_b64"}:
        raise ArenaSignatureError("signature block has unexpected fields")
    if signature.get("algorithm") != SIGNING_ALGORITHM:
        raise ArenaSignatureError("unsupported signing algorithm")
    require_sha256(expected_public_key_hash, "expected_public_key_hash")
    if public_key_hash(public_key_der) != expected_public_key_hash:
        raise ArenaSignatureError("public key does not match the pinned key hash")
    if signature.get("public_key_hash") != expected_public_key_hash:
        raise ArenaSignatureError("signature was made by a different key")
    try:
        raw_signature = base64.b64decode(str(signature.get("signature_b64") or ""), validate=True)
    except (ValueError, TypeError) as exc:
        raise ArenaSignatureError("signature is not base64") from exc
    message = _signed_message(document, hash_field)
    try:
        public_key = serialization.load_der_public_key(bytes(public_key_der))
        public_key.verify(raw_signature, message, ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise ArenaSignatureError("Arena signature invalid") from exc
    return str(document[hash_field])


def load_public_key_from_document(document: Mapping[str, Any]) -> bytes:
    if document.get("schema_version") != SIGNING_KEY_DOCUMENT_SCHEMA_VERSION:
        raise ArenaContractError("unsupported signing key document")
    if document.get("algorithm") != SIGNING_ALGORITHM or document.get("key_spec") != SIGNING_KEY_SPEC:
        raise ArenaContractError("signing key document pins a different algorithm")
    der = base64.b64decode(str(document.get("public_key_der_b64") or ""), validate=True)
    if public_key_hash(der) != document.get("public_key_hash"):
        raise ArenaContractError("signing key document hash mismatch")
    return der
