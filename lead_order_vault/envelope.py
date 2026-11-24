"""Seal a miner's provider key to the vault, and open it inside the vault.

ECIES on secp256k1: an ephemeral key agrees a secret with the vault's
encryption key, HKDF-SHA256 turns it into an AES-256-GCM key, and the
envelope's public header (version, provider, purpose, miner hotkey, vault
key) is the authenticated data. So an envelope opens only inside the vault it
was sealed to, and only for the provider, purpose and miner it names.

Leadpoet stores the envelope and passes it to the vault as the request's
credential. It never holds the key.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Tuple

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PREFIX = "lov1."
PROVIDERS = ("openrouter", "deepline")
# A runtime key makes calls. An OpenRouter management key is used once, at
# registration, to prove the miner owns the runtime key; it never makes calls.
PURPOSES = ("runtime", "management")
_HKDF_INFO = b"leadpoet.lead_order_vault.envelope.v1"
_FINGERPRINT_TAG = b"leadpoet.lead_order_vault.key.v1\x00"
MAX_SECRET_BYTES = 1024
MAX_ENVELOPE_CHARS = 4096
_HOTKEY_CHARS = frozenset("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")


class EnvelopeError(ValueError):
    """The envelope is malformed, sealed elsewhere, or tampered with."""


@dataclass(frozen=True)
class Header:
    provider: str
    purpose: str
    miner_hotkey: str
    vault_key: str  # hex, SEC1 compressed secp256k1


def fingerprint(provider: str, secret: str) -> str:
    """The key's public name on receipts. A miner computes it from their own key."""

    material = _FINGERPRINT_TAG + provider.encode("ascii") + b"\x00" + secret.encode("utf-8")
    return "sha256:" + hashlib.sha256(material).hexdigest()


def _check_header(header: Header) -> None:
    if header.provider not in PROVIDERS:
        raise EnvelopeError("provider_invalid")
    if header.purpose not in PURPOSES or (header.purpose == "management" and header.provider != "openrouter"):
        raise EnvelopeError("purpose_invalid")
    if not (40 <= len(header.miner_hotkey) <= 64) or not set(header.miner_hotkey) <= _HOTKEY_CHARS:
        raise EnvelopeError("miner_hotkey_invalid")
    if len(header.vault_key) != 66 or header.vault_key[:2] not in ("02", "03"):
        raise EnvelopeError("vault_key_invalid")
    try:
        bytes.fromhex(header.vault_key)
    except ValueError:
        raise EnvelopeError("vault_key_invalid") from None


def _aad(header: Header) -> bytes:
    return json.dumps(
        {"v": 1, "provider": header.provider, "purpose": header.purpose,
         "miner_hotkey": header.miner_hotkey, "vault_key": header.vault_key},
        sort_keys=True, separators=(",", ":"),
    ).encode("ascii")


def _aes_key(shared: bytes, ephemeral: bytes, vault_key: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=_HKDF_INFO + ephemeral + vault_key).derive(shared)


def _compressed(public_key: ec.EllipticCurvePublicKey) -> bytes:
    return public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.CompressedPoint)


def seal(secret: str, header: Header) -> str:
    """Encrypt ``secret`` to the vault named in ``header``."""

    _check_header(header)
    plaintext = secret.encode("utf-8")
    if not 1 <= len(plaintext) <= MAX_SECRET_BYTES or secret != secret.strip():
        raise EnvelopeError("secret_invalid")
    vault_key = bytes.fromhex(header.vault_key)
    vault = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), vault_key)
    ephemeral_private = ec.generate_private_key(ec.SECP256K1())
    ephemeral = _compressed(ephemeral_private.public_key())
    key = _aes_key(ephemeral_private.exchange(ec.ECDH(), vault), ephemeral, vault_key)
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, _aad(header))
    document = {
        "v": 1, "provider": header.provider, "purpose": header.purpose, "miner_hotkey": header.miner_hotkey,
        "vault_key": header.vault_key, "epk": ephemeral.hex(), "nonce": nonce.hex(), "ct": ciphertext.hex(),
    }
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")
    return PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def read_header(envelope: str) -> Tuple[Header, dict]:
    """Parse and check the public header without opening the envelope."""

    if not isinstance(envelope, str) or not envelope.startswith(PREFIX) or len(envelope) > MAX_ENVELOPE_CHARS:
        raise EnvelopeError("envelope_invalid")
    body = envelope[len(PREFIX):]
    try:
        document = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except (ValueError, TypeError):
        raise EnvelopeError("envelope_invalid") from None
    expected = {"v", "provider", "purpose", "miner_hotkey", "vault_key", "epk", "nonce", "ct"}
    if not isinstance(document, dict) or set(document) != expected or document["v"] != 1:
        raise EnvelopeError("envelope_invalid")
    if not all(isinstance(document[name], str) for name in expected - {"v"}):
        raise EnvelopeError("envelope_invalid")
    header = Header(document["provider"], document["purpose"], document["miner_hotkey"], document["vault_key"])
    _check_header(header)
    return header, document


def open_envelope(envelope: str, private_key: ec.EllipticCurvePrivateKey) -> Tuple[str, Header]:
    """Decrypt inside the vault. Refuses an envelope sealed to another key."""

    header, document = read_header(envelope)
    own = _compressed(private_key.public_key())
    if header.vault_key != own.hex():
        raise EnvelopeError("sealed_to_another_vault")
    try:
        ephemeral = bytes.fromhex(document["epk"])
        nonce = bytes.fromhex(document["nonce"])
        ciphertext = bytes.fromhex(document["ct"])
        peer = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), ephemeral)
    except ValueError:
        raise EnvelopeError("envelope_invalid") from None
    if len(nonce) != 12:
        raise EnvelopeError("envelope_invalid")
    key = _aes_key(private_key.exchange(ec.ECDH(), peer), ephemeral, own)
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, _aad(header))
    except InvalidTag:
        raise EnvelopeError("envelope_tampered") from None
    try:
        return plaintext.decode("utf-8"), header
    except UnicodeDecodeError:
        raise EnvelopeError("envelope_invalid") from None


__all__ = [
    "EnvelopeError", "Header", "MAX_ENVELOPE_CHARS", "PREFIX", "PROVIDERS", "PURPOSES",
    "fingerprint", "open_envelope", "read_header", "seal",
]
