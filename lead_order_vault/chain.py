"""Check that a vault public key was issued by Phala's KMS to our app.

dstack's GetKey returns a key with a two-link signature chain:

* link 0: the app's root key signs ``LP("dstack-guest-v1-key-claim") ||
  LP(algorithm) || LP(domain) || LP(public_key)``.
* link 1: the KMS root key signs ``"dstack-kms-issued:" || app_id ||
  app_root_public_key``.

Both are 65-byte ``r || s || v`` secp256k1 signatures over keccak256. The KMS
root key is public on chain (``DstackKms.kmsInfo().k256Pubkey``), so anyone
can check that a vault key belongs to one app id, and that only builds the
app's on-chain owner approved can hold its private half.

The app root key is recovered from link 0 with plain curve math, then both
links are verified with ``cryptography``; recovery alone proves nothing.
"""

from __future__ import annotations

import struct
from typing import Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

# secp256k1
_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_G = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)


class ChainError(ValueError):
    """The signature chain does not tie this key to the expected app."""


def keccak256(data: bytes) -> bytes:
    from Crypto.Hash import keccak  # pycryptodome; only the verifier needs it

    return keccak.new(digest_bits=256, data=data).digest()


def _lp(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def claim_message(algorithm: str, domain: str, public_key: bytes) -> bytes:
    return (_lp(b"dstack-guest-v1-key-claim") + _lp(algorithm.encode("ascii"))
            + _lp(domain.encode("utf-8")) + _lp(public_key))


def issued_message(app_id: bytes, app_root_public_key: bytes) -> bytes:
    return b"dstack-kms-issued:" + app_id + app_root_public_key


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a[0] == b[0] and (a[1] + b[1]) % _P == 0:
        return None
    if a == b:
        slope = 3 * a[0] * a[0] * pow(2 * a[1], -1, _P)
    else:
        slope = (b[1] - a[1]) * pow(b[0] - a[0], -1, _P)
    x = (slope * slope - a[0] - b[0]) % _P
    return x, (slope * (a[0] - x) - a[1]) % _P


def _mul(k: int, point):
    result = None
    while k:
        if k & 1:
            result = _add(result, point)
        point = _add(point, point)
        k >>= 1
    return result


def _split(signature: bytes):
    if len(signature) != 65:
        raise ChainError("signature_length")
    r = int.from_bytes(signature[:32], "big")
    s = int.from_bytes(signature[32:64], "big")
    v = signature[64]
    if v >= 27:
        v -= 27
    if not (0 < r < _N and 0 < s < _N) or v > 3:
        raise ChainError("signature_invalid")
    return r, s, v


def recover(digest: bytes, signature: bytes) -> bytes:
    """Return the SEC1-compressed public key that could have made ``signature``."""

    r, s, v = _split(signature)
    x = r + (v >> 1) * _N
    if x >= _P:
        raise ChainError("signature_invalid")
    y = pow((pow(x, 3, _P) + 7) % _P, (_P + 1) // 4, _P)
    if (y * y - x * x * x - 7) % _P:
        raise ChainError("signature_invalid")
    if y & 1 != v & 1:
        y = _P - y
    z = int.from_bytes(digest, "big") % _N
    r_inv = pow(r, -1, _N)
    point = _add(_mul(s * r_inv % _N, (x, y)), _mul((-z * r_inv) % _N, _G))
    if point is None:
        raise ChainError("signature_invalid")
    return bytes([2 + (point[1] & 1)]) + point[0].to_bytes(32, "big")


def verify(public_key: bytes, digest: bytes, signature: bytes) -> None:
    r, s, _ = _split(signature)
    try:
        key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), public_key)
        key.verify(utils.encode_dss_signature(r, s), digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
    except (InvalidSignature, ValueError):
        raise ChainError("signature_invalid") from None


def _compressed(public_key: bytes) -> bytes:
    try:
        key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), public_key)
    except ValueError:
        raise ChainError("public_key_invalid") from None
    return key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.CompressedPoint)


def app_root_key(*, algorithm: str, domain: str, public_key: bytes, link0: bytes) -> bytes:
    """Recover and verify the app root key that signed the claim (link 0)."""

    digest = keccak256(claim_message(algorithm, domain, public_key))
    root = recover(digest, link0)
    verify(root, digest, link0)
    return root


def verify_key(
    *, algorithm: str, domain: str, public_key_hex: str, signature_chain: Sequence[str],
    app_id_hex: str, kms_root_public_key_hex: str,
) -> None:
    """Raise ``ChainError`` unless the KMS issued this key to ``app_id``."""

    if len(signature_chain) != 2:
        raise ChainError("chain_length")
    try:
        public_key = bytes.fromhex(public_key_hex)
        link0, link1 = (bytes.fromhex(link) for link in signature_chain)
        app_id = bytes.fromhex(app_id_hex.removeprefix("0x"))
        kms_root = _compressed(bytes.fromhex(kms_root_public_key_hex.removeprefix("0x")))
    except ValueError:
        raise ChainError("chain_encoding") from None
    root = app_root_key(algorithm=algorithm, domain=domain, public_key=public_key, link0=link0)
    verify(kms_root, keccak256(issued_message(app_id, root)), link1)


__all__ = ["ChainError", "app_root_key", "claim_message", "issued_message", "keccak256", "recover", "verify_key"]
