"""Bittensor signed requests (btauth/1), the scheme the Gamma standard names.

A request carries ``X-Bittensor-Version: 1``, ``X-Bittensor-Hotkey``,
``X-Bittensor-Receiver`` (this market's Gamma service hotkey),
``X-Bittensor-Nonce`` (unix nanoseconds) and ``X-Bittensor-Signature``
(``0x`` and lowercase hex). The signature is sr25519 over eight lines::

    btauth/1, sr25519, METHOD, path?query as sent, sha256hex(body), nonce, hotkey, receiver

(https://www.bittensor.com/docs/guides/signed-requests). A nonce older than
10 s or more than 2 s ahead is refused, and each (hotkey, nonce) pair is
accepted once. Only sr25519 hotkeys sign and the scheme is implied: a
request that sets ``X-Bittensor-Crypto`` is refused, never verified another way.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Callable, Dict, Mapping

from lab_arena.lead_orders.auth import AuthError, BuyerIdentity, NonceSink, SignatureVerifier
from lab_arena.lead_orders.config import is_valid_hotkey

SCHEME = "btauth/1"
MAX_AGE_NS = 10 * 10**9
MAX_SKEW_NS = 2 * 10**9
_NONCE_RE = re.compile(r"[1-9][0-9]{0,19}")
_SIGNATURE_RE = re.compile(r"0x[0-9a-f]{128}")
VERSION_HEADER = "x-bittensor-version"


def payload(method: str, target: str, body: bytes, nonce: str, hotkey: str, receiver: str) -> str:
    return "\n".join((SCHEME, "sr25519", method.upper(), target, hashlib.sha256(body).hexdigest(), nonce,
                      hotkey, receiver))


def sign(keypair: Any, *, method: str, target: str, body: bytes, receiver: str,
         now_ns: Callable[[], int] = time.time_ns) -> Dict[str, str]:
    """The headers for one request; sign again for every retry."""

    nonce = str(now_ns())
    message = payload(method, target, body, nonce, keypair.ss58_address, receiver)
    return {"X-Bittensor-Version": "1", "X-Bittensor-Hotkey": keypair.ss58_address,
            "X-Bittensor-Receiver": receiver, "X-Bittensor-Nonce": nonce,
            "X-Bittensor-Signature": "0x" + keypair.sign(message.encode("utf-8")).hex()}


def verify(headers: Mapping[str, str], body: bytes, *, method: str, target: str, receiver: str,
           verifier: SignatureVerifier, use_nonce: NonceSink,
           now_ns: Callable[[], int] = time.time_ns) -> BuyerIdentity:
    """The verified sender, or ``AuthError`` (401) with a ``btauth_*`` code."""

    get = lambda name: headers.get("x-bittensor-" + name)  # noqa: E731 - header names are case-insensitive
    if get("version") != "1":
        raise AuthError(401, "btauth_unknown_version")
    hotkey, nonce, signature, sent_to = get("hotkey"), get("nonce"), get("signature"), get("receiver")
    if not (hotkey and nonce and signature and sent_to):
        raise AuthError(401, "btauth_missing_header")
    if get("crypto") is not None:
        raise AuthError(401, "btauth_unsupported_scheme")
    if not is_valid_hotkey(hotkey) or not _NONCE_RE.fullmatch(nonce) or not _SIGNATURE_RE.fullmatch(signature):
        raise AuthError(401, "btauth_bad_header")
    if sent_to != receiver:
        raise AuthError(401, "btauth_wrong_receiver")
    age = now_ns() - int(nonce)
    body_sha256, signed_at = hashlib.sha256(body).hexdigest(), int(nonce) // 10**9
    if age > MAX_AGE_NS or age < -MAX_SKEW_NS:
        raise AuthError(401, "btauth_stale_nonce")
    try:
        valid = bool(verifier(hotkey, signature, payload(method, target, body, nonce, hotkey, receiver)))
    except Exception:
        valid = False
    if not valid:
        raise AuthError(401, "btauth_bad_signature")
    # Recorded only once the signature verifies. sr25519 signatures are
    # randomized, so the nonce, not the signature, is what is used once.
    if not use_nonce(hotkey, "sha256:" + hashlib.sha256((SCHEME + ":" + nonce).encode()).hexdigest(),
                     signed_at):
        raise AuthError(401, "btauth_replayed_nonce")
    return BuyerIdentity(hotkey=hotkey, signed_at=signed_at, host=receiver, signature=signature,
                         body_sha256=body_sha256)


__all__ = ["SCHEME", "VERSION_HEADER", "payload", "sign", "verify"]
