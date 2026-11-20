"""Request authentication for the lead-order routes.

Buyers sign each request with their hotkey (sr25519). The signed message is

    METHOD "\\n" HOST "\\n" PATH "\\n" TIMESTAMP "\\n" sha256hex(BODY)

where HOST is the gateway's public host (from its configured public base
URL, lowercase, with the port if the URL names one). The query string is not
signed; no signed route takes a query that changes what is written. It is
sent as ``X-Hotkey``, ``X-Timestamp`` (unix seconds) and ``X-Signature``
(hex). A timestamp more than ``TIMESTAMP_WINDOW_SECONDS`` from server time is
refused, and every signature is accepted once (the nonce table is keyed by
hotkey and signature hash), so a captured request cannot be replayed.

The private runner authenticates with a bearer secret compared in constant
time (``LeadOrdersConfig.runner_secret_matches``).
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

from lab_arena.lead_orders.config import LeadOrdersConfig

TIMESTAMP_WINDOW_SECONDS = 60
_HOTKEY_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,48}$")
_SIG_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]{128}$")

SignatureVerifier = Callable[[str, str, str], bool]
NonceSink = Callable[[str, str, int], bool]


class AuthError(Exception):
    def __init__(self, status: int, code: str) -> None:
        self.status = status
        self.code = code
        super().__init__(code)


def signed_message(method: str, host: str, path: str, timestamp: str, body: bytes) -> str:
    """METHOD, host, path, timestamp and body sha256, one per line.

    The host is this gateway's public host (``LEADPOET_LEAD_ORDERS_PUBLIC_BASE_URL``),
    never the request's Host header, so a signature made for one deployment
    cannot be replayed against another. Arena signing does not use this.
    """

    return "%s\n%s\n%s\n%s\n%s" % (method.upper(), host.lower(), path, timestamp, hashlib.sha256(body).hexdigest())


@dataclass(frozen=True)
class BuyerIdentity:
    hotkey: str
    signed_at: int
    host: str = ""
    # Kept so a signed acceptance (miner terms) can be stored and re-verified.
    signature: str = ""
    body_sha256: str = ""


def verify_buyer_request(
    *,
    method: str,
    host: str,
    path: str,
    body: bytes,
    hotkey: Optional[str],
    timestamp: Optional[str],
    signature: Optional[str],
    verifier: SignatureVerifier,
    use_nonce: NonceSink,
    now: Callable[[], float] = time.time,
) -> BuyerIdentity:
    """Return the verified hotkey or raise ``AuthError`` (401)."""

    if not host:
        raise AuthError(401, "host_unconfigured")
    if not hotkey or not _HOTKEY_RE.fullmatch(hotkey):
        raise AuthError(401, "hotkey_invalid")
    if not timestamp or not timestamp.isdigit() or len(timestamp) > 12:
        raise AuthError(401, "timestamp_invalid")
    signed_at = int(timestamp)
    if abs(now() - signed_at) > TIMESTAMP_WINDOW_SECONDS:
        raise AuthError(401, "timestamp_outside_window")
    if not signature or not _SIG_RE.fullmatch(signature):
        raise AuthError(401, "signature_invalid")
    message = signed_message(method, host, path, timestamp, body)
    try:
        valid = bool(verifier(hotkey, signature, message))
    except Exception:
        valid = False
    if not valid:
        raise AuthError(401, "signature_invalid")
    signature_hash = "sha256:" + hashlib.sha256(signature.lower().removeprefix("0x").encode("ascii")).hexdigest()
    # Record only after the signature verifies, so junk cannot fill the table.
    if not use_nonce(hotkey, signature_hash, signed_at):
        raise AuthError(401, "signature_replayed")
    return BuyerIdentity(hotkey=hotkey, signed_at=signed_at, host=host.lower(), signature=signature,
                         body_sha256=hashlib.sha256(body).hexdigest())


def verify_runner(config: LeadOrdersConfig, authorization: Optional[str]) -> None:
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise AuthError(401, "runner_unauthorized")
    if not config.runner_secret_matches(authorization[len(prefix):]):
        raise AuthError(401, "runner_unauthorized")


def order_token_hash(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


__all__ = [
    "AuthError",
    "BuyerIdentity",
    "TIMESTAMP_WINDOW_SECONDS",
    "order_token_hash",
    "signed_message",
    "verify_buyer_request",
    "verify_runner",
]
