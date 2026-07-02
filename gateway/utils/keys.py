"""Gateway Ed25519 key loader used by startup receipt-signing checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

_PRIVATE_KEY: Optional[Ed25519PrivateKey] = None
_PUBLIC_KEY_HEX: Optional[str] = None


def _candidate_key_paths(configured_path: str) -> list[Path]:
    raw = Path(configured_path).expanduser()
    gateway_root = Path(__file__).resolve().parents[1]
    repo_root = gateway_root.parent
    paths = [raw]
    if not raw.is_absolute():
        paths.extend(
            [
                Path.cwd() / raw,
                repo_root / raw,
                gateway_root / raw,
            ]
        )
        if raw.parts and raw.parts[0] == "gateway":
            paths.append(gateway_root.joinpath(*raw.parts[1:]))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _resolve_key_path(configured_path: str) -> Path:
    for path in _candidate_key_paths(configured_path):
        if path.is_file():
            return path
    return _candidate_key_paths(configured_path)[0]


def load_gateway_keypair() -> bool:
    """Load and validate the gateway signing keypair.

    The private key may be encrypted or unencrypted PEM. The configured public
    key must match the private key when ``GATEWAY_PUBLIC_KEY`` is present.
    """

    global _PRIVATE_KEY, _PUBLIC_KEY_HEX

    from gateway.config import (
        GATEWAY_PRIVATE_KEY_PASSWORD,
        GATEWAY_PRIVATE_KEY_PATH,
        GATEWAY_PUBLIC_KEY,
    )

    path = _resolve_key_path(GATEWAY_PRIVATE_KEY_PATH)
    if not path.is_file():
        print(f"Gateway private key file not found: {path}")
        return False

    password = GATEWAY_PRIVATE_KEY_PASSWORD.encode("utf-8") if GATEWAY_PRIVATE_KEY_PASSWORD else None
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=password)
    except TypeError as exc:
        print(f"Gateway private key password/config error: {exc}")
        return False
    except Exception as exc:
        print(f"Gateway private key load failed: {exc}")
        return False

    if not isinstance(key, Ed25519PrivateKey):
        print(f"Gateway private key must be Ed25519, got {type(key).__name__}")
        return False

    public_hex = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    expected_public_hex = (GATEWAY_PUBLIC_KEY or "").strip().lower()
    if expected_public_hex and expected_public_hex != public_hex:
        print("Gateway public key does not match configured private key")
        return False

    _PRIVATE_KEY = key
    _PUBLIC_KEY_HEX = public_hex
    return True


def gateway_keypair_loaded() -> bool:
    return _PRIVATE_KEY is not None


def get_gateway_public_key_hex() -> Optional[str]:
    return _PUBLIC_KEY_HEX


def sign_gateway_message(message: Union[bytes, str]) -> str:
    if _PRIVATE_KEY is None:
        raise RuntimeError("gateway keypair is not loaded")
    if isinstance(message, str):
        payload = message.encode("utf-8")
    elif isinstance(message, bytes):
        payload = message
    else:
        raise TypeError("message must be bytes or str")
    return _PRIVATE_KEY.sign(payload).hex()
