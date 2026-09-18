"""Pure Bittensor signature verification used by the Arena."""

import logging


logger = logging.getLogger(__name__)


def verify_sr25519_signature(hotkey: str, signature: str, message: str) -> bool:
    """Verify an SR25519 signature from a Bittensor hotkey."""

    try:
        from bittensor import Keypair

        keypair = Keypair(ss58_address=hotkey)
        encoded = signature[2:] if signature.startswith("0x") else signature
        return bool(keypair.verify(message, bytes.fromhex(encoded)))
    except ValueError as exc:
        logger.error("Signature format error: %s", exc)
        return False
    except Exception as exc:
        logger.error("Signature verification error: %s", exc)
        return False
