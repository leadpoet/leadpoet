"""Gateway coordinator identity initialization."""

from __future__ import annotations

from typing import Any, Dict


async def initialize_enclave_identity() -> Dict[str, Any]:
    """Read and validate the measured coordinator runtime identity."""
    from gateway.utils.tee_client import tee_client

    identity = await tee_client.get_event_signing_identity()
    if not isinstance(identity, dict):
        raise RuntimeError("coordinator enclave returned an invalid runtime identity")
    if identity.get("purpose") != "gateway_event_signing":
        raise RuntimeError("coordinator enclave returned the wrong signing purpose")
    if not identity.get("attestation_document_b64"):
        raise RuntimeError("coordinator runtime identity has no Nitro attestation")
    return identity
