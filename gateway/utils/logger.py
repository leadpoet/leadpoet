"""TEE event logging with direct Arweave checkpoint delivery.

The coordinator enclave owns signing and buffering. The hourly batch task
uploads confirmed buffer prefixes to Arweave. Relational transparency-log
persistence was part of the retired lead-validation service and is not in the
current Arena path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

FALLBACK_LOG_DIR = Path(
    os.getenv("GATEWAY_TEE_FALLBACK_LOG_DIR", "gateway/logs/tee_fallback")
).expanduser()
FALLBACK_LOG_DIR.mkdir(parents=True, exist_ok=True)


def compute_payload_hash(payload: dict) -> str:
    """Return the canonical SHA-256 digest for an event payload."""
    payload_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


async def initialize_enclave_event_signing() -> Dict[str, Any]:
    """Initialize and attest the coordinator event signer.

    A new enclave boot starts a new buffer chain. Confirmed buffer prefixes
    are durable in Arweave; startup no longer depends on a mutable relational
    log tip.
    """
    try:
        from gateway.utils.tee_client import tee_client
    except ImportError:
        from utils.tee_client import tee_client

    initialized = await tee_client.initialize_event_signer(None)
    identity = initialized.get("identity")
    restart_log_entry = initialized.get("restart_log_entry")
    tee_buffer = initialized.get("buffer")
    if not isinstance(identity, dict) or not isinstance(restart_log_entry, dict):
        raise RuntimeError("coordinator enclave returned invalid signer initialization")
    if not isinstance(tee_buffer, dict):
        raise RuntimeError("coordinator enclave did not buffer its restart event")
    if identity.get("purpose") != "gateway_event_signing":
        raise RuntimeError("coordinator enclave returned the wrong signing purpose")
    if not identity.get("attestation_document_b64"):
        raise RuntimeError("coordinator event signer has no Nitro attestation")
    return identity


async def log_event(
    event_or_type: str | Dict[str, Any],
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sign and buffer an event in the coordinator enclave.

    The dictionary form remains accepted for the retained ICP and Arweave
    tasks. It is normalized into the signed payload before it reaches the
    enclave. Both forms use the same signed, buffered path.
    """
    if isinstance(event_or_type, dict):
        event = dict(event_or_type)
        event_type = str(event.pop("event_type", "UNKNOWN"))
        event_payload = event.pop("payload", None)
        if not isinstance(event_payload, dict):
            raise ValueError("dictionary events require an object payload")
        normalized_payload = dict(event_payload)
        if event.get("actor_hotkey") is not None:
            normalized_payload.setdefault("actor_hotkey", event["actor_hotkey"])
        return await _log_event_signed_format(event_type, normalized_payload)
    if isinstance(event_or_type, str) and payload is not None:
        return await _log_event_signed_format(event_or_type, payload)
    raise ValueError(
        "Invalid log_event arguments. Use a full event dictionary or "
        "log_event(event_type, payload)."
    )


async def _log_event_signed_format(
    event_type: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    payload_hash = compute_payload_hash(payload)
    try:
        try:
            from gateway.utils.tee_client import tee_client
        except ImportError:
            from utils.tee_client import tee_client
        enclave_result = await tee_client.sign_transparency_event(
            event_type=event_type,
            payload=payload,
            payload_hash=payload_hash,
        )
        if not isinstance(enclave_result, dict):
            raise RuntimeError("coordinator enclave returned an invalid event result")
        log_entry = enclave_result.get("log_entry")
        tee_result = enclave_result.get("buffer")
        if not isinstance(log_entry, dict) or not isinstance(tee_result, dict):
            raise RuntimeError("coordinator enclave returned an invalid event result")
        try:
            signed_event = log_entry["signed_event"]
            event_hash = log_entry["event_hash"]
            sequence = tee_result["sequence"]
            buffer_size = tee_result["buffer_size"]
        except KeyError as exc:
            raise RuntimeError(
                "coordinator enclave omitted a required signed-event field"
            ) from exc
        if not isinstance(signed_event, dict):
            raise RuntimeError("coordinator enclave returned an invalid signed event")
        try:
            monotonic_seq = signed_event["monotonic_seq"]
        except KeyError as exc:
            raise RuntimeError(
                "coordinator enclave omitted a required signed-event field"
            ) from exc
        event_hash = str(event_hash).strip().lower()
        if len(event_hash) != 64 or any(
            char not in "0123456789abcdef" for char in event_hash
        ):
            raise RuntimeError("coordinator enclave returned an invalid event hash")
        if (
            not isinstance(monotonic_seq, int)
            or isinstance(monotonic_seq, bool)
            or monotonic_seq < 0
        ):
            raise RuntimeError("coordinator enclave returned an invalid monotonic sequence")
        for field_name, field_value in (
            ("sequence", sequence),
            ("buffer_size", buffer_size),
        ):
            if (
                not isinstance(field_value, int)
                or isinstance(field_value, bool)
                or field_value < 0
            ):
                raise RuntimeError(
                    f"coordinator enclave returned an invalid {field_name}"
                )
        log_entry["sequence"] = sequence
        log_entry["buffer_size"] = buffer_size
        log_entry["tee_sequence"] = sequence
        log_entry["tee_buffer_size"] = buffer_size
        log_entry["tee_buffered"] = True
        logger.info(
            "Event signed and buffered in coordinator enclave: %s "
            "(seq=%s, hash=%s)",
            event_type,
            monotonic_seq,
            event_hash[:16],
        )
        return log_entry
    except Exception as exc:
        logger.error("Event signing failed: %s - %s", event_type, exc)
        await _fallback_log_to_file(
            {"event_type": event_type, "payload": payload},
            error=str(exc),
        )
        raise RuntimeError(
            f"Failed to sign and buffer event: {event_type}."
        ) from exc


async def _fallback_log_to_file(event: dict, error: str = "") -> None:
    """Keep local evidence when the coordinator enclave is unavailable."""
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        event_type = event.get("event_type", "UNKNOWN")
        filepath = FALLBACK_LOG_DIR / f"{timestamp}_{event_type}_TEE_FAILURE.json"
        fallback_data = {
            "event": event,
            "error": error,
            "failed_at": datetime.utcnow().isoformat(),
            "reason": "TEE buffer write failed",
        }
        with open(filepath, "w") as file_handle:
            json.dump(fallback_data, file_handle, indent=2, sort_keys=True)
        logger.critical(
            "TEE buffer write failed for %s; fallback log: %s",
            event_type,
            filepath,
        )
    except Exception as exc:
        logger.critical("Fallback logging also failed: %s", exc)


def get_signer_info() -> Dict[str, Any]:
    """Return public information about the enclave signer boundary."""
    return {
        "authority": "gateway_coordinator_enclave",
        "host_signer_present": False,
        "identity_endpoint": "/attestation/health",
    }
