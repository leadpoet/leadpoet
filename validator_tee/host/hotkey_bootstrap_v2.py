"""Provision the production validator hotkey directly into Nitro with KMS.

The parent receives the existing KMS ciphertext and AWS KMS
``CiphertextForRecipient`` only. A KMS response containing ``Plaintext`` is a
hard failure and is never forwarded or logged.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json
from validator_tee.enclave.hotkey_authority_v2 import (
    HOTKEY_ENCRYPTION_ALGORITHM,
    HOTKEY_RECIPIENT_SCHEMA_VERSION,
    hotkey_authority_configuration_hash,
    validate_hotkey_authority_configuration,
)
from validator_tee.host.vsock_client import ValidatorEnclaveClient


HOTKEY_ENVELOPE_SCHEMA_VERSION = "leadpoet.validator_hotkey_envelope.v2"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ValidatorHotkeyBootstrapV2Error(RuntimeError):
    """Hotkey configuration or recipient provisioning is unsafe."""


def kms_key_reference_hash(key_id: str) -> str:
    value = str(key_id or "").strip()
    if not value:
        raise ValidatorHotkeyBootstrapV2Error("validator KMS key id is empty")
    return sha256_bytes(("leadpoet-validator-hotkey-kms-v2:" + value).encode())


def validate_hotkey_envelope(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "validator_hotkey",
        "hotkey_public_key",
        "ciphertext_blob_b64",
        "ciphertext_blob_hash",
        "kms_key_id_hash",
        "encryption_context",
        "encryption_context_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey envelope fields are invalid"
        )
    if value.get("schema_version") != HOTKEY_ENVELOPE_SCHEMA_VERSION:
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey envelope schema is invalid"
        )
    try:
        ciphertext = base64.b64decode(
            str(value.get("ciphertext_blob_b64") or ""), validate=True
        )
    except Exception as exc:
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey KMS ciphertext is invalid"
        ) from exc
    if not ciphertext or len(ciphertext) > 64 * 1024:
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey KMS ciphertext is outside limit"
        )
    if sha256_bytes(ciphertext) != value.get("ciphertext_blob_hash"):
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey KMS ciphertext hash mismatch"
        )
    context = value.get("encryption_context")
    if not isinstance(context, Mapping) or not context:
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey encryption context is missing"
        )
    normalized_context = {
        str(name): str(item) for name, item in sorted(context.items())
    }
    if any(
        not name or not item or "\x00" in name + item
        for name, item in normalized_context.items()
    ):
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey encryption context is invalid"
        )
    if sha256_json(normalized_context) != value.get("encryption_context_hash"):
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey encryption context hash mismatch"
        )
    key_hash = str(value.get("kms_key_id_hash") or "").lower()
    public_key = str(value.get("hotkey_public_key") or "").lower()
    hotkey = str(value.get("validator_hotkey") or "")
    if (
        not _HASH_RE.fullmatch(key_hash)
        or not re.fullmatch(r"[0-9a-f]{64}", public_key)
        or not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{40,64}", hotkey)
    ):
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey envelope identity is invalid"
        )
    return {
        **dict(value),
        "validator_hotkey": hotkey,
        "hotkey_public_key": public_key,
        "kms_key_id_hash": key_hash,
        "encryption_context": normalized_context,
        "ciphertext_blob": ciphertext,
    }


def provision_validator_hotkey_v2(
    *,
    hotkey_configuration: Mapping[str, Any],
    hotkey_envelope: Mapping[str, Any],
    client: Optional[ValidatorEnclaveClient] = None,
    kms_client: Any = None,
) -> Dict[str, Any]:
    configuration = validate_hotkey_authority_configuration(
        hotkey_configuration
    )
    envelope = validate_hotkey_envelope(hotkey_envelope)
    if (
        envelope["validator_hotkey"] != configuration["validator_hotkey"]
        or envelope["hotkey_public_key"] != configuration["hotkey_public_key"]
    ):
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey envelope differs from measured configuration"
        )
    enclave_client = client or ValidatorEnclaveClient()
    expected_config_hash = hotkey_authority_configuration_hash(configuration)
    configured = enclave_client.configure_hotkey_authority_v2(
        configuration,
        expected_config_hash,
    )
    if (
        configured.get("validator_hotkey") != envelope["validator_hotkey"]
        or configured.get("hotkey_public_key") != envelope["hotkey_public_key"]
        or configured.get("provisioned") is not False
    ):
        raise ValidatorHotkeyBootstrapV2Error(
            "enclave hotkey authority configured an unexpected identity"
        )
    recipient = enclave_client.get_hotkey_recipient_v2()
    if (
        recipient.get("schema_version") != HOTKEY_RECIPIENT_SCHEMA_VERSION
        or recipient.get("purpose") != "leadpoet.validator_hotkey_unseal.v2"
        or recipient.get("validator_hotkey") != envelope["validator_hotkey"]
        or recipient.get("key_encryption_algorithm")
        != HOTKEY_ENCRYPTION_ALGORITHM
    ):
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey recipient request is invalid"
        )
    try:
        attestation_document = base64.b64decode(
            str(recipient["attestation_document_b64"]), validate=True
        )
    except Exception as exc:
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey recipient attestation is invalid"
        ) from exc
    if kms_client is None:
        import boto3

        kms_client = boto3.client("kms")
    response = kms_client.decrypt(
        CiphertextBlob=envelope["ciphertext_blob"],
        EncryptionContext=envelope["encryption_context"],
        Recipient={
            "KeyEncryptionAlgorithm": HOTKEY_ENCRYPTION_ALGORITHM,
            "AttestationDocument": attestation_document,
        },
    )
    if "Plaintext" in response:
        raise ValidatorHotkeyBootstrapV2Error(
            "KMS returned validator hotkey plaintext to the parent"
        )
    if kms_key_reference_hash(str(response.get("KeyId") or "")) != envelope[
        "kms_key_id_hash"
    ]:
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey KMS response key differs from envelope"
        )
    ciphertext_for_recipient = response.get("CiphertextForRecipient")
    if not isinstance(ciphertext_for_recipient, (bytes, bytearray)) or not (
        ciphertext_for_recipient
    ):
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey recipient ciphertext is missing"
        )
    state = enclave_client.provision_hotkey_v2(
        base64.b64encode(bytes(ciphertext_for_recipient)).decode("ascii")
    )
    if (
        state.get("provisioned") is not True
        or state.get("validator_hotkey") != envelope["validator_hotkey"]
        or state.get("hotkey_public_key") != envelope["hotkey_public_key"]
    ):
        raise ValidatorHotkeyBootstrapV2Error(
            "validator enclave rejected the provisioned hotkey"
        )
    if enclave_client.get_hotkey_state_v2() != state:
        raise ValidatorHotkeyBootstrapV2Error(
            "validator hotkey state readback mismatch"
        )
    return {
        "validator_hotkey": state["validator_hotkey"],
        "hotkey_public_key": state["hotkey_public_key"],
        "provisioned": True,
        "hotkey_configuration_hash": expected_config_hash,
    }


def _load(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatorHotkeyBootstrapV2Error(
            "validator V2 input is unavailable: %s" % path
        ) from exc
    if not isinstance(value, dict):
        raise ValidatorHotkeyBootstrapV2Error(
            "validator V2 input is not an object: %s" % path
        )
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotkey-config", type=Path, required=True)
    parser.add_argument("--hotkey-envelope", type=Path, required=True)
    args = parser.parse_args(argv)
    result = provision_validator_hotkey_v2(
        hotkey_configuration=_load(args.hotkey_config),
        hotkey_envelope=_load(args.hotkey_envelope),
    )
    print("validator_hotkey=%s" % result["validator_hotkey"])
    print("validator_hotkey_provisioned=true")
    print("validator_hotkey_config_hash=%s" % result["hotkey_configuration_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
