"""Seal an Arena seed/policy once, or provision ciphertext into Nitro."""

from __future__ import annotations

import argparse
import base64
import json
import hashlib
import os
import re
import stat
from pathlib import Path

from leadpoet_canonical.lab_arena_rewards import sha256_json
from validator_tee.enclave.arena_hotkey import (
    ENCRYPTION_ALGORITHM, PAYLOAD_SCHEMA, RECIPIENT_SCHEMA,
    ArenaHotkeyError, sealed_payload, validate_policy,
)

ENVELOPE_SCHEMA = "leadpoet.arena.hotkey_envelope.v1"
LEGACY_ENVELOPE_SCHEMA = "leadpoet.validator_hotkey_envelope.v2"


def kms_region(key_id):
    """Use a key ARN's region even in a clean service environment."""
    if not isinstance(key_id, str) or not key_id:
        raise ArenaHotkeyError("KMS key identity is invalid")
    if not key_id.startswith("arn:"):
        return None
    match = re.fullmatch(r"arn:aws(?:-us-gov|-cn)?:kms:([a-z0-9-]+):[0-9]{12}:(?:key|alias)/[A-Za-z0-9/_-]+", key_id)
    if match is None:
        raise ArenaHotkeyError("KMS key ARN is invalid")
    return match.group(1)


def create_kms_client(key_id):
    import boto3
    return boto3.client("kms", region_name=kms_region(key_id))


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _kms_key_reference_hash(key_id):
    if not isinstance(key_id, str) or not key_id:
        raise ArenaHotkeyError("KMS key identity is invalid")
    return "sha256:" + _sha256_bytes(("leadpoet-validator-hotkey-kms-v2:" + key_id).encode())


def _validate_recipient(recipient, policy):
    if (recipient.get("schema_version") != RECIPIENT_SCHEMA
            or recipient.get("purpose") != PAYLOAD_SCHEMA
            or recipient.get("key_encryption_algorithm") != ENCRYPTION_ALGORITHM
            or recipient.get("policy_hash") != sha256_json(policy)
            or recipient.get("policy") != policy):
        raise ArenaHotkeyError("Arena recipient identity or measured policy is invalid")


def seal(seed, policy, *, kms_key_id, kms_client):
    normalized = validate_policy(policy)
    policy_hash = sha256_json(normalized)
    context = {"purpose": PAYLOAD_SCHEMA, "validator_hotkey": normalized["validator_hotkey"], "policy_hash": policy_hash}
    response = kms_client.encrypt(KeyId=kms_key_id, Plaintext=sealed_payload(seed, normalized), EncryptionContext=context)
    if not response.get("KeyId") or not isinstance(response.get("CiphertextBlob"), bytes):
        raise ArenaHotkeyError("KMS did not return an Arena envelope")
    return {
        "schema_version": ENVELOPE_SCHEMA, "kms_key_id": response["KeyId"],
        "policy": normalized, "policy_hash": policy_hash, "encryption_context": context,
        "ciphertext_blob_b64": base64.b64encode(response["CiphertextBlob"]).decode(),
    }


def provision(envelope, *, client, kms_client):
    fields = {"schema_version", "kms_key_id", "policy", "policy_hash", "encryption_context", "ciphertext_blob_b64"}
    if not isinstance(envelope, dict) or set(envelope) != fields or envelope["schema_version"] != ENVELOPE_SCHEMA:
        raise ArenaHotkeyError("Arena envelope fields are invalid")
    policy = validate_policy(envelope["policy"])
    expected_hash = sha256_json(policy)
    context = {"purpose": PAYLOAD_SCHEMA, "validator_hotkey": policy["validator_hotkey"], "policy_hash": expected_hash}
    if envelope["policy_hash"] != expected_hash or envelope["encryption_context"] != context:
        raise ArenaHotkeyError("Arena envelope policy differs")
    current = client.get_arena_hotkey_state_v1()
    if current.get("provisioned") is True:
        if current.get("policy_hash") != expected_hash or current.get("policy") != policy:
            raise ArenaHotkeyError("Provisioned Arena policy differs")
        return current
    recipient = client.get_arena_hotkey_recipient_v1()
    _validate_recipient(recipient, policy)
    try:
        ciphertext = base64.b64decode(envelope["ciphertext_blob_b64"], validate=True)
        attestation = base64.b64decode(recipient["attestation_document_b64"], validate=True)
        if not ciphertext or not attestation:
            raise ValueError("empty ciphertext or attestation")
    except Exception:
        raise ArenaHotkeyError("Arena envelope or attestation encoding is invalid") from None
    response = kms_client.decrypt(
        KeyId=envelope["kms_key_id"], CiphertextBlob=ciphertext, EncryptionContext=context,
        Recipient={"KeyEncryptionAlgorithm": ENCRYPTION_ALGORITHM, "AttestationDocument": attestation},
    )
    # AWS documents a null/empty Plaintext field for recipient responses.
    if response.get("Plaintext") not in (None, b"", ""):
        raise ArenaHotkeyError("KMS returned plaintext to the parent")
    if response.get("KeyId") != envelope["kms_key_id"]:
        raise ArenaHotkeyError("KMS returned a different key identity")
    recipient_ciphertext = response.get("CiphertextForRecipient")
    if not isinstance(recipient_ciphertext, bytes) or not recipient_ciphertext:
        raise ArenaHotkeyError("KMS recipient ciphertext is missing")
    result = client.provision_arena_hotkey_v1(base64.b64encode(recipient_ciphertext).decode())
    if (result.get("provisioned") is not True or result.get("policy_hash") != expected_hash
            or result.get("policy") != policy or client.get_arena_hotkey_state_v1() != result):
        raise ArenaHotkeyError("Arena provisioned policy readback differs")
    return result


def provision_legacy(envelope, *, expected_policy, kms_key_id, client, kms_client):
    """Move a v2 raw-seed KMS envelope into the measured Arena authority."""

    policy = validate_policy(expected_policy)
    fields = {
        "schema_version", "ciphertext_blob_b64", "ciphertext_blob_hash",
        "encryption_context", "encryption_context_hash", "hotkey_public_key",
        "kms_key_id_hash", "validator_hotkey",
    }
    if not isinstance(envelope, dict) or set(envelope) != fields or envelope.get("schema_version") != LEGACY_ENVELOPE_SCHEMA:
        raise ArenaHotkeyError("Legacy hotkey envelope fields are invalid")
    if (envelope["validator_hotkey"] != policy["validator_hotkey"]
            or envelope["hotkey_public_key"] != policy["hotkey_public_key"]):
        raise ArenaHotkeyError("Legacy hotkey identity differs from measured policy")
    if not isinstance(envelope["encryption_context"], dict):
        raise ArenaHotkeyError("Legacy encryption context is invalid")
    try:
        ciphertext = base64.b64decode(envelope["ciphertext_blob_b64"], validate=True)
    except Exception:
        raise ArenaHotkeyError("Legacy ciphertext encoding is invalid") from None
    if (not ciphertext or envelope["ciphertext_blob_hash"] != "sha256:" + _sha256_bytes(ciphertext)
            or envelope["encryption_context_hash"] != sha256_json(envelope["encryption_context"])
            or envelope["kms_key_id_hash"] != _kms_key_reference_hash(kms_key_id)):
        raise ArenaHotkeyError("Legacy envelope integrity differs")
    current = client.get_arena_hotkey_state_v1()
    expected_hash = sha256_json(policy)
    if current.get("provisioned") is True:
        if current.get("policy_hash") != expected_hash or current.get("policy") != policy:
            raise ArenaHotkeyError("Provisioned Arena policy differs")
        return current
    recipient = client.get_arena_hotkey_recipient_v1()
    _validate_recipient(recipient, policy)
    try:
        attestation = base64.b64decode(recipient["attestation_document_b64"], validate=True)
        if not attestation:
            raise ValueError("empty attestation")
    except Exception:
        raise ArenaHotkeyError("Arena attestation encoding is invalid") from None
    response = kms_client.decrypt(
        KeyId=kms_key_id, CiphertextBlob=ciphertext,
        EncryptionContext=envelope["encryption_context"],
        Recipient={"KeyEncryptionAlgorithm": ENCRYPTION_ALGORITHM, "AttestationDocument": attestation},
    )
    if response.get("Plaintext") not in (None, b"", ""):
        raise ArenaHotkeyError("KMS returned plaintext to the parent")
    returned_key_id = response.get("KeyId")
    if not isinstance(returned_key_id, str) or _kms_key_reference_hash(returned_key_id) != envelope["kms_key_id_hash"]:
        raise ArenaHotkeyError("KMS returned a different key identity")
    recipient_ciphertext = response.get("CiphertextForRecipient")
    if not isinstance(recipient_ciphertext, bytes) or not recipient_ciphertext:
        raise ArenaHotkeyError("KMS recipient ciphertext is missing")
    result = client.provision_arena_legacy_hotkey_v1(base64.b64encode(recipient_ciphertext).decode())
    if (result.get("provisioned") is not True or result.get("policy_hash") != expected_hash
            or result.get("policy") != policy or client.get_arena_hotkey_state_v1() != result):
        raise ArenaHotkeyError("Arena provisioned policy readback differs")
    return result


def _private_read(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077
                or metadata.st_uid not in (0, os.geteuid())):
            raise ArenaHotkeyError("Arena bootstrap input must be a private owned file")
        raw = source.read(65537)
        if len(raw) > 65536:
            raise ArenaHotkeyError("Arena bootstrap input is too large")
        return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("seal", help="owner workstation only; seed input is never a command argument")
    prepare.add_argument("--seed-file", type=Path, required=True, help="private file with the 32 raw seed bytes")
    prepare.add_argument("--policy", type=Path, required=True)
    prepare.add_argument("--kms-key-id", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    unseal = commands.add_parser("provision", help="validator host; accepts encrypted envelope only")
    unseal.add_argument("--envelope", type=Path, required=True)
    migrate = commands.add_parser("migrate-legacy", help="validator host; migrate the existing KMS raw-seed envelope")
    migrate.add_argument("--legacy-envelope", type=Path, required=True)
    migrate.add_argument("--policy", type=Path, required=True)
    migrate.add_argument("--kms-key-id", required=True)
    args = parser.parse_args(argv)
    envelope = json.loads(_private_read(args.envelope)) if args.command == "provision" else None
    kms = create_kms_client(envelope["kms_key_id"] if envelope is not None else args.kms_key_id)
    if args.command == "seal":
        seed = bytearray(_private_read(args.seed_file))
        try:
            envelope = seal(bytes(seed), json.loads(_private_read(args.policy)), kms_key_id=args.kms_key_id, kms_client=kms)
        finally:
            for index in range(len(seed)):
                seed[index] = 0
        descriptor = os.open(str(args.output), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(envelope, target, sort_keys=True)
            target.flush()
            os.fsync(target.fileno())
        print("Arena encrypted envelope created")
    elif args.command == "provision":
        from validator_tee.host.vsock_client import ValidatorEnclaveClient
        result = provision(envelope, client=ValidatorEnclaveClient(), kms_client=kms)
        print("Arena hotkey provisioned: %s" % result["validator_hotkey"])
    else:
        from validator_tee.host.vsock_client import ValidatorEnclaveClient
        result = provision_legacy(
            json.loads(_private_read(args.legacy_envelope)),
            expected_policy=json.loads(_private_read(args.policy)),
            kms_key_id=args.kms_key_id,
            client=ValidatorEnclaveClient(), kms_client=kms,
        )
        print("Arena hotkey migrated: %s" % result["validator_hotkey"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
