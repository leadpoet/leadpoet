"""Seal the existing testnet validator key directly into a KMS envelope.

The source Bittensor keyfile never leaves the source validator host.  This
helper reads it once in memory, verifies the key through Bittensor and sr25519,
and writes only a public measured configuration plus KMS ciphertext envelope.
It never creates a raw-seed file or prints secret material.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from leadpoet_canonical.attested_v2 import sha256_json
from validator_tee.enclave.hotkey_authority_v2 import (
    HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
    MEASURED_DRAND_LIBRARY_PATH,
    load_chain_signing_profile,
    validate_hotkey_authority_configuration,
)
from validator_tee.host.hotkey_bootstrap_v2 import build_hotkey_envelope_v2


class TemporaryTestnetHotkeySealError(RuntimeError):
    """The source key cannot safely produce measured ciphertext assets."""


def _private_regular(path: Path, field: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TemporaryTestnetHotkeySealError(f"{field} is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_mode & 0o077
    ):
        raise TemporaryTestnetHotkeySealError(
            f"{field} must be a private regular file"
        )


def _load_keyfile_document(payload: bytearray) -> Dict[str, Any]:
    try:
        value = json.loads(bytes(payload))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TemporaryTestnetHotkeySealError("Bittensor keyfile is invalid") from exc
    if not isinstance(value, Mapping):
        raise TemporaryTestnetHotkeySealError("Bittensor keyfile is invalid")
    return dict(value)


def _seed_from_document(document: Mapping[str, Any]) -> bytearray:
    encoded = str(document.get("secretSeed") or "").removeprefix("0x")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", encoded):
        raise TemporaryTestnetHotkeySealError(
            "Bittensor keyfile does not contain one 32-byte seed"
        )
    return bytearray.fromhex(encoded)


def _write_assets(
    *,
    config_output: Path,
    envelope_output: Path,
    configuration: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> None:
    if config_output.exists() or envelope_output.exists():
        raise TemporaryTestnetHotkeySealError("hotkey output already exists")
    config_output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    envelope_output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    wrote_config = False
    try:
        config_descriptor = os.open(
            config_output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(config_descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(configuration), handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        wrote_config = True
        envelope_descriptor = os.open(
            envelope_output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(envelope_descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(envelope), handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if wrote_config and config_output.is_file() and not config_output.is_symlink():
            config_output.unlink()
        raise


def seal_temporary_testnet_hotkey(
    *,
    keyfile_path: Path,
    expected_hotkey: str,
    kms_key_id: str,
    chain_profile_path: Path,
    drand_hash_path: Path,
    config_output: Path,
    envelope_output: Path,
    kms_client: Any = None,
    keyfile_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, str]:
    """Create ciphertext assets without materializing a raw seed file."""

    _private_regular(keyfile_path, "Bittensor keyfile")
    payload = bytearray(keyfile_path.read_bytes())
    seed = bytearray()
    document: Dict[str, Any] = {}
    keyfile = None
    keypair = None
    try:
        if keyfile_factory is None:
            from bittensor_wallet import Keyfile

            keyfile_factory = Keyfile
        keyfile = keyfile_factory(path=str(keyfile_path))
        if keyfile.is_encrypted():
            raise TemporaryTestnetHotkeySealError(
                "Bittensor keyfile is encrypted and requires an SDK unlock path"
            )
        keypair = keyfile.keypair
        if str(keypair.ss58_address) != expected_hotkey:
            raise TemporaryTestnetHotkeySealError(
                "Bittensor keyfile hotkey differs from expected validator"
            )
        document = _load_keyfile_document(payload)
        seed = _seed_from_document(document)

        import sr25519
        from scalecodec.utils.ss58 import ss58_encode

        public_key, _private_key = sr25519.pair_from_seed(bytes(seed))
        public_key_hex = bytes(public_key).hex()
        if ss58_encode(public_key_hex, ss58_format=42) != expected_hotkey:
            raise TemporaryTestnetHotkeySealError(
                "Bittensor seed derives another validator hotkey"
            )
        observed_public_key = bytes(keypair.public_key).hex()
        if observed_public_key != public_key_hex:
            raise TemporaryTestnetHotkeySealError(
                "Bittensor SDK and sr25519 public keys differ"
            )
        profile = load_chain_signing_profile(chain_profile_path)
        if (
            profile.get("network") != "test"
            or profile.get("chain_endpoint")
            != "wss://test.finney.opentensor.ai:443"
        ):
            raise TemporaryTestnetHotkeySealError(
                "chain signing profile is not the measured test network"
            )
        try:
            drand_hash = drand_hash_path.read_text(encoding="ascii").strip().lower()
        except OSError as exc:
            raise TemporaryTestnetHotkeySealError(
                "pinned drand library hash is unavailable"
            ) from exc
        configuration = validate_hotkey_authority_configuration(
            {
                "schema_version": HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
                "validator_hotkey": expected_hotkey,
                "hotkey_public_key": public_key_hex,
                "chain_signing_profile_hash": sha256_json(profile),
                "drand_library_path": MEASURED_DRAND_LIBRARY_PATH,
                "drand_library_sha256": drand_hash,
            }
        )
        envelope = build_hotkey_envelope_v2(
            validator_hotkey=expected_hotkey,
            hotkey_public_key=public_key_hex,
            seed=bytes(seed),
            kms_key_id=kms_key_id,
            encryption_context={
                "leadpoet:hotkey": expected_hotkey,
                "leadpoet:purpose": "validator-hotkey-unseal-v2",
            },
            kms_client=kms_client,
        )
        _write_assets(
            config_output=config_output,
            envelope_output=envelope_output,
            configuration=configuration,
            envelope=envelope,
        )
        return {
            "validator_hotkey": expected_hotkey,
            "hotkey_public_key": public_key_hex,
            "hotkey_configuration_hash": sha256_json(configuration),
            "ciphertext_blob_hash": str(envelope["ciphertext_blob_hash"]),
        }
    finally:
        if document:
            for field in ("secretSeed", "secretPhrase", "privateKey"):
                document[field] = ""
        for buffer in (seed, payload):
            for index in range(len(buffer)):
                buffer[index] = 0
        keypair = None
        keyfile = None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyfile", type=Path, required=True)
    parser.add_argument("--expected-hotkey", required=True)
    parser.add_argument("--kms-key-id", required=True)
    parser.add_argument("--chain-profile", type=Path, required=True)
    parser.add_argument("--drand-hash", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--envelope-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = seal_temporary_testnet_hotkey(
            keyfile_path=args.keyfile,
            expected_hotkey=args.expected_hotkey,
            kms_key_id=args.kms_key_id,
            chain_profile_path=args.chain_profile,
            drand_hash_path=args.drand_hash,
            config_output=args.config_output,
            envelope_output=args.envelope_output,
        )
    except Exception:
        print(
            json.dumps(
                {
                    "schema_version": "leadpoet.temporary_testnet_hotkey_seal.v1",
                    "status": "failed",
                    "error": "temporary hotkey sealing failed",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
