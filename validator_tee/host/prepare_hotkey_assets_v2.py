"""Build the measured validator hotkey config and KMS ciphertext envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
from typing import Any, Dict, Optional, Sequence

from leadpoet_canonical.attested_v2 import sha256_json
from validator_tee.enclave.hotkey_authority_v2 import (
    HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
    MEASURED_DRAND_LIBRARY_PATH,
    load_chain_signing_profile,
    validate_hotkey_authority_configuration,
)
from validator_tee.host.hotkey_bootstrap_v2 import build_hotkey_envelope_v2


class ValidatorHotkeyAssetPreparationV2Error(RuntimeError):
    """The sensitive operator input cannot produce bound V2 assets."""


def _secure_seed(path: Path) -> bytearray:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValidatorHotkeyAssetPreparationV2Error(
            "validator hotkey seed file is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_mode & 0o077
    ):
        raise ValidatorHotkeyAssetPreparationV2Error(
            "validator hotkey seed file must be a private regular file"
        )
    payload = bytearray(path.read_bytes())
    if len(payload) != 32:
        raise ValidatorHotkeyAssetPreparationV2Error(
            "validator hotkey seed must be exactly 32 bytes"
        )
    return payload


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    path.chmod(0o600)


def prepare_hotkey_assets_v2(
    *,
    seed_file: Path,
    validator_hotkey: str,
    kms_key_id: str,
    chain_profile_path: Path,
    drand_hash_path: Path,
    config_output: Path,
    envelope_output: Path,
    kms_client: Any = None,
) -> Dict[str, str]:
    if config_output.exists() or envelope_output.exists():
        raise ValidatorHotkeyAssetPreparationV2Error(
            "validator V2 hotkey output already exists"
        )
    seed = _secure_seed(seed_file)
    try:
        import sr25519
        from scalecodec.utils.ss58 import ss58_encode

        public_key, _private_key = sr25519.pair_from_seed(bytes(seed))
        public_key_hex = bytes(public_key).hex()
        derived_hotkey = ss58_encode(public_key_hex, ss58_format=42)
        if derived_hotkey != validator_hotkey:
            raise ValidatorHotkeyAssetPreparationV2Error(
                "validator hotkey seed derives another SS58 address"
            )
        chain_profile = load_chain_signing_profile(chain_profile_path)
        try:
            drand_hash = drand_hash_path.read_text(encoding="ascii").strip().lower()
        except OSError as exc:
            raise ValidatorHotkeyAssetPreparationV2Error(
                "pinned drand library hash is unavailable"
            ) from exc
        configuration = validate_hotkey_authority_configuration(
            {
                "schema_version": HOTKEY_AUTHORITY_CONFIG_SCHEMA_VERSION,
                "validator_hotkey": validator_hotkey,
                "hotkey_public_key": public_key_hex,
                "chain_signing_profile_hash": sha256_json(chain_profile),
                "drand_library_path": MEASURED_DRAND_LIBRARY_PATH,
                "drand_library_sha256": drand_hash,
            }
        )
        envelope = build_hotkey_envelope_v2(
            validator_hotkey=validator_hotkey,
            hotkey_public_key=public_key_hex,
            seed=bytes(seed),
            kms_key_id=kms_key_id,
            encryption_context={
                "leadpoet:hotkey": validator_hotkey,
                "leadpoet:purpose": "validator-hotkey-unseal-v2",
            },
            kms_client=kms_client,
        )
        _write_json(config_output, configuration)
        _write_json(envelope_output, envelope)
        return {
            "validator_hotkey": validator_hotkey,
            "hotkey_public_key": public_key_hex,
            "hotkey_configuration_hash": sha256_json(configuration),
        }
    finally:
        for index in range(len(seed)):
            seed[index] = 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--validator-hotkey", required=True)
    parser.add_argument("--kms-key-id", required=True)
    parser.add_argument("--chain-profile", type=Path, required=True)
    parser.add_argument("--drand-hash", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--envelope-output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_hotkey_assets_v2(
        seed_file=args.seed_file,
        validator_hotkey=args.validator_hotkey,
        kms_key_id=args.kms_key_id,
        chain_profile_path=args.chain_profile,
        drand_hash_path=args.drand_hash,
        config_output=args.config_output,
        envelope_output=args.envelope_output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
