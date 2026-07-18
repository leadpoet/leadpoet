"""Refresh release-measured fields in an existing validator hotkey config."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Optional, Sequence

from leadpoet_canonical.attested_v2 import sha256_json
from validator_tee.enclave.hotkey_authority_v2 import (
    MEASURED_DRAND_LIBRARY_PATH,
    load_chain_signing_profile,
    validate_hotkey_authority_configuration,
)


class ValidatorHotkeyConfigRefreshV2Error(RuntimeError):
    """The existing public config cannot be safely rebound to this release."""


def _read_drand_hash(path: Path) -> str:
    try:
        value = path.read_text(encoding="ascii").strip().lower()
    except OSError as exc:
        raise ValidatorHotkeyConfigRefreshV2Error(
            "pinned drand library hash is unavailable"
        ) from exc
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValidatorHotkeyConfigRefreshV2Error(
            "pinned drand library hash is invalid"
        )
    return value


def _atomic_write_private_json(path: Path, value: Dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def refresh_hotkey_config_v2(
    *,
    config_path: Path,
    chain_profile_path: Path,
    drand_hash_path: Path,
) -> Dict[str, str]:
    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatorHotkeyConfigRefreshV2Error(
            "validator hotkey config is unavailable or invalid"
        ) from exc

    normalized = validate_hotkey_authority_configuration(existing)
    refreshed = validate_hotkey_authority_configuration(
        {
            **normalized,
            "chain_signing_profile_hash": sha256_json(
                load_chain_signing_profile(chain_profile_path)
            ),
            "drand_library_path": MEASURED_DRAND_LIBRARY_PATH,
            "drand_library_sha256": _read_drand_hash(drand_hash_path),
        }
    )
    status = "unchanged"
    if refreshed != normalized:
        _atomic_write_private_json(config_path, refreshed)
        status = "refreshed"
    else:
        config_path.chmod(0o600)
    return {
        "status": status,
        "validator_hotkey": refreshed["validator_hotkey"],
        "chain_signing_profile_hash": refreshed["chain_signing_profile_hash"],
        "drand_library_sha256": refreshed["drand_library_sha256"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chain-profile", type=Path, required=True)
    parser.add_argument("--drand-hash", type=Path, required=True)
    args = parser.parse_args(argv)
    result = refresh_hotkey_config_v2(
        config_path=args.config,
        chain_profile_path=args.chain_profile,
        drand_hash_path=args.drand_hash,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
