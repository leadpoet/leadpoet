"""Configure the validator enclave from independently reproduced V2 releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from gateway.tee.release_manifest_v2 import validate_release_manifest
from Leadpoet.utils.subnet_epoch import (
    STATEFUL_EPOCH_MODE,
    get_epoch_mode,
    load_subnet_epoch_cutover,
)
from leadpoet_canonical.attested_v2 import sha256_json, verify_boot_identity_nitro
from validator_tee.host.release_v2 import validator_release_authority
from validator_tee.host.vsock_client import ValidatorEnclaveClient
from validator_tee.enclave.runtime_v2 import (
    VALIDATOR_RUNTIME_CONFIG_SCHEMA_VERSION,
    VALIDATOR_RUNTIME_STATEFUL_CONFIG_SCHEMA_VERSION,
)
from validator_tee.enclave.hotkey_authority_v2 import (
    hotkey_authority_configuration_hash,
    validate_hotkey_authority_configuration,
)


class ValidatorRuntimeBootstrapV2Error(RuntimeError):
    """A local or independently rebuilt release cannot authorize startup."""


def build_runtime_configuration(
    *,
    validator_release: Mapping[str, Any],
    gateway_release: Mapping[str, Any],
    hotkey_authority_config: Mapping[str, Any],
) -> Dict[str, Any]:
    validator = validator_release_authority(validator_release)
    gateway = validate_release_manifest(gateway_release)
    hotkey_config = validate_hotkey_authority_configuration(
        hotkey_authority_config
    )
    if validator["commit_sha"] != gateway["commit_sha"]:
        raise ValidatorRuntimeBootstrapV2Error(
            "validator and gateway V2 releases use different commits"
        )
    expectations = {}
    for role, summary in sorted(gateway["roles"].items()):
        expectations[role] = {
            "commit_sha": summary["commit_sha"],
            "pcr0": summary["pcr0"],
            "build_manifest_hash": summary["execution_manifest_hash"],
        }
    configuration = {
        "schema_version": VALIDATOR_RUNTIME_CONFIG_SCHEMA_VERSION,
        "commit_sha": validator["commit_sha"],
        "build_manifest_hash": validator["app_manifest_hash"],
        "dependency_lock_hash": validator["dependency_lock_hash"],
        "gateway_release_hash": gateway["release_hash"],
        "hotkey_authority_config_hash": hotkey_authority_configuration_hash(
            hotkey_config
        ),
        "gateway_role_expectations": expectations,
    }
    if get_epoch_mode() == STATEFUL_EPOCH_MODE:
        cutover = load_subnet_epoch_cutover()
        configuration.update(
            {
                "schema_version": VALIDATOR_RUNTIME_STATEFUL_CONFIG_SCHEMA_VERSION,
                "epoch_authority": {
                    "mode": STATEFUL_EPOCH_MODE,
                    "cutover_manifest": cutover.to_dict(),
                },
            }
        )
    return configuration


def configure_validator_runtime_v2(
    *,
    validator_release: Mapping[str, Any],
    gateway_release: Mapping[str, Any],
    hotkey_authority_config: Mapping[str, Any],
    client: Optional[ValidatorEnclaveClient] = None,
    boot_verifier=verify_boot_identity_nitro,
) -> Dict[str, Any]:
    configuration = build_runtime_configuration(
        validator_release=validator_release,
        gateway_release=gateway_release,
        hotkey_authority_config=hotkey_authority_config,
    )
    validator = validator_release_authority(validator_release)
    expected_config_hash = sha256_json(configuration)
    enclave_client = client or ValidatorEnclaveClient()
    boot = enclave_client.configure_authoritative_v2(
        configuration,
        expected_config_hash,
    )
    boot_verifier(boot, expected_pcr0=validator["pcr0"])
    expected = {
        "commit_sha": validator["commit_sha"],
        "build_manifest_hash": validator["app_manifest_hash"],
        "dependency_lock_hash": validator["dependency_lock_hash"],
        "config_hash": expected_config_hash,
        "pcr0": validator["pcr0"],
        "role": "validator_weights",
        "physical_role": "validator_weights",
    }
    for field, value in expected.items():
        if boot.get(field) != value:
            raise ValidatorRuntimeBootstrapV2Error(
                "validator boot differs from release at %s" % field
            )
    observed = enclave_client.get_authoritative_v2_boot_identity()
    if observed != boot:
        raise ValidatorRuntimeBootstrapV2Error(
            "validator boot identity readback mismatch"
        )
    return {
        "configuration": configuration,
        "configuration_hash": expected_config_hash,
        "boot_identity": dict(boot),
    }


def _load(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatorRuntimeBootstrapV2Error(
            "V2 release file is unavailable or invalid"
        ) from exc
    if not isinstance(value, dict):
        raise ValidatorRuntimeBootstrapV2Error("V2 release file is not an object")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator-release", type=Path, required=True)
    parser.add_argument("--gateway-release", type=Path, required=True)
    parser.add_argument("--hotkey-config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = configure_validator_runtime_v2(
        validator_release=_load(args.validator_release),
        gateway_release=_load(args.gateway_release),
        hotkey_authority_config=_load(args.hotkey_config),
    )
    boot = result["boot_identity"]
    print("validator_v2_boot_identity_hash=%s" % boot["boot_identity_hash"])
    print("validator_v2_pcr0=%s" % boot["pcr0"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
