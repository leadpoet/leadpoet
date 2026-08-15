"""Configure the validator enclave from independently reproduced V2 releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from leadpoet_observability import (
    capture_failure,
    failure_code_for_exception,
    init_sentry,
    record_stage,
)

from gateway.tee.release_manifest_v2 import validate_release_manifest
from Leadpoet.utils.subnet_epoch import (
    load_subnet_epoch_cutover,
)
from leadpoet_canonical.attested_v2 import sha256_json, verify_boot_identity_nitro
from validator_tee.host.release_v2 import validator_release_authority
from validator_tee.host.vsock_client import ValidatorEnclaveClient
from validator_tee.enclave.runtime_v2 import (
    VALIDATOR_RUNTIME_STATEFUL_CONFIG_SCHEMA_VERSION,
    validate_gateway_release_lineage,
)
from validator_tee.enclave.hotkey_authority_v2 import (
    hotkey_authority_configuration_hash,
    load_chain_signing_profile,
    validate_hotkey_authority_configuration,
)
from leadpoet_canonical.chain_source_v2 import (
    chain_source_boundary_for_profile_v2,
)


class ValidatorRuntimeBootstrapV2Error(RuntimeError):
    """A local or independently rebuilt release cannot authorize startup."""


def build_runtime_configuration(
    *,
    validator_release: Mapping[str, Any],
    gateway_release: Mapping[str, Any],
    gateway_release_lineage: Mapping[str, Any],
    hotkey_authority_config: Mapping[str, Any],
) -> Dict[str, Any]:
    validator = validator_release_authority(validator_release)
    gateway = validate_release_manifest(gateway_release)
    hotkey_config = validate_hotkey_authority_configuration(
        hotkey_authority_config
    )
    chain_profile = load_chain_signing_profile(
        Path(__file__).resolve().parents[1]
        / "enclave"
        / "chain_signing_profile_v2.json",
        expected_hash=hotkey_config["chain_signing_profile_hash"],
    )
    if validator["commit_sha"] != gateway["commit_sha"]:
        raise ValidatorRuntimeBootstrapV2Error(
            "validator and gateway V2 releases use different commits"
        )
    lineage = validate_gateway_release_lineage(gateway_release_lineage)
    if (
        lineage["current_commit_sha"] != gateway["commit_sha"]
        or lineage["current_gateway_release_hash"] != gateway["release_hash"]
    ):
        raise ValidatorRuntimeBootstrapV2Error(
            "current gateway release differs from approved release lineage"
        )
    cutover = load_subnet_epoch_cutover()
    if cutover.network_genesis_hash.lower() != (
        "0x" + str(chain_profile["genesis_hash"]).lower()
    ):
        raise ValidatorRuntimeBootstrapV2Error(
            "chain signing profile differs from the epoch authority"
        )
    configuration = {
        "schema_version": VALIDATOR_RUNTIME_STATEFUL_CONFIG_SCHEMA_VERSION,
        "commit_sha": validator["commit_sha"],
        "build_manifest_hash": validator["app_manifest_hash"],
        "dependency_lock_hash": validator["dependency_lock_hash"],
        "gateway_release_hash": gateway["release_hash"],
        "hotkey_authority_config_hash": hotkey_authority_configuration_hash(
            hotkey_config
        ),
        "chain_source_boundary": chain_source_boundary_for_profile_v2(
            chain_profile
        ),
        "gateway_release_lineage": lineage,
        "epoch_authority": {
            "mode": "stateful_v1",
            "cutover_manifest": cutover.to_dict(),
        },
    }
    return configuration


def configure_validator_runtime_v2(
    *,
    validator_release: Mapping[str, Any],
    gateway_release: Mapping[str, Any],
    gateway_release_lineage: Mapping[str, Any],
    hotkey_authority_config: Mapping[str, Any],
    client: Optional[ValidatorEnclaveClient] = None,
    boot_verifier=verify_boot_identity_nitro,
) -> Dict[str, Any]:
    configuration = build_runtime_configuration(
        validator_release=validator_release,
        gateway_release=gateway_release,
        gateway_release_lineage=gateway_release_lineage,
        hotkey_authority_config=hotkey_authority_config,
    )
    validator = validator_release_authority(validator_release)
    expected_config_hash = sha256_json(configuration)
    enclave_client = client or ValidatorEnclaveClient()
    boot = enclave_client.configure_authoritative_v2(
        configuration,
        expected_config_hash,
    )
    try:
        boot_verifier(boot, expected_pcr0=validator["pcr0"])
    except Exception as exc:
        failure_code = failure_code_for_exception(
            exc,
            default="restart.terminal_failure",
        )
        capture_failure(
            failure_code,
            component="validator",
            stage="validator_boot_attestation_verification",
            exception=exc,
            terminal=True,
            retryable=False,
            fail_closed=True,
            runtime_sha=validator["commit_sha"],
            expected_pcr0_hash=validator["pcr0"],
            observed_pcr0_hash=boot.get("pcr0"),
            boot_identity_hash=boot.get("boot_identity_hash"),
        )
        raise
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
    record_stage(
        component="validator",
        stage="validator_boot_attestation_verification",
        status="passed",
        runtime_sha=validator["commit_sha"],
        expected_pcr0_hash=validator["pcr0"],
        observed_pcr0_hash=boot.get("pcr0"),
        boot_identity_hash=boot.get("boot_identity_hash"),
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
    init_sentry(component="validator-runtime-bootstrap")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator-release", type=Path, required=True)
    parser.add_argument("--gateway-release", type=Path, required=True)
    parser.add_argument("--gateway-release-lineage", type=Path, required=True)
    parser.add_argument("--hotkey-config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = configure_validator_runtime_v2(
        validator_release=_load(args.validator_release),
        gateway_release=_load(args.gateway_release),
        gateway_release_lineage=_load(args.gateway_release_lineage),
        hotkey_authority_config=_load(args.hotkey_config),
    )
    boot = result["boot_identity"]
    print("validator_v2_boot_identity_hash=%s" % boot["boot_identity_hash"])
    print("validator_v2_pcr0=%s" % boot["pcr0"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
