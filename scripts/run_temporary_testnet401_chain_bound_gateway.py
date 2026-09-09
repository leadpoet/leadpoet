"""Start the temporary gateway after binding canonical chain consumers to testnet401."""

from __future__ import annotations


def configure_temporary_testnet401_chain_source(
    *, profile_path, hotkey_config_path, role
):
    """Configure the measured test source before importing chain consumers."""

    import json
    import os
    from pathlib import Path

    expected_endpoint = "wss://test.finney.opentensor.ai:443"
    expected_genesis = (
        "8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
    )
    expected_hotkey = "5CJyMxw6YJJvLhPf58gSpMB7mvSKSCMx9RXhXJum6cNfqMEz"
    if role not in {"gateway", "validator"}:
        raise RuntimeError("temporary chain source role is invalid")
    role_environment = {
        "gateway": {
            "ALLOWED_NETUIDS": "401",
            "PRIMARY_VALIDATOR_HOTKEYS": expected_hotkey,
            "BT_SUBTENSOR_NETWORK": "test",
            "BT_SUBTENSOR_CHAIN_ENDPOINT": expected_endpoint,
        },
        "validator": {
            "BITTENSOR_NETWORK": "test",
            "BITTENSOR_NETUID": "401",
            "VALIDATOR_SUBTENSOR_NETWORK": "test",
            "EXPECTED_CHAIN": expected_endpoint,
        },
    }[role]
    if any(os.environ.get(name) != value for name, value in role_environment.items()):
        raise RuntimeError("temporary chain source environment differs")

    from leadpoet_canonical.attested_v2 import sha256_json
    from leadpoet_canonical.chain_source_v2 import (
        chain_source_boundary_for_profile_v2,
        configure_chain_source_boundary_v2,
    )
    from leadpoet_canonical.hotkey_authority_v2 import (
        validate_chain_signing_profile,
    )

    profile = validate_chain_signing_profile(
        json.loads(Path(profile_path).read_text(encoding="utf-8"))
    )
    if (
        profile["network"] != "test"
        or profile["chain_endpoint"] != expected_endpoint
        or profile["genesis_hash"] != expected_genesis
    ):
        raise RuntimeError("temporary chain signing profile differs")
    raw_hotkey = json.loads(Path(hotkey_config_path).read_text(encoding="utf-8"))
    expected_hotkey_fields = {
        "schema_version", "validator_hotkey", "hotkey_public_key",
        "chain_signing_profile_hash", "drand_library_path",
        "drand_library_sha256",
    }
    if (
        not isinstance(raw_hotkey, dict)
        or set(raw_hotkey) != expected_hotkey_fields
        or raw_hotkey.get("schema_version")
        != "leadpoet.validator_hotkey_config.v2"
        or raw_hotkey.get("validator_hotkey") != expected_hotkey
        or raw_hotkey.get("chain_signing_profile_hash") != sha256_json(profile)
        or raw_hotkey.get("drand_library_path")
        != "/app/validator_tee/enclave/libbittensor_drand_v2.so"
    ):
        raise RuntimeError("temporary chain signing profile binding differs")
    boundary = chain_source_boundary_for_profile_v2(profile)
    configure_chain_source_boundary_v2(
        chain_host=boundary["chain_host"],
        chain_archive_host=boundary["chain_archive_host"],
    )

    # This import also loads the chain-evidence consumers. It must remain after
    # the one-time boundary configuration above.
    from validator_tee.enclave.hotkey_authority_v2 import (
        validate_hotkey_authority_configuration,
    )

    hotkey = validate_hotkey_authority_configuration(
        raw_hotkey
    )
    if (
        hotkey["validator_hotkey"] != expected_hotkey
        or hotkey["chain_signing_profile_hash"] != sha256_json(profile)
    ):
        raise RuntimeError("temporary chain signing profile binding differs")

    from leadpoet_canonical import compact_auditor_authority_v2
    from leadpoet_canonical import weight_authority_v2

    expected_host = boundary["chain_host"]
    if (
        weight_authority_v2.CHAIN_ENDPOINT_HOST != expected_host
        or compact_auditor_authority_v2.CHAIN_ENDPOINT_HOST != expected_host
        or weight_authority_v2.CHAIN_ARCHIVE_ENDPOINT_HOST
        != boundary["chain_archive_host"]
        or compact_auditor_authority_v2.CHAIN_ARCHIVE_ENDPOINT_HOST
        != boundary["chain_archive_host"]
    ):
        raise RuntimeError("temporary canonical chain consumer binding differs")
    return {
        "network": "test",
        "netuid": 401,
        "chain_signing_profile_hash": sha256_json(profile),
        "chain_source_policy_hash": boundary["chain_source_policy_hash"],
    }


def main() -> int:
    import argparse
    from pathlib import Path
    import runpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-profile", type=Path, required=True)
    parser.add_argument("--hotkey-config", type=Path, required=True)
    args = parser.parse_args()
    configure_temporary_testnet401_chain_source(
        profile_path=args.chain_profile,
        hotkey_config_path=args.hotkey_config,
        role="gateway",
    )
    runpy.run_module("gateway.main", run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
