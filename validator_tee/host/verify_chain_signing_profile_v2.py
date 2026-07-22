"""Fail closed when the measured signing profile differs from the live chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import bittensor as bt

from validator_tee.enclave.hotkey_authority_v2 import (
    load_chain_signing_profile,
)


class ChainSigningProfileV2Error(RuntimeError):
    """The measured chain signing profile cannot authorize the live runtime."""


def _rpc_result(value: Any, field: str) -> Any:
    if not isinstance(value, Mapping) or "result" not in value:
        raise ChainSigningProfileV2Error(f"{field} RPC response is invalid")
    return value["result"]


def verify_chain_signing_profile_v2(
    *,
    profile: Mapping[str, Any],
    runtime_version: Mapping[str, Any],
    genesis_hash: str,
) -> Dict[str, Any]:
    expected = dict(profile)
    observed_genesis = str(genesis_hash or "").lower().removeprefix("0x")
    try:
        observed_spec = int(runtime_version["specVersion"])
        observed_transaction = int(runtime_version["transactionVersion"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ChainSigningProfileV2Error(
            "live runtime version response is invalid"
        ) from exc

    mismatches = []
    if observed_spec != int(expected["spec_version"]):
        mismatches.append(
            f"specVersion live={observed_spec} measured={expected['spec_version']}"
        )
    if observed_transaction != int(expected["transaction_version"]):
        mismatches.append(
            "transactionVersion live=%s measured=%s"
            % (observed_transaction, expected["transaction_version"])
        )
    if observed_genesis != str(expected["genesis_hash"]).lower():
        mismatches.append("genesis hash differs from measured profile")
    if mismatches:
        raise ChainSigningProfileV2Error(
            "chain signing profile differs from live runtime: "
            + "; ".join(mismatches)
        )

    return {
        "schema_version": "leadpoet.chain_signing_profile_compatibility.v2",
        "status": "ready",
        "spec_version": observed_spec,
        "transaction_version": observed_transaction,
        "genesis_hash": observed_genesis,
    }


def read_live_chain_signing_state(network: str) -> Dict[str, Any]:
    subtensor = bt.Subtensor(network=str(network))
    runtime_version = _rpc_result(
        subtensor.substrate.rpc_request("state_getRuntimeVersion", []),
        "state_getRuntimeVersion",
    )
    genesis_hash = _rpc_result(
        subtensor.substrate.rpc_request("chain_getBlockHash", [0]),
        "chain_getBlockHash",
    )
    if not isinstance(runtime_version, Mapping):
        raise ChainSigningProfileV2Error(
            "state_getRuntimeVersion result is invalid"
        )
    return {
        "runtime_version": dict(runtime_version),
        "genesis_hash": str(genesis_hash or ""),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="finney")
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args(argv)

    profile = load_chain_signing_profile(args.profile)
    live = read_live_chain_signing_state(args.network)
    result = verify_chain_signing_profile_v2(
        profile=profile,
        runtime_version=live["runtime_version"],
        genesis_hash=live["genesis_hash"],
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
