"""Fail closed when the measured signing profile differs from the live chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import bittensor as bt

from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.chain_source_v2 import (
    ChainSourceV2Error,
    parse_runtime_version,
)
from validator_tee.enclave.hotkey_authority_v2 import (
    load_chain_signing_profile,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    HotkeyAuthorityV2Error,
    select_chain_signing_profile,
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
        normalized_runtime = parse_runtime_version(runtime_version)
        observed_spec = normalized_runtime["spec_version"]
        observed_transaction = normalized_runtime["transaction_version"]
    except ChainSourceV2Error as exc:
        raise ChainSigningProfileV2Error(
            "live runtime version response is invalid"
        ) from exc

    try:
        selected = select_chain_signing_profile(
            expected,
            runtime_version={
                "specVersion": observed_spec,
                "transactionVersion": observed_transaction,
            },
            genesis_hash=observed_genesis,
        )
    except HotkeyAuthorityV2Error as exc:
        raise ChainSigningProfileV2Error(
            "chain signing profile differs from live runtime: %s" % exc
        ) from exc

    return {
        "schema_version": "leadpoet.chain_signing_profile_compatibility.v2",
        "status": "ready",
        "spec_version": observed_spec,
        "transaction_version": observed_transaction,
        "genesis_hash": observed_genesis,
        "selected_profile_hash": sha256_json(selected),
    }


def read_live_chain_signing_state(network: str) -> Dict[str, Any]:
    subtensor = bt.Subtensor(network=str(network))
    finalized_hash = str(
        _rpc_result(
            subtensor.substrate.rpc_request(
                "chain_getFinalizedHead", []
            ),
            "chain_getFinalizedHead",
        )
        or ""
    )
    runtime_version = _rpc_result(
        subtensor.substrate.rpc_request(
            "state_getRuntimeVersion", [finalized_hash]
        ),
        "state_getRuntimeVersion",
    )
    finalized_header = _rpc_result(
        subtensor.substrate.rpc_request(
            "chain_getHeader", [finalized_hash]
        ),
        "chain_getHeader",
    )
    genesis_hash = _rpc_result(
        subtensor.substrate.rpc_request("chain_getBlockHash", [0]),
        "chain_getBlockHash",
    )
    if (
        not finalized_hash.startswith("0x")
        or len(finalized_hash) != 66
        or not isinstance(runtime_version, Mapping)
        or not isinstance(finalized_header, Mapping)
    ):
        raise ChainSigningProfileV2Error(
            "exact finalized runtime response is invalid"
        )
    try:
        finalized_block = int(str(finalized_header["number"]), 16)
    except (KeyError, TypeError, ValueError) as exc:
        raise ChainSigningProfileV2Error(
            "finalized runtime header is invalid"
        ) from exc
    return {
        "runtime_version": dict(runtime_version),
        "genesis_hash": str(genesis_hash or ""),
        "finalized_block_hash": finalized_hash,
        "finalized_block": finalized_block,
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
    result["finalized_block"] = live["finalized_block"]
    result["finalized_block_hash"] = live["finalized_block_hash"]
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
