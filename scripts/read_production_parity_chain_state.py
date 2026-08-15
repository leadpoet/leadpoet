#!/usr/bin/env python3
"""Read finalized testnet weight state for isolated parity validators.

This is an independent acceptance probe. It never signs or submits an
extrinsic and does not calculate a replacement weight vector.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Sequence
from urllib.parse import urlsplit


HOTKEY_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{40,64}$")
HEX_HASH_RE = re.compile(r"^0x[0-9a-f]{64}$")
SCHEMA_VERSION = "leadpoet.production_parity_chain_readback.v1"


class ProductionParityChainReadbackError(RuntimeError):
    pass


def _value(result: Any) -> Any:
    return getattr(result, "value", result)


def _normalized_endpoint(value: str) -> str:
    endpoint = str(value or "").strip()
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "wss"
        or parsed.hostname != "test.finney.opentensor.ai"
        or parsed.port not in (None, 443)
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ProductionParityChainReadbackError(
            "chain readback endpoint is not the official testnet authority"
        )
    return endpoint


def _normalized_hash(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not HEX_HASH_RE.fullmatch(normalized):
        raise ProductionParityChainReadbackError(f"{field} is invalid")
    return normalized


def _weights(value: Any) -> list[list[int]]:
    rows = _value(value)
    if not isinstance(rows, (list, tuple)):
        raise ProductionParityChainReadbackError(
            "finalized Weights storage response is invalid"
        )
    result: list[list[int]] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ProductionParityChainReadbackError(
                "finalized Weights row is invalid"
            )
        uid = int(row[0])
        weight = int(row[1])
        if uid < 0 or uid in seen or not 0 <= weight <= 65535:
            raise ProductionParityChainReadbackError(
                "finalized Weights row contains invalid values"
            )
        seen.add(uid)
        result.append([uid, weight])
    return sorted(result)


def read_finalized_state(
    *,
    endpoint: str,
    netuid: int,
    hotkeys: Sequence[str],
    expected_genesis_hash: str,
    subtensor_factory: Any = None,
) -> dict[str, Any]:
    normalized_endpoint = _normalized_endpoint(endpoint)
    normalized_genesis = _normalized_hash(
        expected_genesis_hash, field="expected genesis hash"
    )
    normalized_hotkeys = [str(item or "").strip() for item in hotkeys]
    if (
        not isinstance(netuid, int)
        or isinstance(netuid, bool)
        or netuid <= 0
        or len(normalized_hotkeys) != 3
        or len(set(normalized_hotkeys)) != 3
        or any(not HOTKEY_RE.fullmatch(item) for item in normalized_hotkeys)
    ):
        raise ProductionParityChainReadbackError(
            "chain readback validator identity is invalid"
        )

    if subtensor_factory is None:
        import bittensor as bt

        subtensor_factory = bt.Subtensor

    subtensor = subtensor_factory(network=normalized_endpoint)
    try:
        substrate = subtensor.substrate
        genesis_hash = _normalized_hash(
            substrate.get_block_hash(0), field="observed genesis hash"
        )
        if genesis_hash != normalized_genesis:
            raise ProductionParityChainReadbackError(
                "chain readback genesis hash differs"
            )
        finalized_hash = _normalized_hash(
            substrate.get_chain_finalised_head(), field="finalized block hash"
        )
        finalized_block = int(substrate.get_block_number(finalized_hash))
        if finalized_block < 0:
            raise ProductionParityChainReadbackError(
                "finalized block number is invalid"
            )
        last_updates = _value(
            substrate.query(
                module="SubtensorModule",
                storage_function="LastUpdate",
                params=[netuid],
                block_hash=finalized_hash,
            )
        )
        if not isinstance(last_updates, (list, tuple)):
            raise ProductionParityChainReadbackError(
                "finalized LastUpdate storage response is invalid"
            )

        validators = []
        seen_uids: set[int] = set()
        for hotkey in normalized_hotkeys:
            uid_value = _value(
                substrate.query(
                    module="SubtensorModule",
                    storage_function="Uids",
                    params=[netuid, hotkey],
                    block_hash=finalized_hash,
                )
            )
            if uid_value is None:
                raise ProductionParityChainReadbackError(
                    "staging validator is not registered on testnet"
                )
            uid = int(uid_value)
            if uid < 0 or uid in seen_uids or uid >= len(last_updates):
                raise ProductionParityChainReadbackError(
                    "staging validator UID or LastUpdate is invalid"
                )
            seen_uids.add(uid)
            validators.append(
                {
                    "hotkey": hotkey,
                    "uid": uid,
                    "last_update": int(last_updates[uid]),
                    "weights": _weights(
                        substrate.query(
                            module="SubtensorModule",
                            storage_function="Weights",
                            params=[netuid, uid],
                            block_hash=finalized_hash,
                        )
                    ),
                }
            )
        body = {
            "schema_version": SCHEMA_VERSION,
            "network": "test",
            "chain_endpoint_host": "test.finney.opentensor.ai",
            "network_genesis_hash": genesis_hash,
            "netuid": netuid,
            "finalized_block": finalized_block,
            "finalized_block_hash": finalized_hash,
            "validators": validators,
        }
        from leadpoet_canonical.production_parity import sha256_json

        return {**body, "readback_hash": sha256_json(body)}
    finally:
        close = getattr(subtensor, "close", None)
        if callable(close):
            close()
        else:
            substrate_close = getattr(getattr(subtensor, "substrate", None), "close", None)
            if callable(substrate_close):
                substrate_close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--hotkey", action="append", required=True)
    parser.add_argument("--expected-genesis-hash", required=True)
    args = parser.parse_args(argv)
    try:
        result = read_finalized_state(
            endpoint=args.endpoint,
            netuid=args.netuid,
            hotkeys=args.hotkey,
            expected_genesis_hash=args.expected_genesis_hash,
        )
    except (ProductionParityChainReadbackError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
