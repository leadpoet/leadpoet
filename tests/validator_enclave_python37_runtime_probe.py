#!/usr/bin/env python3
"""Execute the enclave finalization proof under its production interpreter."""

import importlib.util
from pathlib import Path

from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    sha256_bytes,
    sha256_json,
)
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ARCHIVE_ENDPOINT_HOST,
    CHAIN_ENDPOINT_HOST,
    chain_source_policy_hash,
    timelocked_weight_commits_storage_key,
)
from leadpoet_canonical.hotkey_authority_v2 import signed_extrinsic_hash_v2
from leadpoet_canonical.weight_computation import _ordered_float_sum


CHAIN_SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "validator_tee"
    / "enclave"
    / "chain_source_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "leadpoet_validator_enclave_chain_source_v2",
    str(CHAIN_SOURCE_PATH),
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("validator enclave chain source could not be loaded")
CHAIN_SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHAIN_SOURCE)
ValidatorChainSourceV2 = CHAIN_SOURCE.ValidatorChainSourceV2


BLOCK = 8_692_918
EPOCH = 24_134
SUBNET_EPOCH_INDEX = 24_081
OWNER = bytes.fromhex("11" * 32)


def transport_attempt(kwargs, *, archive=False):
    operation_prefix = kwargs["job_id"] + ":"
    logical_operation_id = kwargs["logical_operation_id"]
    if not logical_operation_id.startswith(operation_prefix):
        raise AssertionError("unexpected logical operation identity")
    operation = logical_operation_id[len(operation_prefix) :]
    body = operation.encode("utf-8")
    return build_transport_attempt(
        request_id="%032x" % kwargs["request_id"],
        logical_operation_id=logical_operation_id,
        job_id=kwargs["job_id"],
        purpose=kwargs["purpose"],
        provider_id="bittensor_archive" if archive else "bittensor_chain",
        attempt_number=0,
        method="POST",
        destination_host=(
            CHAIN_ARCHIVE_ENDPOINT_HOST if archive else CHAIN_ENDPOINT_HOST
        ),
        destination_port=443,
        path_hash=sha256_json({"path": "/"}),
        nonsecret_headers_hash=sha256_json({"content-type": "application/json"}),
        body_hash=sha256_bytes(body),
        credential_ref_hash=sha256_json({"credential": "none"}),
        retry_policy_hash=chain_source_policy_hash(),
        timeout_ms=30_000,
        started_at="2026-07-24T00:00:00Z",
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=sha256_bytes(body),
        request_artifact_hash=sha256_bytes(body),
        response_artifact_hash=sha256_bytes(body),
        tls_peer_chain_hash=sha256_json([sha256_bytes(b"certificate")]),
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at="2026-07-24T00:00:00Z",
    )


def main() -> None:
    # Python 3.12's built-in sum() returns 1.0 for this sequence while the
    # validator enclave's Python 3.7 returns 0.0. Consensus arithmetic must
    # retain the established enclave result on both runtimes.
    if _ordered_float_sum([1e16, 1.0, -1e16]) != 0.0:
        raise SystemExit("canonical float summation depends on the interpreter")

    extrinsic = b"\x10\x84python37-finalization-rehearsal"
    extrinsic_hash = signed_extrinsic_hash_v2(extrinsic)
    commitment = b"python37-commitment"
    reveal_round = 30_702_160
    storage = b"".join(
        (
            b"\x04",
            OWNER,
            BLOCK.to_bytes(8, "little"),
            bytes((len(commitment) << 2,)),
            commitment,
            reveal_round.to_bytes(8, "little"),
        )
    )

    def rpc_call(**kwargs):
        method = kwargs["method"]
        if method == "chain_getFinalizedHead":
            result = "0x" + "ab" * 32
        elif method == "chain_getHeader":
            result = {
                "number": hex(BLOCK),
                "stateRoot": "0x" + "12" * 32,
                "parentHash": "0x" + "34" * 32,
                "extrinsicsRoot": "0x" + "56" * 32,
            }
        elif method == "chain_getBlockHash":
            result = "0x" + "cd" * 32
        elif method == "chain_getBlock":
            result = {
                "block": {
                    "header": {
                        "number": hex(BLOCK),
                        "stateRoot": "0x" + "12" * 32,
                        "parentHash": "0x" + "34" * 32,
                        "extrinsicsRoot": "0x" + "56" * 32,
                    },
                    "extrinsics": ["0x" + extrinsic.hex()],
                },
                "justifications": None,
            }
        elif method == "state_getStorage":
            expected_key = timelocked_weight_commits_storage_key(
                netuid=71,
                subnet_epoch_index=SUBNET_EPOCH_INDEX,
            )
            if kwargs["params"][0] != expected_key:
                raise AssertionError("unexpected finalized storage key")
            result = "0x" + storage.hex()
        else:
            raise AssertionError("unexpected RPC method: %s" % method)
        return {
            "result": result,
            "attempts": [transport_attempt(kwargs)],
            "artifacts": [],
        }

    def archive_rpc_call(**kwargs):
        result = rpc_call(**kwargs)
        return {
            **result,
            "attempts": [transport_attempt(kwargs, archive=True)],
        }

    result = ValidatorChainSourceV2(
        rpc_call=rpc_call,
        archive_rpc_call=archive_rpc_call,
        finalization_sleep=lambda _seconds: None,
        epoch_authority_supplier=lambda: {
            "mode": "stateful_v1",
            "cutover_manifest": {},
        },
    ).find_finalized_extrinsic_inclusion(
        expected_extrinsics={extrinsic_hash: extrinsic.hex()},
        expected_commitments={
            extrinsic_hash: {
                "netuid": 71,
                "subnet_epoch_index": SUBNET_EPOCH_INDEX,
                "hotkey_public_key": OWNER.hex(),
                "commitment_hex": commitment.hex(),
                "reveal_round": reveal_round,
            }
        },
        minimum_block=BLOCK - 1,
        maximum_block=BLOCK,
        epoch_id=EPOCH,
        finalization_scan_id="sha256:" + "9" * 64,
    )
    if result["extrinsic_hash"] != extrinsic_hash:
        raise SystemExit("finalization proof returned the wrong extrinsic")
    if result["finalized_block"] != BLOCK:
        raise SystemExit("finalization proof returned the wrong block")
    print("PYTHON37_FINALIZATION_PROBE_SUCCESS")


if __name__ == "__main__":
    main()
