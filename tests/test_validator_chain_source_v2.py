from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from leadpoet_canonical.attested_v2 import build_transport_attempt, sha256_bytes, sha256_json
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ENDPOINT_HOST,
    chain_source_policy_hash,
)
from leadpoet_canonical.hotkey_authority_v2 import signed_extrinsic_hash_v2
from validator_tee.enclave.chain_source_v2 import (
    EnclaveChainRpcTransportV2,
    ValidatorChainSourceV2,
    ValidatorChainSourceV2Error,
)


OWNER = bytes.fromhex(
    "924620afb270acb1ee27bd034aa9e97108ef276da5079db982883cd70294741a"
)
SECOND = bytes.fromhex(
    "74adb27b7edd7126a81f5bac79e9bda1a4c8ec94d2c4f2ce795e0c56932a5383"
)
BLOCK = 8_597_161
EPOCH = BLOCK // 360


def _selective_result() -> str:
    encoded = bytearray((1, 0x1D, 0x01))
    encoded.extend(b"\x00" * 4)
    encoded.extend(b"\x01" + OWNER)
    encoded.extend(b"\x00")
    encoded.extend(b"\x01" + ((BLOCK << 2) | 2).to_bytes(4, "little"))
    encoded.extend(b"\x00" * 44)
    encoded.extend(b"\x01\x08" + OWNER + SECOND)
    encoded.extend(b"\x00" * 21)
    return "0x" + bytes(encoded).hex()


def _attempt(*, job_id: str, purpose: str, operation: str, request_id: str):
    body = json.dumps({"operation": operation}, sort_keys=True).encode()
    return build_transport_attempt(
        request_id=request_id,
        logical_operation_id=job_id + ":" + operation,
        job_id=job_id,
        purpose=purpose,
        provider_id="bittensor_chain",
        attempt_number=0,
        method="POST",
        destination_host=CHAIN_ENDPOINT_HOST,
        destination_port=443,
        path_hash=sha256_json({"path": "/"}),
        nonsecret_headers_hash=sha256_json({"content-type": "application/json"}),
        body_hash=sha256_bytes(body),
        credential_ref_hash=sha256_json({"credential": "none"}),
        retry_policy_hash=chain_source_policy_hash(),
        timeout_ms=30_000,
        started_at="2026-07-10T00:00:00Z",
        terminal_status="authenticated_response",
        http_status=200,
        response_hash=sha256_bytes(body),
        request_artifact_hash=sha256_bytes(body),
        response_artifact_hash=sha256_bytes(body),
        tls_peer_chain_hash=sha256_json([sha256_bytes(b"cert")]),
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at="2026-07-10T00:00:00Z",
    )


def test_finalized_source_uses_one_block_for_header_and_metagraph():
    calls = []

    def rpc_call(**kwargs):
        calls.append(kwargs)
        request_id = kwargs["request_id"]
        if request_id == 1:
            result = "0x" + "ab" * 32
        elif request_id == 2:
            result = {
                "number": hex(BLOCK),
                "stateRoot": "0x" + "12" * 32,
                "parentHash": "0x" + "34" * 32,
                "extrinsicsRoot": "0x" + "56" * 32,
            }
        else:
            result = _selective_result()
        attempt = _attempt(
            job_id=kwargs["job_id"],
            purpose=kwargs["purpose"],
            operation=str(request_id),
            request_id=("%032x" % request_id),
        )
        return {
            "result": result,
            "attempts": [attempt],
            "artifacts": [
                {
                    "artifact_hash": attempt["response_artifact_hash"],
                    "kind": "chain_rpc_response",
                    "body_b64": "e30=",
                }
            ],
        }

    result = ValidatorChainSourceV2(rpc_call=rpc_call).read_finalized_snapshot(
        netuid=71, epoch_id=EPOCH
    )
    assert result["header"]["block"] == BLOCK
    assert result["metagraph"]["block"] == BLOCK
    assert len(result["metagraph"]["hotkeys"]) == 2
    assert calls[2]["params"][2] == "0x" + "ab" * 32
    assert [attempt["purpose"] for attempt in result["attempts"]] == [
        "validator.chain_state.v2",
        "validator.chain_state.v2",
        "validator.metagraph_state.v2",
    ]


def test_finalized_source_rejects_cross_epoch_or_metagraph_block_mismatch():
    def rpc_call(**kwargs):
        request_id = kwargs["request_id"]
        result = (
            "0x" + "ab" * 32
            if request_id == 1
            else {
                "number": hex(BLOCK),
                "stateRoot": "0x" + "12" * 32,
                "parentHash": "0x" + "34" * 32,
                "extrinsicsRoot": "0x" + "56" * 32,
            }
            if request_id == 2
            else _selective_result()
        )
        return {
            "result": result,
            "attempts": [
                _attempt(
                    job_id=kwargs["job_id"],
                    purpose=kwargs["purpose"],
                    operation=str(request_id),
                    request_id=("%032x" % request_id),
                )
            ],
            "artifacts": [],
        }

    with pytest.raises(ValidatorChainSourceV2Error, match="requested epoch"):
        ValidatorChainSourceV2(rpc_call=rpc_call).read_finalized_snapshot(
            netuid=71, epoch_id=EPOCH + 1
        )


def test_transport_retries_malformed_authenticated_reply_without_duplicate_terminal_record():
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    transport = EnclaveChainRpcTransportV2(
        clock=lambda: now,
        sleep=lambda _seconds: None,
    )
    responses = [
        b"not-json",
        b'{"jsonrpc":"2.0","id":1,"result":"0x' + b"ab" * 32 + b'"}',
    ]

    def http_post(_body):
        return {
            "status": 200,
            "body": responses.pop(0),
            "tls_peer_chain_hash": sha256_json([sha256_bytes(b"cert")]),
            "tls_protocol": "TLSv1.3",
        }

    transport._http_post = http_post
    result = transport.call(
        method="chain_getFinalizedHead",
        params=[],
        request_id=1,
        job_id="chain-state:1",
        purpose="validator.chain_state.v2",
        logical_operation_id="chain-state:1:head",
    )
    assert result["result"] == "0x" + "ab" * 32
    assert [attempt["attempt_number"] for attempt in result["attempts"]] == [0, 1]
    assert all(
        attempt["terminal_status"] == "authenticated_response"
        for attempt in result["attempts"]
    )


def test_finalized_extrinsic_requires_exact_bytes_and_committed_chain_state():
    extrinsic = b"\x10\x84measured-weight-extrinsic"
    extrinsic_hash = signed_extrinsic_hash_v2(extrinsic)
    commitment = b"timelocked-commitment"
    reveal_round = 998877
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
            result = "0x" + storage.hex()
        else:
            raise AssertionError(method)
        attempt = _attempt(
            job_id=kwargs["job_id"],
            purpose=kwargs["purpose"],
            operation=method + ":" + str(kwargs["request_id"]),
            request_id=("%032x" % kwargs["request_id"]),
        )
        return {"result": result, "attempts": [attempt], "artifacts": []}

    result = ValidatorChainSourceV2(
        rpc_call=rpc_call
    ).find_finalized_extrinsic_inclusion(
        expected_extrinsics={extrinsic_hash: extrinsic.hex()},
        expected_commitments={
            extrinsic_hash: {
                "netuid": 71,
                "mechid": 0,
                "hotkey_public_key": OWNER.hex(),
                "commitment_hex": commitment.hex(),
                "reveal_round": reveal_round,
            }
        },
        minimum_block=BLOCK,
        maximum_block=BLOCK,
        epoch_id=EPOCH,
    )
    assert result["extrinsic_hash"] == extrinsic_hash
    assert result["finalized_block"] == BLOCK
    assert result["state_transition_hash"].startswith("sha256:")


def test_finalized_extrinsic_rejects_inclusion_without_expected_state_change():
    extrinsic = b"\x10\x84measured-weight-extrinsic"
    extrinsic_hash = signed_extrinsic_hash_v2(extrinsic)

    def rpc_call(**kwargs):
        method = kwargs["method"]
        values = {
            "chain_getFinalizedHead": "0x" + "ab" * 32,
            "chain_getHeader": {
                "number": hex(BLOCK),
                "stateRoot": "0x" + "12" * 32,
                "parentHash": "0x" + "34" * 32,
                "extrinsicsRoot": "0x" + "56" * 32,
            },
            "chain_getBlockHash": "0x" + "cd" * 32,
            "chain_getBlock": {
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
            },
            "state_getStorage": "0x00",
        }
        return {"result": values[method], "attempts": [], "artifacts": []}

    with pytest.raises(
        ValidatorChainSourceV2Error, match="expected chain state"
    ):
        ValidatorChainSourceV2(
            rpc_call=rpc_call
        ).find_finalized_extrinsic_inclusion(
            expected_extrinsics={extrinsic_hash: extrinsic.hex()},
            expected_commitments={
                extrinsic_hash: {
                    "netuid": 71,
                    "mechid": 0,
                    "hotkey_public_key": OWNER.hex(),
                    "commitment_hex": "aa",
                    "reveal_round": 1,
                }
            },
            minimum_block=BLOCK,
            maximum_block=BLOCK,
            epoch_id=EPOCH,
        )
