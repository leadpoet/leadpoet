from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from leadpoet_canonical.attested_v2 import build_transport_attempt, sha256_bytes, sha256_json
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ARCHIVE_ENDPOINT_HOST,
    CHAIN_ENDPOINT_HOST,
    chain_source_policy_hash,
    decode_subnet_epoch_storage,
    decode_timestamp_now_storage,
    timestamp_now_storage_key,
    subnet_epoch_storage_key,
    timelocked_weight_commits_storage_key,
)
from leadpoet_canonical.hotkey_authority_v2 import signed_extrinsic_hash_v2
from validator_tee.enclave.chain_source_v2 import (
    EnclaveChainRpcTransportV2,
    FINALIZATION_RPC_PACING_SECONDS,
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
STATEFUL_BLOCK = 8_637_160
STATEFUL_LAST_EPOCH_BLOCK = 8_637_156
STATEFUL_SUBNET_EPOCH_INDEX = 23_927
STATEFUL_SETTLEMENT_EPOCH_ID = 23_992
GENESIS_HASH = "0x" + "11" * 32
CUTOVER_HASH = "0x" + "22" * 32
PREDECESSOR_HASH = "0x" + "21" * 32
FINALIZED_HASH = "0x" + "ab" * 32
TIMESTAMP_MS = 1_752_710_400_123


def _selective_result(block: int = BLOCK) -> str:
    encoded = bytearray((1, 0x1D, 0x01))
    encoded.extend(b"\x00" * 4)
    encoded.extend(b"\x01" + OWNER)
    encoded.extend(b"\x00")
    encoded.extend(b"\x01" + ((int(block) << 2) | 2).to_bytes(4, "little"))
    encoded.extend(b"\x00" * 44)
    encoded.extend(b"\x01\x08" + OWNER + SECOND)
    encoded.extend(b"\x00" * 24)
    return "0x" + bytes(encoded).hex()


def _stateful_cutover(**updates):
    body = {
        "schema_version": "leadpoet.subnet_epoch_cutover.v1",
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": GENESIS_HASH,
        "netuid": 71,
        "cutover_block": STATEFUL_LAST_EPOCH_BLOCK,
        "cutover_block_hash": CUTOVER_HASH,
        "first_subnet_epoch_index": STATEFUL_SUBNET_EPOCH_INDEX,
        "first_settlement_epoch_id": STATEFUL_SETTLEMENT_EPOCH_ID,
        "last_legacy_epoch_id": STATEFUL_SETTLEMENT_EPOCH_ID - 1,
    }
    body.update(updates)
    return {**body, "mapping_hash": sha256_json(body)}


def _stateful_rpc(
    *,
    storage_updates=None,
    genesis_hash=GENESIS_HASH,
    finalized_hash=FINALIZED_HASH,
    finalized_block=STATEFUL_BLOCK,
):
    calls = []
    storage = {
        "Tempo": 360,
        "LastEpochBlock": STATEFUL_LAST_EPOCH_BLOCK,
        "PendingEpochAt": 0,
        "SubnetEpochIndex": STATEFUL_SUBNET_EPOCH_INDEX,
        "BlocksSinceLastStep": 4,
    }
    storage.update(storage_updates or {})
    storage_names = {
        subnet_epoch_storage_key(storage_name=name, netuid=71): name
        for name in storage
    }
    timestamp_key = timestamp_now_storage_key()

    def rpc_call(**kwargs):
        calls.append(kwargs)
        method = kwargs["method"]
        params = kwargs["params"]
        if method == "chain_getFinalizedHead":
            result = finalized_hash
        elif method == "chain_getHeader":
            block = (
                finalized_block
                if params == [finalized_hash]
                else STATEFUL_LAST_EPOCH_BLOCK
            )
            result = {
                "number": hex(block),
                "stateRoot": "0x" + "12" * 32,
                "parentHash": "0x" + "34" * 32,
                "extrinsicsRoot": "0x" + "56" * 32,
            }
        elif method == "chain_getBlockHash":
            if params == [0]:
                result = genesis_hash
            elif params == [STATEFUL_LAST_EPOCH_BLOCK]:
                result = CUTOVER_HASH
            else:
                assert params == [STATEFUL_LAST_EPOCH_BLOCK - 1]
                result = PREDECESSOR_HASH
        elif method == "state_getStorage":
            if params[0] == timestamp_key:
                result = "0x" + TIMESTAMP_MS.to_bytes(8, "little").hex()
            else:
                name = storage_names[params[0]]
                value = storage[name]
                if (
                    params[1] == PREDECESSOR_HASH
                    and name == "SubnetEpochIndex"
                ):
                    value = int(storage["SubnetEpochIndex"]) - 1
                if (
                    params[1] == CUTOVER_HASH
                    and name == "BlocksSinceLastStep"
                ):
                    value = 0
                width = 2 if name == "Tempo" else 8
                result = "0x" + int(value).to_bytes(width, "little").hex()
        elif method == "state_call":
            result = _selective_result(finalized_block)
        else:
            raise AssertionError(method)
        attempt = _attempt(
            job_id=kwargs["job_id"],
            purpose=kwargs["purpose"],
            operation=kwargs["logical_operation_id"].removeprefix(
                kwargs["job_id"] + ":"
            ),
            request_id=("%032x" % kwargs["request_id"]),
        )
        return {"result": result, "attempts": [attempt], "artifacts": []}

    return calls, rpc_call


def _attempt(
    *,
    job_id: str,
    purpose: str,
    operation: str,
    request_id: str,
    provider_id: str = "bittensor_chain",
    destination_host: str = CHAIN_ENDPOINT_HOST,
):
    body = json.dumps({"operation": operation}, sort_keys=True).encode()
    return build_transport_attempt(
        request_id=request_id,
        logical_operation_id=job_id + ":" + operation,
        job_id=job_id,
        purpose=purpose,
        provider_id=provider_id,
        attempt_number=0,
        method="POST",
        destination_host=destination_host,
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


def _archive_adapter(rpc_call):
    def archive_attempt(attempt):
        return build_transport_attempt(
            request_id=attempt["request_id"],
            logical_operation_id=attempt["logical_operation_id"],
            job_id=attempt["job_id"],
            purpose=attempt["purpose"],
            provider_id="bittensor_archive",
            attempt_number=attempt["attempt_number"],
            method=attempt["method"],
            destination_host=CHAIN_ARCHIVE_ENDPOINT_HOST,
            destination_port=attempt["destination_port"],
            path_hash=attempt["path_hash"],
            nonsecret_headers_hash=attempt["nonsecret_headers_hash"],
            body_hash=attempt["body_hash"],
            credential_ref_hash=attempt["credential_ref_hash"],
            egress_proxy_ref_hash=attempt["egress_proxy_ref_hash"],
            retry_policy_hash=attempt["retry_policy_hash"],
            timeout_ms=attempt["timeout_ms"],
            started_at=attempt["started_at"],
            terminal_status=attempt["terminal_status"],
            http_status=attempt["http_status"],
            response_hash=attempt["response_hash"],
            request_artifact_hash=attempt["request_artifact_hash"],
            response_artifact_hash=attempt["response_artifact_hash"],
            tls_peer_chain_hash=attempt["tls_peer_chain_hash"],
            tls_protocol=attempt["tls_protocol"],
            failure_code=attempt["failure_code"],
            completed_at=attempt["completed_at"],
        )

    def archive_rpc_call(**kwargs):
        result = rpc_call(**kwargs)
        return {
            **result,
            "attempts": [archive_attempt(attempt) for attempt in result["attempts"]],
        }

    return archive_rpc_call


def _runtime_rpc(
    *,
    runtime_block=99,
    finalized_block=100,
    canonical=True,
    runtime_version=None,
):
    calls = []
    requested_hash = "0x" + "cd" * 32
    finalized_hash = "0x" + "ab" * 32
    version = runtime_version or {
        "specVersion": 438,
        "transactionVersion": 1,
    }

    def rpc_call(**kwargs):
        calls.append(kwargs)
        method = kwargs["method"]
        params = kwargs["params"]
        if method == "chain_getFinalizedHead":
            result = finalized_hash
        elif method == "chain_getHeader":
            result = {
                "number": hex(
                    finalized_block
                    if params == [finalized_hash]
                    else runtime_block
                ),
                "stateRoot": "0x" + "12" * 32,
                "parentHash": "0x" + "34" * 32,
                "extrinsicsRoot": "0x" + "56" * 32,
            }
        elif method == "chain_getBlockHash":
            if params == [0]:
                result = GENESIS_HASH
            else:
                assert params == [runtime_block]
                result = (
                    requested_hash if canonical else "0x" + "ef" * 32
                )
        elif method == "state_getRuntimeVersion":
            assert params == [requested_hash]
            result = version
        else:
            raise AssertionError(method)
        attempt = _attempt(
            job_id=kwargs["job_id"],
            purpose=kwargs["purpose"],
            operation=kwargs["logical_operation_id"].split(":")[-1],
            request_id="%032x" % kwargs["request_id"],
        )
        return {"result": result, "attempts": [attempt], "artifacts": []}

    return calls, requested_hash, rpc_call


def test_chain_signing_runtime_is_exact_canonical_and_finalized():
    calls, requested_hash, rpc_call = _runtime_rpc()
    result = ValidatorChainSourceV2(
        rpc_call=rpc_call,
        epoch_authority_supplier=lambda: None,
    ).read_chain_signing_runtime(
        runtime_block_hash=requested_hash,
        max_block_drift=8,
    )

    assert result["runtime_block"] == 99
    assert result["finalized_block"] == 100
    assert result["spec_version"] == 438
    assert result["transaction_version"] == 1
    assert result["genesis_hash"] == GENESIS_HASH[2:]
    assert [item["method"] for item in calls] == [
        "chain_getFinalizedHead",
        "chain_getHeader",
        "chain_getHeader",
        "chain_getBlockHash",
        "state_getRuntimeVersion",
        "chain_getBlockHash",
    ]
    assert {item["purpose"] for item in calls} == {"validator.chain_state.v2"}
    assert calls[4]["params"] == [requested_hash]


@pytest.mark.parametrize(
    ("runtime_block", "finalized_block", "canonical", "match"),
    [
        (99, 100, False, "not canonical"),
        (90, 100, True, "not within the finalized signing window"),
        (101, 100, True, "not within the finalized signing window"),
    ],
)
def test_chain_signing_runtime_fails_closed_on_invalid_exact_block(
    runtime_block, finalized_block, canonical, match
):
    _calls, requested_hash, rpc_call = _runtime_rpc(
        runtime_block=runtime_block,
        finalized_block=finalized_block,
        canonical=canonical,
    )
    with pytest.raises(ValidatorChainSourceV2Error, match=match):
        ValidatorChainSourceV2(
            rpc_call=rpc_call,
            epoch_authority_supplier=lambda: None,
        ).read_chain_signing_runtime(
            runtime_block_hash=requested_hash,
            max_block_drift=8,
        )


@pytest.mark.parametrize(
    "runtime_version",
    [
        {"specVersion": True, "transactionVersion": 1},
        {"specVersion": 438, "transactionVersion": "1"},
        {"specVersion": 438},
    ],
)
def test_chain_signing_runtime_rejects_malformed_runtime_version(
    runtime_version,
):
    _calls, requested_hash, rpc_call = _runtime_rpc(
        runtime_version=runtime_version
    )
    with pytest.raises(ValidatorChainSourceV2Error):
        ValidatorChainSourceV2(
            rpc_call=rpc_call,
            epoch_authority_supplier=lambda: None,
        ).read_chain_signing_runtime(
            runtime_block_hash=requested_hash,
            max_block_drift=8,
        )


def test_finalized_source_uses_one_block_for_header_and_metagraph():
    calls, rpc_call = _stateful_rpc()
    result = ValidatorChainSourceV2(
        rpc_call=rpc_call,
        archive_rpc_call=_archive_adapter(rpc_call),
        epoch_authority_supplier=lambda: {
            "mode": "stateful_v1",
            "cutover_manifest": _stateful_cutover(),
        },
    ).read_finalized_snapshot(
        netuid=71, epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID
    )
    assert result["header"]["block"] == STATEFUL_BLOCK
    assert result["metagraph"]["block"] == STATEFUL_BLOCK
    assert len(result["metagraph"]["hotkeys"]) == 2
    metagraph_call = next(item for item in calls if item["method"] == "state_call")
    assert metagraph_call["params"][2] == FINALIZED_HASH


def test_stateful_storage_keys_and_fixed_width_decoding_match_sn71_vector():
    assert subnet_epoch_storage_key(storage_name="Tempo", netuid=71) == (
        "0x658faa385070e074c85bf6b568cf0555"
        "7641384bb339f3758acddfd7053d33174700"
    )
    assert subnet_epoch_storage_key(
        storage_name="SubnetEpochIndex", netuid=71
    ) == (
        "0x658faa385070e074c85bf6b568cf0555"
        "4f101d7a30ae31c7ab3099206c5ae12b4700"
    )
    assert decode_subnet_epoch_storage(
        "0x6801", storage_name="Tempo"
    ) == 360
    assert decode_subnet_epoch_storage(
        "0x" + STATEFUL_SUBNET_EPOCH_INDEX.to_bytes(8, "little").hex(),
        storage_name="SubnetEpochIndex",
    ) == STATEFUL_SUBNET_EPOCH_INDEX
    assert timestamp_now_storage_key() == (
        "0xf0c365c3cf59d671eb72da0e7a4113c4"
        "9f1f0515f462cdcf84e0f1d6045dfcbb"
    )
    assert decode_timestamp_now_storage(
        "0x" + TIMESTAMP_MS.to_bytes(8, "little").hex()
    ) == TIMESTAMP_MS


def test_stateful_finalized_source_binds_official_epoch_and_chain_anchors():
    calls, rpc_call = _stateful_rpc()
    cutover = _stateful_cutover()
    result = ValidatorChainSourceV2(
        rpc_call=rpc_call,
        archive_rpc_call=_archive_adapter(rpc_call),
        epoch_authority_supplier=lambda: {
            "mode": "stateful_v1",
            "cutover_manifest": cutover,
        },
    ).read_finalized_snapshot(
        netuid=71,
        epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
    )

    authority = result["epoch_authority"]
    assert authority["subnet_epoch_index"] == STATEFUL_SUBNET_EPOCH_INDEX
    assert authority["settlement_epoch_id"] == STATEFUL_SETTLEMENT_EPOCH_ID
    assert authority["epoch_block"] == 4
    assert authority["blocks_remaining"] == 356
    assert authority["last_epoch_block"] == STATEFUL_LAST_EPOCH_BLOCK
    assert authority["block_hash"] == FINALIZED_HASH
    assert authority["cutover_mapping_hash"] == cutover["mapping_hash"]
    anchor_calls = [
        item for item in calls if item["method"] == "chain_getBlockHash"
    ]
    assert [item["params"] for item in anchor_calls] == [
        [0],
        [STATEFUL_LAST_EPOCH_BLOCK],
        [STATEFUL_LAST_EPOCH_BLOCK - 1],
    ]
    storage_calls = [item for item in calls if item["method"] == "state_getStorage"]
    assert {item["params"][1] for item in storage_calls} == {
        FINALIZED_HASH,
        CUTOVER_HASH,
        PREDECESSOR_HASH,
    }
    assert authority["observed_at"] == "2025-07-17T00:00:00Z"
    assert result["epoch_boundary"]["observed_at"] == authority["observed_at"]
    archive_attempts = [
        attempt
        for attempt in result["attempts"]
        if attempt["provider_id"] == "bittensor_archive"
    ]
    live_attempts = [
        attempt
        for attempt in result["attempts"]
        if attempt["provider_id"] == "bittensor_chain"
    ]
    assert archive_attempts
    assert live_attempts
    assert {
        attempt["destination_host"] for attempt in archive_attempts
    } == {CHAIN_ARCHIVE_ENDPOINT_HOST}
    assert {
        attempt["destination_host"] for attempt in live_attempts
    } == {CHAIN_ENDPOINT_HOST}
    assert any(
        attempt["logical_operation_id"].endswith(":cutover-subnet-epoch-index")
        for attempt in archive_attempts
    )
    assert any(
        ":epoch-storage-subnetepochindex" in attempt["logical_operation_id"]
        for attempt in live_attempts
    )
    assert any(
        attempt["purpose"] == "validator.metagraph_state.v2"
        for attempt in live_attempts
    )
    assert not any(
        attempt["purpose"] == "validator.metagraph_state.v2"
        for attempt in archive_attempts
    )


def test_stateful_source_fails_closed_without_a_separate_archive_adapter():
    calls, rpc_call = _stateful_rpc()
    with pytest.raises(
        ValidatorChainSourceV2Error,
        match="archive RPC adapter is required",
    ):
        ValidatorChainSourceV2(
            rpc_call=rpc_call,
            epoch_authority_supplier=lambda: {
                "mode": "stateful_v1",
                "cutover_manifest": _stateful_cutover(),
            },
        ).read_finalized_snapshot(
            netuid=71,
            epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
        )
    assert [item["method"] for item in calls] == [
        "chain_getFinalizedHead",
        "chain_getHeader",
    ]


def test_stateful_source_rejects_live_endpoint_evidence_for_archive_reads():
    _calls, rpc_call = _stateful_rpc()
    with pytest.raises(
        ValidatorChainSourceV2Error,
        match="wrong measured endpoint",
    ):
        ValidatorChainSourceV2(
            rpc_call=rpc_call,
            archive_rpc_call=rpc_call,
            epoch_authority_supplier=lambda: {
                "mode": "stateful_v1",
                "cutover_manifest": _stateful_cutover(),
            },
        ).read_finalized_snapshot(
            netuid=71,
            epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
        )


def test_explicit_boundary_capture_proves_historical_cutover():
    calls, rpc_call = _stateful_rpc()
    cutover = _stateful_cutover()
    source = ValidatorChainSourceV2(
        rpc_call=rpc_call,
        archive_rpc_call=_archive_adapter(rpc_call),
        epoch_authority_supplier=lambda: {
            "mode": "stateful_v1",
            "cutover_manifest": cutover,
        },
    )
    result = source.capture_stateful_epoch_boundary(
        cutover_manifest=cutover,
        settlement_epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
        capture_scope="sha256:" + "9" * 64,
    )

    assert result["epoch_boundary"]["current_block"] == STATEFUL_LAST_EPOCH_BLOCK
    assert result["epoch_boundary"]["block_hash"] == CUTOVER_HASH
    assert result["epoch_authority"] == result["epoch_boundary"]
    assert result["epoch_authority"]["settlement_epoch_id"] == (
        STATEFUL_SETTLEMENT_EPOCH_ID
    )
    assert result["finalized_block_hash"] == FINALIZED_HASH.removeprefix("0x")
    assert result["header"]["block"] == STATEFUL_BLOCK
    assert result["jobs"]["subnet_epoch_snapshot"].startswith(
        "subnet-epoch-capture-current:"
    )
    assert result["jobs"]["subnet_epoch_snapshot"].endswith("9" * 64)
    assert all(
        attempt["logical_operation_id"].startswith(
            result["jobs"]["subnet_epoch_snapshot"] + ":"
        )
        for attempt in result["attempts"]
    )
    assert result["jobs"]["subnet_epoch_boundary"] == result["jobs"][
        "subnet_epoch_snapshot"
    ]
    assert {
        attempt["job_id"] for attempt in result["attempts"]
    } == set(result["jobs"].values())
    assert all(
        attempt["purpose"] == "validator.subnet_epoch_snapshot.v2"
        for attempt in result["attempts"]
    )
    assert any(
        item["method"] == "state_getStorage"
        and item["params"][1] == PREDECESSOR_HASH
        for item in calls
    )


def test_explicit_boundary_capture_rejects_manifest_block_hash_mismatch():
    _calls, rpc_call = _stateful_rpc(
        finalized_hash=CUTOVER_HASH,
        finalized_block=STATEFUL_LAST_EPOCH_BLOCK,
    )
    cutover = _stateful_cutover(cutover_block_hash="0x" + "99" * 32)
    with pytest.raises(ValidatorChainSourceV2Error, match="not canonical"):
        ValidatorChainSourceV2(
            rpc_call=rpc_call,
            archive_rpc_call=_archive_adapter(rpc_call),
            epoch_authority_supplier=lambda: {
                "mode": "stateful_v1",
                "cutover_manifest": cutover,
            },
        ).capture_stateful_epoch_boundary(
            cutover_manifest=cutover,
            settlement_epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
            capture_scope="sha256:" + "9" * 64,
        )


def test_explicit_boundary_capture_rejects_an_unfinalized_boundary():
    _calls, rpc_call = _stateful_rpc(
        finalized_hash=PREDECESSOR_HASH,
        finalized_block=STATEFUL_LAST_EPOCH_BLOCK - 1,
    )
    with pytest.raises(ValidatorChainSourceV2Error, match="not finalized"):
        ValidatorChainSourceV2(
            rpc_call=rpc_call,
            archive_rpc_call=_archive_adapter(rpc_call),
            epoch_authority_supplier=lambda: {
                "mode": "stateful_v1",
                "cutover_manifest": _stateful_cutover(),
            },
        ).capture_stateful_epoch_boundary(
            cutover_manifest=_stateful_cutover(),
            settlement_epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
            capture_scope="sha256:" + "9" * 64,
        )


def test_stateful_source_honors_pending_due_boundary_without_rollover():
    pending_block = STATEFUL_BLOCK + 10
    _calls, rpc_call = _stateful_rpc(
        storage_updates={"PendingEpochAt": pending_block}
    )
    cutover = _stateful_cutover()
    result = ValidatorChainSourceV2(
        rpc_call=rpc_call,
        archive_rpc_call=_archive_adapter(rpc_call),
        epoch_authority_supplier=lambda: {
            "mode": "stateful_v1",
            "cutover_manifest": cutover,
        },
    ).read_finalized_snapshot(
        netuid=71,
        epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
    )
    authority = result["epoch_authority"]
    assert authority["next_epoch_block"] == pending_block
    assert authority["blocks_remaining"] == 10
    assert authority["subnet_epoch_index"] == STATEFUL_SUBNET_EPOCH_INDEX


@pytest.mark.parametrize(
    ("blocks_since_last_step", "blocks_until_safety"),
    ((50_399, 2), (50_400, 1), (50_401, 0)),
)
def test_stateful_source_uses_post_block_safety_deadline(
    blocks_since_last_step,
    blocks_until_safety,
):
    _calls, rpc_call = _stateful_rpc(
        storage_updates={"BlocksSinceLastStep": blocks_since_last_step}
    )
    result = ValidatorChainSourceV2(
        rpc_call=rpc_call,
        archive_rpc_call=_archive_adapter(rpc_call),
        epoch_authority_supplier=lambda: {
            "mode": "stateful_v1",
            "cutover_manifest": _stateful_cutover(),
        },
    ).read_finalized_snapshot(
        netuid=71,
        epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
    )

    authority = result["epoch_authority"]
    assert authority["next_epoch_block"] == (
        STATEFUL_BLOCK + blocks_until_safety
    )
    assert authority["blocks_remaining"] == blocks_until_safety


@pytest.mark.parametrize(
    ("rpc_options", "manifest_updates", "match"),
    (
        (
            {"genesis_hash": "0x" + "99" * 32},
            {},
            "genesis differs",
        ),
        (
            {"storage_updates": {"SubnetEpochIndex": 23_926}},
            {
                "first_subnet_epoch_index": 23_926,
                "first_settlement_epoch_id": 23_991,
                "last_legacy_epoch_id": 23_990,
            },
            "requested settlement epoch",
        ),
    ),
)
def test_stateful_source_rejects_tampered_anchor_or_finalized_previous_index(
    rpc_options,
    manifest_updates,
    match,
):
    _calls, rpc_call = _stateful_rpc(**rpc_options)
    cutover = _stateful_cutover(**manifest_updates)
    with pytest.raises(ValidatorChainSourceV2Error, match=match):
        ValidatorChainSourceV2(
            rpc_call=rpc_call,
            archive_rpc_call=_archive_adapter(rpc_call),
            epoch_authority_supplier=lambda: {
                "mode": "stateful_v1",
                "cutover_manifest": cutover,
            },
        ).read_finalized_snapshot(
            netuid=71,
            epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
        )


def test_stateful_source_rejects_tampered_mapping_before_chain_requests():
    calls, rpc_call = _stateful_rpc()
    cutover = {
        **_stateful_cutover(),
        "mapping_hash": "sha256:" + "00" * 32,
    }
    with pytest.raises(ValidatorChainSourceV2Error, match="mapping"):
        ValidatorChainSourceV2(
            rpc_call=rpc_call,
            archive_rpc_call=_archive_adapter(rpc_call),
            epoch_authority_supplier=lambda: {
                "mode": "stateful_v1",
                "cutover_manifest": cutover,
            },
        ).read_finalized_snapshot(
            netuid=71,
            epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
        )
    assert [item["method"] for item in calls] == [
        "chain_getFinalizedHead",
        "chain_getHeader",
    ]


def test_stateful_boundary_search_uses_index_transition_not_reset_anchor():
    cutover_block = 100
    boundary_block = 120
    reset_anchor = 150
    finalized_block = 160
    first_index = 10
    current_index = 11
    first_settlement = 200
    genesis = "0x" + "41" * 32

    def block_hash(block):
        return "0x" + ("%064x" % int(block))

    body = {
        "schema_version": "leadpoet.subnet_epoch_cutover.v1",
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": genesis,
        "netuid": 71,
        "cutover_block": cutover_block,
        "cutover_block_hash": block_hash(cutover_block),
        "first_subnet_epoch_index": first_index,
        "first_settlement_epoch_id": first_settlement,
        "last_legacy_epoch_id": first_settlement - 1,
    }
    cutover = {**body, "mapping_hash": sha256_json(body)}
    scheduler_keys = {
        subnet_epoch_storage_key(storage_name=name, netuid=71): name
        for name in (
            "Tempo",
            "LastEpochBlock",
            "PendingEpochAt",
            "SubnetEpochIndex",
            "BlocksSinceLastStep",
        )
    }
    calls = []

    def rpc_call(**kwargs):
        calls.append(kwargs)
        method = kwargs["method"]
        params = kwargs["params"]
        if method == "chain_getFinalizedHead":
            result = block_hash(finalized_block)
        elif method == "chain_getBlockHash":
            result = genesis if params == [0] else block_hash(params[0])
        elif method == "chain_getHeader":
            block = int(params[0], 16)
            result = {
                "number": hex(block),
                "stateRoot": "0x" + "12" * 32,
                "parentHash": "0x" + "34" * 32,
                "extrinsicsRoot": "0x" + "56" * 32,
            }
        elif method == "state_getStorage":
            block = int(params[1], 16)
            if params[0] == timestamp_now_storage_key():
                value = TIMESTAMP_MS + block
            else:
                name = scheduler_keys[params[0]]
                index = (
                    first_index - 1
                    if block < cutover_block
                    else first_index
                    if block < boundary_block
                    else current_index
                )
                if name == "SubnetEpochIndex":
                    value = index
                elif name == "Tempo":
                    value = 360
                elif name == "LastEpochBlock":
                    value = (
                        boundary_block
                        if block == boundary_block
                        else reset_anchor
                    )
                elif name == "PendingEpochAt":
                    value = 0
                else:
                    value = 0 if block == boundary_block else 40
            width = 2 if params[0] == next(
                key for key, name in scheduler_keys.items() if name == "Tempo"
            ) else 8
            result = "0x" + int(value).to_bytes(width, "little").hex()
        elif method == "state_call":
            result = _selective_result(finalized_block)
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
        rpc_call=rpc_call,
        archive_rpc_call=_archive_adapter(rpc_call),
        epoch_authority_supplier=lambda: {
            "mode": "stateful_v1",
            "cutover_manifest": cutover,
        },
    ).read_finalized_snapshot(netuid=71, epoch_id=first_settlement + 1)

    assert result["epoch_authority"]["last_epoch_block"] == reset_anchor
    assert result["epoch_boundary"]["current_block"] == boundary_block
    assert result["epoch_boundary"]["last_epoch_block"] == boundary_block
    assert result["epoch_boundary"]["subnet_epoch_index"] == current_index
    selector_job = "subnet-epoch-selector:%d" % (first_settlement + 1)
    assert result["jobs"]["subnet_epoch_snapshot"] == selector_job
    epoch_jobs = {
        result["jobs"]["subnet_epoch_snapshot"],
        result["jobs"]["subnet_epoch_boundary"],
    }
    epoch_attempts = [
        attempt
        for attempt in result["attempts"]
        if attempt["purpose"] == "validator.subnet_epoch_snapshot.v2"
    ]
    assert epoch_attempts
    assert {attempt["job_id"] for attempt in epoch_attempts} <= epoch_jobs
    probed_blocks = {
        int(item["params"][1], 16)
        for item in calls
        if item["method"] == "state_getStorage"
        and item["params"][0]
        == subnet_epoch_storage_key(
            storage_name="SubnetEpochIndex",
            netuid=71,
        )
    }
    assert boundary_block in probed_blocks
    assert boundary_block - 1 in probed_blocks


def test_stateful_boundary_search_rejects_a_skipped_epoch_index_transition():
    cutover_block = 100
    boundary_block = 120
    finalized_block = 130
    first_index = 10
    current_index = 12
    first_settlement = 200
    genesis = "0x" + "41" * 32

    def block_hash(block):
        return "0x" + ("%064x" % int(block))

    body = {
        "schema_version": "leadpoet.subnet_epoch_cutover.v1",
        "epoch_scheme": "bittensor.subnet_epoch_index.v1",
        "network_genesis_hash": genesis,
        "netuid": 71,
        "cutover_block": cutover_block,
        "cutover_block_hash": block_hash(cutover_block),
        "first_subnet_epoch_index": first_index,
        "first_settlement_epoch_id": first_settlement,
        "last_legacy_epoch_id": first_settlement - 1,
    }
    cutover = {**body, "mapping_hash": sha256_json(body)}
    scheduler_keys = {
        subnet_epoch_storage_key(storage_name=name, netuid=71): name
        for name in (
            "Tempo",
            "LastEpochBlock",
            "PendingEpochAt",
            "SubnetEpochIndex",
            "BlocksSinceLastStep",
        )
    }

    def rpc_call(**kwargs):
        method = kwargs["method"]
        params = kwargs["params"]
        if method == "chain_getFinalizedHead":
            result = block_hash(finalized_block)
        elif method == "chain_getBlockHash":
            result = genesis if params == [0] else block_hash(params[0])
        elif method == "chain_getHeader":
            block = int(params[0], 16)
            result = {
                "number": hex(block),
                "stateRoot": "0x" + "12" * 32,
                "parentHash": "0x" + "34" * 32,
                "extrinsicsRoot": "0x" + "56" * 32,
            }
        elif method == "state_getStorage":
            block = int(params[1], 16)
            if params[0] == timestamp_now_storage_key():
                value = TIMESTAMP_MS + block
                width = 8
            else:
                name = scheduler_keys[params[0]]
                index = (
                    first_index - 1
                    if block < cutover_block
                    else first_index
                    if block < boundary_block
                    else current_index
                )
                value = {
                    "Tempo": 360,
                    "LastEpochBlock": boundary_block,
                    "PendingEpochAt": 0,
                    "SubnetEpochIndex": index,
                    "BlocksSinceLastStep": block - boundary_block,
                }[name]
                width = 2 if name == "Tempo" else 8
            result = "0x" + int(value).to_bytes(width, "little").hex()
        elif method == "state_call":
            result = _selective_result(finalized_block)
        else:
            raise AssertionError(method)
        attempt = _attempt(
            job_id=kwargs["job_id"],
            purpose=kwargs["purpose"],
            operation=method + ":" + str(kwargs["request_id"]),
            request_id=("%032x" % kwargs["request_id"]),
        )
        return {"result": result, "attempts": [attempt], "artifacts": []}

    with pytest.raises(
        ValidatorChainSourceV2Error,
        match="boundary transition is invalid",
    ):
        ValidatorChainSourceV2(
            rpc_call=rpc_call,
            archive_rpc_call=_archive_adapter(rpc_call),
            epoch_authority_supplier=lambda: {
                "mode": "stateful_v1",
                "cutover_manifest": cutover,
            },
        ).read_finalized_snapshot(
            netuid=71,
            epoch_id=first_settlement + (current_index - first_index),
        )


def test_finalized_source_rejects_cross_epoch_or_metagraph_block_mismatch():
    _calls, rpc_call = _stateful_rpc()

    with pytest.raises(
        ValidatorChainSourceV2Error,
        match="requested settlement epoch",
    ):
        ValidatorChainSourceV2(
            rpc_call=rpc_call,
            archive_rpc_call=_archive_adapter(rpc_call),
            epoch_authority_supplier=lambda: {
                "mode": "stateful_v1",
                "cutover_manifest": _stateful_cutover(),
            },
        ).read_finalized_snapshot(
            netuid=71, epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID + 1
        )


def test_finalized_source_selects_just_finished_official_epoch():
    current_block = STATEFUL_BLOCK + 360
    current_last_epoch_block = STATEFUL_LAST_EPOCH_BLOCK + 360
    target_block = current_last_epoch_block - 1
    current_hash = "0x" + "cc" * 32
    target_hash = "0x" + "dd" * 32
    calls = []
    storage_names = {
        subnet_epoch_storage_key(storage_name=name, netuid=71): name
        for name in ("SubnetEpochIndex", "LastEpochBlock")
    }

    def rpc_call(**kwargs):
        calls.append(kwargs)
        method = kwargs["method"]
        params = kwargs["params"]
        if method == "chain_getFinalizedHead":
            result = current_hash
        elif method == "chain_getHeader":
            block = target_block if params == [target_hash] else current_block
            result = {
                "number": hex(block),
                "stateRoot": "0x" + "12" * 32,
                "parentHash": "0x" + "34" * 32,
                "extrinsicsRoot": "0x" + "56" * 32,
            }
        elif method == "chain_getBlockHash":
            assert params == [target_block]
            result = target_hash
        elif method == "state_getStorage":
            name = storage_names[params[0]]
            value = {
                "SubnetEpochIndex": STATEFUL_SUBNET_EPOCH_INDEX + 1,
                "LastEpochBlock": current_last_epoch_block,
            }[name]
            result = "0x" + int(value).to_bytes(8, "little").hex()
        elif method == "state_call":
            assert params[-1] == target_hash
            result = _selective_result(target_block)
        else:
            raise AssertionError(method)
        attempt = _attempt(
            job_id=kwargs["job_id"],
            purpose=kwargs["purpose"],
            operation=kwargs["logical_operation_id"].removeprefix(
                kwargs["job_id"] + ":"
            ),
            request_id=("%032x" % kwargs["request_id"]),
        )
        return {"result": result, "attempts": [attempt], "artifacts": []}

    source = ValidatorChainSourceV2(
        rpc_call=rpc_call,
        archive_rpc_call=_archive_adapter(rpc_call),
        epoch_authority_supplier=lambda: {
            "mode": "stateful_v1",
            "cutover_manifest": _stateful_cutover(),
        },
    )
    observed = {}

    def read_stateful(**kwargs):
        observed.update(kwargs)
        return {
            "authority": {"settlement_epoch_id": STATEFUL_SETTLEMENT_EPOCH_ID},
            "boundary_snapshot": {"settlement_epoch_id": STATEFUL_SETTLEMENT_EPOCH_ID},
            "snapshot_job": "snapshot",
            "boundary_job": "boundary",
            "attempts": [],
            "artifacts": [],
            "next_request_id": 20,
        }

    source._read_stateful_epoch_authority = read_stateful
    result = source.read_finalized_snapshot(
        netuid=71,
        epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
    )

    assert result["header"]["block"] == target_block
    assert result["metagraph"]["block"] == target_block
    assert observed["finalized_hash"] == target_hash.removeprefix("0x")
    assert observed["historical_snapshot"] is True
    assert observed["snapshot_job_override"] == (
        "subnet-epoch-selector:%d" % STATEFUL_SETTLEMENT_EPOCH_ID
    )
    selector_attempts = [
        attempt
        for attempt in result["attempts"]
        if attempt["purpose"] == "validator.subnet_epoch_snapshot.v2"
    ]
    assert selector_attempts
    assert {
        attempt["job_id"] for attempt in selector_attempts
    } == {observed["snapshot_job_override"]}
    metagraph_attempt = result["attempts"][-1]
    assert metagraph_attempt["provider_id"] == "bittensor_archive"
    assert metagraph_attempt["destination_host"] == CHAIN_ARCHIVE_ENDPOINT_HOST


def test_finalized_source_rejects_more_than_one_epoch_of_drift():
    calls, rpc_call = _stateful_rpc(
        storage_updates={
            "SubnetEpochIndex": STATEFUL_SUBNET_EPOCH_INDEX + 2,
            "LastEpochBlock": STATEFUL_LAST_EPOCH_BLOCK + 720,
        },
        finalized_block=STATEFUL_BLOCK + 720,
    )
    with pytest.raises(
        ValidatorChainSourceV2Error,
        match="not current or just finalized",
    ):
        ValidatorChainSourceV2(
            rpc_call=rpc_call,
            archive_rpc_call=_archive_adapter(rpc_call),
            epoch_authority_supplier=lambda: {
                "mode": "stateful_v1",
                "cutover_manifest": _stateful_cutover(),
            },
        ).read_finalized_snapshot(
            netuid=71,
            epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
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


def test_archive_transport_records_only_the_exact_archive_destination():
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    transport = EnclaveChainRpcTransportV2(
        destination_host=CHAIN_ARCHIVE_ENDPOINT_HOST,
        clock=lambda: now,
        sleep=lambda _seconds: None,
    )
    body = b'{"jsonrpc":"2.0","id":1,"result":"0x' + b"ab" * 32 + b'"}'
    transport._http_post = lambda _body: {
        "status": 200,
        "body": body,
        "tls_peer_chain_hash": sha256_json([sha256_bytes(b"cert")]),
        "tls_protocol": "TLSv1.3",
    }
    result = transport.call(
        method="chain_getBlockHash",
        params=[1],
        request_id=1,
        job_id="subnet-epoch-boundary:1",
        purpose="validator.subnet_epoch_snapshot.v2",
        logical_operation_id="subnet-epoch-boundary:1:block-hash",
    )
    assert {
        (attempt["provider_id"], attempt["destination_host"])
        for attempt in result["attempts"]
    } == {("bittensor_archive", CHAIN_ARCHIVE_ENDPOINT_HOST)}
    with pytest.raises(ValidatorChainSourceV2Error, match="outside measured policy"):
        EnclaveChainRpcTransportV2(destination_host="archive.attacker.example")


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
            assert kwargs["params"][0] == timelocked_weight_commits_storage_key(
                netuid=71,
                subnet_epoch_index=STATEFUL_SUBNET_EPOCH_INDEX,
            )
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

    pacing = []
    result = ValidatorChainSourceV2(
        rpc_call=rpc_call,
        archive_rpc_call=_archive_adapter(rpc_call),
        finalization_sleep=pacing.append,
        epoch_authority_supplier=lambda: {
            "mode": "stateful_v1",
            "cutover_manifest": _stateful_cutover(),
        },
    ).find_finalized_extrinsic_inclusion(
        expected_extrinsics={extrinsic_hash: extrinsic.hex()},
        expected_commitments={
            extrinsic_hash: {
                "netuid": 71,
                "subnet_epoch_index": STATEFUL_SUBNET_EPOCH_INDEX,
                "hotkey_public_key": OWNER.hex(),
                "commitment_hex": commitment.hex(),
                "reveal_round": reveal_round,
            }
        },
        minimum_block=BLOCK - 1,
        maximum_block=BLOCK,
        epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
        finalization_scan_id="sha256:" + "9" * 64,
    )
    assert result["extrinsic_hash"] == extrinsic_hash
    assert result["finalized_block"] == BLOCK
    assert result["state_transition_hash"].startswith("sha256:")
    assert result["job_id"] == (
        "weight-finalization:%d:%s"
        % (STATEFUL_SETTLEMENT_EPOCH_ID, "9" * 64)
    )
    providers = [attempt["provider_id"] for attempt in result["attempts"]]
    assert providers[-1] == "bittensor_archive"
    assert set(providers[:-1]) == {"bittensor_chain"}
    assert pacing == [FINALIZATION_RPC_PACING_SECONDS] * 4


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
            rpc_call=rpc_call,
            finalization_sleep=lambda _seconds: None,
            epoch_authority_supplier=lambda: {
                "mode": "stateful_v1",
                "cutover_manifest": _stateful_cutover(),
            },
        ).find_finalized_extrinsic_inclusion(
            expected_extrinsics={extrinsic_hash: extrinsic.hex()},
            expected_commitments={
                extrinsic_hash: {
                    "netuid": 71,
                    "subnet_epoch_index": STATEFUL_SUBNET_EPOCH_INDEX,
                    "hotkey_public_key": OWNER.hex(),
                    "commitment_hex": "aa",
                    "reveal_round": 1,
                }
            },
            minimum_block=BLOCK,
            maximum_block=BLOCK,
            epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID,
            finalization_scan_id="sha256:" + "9" * 64,
        )
