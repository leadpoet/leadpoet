from __future__ import annotations

import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from Leadpoet.utils.subnet_epoch import SubnetEpochCutover
from gateway.tee import coordinator_chain_source_v2 as chain_source_module
from gateway.tee.coordinator_chain_source_v2 import (
    CoordinatorChainSourceV2,
    CoordinatorChainSourceV2Error,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    sha256_bytes,
    sha256_json,
)
from leadpoet_canonical.chain_source_v2 import (
    last_update_storage_key,
    reveal_period_epochs_storage_key,
    ss58_encode_account_id,
    subnet_epoch_storage_key,
    system_event_count_storage_key,
    system_events_storage_key,
    timelocked_weight_commits_storage_key,
    weights_storage_key,
)


HASH = "sha256:" + "a" * 64
OWNER = bytes.fromhex("924620afb270acb1ee27bd034aa9e97108ef276da5079db982883cd70294741a")
MINER = bytes.fromhex("74adb27b7edd7126a81f5bac79e9bda1a4c8ec94d2c4f2ce795e0c56932a5383")


def _selective_fixture(block, *, last_field=76):
    encoded = bytearray((1, 0x1D, 0x01))
    encoded.extend(b"\x00" * 4)
    encoded.extend(b"\x01" + OWNER)
    encoded.extend(b"\x00")
    encoded.extend(b"\x01" + ((int(block) << 2) | 2).to_bytes(4, "little"))
    encoded.extend(b"\x00" * 44)
    encoded.extend(b"\x01\x08" + OWNER + MINER)
    encoded.extend(b"\x00" * (int(last_field) - 52))
    return "0x" + bytes(encoded).hex()


class FakeBroker:
    def __init__(self, block, *, cutover=None, predecessor_index=None):
        self.block = int(block)
        self.cutover = cutover
        self.predecessor_index = predecessor_index
        self.calls = []

    def execute(self, request):
        self.calls.append(dict(request))
        request_body = base64.b64decode(request["body_b64"])
        if request["provider_id"] == "coingecko":
            response_body = json.dumps(
                {"bittensor": {"usd": 201.25}},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        else:
            rpc = json.loads(request_body)
            if rpc["method"] == "chain_getFinalizedHead":
                result = "0x" + "b" * 64
            elif rpc["method"] == "chain_getBlockHash":
                requested_block = int(rpc["params"][0])
                if requested_block == 0 and self.cutover is not None:
                    result = self.cutover.network_genesis_hash
                elif (
                    self.cutover is not None
                    and requested_block == self.cutover.cutover_block
                ):
                    result = self.cutover.cutover_block_hash
                elif (
                    self.cutover is not None
                    and requested_block == self.cutover.cutover_block - 1
                ):
                    result = "0x" + "9" * 64
                else:
                    raise AssertionError("unexpected block-hash request")
            elif rpc["method"] == "chain_getHeader":
                at_hash = str(rpc["params"][0])
                is_cutover = (
                    self.cutover is not None
                    and at_hash == self.cutover.cutover_block_hash
                )
                result = {
                    "number": hex(
                        self.cutover.cutover_block if is_cutover else self.block
                    ),
                    "stateRoot": "0x" + "c" * 64,
                    "parentHash": (
                        "0x" + "9" * 64 if is_cutover else "0x" + "d" * 64
                    ),
                    "extrinsicsRoot": "0x" + "e" * 64,
                    "digest": {"logs": []},
                }
            elif rpc["method"] == "state_call":
                result = (
                    _selective_fixture(self.block)
                    if rpc["params"][0]
                    == "SubnetInfoRuntimeApi_get_selective_mechagraph"
                    else "0x9a0f4f0000000000"
                )
            elif rpc["method"] == "state_getStorage":
                if self.cutover is None:
                    raise AssertionError("unexpected stateful storage request")
                storage_key, at_hash = rpc["params"]
                storage_name = next(
                    name
                    for name in (
                        "Tempo",
                        "LastEpochBlock",
                        "PendingEpochAt",
                        "SubnetEpochIndex",
                        "BlocksSinceLastStep",
                    )
                    if storage_key
                    == subnet_epoch_storage_key(storage_name=name, netuid=71)
                )
                if at_hash == self.cutover.cutover_block_hash:
                    values = {
                        "SubnetEpochIndex": self.cutover.first_subnet_epoch_index,
                        "LastEpochBlock": self.cutover.cutover_block,
                    }
                elif at_hash == "0x" + "9" * 64:
                    values = {
                        "SubnetEpochIndex": (
                            self.cutover.first_subnet_epoch_index - 1
                            if self.predecessor_index is None
                            else self.predecessor_index
                        )
                    }
                else:
                    values = {
                        "Tempo": 360,
                        "LastEpochBlock": self.cutover.cutover_block,
                        "PendingEpochAt": 0,
                        "SubnetEpochIndex": self.cutover.first_subnet_epoch_index,
                        "BlocksSinceLastStep": (
                            self.block - self.cutover.cutover_block
                        ),
                    }
                width = 2 if storage_name == "Tempo" else 8
                result = "0x" + int(values[storage_name]).to_bytes(
                    width, "little"
                ).hex()
            else:
                result = "0x9a0f4f0000000000"
            response_body = json.dumps(
                {"jsonrpc": "2.0", "id": rpc["id"], "result": result},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        attempt = build_transport_attempt(
            request_id=("%032x" % len(self.calls)),
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id=request["provider_id"],
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host=(
                "api.coingecko.com"
                if request["provider_id"] == "coingecko"
                else (
                    "archive.chain.opentensor.ai"
                    if request["provider_id"] == "bittensor_archive"
                    else "entrypoint-finney.opentensor.ai"
                )
            ),
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH,
            body_hash=sha256_bytes(request_body),
            credential_ref_hash=HASH,
            retry_policy_hash=request["retry_policy_hash"],
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T20:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=sha256_bytes(response_body),
            request_artifact_hash=sha256_json(
                {"request": len(self.calls), "provider": request["provider_id"]}
            ),
            response_artifact_hash=sha256_bytes(response_body),
            tls_peer_chain_hash=HASH,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T20:00:00Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "body_b64": base64.b64encode(response_body).decode(),
            "transport_attempt": attempt,
        }


def test_finalized_metagraph_and_prices_are_bound_to_terminal_records():
    cutover = _stateful_cutover()
    broker = FakeBroker(1_310, cutover=cutover)
    source = CoordinatorChainSourceV2(
        execute_provider=broker.execute,
        retry_policy_hashes={
            "bittensor_chain": "sha256:" + "1" * 64,
            "bittensor_archive": "sha256:" + "2" * 64,
            "coingecko": "sha256:" + "3" * 64,
        },
        epoch_authority={
            "mode": "stateful_v1",
            "cutover": cutover.to_dict(),
        },
        sleep=lambda _seconds: None,
        clock=lambda: datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc),
    )
    context = ExecutionContextV2(
        job_id="allocation-v2:test",
        purpose="research_lab.allocation.v2",
        epoch_id=101,
    )

    result = source.resolve_live_prices(netuid=71, context=context)

    assert result["header"]["block"] == 1_310
    assert result["workflow_epoch_id"] == 101
    assert result["official_subnet_epoch_id"] == 10
    assert len(result["metagraph"]["hotkeys"]) == 2
    assert result["tao_per_alpha"] == 0.005181338
    assert result["tao_usd"] == 201.25
    assert len(context.transport_attempts) == 17
    assert len(broker.calls) == 17
    assert {call["provider_id"] for call in broker.calls} == {
        "bittensor_archive",
        "bittensor_chain",
        "coingecko",
    }


def _stateful_cutover():
    return SubnetEpochCutover(
        network_genesis_hash=(
            "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44"
            "b090c175ffde2a7e5ab36c03"
        ),
        netuid=71,
        cutover_block=1_000,
        cutover_block_hash="0x" + "2" * 64,
        first_subnet_epoch_index=10,
        first_settlement_epoch_id=101,
        last_legacy_epoch_id=100,
    )


def _stateful_source(broker, cutover):
    chain_signing_profile = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ).read_text(encoding="utf-8")
    )
    return CoordinatorChainSourceV2(
        execute_provider=broker.execute,
        retry_policy_hashes={
            "bittensor_chain": "sha256:" + "1" * 64,
            "bittensor_archive": "sha256:" + "2" * 64,
            "coingecko": "sha256:" + "3" * 64,
        },
        epoch_authority={
            "mode": "stateful_v1",
            "cutover": cutover.to_dict(),
            "chain_signing_profile": chain_signing_profile,
        },
        sleep=lambda _seconds: None,
    )


def test_stateful_finalized_metagraph_uses_exact_scheduler_and_cutover_proof():
    cutover = _stateful_cutover()
    broker = FakeBroker(1_310, cutover=cutover)
    source = _stateful_source(broker, cutover)
    context = ExecutionContextV2(
        job_id="allocation-v2:stateful",
        purpose="research_lab.allocation.v2",
        epoch_id=101,
    )

    result = source.read_finalized_metagraph(netuid=71, context=context)

    assert result["workflow_epoch_id"] == 101
    assert result["official_subnet_epoch_id"] == 10
    assert result["epoch_authority"] == {
        "mode": "stateful_v1",
        "workflow_epoch_id": 101,
        "official_subnet_epoch_id": 10,
        "cutover_mapping_hash": cutover.mapping_hash,
        "state": {
            "Tempo": 360,
            "LastEpochBlock": 1_000,
            "PendingEpochAt": 0,
            "SubnetEpochIndex": 10,
            "BlocksSinceLastStep": 310,
        },
    }
    assert len(broker.calls) == 15
    assert len(context.transport_attempts) == 15
    historical_calls = [
        call
        for call in broker.calls
        if call["provider_id"] == "bittensor_archive"
    ]
    live_calls = [
        call
        for call in broker.calls
        if call["provider_id"] == "bittensor_chain"
    ]
    assert len(historical_calls) == 7
    assert len(live_calls) == 8
    for call in historical_calls:
        rpc = json.loads(base64.b64decode(call["body_b64"]))
        if rpc["method"] == "state_getStorage":
            assert rpc["params"][1] in {
                cutover.cutover_block_hash,
                "0x" + "9" * 64,
            }
    for call in live_calls:
        rpc = json.loads(base64.b64decode(call["body_b64"]))
        assert not (
            rpc["method"] == "state_getStorage"
            and rpc["params"][1]
            in {cutover.cutover_block_hash, "0x" + "9" * 64}
        )
    assert sum(
        json.loads(base64.b64decode(call["body_b64"]))["method"]
        == "state_getStorage"
        for call in broker.calls
    ) == 8


def test_stateful_coordinator_rejects_skipped_cutover_index():
    cutover = _stateful_cutover()
    broker = FakeBroker(
        1_310,
        cutover=cutover,
        predecessor_index=cutover.first_subnet_epoch_index - 2,
    )
    source = _stateful_source(broker, cutover)
    context = ExecutionContextV2(
        job_id="allocation-v2:skipped-index",
        purpose="research_lab.allocation.v2",
        epoch_id=101,
    )

    with pytest.raises(
        CoordinatorChainSourceV2Error,
        match="not an official transition",
    ):
        source.read_finalized_metagraph(netuid=71, context=context)


@pytest.mark.parametrize(
    ("reveal_period_override", "runtime_spec_version", "expected_error"),
    (
        (1, 452, None),
        (None, 452, None),
        (1, 440, "metadata is invalid"),
        (1, 436, "not explicitly supported"),
        (2, 452, "reveal period differs"),
    ),
)
def test_stateful_epoch_close_is_live_finalized_and_exact_archive_state(
    monkeypatch,
    reveal_period_override,
    runtime_spec_version,
    expected_error,
):
    cutover = _stateful_cutover()
    source = _stateful_source(FakeBroker(1_800, cutover=cutover), cutover)
    calls = []
    if runtime_spec_version == 452:
        monkeypatch.setattr(
            chain_source_module,
            "resolve_reveal_period_metadata_default_v2",
            lambda **_kwargs: 1,
        )

    def block_hash(block):
        return "%064x" % int(block)

    def header(block):
        return {
            "number": hex(int(block)),
            "stateRoot": "0x" + ("%064x" % (int(block) + 10_000)),
            "parentHash": "0x" + block_hash(int(block) - 1),
            "extrinsicsRoot": "0x" + "e" * 64,
            "digest": {"logs": []},
        }

    def live_call(**kwargs):
        calls.append(("live", kwargs["method"], tuple(kwargs["params"])))
        if kwargs["method"] == "chain_getFinalizedHead":
            return "0x" + block_hash(1_800)
        if kwargs["method"] == "chain_getHeader":
            return header(1_800)
        raise AssertionError(kwargs)

    def archive_call(**kwargs):
        method = kwargs["method"]
        params = tuple(kwargs["params"])
        calls.append(("archive", method, params))
        if method == "chain_getBlockHash":
            return "0x" + block_hash(int(params[0]))
        if method == "chain_getHeader":
            return header(int(str(params[0]), 16))
        if method == "state_call":
            return _selective_fixture(1_719)
        if method == "state_getRuntimeVersion":
            return {
                "specVersion": runtime_spec_version,
                "transactionVersion": 1,
            }
        if method == "state_getMetadata":
            return "0x6d6574610e"
        if method == "state_getStorage":
            storage_key, at_hash = params
            block = int(str(at_hash), 16)
            if storage_key == subnet_epoch_storage_key(
                storage_name="SubnetEpochIndex",
                netuid=71,
            ):
                epoch_index = (
                    9
                    if block < 1_000
                    else 10
                    if block < 1_360
                    else 11
                    if block < 1_720
                    else 12
                )
                return "0x" + epoch_index.to_bytes(8, "little").hex()
            if storage_key == subnet_epoch_storage_key(
                storage_name="LastEpochBlock",
                netuid=71,
            ):
                # A set_tempo call can reset this value without changing the
                # official SubnetEpochIndex. It is not an epoch identity.
                return "0x" + (1_600).to_bytes(8, "little").hex()
            if storage_key == weights_storage_key(
                netuid=71,
                validator_uid=0,
            ):
                return "0x" + (
                    b"\x08"
                    + (0).to_bytes(2, "little")
                    + (65_535).to_bytes(2, "little")
                    + (1).to_bytes(2, "little")
                    + (16_384).to_bytes(2, "little")
                ).hex()
            if storage_key == last_update_storage_key(netuid=71):
                return "0x" + (
                    b"\x08"
                    + (1_345).to_bytes(8, "little")
                    + (1_200).to_bytes(8, "little")
                ).hex()
            if storage_key == reveal_period_epochs_storage_key(netuid=71):
                return (
                    None
                    if reveal_period_override is None
                    else "0x"
                    + int(reveal_period_override).to_bytes(8, "little").hex()
                )
        raise AssertionError(kwargs)

    monkeypatch.setattr(source, "_chain_call", live_call)
    monkeypatch.setattr(source, "_archive_call", archive_call)
    context = ExecutionContextV2(
        job_id="chain-realized:102",
        purpose="research_lab.chain_weight_observation.v1",
        epoch_id=102,
    )

    if expected_error is not None:
        with pytest.raises(
            CoordinatorChainSourceV2Error,
            match=expected_error,
        ):
            source.read_stateful_epoch_close_weights(
                netuid=71,
                epoch_id=102,
                validator_hotkey=ss58_encode_account_id(OWNER),
                context=context,
            )
        return

    result = source.read_stateful_epoch_close_weights(
        netuid=71,
        epoch_id=102,
        validator_hotkey=ss58_encode_account_id(OWNER),
        context=context,
    )

    assert result["close_block"] == 1_719
    assert result["next_epoch_block"] == 1_720
    assert result["official_subnet_epoch_id"] == 11
    assert result["last_update_block"] == 1_345
    assert result["latest_commit_source_epoch_id"] == 101
    assert result["epoch_start_block"] == 1_360
    assert result["epoch_start_block_hash"] == block_hash(1_360)
    assert result["reveal_window_start_block"] == 1_360
    assert result["reveal_window_start_block_hash"] == block_hash(1_360)
    assert result["scheduled_reveal_subnet_epoch_id"] == 10
    assert result["scheduled_reveal_source_epoch_id"] == 101
    assert result["subnet_reveal_period_epochs"] == 1
    assert result["reveal_period_storage_key"] == (
        reveal_period_epochs_storage_key(netuid=71)
    )
    assert result["reveal_period_storage_override"] == reveal_period_override
    assert result["reveal_period_metadata_hash"] == sha256_bytes(b"meta\x0e")
    assert result["reveal_period_runtime_spec_version"] == runtime_spec_version
    assert result["chain_signing_profile"]["network"] == "finney"
    assert result["chain_signing_profile_hash"] == sha256_json(
        result["chain_signing_profile"]
    )
    assert result["weights"] == [[0, 65_535], [1, 16_384]]
    assert [item[1] for item in calls if item[0] == "live"] == [
        "chain_getFinalizedHead",
        "chain_getHeader",
    ]
    assert not any(
        source_kind == "archive" and method == "chain_getFinalizedHead"
        for source_kind, method, _params in calls
    )
    assert not any(
        method == "state_getStorage"
        and params[0]
        == subnet_epoch_storage_key(
            storage_name="LastEpochBlock",
            netuid=71,
        )
        for _source_kind, method, params in calls
    )


class HistoricalBroker:
    def __init__(self, *, epoch=100, fail_first=False, last_field=73):
        self.epoch = int(epoch)
        self.target = (self.epoch + 1) * 360 - 1
        self.fail_first = bool(fail_first)
        self.last_field = int(last_field)
        self.calls = []

    def execute(self, request):
        self.calls.append(dict(request))
        body = base64.b64decode(request["body_b64"])
        rpc = json.loads(body)
        if self.fail_first and len(self.calls) == 1:
            response = b'{"error":"busy"}'
            http_status = 503
        else:
            if rpc["method"] == "chain_getFinalizedHead":
                value = "0x" + "a" * 64
            elif rpc["method"] == "chain_getBlockHash":
                value = "0x" + "b" * 64
            elif rpc["method"] == "chain_getHeader":
                is_target = rpc["params"][0] == "0x" + "b" * 64
                value = {
                    "number": hex(self.target if is_target else self.target + 20),
                    "stateRoot": "0x" + "c" * 64,
                    "parentHash": "0x" + "d" * 64,
                    "extrinsicsRoot": "0x" + "e" * 64,
                    "digest": {"logs": []},
                }
            elif rpc["method"] == "state_call":
                value = _selective_fixture(
                    self.target,
                    last_field=self.last_field,
                )
            else:
                value = "0x" + (
                    b"\x08"
                    + (1).to_bytes(2, "little")
                    + (1000).to_bytes(2, "little")
                    + (4).to_bytes(2, "little")
                    + (2000).to_bytes(2, "little")
                ).hex()
            response = json.dumps(
                {"jsonrpc": "2.0", "id": rpc["id"], "result": value},
                separators=(",", ":"),
            ).encode()
            http_status = 200
        attempt = build_transport_attempt(
            request_id=("%032x" % len(self.calls)),
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id="bittensor_archive",
            attempt_number=request["attempt_number"],
            method="POST",
            destination_host="archive.chain.opentensor.ai",
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH,
            body_hash=sha256_bytes(body),
            credential_ref_hash=HASH,
            retry_policy_hash=request["retry_policy_hash"],
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T20:00:00Z",
            terminal_status="authenticated_response",
            http_status=http_status,
            response_hash=sha256_bytes(response),
            request_artifact_hash=sha256_json(
                {"archive_request": len(self.calls)}
            ),
            response_artifact_hash=sha256_bytes(response),
            tls_peer_chain_hash=HASH,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T20:00:00Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": http_status,
            "body_b64": base64.b64encode(response).decode(),
            "transport_attempt": attempt,
        }


def test_historical_weights_use_archive_epoch_end_and_exact_validator_uid():
    broker = HistoricalBroker()
    sleeps = []
    source = CoordinatorChainSourceV2(
        execute_provider=broker.execute,
        retry_policy_hashes={
            "bittensor_chain": "sha256:" + "1" * 64,
            "bittensor_archive": "sha256:" + "2" * 64,
            "coingecko": "sha256:" + "3" * 64,
        },
        epoch_authority={
            "mode": "stateful_v1",
            "cutover": _stateful_cutover().to_dict(),
        },
        sleep=sleeps.append,
    )
    context = ExecutionContextV2(
        job_id="legacy-settlement:test",
        purpose="research_lab.legacy_finalized_allocation.v2",
        epoch_id=101,
    )
    result = source.read_historical_finalized_weights(
        netuid=71,
        epoch_id=100,
        validator_hotkey=ss58_encode_account_id(OWNER),
        context=context,
    )
    assert result["target_block"] == 101 * 360 - 1
    assert result["validator_uid"] == 0
    assert result["weights"] == [[1, 1000], [4, 2000]]
    assert len(broker.calls) == 6
    assert all(call["provider_id"] == "bittensor_archive" for call in broker.calls)
    assert sleeps == []


def test_historical_archive_retries_are_recorded_and_bounded():
    broker = HistoricalBroker(fail_first=True)
    sleeps = []
    source = CoordinatorChainSourceV2(
        execute_provider=broker.execute,
        retry_policy_hashes={
            "bittensor_chain": "sha256:" + "1" * 64,
            "bittensor_archive": "sha256:" + "2" * 64,
            "coingecko": "sha256:" + "3" * 64,
        },
        epoch_authority={
            "mode": "stateful_v1",
            "cutover": _stateful_cutover().to_dict(),
        },
        sleep=sleeps.append,
    )
    context = ExecutionContextV2(
        job_id="legacy-settlement:retry",
        purpose="research_lab.legacy_finalized_allocation.v2",
        epoch_id=101,
    )
    result = source.read_historical_finalized_weights(
        netuid=71,
        epoch_id=100,
        validator_hotkey=ss58_encode_account_id(OWNER),
        context=context,
    )
    assert result["epoch_id"] == 100
    assert sleeps == [1.0]
    assert len(context.transport_attempts) == 7


REVEAL_EVENT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests/fixtures/subtensor_events_spec452_block8984916.json"
)


def _scale_compact(value):
    normalized = int(value)
    if normalized < 1 << 6:
        return bytes((normalized << 2,))
    if normalized < 1 << 14:
        return ((normalized << 2) | 1).to_bytes(2, "little")
    if normalized < 1 << 30:
        return ((normalized << 2) | 2).to_bytes(4, "little")
    raise AssertionError("test SCALE compact value is too large")


def _encode_timelocked_commits(entries):
    encoded = bytearray(_scale_compact(len(entries)))
    for entry in entries:
        commitment = bytes.fromhex(entry["commitment_hex"])
        encoded.extend(bytes.fromhex(entry["hotkey_public_key"]))
        encoded.extend(int(entry["submitted_at"]).to_bytes(8, "little"))
        encoded.extend(_scale_compact(len(commitment)))
        encoded.extend(commitment)
        encoded.extend(int(entry["reveal_round"]).to_bytes(8, "little"))
    return "0x" + bytes(encoded).hex()


def _encode_weights(entries):
    encoded = bytearray(_scale_compact(len(entries)))
    for uid, weight in entries:
        encoded.extend(int(uid).to_bytes(2, "little"))
        encoded.extend(int(weight).to_bytes(2, "little"))
    return "0x" + bytes(encoded).hex()


def _selective_reveal_fixture(*, block, validator_uid, validator_account):
    hotkeys = [bytes((index,)) * 32 for index in range(24)]
    hotkeys[int(validator_uid)] = bytes(validator_account)
    encoded = bytearray((1,))
    encoded.extend(_scale_compact(71))
    encoded.extend(b"\x00" * 4)
    encoded.extend(b"\x01" + OWNER)
    encoded.extend(b"\x00")
    encoded.extend(b"\x01" + _scale_compact(block))
    encoded.extend(b"\x00" * 44)
    encoded.extend(b"\x01" + _scale_compact(len(hotkeys)))
    encoded.extend(b"".join(hotkeys))
    encoded.extend(b"\x00" * 24)
    return "0x" + bytes(encoded).hex()


class RevealProofArchive:
    def __init__(
        self,
        *,
        removal_block=None,
        event_mode="exact",
        multiple_pre_entries=False,
        runtime_spec_version=452,
    ):
        self.fixture = json.loads(
            REVEAL_EVENT_FIXTURE_PATH.read_text(encoding="utf-8")
        )
        self.profile = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "leadpoet_canonical/subtensor_events_profile_v2.json"
            ).read_text(encoding="utf-8")
        )
        self.reveal_block = int(self.fixture["block_number"])
        self.removal_block = int(
            self.reveal_block if removal_block is None else removal_block
        )
        self.event_mode = str(event_mode)
        self.multiple_pre_entries = bool(multiple_pre_entries)
        self.runtime_spec_version = int(runtime_spec_version)
        self.public_key = str(self.fixture["expected"]["account_id_hex"])
        self.validator_uid = int(self.fixture["expected"]["uid"])
        self.commitment = b"production-shaped-reveal-proof"
        self.reveal_round = 8_984_916
        self.exact_entry = {
            "hotkey_public_key": self.public_key,
            "submitted_at": self.reveal_block - 200,
            "commitment_hex": self.commitment.hex(),
            "reveal_round": self.reveal_round,
        }
        self.other_entry = {
            "hotkey_public_key": self.public_key,
            "submitted_at": self.reveal_block - 199,
            "commitment_hex": b"other-queued-commitment".hex(),
            "reveal_round": self.reveal_round + 1,
        }
        self.weights = [[3, 12_345], [8, 54_321]]
        self.metadata_raw = gzip.decompress(
            (
                Path(__file__).resolve().parents[1]
                / "tests/restart_rehearsal/fixtures/subtensor_metadata_spec452_parent8984915.scale.gz"
            ).read_bytes()
        )
        self.calls = []
        self._block_hashes = {}
        self._hash_blocks = {}

    def block_hash(self, block):
        normalized = int(block)
        observed = self._block_hashes.get(normalized)
        if observed is None:
            if normalized == self.reveal_block:
                observed = str(self.fixture["block_hash"])[2:]
            elif normalized == self.reveal_block - 1:
                observed = str(self.profile["measurement"]["parent_hash"])[2:]
            else:
                observed = hashlib.sha256(
                    ("reveal-proof-block:%d" % normalized).encode("ascii")
                ).hexdigest()
            self._block_hashes[normalized] = observed
            self._hash_blocks[observed] = normalized
        return observed

    def _block_for_hash(self, value):
        normalized = str(value)
        if normalized.startswith("0x"):
            normalized = normalized[2:]
        if normalized not in self._hash_blocks:
            raise AssertionError("unmeasured test block hash")
        return self._hash_blocks[normalized]

    def _header(self, block):
        normalized = int(block)
        return {
            "number": hex(normalized),
            "stateRoot": "0x"
            + hashlib.sha256(
                ("reveal-proof-state:%d" % normalized).encode("ascii")
            ).hexdigest(),
            "parentHash": "0x" + self.block_hash(normalized - 1),
            "extrinsicsRoot": "0x"
            + hashlib.sha256(
                ("reveal-proof-extrinsics:%d" % normalized).encode("ascii")
            ).hexdigest(),
            "digest": {"logs": []},
        }

    def _events(self):
        events = bytes.fromhex(str(self.fixture["system_events"])[2:])
        event_count = str(self.fixture["system_event_count"])
        if self.event_mode == "exact":
            return "0x" + events.hex(), event_count
        if self.event_mode == "wrong_identity":
            account = bytes.fromhex(self.public_key)
            assert events.count(account) == 1
            return "0x" + events.replace(account, b"\x11" * 32, 1).hex(), event_count
        if self.event_mode == "empty":
            return "0x00", "0x00000000"
        raise AssertionError("unknown test event mode")

    def call(self, *, method, params, **_kwargs):
        self.calls.append((str(method), tuple(params)))
        if method == "chain_getBlockHash":
            return "0x" + self.block_hash(int(params[0]))
        if method == "chain_getHeader":
            return self._header(self._block_for_hash(params[0]))
        if method == "state_getRuntimeVersion":
            return {
                "specVersion": self.runtime_spec_version,
                "transactionVersion": 1,
            }
        if method == "state_getMetadata":
            return "0x" + self.metadata_raw.hex()
        if method == "state_getStorageHash":
            assert params[0] == chain_source_module.RUNTIME_CODE_STORAGE_KEY
            return self.profile["runtime_code_storage_hash"]
        if method == "state_call":
            block = self._block_for_hash(params[2])
            assert block == self.removal_block
            return _selective_reveal_fixture(
                block=block,
                validator_uid=self.validator_uid,
                validator_account=bytes.fromhex(self.public_key),
            )
        if method != "state_getStorage":
            raise AssertionError("unexpected archive method %s" % method)
        storage_key, at_hash = params
        block = self._block_for_hash(at_hash)
        commit_key = timelocked_weight_commits_storage_key(
            netuid=71,
            subnet_epoch_index=24_945,
        )
        if storage_key == commit_key:
            entries = [self.exact_entry] if block < self.removal_block else []
            if self.multiple_pre_entries and block == self.removal_block - 1:
                entries.append(self.other_entry)
            return _encode_timelocked_commits(entries)
        if storage_key == reveal_period_epochs_storage_key(netuid=71):
            return "0x" + (1).to_bytes(8, "little").hex()
        if storage_key == weights_storage_key(
            netuid=71,
            validator_uid=self.validator_uid,
        ):
            assert block == self.removal_block
            return _encode_weights(self.weights)
        events, event_count = self._events()
        if storage_key == system_events_storage_key():
            assert block == self.removal_block
            return events
        if storage_key == system_event_count_storage_key():
            assert block == self.removal_block
            return event_count
        raise AssertionError("unexpected archive storage key")


def _reveal_proof_case(
    monkeypatch,
    *,
    final_block,
    window_start_block,
    close_block,
    removal_block=None,
    event_mode="exact",
    multiple_pre_entries=False,
    runtime_spec_version=452,
):
    archive = RevealProofArchive(
        removal_block=removal_block,
        event_mode=event_mode,
        multiple_pre_entries=multiple_pre_entries,
        runtime_spec_version=runtime_spec_version,
    )
    source = _stateful_source(
        FakeBroker(close_block, cutover=_stateful_cutover()),
        _stateful_cutover(),
    )
    monkeypatch.setattr(source, "_archive_call", archive.call)
    validator_hotkey = ss58_encode_account_id(bytes.fromhex(archive.public_key))
    chain_state = {
        "netuid": 71,
        "official_subnet_epoch_id": 24_946,
        "scheduled_reveal_subnet_epoch_id": 24_945,
        "scheduled_reveal_source_epoch_id": 25_036,
        "subnet_reveal_period_epochs": 1,
        "validator_hotkey": validator_hotkey,
        "validator_uid": archive.validator_uid,
        "weights": archive.weights,
        "reveal_window_start_block": int(window_start_block),
        "reveal_window_start_block_hash": archive.block_hash(window_start_block),
        "close_block": int(close_block),
        "close_block_hash": archive.block_hash(close_block),
    }
    authority = {
        "bundle_hash": "sha256:" + "b" * 64,
        "netuid": 71,
        "epoch_id": 25_036,
        "subnet_epoch_index": 24_945,
        "validator_hotkey": validator_hotkey,
        "uids": [item[0] for item in archive.weights],
        "weights_u16": [item[1] for item in archive.weights],
        "finalized_block": int(final_block),
        "finalized_block_hash": archive.block_hash(final_block),
        "state_transition_hash": sha256_json(archive.exact_entry),
        "extrinsic_authorization": {
            "netuid": 71,
            "epoch_id": 25_036,
            "subnet_epoch_index": 24_945,
            "validator_hotkey": validator_hotkey,
            "hotkey_public_key": archive.public_key,
            "commitment_hex": archive.commitment.hex(),
            "commitment_hash": sha256_bytes(archive.commitment),
            "reveal_round": archive.reveal_round,
        },
    }
    context = ExecutionContextV2(
        job_id="chain-realized:25037",
        purpose="research_lab.chain_weight_observation.v1",
        epoch_id=25_037,
    )
    return source, archive, chain_state, authority, context


def test_timelocked_reveal_proof_uses_real_spec452_event_pair(monkeypatch):
    event_block = 8_984_916
    source, archive, chain_state, authority, context = _reveal_proof_case(
        monkeypatch,
        final_block=event_block - 24,
        window_start_block=event_block - 60,
        close_block=event_block + 8,
    )

    proof = source.read_timelocked_reveal_proof(
        chain_state=chain_state,
        authority=authority,
        context=context,
    )

    assert proof is not None
    assert proof["reveal_block"] == event_block
    assert proof["reveal_block_hash"] == str(archive.fixture["block_hash"])[2:]
    assert proof["event_witness"]["event_count"] == 196
    assert proof["event_witness"]["weights_set_record_index"] == 1
    assert proof["event_witness"]["reveal_record_index"] == 2
    assert proof["event_witness"]["account_id_hex"] == archive.public_key
    assert proof["revealed_weights"] == archive.weights
    assert proof["proof_hash"] == sha256_json(
        {key: value for key, value in proof.items() if key != "proof_hash"}
    )


def test_timelocked_reveal_proof_accepts_normal_span_over_96_blocks(monkeypatch):
    event_block = 8_984_916
    source, _archive, chain_state, authority, context = _reveal_proof_case(
        monkeypatch,
        final_block=event_block - 220,
        window_start_block=event_block - 360,
        close_block=event_block + 12,
    )
    assert (
        chain_state["close_block"]
        - max(
            authority["finalized_block"],
            chain_state["reveal_window_start_block"],
        )
        > 96
    )

    proof = source.read_timelocked_reveal_proof(
        chain_state=chain_state,
        authority=authority,
        context=context,
    )

    assert proof is not None
    assert proof["reveal_block"] == event_block


def test_timelocked_reveal_proof_rejects_missing_exact_event(monkeypatch):
    event_block = 8_984_916
    source, _archive, chain_state, authority, context = _reveal_proof_case(
        monkeypatch,
        final_block=event_block - 24,
        window_start_block=event_block - 60,
        close_block=event_block + 8,
        event_mode="wrong_identity",
    )

    assert source.read_timelocked_reveal_proof(
        chain_state=chain_state,
        authority=authority,
        context=context,
    ) is None


def test_timelocked_reveal_proof_rejects_tuple_removal_without_event(monkeypatch):
    event_block = 8_984_916
    source, _archive, chain_state, authority, context = _reveal_proof_case(
        monkeypatch,
        final_block=event_block - 24,
        window_start_block=event_block - 60,
        close_block=event_block + 8,
        event_mode="empty",
    )

    assert source.read_timelocked_reveal_proof(
        chain_state=chain_state,
        authority=authority,
        context=context,
    ) is None


def test_timelocked_reveal_proof_rejects_multiple_pre_reveal_entries(monkeypatch):
    event_block = 8_984_916
    source, _archive, chain_state, authority, context = _reveal_proof_case(
        monkeypatch,
        final_block=event_block - 24,
        window_start_block=event_block - 60,
        close_block=event_block + 8,
        multiple_pre_entries=True,
    )

    assert source.read_timelocked_reveal_proof(
        chain_state=chain_state,
        authority=authority,
        context=context,
    ) is None


def test_timelocked_reveal_proof_rejects_removal_before_window(monkeypatch):
    event_block = 8_984_916
    removal_block = event_block - 40
    source, _archive, chain_state, authority, context = _reveal_proof_case(
        monkeypatch,
        final_block=event_block - 120,
        window_start_block=event_block - 20,
        close_block=event_block + 8,
        removal_block=removal_block,
    )

    assert source.read_timelocked_reveal_proof(
        chain_state=chain_state,
        authority=authority,
        context=context,
    ) is None


def test_timelocked_reveal_proof_rejects_unknown_runtime(monkeypatch):
    event_block = 8_984_916
    source, _archive, chain_state, authority, context = _reveal_proof_case(
        monkeypatch,
        final_block=event_block - 24,
        window_start_block=event_block - 60,
        close_block=event_block + 8,
        runtime_spec_version=451,
    )

    assert source.read_timelocked_reveal_proof(
        chain_state=chain_state,
        authority=authority,
        context=context,
    ) is None
