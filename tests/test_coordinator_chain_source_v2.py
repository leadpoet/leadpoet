from __future__ import annotations

import base64
from datetime import datetime, timezone
import json

from gateway.tee.coordinator_chain_source_v2 import CoordinatorChainSourceV2
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    sha256_bytes,
    sha256_json,
)
from leadpoet_canonical.chain_source_v2 import ss58_encode_account_id


HASH = "sha256:" + "a" * 64
OWNER = bytes.fromhex("924620afb270acb1ee27bd034aa9e97108ef276da5079db982883cd70294741a")
MINER = bytes.fromhex("74adb27b7edd7126a81f5bac79e9bda1a4c8ec94d2c4f2ce795e0c56932a5383")


def _selective_fixture(block):
    encoded = bytearray((1, 0x1D, 0x01))
    encoded.extend(b"\x00" * 4)
    encoded.extend(b"\x01" + OWNER)
    encoded.extend(b"\x00")
    encoded.extend(b"\x01" + ((int(block) << 2) | 2).to_bytes(4, "little"))
    encoded.extend(b"\x00" * 44)
    encoded.extend(b"\x01\x08" + OWNER + MINER)
    encoded.extend(b"\x00" * 21)
    return "0x" + bytes(encoded).hex()


class FakeBroker:
    def __init__(self, block):
        self.block = int(block)
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
            elif rpc["method"] == "chain_getHeader":
                result = {
                    "number": hex(self.block),
                    "stateRoot": "0x" + "c" * 64,
                    "parentHash": "0x" + "d" * 64,
                    "extrinsicsRoot": "0x" + "e" * 64,
                    "digest": {"logs": []},
                }
            elif rpc["params"][0] == "SubnetInfoRuntimeApi_get_selective_mechagraph":
                result = _selective_fixture(self.block)
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
                else "entrypoint-finney.opentensor.ai"
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
    broker = FakeBroker(100 * 360 + 10)
    source = CoordinatorChainSourceV2(
        execute_provider=broker.execute,
        retry_policy_hashes={
            "bittensor_chain": "sha256:" + "1" * 64,
            "coingecko": "sha256:" + "2" * 64,
        },
        sleep=lambda _seconds: None,
        clock=lambda: datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc),
    )
    context = ExecutionContextV2(
        job_id="allocation-v2:test",
        purpose="research_lab.allocation.v2",
        epoch_id=100,
    )

    result = source.resolve_live_prices(netuid=71, context=context)

    assert result["header"]["block"] == 100 * 360 + 10
    assert len(result["metagraph"]["hotkeys"]) == 2
    assert result["tao_per_alpha"] == 0.005181338
    assert result["tao_usd"] == 201.25
    assert len(context.transport_attempts) == 5
    assert len(broker.calls) == 5
    assert {call["provider_id"] for call in broker.calls} == {
        "bittensor_chain",
        "coingecko",
    }


class HistoricalBroker:
    def __init__(self, *, epoch=100, fail_first=False):
        self.epoch = int(epoch)
        self.target = (self.epoch + 1) * 360 - 1
        self.fail_first = bool(fail_first)
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
                value = _selective_fixture(self.target)
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
