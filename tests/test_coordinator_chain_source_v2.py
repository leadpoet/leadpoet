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
