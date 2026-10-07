"""Trusted settlement proof at the worker socket boundary."""

from __future__ import annotations

import base64
import json
import socket
import tempfile
from pathlib import Path

import pytest

from lab_arena import contracts, runner, shim
from tests.lab_arena.test_lab_arena_runner import lease


OPERATION = "deepline.execute"
PARAMETERS = {"tool": "exa_search", "payload": {"query": "x"}}
IDENTITY = "sha256:" + "a" * 64


def response_document(*, body=None, headers=None, call=None):
    payload = {"results": [{"url": "https://example.com"}]} if body is None else body
    encoded = (
        base64.b64encode(contracts.canonical_json(payload).encode()).decode()
        if not isinstance(payload, bytes)
        else base64.b64encode(payload).decode()
    )
    settled = {
        "operation_id": OPERATION,
        "provider": "deepline",
        "action_sequence": 0,
        "outcome": "settled",
        "actual_microusd": 0,
        "call_identity": IDENTITY,
    }
    if call:
        settled.update(call)
    return {
        "status": 200,
        "headers": {"content-type": "application/json", **(headers or {})},
        "body_b64": encoded,
        "call": settled,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"operation_id": "openrouter.responses"},
        {"provider": "openrouter"},
        {"action_sequence": 1},
        {"action_sequence": True},
        {"outcome": "unknown"},
        {"actual_microusd": -1},
        {"actual_microusd": True},
        {"call_identity": "not-a-hash"},
    ],
)
def test_settlement_header_requires_exact_bound_deepline_call(change):
    document = response_document(
        headers={
            "X-LeadPoet-Settled-MicroUSD": "999",
            "X-LeadPoet-Call-Identity": "sha256:" + "f" * 64,
        },
        call=change,
    )

    headers = runner.WorkerSocketServer._socket_response_headers(
        document, OPERATION, 0
    )

    assert runner.SETTLED_MICROUSD_HEADER not in {
        str(name).lower() for name in headers
    }
    identity_bound = not set(change) & {
        "operation_id", "provider", "action_sequence", "call_identity"
    }
    if identity_bound:
        assert headers == {
            "content-type": "application/json",
            runner.CALL_IDENTITY_HEADER: IDENTITY,
        }
    else:
        assert runner.CALL_IDENTITY_HEADER not in {
            str(name).lower() for name in headers
        }
        assert headers == {"content-type": "application/json"}


@pytest.mark.parametrize(
    "body",
    [
        [],
        b"not-json",
    ],
)
def test_settlement_header_requires_json_mapping_body(body):
    document = response_document(
        body=body, headers={
            runner.SETTLED_MICROUSD_HEADER.upper(): "999",
            runner.CALL_IDENTITY_HEADER.upper(): "sha256:" + "f" * 64,
        }
    )

    headers = runner.WorkerSocketServer._socket_response_headers(
        document, OPERATION, 0
    )

    assert headers == {
        "content-type": "application/json",
        runner.CALL_IDENTITY_HEADER: IDENTITY,
    }


@pytest.mark.parametrize("operation_id", ["openrouter.responses", "exa.search", "exa.contents"])
def test_other_operations_strip_spoof_without_parsing_body(operation_id):
    document = response_document(
        body=b"not-json",
        headers={
            "X-LeadPoet-Settled-MicroUSD": "999",
            "X-LeadPoet-Call-Identity": "sha256:" + "f" * 64,
        },
    )

    headers = runner.WorkerSocketServer._socket_response_headers(
        document, operation_id, 0
    )

    assert headers == {"content-type": "application/json"}


@pytest.mark.parametrize("billing", [
    None,
    {},
    {"pricing_status": "pending", "estimated_cost_usd": 0.002},
    {"pricing_status": "provisional", "cost_usd": 0.002},
    {"credits_charged": 0.02, "cost_usd": 0.002},
    {"pricing_status": "final", "credits_charged": 0, "cost_usd": 0},
    "unknown",
])
@pytest.mark.parametrize("actual_microusd", [0, 2000])
def test_settlement_header_exposes_ledger_charge_independently_of_provider_billing(
    billing, actual_microusd
):
    document = response_document(body={"billing": billing, "results": []})
    document["call"]["actual_microusd"] = actual_microusd
    original_body = document["body_b64"]

    headers = runner.WorkerSocketServer._socket_response_headers(
        document, OPERATION, 0
    )

    assert headers[runner.SETTLED_MICROUSD_HEADER] == str(actual_microusd)
    assert headers[runner.CALL_IDENTITY_HEADER] == IDENTITY
    assert document["body_b64"] == original_body


@pytest.mark.parametrize("outcome", ["uncertain", "reserved", "unknown"])
def test_pending_ledger_never_exposes_provider_estimate_as_confirmed_cost(outcome):
    document = response_document(
        body={"billing": {"pricing_status": "pending", "estimated_cost_usd": 0.002}},
        headers={runner.SETTLED_MICROUSD_HEADER.upper(): "2000"},
        call={"outcome": outcome, "actual_microusd": 0},
    )

    headers = runner.WorkerSocketServer._socket_response_headers(document, OPERATION, 0)

    assert headers == {
        "content-type": "application/json",
        runner.CALL_IDENTITY_HEADER: IDENTITY,
    }


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        assert chunk
        result.extend(chunk)
    return bytes(result)


def test_real_worker_socket_replaces_spoof_and_preserves_three_key_envelope():
    original_body = contracts.canonical_json(
        {
            "results": [{"url": "https://example.com"}],
            "billing": {"pricing_status": "pending", "estimated_cost_usd": 0.002},
        }
    ).encode()

    class Api:
        frames = []

        def provider(self, _run_id, _lease_token, frame):
            self.frames.append(dict(frame))
            return response_document(
                body=original_body,
                headers={
                    "X-LeadPoet-Settled-MicroUSD": "999",
                    "x-LEADPOET-settled-microusd": "998",
                    "X-LeadPoet-Call-Identity": "sha256:" + "f" * 64,
                    "x-LEADPOET-call-identity": "sha256:" + "e" * 64,
                },
                call={
                    "action_sequence": frame["action_sequence"],
                    "actual_microusd": 2000,
                },
            )

    api = Api()
    with tempfile.TemporaryDirectory(prefix="lah-", dir="/tmp") as directory:
        path = Path(directory) / "worker.sock"
        worker = runner.WorkerSocketServer(
            path,
            api,
            runner.RunState(lease=lease("settled-header"), lease_token="token"),
        )
        worker.start()
        try:
            frame = shim.build_operation_frame(OPERATION, PARAMETERS, 5_000)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(path))
                connection.sendall(len(frame).to_bytes(4, "big") + frame)
                size = int.from_bytes(_recv_exact(connection, 4), "big")
                response = json.loads(_recv_exact(connection, size))
        finally:
            worker.stop()

    assert set(response) == {"status", "headers", "body_b64"}
    assert response["headers"] == {
        "content-type": "application/json",
        runner.CALL_IDENTITY_HEADER: IDENTITY,
        runner.SETTLED_MICROUSD_HEADER: "2000",
    }
    assert base64.b64decode(response["body_b64"], validate=True) == original_body
    status, headers, body = shim.parse_worker_response(response)
    assert status == 200
    assert headers[runner.SETTLED_MICROUSD_HEADER] == "2000"
    assert headers[runner.CALL_IDENTITY_HEADER] == IDENTITY
    assert body == original_body
    assert api.frames[0]["operation_id"] == OPERATION
    assert api.frames[0]["action_sequence"] == 0


def test_plain_http_bridge_replaces_upstream_spoof_with_bound_proof(tmp_path):
    original_body = {
        "billing": {"pricing_status": "pending", "estimated_cost_usd": 0.002},
        "results": [{"url": "https://example.com"}],
    }

    class Api:
        def provider(self, _run_id, _lease_token, frame):
            return response_document(
                body=original_body,
                headers={
                    "X-LeadPoet-Settled-MicroUSD": "999",
                    "X-LeadPoet-Call-Identity": "sha256:" + "f" * 64,
                },
                call={"action_sequence": frame["action_sequence"], "actual_microusd": 2000},
            )

    worker = runner.WorkerSocketServer(
        tmp_path / "worker.sock",
        Api(),
        runner.RunState(lease=lease("http-header"), lease_token="token"),
    )
    status, headers, body = worker.handle_http(
        "POST",
        "https://code.deepline.com/api/v2/integrations/exa_search/execute",
        contracts.canonical_json({"payload": {"query": "x"}}).encode(),
        {"content-type": "application/json"},
    )

    assert status == 200
    assert headers[runner.SETTLED_MICROUSD_HEADER] == "2000"
    assert headers[runner.CALL_IDENTITY_HEADER] == IDENTITY
    assert json.loads(body) == original_body


def test_shim_preserves_host_settlement_receipt_and_body():
    document = response_document(headers={
        runner.SETTLED_MICROUSD_HEADER: "3000", runner.CALL_IDENTITY_HEADER: IDENTITY,
        "x-provider-secret": "removed",
    })
    status, headers, body = shim.parse_worker_response({key: document[key] for key in ("status", "headers", "body_b64")})
    assert status == 200 and json.loads(body)["results"]
    assert headers[runner.SETTLED_MICROUSD_HEADER] == "3000"
    assert headers[runner.CALL_IDENTITY_HEADER] == IDENTITY
    assert "x-provider-secret" not in headers


@pytest.mark.parametrize("headers", [
    {runner.SETTLED_MICROUSD_HEADER: "3000"},
    {runner.CALL_IDENTITY_HEADER: "bad-hash"},
    {runner.CALL_IDENTITY_HEADER: IDENTITY, runner.SETTLED_MICROUSD_HEADER: "-1"},
    {runner.CALL_IDENTITY_HEADER: IDENTITY, runner.SETTLED_MICROUSD_HEADER: "0.003"},
    {runner.CALL_IDENTITY_HEADER: IDENTITY, runner.SETTLED_MICROUSD_HEADER: "9223372036854775808"},
    {runner.CALL_IDENTITY_HEADER: IDENTITY, runner.CALL_IDENTITY_HEADER.upper(): IDENTITY},
    {runner.CALL_IDENTITY_HEADER: IDENTITY, runner.SETTLED_MICROUSD_HEADER: "0",
        runner.SETTLED_MICROUSD_HEADER.upper(): "3000"},
])
def test_shim_rejects_malformed_or_unbound_host_proof(headers):
    document = response_document(headers=headers)
    with pytest.raises(shim.ShimTransportError, match="invalid_response"):
        shim.parse_worker_response({key: document[key] for key in ("status", "headers", "body_b64")})
