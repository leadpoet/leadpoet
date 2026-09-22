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
        {"billing": {"cost_usd": 0}, "results": []},
        {"billing": "unknown", "results": []},
        [],
        b"not-json",
    ],
)
def test_settlement_header_does_not_compete_with_billing_or_malformed_body(body):
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


def test_non_deepline_response_strips_spoof_without_parsing_body():
    document = response_document(
        body=b"not-json",
        headers={
            "X-LeadPoet-Settled-MicroUSD": "999",
            "X-LeadPoet-Call-Identity": "sha256:" + "f" * 64,
        },
    )

    headers = runner.WorkerSocketServer._socket_response_headers(
        document, "openrouter.responses", 0
    )

    assert headers == {"content-type": "application/json"}


@pytest.mark.parametrize("billing", [None, {}])
def test_settlement_header_allows_null_or_empty_billing_mapping(billing):
    document = response_document(body={"billing": billing, "results": []})
    document["call"]["actual_microusd"] = 12_003

    headers = runner.WorkerSocketServer._socket_response_headers(
        document, OPERATION, 0
    )

    assert headers[runner.SETTLED_MICROUSD_HEADER] == "12003"
    assert headers[runner.CALL_IDENTITY_HEADER] == IDENTITY


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        assert chunk
        result.extend(chunk)
    return bytes(result)


def test_real_worker_socket_replaces_spoof_and_preserves_three_key_envelope():
    original_body = contracts.canonical_json(
        {"results": [{"url": "https://example.com"}]}
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
                    "actual_microusd": 0,
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
        runner.SETTLED_MICROUSD_HEADER: "0",
    }
    assert base64.b64decode(response["body_b64"], validate=True) == original_body
    assert api.frames[0]["operation_id"] == OPERATION
    assert api.frames[0]["action_sequence"] == 0


def test_plain_http_bridge_strips_reserved_header_without_adding_proof(tmp_path):
    class Api:
        def provider(self, _run_id, _lease_token, frame):
            return response_document(
                headers={
                    "X-LeadPoet-Settled-MicroUSD": "999",
                    "X-LeadPoet-Call-Identity": "sha256:" + "f" * 64,
                },
                call={"action_sequence": frame["action_sequence"]},
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
    assert runner.SETTLED_MICROUSD_HEADER not in {
        name.lower() for name in headers
    }
    assert runner.CALL_IDENTITY_HEADER not in {
        name.lower() for name in headers
    }
    assert json.loads(body)["results"]
