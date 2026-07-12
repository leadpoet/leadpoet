from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from gateway.tee.sandbox_http_shim_v2 import execute
from gateway.tee.sandbox_provider_socket_v2 import SandboxProviderSocketServerV2
from leadpoet_canonical.attested_v2 import build_transport_attempt


def _hash(character):
    return "sha256:" + character * 64


def test_sandbox_socket_preserves_shared_attempt_scope_and_strips_credentials(
    tmp_path, monkeypatch
):
    requests = []
    terminals = []

    def broker(request):
        requests.append(dict(request))
        attempt = build_transport_attempt(
            request_id=("a" if len(requests) == 1 else "b") * 32,
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id=request["provider_id"],
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host="openrouter.ai",
            destination_port=443,
            path_hash=_hash("1"),
            nonsecret_headers_hash=_hash("2"),
            body_hash=_hash("3"),
            credential_ref_hash=_hash("4"),
            retry_policy_hash=request["retry_policy_hash"],
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T00:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=_hash("5"),
            request_artifact_hash=_hash("6"),
            response_artifact_hash=_hash("5"),
            tls_peer_chain_hash=_hash("7"),
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T00:00:01Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(b'{"ok":true}').decode(),
            "encrypted_request_artifact_id": _hash("6"),
            "encrypted_artifact_id": _hash("5"),
            "transport_attempt": attempt,
        }

    transport = BrokeredProviderTransportV2(broker)
    scope = transport.create_scope(
        job_id="model-job-1",
        purpose="research_lab.private_model_run.v2",
        logical_operation_id="model-job-1",
        retry_policy_hashes={"openrouter": _hash("8")},
        terminal_sink=lambda attempt: terminals.append(dict(attempt)),
    )
    socket_path = Path("/tmp") / (
        "lp-sandbox-%s.sock"
        % hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    )
    server = SandboxProviderSocketServerV2(
        socket_path=socket_path,
        transport=transport,
        execution_scope=scope,
    )
    server.start()
    monkeypatch.setenv("LEADPOET_SANDBOX_PROVIDER_SOCKET", str(socket_path))
    try:
        first = execute(
            method="POST",
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer sandbox-value", "x-title": "Leadpoet"},
            body=b"{}",
            timeout_ms=30000,
        )
        second = execute(
            method="POST",
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer another-value", "x-title": "Leadpoet"},
            body=b"{}",
            timeout_ms=30000,
        )
    finally:
        server.close()
        transport.restore()

    assert base64.b64decode(first["body_b64"]) == b'{"ok":true}'
    assert base64.b64decode(second["body_b64"]) == b'{"ok":true}'
    assert [request["attempt_number"] for request in requests] == [0, 1]
    assert all("Authorization" not in request["headers"] for request in requests)
    assert len(terminals) == 2
