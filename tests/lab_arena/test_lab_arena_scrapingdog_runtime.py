"""Scrapingdog SDK handle compatibility without exposing provider secrets."""

from urllib.parse import urlencode

import pytest

from lab_arena import operations, runtime
from tests.lab_arena.test_lab_arena_runtime import make_spec


@pytest.mark.parametrize(
    "path, parameters, operation_id",
    [
        ("/profile", {"type": "company", "id": "microsoft"}, "scrapingdog.profile"),
        ("/x/profile", {"profileId": "scrapingdog"}, "scrapingdog.x_profile"),
        ("/google", {"query": "company launch"}, "scrapingdog.google"),
    ],
)
def test_runtime_handle_is_removed_before_dispatch(path, parameters, operation_id):
    url = "https://api.scrapingdog.com" + path + "?" + urlencode(
        dict(parameters, api_key=operations.SCRAPINGDOG_RUNTIME_HANDLE)
    )
    matched, normalized = operations.match_request("GET", url, b"", {})
    assert matched == operation_id
    assert "api_key" not in normalized
    assert operations.SCRAPINGDOG_RUNTIME_HANDLE not in repr(normalized)
    assert operations.match_request(
        "GET", "https://api.scrapingdog.com" + path + "?" + urlencode(parameters), b"", {}
    ) == (matched, normalized)


@pytest.mark.parametrize("key", ["real-provider-secret-value", "", "lab-arena-brokered-scrapingdog "])
def test_real_or_malformed_caller_keys_remain_forbidden(key):
    url = "https://api.scrapingdog.com/x/profile?" + urlencode(
        {"profileId": "scrapingdog", "api_key": key}
    )
    with pytest.raises(operations.OperationRequestError, match="forbidden_field"):
        operations.match_request("GET", url, b"", {})


def test_duplicate_key_cannot_hide_behind_removed_handle():
    url = (
        "https://api.scrapingdog.com/x/profile?profileId=scrapingdog&api_key="
        + operations.SCRAPINGDOG_RUNTIME_HANDLE
        + "&api_key=" + operations.SCRAPINGDOG_RUNTIME_HANDLE
    )
    with pytest.raises(operations.OperationRequestError, match="invalid_query"):
        operations.match_request("GET", url, b"", {})


def test_handle_does_not_allow_credentials_in_broker_parameters_or_headers():
    with pytest.raises(operations.OperationRequestError, match="forbidden_field"):
        operations.validate_operation_request(
            "scrapingdog.x_profile",
            {"profileId": "scrapingdog", "api_key": operations.SCRAPINGDOG_RUNTIME_HANDLE},
        )
    with pytest.raises(operations.OperationRequestError, match="forbidden_header"):
        operations.match_request(
            "GET", "https://api.scrapingdog.com/x/profile?profileId=scrapingdog", b"",
            {"Authorization": "Bearer " + operations.SCRAPINGDOG_RUNTIME_HANDLE},
        )


def test_environment_contains_only_public_handle_not_host_key(tmp_path, monkeypatch):
    secret = "host-scrapingdog-secret-must-not-enter-model"
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", secret)
    monkeypatch.setenv("LAB_ARENA_SCRAPINGDOG_API_KEY", secret)
    spec = make_spec(tmp_path, extra_environment={"SCRAPINGDOG_API_KEY": secret})
    environment = runtime.sandbox_environment(spec)
    assert "SCRAPINGDOG_API_KEY" not in environment
    assert secret not in repr(environment)
    configured = make_spec(tmp_path, extra_environment={
        "SCRAPINGDOG_API_KEY": operations.SCRAPINGDOG_RUNTIME_HANDLE,
    })
    assert runtime.sandbox_environment(configured)["SCRAPINGDOG_API_KEY"] == operations.SCRAPINGDOG_RUNTIME_HANDLE


@pytest.mark.parametrize("configured", [True, False, None, "true"])
def test_runner_exposes_optional_handle_only_when_gateway_confirms_it(tmp_path, configured):
    from lab_arena import runner
    from tests.lab_arena.test_lab_arena_runner import BridgingRuntime, FakeApi, lease, make_config

    claim = lease()
    if configured is not None:
        claim["scrapingdog_configured"] = configured
    api = FakeApi([claim])
    sandbox = BridgingRuntime(output={"companies": []}, calls=0)
    (tmp_path / "work").mkdir()
    worker = runner.Runner(make_config(tmp_path, api, sandbox))
    try:
        assert worker.run_once() == 1
        environment = runtime.sandbox_environment(sandbox.specs[0])
        assert bool(environment.get("SCRAPINGDOG_API_KEY")) is (configured is True)
        assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"
    finally:
        worker.close()


def test_standard_http_client_handle_crosses_worker_without_credentials():
    """Unit boundary check; the deployed provider proof is a separate live run."""
    import json
    import tempfile
    from pathlib import Path

    import httpx

    from lab_arena import runner
    from tests.lab_arena.test_lab_arena_runner import FakeApi, lease

    api = FakeApi([])
    state = runner.RunState(lease=lease("r1"), lease_token="tok-r1")
    with tempfile.TemporaryDirectory(prefix="sd-worker-", dir="/tmp") as directory:
        socket_path = Path(directory) / runtime.SANDBOX_SOCKET_NAME
        server = runner.WorkerSocketServer(socket_path, api, state)
        server.start()
        try:
            with httpx.Client(transport=httpx.HTTPTransport(uds=str(socket_path))) as client:
                response = client.get(
                    # Plain HTTP stays on the private Unix socket. The broker
                    # resolves the approved host to the real HTTPS upstream.
                    "http://api.scrapingdog.com/profile",
                    params={
                        "type": "company", "id": "microsoft",
                        "api_key": operations.SCRAPINGDOG_RUNTIME_HANDLE,
                    },
                )
            assert response.status_code == 200
            assert response.json()["results"]
            assert api.provider_frames[-1]["operation_id"] == "scrapingdog.profile"
            frame = json.dumps(api.provider_frames[-1])
            assert "api_key" not in frame
            assert operations.SCRAPINGDOG_RUNTIME_HANDLE not in frame
            assert len(state.calls) == 1
        finally:
            server.stop()


@pytest.mark.parametrize("fails", [False, True])
def test_provider_transport_never_logs_key_at_debug_level(caplog, fails):
    import logging

    import httpx

    from lab_arena.broker import HttpxProviderTransport, ProviderTransportError

    secret = "unit-test-scrapingdog-key-not-a-real-secret"
    url = "https://api.scrapingdog.com/google?" + urlencode({"api_key": secret, "query": "companies"})

    def provider(request):
        # HTTP/1.1 debug tracing can include response headers. HTTPX also logs
        # the complete request URL. Neither belongs in the service log.
        for name in ("httpcore.connection", "httpcore.http11", "httpcore.proxy"):
            logging.getLogger(name).debug("provider header echoed %s", secret)
        if fails:
            raise httpx.ConnectError("provider request failed", request=request)
        return httpx.Response(200, json={"organic_results": []})

    caplog.set_level(logging.DEBUG)
    # Exercise each transport logger even when another test configured it above
    # DEBUG. caplog restores each logger's prior level after this test.
    for name in ("httpx", "httpcore.connection", "httpcore.http11", "httpcore.proxy"):
        caplog.set_level(logging.DEBUG, logger=name)
    with httpx.Client(transport=httpx.MockTransport(provider)) as client:
        transport = HttpxProviderTransport(client=client)
        if fails:
            with pytest.raises(ProviderTransportError):
                transport.send(method="GET", url=url, headers={}, body=b"", timeout_seconds=5)
        else:
            assert transport.send(method="GET", url=url, headers={}, body=b"", timeout_seconds=5).status == 200
    logging.getLogger("httpx").info("unrelated HTTP diagnostic still visible")
    assert secret not in caplog.text
    assert "unrelated HTTP diagnostic still visible" in caplog.text
