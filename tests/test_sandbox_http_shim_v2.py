from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
from types import ModuleType

import pytest

from gateway.tee.inter_enclave_tls import REPLAY_WAIT_SECONDS
from leadpoet_canonical.attested_v2 import canonical_json
from research_lab.eval.provider_evidence_cache import (
    canonical_request_fingerprint,
)


def _install_qualification_route_module(monkeypatch, transport_headers):
    package = ModuleType("sourcing_model")
    package.__path__ = []
    module = ModuleType("sourcing_model.qualification_route")
    module.transport_headers = transport_headers
    package.qualification_route = module
    monkeypatch.setitem(sys.modules, "sourcing_model", package)
    monkeypatch.setitem(
        sys.modules,
        "sourcing_model.qualification_route",
        module,
    )


def test_route_commitment_hook_injects_once_and_overrides_caller_header(
    monkeypatch,
) -> None:
    import gateway.tee.sandbox_http_shim_v2 as shim

    calls = []
    broker_requests = []

    def transport_headers():
        calls.append("called")
        return {
            "X-Leadpoet-Qualification-Route-Commitment": "a" * 64,
        }

    _install_qualification_route_module(monkeypatch, transport_headers)
    monkeypatch.setenv(shim.SOCKET_ENV, "/tmp/provider.sock")
    monkeypatch.setattr(shim, "_snapshot_terminal", lambda **_kwargs: None)
    monkeypatch.setattr(shim, "_cached_terminal", lambda **_kwargs: None)

    def execute_broker_request(*, socket_path, encoded):
        assert socket_path == "/tmp/provider.sock"
        broker_requests.append(json.loads(encoded))
        return {"terminal_status": "transport_failure"}

    monkeypatch.setattr(
        shim,
        "_execute_broker_request",
        execute_broker_request,
    )

    shim.execute(
        method="GET",
        url="https://api.exa.ai/search",
        headers={
            "X-Leadpoet-Qualification-Route-Commitment": "f" * 64,
            "Accept": "application/json",
        },
        body=b"",
        timeout_ms=1,
    )

    assert calls == ["called"]
    assert broker_requests[0]["headers"] == {
        "Accept": "application/json",
        "X-Leadpoet-Qualification-Route-Commitment": "a" * 64,
    }


@pytest.mark.parametrize(
    "transport_headers",
    [
        lambda: {
            "X-Leadpoet-Qualification-Route-Commitment": "A" * 64,
        },
        lambda: {"X-Untrusted": "value"},
        lambda: None,
    ],
)
def test_invalid_route_commitment_hook_rejects_before_broker(
    monkeypatch,
    transport_headers,
) -> None:
    import gateway.tee.sandbox_http_shim_v2 as shim

    _install_qualification_route_module(monkeypatch, transport_headers)
    monkeypatch.setenv(shim.SOCKET_ENV, "/tmp/provider.sock")
    monkeypatch.setattr(shim, "_snapshot_terminal", lambda **_kwargs: None)
    monkeypatch.setattr(shim, "_cached_terminal", lambda **_kwargs: None)
    broker_calls = []
    monkeypatch.setattr(
        shim,
        "_execute_broker_request",
        lambda **kwargs: broker_calls.append(kwargs),
    )

    with pytest.raises(shim.SandboxHTTPShimV2Error, match="headers are invalid"):
        shim.execute(
            method="GET",
            url="https://api.exa.ai/search",
            headers={},
            body=b"",
            timeout_ms=1,
        )

    assert broker_calls == []


def test_absent_hook_is_legacy_safe_but_v2_fails_closed(monkeypatch) -> None:
    import gateway.tee.sandbox_http_shim_v2 as shim

    package = ModuleType("sourcing_model")
    package.__path__ = []
    module = ModuleType("sourcing_model.qualification_route")
    package.qualification_route = module
    monkeypatch.setitem(sys.modules, "sourcing_model", package)
    monkeypatch.setitem(
        sys.modules,
        "sourcing_model.qualification_route",
        module,
    )

    assert shim._qualification_route_transport_headers() == {}
    monkeypatch.setenv(shim.QUALIFICATION_PROTOCOL_V2_ENV, "1")
    with pytest.raises(shim.SandboxHTTPShimV2Error, match="hook is invalid"):
        shim._qualification_route_transport_headers()


def test_live_socket_waits_for_measured_operation_completion(monkeypatch) -> None:
    import gateway.tee.sandbox_http_shim_v2 as shim

    response = canonical_json(
        {"result": {"terminal_status": "transport_failure"}}
    ).encode("utf-8")
    chunks = [len(response).to_bytes(4, "big"), response]
    observed = {}

    class Connection:
        def settimeout(self, timeout):
            observed["timeout"] = timeout

        def connect(self, path):
            observed["path"] = path

        def sendall(self, payload):
            observed["request"] = bytes(payload)

        def recv(self, size):
            chunk = chunks.pop(0)
            assert len(chunk) == size
            return chunk

        def close(self):
            observed["closed"] = True

    monkeypatch.setenv(shim.SOCKET_ENV, "/tmp/provider.sock")
    monkeypatch.setattr(shim.socket, "socket", lambda *_args: Connection())

    result = shim.execute(
        method="GET",
        url="https://api.exa.ai/search",
        headers={},
        body=b"",
        timeout_ms=1,
    )

    assert result == {"terminal_status": "transport_failure"}
    assert observed["timeout"] == REPLAY_WAIT_SECONDS
    assert observed["path"] == "/tmp/provider.sock"
    assert observed["closed"] is True


def test_broker_request_retains_valid_result_until_required_close_succeeds(
    monkeypatch,
) -> None:
    import gateway.tee.sandbox_http_shim_v2 as shim

    terminal = {"terminal_status": "transport_failure", "failure_code": "timeout"}
    response = canonical_json({"result": terminal}).encode("utf-8")

    class Connection:
        def __init__(self, *, allow_close):
            self.allow_close = allow_close
            self.chunks = [len(response).to_bytes(4, "big"), response]

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            pass

        def sendall(self, _payload):
            pass

        def recv(self, size):
            chunk = self.chunks.pop(0)
            assert len(chunk) == size
            return chunk

        def close(self):
            return self.allow_close

    connection = Connection(allow_close=False)
    recovered_connection = Connection(allow_close=True)
    connections = iter((connection, recovered_connection))
    monkeypatch.setenv(shim.SOCKET_ENV, "/tmp/provider.sock")
    monkeypatch.setattr(shim.socket, "socket", lambda *_args: next(connections))

    with pytest.raises(shim.SandboxHTTPShimTransportCleanupError) as raised:
        shim.execute(
            method="GET",
            url="https://api.exa.ai/search",
            headers={},
            body=b"",
            timeout_ms=1,
        )

    assert raised.value._resource is connection
    assert raised.value._result == terminal
    assert raised.value.primary_error is raised.value.__cause__
    assert shim._RETIRED_CLEANUP_RESOURCES[id(connection)][0] is connection
    connection.allow_close = True
    assert shim.execute(
        method="GET",
        url="https://api.exa.ai/search",
        headers={},
        body=b"",
        timeout_ms=1,
    ) == terminal
    assert shim._RETIRED_CLEANUP_RESOURCES == {}


def test_broker_request_cleanup_preserves_original_request_error(monkeypatch) -> None:
    import gateway.tee.sandbox_http_shim_v2 as shim

    primary = OSError("connect failed")

    class Connection:
        def __init__(self):
            self.allow_close = False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            raise primary

        def close(self):
            return self.allow_close

    connection = Connection()
    monkeypatch.setenv(shim.SOCKET_ENV, "/tmp/provider.sock")
    monkeypatch.setattr(shim.socket, "socket", lambda *_args: connection)

    with pytest.raises(shim.SandboxHTTPShimTransportCleanupError) as raised:
        shim.execute(
            method="GET",
            url="https://api.exa.ai/search",
            headers={},
            body=b"",
            timeout_ms=1,
        )

    assert raised.value.primary_error is primary
    assert raised.value.__cause__ is primary
    assert raised.value._result is None
    connection.allow_close = True
    shim._require_retired_cleanup()


def test_cached_terminal_still_retires_prior_live_socket(monkeypatch) -> None:
    import gateway.tee.sandbox_http_shim_v2 as shim

    class RetiredConnection:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    retired = RetiredConnection()
    terminal = {
        "terminal_status": "attested_local_response",
        "http_status": 200,
        "headers": {},
        "body_b64": "",
        "failure_code": None,
    }
    shim._retain_cleanup_failure(
        retired,
        primary_error=OSError("prior cleanup failed"),
        result=None,
    )
    monkeypatch.setattr(
        shim,
        "_snapshot_terminal",
        lambda **_kwargs: dict(terminal),
    )

    assert shim.execute(
        method="GET",
        url="https://api.exa.ai/search",
        headers={},
        body=b"",
        timeout_ms=1,
    ) == terminal
    assert retired.closed is True
    assert shim._RETIRED_CLEANUP_RESOURCES == {}


def test_frozen_evidence_miss_emits_bounded_sentinel_before_raise(
    monkeypatch,
) -> None:
    import gateway.tee.sandbox_http_shim_v2 as shim

    writes = []

    def capture(fd, data):
        writes.append((fd, bytes(data)))
        return len(data)

    monkeypatch.setattr(shim, "_EVIDENCE_MISS_WRITE", capture)
    method = "POST"
    url = "https://api.exa.ai/search"
    body = b'{"query":"bounded"}'
    fingerprint = canonical_request_fingerprint(method, url, body)

    with pytest.raises(
        shim.SandboxHTTPShimV2Error,
        match=shim.EVIDENCE_MISS_SENTINEL + fingerprint,
    ):
        shim._cached_terminal(
            method=method,
            url=url,
            body=body,
            mode="frozen",
            cache={},
        )

    assert writes == [
        (
            2,
            (shim.EVIDENCE_MISS_SENTINEL + fingerprint + "\n").encode(
                "ascii"
            ),
        )
    ]


def test_v2_frozen_miss_captures_route_once_and_never_reaches_broker(
    monkeypatch,
) -> None:
    import gateway.tee.sandbox_http_shim_v2 as shim

    hook_calls = []

    def transport_headers():
        hook_calls.append("called")
        return {
            "X-Leadpoet-Qualification-Route-Commitment": "e" * 64,
        }

    _install_qualification_route_module(monkeypatch, transport_headers)
    monkeypatch.setenv(shim.QUALIFICATION_PROTOCOL_V2_ENV, "1")
    monkeypatch.setenv(shim.EVIDENCE_MODE_ENV, "frozen")
    monkeypatch.setenv(shim.SOCKET_ENV, "/tmp/provider.sock")
    monkeypatch.setattr(shim, "_snapshot_terminal", lambda **_kwargs: None)
    monkeypatch.setattr(shim, "_evidence_cache", lambda: {})
    monkeypatch.setattr(
        shim,
        "_execute_broker_request",
        lambda **_kwargs: pytest.fail("frozen replay miss reached broker"),
    )

    with pytest.raises(shim.SandboxHTTPShimV2Error, match="EVIDENCE_MISS"):
        shim.execute(
            method="POST",
            url="https://api.exa.ai/search",
            headers={},
            body=b"{}",
            timeout_ms=1000,
        )

    assert hook_calls == ["called"]


@pytest.mark.parametrize("write_result", ("short", "error"))
def test_frozen_evidence_miss_write_is_diagnostic_best_effort(
    monkeypatch,
    write_result,
) -> None:
    import gateway.tee.sandbox_http_shim_v2 as shim

    def diagnostic_write(_fd, _data):
        if write_result == "error":
            raise OSError("diagnostic sink unavailable")
        return 0

    monkeypatch.setattr(shim, "_EVIDENCE_MISS_WRITE", diagnostic_write)
    fingerprint = canonical_request_fingerprint(
        "GET",
        "https://api.exa.ai/search",
        b"",
    )

    with pytest.raises(shim.SandboxHTTPShimV2Error) as raised:
        shim._cached_terminal(
            method="GET",
            url="https://api.exa.ai/search",
            body=b"",
            mode="frozen",
            cache={},
        )

    assert str(raised.value) == shim.EVIDENCE_MISS_SENTINEL + fingerprint


def test_retired_cleanup_allows_concurrent_owner_transfer(monkeypatch) -> None:
    import gateway.tee.sandbox_http_shim_v2 as shim

    close_started = threading.Event()
    release_close = threading.Event()

    class BlockingConnection:
        def shutdown(self, _how):
            return None

        def close(self):
            close_started.set()
            assert release_close.wait(timeout=1)
            return None

    class ConcurrentConnection:
        def shutdown(self, _how):
            return None

        def close(self):
            return None

    first = BlockingConnection()
    concurrent = ConcurrentConnection()
    monkeypatch.setattr(shim, "_RETIRED_CLEANUP_RESOURCES", {})
    shim._retain_cleanup_failure(
        first,
        primary_error=OSError("first cleanup"),
        result=None,
    )
    recovery_errors = []

    def recover():
        try:
            shim._require_retired_cleanup()
        except Exception as exc:
            recovery_errors.append(exc)

    recovery = threading.Thread(target=recover)
    recovery.start()
    assert close_started.wait(timeout=1)
    retained = threading.Event()

    def retain():
        shim._retain_cleanup_failure(
            concurrent,
            primary_error=OSError("concurrent cleanup"),
            result=None,
        )
        retained.set()

    retention = threading.Thread(target=retain)
    retention.start()
    assert retained.wait(timeout=1)
    assert shim._RETIRED_CLEANUP_RESOURCES[id(concurrent)][0] is concurrent
    release_close.set()
    recovery.join(timeout=1)
    retention.join(timeout=1)

    assert len(recovery_errors) == 1
    assert list(shim._RETIRED_CLEANUP_RESOURCES.values())[0][0] is concurrent
    shim._require_retired_cleanup()
    assert shim._RETIRED_CLEANUP_RESOURCES == {}


def test_all_supported_http_clients_use_the_same_frozen_evidence(
    tmp_path: Path,
) -> None:
    url = "https://provider.example/v1/search?query=tree"
    response_doc = {"companies": [{"domain": "example.com"}], "source": "frozen"}
    response_body = json.dumps(response_doc, separators=(",", ":")).encode("utf-8")
    fingerprint = canonical_request_fingerprint("GET", url, b"")
    cache_path = tmp_path / "provider-evidence.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "entries": {
                    fingerprint: {
                        "status": 200,
                        "body_b64": base64.b64encode(response_body).decode("ascii"),
                    }
                },
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    script = textwrap.dedent(
        f"""
        import asyncio
        import json
        from types import SimpleNamespace
        import urllib.request

        import aiohttp
        import httpx
        import requests

        import gateway.tee.sandbox_http_shim_v2 as shim

        url = {url!r}
        calls = []
        original_execute = shim.execute

        def guarded_execute(**kwargs):
            calls.append(
                (
                    kwargs["method"],
                    kwargs["url"],
                    bytes(kwargs["body"]),
                    kwargs["timeout_ms"],
                )
            )
            return original_execute(**kwargs)

        def forbidden_socket(*args, **kwargs):
            raise AssertionError("frozen evidence attempted a network socket")

        shim.execute = guarded_execute
        shim.socket = SimpleNamespace(
            AF_UNIX=object(), SOCK_STREAM=object(), socket=forbidden_socket
        )
        shim.install()

        with urllib.request.urlopen(url, timeout=2) as response:
            urllib_doc = json.loads(response.read().decode("utf-8"))
        requests_doc = requests.get(url, timeout=(3, 5)).json()
        httpx_doc = httpx.get(
            url,
            timeout=httpx.Timeout(180, connect=7),
        ).json()

        async def aiohttp_request():
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=11),
                ) as response:
                    return await response.json()

        aiohttp_doc = asyncio.run(aiohttp_request())
        expected = {response_doc!r}
        assert urllib_doc == expected
        assert requests_doc == expected
        assert httpx_doc == expected
        assert aiohttp_doc == expected
        assert calls == [
            ("GET", url, b"", 2_000),
            ("GET", url, b"", 5_000),
            ("GET", url, b"", 180_000),
            ("GET", url, b"", 11_000),
        ]
        print(json.dumps({{"clients": 4, "fingerprint": {fingerprint!r}}}))
        """
    )
    env = dict(os.environ)
    env.update(
        {
            "RESEARCH_LAB_PROVIDER_EVIDENCE_MODE": "frozen",
            "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH": str(cache_path),
            "LEADPOET_SANDBOX_PROVIDER_SOCKET": "/nonexistent/provider.sock",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "clients": 4,
        "fingerprint": fingerprint,
    }


def test_all_supported_http_clients_preserve_attested_transport_failure() -> None:
    script = textwrap.dedent(
        """
        import asyncio
        import json
        import urllib.request

        import aiohttp
        import httpx
        import requests

        import gateway.tee.sandbox_http_shim_v2 as shim

        url = "https://provider.example/v1/search"
        failure = {
            "terminal_status": "transport_failure",
            "failure_code": "timeout",
        }
        shim.execute = lambda **_kwargs: dict(failure)
        shim.install()

        errors = {}
        for name, call in {
            "urllib": lambda: urllib.request.urlopen(url, timeout=1),
            "requests": lambda: requests.get(url, timeout=1),
            "httpx": lambda: httpx.get(url, timeout=1),
        }.items():
            try:
                call()
            except Exception as exc:
                errors[name] = [type(exc).__name__, str(exc)]

        async def call_aiohttp():
            try:
                async with aiohttp.ClientSession() as session:
                    await session.get(url, timeout=1)
            except Exception as exc:
                errors["aiohttp"] = [type(exc).__name__, str(exc)]

        asyncio.run(call_aiohttp())
        assert set(errors) == {"urllib", "requests", "httpx", "aiohttp"}
        assert errors["urllib"][0] == "URLError"
        assert errors["requests"][0] == "ConnectionError"
        assert errors["httpx"][0] == "TransportError"
        assert errors["aiohttp"][0] == "ClientConnectionError"
        assert all(
            "attested transport failure: timeout" in value[1]
            for value in errors.values()
        )
        print(json.dumps(errors, sort_keys=True))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert set(json.loads(completed.stdout)) == {
        "urllib",
        "requests",
        "httpx",
        "aiohttp",
    }


@pytest.mark.parametrize("client_name", ("httpx", "aiohttp"))
def test_async_client_cancellation_waits_for_started_transport(
    client_name: str,
) -> None:
    script = textwrap.dedent(
        f"""
        import asyncio
        import json
        import os
        import sys
        import threading
        from types import ModuleType

        import aiohttp
        import httpx

        import gateway.tee.sandbox_http_shim_v2 as shim

        client_name = {client_name!r}
        transport_started = threading.Event()
        release_transport = threading.Event()
        transport_finished = threading.Event()
        route_active = {{"value": True}}
        observations = []

        package = ModuleType("sourcing_model")
        package.__path__ = []
        route_module = ModuleType("sourcing_model.qualification_route")

        def transport_headers():
            transport_started.set()
            assert release_transport.wait(5)
            observations.append(route_active["value"])
            if not route_active["value"]:
                raise RuntimeError("route binding already closed")
            return {{
                "X-Leadpoet-Qualification-Route-Commitment": "a" * 64,
            }}

        route_module.transport_headers = transport_headers
        package.qualification_route = route_module
        sys.modules["sourcing_model"] = package
        sys.modules["sourcing_model.qualification_route"] = route_module
        os.environ[shim.QUALIFICATION_PROTOCOL_V2_ENV] = "1"
        os.environ[shim.SOCKET_ENV] = "/tmp/provider.sock"
        shim._snapshot_terminal = lambda **_kwargs: None
        shim._cached_terminal = lambda **_kwargs: None

        def execute_broker_request(**_kwargs):
            transport_finished.set()
            return {{
                "terminal_status": "authenticated_response",
                "http_status": 200,
                "headers": {{"content-type": "application/json"}},
                "body_b64": "e30=",
            }}

        shim._execute_broker_request = execute_broker_request
        shim.install()

        async def request():
            if client_name == "httpx":
                async with httpx.AsyncClient() as client:
                    await client.get("https://provider.example/search")
                return
            async with aiohttp.ClientSession() as session:
                await session.get("https://provider.example/search")

        async def exercise():
            task = asyncio.create_task(request())
            assert await asyncio.to_thread(transport_started.wait, 2)
            asyncio.get_running_loop().call_later(
                0.05,
                release_transport.set,
            )
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            route_active["value"] = False
            assert await asyncio.to_thread(transport_finished.wait, 2)

        asyncio.run(exercise())
        assert observations == [True], observations
        print(json.dumps(observations))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [True]
