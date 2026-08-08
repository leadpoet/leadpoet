from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

from gateway.tee.inter_enclave_tls import REPLAY_WAIT_SECONDS
from leadpoet_canonical.attested_v2 import canonical_json
from research_lab.eval.provider_evidence_cache import (
    canonical_request_fingerprint,
)


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
