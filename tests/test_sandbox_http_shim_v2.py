from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

from research_lab.eval.provider_evidence_cache import (
    canonical_request_fingerprint,
)


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
            calls.append((kwargs["method"], kwargs["url"], bytes(kwargs["body"])))
            return original_execute(**kwargs)

        def forbidden_socket(*args, **kwargs):
            raise AssertionError("frozen evidence attempted a network socket")

        shim.execute = guarded_execute
        shim.socket = SimpleNamespace(
            AF_UNIX=object(), SOCK_STREAM=object(), socket=forbidden_socket
        )
        shim.install()

        with urllib.request.urlopen(url, timeout=1) as response:
            urllib_doc = json.loads(response.read().decode("utf-8"))
        requests_doc = requests.get(url, timeout=1).json()
        httpx_doc = httpx.get(url, timeout=1).json()

        async def aiohttp_request():
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=1) as response:
                    return await response.json()

        aiohttp_doc = asyncio.run(aiohttp_request())
        expected = {response_doc!r}
        assert urllib_doc == expected
        assert requests_doc == expected
        assert httpx_doc == expected
        assert aiohttp_doc == expected
        assert calls == [("GET", url, b"")] * 4
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
