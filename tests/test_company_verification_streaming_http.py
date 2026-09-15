"""Exercise homepage verification over real, fragmented loopback HTTP."""

import asyncio
import socket

import aiohttp
from aiohttp import web
import pytest

from qualification.scoring.company_verification import verify_company_exists


class _LoopbackResolver(aiohttp.abc.AbstractResolver):
    async def resolve(self, host, port=0, family=socket.AF_INET):
        assert host == "example.com"
        return [{
            "hostname": host,
            "host": "127.0.0.1",
            "port": port,
            "family": socket.AF_INET,
            "proto": 0,
            "flags": 0,
        }]

    async def close(self):
        pass


@pytest.mark.parametrize("first_chunk_size", [64, 159 * 1024, 229 * 1024])
@pytest.mark.parametrize(
    "encoding,content_type",
    [
        ("utf-8", "text/html; charset=utf-8"),
        ("cp1252", "text/html; charset=windows-1252"),
        ("utf-16", "text/html"),
    ],
)
def test_real_http_stream_keeps_late_homepage_identity(
    monkeypatch, first_chunk_size, encoding, content_type
):
    original_session = aiohttp.ClientSession
    title = "<title>Example Company</title>" if encoding == "utf-8" else ""
    footer = (
        '<footer>© 2026 Example Company Ltd. All rights reserved '
        '<a href="https://www.linkedin.com/company/example-company">'
        "LinkedIn</a></footer>"
    )
    payload = (
        title + " " * (408 * 1024 - len(title) - len(footer)) + footer
    ).encode(encoding)

    async def homepage(request):
        response = web.StreamResponse(headers={"Content-Type": content_type})
        await response.prepare(request)
        await response.write(payload[:first_chunk_size])
        await asyncio.sleep(0.02)
        await response.write(payload[first_chunk_size:])
        await response.write_eof()
        return response

    def local_session(**kwargs):
        return original_session(
            connector=aiohttp.TCPConnector(resolver=_LoopbackResolver()), **kwargs
        )

    async def run():
        app = web.Application()
        app.router.add_get("/", homepage)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            monkeypatch.setattr(aiohttp, "ClientSession", local_session)

            result = await verify_company_exists(
                "Example Company",
                f"http://example.com:{port}/",
                company_linkedin="https://www.linkedin.com/company/example-company",
            )

            assert result.decision == "match", result.reason
            assert result.details["identity"]["evidence_source"] == "company_homepage"
        finally:
            await runner.cleanup()

    asyncio.run(run())
