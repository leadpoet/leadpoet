import gzip

import pytest

from gateway.middleware.body_size import BodySizeLimitMiddleware


@pytest.mark.asyncio
async def test_body_guard_never_expands_unauthenticated_gzip() -> None:
    compressed = gzip.compress(b"x" * (1024 * 1024), mtime=0)
    observed = []
    sent = []

    async def app(scope, receive, send):
        observed.append((scope, await receive()))
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = BodySizeLimitMiddleware(
        app,
        max_body_bytes=len(compressed) + 1,
    )
    scope = {
        "type": "http",
        "path": "/weights/submit/v2",
        "headers": [
            (b"content-encoding", b"gzip"),
            (b"content-length", str(len(compressed)).encode("ascii")),
        ],
    }
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {
            "type": "http.request",
            "body": compressed,
            "more_body": False,
        }

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)

    assert observed[0][0] is scope
    assert observed[0][1]["body"] == compressed
    assert sent[0]["status"] == 204
