import gzip

import pytest

from gateway.middleware.body_size import (
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_WEIGHT_AUTHORITY_MAX_BODY_BYTES,
    WEIGHT_AUTHORITY_GRAPH_PATHS,
    BodySizeLimitMiddleware,
)


_FAILED_PRODUCTION_WIRE_BYTES = 10_701_583


def _receipt_graph_json_body(size: int) -> bytes:
    prefix = b'{"receipt_graph":{"transport_attempts":["'
    suffix = b'"]}}'
    return prefix + (b"x" * (size - len(prefix) - len(suffix))) + suffix


async def _drive_request(
    middleware: BodySizeLimitMiddleware,
    *,
    path: str,
    body: bytes,
    method: str = "POST",
    include_content_length: bool = True,
) -> tuple[list[bytes], list[dict]]:
    observed = []
    sent = []

    async def app(scope, receive, send):
        message = await receive()
        observed.append(message.get("body") or b"")
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware.app = app
    headers = []
    if include_content_length:
        headers.append((b"content-length", str(len(body)).encode("ascii")))
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers,
    }
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)
    return observed, sent


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/weights/submit/v2",
        "/weights/finalize/v2",
    ],
)
async def test_weight_authority_routes_accept_failed_production_body_size(
    path: str,
) -> None:
    body = _receipt_graph_json_body(_FAILED_PRODUCTION_WIRE_BYTES)
    assert len(body) > DEFAULT_MAX_BODY_BYTES

    middleware = BodySizeLimitMiddleware(lambda *_args: None)
    observed, sent = await _drive_request(
        middleware,
        path=path,
        body=body,
    )

    assert observed == [body]
    assert sent[0]["status"] == 204


@pytest.mark.asyncio
async def test_large_body_allowance_is_exact_and_post_only() -> None:
    body = _receipt_graph_json_body(DEFAULT_MAX_BODY_BYTES + 1)
    middleware = BodySizeLimitMiddleware(lambda *_args: None)

    for path, method in (
        ("/fulfillment/scoring", "POST"),
        ("/weights/submit/v2/typo", "POST"),
        ("/weights/submit/v2", "GET"),
    ):
        observed, sent = await _drive_request(
            middleware,
            path=path,
            method=method,
            body=body,
        )
        assert observed == []
        assert sent[0]["status"] == 413


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(WEIGHT_AUTHORITY_GRAPH_PATHS))
async def test_all_weight_authority_graph_routes_share_scoped_limit(
    path: str,
) -> None:
    middleware = BodySizeLimitMiddleware(
        lambda *_args: None,
        max_body_bytes=4,
        max_weight_authority_body_bytes=8,
    )
    observed, sent = await _drive_request(
        middleware,
        path=path,
        body=b"12345678",
    )

    assert observed == [b"12345678"]
    assert sent[0]["status"] == 204


@pytest.mark.asyncio
async def test_weight_authority_limit_still_rejects_oversized_body() -> None:
    middleware = BodySizeLimitMiddleware(lambda *_args: None)
    sent = []
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/weights/finalize/v2",
        "headers": [
            (
                b"content-length",
                str(DEFAULT_WEIGHT_AUTHORITY_MAX_BODY_BYTES + 1).encode("ascii"),
            ),
        ],
    }

    async def receive():
        raise AssertionError("declared oversized body must not be consumed")

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)

    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_weight_authority_limit_rejects_oversized_streamed_body() -> None:
    middleware = BodySizeLimitMiddleware(
        lambda *_args: None,
        max_body_bytes=4,
        max_weight_authority_body_bytes=8,
    )
    observed, sent = await _drive_request(
        middleware,
        path="/weights/submit/v2",
        body=b"123456789",
        include_content_length=False,
    )

    assert observed == []
    assert sent[0]["status"] == 413
