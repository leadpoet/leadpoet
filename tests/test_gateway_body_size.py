import gzip

import pytest

from gateway.middleware.body_size import BodySizeLimitMiddleware


async def _invoke(
    middleware,
    *,
    path,
    body,
    content_encoding=None,
    fragment_size=None,
):
    headers = [(b"content-length", str(len(body)).encode("ascii"))]
    if content_encoding is not None:
        headers.append((b"content-encoding", content_encoding.encode("ascii")))
    scope = {"type": "http", "path": path, "headers": headers}
    chunk_size = fragment_size or max(1, len(body))
    chunks = [
        body[index : index + chunk_size]
        for index in range(0, len(body), chunk_size)
    ] or [b""]
    received = []
    sent = []

    async def receive():
        chunk = chunks.pop(0)
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": bool(chunks),
        }

    async def send(message):
        sent.append(message)

    async def app(observed_scope, observed_receive, observed_send):
        request = await observed_receive()
        received.append((observed_scope, request))
        await observed_send(
            {"type": "http.response.start", "status": 204, "headers": []}
        )
        await observed_send(
            {"type": "http.response.body", "body": b"", "more_body": False}
        )

    middleware.app = app
    await middleware(scope, receive, send)
    return received, sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    (
        "/weights/submit/v2",
        "/weights/finalize/v2",
        "/weights/subnet-epoch/candidate/v1",
        "/weights/subnet-epoch/boundary/v1",
    ),
)
async def test_large_weight_authority_gzip_is_bounded_then_decompressed(path):
    body = b'{"receipt_graph":"' + b"x" * 2048 + b'"}'
    compressed = gzip.compress(body, compresslevel=1, mtime=0)
    middleware = BodySizeLimitMiddleware(
        None,
        max_body_bytes=1024,
        max_weight_body_bytes=4096,
    )

    received, sent = await _invoke(
        middleware,
        path=path,
        body=compressed,
        content_encoding="gzip",
        fragment_size=7,
    )

    assert received[0][1]["body"] == body
    headers = dict(received[0][0]["headers"])
    assert b"content-encoding" not in headers
    assert headers[b"content-length"] == str(len(body)).encode("ascii")
    assert sent[0]["status"] == 204


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "expected_status"),
    (
        (b"not-gzip", 400),
        (gzip.compress(b"x" * 101, mtime=0), 400),
        (gzip.compress(b"{}", mtime=0) + b"trailing", 400),
    ),
)
async def test_invalid_or_expanded_oversized_weight_gzip_fails_closed(
    body,
    expected_status,
):
    middleware = BodySizeLimitMiddleware(
        None,
        max_body_bytes=1024,
        max_weight_body_bytes=100,
    )

    received, sent = await _invoke(
        middleware,
        path="/weights/submit/v2",
        body=body,
        content_encoding="gzip",
    )

    assert received == []
    assert sent[0]["status"] == expected_status


@pytest.mark.asyncio
async def test_compressed_request_is_rejected_outside_weight_authority_paths():
    middleware = BodySizeLimitMiddleware(None, max_body_bytes=1024)

    received, sent = await _invoke(
        middleware,
        path="/research-lab/status",
        body=gzip.compress(b"{}", mtime=0),
        content_encoding="gzip",
    )

    assert received == []
    assert sent[0]["status"] == 415


@pytest.mark.asyncio
async def test_compressed_weight_request_rejects_oversized_wire_body():
    middleware = BodySizeLimitMiddleware(
        None,
        max_body_bytes=8,
        max_weight_body_bytes=4096,
    )

    received, sent = await _invoke(
        middleware,
        path="/weights/submit/v2",
        body=gzip.compress(b"{}" * 64, mtime=0),
        content_encoding="gzip",
        fragment_size=3,
    )

    assert received == []
    assert sent[0]["status"] == 413
