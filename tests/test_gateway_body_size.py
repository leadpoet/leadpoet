import pytest

from gateway.middleware.body_size import BodySizeLimitMiddleware


async def _invoke(middleware, *, body, headers=(), chunks=None):
    observed, sent = [], []

    async def app(scope, receive, send):
        while True:
            message = await receive()
            observed.append(message)
            if message.get("type") != "http.request" or not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware.app = app
    parts = list(chunks) if chunks is not None else [body]
    messages = [
        {"type": "http.request", "body": part, "more_body": index < len(parts) - 1}
        for index, part in enumerate(parts)
    ]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await middleware(
        {"type": "http", "method": "POST", "path": "/arena/v1/score", "headers": list(headers)},
        receive,
        send,
    )
    return observed, sent


@pytest.mark.asyncio
async def test_plain_body_within_limit_is_delivered_unchanged():
    body = b'{"score":1}'
    observed, sent = await _invoke(
        BodySizeLimitMiddleware(None, max_body_bytes=len(body)),
        body=body,
        headers=[(b"content-length", str(len(body)).encode())],
    )
    assert sent[0]["status"] == 204
    assert observed == [{"type": "http.request", "body": body, "more_body": False}]


@pytest.mark.asyncio
async def test_streamed_overflow_emits_one_rejection():
    observed, sent = await _invoke(
        BodySizeLimitMiddleware(None, max_body_bytes=5),
        body=b"",
        chunks=[b"123", b"456"],
    )
    assert observed == [
        {"type": "http.request", "body": b"123", "more_body": True},
        {"type": "http.disconnect"},
    ]
    assert [item["type"] for item in sent] == ["http.response.start", "http.response.body"]
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_compressed_body_is_rejected_before_delivery():
    observed, sent = await _invoke(
        BodySizeLimitMiddleware(None, max_body_bytes=100),
        body=b"compressed",
        headers=[(b"content-encoding", b"gzip")],
    )
    assert observed == []
    assert sent[0]["status"] == 415
