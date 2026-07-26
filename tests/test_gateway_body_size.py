import base64
import gzip
import json

import pytest

from gateway.middleware.body_size import BodySizeLimitMiddleware
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_bytes,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    build_weight_transport_authorization_v2,
    weight_transport_authorization_message_v2,
)


HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"


def _transport_headers(
    *,
    wire_body,
    logical_body,
    path="/weights/submit/v2",
    hotkey=HOTKEY,
    signature="a" * 128,
):
    authorization = build_weight_transport_authorization_v2(
        validator_hotkey=hotkey,
        path=path,
        wire_body_hash=sha256_bytes(wire_body),
        wire_body_bytes=len(wire_body),
        logical_body_hash=sha256_bytes(logical_body),
        logical_body_bytes=len(logical_body),
    )
    return authorization, [
        (b"content-type", b"application/json"),
        (b"content-encoding", b"gzip"),
        (b"content-length", str(len(wire_body)).encode("ascii")),
        (
            b"x-leadpoet-weight-transport",
            base64.b64encode(
                canonical_json(authorization).encode("utf-8")
            ),
        ),
        (
            b"x-leadpoet-weight-transport-signature",
            signature.encode("ascii"),
        ),
    ]


async def _invoke(
    middleware,
    *,
    body,
    headers,
    path="/weights/submit/v2",
    method="POST",
    chunks=None,
):
    observed = []
    sent = []

    async def app(scope, receive, send):
        request = await receive()
        observed.append((scope, request))
        await send(
            {"type": "http.response.start", "status": 204, "headers": []}
        )
        await send({"type": "http.response.body", "body": b""})

    middleware.app = app
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": list(headers),
    }
    messages = []
    parts = list(chunks) if chunks is not None else [body]
    for index, part in enumerate(parts):
        messages.append(
            {
                "type": "http.request",
                "body": part,
                "more_body": index < len(parts) - 1,
            }
        )

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)
    return observed, sent


@pytest.mark.asyncio
async def test_authenticated_gzip_is_verified_then_delivered_as_exact_json(
    monkeypatch,
) -> None:
    logical = canonical_json(
        {"validator_hotkey": HOTKEY, "receipt_graph": ["x" * 4096]}
    ).encode("utf-8")
    compressed = gzip.compress(logical, compresslevel=1, mtime=0)
    _authorization, headers = _transport_headers(
        wire_body=compressed,
        logical_body=logical,
    )
    monkeypatch.setenv("PRIMARY_VALIDATOR_HOTKEYS", HOTKEY)
    monkeypatch.setattr(
        "gateway.middleware.body_size._verify_transport_signature",
        lambda **_kwargs: True,
    )
    middleware = BodySizeLimitMiddleware(
        None,
        max_body_bytes=len(compressed) + 1,
        max_weight_body_bytes=len(logical) + 1,
    )

    observed, sent = await _invoke(
        middleware,
        body=compressed,
        headers=headers,
        chunks=[compressed[:7], compressed[7:]],
    )

    assert sent[0]["status"] == 204
    assert observed[0][1]["body"] == logical
    delivered_headers = dict(observed[0][0]["headers"])
    assert delivered_headers[b"content-length"] == str(len(logical)).encode()
    assert b"content-encoding" not in delivered_headers
    assert b"x-leadpoet-weight-transport" not in delivered_headers
    assert b"x-leadpoet-weight-transport-signature" not in delivered_headers


@pytest.mark.asyncio
async def test_authenticated_gzip_uses_real_sr25519_signature(
    monkeypatch,
) -> None:
    from bittensor import Keypair

    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    logical = b'{"value":"exact"}'
    compressed = gzip.compress(logical, mtime=0)
    authorization, headers = _transport_headers(
        wire_body=compressed,
        logical_body=logical,
        hotkey=keypair.ss58_address,
        signature="0" * 128,
    )
    signature = bytes(
        keypair.sign(
            weight_transport_authorization_message_v2(
                authorization
            ).encode("utf-8")
        )
    ).hex()
    headers[-1] = (
        b"x-leadpoet-weight-transport-signature",
        signature.encode("ascii"),
    )
    monkeypatch.setenv(
        "PRIMARY_VALIDATOR_HOTKEYS",
        keypair.ss58_address,
    )
    middleware = BodySizeLimitMiddleware(
        None,
        max_body_bytes=len(compressed) + 1,
        max_weight_body_bytes=len(logical) + 1,
    )

    observed, sent = await _invoke(
        middleware,
        body=compressed,
        headers=headers,
    )

    assert sent[0]["status"] == 204
    assert observed[0][1]["body"] == logical


@pytest.mark.asyncio
async def test_unauthenticated_or_duplicate_gzip_is_rejected(
    monkeypatch,
) -> None:
    logical = b'{"value":"exact"}'
    compressed = gzip.compress(logical, mtime=0)
    _authorization, valid_headers = _transport_headers(
        wire_body=compressed,
        logical_body=logical,
    )
    monkeypatch.setenv("PRIMARY_VALIDATOR_HOTKEYS", HOTKEY)
    monkeypatch.setattr(
        "gateway.middleware.body_size._verify_transport_signature",
        lambda **_kwargs: True,
    )
    middleware = BodySizeLimitMiddleware(
        None,
        max_body_bytes=len(compressed) + 1,
    )

    observed, sent = await _invoke(
        middleware,
        body=compressed,
        headers=[
            (b"content-encoding", b"gzip"),
            (b"content-length", str(len(compressed)).encode()),
        ],
    )
    assert observed == []
    assert sent[0]["status"] == 401

    observed, sent = await _invoke(
        middleware,
        body=compressed,
        headers=valid_headers + [valid_headers[-1]],
    )
    assert observed == []
    assert sent[0]["status"] == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tamper", "status"),
    [
        ("wire", 400),
        ("logical", 400),
        ("hotkey", 401),
        ("method", 401),
    ],
)
async def test_authenticated_gzip_fails_closed_on_tampering(
    monkeypatch,
    tamper,
    status,
) -> None:
    logical = b'{"value":"exact"}'
    compressed = gzip.compress(logical, mtime=0)
    authorization, headers = _transport_headers(
        wire_body=compressed,
        logical_body=(
            b'{"value":"different"}' if tamper == "logical" else logical
        ),
        hotkey=(
            "5GcFM97at7gaatFieL1qBHXs6fCD8Xqui3nwmdaZUaUoYAAE"
            if tamper == "hotkey"
            else HOTKEY
        ),
    )
    assert authorization
    monkeypatch.setenv("PRIMARY_VALIDATOR_HOTKEYS", HOTKEY)
    monkeypatch.setattr(
        "gateway.middleware.body_size._verify_transport_signature",
        lambda **_kwargs: True,
    )
    middleware = BodySizeLimitMiddleware(
        None,
        max_body_bytes=len(compressed) + 1,
        max_weight_body_bytes=1024,
    )

    observed, sent = await _invoke(
        middleware,
        body=(
            compressed[:-1] + bytes([compressed[-1] ^ 1])
            if tamper == "wire"
            else compressed
        ),
        headers=headers,
        method="GET" if tamper == "method" else "POST",
    )

    assert observed == []
    assert sent[0]["status"] == status


@pytest.mark.asyncio
async def test_authenticated_gzip_rejects_expansion_over_signed_limit(
    monkeypatch,
) -> None:
    actual_logical = b"x" * 4096
    claimed_logical = b"x" * 32
    compressed = gzip.compress(actual_logical, mtime=0)
    _authorization, headers = _transport_headers(
        wire_body=compressed,
        logical_body=claimed_logical,
    )
    monkeypatch.setenv("PRIMARY_VALIDATOR_HOTKEYS", HOTKEY)
    monkeypatch.setattr(
        "gateway.middleware.body_size._verify_transport_signature",
        lambda **_kwargs: True,
    )
    middleware = BodySizeLimitMiddleware(
        None,
        max_body_bytes=len(compressed) + 1,
        max_weight_body_bytes=64,
    )

    observed, sent = await _invoke(
        middleware,
        body=compressed,
        headers=headers,
    )

    assert observed == []
    assert sent[0]["status"] == 400


@pytest.mark.asyncio
async def test_compression_is_rejected_outside_exact_weight_paths() -> None:
    logical = b'{"value":"exact"}'
    compressed = gzip.compress(logical, mtime=0)
    _authorization, headers = _transport_headers(
        wire_body=compressed,
        logical_body=logical,
    )
    middleware = BodySizeLimitMiddleware(
        None,
        max_body_bytes=len(compressed) + 1,
    )

    observed, sent = await _invoke(
        middleware,
        body=compressed,
        headers=headers,
        path="/health",
    )

    assert observed == []
    assert sent[0]["status"] == 415


@pytest.mark.asyncio
async def test_uncompressed_body_behavior_is_unchanged() -> None:
    body = b'{"value":"exact"}'
    middleware = BodySizeLimitMiddleware(None, max_body_bytes=len(body) + 1)

    observed, sent = await _invoke(
        middleware,
        body=body,
        headers=[
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    )

    assert sent[0]["status"] == 204
    assert observed[0][1]["body"] == body


@pytest.mark.asyncio
async def test_streamed_overflow_emits_only_one_response() -> None:
    sent = []

    async def app(_scope, receive, send):
        message = await receive()
        assert message["type"] == "http.disconnect"
        await send(
            {"type": "http.response.start", "status": 500, "headers": []}
        )
        await send({"type": "http.response.body", "body": b"error"})

    messages = [
        {
            "type": "http.request",
            "body": b"x" * 11,
            "more_body": False,
        }
    ]

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    middleware = BodySizeLimitMiddleware(app, max_body_bytes=10)
    await middleware(
        {"type": "http", "method": "POST", "path": "/submit", "headers": []},
        receive,
        send,
    )

    statuses = [
        message["status"]
        for message in sent
        if message["type"] == "http.response.start"
    ]
    assert statuses == [413]
