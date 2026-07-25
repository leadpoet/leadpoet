"""ASGI request body size guard."""

from __future__ import annotations

import os
import zlib
from typing import Awaitable, Callable


_COMPRESSED_WEIGHT_PATHS = frozenset(
    {
        "/weights/submit/v2",
        "/weights/finalize/v2",
        "/weights/subnet-epoch/candidate/v1",
        "/weights/subnet-epoch/boundary/v1",
    }
)
_MAX_WEIGHT_BODY_BYTES = 64 * 1024 * 1024


class BodySizeLimitMiddleware:
    def __init__(
        self,
        app,
        max_body_bytes: int | None = None,
        max_weight_body_bytes: int = _MAX_WEIGHT_BODY_BYTES,
    ) -> None:
        self.app = app
        self.max_body_bytes = int(
            max_body_bytes
            if max_body_bytes is not None
            else os.getenv("GATEWAY_MAX_BODY_BYTES", "10485760")
        )
        self.max_weight_body_bytes = int(max_weight_body_bytes)

    async def __call__(self, scope, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.lower(): value
            for key, value in scope.get("headers", [])
        }
        content_encoding = headers.get(b"content-encoding", b"").strip().lower()
        if content_encoding and content_encoding != b"identity":
            if (
                content_encoding != b"gzip"
                or scope.get("path") not in _COMPRESSED_WEIGHT_PATHS
            ):
                await self._reject(
                    send,
                    status=415,
                    detail="Unsupported request content encoding",
                )
                return
            compressed = await self._read_bounded_body(
                receive,
                send,
                limit=self.max_body_bytes,
            )
            if compressed is None:
                return
            try:
                body = self._decompress_gzip(compressed)
            except ValueError:
                await self._reject(
                    send,
                    status=400,
                    detail="Invalid compressed request body",
                )
                return
            rewritten_headers = [
                (key, value)
                for key, value in scope.get("headers", [])
                if key.lower() not in {b"content-encoding", b"content-length"}
            ]
            rewritten_headers.append(
                (b"content-length", str(len(body)).encode("ascii"))
            )
            rewritten_scope = dict(scope)
            rewritten_scope["headers"] = rewritten_headers
            delivered = False

            async def decompressed_receive():
                nonlocal delivered
                if delivered:
                    return {
                        "type": "http.request",
                        "body": b"",
                        "more_body": False,
                    }
                delivered = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }

            await self.app(rewritten_scope, decompressed_receive, send)
            return

        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send)
                return

        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message.get("type") == "http.request":
                consumed += len(message.get("body") or b"")
                if consumed > self.max_body_bytes:
                    await self._reject(send)
                    return {
                        "type": "http.disconnect",
                    }
            return message

        await self.app(scope, limited_receive, send)

    async def _read_bounded_body(
        self,
        receive: Callable,
        send: Callable,
        *,
        limit: int,
    ) -> bytes | None:
        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return None
            if message.get("type") != "http.request":
                continue
            body.extend(message.get("body") or b"")
            if len(body) > limit:
                await self._reject(send)
                return None
            if not message.get("more_body", False):
                return bytes(body)

    def _decompress_gzip(self, body: bytes) -> bytes:
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            decoded = decompressor.decompress(
                body,
                self.max_weight_body_bytes + 1,
            )
        except zlib.error as exc:
            raise ValueError("invalid gzip body") from exc
        if (
            len(decoded) > self.max_weight_body_bytes
            or not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            raise ValueError("invalid gzip body")
        return decoded

    async def _reject(
        self,
        send: Callable[..., Awaitable[None]],
        *,
        status: int = 413,
        detail: str = "Request body too large",
    ) -> None:
        body = ('{"detail":"%s"}' % detail).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})
