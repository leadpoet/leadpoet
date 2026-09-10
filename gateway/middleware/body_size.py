"""Bound request bodies before gateway API handlers consume them."""
from __future__ import annotations
import os
from typing import Awaitable, Callable


class BodySizeLimitMiddleware:
    def __init__(
        self,
        app,
        max_body_bytes: int | None = None,
    ) -> None:
        self.app = app
        self.max_body_bytes = int(
            max_body_bytes
            if max_body_bytes is not None
            else os.getenv("GATEWAY_MAX_BODY_BYTES", "10485760")
        )

    async def __call__(self, scope, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        encoding = headers.get(b"content-encoding", b"").strip().lower()
        if encoding and encoding != b"identity":
            await self._reject(send, status=415, detail="Unsupported request content encoding")
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
        rejected = False

        async def guarded_send(message):
            # Once the middleware has emitted the 413, discard a downstream
            # error response triggered by the synthetic disconnect.
            if rejected:
                return
            await send(message)

        async def limited_receive():
            nonlocal consumed, rejected
            message = await receive()
            if message.get("type") == "http.request":
                consumed += len(message.get("body") or b"")
                if consumed > self.max_body_bytes:
                    if not rejected:
                        await self._reject(send)
                        rejected = True
                    return {
                        "type": "http.disconnect",
                    }
            return message

        await self.app(scope, limited_receive, guarded_send)

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
