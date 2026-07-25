"""ASGI request body size guard."""

from __future__ import annotations

import os
from typing import Awaitable, Callable


class BodySizeLimitMiddleware:
    def __init__(self, app, max_body_bytes: int | None = None) -> None:
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

        headers = {
            key.lower(): value
            for key, value in scope.get("headers", [])
        }
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

    async def _reject(self, send: Callable[..., Awaitable[None]]) -> None:
        body = b'{"detail":"Request body too large"}'
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})
