"""ASGI request body size guard."""

from __future__ import annotations

import logging
import os
from typing import Awaitable, Callable


logger = logging.getLogger(__name__)

DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024
DEFAULT_WEIGHT_AUTHORITY_MAX_BODY_BYTES = 32 * 1024 * 1024

# These POST endpoints carry complete attested receipt graphs.  Keep the larger
# allowance exact so it cannot silently broaden ordinary unauthenticated
# gateway ingress.  PriorityMiddleware separately bounds their concurrency.
WEIGHT_AUTHORITY_GRAPH_PATHS = frozenset(
    {
        "/weights/subnet-epoch/candidate/v1",
        "/weights/subnet-epoch/boundary/v1",
        "/weights/submit/v2",
        "/weights/finalize/v2",
    }
)


class _RequestBodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    def __init__(
        self,
        app,
        max_body_bytes: int | None = None,
        max_weight_authority_body_bytes: int | None = None,
    ) -> None:
        self.app = app
        self.max_body_bytes = int(
            max_body_bytes
            if max_body_bytes is not None
            else os.getenv("GATEWAY_MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES))
        )
        self.max_weight_authority_body_bytes = int(
            max_weight_authority_body_bytes
            if max_weight_authority_body_bytes is not None
            else os.getenv(
                "GATEWAY_WEIGHT_AUTHORITY_MAX_BODY_BYTES",
                str(DEFAULT_WEIGHT_AUTHORITY_MAX_BODY_BYTES),
            )
        )
        if self.max_body_bytes <= 0:
            raise ValueError("GATEWAY_MAX_BODY_BYTES must be positive")
        if self.max_weight_authority_body_bytes <= 0:
            raise ValueError(
                "GATEWAY_WEIGHT_AUTHORITY_MAX_BODY_BYTES must be positive"
            )

    async def __call__(self, scope, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        max_body_bytes = self._max_body_bytes_for_scope(scope)
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                declared_bytes = int(content_length)
                if declared_bytes < 0 or declared_bytes > max_body_bytes:
                    await self._reject(
                        scope=scope,
                        send=send,
                        max_body_bytes=max_body_bytes,
                        observed_bytes=declared_bytes,
                        reason="content_length",
                    )
                    return
            except ValueError:
                await self._reject(
                    scope=scope,
                    send=send,
                    max_body_bytes=max_body_bytes,
                    observed_bytes=None,
                    reason="invalid_content_length",
                )
                return

        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message.get("type") == "http.request":
                consumed += len(message.get("body") or b"")
                if consumed > max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await self._reject(
                scope=scope,
                send=send,
                max_body_bytes=max_body_bytes,
                observed_bytes=consumed,
                reason="streamed_body",
            )

    def _max_body_bytes_for_scope(self, scope) -> int:
        if (
            str(scope.get("method") or "").upper() == "POST"
            and scope.get("path") in WEIGHT_AUTHORITY_GRAPH_PATHS
        ):
            return self.max_weight_authority_body_bytes
        return self.max_body_bytes

    async def _reject(
        self,
        *,
        scope,
        send: Callable[..., Awaitable[None]],
        max_body_bytes: int,
        observed_bytes: int | None,
        reason: str,
    ) -> None:
        logger.warning(
            "gateway_request_body_rejected tag=gateway_body_limit method=%s "
            "path=%s reason=%s observed_bytes=%s max_body_bytes=%s",
            scope.get("method", ""),
            scope.get("path", ""),
            reason,
            observed_bytes,
            max_body_bytes,
        )
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
