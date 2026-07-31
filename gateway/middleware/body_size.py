"""ASGI request body size guard."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import zlib
from typing import Awaitable, Callable

from leadpoet_canonical.attested_v2 import sha256_bytes
from leadpoet_canonical.hotkey_authority_v2 import (
    MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES,
    WEIGHT_TRANSPORT_PATHS,
    validate_weight_transport_authorization_v2,
    weight_transport_authorization_message_v2,
)


_TRANSPORT_AUTHORIZATION_HEADER = b"x-leadpoet-weight-transport"
_TRANSPORT_SIGNATURE_HEADER = b"x-leadpoet-weight-transport-signature"
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")


def _verify_transport_signature(
    *,
    validator_hotkey: str,
    signature: str,
    message: bytes,
) -> bool:
    try:
        from bittensor import Keypair

        return bool(
            Keypair(ss58_address=validator_hotkey).verify(
                message,
                bytes.fromhex(signature),
            )
        )
    except Exception:
        return False


class BodySizeLimitMiddleware:
    def __init__(
        self,
        app,
        max_body_bytes: int | None = None,
        max_weight_body_bytes: int = MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES,
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

        header_items = list(scope.get("headers", []))
        headers = {key.lower(): value for key, value in header_items}
        content_encoding = headers.get(b"content-encoding", b"").strip().lower()
        path = str(scope.get("path") or "")
        if content_encoding and content_encoding != b"identity":
            if content_encoding != b"gzip" or path not in WEIGHT_TRANSPORT_PATHS:
                await self._reject(
                    send,
                    status=415,
                    detail="Unsupported request content encoding",
                )
                return
            if (
                sum(
                    1
                    for key, _value in header_items
                    if key.lower() == _TRANSPORT_AUTHORIZATION_HEADER
                )
                != 1
                or sum(
                    1
                    for key, _value in header_items
                    if key.lower() == _TRANSPORT_SIGNATURE_HEADER
                )
                != 1
            ):
                await self._reject(
                    send,
                    status=401,
                    detail="Compressed weight transport is not authorized",
                )
                return
            # The authorization check imports bittensor and runs an sr25519
            # verify; the body verification hashes up to 10 MiB compressed +
            # 128 MiB logical and inflates the gzip. All of that is synchronous
            # CPU that would otherwise stall the single gateway event loop
            # (every miner and validator request) for ~100ms+ right at the
            # weight-submission window. Offload to a worker thread; the check
            # order and every fail-closed rejection are unchanged.
            authorization = await asyncio.to_thread(
                self._transport_authorization,
                scope=scope,
                headers=headers,
            )
            if authorization is None:
                await self._reject(
                    send,
                    status=401,
                    detail="Compressed weight transport is not authorized",
                )
                return
            compressed = await self._read_bounded_body(
                receive,
                send,
                limit=self.max_body_bytes,
            )
            if compressed is None:
                return
            body, verify_error = await asyncio.to_thread(
                self._verify_wire_and_decompress,
                compressed,
                authorization,
            )
            if body is None:
                await self._reject(
                    send,
                    status=400,
                    detail=verify_error or "Invalid compressed request body",
                )
                return
            rewritten_headers = [
                (key, value)
                for key, value in header_items
                if key.lower()
                not in {
                    b"content-encoding",
                    b"content-length",
                    _TRANSPORT_AUTHORIZATION_HEADER,
                    _TRANSPORT_SIGNATURE_HEADER,
                }
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
        if (
            _TRANSPORT_AUTHORIZATION_HEADER in headers
            or _TRANSPORT_SIGNATURE_HEADER in headers
        ):
            await self._reject(
                send,
                status=400,
                detail="Weight transport authorization requires gzip",
            )
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

    def _transport_authorization(self, *, scope, headers):
        try:
            encoded = headers[_TRANSPORT_AUTHORIZATION_HEADER]
            if len(encoded) > 4096:
                return None
            document = json.loads(
                base64.b64decode(encoded, validate=True).decode("utf-8")
            )
            authorization = validate_weight_transport_authorization_v2(
                document
            )
            signature = headers[_TRANSPORT_SIGNATURE_HEADER].decode(
                "ascii"
            ).lower()
            allowed_hotkeys = {
                item.strip()
                for item in os.getenv(
                    "PRIMARY_VALIDATOR_HOTKEYS", ""
                ).split(",")
                if item.strip()
            }
            if (
                str(scope.get("method") or "").upper() != "POST"
                or authorization["method"] != "POST"
                or authorization["path"] != str(scope.get("path") or "")
                or authorization["content_encoding"] != "gzip"
                or authorization["validator_hotkey"] not in allowed_hotkeys
                or authorization["logical_body_bytes"]
                > self.max_weight_body_bytes
                or not _SIGNATURE_RE.fullmatch(signature)
            ):
                return None
            message = weight_transport_authorization_message_v2(
                authorization
            ).encode("utf-8")
            if not _verify_transport_signature(
                validator_hotkey=authorization["validator_hotkey"],
                signature=signature,
                message=message,
            ):
                return None
            return authorization
        except Exception:
            return None

    async def _read_bounded_body(
        self,
        receive: Callable,
        send: Callable,
        *,
        limit: int,
    ):
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

    def _verify_wire_and_decompress(
        self,
        compressed: bytes,
        authorization,
    ):
        """Runs in a worker thread: byte-count/hash checks and the bounded
        gzip inflate, in the exact order the inline code used. Returns
        (logical_body, None) on success or (None, rejection_detail)."""
        if (
            len(compressed) != authorization["wire_body_bytes"]
            or sha256_bytes(compressed) != authorization["wire_body_hash"]
        ):
            return None, "Compressed weight transport body differs"
        try:
            body = self._decompress_gzip(compressed)
        except ValueError:
            return None, "Invalid compressed request body"
        if (
            len(body) != authorization["logical_body_bytes"]
            or sha256_bytes(body) != authorization["logical_body_hash"]
        ):
            return None, "Compressed weight transport payload differs"
        return body, None

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
