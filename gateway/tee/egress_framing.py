"""Explicit framing for the measured artifact egress tunnel.

Only the enclave-to-parent hop is framed. TLS remains end to end between the
enclave client and the public origin, while an explicit zero-length frame
represents directional EOF without depending on AF_VSOCK shutdown ordering.
"""

from __future__ import annotations

import errno
import select
import socket
import time
from typing import Any, Callable, Dict, Optional


TUNNEL_FRAMING_HEADER = "x-leadpoet-egress-tunnel-framing"
TUNNEL_FRAMING_MODE = "length-v1"
TUNNEL_FRAME_BYTES = 64 * 1024
_PEER_CLOSE_ERRNOS = frozenset(
    value
    for value in (
        errno.EPIPE,
        errno.ECONNRESET,
        errno.ENOTCONN,
        getattr(errno, "ESHUTDOWN", None),
    )
    if value is not None
)


class EgressTunnelFramingError(RuntimeError):
    """The framed enclave-to-parent tunnel violated its wire contract."""


def _receive_exact_until(
    connection: Any,
    size: int,
    *,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        remaining = deadline - clock()
        if remaining <= 0:
            raise EgressTunnelFramingError("egress tunnel frame timed out")
        readable, _writable, _exceptional = select.select(
            [connection], [], [], min(1.0, remaining)
        )
        if not readable:
            continue
        try:
            chunk = connection.recv(size - len(chunks))
        except OSError as exc:
            if int(getattr(exc, "errno", 0) or 0) in _PEER_CLOSE_ERRNOS:
                chunk = b""
            else:
                raise
        if not chunk:
            raise EgressTunnelFramingError(
                "egress tunnel closed before an explicit EOF frame"
            )
        chunks.extend(chunk)
    return bytes(chunks)


def send_tunnel_frame(connection: Any, payload: Optional[bytes]) -> None:
    """Send one data frame, or an explicit directional EOF for ``None``."""

    if payload is None:
        connection.sendall((0).to_bytes(4, byteorder="big"))
        return
    if not isinstance(payload, bytes) or not 1 <= len(payload) <= TUNNEL_FRAME_BYTES:
        raise EgressTunnelFramingError("egress tunnel frame size is invalid")
    connection.sendall(len(payload).to_bytes(4, byteorder="big") + payload)


def receive_tunnel_frame(
    connection: Any,
    *,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> Optional[bytes]:
    """Receive one complete frame; ``None`` is the explicit EOF marker."""

    prefix = _receive_exact_until(
        connection,
        4,
        deadline=deadline,
        clock=clock,
    )
    size = int.from_bytes(prefix, byteorder="big")
    if size == 0:
        return None
    if size > TUNNEL_FRAME_BYTES:
        raise EgressTunnelFramingError("egress tunnel frame exceeds limit")
    return _receive_exact_until(
        connection,
        size,
        deadline=deadline,
        clock=clock,
    )


def relay_raw_and_framed(
    raw: Any,
    framed: Any,
    *,
    idle_timeout_seconds: float,
    max_bytes_per_direction: int,
    raw_label: str,
    framed_label: str,
    clock: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    """Relay a raw full-duplex stream over a length-framed full-duplex stream."""

    active = {raw, framed}
    transferred = {raw: 0, framed: 0}
    first_closed = ""
    write_closed = ""
    raw_write_available = True
    last_activity = clock()
    while active:
        remaining = max(0.0, idle_timeout_seconds - (clock() - last_activity))
        if remaining <= 0:
            raise EgressTunnelFramingError("egress tunnel idle timeout")
        readable, _writable, _exceptional = select.select(
            list(active), [], [], min(1.0, remaining)
        )
        if not readable:
            continue
        for source in readable:
            deadline = clock() + remaining
            if source is raw:
                try:
                    payload = raw.recv(TUNNEL_FRAME_BYTES)
                except OSError as exc:
                    if int(getattr(exc, "errno", 0) or 0) not in _PEER_CLOSE_ERRNOS:
                        raise
                    payload = b""
                if not payload:
                    if not first_closed:
                        first_closed = raw_label
                    active.discard(raw)
                    send_tunnel_frame(framed, None)
                    last_activity = clock()
                    continue
                next_total = transferred[raw] + len(payload)
                if next_total > max_bytes_per_direction:
                    raise EgressTunnelFramingError(
                        "egress tunnel byte limit exceeded"
                    )
                send_tunnel_frame(framed, payload)
                transferred[raw] = next_total
                last_activity = clock()
                continue

            payload = receive_tunnel_frame(
                framed,
                deadline=deadline,
                clock=clock,
            )
            if payload is None:
                if not first_closed:
                    first_closed = framed_label
                active.discard(framed)
                if raw_write_available:
                    try:
                        raw.shutdown(socket.SHUT_WR)
                    except Exception:
                        pass
                last_activity = clock()
                continue
            next_total = transferred[framed] + len(payload)
            if next_total > max_bytes_per_direction:
                raise EgressTunnelFramingError("egress tunnel byte limit exceeded")
            if raw_write_available:
                try:
                    raw.sendall(payload)
                except OSError as exc:
                    if int(getattr(exc, "errno", 0) or 0) not in _PEER_CLOSE_ERRNOS:
                        raise
                    raw_write_available = False
                    write_closed = raw_label
                    if not first_closed:
                        first_closed = raw_label
                    if raw in active:
                        active.discard(raw)
                        send_tunnel_frame(framed, None)
            transferred[framed] = next_total
            last_activity = clock()
    result = {
        "%s_to_%s_bytes" % (raw_label, framed_label): transferred[raw],
        "%s_to_%s_bytes" % (framed_label, raw_label): transferred[framed],
        "first_closed": first_closed or "unknown",
        "framing": TUNNEL_FRAMING_MODE,
    }
    if write_closed:
        result["write_closed"] = write_closed
    return result
