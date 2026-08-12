"""Explicit framing for the measured artifact egress tunnel.

Only the enclave-to-parent hop is framed. TLS remains end to end between the
enclave client and the public origin. Directional EOF and its acknowledgement
are explicit records, so neither peer closes AF_VSOCK until both terminal
records have been consumed.
"""

from __future__ import annotations

import errno
import select
import socket
import threading
import time
from typing import Any, Callable, Dict, Optional


TUNNEL_FRAMING_HEADER = "x-leadpoet-egress-tunnel-framing"
TUNNEL_FRAMING_MODE = "length-v2"
TUNNEL_FRAME_BYTES = 64 * 1024
_TUNNEL_EOF_ACK_SIZE = (1 << 32) - 1
_TUNNEL_CLOSE_SIZE = (1 << 32) - 2
_TUNNEL_CLOSE_ACK_SIZE = (1 << 32) - 3
_TUNNEL_EOF_ACK = object()
_TUNNEL_CLOSE = object()
_TUNNEL_CLOSE_ACK = object()
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


def _send_tunnel_eof_ack(connection: Any) -> None:
    connection.sendall(_TUNNEL_EOF_ACK_SIZE.to_bytes(4, byteorder="big"))


def _send_tunnel_control(connection: Any, size: int) -> None:
    connection.sendall(size.to_bytes(4, byteorder="big"))


def receive_tunnel_frame(
    connection: Any,
    *,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """Receive one complete data or terminal-control record."""

    prefix = _receive_exact_until(
        connection,
        4,
        deadline=deadline,
        clock=clock,
    )
    size = int.from_bytes(prefix, byteorder="big")
    if size == 0:
        return None
    if size == _TUNNEL_EOF_ACK_SIZE:
        return _TUNNEL_EOF_ACK
    if size == _TUNNEL_CLOSE_SIZE:
        return _TUNNEL_CLOSE
    if size == _TUNNEL_CLOSE_ACK_SIZE:
        return _TUNNEL_CLOSE_ACK
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
    terminal_initiator: bool,
    clock: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    """Relay a raw full-duplex stream over a length-framed full-duplex stream."""

    if not isinstance(terminal_initiator, bool):
        raise EgressTunnelFramingError("egress tunnel terminal role is invalid")
    transferred = {raw: 0, framed: 0}
    state_lock = threading.Lock()
    framed_write_lock = threading.Lock()
    abort = threading.Event()
    stop_raw_read = threading.Event()
    own_eof_sent = threading.Event()
    own_eof_acked = threading.Event()
    peer_eof_received = threading.Event()
    first_closed = [""]
    write_closed = [""]
    last_activity = [clock()]
    errors: list[BaseException] = []

    def touch() -> None:
        with state_lock:
            last_activity[0] = clock()

    def mark_first_closed(label: str) -> None:
        with state_lock:
            if not first_closed[0]:
                first_closed[0] = label

    def record_error(exc: BaseException) -> None:
        with state_lock:
            if not errors:
                errors.append(exc)
        abort.set()

    def send_frame(payload: Optional[bytes]) -> None:
        with framed_write_lock:
            send_tunnel_frame(framed, payload)

    def send_eof_ack() -> None:
        with framed_write_lock:
            _send_tunnel_eof_ack(framed)

    def relay_raw_to_framed() -> None:
        try:
            while not abort.is_set() and not stop_raw_read.is_set():
                remaining = idle_timeout_seconds - (clock() - last_activity[0])
                if remaining <= 0:
                    raise EgressTunnelFramingError("egress tunnel idle timeout")
                readable, _writable, _exceptional = select.select(
                    [raw], [], [], min(1.0, remaining)
                )
                if not readable:
                    continue
                try:
                    payload = raw.recv(TUNNEL_FRAME_BYTES)
                except OSError as exc:
                    if int(getattr(exc, "errno", 0) or 0) not in _PEER_CLOSE_ERRNOS:
                        raise
                    payload = b""
                if not payload:
                    break
                next_total = transferred[raw] + len(payload)
                if next_total > max_bytes_per_direction:
                    raise EgressTunnelFramingError(
                        "egress tunnel byte limit exceeded"
                    )
                send_frame(payload)
                transferred[raw] = next_total
                touch()
            if not abort.is_set():
                mark_first_closed(raw_label)
                own_eof_sent.set()
                send_frame(None)
                touch()
        except BaseException as exc:  # noqa: BLE001 - relay failure must unblock its peer
            record_error(exc)

    def relay_framed_to_raw() -> None:
        raw_write_available = True
        try:
            while not abort.is_set() and not (
                own_eof_acked.is_set() and peer_eof_received.is_set()
            ):
                remaining = idle_timeout_seconds - (clock() - last_activity[0])
                if remaining <= 0:
                    raise EgressTunnelFramingError("egress tunnel idle timeout")
                payload = receive_tunnel_frame(
                    framed,
                    deadline=clock() + remaining,
                    clock=clock,
                )
                if payload is _TUNNEL_EOF_ACK:
                    if not own_eof_sent.is_set() or own_eof_acked.is_set():
                        raise EgressTunnelFramingError(
                            "egress tunnel EOF acknowledgement is invalid"
                        )
                    own_eof_acked.set()
                    touch()
                    continue
                if payload in {_TUNNEL_CLOSE, _TUNNEL_CLOSE_ACK}:
                    raise EgressTunnelFramingError(
                        "egress tunnel close control arrived before stream completion"
                    )
                if payload is None:
                    if peer_eof_received.is_set():
                        raise EgressTunnelFramingError(
                            "egress tunnel EOF frame is duplicated"
                        )
                    mark_first_closed(framed_label)
                    peer_eof_received.set()
                    send_eof_ack()
                    if raw_write_available:
                        try:
                            raw.shutdown(socket.SHUT_WR)
                        except Exception:
                            pass
                    if terminal_initiator:
                        # The authenticated upstream EOF proves that no later
                        # client bytes can reach the provider. A pooled HTTP
                        # client may otherwise remain idle indefinitely after
                        # consuming the complete response, leaving both framed
                        # peers waiting for its directional EOF. Stop that
                        # impossible direction and complete the authenticated
                        # terminal handshake without relaxing response-body or
                        # TLS verification at the application boundary.
                        stop_raw_read.set()
                        try:
                            raw.shutdown(socket.SHUT_RD)
                        except Exception:
                            pass
                    touch()
                    continue
                if peer_eof_received.is_set():
                    raise EgressTunnelFramingError(
                        "egress tunnel data followed its EOF frame"
                    )
                next_total = transferred[framed] + len(payload)
                if next_total > max_bytes_per_direction:
                    raise EgressTunnelFramingError(
                        "egress tunnel byte limit exceeded"
                    )
                if raw_write_available:
                    try:
                        raw.sendall(payload)
                    except OSError as exc:
                        if int(getattr(exc, "errno", 0) or 0) not in _PEER_CLOSE_ERRNOS:
                            raise
                        raw_write_available = False
                        with state_lock:
                            write_closed[0] = raw_label
                        mark_first_closed(raw_label)
                        stop_raw_read.set()
                transferred[framed] = next_total
                touch()
        except BaseException as exc:  # noqa: BLE001 - relay failure must unblock its peer
            record_error(exc)

    threads = (
        threading.Thread(target=relay_raw_to_framed, daemon=True),
        threading.Thread(target=relay_framed_to_raw, daemon=True),
    )
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        for thread in threads:
            thread.join(timeout=0.05)
        with state_lock:
            idle_expired = clock() - last_activity[0] >= idle_timeout_seconds
            failed = bool(errors)
        if failed or idle_expired:
            if idle_expired and not failed:
                record_error(EgressTunnelFramingError("egress tunnel idle timeout"))
            abort.set()
            try:
                framed.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            break
    for thread in threads:
        thread.join(timeout=1.1)
    if any(thread.is_alive() for thread in threads):
        try:
            raw.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        for thread in threads:
            thread.join(timeout=1.0)
    if any(thread.is_alive() for thread in threads):
        raise EgressTunnelFramingError("egress tunnel relay did not stop")
    if errors:
        raise errors[0]
    if not (
        own_eof_sent.is_set()
        and own_eof_acked.is_set()
        and peer_eof_received.is_set()
    ):
        raise EgressTunnelFramingError("egress tunnel terminal state is incomplete")

    deadline = clock() + idle_timeout_seconds
    if terminal_initiator:
        _send_tunnel_control(framed, _TUNNEL_CLOSE_SIZE)
        if receive_tunnel_frame(framed, deadline=deadline, clock=clock) is not _TUNNEL_CLOSE_ACK:
            raise EgressTunnelFramingError(
                "egress tunnel close acknowledgement is invalid"
            )
        # The responder deliberately waits for this physical close. It proves
        # that its final acknowledgement reached the initiator before either
        # endpoint releases the AF_VSOCK tunnel.
        framed.close()
    else:
        if receive_tunnel_frame(framed, deadline=deadline, clock=clock) is not _TUNNEL_CLOSE:
            raise EgressTunnelFramingError("egress tunnel close request is invalid")
        _send_tunnel_control(framed, _TUNNEL_CLOSE_ACK_SIZE)
        while True:
            remaining = deadline - clock()
            if remaining <= 0:
                raise EgressTunnelFramingError(
                    "egress tunnel physical close timed out"
                )
            readable, _writable, _exceptional = select.select(
                [framed], [], [], min(1.0, remaining)
            )
            if not readable:
                continue
            try:
                trailing = framed.recv(1)
            except OSError as exc:
                if int(getattr(exc, "errno", 0) or 0) in _PEER_CLOSE_ERRNOS:
                    trailing = b""
                else:
                    raise
            if trailing:
                raise EgressTunnelFramingError(
                    "egress tunnel data followed its close acknowledgement"
                )
            break
    result = {
        "%s_to_%s_bytes" % (raw_label, framed_label): transferred[raw],
        "%s_to_%s_bytes" % (framed_label, raw_label): transferred[framed],
        "first_closed": first_closed[0] or "unknown",
        "framing": TUNNEL_FRAMING_MODE,
    }
    if write_closed[0]:
        result["write_closed"] = write_closed[0]
    return result
