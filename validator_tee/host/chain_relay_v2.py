"""Parent-side opaque relay for validator-enclave chain TLS.

The relay can connect only to the measured Finney live or archive endpoint.  It
never sees an HTTP request or response because the TLS session originates and
terminates in the validator enclave.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import select
import socket
import threading
import time
from typing import Any, Callable, Iterable, Optional

from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ARCHIVE_ENDPOINT_HOST,
    CHAIN_ENDPOINT_HOST,
    CHAIN_ENDPOINT_PORT,
    chain_source_policy_hash,
)


AF_VSOCK = 40
VMADDR_CID_ANY = 0xFFFFFFFF
logger = logging.getLogger(__name__)

CHAIN_RELAY_VSOCK_PORT = 5002
CHAIN_RELAY_SUPERVISION_INTERVAL_SECONDS = 15
MAX_CONTROL_BYTES = 16 * 1024
MAX_BYTES_PER_DIRECTION = 32 * 1024 * 1024
RELAY_CHUNK_BYTES = 64 * 1024
CONNECT_TIMEOUT_SECONDS = 15.0
# Bounds the client control handshake so a peer that connects and stalls
# before/while sending its control frame cannot pin a handler thread and its
# accepted fd forever — that leak is what cumulatively exhausts fds and drives
# the accept() failures the loop now survives. _relay keeps its own idle
# timeout, so the connection returns to blocking mode for the data pump.
CONTROL_HANDSHAKE_TIMEOUT_SECONDS = 15.0
IDLE_TIMEOUT_SECONDS = 90.0


class ValidatorChainRelayV2Error(RuntimeError):
    """The parent relay request or destination violates the fixed policy."""


def _recv_exact(connection: Any, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = connection.recv(min(64 * 1024, size - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


def _send_control(connection: Any, value: Any) -> None:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if len(encoded) < 2 or len(encoded) > MAX_CONTROL_BYTES:
        raise ValidatorChainRelayV2Error("relay response is outside size limit")
    connection.sendall(len(encoded).to_bytes(4, "big") + encoded)


def _read_control(connection: Any) -> dict:
    prefix = _recv_exact(connection, 4)
    if len(prefix) != 4:
        raise ValidatorChainRelayV2Error("relay control prefix is incomplete")
    size = int.from_bytes(prefix, "big")
    if size < 2 or size > MAX_CONTROL_BYTES:
        raise ValidatorChainRelayV2Error("relay control size is invalid")
    body = _recv_exact(connection, size)
    if len(body) != size:
        raise ValidatorChainRelayV2Error("relay control body is incomplete")
    try:
        value = json.loads(body.decode("ascii"))
    except Exception as exc:
        raise ValidatorChainRelayV2Error("relay control JSON is invalid") from exc
    if not isinstance(value, dict):
        raise ValidatorChainRelayV2Error("relay control must be an object")
    return value


def _validate_control(value: Any) -> str:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "host",
        "port",
        "policy_hash",
    }:
        raise ValidatorChainRelayV2Error("relay control fields are invalid")
    if value.get("schema_version") != "leadpoet.validator_chain_relay.v2":
        raise ValidatorChainRelayV2Error("relay schema is invalid")
    destination_host = value.get("host")
    if (
        destination_host
        not in (CHAIN_ENDPOINT_HOST, CHAIN_ARCHIVE_ENDPOINT_HOST)
        or value.get("port") != CHAIN_ENDPOINT_PORT
    ):
        raise ValidatorChainRelayV2Error("relay destination is not the measured chain")
    if value.get("policy_hash") != chain_source_policy_hash():
        raise ValidatorChainRelayV2Error("relay policy hash differs")
    return str(destination_host)


def _global_addresses(
    destination_host: str,
    resolver: Callable[..., Iterable[Any]] = socket.getaddrinfo,
) -> list:
    if destination_host not in (CHAIN_ENDPOINT_HOST, CHAIN_ARCHIVE_ENDPOINT_HOST):
        raise ValidatorChainRelayV2Error("chain destination is not measured")
    try:
        entries = resolver(
            destination_host,
            CHAIN_ENDPOINT_PORT,
            type=socket.SOCK_STREAM,
        )
    except Exception as exc:
        raise ValidatorChainRelayV2Error("chain DNS resolution failed") from exc
    result = []
    for family, socktype, protocol, _canonical, address in entries:
        if socktype != socket.SOCK_STREAM or not address:
            continue
        try:
            parsed = ipaddress.ip_address(str(address[0]))
        except ValueError as exc:
            raise ValidatorChainRelayV2Error("chain DNS address is invalid") from exc
        if not parsed.is_global:
            raise ValidatorChainRelayV2Error("chain DNS returned a non-global address")
        result.append((family, socktype, protocol, address))
    if not result:
        raise ValidatorChainRelayV2Error("chain endpoint has no global address")
    return result


def _connect_chain(
    destination_host: str,
    *,
    resolver: Callable[..., Iterable[Any]] = socket.getaddrinfo,
    socket_factory: Callable[..., Any] = socket.socket,
) -> Any:
    last_error = None
    for family, socktype, protocol, address in _global_addresses(
        destination_host,
        resolver,
    ):
        connection = socket_factory(family, socktype, protocol)
        try:
            connection.settimeout(CONNECT_TIMEOUT_SECONDS)
            connection.connect(address)
            connection.settimeout(None)
            return connection
        except Exception as exc:
            last_error = exc
            connection.close()
    raise ValidatorChainRelayV2Error("chain connection failed") from last_error


def _relay(left: Any, right: Any) -> None:
    peers = {left: right, right: left}
    active = {left, right}
    counts = {left: 0, right: 0}
    last_activity = time.monotonic()
    while active:
        remaining = IDLE_TIMEOUT_SECONDS - (time.monotonic() - last_activity)
        if remaining <= 0:
            raise ValidatorChainRelayV2Error("chain relay idle timeout")
        readable, _writable, _exceptional = select.select(
            list(active), [], [], min(1.0, remaining)
        )
        for source in readable:
            data = source.recv(RELAY_CHUNK_BYTES)
            target = peers[source]
            if not data:
                active.discard(source)
                try:
                    target.shutdown(socket.SHUT_WR)
                except Exception:
                    pass
                continue
            counts[source] += len(data)
            if counts[source] > MAX_BYTES_PER_DIRECTION:
                raise ValidatorChainRelayV2Error("chain relay byte limit exceeded")
            target.sendall(data)
            last_activity = time.monotonic()


def handle_chain_relay_connection(
    connection: Any, *, connector: Callable[[str], Any] = _connect_chain
) -> None:
    upstream = None
    try:
        connection.settimeout(CONTROL_HANDSHAKE_TIMEOUT_SECONDS)
        request = _read_control(connection)
        destination_host = _validate_control(request)
        upstream = connector(destination_host)
        _send_control(
            connection,
            {
                "status": "connected",
                "policy_hash": chain_source_policy_hash(),
            },
        )
        # Handshake done; the data pump is bounded by _relay's own idle
        # timeout, so restore blocking mode for it.
        connection.settimeout(None)
        _relay(connection, upstream)
    finally:
        for candidate in (upstream, connection):
            if candidate is not None:
                try:
                    candidate.close()
                except Exception:
                    pass


class ValidatorChainRelayV2:
    def __init__(
        self,
        *,
        port: int = CHAIN_RELAY_VSOCK_PORT,
        socket_factory: Callable[..., Any] = socket.socket,
        connector: Callable[[str], Any] = _connect_chain,
    ) -> None:
        self.port = int(port)
        self._socket_factory = socket_factory
        self._connector = connector
        self._listener = None  # type: Optional[Any]
        self._thread = None  # type: Optional[threading.Thread]
        self._stop = threading.Event()

    def start(self) -> dict:
        if self._thread is not None and self._thread.is_alive():
            return self.status()
        # Recovery-safe (re)start: close any stale listener first so a restart
        # after an unexpected loop-thread death can rebind the port instead of
        # failing address-in-use, and clear the stop flag so the fresh loop
        # runs (supervision in main() relies on this).
        if self._listener is not None:
            try:
                self._listener.close()
            except Exception:
                pass
            self._listener = None
        self._stop.clear()
        listener = self._socket_factory(AF_VSOCK, socket.SOCK_STREAM)
        listener.bind((VMADDR_CID_ANY, self.port))
        listener.listen(8)
        self._listener = listener
        self._thread = threading.Thread(
            target=self._accept_loop,
            name="validator-chain-relay-v2",
            daemon=True,
        )
        self._thread.start()
        return self.status()

    def status(self) -> dict:
        return {
            "status": "running"
            if self._thread is not None and self._thread.is_alive()
            else "stopped",
            "port": self.port,
            "policy_hash": chain_source_policy_hash(),
        }

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
        self._listener = None

    def _accept_loop(self) -> None:
        # A transient accept() failure (EMFILE under fd pressure, ECONNABORTED,
        # ENOMEM) must not permanently kill this loop: the relay is the
        # enclave's only chain-RPC path, so a dead accept loop silently stops
        # all weight-extrinsic signing with the host process still looking
        # healthy (main() never relaunches it). Only exit when stop() has been
        # requested; otherwise log and keep serving after a short pause so a
        # persistently broken listener cannot hot-spin.
        consecutive_errors = 0
        while not self._stop.is_set():
            try:
                connection, _address = self._listener.accept()
            except Exception as exc:
                if self._stop.is_set():
                    return
                consecutive_errors += 1
                # Log the first failure and then only periodically, so a
                # sustained fd/socket outage cannot spam the log at the retry
                # cadence while still leaving one clear, greppable signal plus
                # the recovery below.
                if consecutive_errors == 1 or consecutive_errors % 60 == 0:
                    logger.warning(
                        "chain_relay_accept_error type=%s error=%s "
                        "consecutive=%d; continuing to serve chain RPC",
                        type(exc).__name__,
                        str(exc)[:200],
                        consecutive_errors,
                    )
                # Interruptible: a stop() during the backoff exits immediately.
                self._stop.wait(0.5)
                continue
            if consecutive_errors:
                logger.info(
                    "chain_relay_accept_recovered after=%d errors",
                    consecutive_errors,
                )
                consecutive_errors = 0
            # Spawning the handler must not be able to kill the accept loop
            # either (thread exhaustion raises from start()): on failure close
            # the connection and keep serving.
            try:
                threading.Thread(
                    target=handle_chain_relay_connection,
                    args=(connection,),
                    kwargs={"connector": self._connector},
                    daemon=True,
                ).start()
            except Exception as exc:
                logger.warning(
                    "chain_relay_handler_spawn_failed type=%s error=%s",
                    type(exc).__name__,
                    str(exc)[:200],
                )
                try:
                    connection.close()
                except Exception:
                    pass


def main() -> None:
    relay = ValidatorChainRelayV2()
    print(json.dumps(relay.start(), sort_keys=True), flush=True)
    # Supervise: if the accept-loop thread ever dies for a reason the loop
    # itself did not handle, restart it instead of leaving the enclave with no
    # chain-RPC path (and thus no weight signing) while the process looks
    # healthy. A clean stop() is respected. start() is recovery-safe.
    while True:
        time.sleep(CHAIN_RELAY_SUPERVISION_INTERVAL_SECONDS)
        if relay._stop.is_set():
            continue
        if relay.status().get("status") != "running":
            logger.error(
                "chain_relay_thread_dead; restarting the validator chain relay"
            )
            try:
                relay.start()
            except Exception as exc:
                logger.error(
                    "chain_relay_restart_failed type=%s error=%s",
                    type(exc).__name__,
                    str(exc)[:200],
                )


if __name__ == "__main__":
    main()
