"""Opaque parent relay for mutually attested gateway-enclave TLS channels."""

from __future__ import annotations

import argparse
import errno
import json
import re
import select
import socket
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from gateway.tee.topology import ROLE_SPECS, topology_hash
from gateway.utils.tee_client import AF_VSOCK, _recv_exact


VMADDR_CID_ANY = 0xFFFFFFFF
DEFAULT_RELAY_PORT = 5002
TARGET_TLS_PORT = 5003
MAX_CONTROL_BYTES = 16 * 1024
MAX_CHANNEL_BYTES_PER_DIRECTION = 512 * 1024 * 1024
RELAY_CHUNK_BYTES = 64 * 1024
IDLE_TIMEOUT_SECONDS = 1800.0
RELAY_SUPERVISOR_POLL_SECONDS = 0.25
RELAY_TRANSPORT_HEALTH_SCHEMA_VERSION = (
    "leadpoet.inter_enclave_relay_transport_health.v2"
)
MAX_RELAY_CLEANUP_EVENT_COUNT = (1 << 63) - 1
RELAY_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE = 1
MAX_RELAY_CLEANUP_ATTEMPT_COUNT = (1 << 63) - 1
_CHANNEL_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_TRANSIENT_ACCEPT_ERRNOS = frozenset(
    value
    for value in (
        errno.EINTR,
        errno.ECONNABORTED,
        getattr(errno, "EPROTO", None),
    )
    if value is not None
)

_CID_BY_ROLE = {role: int(spec["cid"]) for role, spec in ROLE_SPECS.items()}
_COORDINATOR_CID = _CID_BY_ROLE["gateway_coordinator"]
_APPROVED_PAIRS = frozenset(
    {
        (_COORDINATOR_CID, _CID_BY_ROLE["gateway_scoring"]),
        (_CID_BY_ROLE["gateway_scoring"], _COORDINATOR_CID),
    }
)
_RELAY_TRANSPORT_HEALTH_LOCK = threading.Lock()
_relay_cleanup_attempt_count = 0
_relay_cleanup_failure_count = 0
_relay_last_primary_error_type = ""
_relay_last_cleanup_error_type = ""
_RELAY_TERMINAL_FAILURE_EVENT = threading.Event()
_RELAY_RECOVERY_LOCK = threading.Lock()
_relay_pending_cleanup_failures = []
_relay_cleanup_recovery_count = 0


class InterEnclaveRelayError(RuntimeError):
    """A relay request violates the fixed V2 topology or framing contract."""


class InterEnclaveRelayCleanupError(InterEnclaveRelayError):
    """An accepted relay transport could not prove descriptor release."""

    def __init__(
        self,
        *,
        primary_error: BaseException,
        cleanup_error: BaseException,
        resources: Sequence[Any],
    ) -> None:
        super().__init__("inter-enclave relay transport cleanup failed")
        self.primary_error_type = type(primary_error).__name__
        self.cleanup_error_type = type(cleanup_error).__name__
        # Keep ownership reachable until the process supervisor observes the
        # terminal cleanup failure. These resources are never serialized.
        self._resources = tuple(resources)
        self._cleanup_attempt_counts = tuple(1 for _resource in self._resources)


class _ExplicitCloseFailure(RuntimeError):
    """A transport adapter explicitly reported that it retained ownership."""


def _close_transport_required(candidate: Any) -> Optional[BaseException]:
    """Attempt full-duplex shutdown and return any descriptor-release failure."""

    try:
        candidate.shutdown(socket.SHUT_RDWR)
    except Exception:
        # Relay peers normally half-close before the relay exits. Descriptor
        # ownership is therefore proven by close(), not by shutdown().
        pass
    try:
        if candidate.close() is False:
            return _ExplicitCloseFailure("relay transport close was not confirmed")
    except BaseException as exc:
        return exc
    return None


def _record_relay_transport_cleanup(
    *,
    primary_error: Optional[BaseException],
    cleanup_error: Optional[BaseException],
) -> None:
    global _relay_cleanup_attempt_count
    global _relay_cleanup_failure_count
    global _relay_last_primary_error_type
    global _relay_last_cleanup_error_type
    with _RELAY_TRANSPORT_HEALTH_LOCK:
        _relay_cleanup_attempt_count = min(
            MAX_RELAY_CLEANUP_EVENT_COUNT,
            _relay_cleanup_attempt_count + 1,
        )
        if cleanup_error is not None:
            _relay_cleanup_failure_count = min(
                MAX_RELAY_CLEANUP_EVENT_COUNT,
                _relay_cleanup_failure_count + 1,
            )
            _relay_last_primary_error_type = (
                type(primary_error).__name__
                if primary_error is not None
                else type(cleanup_error).__name__
            )
            _relay_last_cleanup_error_type = type(cleanup_error).__name__


def _retain_relay_cleanup_failure(
    failure: InterEnclaveRelayCleanupError,
) -> None:
    """Transfer failed descriptor ownership to the listener supervisor."""

    with _RELAY_TRANSPORT_HEALTH_LOCK:
        _relay_pending_cleanup_failures.append(failure)
        _RELAY_TERMINAL_FAILURE_EVENT.set()


def _recover_relay_cleanup_failures() -> bool:
    global _relay_cleanup_recovery_count
    with _RELAY_RECOVERY_LOCK:
        with _RELAY_TRANSPORT_HEALTH_LOCK:
            snapshot = tuple(
                (failure, failure._resources, failure._cleanup_attempt_counts)
                for failure in _relay_pending_cleanup_failures
            )
        outcomes = []
        for failure, resources, attempt_counts in snapshot:
            unresolved_resources = []
            unresolved_counts = []
            recovered_count = 0
            for resource, attempt_count in zip(resources, attempt_counts):
                cleanup_error = None  # type: Optional[BaseException]
                for _attempt in range(
                    RELAY_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE
                ):
                    attempt_count = min(
                        MAX_RELAY_CLEANUP_ATTEMPT_COUNT,
                        attempt_count + 1,
                    )
                    cleanup_error = _close_transport_required(resource)
                    if cleanup_error is None:
                        recovered_count += 1
                        break
                    failure.cleanup_error_type = type(cleanup_error).__name__
                if cleanup_error is not None:
                    unresolved_resources.append(resource)
                    unresolved_counts.append(attempt_count)
            outcomes.append(
                (
                    failure,
                    tuple(unresolved_resources),
                    tuple(unresolved_counts),
                    recovered_count,
                )
            )
        with _RELAY_TRANSPORT_HEALTH_LOCK:
            outcome_by_id = {id(item[0]): item for item in outcomes}
            retained = []
            recovered_total = 0
            for failure in _relay_pending_cleanup_failures:
                outcome = outcome_by_id.get(id(failure))
                if outcome is None:
                    retained.append(failure)
                    continue
                _failure, resources, counts, recovered_count = outcome
                recovered_total += recovered_count
                if resources:
                    failure._resources = resources
                    failure._cleanup_attempt_counts = counts
                    retained.append(failure)
            _relay_pending_cleanup_failures[:] = retained
            _relay_cleanup_recovery_count += recovered_total
            if retained:
                _RELAY_TERMINAL_FAILURE_EVENT.set()
            else:
                _RELAY_TERMINAL_FAILURE_EVENT.clear()
            return not retained


def relay_transport_health() -> dict[str, Any]:
    """Return a bounded, text-free projection of relay cleanup health."""

    with _RELAY_TRANSPORT_HEALTH_LOCK:
        failures = tuple(_relay_pending_cleanup_failures)
        return {
            "schema_version": RELAY_TRANSPORT_HEALTH_SCHEMA_VERSION,
            "status": "error" if failures else "healthy",
            "cleanup_attempt_count": _relay_cleanup_attempt_count,
            "cleanup_failure_count": _relay_cleanup_failure_count,
            "last_primary_error_type": _relay_last_primary_error_type,
            "last_cleanup_error_type": _relay_last_cleanup_error_type,
            "terminal_failure_latched": (
                bool(failures)
            ),
            "retained_resource_count": (
                sum(len(failure._resources) for failure in failures)
            ),
            "cleanup_recovery_count": _relay_cleanup_recovery_count,
        }


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _send_frame(connection: Any, value: Mapping[str, Any]) -> None:
    encoded = _canonical_json(value)
    if len(encoded) < 2 or len(encoded) > MAX_CONTROL_BYTES:
        raise InterEnclaveRelayError("relay control response is outside size limit")
    connection.sendall(len(encoded).to_bytes(4, "big") + encoded)


def _read_control(connection: Any) -> dict[str, Any]:
    prefix = _recv_exact(connection, 4)
    if len(prefix) != 4:
        raise InterEnclaveRelayError("relay control prefix is incomplete")
    size = int.from_bytes(prefix, "big")
    if size < 2 or size > MAX_CONTROL_BYTES:
        raise InterEnclaveRelayError("relay control frame is outside size limit")
    encoded = _recv_exact(connection, size)
    if len(encoded) != size:
        raise InterEnclaveRelayError("relay control body is incomplete")
    try:
        value = json.loads(encoded.decode("ascii"))
    except Exception as exc:
        raise InterEnclaveRelayError("relay control JSON is invalid") from exc
    if not isinstance(value, dict):
        raise InterEnclaveRelayError("relay control request must be an object")
    return value


def _validated_target(request: Mapping[str, Any], *, source_cid: int) -> tuple[int, int, str]:
    if set(request) != {
        "schema_version",
        "channel_id",
        "source_cid",
        "target_cid",
        "target_port",
        "topology_hash",
    }:
        raise InterEnclaveRelayError("relay control fields are invalid")
    if request.get("schema_version") != "leadpoet.inter_enclave_relay.v2":
        raise InterEnclaveRelayError("relay control schema is invalid")
    if int(request.get("source_cid", -1)) != int(source_cid):
        raise InterEnclaveRelayError("relay source CID differs from socket peer")
    target_cid = int(request.get("target_cid", -1))
    target_port = int(request.get("target_port", -1))
    if (int(source_cid), target_cid) not in _APPROVED_PAIRS:
        raise InterEnclaveRelayError("relay CID pair is not authorized")
    if target_port != TARGET_TLS_PORT:
        raise InterEnclaveRelayError("relay target port is not authorized")
    if request.get("topology_hash") != topology_hash():
        raise InterEnclaveRelayError("relay topology hash mismatch")
    channel_id = str(request.get("channel_id") or "").lower()
    if not _CHANNEL_ID_RE.fullmatch(channel_id):
        raise InterEnclaveRelayError("relay channel ID is invalid")
    return target_cid, target_port, channel_id


def _relay_opaque(left: Any, right: Any, *, idle_timeout: float) -> None:
    peers = {left: right, right: left}
    active = {left, right}
    totals = {left: 0, right: 0}
    last_activity = time.monotonic()
    while active:
        remaining = idle_timeout - (time.monotonic() - last_activity)
        if remaining <= 0:
            raise InterEnclaveRelayError("relay channel idle timeout")
        readable, _, _ = select.select(list(active), [], [], min(1.0, remaining))
        if not readable:
            continue
        for source in readable:
            data = source.recv(RELAY_CHUNK_BYTES)
            destination = peers[source]
            if not data:
                active.discard(source)
                try:
                    destination.shutdown(socket.SHUT_WR)
                except Exception:
                    pass
                continue
            totals[source] += len(data)
            if totals[source] > MAX_CHANNEL_BYTES_PER_DIRECTION:
                raise InterEnclaveRelayError("relay channel byte limit exceeded")
            destination.sendall(data)
            last_activity = time.monotonic()


def _connect_target(target_cid: int, target_port: int) -> Any:
    connection = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    try:
        connection.connect((target_cid, target_port))
    except BaseException as exc:
        cleanup_error = _close_transport_required(connection)
        if cleanup_error is not None:
            raise InterEnclaveRelayCleanupError(
                primary_error=exc,
                cleanup_error=cleanup_error,
                resources=(connection,),
            ) from exc
        raise
    return connection


def _handle_connection(
    connection: Any,
    *,
    source_cid: int,
    connector: Callable[[int, int], Any] = _connect_target,
    idle_timeout: float = IDLE_TIMEOUT_SECONDS,
    cleanup_failure_callback: Optional[
        Callable[[InterEnclaveRelayCleanupError], None]
    ] = None,
) -> None:
    target = None
    primary_error = None  # type: Optional[BaseException]
    try:
        request = _read_control(connection)
        target_cid, target_port, channel_id = _validated_target(
            request,
            source_cid=source_cid,
        )
        target = connector(target_cid, target_port)
        _send_frame(
            connection,
            {
                "result": {
                    "status": "connected",
                    "channel_id": channel_id,
                    "topology_hash": topology_hash(),
                }
            },
        )
        _relay_opaque(connection, target, idle_timeout=idle_timeout)
    except Exception as exc:
        primary_error = exc
        try:
            _send_frame(
                connection,
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                },
            )
        except Exception:
            pass
    cleanup_failures = []
    for candidate in (target, connection):
        if candidate is None:
            continue
        cleanup_error = _close_transport_required(candidate)
        if cleanup_error is not None:
            cleanup_failures.append((candidate, cleanup_error))
    _record_relay_transport_cleanup(
        primary_error=primary_error,
        cleanup_error=(
            cleanup_failures[0][1]
            if cleanup_failures
            else primary_error
            if isinstance(primary_error, InterEnclaveRelayCleanupError)
            else None
        ),
    )
    failure = (
        primary_error
        if isinstance(primary_error, InterEnclaveRelayCleanupError)
        else None
    )
    if cleanup_failures:
        cleanup_error = cleanup_failures[0][1]
        cleanup_primary = primary_error or cleanup_error
        resources = []
        attempt_counts = []
        seen_resource_ids = set()
        if isinstance(primary_error, InterEnclaveRelayCleanupError):
            for resource, attempt_count in zip(
                primary_error._resources,
                primary_error._cleanup_attempt_counts,
            ):
                resource_id = id(resource)
                if resource_id in seen_resource_ids:
                    continue
                seen_resource_ids.add(resource_id)
                resources.append(resource)
                attempt_counts.append(attempt_count)
        for candidate, _error in cleanup_failures:
            resource_id = id(candidate)
            if resource_id in seen_resource_ids:
                continue
            seen_resource_ids.add(resource_id)
            resources.append(candidate)
            attempt_counts.append(1)
        failure = InterEnclaveRelayCleanupError(
            primary_error=cleanup_primary,
            cleanup_error=cleanup_error,
            resources=tuple(resources),
        )
        failure._cleanup_attempt_counts = tuple(attempt_counts)
    if failure is not None:
        if cleanup_failure_callback is not None:
            try:
                cleanup_failure_callback(failure)
            except BaseException:
                raise InterEnclaveRelayError(
                    "relay cleanup ownership transfer failed"
                ) from failure
        if cleanup_failures:
            raise failure from cleanup_primary
        raise failure


def serve_forever(*, port: int = DEFAULT_RELAY_PORT) -> None:
    listener = None
    primary_error = None  # type: Optional[BaseException]
    try:
        listener = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        listener.bind((VMADDR_CID_ANY, int(port)))
        listener.listen(64)
        listener.settimeout(RELAY_SUPERVISOR_POLL_SECONDS)
        while True:
            if _RELAY_TERMINAL_FAILURE_EVENT.is_set():
                if not _recover_relay_cleanup_failures():
                    time.sleep(RELAY_SUPERVISOR_POLL_SECONDS)
                continue
            try:
                connection, address = listener.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if exc.errno in _TRANSIENT_ACCEPT_ERRNOS:
                    continue
                raise
            if _RELAY_TERMINAL_FAILURE_EVENT.is_set():
                cleanup_error = _close_transport_required(connection)
                _record_relay_transport_cleanup(
                    primary_error=None,
                    cleanup_error=cleanup_error,
                )
                if cleanup_error is not None:
                    _retain_relay_cleanup_failure(
                        InterEnclaveRelayCleanupError(
                            primary_error=cleanup_error,
                            cleanup_error=cleanup_error,
                            resources=(connection,),
                        )
                    )
                if not _recover_relay_cleanup_failures():
                    time.sleep(RELAY_SUPERVISOR_POLL_SECONDS)
                continue
            try:
                source_cid = int(address[0])
                worker = threading.Thread(
                    target=_handle_connection,
                    kwargs={
                        "connection": connection,
                        "source_cid": source_cid,
                        "cleanup_failure_callback": (
                            _retain_relay_cleanup_failure
                        ),
                    },
                    name="gateway-inter-enclave-relay",
                    daemon=True,
                )
                worker.start()
            except BaseException as exc:
                cleanup_error = _close_transport_required(connection)
                if cleanup_error is not None:
                    failure = InterEnclaveRelayCleanupError(
                        primary_error=exc,
                        cleanup_error=cleanup_error,
                        resources=(connection,),
                    )
                    _record_relay_transport_cleanup(
                        primary_error=exc,
                        cleanup_error=cleanup_error,
                    )
                    _retain_relay_cleanup_failure(failure)
                    _recover_relay_cleanup_failures()
                raise
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if listener is not None:
            cleanup_error = _close_transport_required(listener)
            if cleanup_error is not None:
                cleanup_primary = primary_error or cleanup_error
                failure = InterEnclaveRelayCleanupError(
                    primary_error=cleanup_primary,
                    cleanup_error=cleanup_error,
                    resources=(listener,),
                )
                _record_relay_transport_cleanup(
                    primary_error=primary_error,
                    cleanup_error=cleanup_error,
                )
                _retain_relay_cleanup_failure(failure)
                if not _recover_relay_cleanup_failures():
                    raise failure from cleanup_primary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_RELAY_PORT)
    args = parser.parse_args(argv)
    serve_forever(port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
