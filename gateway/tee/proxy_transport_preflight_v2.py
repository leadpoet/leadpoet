"""Pre-shutdown capability checks for sealed V2 worker proxy profiles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from gateway.tee.egress_proxy import (
    EnclaveEgressProxy,
    EnclaveEgressProxyCleanupError,
    _shutdown_and_close_socket,
)
from gateway.tee.provider_broker_v2 import _validated_tls_proxy_url
from gateway.utils.tee_egress_forwarder import (
    TEEEgressForwarderCleanupError,
    _connect_public_destination,
)


_PROBE_DESTINATIONS = {
    "gateway_scoring": (
        ("openrouter.ai", 443),
        ("api.exa.ai", 443),
        ("api.scrapingdog.com", 443),
        ("code.deepline.com", 443),
    ),
}
_DEFAULT_ATTEMPTS = 2
_DEFAULT_MAX_WORKERS = 8
_DEFAULT_TIMEOUT_SECONDS = 5.0


class WorkerProxyTransportPreflightV2Error(RuntimeError):
    """A configured worker proxy cannot carry the measured V2 transport."""


class WorkerProxyTransportCleanupV2Error(WorkerProxyTransportPreflightV2Error):
    """A preflight transport could not prove required local cleanup."""

    def __init__(self, *, stage: str, primary_error: BaseException) -> None:
        super().__init__("worker proxy preflight transport cleanup failed")
        self.stage = str(stage)
        self.primary_error = primary_error


_RETIRED_CLEANUP_LOCK = threading.RLock()
_RETIRED_CLEANUP_RECOVERY_LOCK = threading.Lock()
_RETIRED_CLEANUP_RESOURCES: dict[int, tuple[str, Any, BaseException]] = {}


def _retain_cleanup_resource(
    resource: Any,
    *,
    kind: str,
    primary_error: BaseException,
) -> None:
    with _RETIRED_CLEANUP_LOCK:
        _RETIRED_CLEANUP_RESOURCES[id(resource)] = (
            str(kind),
            resource,
            primary_error,
        )


def _retry_retired_cleanup() -> tuple[str, BaseException] | None:
    """Retry every unproven close without racing a concurrent retention."""

    with _RETIRED_CLEANUP_RECOVERY_LOCK:
        with _RETIRED_CLEANUP_LOCK:
            snapshot = tuple(_RETIRED_CLEANUP_RESOURCES.items())
        resolved = []
        for resource_id, (kind, resource, _primary_error) in snapshot:
            if kind == "probe":
                cleaned = resource._retry_retired_cleanup()
            else:
                cleaned = _shutdown_and_close_socket(resource)
            if cleaned:
                resolved.append((resource_id, resource))
        with _RETIRED_CLEANUP_LOCK:
            for resource_id, resource in resolved:
                current = _RETIRED_CLEANUP_RESOURCES.get(resource_id)
                if current is not None and current[1] is resource:
                    _RETIRED_CLEANUP_RESOURCES.pop(resource_id, None)
            pending = next(iter(_RETIRED_CLEANUP_RESOURCES.values()), None)
            return None if pending is None else (pending[0], pending[2])


class _HostProxyProbe(EnclaveEgressProxy):
    def __init__(
        self,
        *,
        connector: Callable[[str, int], Any],
        timeout_seconds: float,
    ) -> None:
        super().__init__(recv_exact=lambda _connection, _length: b"")
        self._host_connector = connector
        self._host_timeout_seconds = float(timeout_seconds)

    def _open_parent_tunnel(
        self,
        host: str,
        port: int,
        *,
        purpose: str = "provider",
    ) -> Any:
        if purpose != "upstream_proxy":
            raise WorkerProxyTransportPreflightV2Error(
                "proxy preflight attempted a non-proxy parent tunnel"
            )
        try:
            connection = self._host_connector(host, port)
        except TEEEgressForwarderCleanupError as connector_error:
            for resource in connector_error._resources:
                self._retain_cleanup_resource(
                    resource,
                    stage="host_proxy_connector_cleanup",
                )
            raise EnclaveEgressProxyCleanupError(
                stage="host_proxy_connector_cleanup",
                primary_error=connector_error.primary_error,
                resources=connector_error._resources,
            ) from connector_error.primary_error
        try:
            connection.settimeout(self._host_timeout_seconds)
        except Exception as primary_error:
            if not _shutdown_and_close_socket(connection):
                self._retain_cleanup_resource(
                    connection,
                    stage="host_proxy_timeout_cleanup",
                )
                raise EnclaveEgressProxyCleanupError(
                    stage="host_proxy_timeout_cleanup",
                    primary_error=primary_error,
                    resources=(connection,),
                ) from primary_error
            raise
        return connection


def verify_tls_proxy_connect_v2(
    proxy_url: str,
    *,
    destination_host: str,
    destination_port: int = 443,
    attempts: int = _DEFAULT_ATTEMPTS,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    connector: Callable[[str, int], Any] = _connect_public_destination,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Exercise the same authenticated CONNECT handshake used in-enclave."""

    pending_failure = _retry_retired_cleanup()
    if pending_failure is not None:
        stage, primary_error = pending_failure
        raise WorkerProxyTransportCleanupV2Error(
            stage="retired_" + stage + "_cleanup",
            primary_error=primary_error,
        ) from primary_error
    normalized_proxy_url = _validated_tls_proxy_url(proxy_url)
    normalized_attempts = max(1, int(attempts))
    last_error: Exception | None = None
    for attempt in range(normalized_attempts):
        probe = None
        tunnel = None
        attempt_error = None
        cleanup_error = None
        try:
            probe = _HostProxyProbe(
                connector=connector,
                timeout_seconds=timeout_seconds,
            )
            tunnel = probe._open_upstream_proxy_tunnel(
                proxy_url=normalized_proxy_url,
                destination_host=destination_host,
                destination_port=destination_port,
            )
        except Exception as exc:
            attempt_error = exc
        finally:
            if tunnel is not None:
                if not _shutdown_and_close_socket(tunnel):
                    cleanup_error = WorkerProxyTransportPreflightV2Error(
                        "worker proxy CONNECT stream cleanup failed"
                    )
                    _retain_cleanup_resource(
                        tunnel,
                        kind="transport",
                        primary_error=attempt_error or cleanup_error,
                    )
            if probe is not None and not probe._retry_retired_cleanup():
                cleanup_error = cleanup_error or (
                    WorkerProxyTransportPreflightV2Error(
                        "worker proxy internal transport cleanup failed"
                    )
                )
                _retain_cleanup_resource(
                    probe,
                    kind="probe",
                    primary_error=attempt_error or cleanup_error,
                )
        if cleanup_error is not None:
            primary_error = attempt_error or cleanup_error
            raise WorkerProxyTransportCleanupV2Error(
                stage="connect_transport_cleanup",
                primary_error=primary_error,
            ) from primary_error
        if attempt_error is None:
            return
        last_error = attempt_error
        if attempt + 1 < normalized_attempts:
            sleep(0.2)
    raise WorkerProxyTransportPreflightV2Error(
        "worker proxy failed authenticated CONNECT preflight"
    ) from last_error


def verify_worker_proxy_fleets_v2(
    fleets: Mapping[str, Sequence[str]],
    *,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    verify_proxy: Callable[..., None] = verify_tls_proxy_connect_v2,
) -> dict[str, tuple[str, ...]]:
    """Return profiles that pass every measured destination check."""

    tasks = []
    for role, values in fleets.items():
        destinations = _PROBE_DESTINATIONS.get(str(role))
        if destinations is None:
            raise WorkerProxyTransportPreflightV2Error(
                "worker proxy execution role is not measured"
            )
        if not values:
            raise WorkerProxyTransportPreflightV2Error(
                "%s worker proxy fleet is unavailable" % role
            )
        for index, value in enumerate(values):
            for destination in destinations:
                tasks.append((str(role), index, str(value), destination))
    if not tasks:
        raise WorkerProxyTransportPreflightV2Error(
            "worker proxy fleet is unavailable"
        )

    failures = []
    failed_profiles = set()
    worker_count = min(max(1, int(max_workers)), len(tasks))
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="v2-proxy-preflight",
    ) as executor:
        futures = {
            executor.submit(
                verify_proxy,
                value,
                destination_host=destination[0],
                destination_port=destination[1],
            ): (role, index, destination)
            for role, index, value, destination in tasks
        }
        for future in as_completed(futures):
            role, index, destination = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((role, index, destination, exc))
                failed_profiles.add((role, index))

    verified_fleets = {}
    for role, values in fleets.items():
        verified = tuple(
            str(value)
            for index, value in enumerate(values)
            if (str(role), index) not in failed_profiles
        )
        if verified:
            verified_fleets[str(role)] = verified
            continue
        role_failures = sorted(
            (
                failure
                for failure in failures
                if failure[0] == str(role)
            ),
            key=lambda item: (item[1], item[2]),
        )
        if not role_failures:
            raise WorkerProxyTransportPreflightV2Error(
                "%s worker proxy fleet has no verified profiles" % role
            )
        _, index, destination, error = role_failures[0]
        raise WorkerProxyTransportPreflightV2Error(
            "%s worker proxy fleet has no verified profiles; proxy %d failed "
            "V2 TLS CONNECT preflight to %s:%d (%d/%d role probes failed)"
            % (
                role,
                index + 1,
                destination[0],
                destination[1],
                len(role_failures),
                len(values) * len(_PROBE_DESTINATIONS[str(role)]),
            )
        ) from error
    return verified_fleets
