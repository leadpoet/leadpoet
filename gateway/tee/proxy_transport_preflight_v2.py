"""Pre-shutdown capability checks for sealed V2 worker proxy profiles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Any, Callable, Mapping, Sequence

from gateway.tee.egress_proxy import EnclaveEgressProxy
from gateway.tee.provider_broker_v2 import _validated_tls_proxy_url
from gateway.utils.tee_egress_forwarder import _connect_public_destination


_PROBE_DESTINATIONS = {
    "gateway_autoresearch": ("openrouter.ai", 443),
    "gateway_scoring": ("api.exa.ai", 443),
}
_DEFAULT_ATTEMPTS = 2
_DEFAULT_MAX_WORKERS = 8
_DEFAULT_TIMEOUT_SECONDS = 5.0


class WorkerProxyTransportPreflightV2Error(RuntimeError):
    """A configured worker proxy cannot carry the measured V2 transport."""


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

    def _open_parent_tunnel(self, host: str, port: int) -> Any:
        connection = self._host_connector(host, port)
        connection.settimeout(self._host_timeout_seconds)
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
    """Exercise the same certificate-verified CONNECT handshake used in-enclave."""

    normalized_proxy_url = _validated_tls_proxy_url(proxy_url)
    normalized_attempts = max(1, int(attempts))
    last_error: Exception | None = None
    for attempt in range(normalized_attempts):
        tunnel = None
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
            return
        except Exception as exc:
            last_error = exc
            if attempt + 1 < normalized_attempts:
                sleep(0.2)
        finally:
            if tunnel is not None:
                try:
                    tunnel.close()
                except Exception:
                    pass
    raise WorkerProxyTransportPreflightV2Error(
        "worker proxy failed certificate-verified authenticated CONNECT preflight"
    ) from last_error


def verify_worker_proxy_fleets_v2(
    fleets: Mapping[str, Sequence[str]],
    *,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    verify_proxy: Callable[..., None] = verify_tls_proxy_connect_v2,
) -> None:
    """Verify every selected worker credential without exposing proxy URLs."""

    tasks = []
    for role, values in fleets.items():
        destination = _PROBE_DESTINATIONS.get(str(role))
        if destination is None:
            raise WorkerProxyTransportPreflightV2Error(
                "worker proxy execution role is not measured"
            )
        for index, value in enumerate(values):
            tasks.append((str(role), index, str(value), destination))
    if not tasks:
        raise WorkerProxyTransportPreflightV2Error(
            "worker proxy fleet is unavailable"
        )

    failures = []
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
            ): (role, index)
            for role, index, value, destination in tasks
        }
        for future in as_completed(futures):
            role, index = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((role, index, exc))
    if failures:
        role, index, error = sorted(
            failures,
            key=lambda item: (item[0], item[1]),
        )[0]
        raise WorkerProxyTransportPreflightV2Error(
            "%s worker proxy %d failed V2 TLS CONNECT preflight (%d/%d failed)"
            % (role, index + 1, len(failures), len(tasks))
        ) from error
