#!/usr/bin/env python3
"""Verify pinned DDGS TLS handshakes against the Arena CONNECT inspector.

This explicit Linux/root probe installs the submitted requirement through the
normal SourceCache path. It makes no web requests. A local proxy attempts every
browser profile that DDGS 9.8.0 advertises, reports profiles that the installed
platform wheel does not implement, and checks all emitted ClientHellos. It also
requires the explicit DuckDuckGo backend to use matching public SNI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab_arena import source_bundle  # noqa: E402
from lab_arena.runner import SourceCache  # noqa: E402
from lab_arena.web_egress import _read_client_hello  # noqa: E402


DDGS_VERSION = "9.8.0"
PRIMP_VERSION = "0.15.0"
DESTINATION_HOST = "example.com"
EXPECTED_PROFILE_COUNT = 43
MAX_PROFILE_CONNECTIONS = 4
MAX_BACKEND_CONNECTIONS = 16

PROFILE_INVENTORY_CLIENT = r"""
import json
from importlib.metadata import version
import sys

sys.path.insert(0, sys.argv[1])
from ddgs.http_client import HttpClient

if version("ddgs") != sys.argv[2]:
    raise RuntimeError("installed DDGS version does not match the probe pin")
if version("primp") != sys.argv[3]:
    raise RuntimeError("installed primp version does not match the probe pin")
print(json.dumps({
    "version": version("ddgs"),
    "primp_version": version("primp"),
    "labels": HttpClient._impersonates,
}))
"""

PROFILE_CLIENT = r"""
import json
from importlib.metadata import version
import sys

sys.path.insert(0, sys.argv[1])
import primp

if version("ddgs") != sys.argv[2]:
    raise RuntimeError("installed DDGS version does not match the probe pin")
proxy_url = sys.argv[3]
profile = sys.argv[4]
destination = sys.argv[5]
error_type = ""
error_detail = ""
try:
    primp.Client(
        proxy=proxy_url,
        timeout=2,
        impersonate=profile,
        verify=False,
    ).get("https://" + destination + "/")
except Exception as exc:
    error_type = type(exc).__name__
    error_detail = str(exc)[:200]
print(json.dumps({
    "version": version("ddgs"),
    "profile": profile,
    "client_error_type": error_type,
    "client_error_detail": error_detail,
}))
"""

DDGS_CLIENT = r"""
import json
from importlib.metadata import version
import sys

sys.path.insert(0, sys.argv[1])
from ddgs import DDGS

if version("ddgs") != sys.argv[2]:
    raise RuntimeError("installed DDGS version does not match the probe pin")
proxy_url = sys.argv[3]
try:
    DDGS(proxy=proxy_url, timeout=2, verify=False).text(
        "arena proxy compatibility",
        backend="duckduckgo",
        max_results=1,
    )
except Exception:
    pass
print(json.dumps({"version": version("ddgs"), "backend": "duckduckgo"}))
"""


def _source_archive(root: Path) -> bytes:
    source = root / "source"
    source.mkdir()
    (source / "harness.py").write_text(
        "def run_icp(icp):\n    return []\n", encoding="utf-8"
    )
    (source / "requirements.txt").write_text(
        "ddgs==%s\nprimp==%s\n" % (DDGS_VERSION, PRIMP_VERSION), encoding="utf-8"
    )
    archive = root / "source.tar.gz"
    source_bundle.write_source_archive(source, archive)
    return archive.read_bytes()


def _capture(
    listener: socket.socket,
    done: threading.Event,
    observed: list[object],
    maximum_connections: int,
) -> None:
    listener.settimeout(0.2)
    while not done.is_set() and len(observed) < maximum_connections:
        try:
            connection, _address = listener.accept()
        except socket.timeout:
            continue
        except OSError:
            if done.is_set():
                return
            raise
        try:
            connection.settimeout(4)
            headers = bytearray()
            while b"\r\n\r\n" not in headers and len(headers) < 64 * 1024:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                headers.extend(chunk)
            connection.sendall(
                b"HTTP/1.1 200 Connection Established\r\nConnection: close\r\n\r\n"
            )
            _raw, server_name = _read_client_hello(connection)
            request_line = bytes(headers).split(b"\r\n", 1)[0]
            parts = request_line.split(b" ")
            authority = (
                parts[1].decode("ascii")
                if len(parts) == 3 and parts[0] == b"CONNECT"
                else ""
            )
            observed.append({"authority": authority, "server_name": server_name})
        except Exception as exc:
            observed.append({"error_type": type(exc).__name__})
        finally:
            connection.close()


def _capture_client(
    dependencies: Path,
    client_source: str,
    *client_arguments: str,
    maximum_connections: int,
) -> tuple[dict[str, object], list[object]]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(maximum_connections)
    proxy_url = "http://127.0.0.1:%d" % listener.getsockname()[1]
    done = threading.Event()
    observed: list[object] = []
    thread = threading.Thread(
        target=_capture,
        args=(listener, done, observed, maximum_connections),
        daemon=True,
    )
    thread.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                client_source,
                str(dependencies),
                DDGS_VERSION,
                proxy_url,
                *client_arguments,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=False,
            text=True,
        )
    finally:
        done.set()
        listener.close()
        thread.join(timeout=5)
    if result.returncode != 0:
        raise RuntimeError("DDGS ClientHello probe process failed")
    try:
        report = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("DDGS ClientHello probe report is invalid") from exc
    if not isinstance(report, dict):
        raise RuntimeError("DDGS ClientHello probe report is invalid")
    return report, observed


def _profile_inventory(dependencies: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            PROFILE_INVENTORY_CLIENT,
            str(dependencies),
            DDGS_VERSION,
            PRIMP_VERSION,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("DDGS browser profile inventory process failed")
    try:
        report = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("DDGS browser profile inventory is invalid") from exc
    if not isinstance(report, dict):
        raise RuntimeError("DDGS browser profile inventory is invalid")
    return report


def _authority_host(value: object) -> str:
    authority = str(value or "")
    host, separator, port = authority.rpartition(":")
    if not separator or port != "443":
        return ""
    return host.lower().rstrip(".")


def _validated_profile_inventory(labels: object) -> list[str]:
    if (
        not isinstance(labels, list)
        or len(labels) != EXPECTED_PROFILE_COUNT
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        raise RuntimeError("DDGS browser profile inventory does not match the pin")
    return labels


def _validated_profile_connection_count(profile: str, observed: list[object]) -> int:
    if not observed:
        raise RuntimeError(
            "DDGS browser profile did not use its explicit proxy: " + profile
        )
    if len(observed) > MAX_PROFILE_CONNECTIONS:
        raise RuntimeError(
            "DDGS browser profile exceeded the connection bound: " + profile
        )
    for item in observed:
        if not isinstance(item, dict):
            raise RuntimeError(
                "DDGS browser profile ClientHello is invalid: " + profile
            )
        authority_host = _authority_host(item.get("authority"))
        if (
            authority_host != DESTINATION_HOST
            or item.get("server_name") != authority_host
        ):
            raise RuntimeError(
                "DDGS browser profile failed Arena SNI inspection: " + profile
            )
    return len(observed)


def _is_unsupported_linux_profile(profile: str, report: dict[str, object]) -> bool:
    return (
        report.get("client_error_type") == "BuilderError"
        and report.get("client_error_detail") == 'Invalid impersonate: "%s"' % profile
    )


def _validated_backend_count(observed: list[object]) -> int:
    if not observed:
        raise RuntimeError("DDGS DuckDuckGo backend did not use its explicit proxy")
    if len(observed) > MAX_BACKEND_CONNECTIONS:
        raise RuntimeError("DDGS DuckDuckGo backend exceeded the connection bound")
    for item in observed:
        if not isinstance(item, dict):
            raise RuntimeError("DDGS DuckDuckGo ClientHello is invalid")
        authority_host = _authority_host(item.get("authority"))
        if (
            not (
                authority_host == "duckduckgo.com"
                or authority_host.endswith(".duckduckgo.com")
            )
            or item.get("server_name") != authority_host
        ):
            raise RuntimeError("DDGS DuckDuckGo backend failed Arena SNI inspection")
    return len(observed)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("DDGS SourceCache probe requires Linux root")
    with tempfile.TemporaryDirectory(prefix="arena-ddgs-probe-") as temporary:
        root = Path(temporary)
        payload = _source_archive(root)
        cache = SourceCache(root / "cache", lambda _run, _token: payload)
        with cache.acquire(
            "ddgs-clienthello-probe",
            "probe",
            "test/ddgs/source.tar.gz",
            "ddgs-" + DDGS_VERSION,
            len(payload),
        ) as (_source, dependencies):
            inventory_report = _profile_inventory(dependencies)
            profiles = _validated_profile_inventory(inventory_report.get("labels"))
            profile_connection_count = 0
            supported_profile_count = 0
            unsupported_profile_count = 0
            for profile in profiles:
                profile_report, profile_hellos = _capture_client(
                    dependencies,
                    PROFILE_CLIENT,
                    profile,
                    DESTINATION_HOST,
                    maximum_connections=MAX_PROFILE_CONNECTIONS,
                )
                if (
                    profile_report.get("version") != DDGS_VERSION
                    or profile_report.get("profile") != profile
                ):
                    raise RuntimeError(
                        "DDGS browser profile report does not match the pin"
                    )
                if not profile_hellos and _is_unsupported_linux_profile(
                    profile, profile_report
                ):
                    unsupported_profile_count += 1
                    continue
                if not profile_hellos and profile_report.get("client_error_type"):
                    raise RuntimeError(
                        "DDGS browser profile did not open its explicit proxy: %s (%s)"
                        % (
                            profile,
                            "%s: %s"
                            % (
                                profile_report["client_error_type"],
                                profile_report.get("client_error_detail", ""),
                            ),
                        )
                    )
                supported_profile_count += 1
                profile_connection_count += _validated_profile_connection_count(
                    profile, profile_hellos
                )
            if supported_profile_count + unsupported_profile_count != len(profiles):
                raise RuntimeError("DDGS browser profile accounting is incomplete")
            backend_report, backend_hellos = _capture_client(
                dependencies,
                DDGS_CLIENT,
                maximum_connections=MAX_BACKEND_CONNECTIONS,
            )
            backend_count = _validated_backend_count(backend_hellos)
            if (
                inventory_report.get("version") != DDGS_VERSION
                or inventory_report.get("primp_version") != PRIMP_VERSION
                or backend_report.get("version") != DDGS_VERSION
                or backend_report.get("backend") != "duckduckgo"
            ):
                raise RuntimeError("DDGS ClientHello report does not match the pin")
            print(
                json.dumps(
                    {
                        "browser_profile_connection_count": profile_connection_count,
                        "browser_profile_count": len(profiles),
                        "browser_profile_matching_sni_count": profile_connection_count,
                        "browser_profile_supported_count": supported_profile_count,
                        "browser_profile_unsupported_linux_count": unsupported_profile_count,
                        "ddgs_backend": "duckduckgo",
                        "ddgs_backend_connection_count": backend_count,
                        "ddgs_backend_matching_sni_count": backend_count,
                        "ddgs_version": DDGS_VERSION,
                        "primp_version": PRIMP_VERSION,
                        "source_cache_install": "passed",
                    },
                    sort_keys=True,
                )
            )
    print("ARENA_DDGS_CLIENT_HELLO_PROBE_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
