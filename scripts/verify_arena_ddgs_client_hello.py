#!/usr/bin/env python3
"""Verify pinned DDGS TLS handshakes against the Arena CONNECT inspector.

This explicit Linux/root probe installs the submitted requirement through the
normal SourceCache path. It makes no web requests: a local proxy captures the
ClientHello from every browser profile that DDGS 9.8.0 can select, plus its
explicit DuckDuckGo backend, and checks the visible SNI with the production
Arena parser.
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
DESTINATION_HOST = "public.example"

CLIENT = r"""
import json
from importlib.metadata import version
import socket
import sys

sys.path.insert(0, sys.argv[1])
from ddgs import DDGS
from ddgs.http_client import HttpClient
import primp

if version("ddgs") != sys.argv[2]:
    raise RuntimeError("installed DDGS version does not match the probe pin")
proxy_url = sys.argv[3]
destination = sys.argv[4]
labels = []
for profile in HttpClient._impersonates:
    labels.append(profile)
    try:
        primp.Client(
            proxy=proxy_url,
            timeout=2,
            impersonate=profile,
            verify=False,
        ).get("https://" + destination + "/")
    except Exception:
        pass
labels.append("duckduckgo-httpx")
try:
    DDGS(proxy=proxy_url, timeout=2, verify=False).text(
        "arena proxy compatibility",
        backend="duckduckgo",
        max_results=1,
    )
except Exception:
    pass
print(json.dumps({"version": version("ddgs"), "labels": labels}))
"""


def _source_archive(root: Path) -> bytes:
    source = root / "source"
    source.mkdir()
    (source / "harness.py").write_text(
        "def run_icp(icp):\n    return []\n", encoding="utf-8"
    )
    (source / "requirements.txt").write_text(
        "ddgs==%s\n" % DDGS_VERSION, encoding="utf-8"
    )
    archive = root / "source.tar.gz"
    source_bundle.write_source_archive(source, archive)
    return archive.read_bytes()


def _capture(
    listener: socket.socket, done: threading.Event, observed: list[object]
) -> None:
    listener.settimeout(0.2)
    while not done.is_set():
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
            observed.append(server_name)
        except Exception as exc:
            observed.append({"error_type": type(exc).__name__, "detail": str(exc)})
        finally:
            connection.close()


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
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen(64)
            proxy_url = "http://127.0.0.1:%d" % listener.getsockname()[1]
            done = threading.Event()
            observed: list[object] = []
            thread = threading.Thread(
                target=_capture, args=(listener, done, observed), daemon=True
            )
            thread.start()
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-c",
                        CLIENT,
                        str(dependencies),
                        DDGS_VERSION,
                        proxy_url,
                        DESTINATION_HOST,
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
            report = json.loads(result.stdout)
            labels = report["labels"]
            if len(observed) != len(labels):
                raise RuntimeError(
                    "DDGS ClientHello count does not match profile count"
                )
            failures = [
                label
                for label, server_name in zip(labels, observed)
                if server_name != DESTINATION_HOST
            ]
            if failures:
                raise RuntimeError(
                    "DDGS ClientHello failed Arena SNI inspection for: "
                    + ", ".join(failures)
                )
            print(
                json.dumps(
                    {
                        "ddgs_version": report["version"],
                        "client_hello_count": len(labels),
                        "matching_visible_sni_count": len(observed),
                        "source_cache_install": "passed",
                    },
                    sort_keys=True,
                )
            )
    print("ARENA_DDGS_CLIENT_HELLO_PROBE_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
