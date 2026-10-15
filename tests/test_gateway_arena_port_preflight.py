"""The real controller bind probes must match the serving socket's reuse policy."""
from __future__ import annotations

import errno
from pathlib import Path
import re
import socket
import subprocess
import sys

import pytest

SOURCE = (Path(__file__).resolve().parents[1] / "gw_restart.sh").read_text()
BLOCKS = [block for block in re.findall(r"<<'PY'\n(.*?)\nPY\n", SOURCE, re.S)
          if "listener.bind(" in block and "socket.socket(" in block]
assert len(BLOCKS) == 2


def _probe(block: str, ports: list[int]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-", *(str(port) for port in ports)],
                          input=block, capture_output=True, text=True, timeout=5)


def _closed_server_port() -> int:
    # Uvicorn's listener uses SO_REUSEADDR. Make the accepted server endpoint
    # close first, so a real TCP TIME_WAIT remains after all sockets are closed.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            accepted, _ = listener.accept()
            with accepted:
                accepted.settimeout(2)
                accepted.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""
                client.shutdown(socket.SHUT_WR)
                assert accepted.recv(1) == b""
    return port


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.mark.parametrize("block", BLOCKS, ids=["lock-handoff", "normal-overlap"])
@pytest.mark.parametrize("endpoint", [0, 1], ids=["api", "edge"])
def test_real_bind_block_accepts_time_wait_and_old_probe_refuses(block: str, endpoint: int):
    retired_port = _closed_server_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as raw:
        with pytest.raises(OSError) as rejected:
            raw.bind(("127.0.0.1", retired_port))
        assert rejected.value.errno == errno.EADDRINUSE
    ports = [_free_port(), _free_port()]
    ports[endpoint] = retired_port
    old_block = block.replace("        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n", "")
    assert _probe(old_block, ports).returncode != 0
    result = _probe(block, ports)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("block", BLOCKS, ids=["lock-handoff", "normal-overlap"])
@pytest.mark.parametrize("endpoint", [0, 1], ids=["api", "edge"])
def test_real_bind_block_still_refuses_active_reuseaddr_listener(block: str, endpoint: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as active:
        active.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        active.bind(("127.0.0.1", 0))
        active.listen()
        active.settimeout(2)
        ports = [_free_port(), _free_port()]
        ports[endpoint] = active.getsockname()[1]
        assert _probe(block, ports).returncode != 0
        # A refused preflight leaves the unrelated listener usable.
        with socket.create_connection(active.getsockname(), timeout=2):
            accepted, _ = active.accept()
            accepted.close()
