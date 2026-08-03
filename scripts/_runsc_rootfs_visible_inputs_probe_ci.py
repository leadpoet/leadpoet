#!/usr/bin/env python3
"""Exercise the exact pinned runsc source and broker visibility contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import urllib.request

from gateway.tee.model_sandbox_v2 import (
    MODEL_SANDBOX_BROKER_DIRECTORY,
    MODEL_SANDBOX_SOURCE_DIRECTORY,
    RunscSandboxConfigV2,
    _normalize_source_permissions,
    _oci_config,
    _runsc_run_command,
    _sandbox_visible_workspace,
)
from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes


def _pinned_runsc(destination: Path) -> Path:
    lock_path = Path("gateway/tee/runsc-runtime.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    installed = os.environ.get("LEADPOET_RUNSC_PROBE_BINARY")
    if installed:
        payload = Path(installed).read_bytes()
        if (
            len(payload) != int(lock["size_bytes"])
            or sha256_bytes(payload) != str(lock["sha256"])
            or hashlib.sha512(payload).hexdigest() != str(lock["sha512"])
        ):
            raise RuntimeError("installed pinned runsc artifact differs")
        return Path(installed)
    request = urllib.request.Request(
        str(lock["source_url"]),
        headers={"User-Agent": "leadpoet-runsc-input-probe-v2"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if (
        len(payload) != int(lock["size_bytes"])
        or sha256_bytes(payload) != str(lock["sha256"])
        or hashlib.sha512(payload).hexdigest() != str(lock["sha512"])
    ):
        raise RuntimeError("pinned runsc artifact differs")
    destination.write_bytes(payload)
    destination.chmod(0o555)
    return destination


def _serve_once(listener: socket.socket, served: threading.Event) -> None:
    connection, _ = listener.accept()
    with connection:
        if connection.recv(4) != b"ping":
            return
        connection.sendall(b"pong")
        served.set()


def main() -> int:
    if os.geteuid() != 0:
        raise RuntimeError("native runsc input probe must execute as root")
    if os.uname().machine not in {"x86_64", "amd64"}:
        raise RuntimeError("native runsc input probe requires an x86_64 host")
    with tempfile.TemporaryDirectory(
        prefix="leadpoet-runsc-input-probe-",
        dir="/var/tmp",
    ) as raw:
        host_root = Path(raw)
        host_root.chmod(0o755)
        runsc = _pinned_runsc(host_root / "runsc")
        config = RunscSandboxConfigV2(
            runsc_path=runsc,
            runsc_sha256=sha256_bytes(runsc.read_bytes()),
            rootfs_path=Path("/"),
            rootfs_manifest_hash="sha256:" + "0" * 64,
            python_path="/usr/bin/python3",
        )
        visible_parent = Path("/") / "leadpoet-model-sandboxes"
        visible_parent.mkdir(mode=0o711, exist_ok=True)
        visible_parent.chmod(0o711)
        with _sandbox_visible_workspace(config) as visible_root:
            source_root = visible_root / MODEL_SANDBOX_SOURCE_DIRECTORY
            source_root.mkdir(mode=0o755)
            (source_root / "source-marker").write_text(
                "source-visible\n",
                encoding="utf-8",
            )
            _normalize_source_permissions(source_root)

            broker_root = visible_root / MODEL_SANDBOX_BROKER_DIRECTORY
            broker_root.mkdir(mode=0o700)
            os.chown(broker_root, config.uid, config.gid)
            socket_path = broker_root / "provider.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(socket_path))
            listener.listen(1)
            os.chown(socket_path, config.uid, config.gid)
            socket_path.chmod(0o600)
            served = threading.Event()
            server = threading.Thread(
                target=_serve_once,
                args=(listener, served),
                daemon=True,
            )
            server.start()

            bundle = host_root / "bundle"
            runsc_root = host_root / "runsc-root"
            bundle.mkdir(mode=0o700)
            runsc_root.mkdir(mode=0o700)
            sandbox_id = "lp-rootfs-visible-inputs"
            script = (
                "import os,socket;from pathlib import Path;"
                "root=Path(os.environ['LEADPOET_MODEL_SOURCE_ROOT']);"
                "assert (root/'source-marker').read_text()=='source-visible\\n';"
                "client=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);"
                "client.connect(os.environ['LEADPOET_SANDBOX_PROVIDER_SOCKET']);"
                "client.sendall(b'ping');assert client.recv(4)==b'pong';"
                "client.close();print('ROOTFS_VISIBLE_INPUTS_OK')"
            )
            document = _oci_config(
                config=config,
                source_root=source_root,
                broker_root=broker_root,
                process_args=[config.python_path, "-c", script],
                environment={},
            )
            (bundle / "config.json").write_text(
                canonical_json(document),
                encoding="utf-8",
            )
            command = _runsc_run_command(
                config=config,
                runsc_root=runsc_root,
                bundle=bundle,
                sandbox_id=sandbox_id,
                host_uds=True,
            )
            try:
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=120,
                    env={"HOME": str(host_root), "PATH": "/usr/local/bin:/usr/bin:/bin"},
                    check=False,
                )
            finally:
                subprocess.run(
                    [
                        str(runsc),
                        "--root=" + str(runsc_root),
                        "delete",
                        "--force",
                        sandbox_id,
                    ],
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                listener.close()
            server.join(timeout=2)
            if (
                completed.returncode != 0
                or completed.stdout.strip() != "ROOTFS_VISIBLE_INPUTS_OK"
                or not served.is_set()
            ):
                raise RuntimeError(
                    "native runsc input probe failed "
                    + canonical_json(
                        {
                            "returncode": completed.returncode,
                            "stderr_hash": sha256_bytes(
                                str(completed.stderr or "").encode("utf-8")
                            ),
                            "stdout_hash": sha256_bytes(
                                str(completed.stdout or "").encode("utf-8")
                            ),
                        }
                    )
                )
    print("RUNSC_ROOTFS_VISIBLE_INPUTS_PROBE_SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
