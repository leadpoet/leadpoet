#!/usr/bin/env python3
"""Exercise the pinned production runsc OCI shape on a native Linux host."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import urllib.request

from gateway.tee.model_sandbox_v2 import RunscSandboxConfigV2, _oci_config


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "gateway/tee/runsc-runtime.lock.json"


def _download_runsc(target: Path) -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    with urllib.request.urlopen(lock["source_url"], timeout=120) as response:
        data = response.read()
    observed = "sha256:" + hashlib.sha256(data).hexdigest()
    if len(data) != lock["size_bytes"] or observed != lock["sha256"]:
        raise RuntimeError("pinned runsc download differs")
    target.write_bytes(data)
    target.chmod(0o755)


def _serve_once(path: Path, ready: threading.Event) -> None:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(path))
        os.chown(path, 65534, 65534)
        path.chmod(0o600)
        server.listen(1)
        ready.set()
        connection, _ = server.accept()
        with connection:
            connection.sendall(b"pong")
    finally:
        server.close()


def _run_variant(runsc: Path, root: Path, *, delegated: bool) -> dict[str, object]:
    broker_root = root / ("broker-delegated" if delegated else "broker-default")
    broker_root.mkdir(mode=0o700)
    os.chown(broker_root, 65534, 65534)
    socket_path = broker_root / "provider.sock"
    ready = threading.Event()
    server = threading.Thread(target=_serve_once, args=(socket_path, ready), daemon=True)
    server.start()
    if not ready.wait(timeout=10):
        raise RuntimeError("probe socket did not start")

    source_root = root / "source"
    source_root.mkdir(exist_ok=True)
    bundle = root / ("bundle-delegated" if delegated else "bundle-default")
    bundle.mkdir()
    runsc_root = root / ("runsc-delegated" if delegated else "runsc-default")
    runsc_root.mkdir()
    sandbox_id = "lp-cgroup-delegated" if delegated else "lp-cgroup-default"
    config = RunscSandboxConfigV2(
        runsc_path=runsc,
        runsc_sha256="sha256:" + "0" * 64,
        rootfs_path=Path("/"),
        rootfs_manifest_hash="sha256:" + "0" * 64,
        python_path="/usr/bin/python3",
    )
    document = _oci_config(
        config=config,
        source_root=source_root,
        broker_root=broker_root,
        process_args=[
            "/usr/bin/python3",
            "-c",
            (
                "import os,socket;"
                "s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);"
                "s.connect(os.environ['LEADPOET_SANDBOX_PROVIDER_SOCKET']);"
                "assert s.recv(4)==b'pong';print('ok')"
            ),
        ],
        environment={},
    )
    if delegated:
        document["linux"]["cgroupsPath"] = "leadpoet-model/" + sandbox_id
    (bundle / "config.json").write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    command = [
        str(runsc),
        "--root=" + str(runsc_root),
        "--rootless=true",
        "--network=none",
        "--host-uds=open",
        "--platform=ptrace",
        "run",
        "--bundle=" + str(bundle),
        sandbox_id,
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=120)
    subprocess.run(
        [str(runsc), "--root=" + str(runsc_root), "delete", "--force", sandbox_id],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return {
        "delegated": delegated,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip()[:2000],
    }


def main() -> int:
    if os.geteuid() != 0:
        raise RuntimeError("native runsc probe must execute as root")
    with tempfile.TemporaryDirectory(prefix="leadpoet-runsc-probe-") as raw_root:
        root = Path(raw_root)
        root.chmod(0o755)
        runsc = root / "runsc"
        _download_runsc(runsc)
        default = _run_variant(runsc, root, delegated=False)
        delegated = _run_variant(runsc, root, delegated=True)
        print(json.dumps({"default": default, "delegated": delegated}, sort_keys=True))
        if delegated["returncode"] != 0 or delegated["stdout"] != "ok":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
