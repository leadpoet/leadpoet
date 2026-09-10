#!/usr/bin/env python3
"""Exercise the Lab Arena sandbox contract under runsc
(labarena.md sections 3.3, 9.2, 18.4).

Runs as root on a Linux x86_64 host (the deploy-checks lane, not pytest):
downloads the current official runsc when needed, then executes four sandboxes through
``lab_arena.runtime.RunscRuntime`` with the host root as the read-only
image:

1. A model that reads ``/input/icp.json``, calls one closed operation over
   the worker socket, proves the root is read-only, proves raw TCP reaches
   nothing, and writes ``/output/companies.json``.
2. A model that sleeps past the wall clock: it must be killed and report a
   timeout with the sandbox deleted afterwards.
3. A model that writes an oversized output: it must be rejected by the
   bounded reader.
4. The production agent entrypoint loads an admitted ``harness.py`` through
   the source and dependency mounts and writes an empty company list.

``--dry-run`` validates the OCI documents and prints the commands without
executing anything, which is what the repository test exercises.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import socket
import sys
import tempfile
import tarfile
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import contracts, runtime, shim  # noqa: E402
from lab_arena.runner import RunState, WorkerSocketServer, _stage_agent_entrypoint  # noqa: E402

MODEL_OK = r'''
import json, os, socket
icp = json.load(open("/input/icp.json"))["icp"]
assert icp["prompt"] == "probe"
try:
    open("/etc/lab-arena-probe-write", "w").write("x")
    raise SystemExit("ROOT_WRITABLE")
except OSError:
    pass
try:
    s = socket.create_connection(("8.8.8.8", 443), timeout=3)
    s.close()
    raise SystemExit("NETWORK_REACHABLE")
except OSError:
    pass
import lab_arena.shim as shim
status, headers, body = shim.dispatch("exa.search", {"query": "probe"}, 5000)
assert status == 200, status
json.dump({"companies": [{"company_name": "Probe Co", "company_website": "https://probe.example.com", "industry": "Software", "employee_count": "51-200", "country": "United States", "intent_signals": [{"source": "news", "description": "probe", "url": "https://news.example.com/1", "date": "2026-08-01", "snippet": "probe", "matched_icp_signal": 0}]}]}, open("/output/companies.json", "w"))
print("LAB_ARENA_MODEL_OK")
'''
MODEL_SLEEP = "import time\ntime.sleep(600)\n"
MODEL_BIG = "open('/output/companies.json', 'w').write('[' + ('1,' * 400000) + '1]')\n"
AGENT_HARNESS = "def run_icp(icp: dict) -> list[dict]:\n    assert icp == {'prompt': 'probe', 'max_companies': 5}\n    return []\n"
RUNSC_DOWNLOAD_URL = "https://storage.googleapis.com/gvisor/releases/release/latest/x86_64/gvisor.tar.bz2"


class ProbeApi:
    """Answers provider frames locally; the probe has no service."""

    def __init__(self) -> None:
        self.frames = []

    def provider(self, run_id, lease_token, frame):
        self.frames.append(frame)
        body = json.dumps({"results": [{"url": "https://probe.example.com"}]}).encode()
        import base64

        return {"status": 200, "headers": {"content-type": "application/json"}, "body_b64": base64.b64encode(body).decode(), "call": {"call_identity": contracts.document_hash(frame), "operation_id": frame["operation_id"], "reserved_microusd": 5000, "actual_microusd": 5000, "outcome": "settled", "status": 200, "request_hash": contracts.document_hash(frame["parameters"]), "response_hash": contracts.hash_bytes(body)}}


def probe_runsc(destination: Path) -> Path:
    if destination.exists():
        runtime.require_runsc_executable(destination)
        return destination
    # Current gVisor releases require companion binaries next to runsc.
    # Fetch and verify the official bundle instead of a standalone executable.
    def download(url):
        request = urllib.request.Request(url, headers={"User-Agent": "leadpoet-lab-arena-runsc-probe/1"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()

    data = download(RUNSC_DOWNLOAD_URL)
    expected = download(RUNSC_DOWNLOAD_URL + ".sha512").decode("ascii").split()[0]
    if hashlib.sha512(data).hexdigest() != expected:
        raise RuntimeError("gVisor bundle checksum differs")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError("gVisor bundle contains an unsafe path")
        archive.extractall(destination.parent, members=members)
    installed = destination.parent / "runsc"
    if installed != destination:
        raise RuntimeError("gVisor probe destination must be named runsc")
    runtime.require_runsc_executable(destination)
    return destination


def make_spec(work: Path, name: str, model_source: str, *, wall_clock: int) -> runtime.SandboxSpec:
    model_dir = work / name / "model"
    input_dir = work / name / "input"
    output_dir = work / name / "output"
    socket_dir = Path(tempfile.mkdtemp(prefix="la", dir="/tmp"))
    for directory in (model_dir, input_dir, output_dir):
        directory.mkdir(parents=True)
    (model_dir / "main.py").write_text(model_source, encoding="utf-8")
    (input_dir / runtime.INPUT_FILE_NAME).write_text(json.dumps({"schema_version": "leadpoet.lab_arena.icp_input.v1", "icp": {"prompt": "probe", "max_companies": 5}, "evaluation_date": "2026-09-02", "company_limit": 5, "provider_operations": sorted(__import__("lab_arena.operations", fromlist=["OPERATIONS"]).OPERATIONS)}), encoding="utf-8")
    os.chmod(output_dir, 0o777)
    return runtime.SandboxSpec(
        sandbox_id="lab-arena-probe-%s" % name, rootfs_path=Path("/"), input_dir=input_dir, output_dir=output_dir,
        # The probe runs on the host root filesystem, so the model's host path is its path inside the sandbox.
        socket_path=socket_dir / runtime.SANDBOX_SOCKET_NAME, entry_command=("python3", str(model_dir / "main.py")), evaluation_date="2026-09-02", random_seed=7, wall_clock_seconds=wall_clock,
    )


def make_agent_spec(work: Path) -> runtime.SandboxSpec:
    source_dir = work / "agent" / "source"
    dependency_dir = work / "agent" / "deps"
    input_dir = work / "agent" / "input"
    output_dir = work / "agent" / "output"
    run_dir = work / "agent" / "run"
    socket_dir = work / "agent" / "socket"
    for directory in (source_dir, dependency_dir, input_dir, output_dir, run_dir, socket_dir):
        directory.mkdir(parents=True)
    (source_dir / "harness.py").write_text(AGENT_HARNESS, encoding="utf-8")
    (input_dir / runtime.INPUT_FILE_NAME).write_text(json.dumps({"schema_version": "leadpoet.lab_arena.icp_input.v1", "icp": {"prompt": "probe", "max_companies": 5}, "evaluation_date": "2026-09-02", "company_limit": 5, "provider_operations": sorted(__import__("lab_arena.operations", fromlist=["OPERATIONS"]).OPERATIONS)}), encoding="utf-8")
    entrypoint = _stage_agent_entrypoint(ROOT / "lab_arena" / "agent_entrypoint.py", run_dir)
    return runtime.SandboxSpec(
        sandbox_id="lab-arena-probe-agent", rootfs_path=Path("/"), input_dir=input_dir, output_dir=output_dir,
        socket_path=socket_dir / runtime.SANDBOX_SOCKET_NAME, source_dir=source_dir, dependency_dir=dependency_dir,
        agent_entrypoint_path=entrypoint, entry_command=runtime.AGENT_ENTRY_COMMAND, working_dir=runtime.AGENT_WORKING_DIR,
        evaluation_date="2026-09-02", random_seed=7, wall_clock_seconds=60,
    )


def run_probe(*, dry_run: bool, runsc_path: Path = None) -> int:
    if not dry_run:
        if os.geteuid() != 0:
            raise RuntimeError("the Lab Arena runsc probe must execute as root")
        runtime.require_linux_x86_64()
    with tempfile.TemporaryDirectory(prefix="lab-arena-runsc-probe-", dir="/var/tmp" if os.path.isdir("/var/tmp") else None) as raw:
        work = Path(raw)
        work.chmod(0o755)
        specs = {
            "ok": make_spec(work, "ok", MODEL_OK, wall_clock=120),
            "timeout": make_spec(work, "timeout", MODEL_SLEEP, wall_clock=30),
            "big": make_spec(work, "big", MODEL_BIG, wall_clock=60),
            "agent": make_agent_spec(work),
        }
        for name, spec in specs.items():
            document = runtime.oci_spec(spec)
            assert document["root"]["readonly"] is True and document["process"]["user"]["uid"] == runtime.SANDBOX_UID
            assert "network" not in json.dumps(document.get("linux", {}).get("namespaces", []))
            config = runtime.RuntimeConfig(runsc_path=runsc_path or work / "runsc", work_dir=work / "sandboxes")
            bundle = work / ("bundle-" + name)
            command = runtime.runsc_run_command(
                config,
                work / "runsc-root",
                bundle,
                spec.sandbox_id,
                pid_file=bundle / "sandbox.pid",
            )
            assert (
                "--network=none" in command
                and "--rootless=false" in command
                and "--pid-file=%s" % (bundle / "sandbox.pid") in command
            )
            print("PLAN", name, json.dumps({"command": command, "wall_clock_seconds": spec.wall_clock_seconds, "entry": list(spec.argv)}))
        if dry_run:
            print("LAB_ARENA_RUNSC_PROBE_DRY_RUN_OK")
            return 0
        runsc = runtime.require_runsc_executable(runsc_path) if runsc_path else probe_runsc(work / "runsc")
        sandbox_work = work / "sandboxes"
        sandbox_work.mkdir(mode=0o700)
        config = runtime.RuntimeConfig(runsc_path=runsc, work_dir=sandbox_work)
        sandbox_runtime = runtime.RunscRuntime(config)
        api = ProbeApi()
        outcomes = {}
        for name, spec in specs.items():
            state = RunState(lease={"run_id": "probe-" + name}, lease_token="probe")
            server = WorkerSocketServer(spec.socket_path, api, state)
            server.start()
            try:
                result = sandbox_runtime.run_icp(spec)
            finally:
                server.stop()
            outcomes[name] = {"exit_code": result.exit_code, "timed_out": result.timed_out, "has_output": result.output_bytes is not None, "output_error": result.output_error, "stdout": result.stdout.decode(errors="replace")[-200:], "output": json.loads(result.output_bytes) if result.output_bytes is not None else None}
        checks = {
            "ok_exit": outcomes["ok"]["exit_code"] == 0 and "LAB_ARENA_MODEL_OK" in outcomes["ok"]["stdout"] and outcomes["ok"]["has_output"],
            "ok_provider_call": len(api.frames) == 1 and api.frames[0]["operation_id"] == "exa.search",
            "timeout_killed": outcomes["timeout"]["timed_out"] is True,
            "big_rejected": outcomes["big"]["output_error"] is not None,
            "agent_entrypoint": outcomes["agent"]["exit_code"] == 0 and outcomes["agent"]["output"] == {"companies": []},
            "agent_no_provider_call": len(api.frames) == 1,
        }
        print(json.dumps({"outcomes": outcomes, "checks": checks}))
        if not all(checks.values()):
            raise RuntimeError("Lab Arena runsc probe failed: %s" % json.dumps(checks))
    print("LAB_ARENA_RUNSC_PROBE_SUCCESS")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Lab Arena pinned-runsc sandbox probe")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runsc-path", type=Path, help="probe this installed runtime and its companion binaries")
    args = parser.parse_args(argv)
    return run_probe(dry_run=bool(args.dry_run), runsc_path=args.runsc_path)


if __name__ == "__main__":
    raise SystemExit(main())
