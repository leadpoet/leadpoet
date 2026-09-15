#!/usr/bin/env python3
"""Short real-gVisor checkpoint probe in an operator-supplied isolated workdir.

Run on a Linux x86_64 root host with the installed trusted Python rootfs:

  python3 scripts/probe_arena_checkpoint_gvisor.py \
    --runsc /usr/local/bin/runsc --rootfs /path/to/trusted/rootfs \
    --work-dir /tmp/arena-checkpoint-proof

The probe uses no providers, database, Arena API, or live model submission.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab_arena import contracts, runtime
from lab_arena.output import OutputInvalid, output_document_from_bytes


VALID = b'{"companies":[]}'


def _valid_output(candidate: bytes) -> bool:
    try:
        output_document_from_bytes(candidate)
    except OutputInvalid:
        return False
    return True


def _trusted_copy(source: Path, destination: Path) -> Path:
    shutil.copyfile(source, destination)
    destination.chmod(0o444)
    return destination


def _run_case(config: runtime.RuntimeConfig, rootfs: Path, name: str, harness: str,
              *, inject_late: bool) -> runtime.SandboxResult:
    case = Path(tempfile.mkdtemp(prefix="checkpoint-%s-" % name, dir=config.work_dir))
    socket_handle = socket.socket(socket.AF_UNIX)
    try:
        source = case / "source"
        deps = case / "deps"
        inp = case / "input"
        out = case / "output"
        sockets = case / "sockets"
        for directory in (source, deps, inp, out, sockets):
            directory.mkdir(mode=0o755)
        (source / "harness.py").write_text(harness, encoding="utf-8")
        (inp / runtime.INPUT_FILE_NAME).write_text(
            json.dumps({"icp": {}, "company_limit": 5}), encoding="utf-8",
        )
        socket_path = sockets / runtime.SANDBOX_SOCKET_NAME
        if len(str(socket_path).encode()) >= 100:
            raise RuntimeError("probe workdir is too long for the worker socket")
        socket_handle.bind(str(socket_path))
        spec = runtime.SandboxSpec(
            sandbox_id="checkpoint-%s" % name,
            rootfs_path=rootfs, input_dir=inp, output_dir=out,
            socket_path=socket_path, source_dir=source,
            dependency_dir=deps,
            agent_entrypoint_path=_trusted_copy(
                ROOT / "lab_arena" / "agent_entrypoint.py", case / "entrypoint.py",
            ),
            checkpoint_module_path=_trusted_copy(
                ROOT / "lab_arena" / "lab_arena_checkpoint.py",
                case / "lab_arena_checkpoint.py",
            ),
            entry_command=runtime.AGENT_ENTRY_COMMAND,
            working_dir=runtime.AGENT_WORKING_DIR,
            evaluation_date="2026-09-15", random_seed=1,
            wall_clock_seconds=5,
            checkpoint_deadline_policy=contracts.CHECKPOINT_DEADLINE_POLICY,
            checkpoint_validator=_valid_output,
        )
        launcher_processes = []
        def runner(argv, **kwargs):
            if inject_late and "kill" in argv and "--root=" in " ".join(argv):
                # Host-side race simulation: after the monotonic cutoff, before
                # runsc receives KILL, the output becomes valid. It must not
                # replace the snapshot frozen before cleanup.
                spec.output_path.write_bytes(VALID)
            if "run" in argv and any(token.startswith("--bundle=") for token in argv):
                launcher = runtime._RusagePopen(argv, **kwargs)
                launcher_processes.append(launcher)
                return launcher
            return subprocess.Popen(argv, **kwargs)

        result = runtime.RunscRuntime(config).run_icp(
            spec, process_runner=runner,
        )
        if (len(launcher_processes) != 1
                or launcher_processes[0].child_rusage is None
                or result.cpu_seconds != launcher_processes[0].child_rusage[0]
                or result.max_rss_bytes != launcher_processes[0].child_rusage[1]):
            raise RuntimeError("probe did not capture this launcher child's wait4 usage")
        return result
    finally:
        socket_handle.close()
        shutil.rmtree(case, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runsc", type=Path, required=True)
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.work_dir.is_dir() or not args.rootfs.is_dir():
        parser.error("isolated workdir and trusted rootfs must already exist")
    config = runtime.RuntimeConfig(runsc_path=args.runsc, work_dir=args.work_dir)
    valid_then_malformed = (
        "import time\n"
        "def run_icp(icp):\n"
        "    import lab_arena_checkpoint\n"
        "    lab_arena_checkpoint.write([])\n"
        "    time.sleep(1)\n"
        "    open('/output/companies.json','w').write('{')\n"
        "    while True: time.sleep(.1)\n"
    )
    empty_until_kill = (
        "import time\n"
        "def run_icp(icp):\n"
        "    while True: time.sleep(.1)\n"
    )
    first = _run_case(config, args.rootfs, "valid", valid_then_malformed,
                      inject_late=True)
    second = _run_case(config, args.rootfs, "late", empty_until_kill,
                       inject_late=True)
    if not (first.timed_out and first.output_bytes == VALID
            and second.timed_out and second.output_bytes is None):
        raise RuntimeError("real gVisor checkpoint cutoff probe failed")
    print("real gVisor checkpoint cutoff probe passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
