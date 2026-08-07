"""Identity-scoped shutdown for gateway-managed host processes.

Replaces the global name-pattern ``pkill -9 -f ...`` block in
``gw_restart.sh``. Those patterns killed by argv substring across the entire
host, with no notion of which checkout owned the process. On a host that also
runs CI builds and restart rehearsals of these same scripts, that is how the
production gateway died on 2026-08-02 02:48Z: a SIGKILL-class stop with no
shutdown marker, mid-write, while repeated restart-controller CI fixture jobs
were failing on the shared runner — followed by fourteen hours with no
published weight bundle.

This module terminates a managed component only when ALL of the following
hold, mirroring the discipline ``host_memory_guard_v2`` already applies in the
opposite direction (it may only kill inside disposable test roots; we may only
kill inside managed production roots):

- the process belongs to the invoking uid;
- its argv matches a known managed-component pattern;
- its current working directory is under one of the ``--root`` directories
  passed by the launcher; and
- its identity snapshot (pid, start ticks, uid, cwd, argv) is unchanged at
  the moment of each signal, so a recycled pid is never signalled.

Termination is SIGTERM first, then SIGKILL only for processes that survive
the bounded grace period. Matching processes OUTSIDE the managed roots are
reported but never signalled — a CI checkout or rehearsal tree running the
same argv must survive a production shutdown, and vice versa.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time


class ScopedShutdownV2Error(RuntimeError):
    pass


# Substring patterns identifying the components gw_restart.sh manages. These
# are matched against the space-joined argv, exactly like the pkill -f
# patterns they replace. Scope safety comes from the cwd check, not from
# making these narrower.
MANAGED_COMPONENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("gateway.main", "python3 main.py"),
    ("gateway.main", "python3 -u main.py"),
    ("gateway.main", "-m gateway.main"),
    ("gateway.main", "uvicorn"),
    ("research_lab.worker", "gateway/research_lab/worker_process.py"),
    ("research_lab.worker", "run_research_lab_hosted_worker"),
    ("research_lab.scoring_worker", "run_research_lab_scoring_worker"),
    ("provider_evidence_proxy", "gateway.research_lab.provider_evidence_proxy"),
    ("provider_evidence_proxy", "provider_evidence_proxy"),
    ("tee_inter_enclave_relay", "gateway.utils.tee_inter_enclave_relay"),
    ("tee_egress_forwarder", "gateway.utils.tee_egress_forwarder"),
)


def _read_process(proc_root: Path, pid: int) -> dict | None:
    process_root = proc_root / str(pid)
    try:
        status = (process_root / "status").read_text(encoding="utf-8")
        stat_fields = (process_root / "stat").read_text(encoding="utf-8").split()
        argv = tuple(
            value.decode("utf-8", errors="replace")
            for value in (process_root / "cmdline").read_bytes().split(b"\0")
            if value
        )
        cwd = Path(os.readlink(process_root / "cwd"))
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None

    fields = {}
    for line in status.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key] = value.strip()
    try:
        uid = int(fields["Uid"].split()[0])
        start_ticks = int(stat_fields[21])
    except (KeyError, IndexError, ValueError):
        return None
    return {
        "pid": pid,
        "uid": uid,
        "start_ticks": start_ticks,
        "cwd": cwd,
        "argv": argv,
    }


def _processes(proc_root: Path) -> list[dict]:
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise ScopedShutdownV2Error(
            f"cannot inspect process table: {exc}"
        ) from exc
    found = []
    for entry in entries:
        if entry.name.isdigit():
            snapshot = _read_process(proc_root, int(entry.name))
            if snapshot is not None:
                found.append(snapshot)
    return found


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _component_for(argv: tuple[str, ...]) -> str | None:
    joined = " ".join(argv)
    for component, pattern in MANAGED_COMPONENT_PATTERNS:
        if pattern in joined:
            return component
    return None


def _same_process(proc_root: Path, expected: dict) -> bool:
    current = _read_process(proc_root, expected["pid"])
    return current is not None and (
        current["start_ticks"],
        current["uid"],
        current["cwd"],
        current["argv"],
    ) == (
        expected["start_ticks"],
        expected["uid"],
        expected["cwd"],
        expected["argv"],
    )


def shutdown_managed_processes(
    *,
    roots: list[Path],
    proc_root: Path = Path("/proc"),
    terminate_timeout_seconds: float = 10.0,
    kill: "callable" = os.kill,
    sleep: "callable" = time.sleep,
    monotonic: "callable" = time.monotonic,
) -> dict:
    if not roots:
        raise ScopedShutdownV2Error("at least one managed --root is required")
    resolved_roots = []
    for root in roots:
        try:
            resolved_roots.append(Path(root).resolve(strict=True))
        except OSError as exc:
            raise ScopedShutdownV2Error(
                f"managed root is unavailable: {root}: {exc}"
            ) from exc

    uid = os.getuid()
    self_pids = {os.getpid(), os.getppid()}

    selected: list[dict] = []
    out_of_scope: list[dict] = []
    for process in _processes(proc_root):
        if process["pid"] in self_pids or process["uid"] != uid:
            continue
        component = _component_for(process["argv"])
        if component is None:
            continue
        process["component"] = component
        if any(_is_under(process["cwd"], root) for root in resolved_roots):
            selected.append(process)
        else:
            # Same argv shape, different owner tree: a CI checkout or a
            # rehearsal root. Reported for visibility, never signalled.
            out_of_scope.append(process)

    for process in selected:
        if not _same_process(proc_root, process):
            process["outcome"] = "exited_before_sigterm"
            continue
        try:
            kill(process["pid"], signal.SIGTERM)
        except ProcessLookupError:
            process["outcome"] = "exited_before_sigterm"

    deadline = monotonic() + terminate_timeout_seconds
    remaining = [
        process for process in selected if "outcome" not in process
    ]
    while remaining and monotonic() < deadline:
        sleep(0.1)
        remaining = [
            process
            for process in remaining
            if _same_process(proc_root, process)
        ]

    for process in selected:
        if "outcome" in process:
            continue
        if _same_process(proc_root, process):
            try:
                kill(process["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            process["outcome"] = "sigkill_forced"
        else:
            process["outcome"] = "terminated"

    return {
        "schema_version": "leadpoet.gateway_scoped_shutdown.v2",
        "managed_roots": [str(root) for root in resolved_roots],
        "terminated": [
            {
                "pid": process["pid"],
                "component": process["component"],
                "cwd": str(process["cwd"]),
                "outcome": process["outcome"],
            }
            for process in selected
        ],
        "out_of_scope_matches": [
            {
                "pid": process["pid"],
                "component": process["component"],
                "cwd": str(process["cwd"]),
            }
            for process in out_of_scope
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Terminate gateway-managed processes whose cwd is under a managed "
            "root. Never signals matching processes outside those roots."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        default=[],
        help="Managed root directory (repeatable, at least one required).",
    )
    parser.add_argument(
        "--terminate-timeout-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--proc-root",
        default="/proc",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if not args.roots:
        parser.error("at least one --root is required")
    if args.terminate_timeout_seconds < 1.0:
        parser.error("--terminate-timeout-seconds must be at least 1.0")
    try:
        report = shutdown_managed_processes(
            roots=[Path(root) for root in args.roots],
            proc_root=Path(args.proc_root),
            terminate_timeout_seconds=args.terminate_timeout_seconds,
        )
    except ScopedShutdownV2Error as exc:
        print(f"scoped_shutdown_v2 failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
