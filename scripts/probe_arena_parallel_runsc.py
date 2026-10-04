#!/usr/bin/env python3
"""Measure isolated runsc ICPs and local worker sockets without Arena leases.

This small 256 MiB workload tests process concurrency, isolation, accounting,
failure independence, and cleanup. It does not size a real 2 GiB model workload.
Run only on an idle rootful Linux x86_64 probe host.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import contracts, runtime
from lab_arena.runtime_host import require_parallel_memory
from lab_arena.runner import RunState, WorkerSocketServer
from scripts._lab_arena_runsc_probe_ci import MODEL_OK, ProbeApi, make_spec, probe_work_dir

MEMORY_BYTES = 256 * 1024 * 1024
HOLD_SECONDS = 20  # Keep lightweight parallel processes alive for observation.


def _worker_count(value: str) -> int:
    count = int(value)
    if not 2 <= count <= 20:
        raise argparse.ArgumentTypeError("workers must be between 2 and 20")
    return count


def _source(index: int, *, parallel: bool) -> str:
    source = MODEL_OK.replace('"query": "probe"', f'"query": "probe-{index}"')
    source = source.replace('"Probe Co"', f'"Probe {index}"')
    if parallel:
        if index == 0:
            return source + f'\nimport time\ntime.sleep({HOLD_SECONDS})\nraise SystemExit("INTENTIONAL_FAILURE")\n'
        if index == 1:
            return source + '\nimport time\ntime.sleep(600)\n'
        return source + f'\nimport time\ntime.sleep({HOLD_SECONDS})\n'
    return source


def _one(engine, spec):
    api = ProbeApi()
    state = RunState(lease={"run_id": spec.sandbox_id}, lease_token="probe")
    server = WorkerSocketServer(spec.socket_path, api, state)
    start = time.monotonic()
    try:
        server.start()
        result = engine.run_icp(spec)
    finally:
        server.stop()
    frames = api.frames
    calls = state.calls
    assert len(frames) == len(calls) == 1
    frame = frames[0]
    assert frame["operation_id"] == "exa.search"
    assert calls[0]["actual_microusd"] == 5000
    assert calls[0]["call_identity"] == contracts.document_hash(frame)
    return {
        "exit_code": result.exit_code, "timed_out": result.timed_out,
        "output_error": result.output_error,
        "output": json.loads(result.output_bytes) if result.output_bytes else None,
        "call_identity": calls[0]["call_identity"],
        "actual_microusd": calls[0]["actual_microusd"],
        "wall_seconds": round(time.monotonic() - start, 3),
        "runsc_max_rss_bytes": result.max_rss_bytes,
    }


def _live_count(work: Path) -> int:
    live = 0
    for path in (work / "sandboxes").glob("*/sandbox.pid"):
        try:
            pid = int(path.read_text().strip())
            # The pid file persists until cleanup; count only live sandbox init.
            state = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1][0]
            live += state != "Z"
        except (OSError, ValueError, IndexError):
            continue
    return live


def _batch(engine, specs, *, parallel: bool, work: Path):
    high_water = 0
    stop = threading.Event()

    def sample():
        nonlocal high_water
        while not stop.wait(0.05):
            high_water = max(high_water, _live_count(work))

    observer = threading.Thread(target=sample, daemon=True)
    started = time.monotonic()
    observer.start()
    try:
        if parallel:
            with ThreadPoolExecutor(max_workers=len(specs)) as executor:
                results = list(executor.map(lambda spec: _one(engine, spec), specs))
        else:
            results = [_one(engine, spec) for spec in specs]
    finally:
        stop.set()
        observer.join(timeout=2)
        assert _live_count(work) == 0, "live sandbox remains after execution"
        assert not any((work / "sandboxes").iterdir()), "sandbox bundle remains after execution"
    return results, high_water, round(time.monotonic() - started, 3)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=_worker_count, default=20)
    parser.add_argument("--runsc-path", type=Path, default=Path("/usr/local/bin/runsc"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run:
        if os.geteuid() != 0:
            raise RuntimeError("runsc parallel probe requires root")
        runtime.require_linux_x86_64()
        require_parallel_memory(args.workers, MEMORY_BYTES)
    with probe_work_dir(dry_run=args.dry_run) as work:
        work.chmod(0o755)
        rootfs = Path("/") if args.dry_run else work / "rootfs"
        if not args.dry_run:
            for name in ("etc", "usr", "input", "output", "run/lab_arena", "agent", "model"):
                (rootfs / name).mkdir(parents=True, exist_ok=True)
            for name, target in (("bin", "usr/bin"), ("lib", "usr/lib"), ("lib64", "usr/lib64")):
                (rootfs / name).symlink_to(target)
            subprocess.run(["mount", "--bind", "/usr", str(rootfs / "usr")], check=True)
        socket_dirs = []
        try:
            if not args.dry_run:
                subprocess.run(["mount", "-o", "remount,bind,ro", str(rootfs / "usr")], check=True)
            batches = []
            for parallel in (False, True):
                specs = []
                for index in range(args.workers):
                    name = ("parallel" if parallel else "reference") + str(index)
                    spec = make_spec(work, name, _source(index, parallel=parallel),
                                     wall_clock=30 if parallel and index == 1 else 90,
                                     rootfs_path=rootfs)
                    socket_dirs.append(spec.socket_path.parent)
                    spec = replace(spec, memory_limit_bytes=MEMORY_BYTES)
                    assert runtime.oci_spec(spec)["linux"]["resources"]["memory"]["limit"] == MEMORY_BYTES
                    specs.append(spec)
                batches.append(specs)
            if args.dry_run:
                print(json.dumps({"workers": args.workers, "sandboxes": 2 * args.workers,
                                  "memory_limit_bytes_each": MEMORY_BYTES, "check": "dry_run_ok"}))
                return 0
            (work / "sandboxes").mkdir(mode=0o700)
            engine = runtime.RunscRuntime(runtime.RuntimeConfig(args.runsc_path, work / "sandboxes"))
            reference, sequential_high, sequential_wall = _batch(engine, batches[0], parallel=False, work=work)
            measured, parallel_high, parallel_wall = _batch(engine, batches[1], parallel=True, work=work)
            for index, (before, after) in enumerate(zip(reference, measured)):
                assert before["exit_code"] == 0 and not before["timed_out"]
                assert before["output"] is not None and before["output_error"] is None
                assert (before["call_identity"], before["actual_microusd"]) == (
                    after["call_identity"], after["actual_microusd"])
                if index == 0:
                    assert after["exit_code"] != 0 and not after["timed_out"]
                elif index == 1:
                    assert after["timed_out"]
                else:
                    assert after["exit_code"] == 0 and not after["timed_out"]
                    assert after["output_error"] is None and after["output"] == before["output"]
            print(json.dumps({"workers": args.workers, "observed_live_high_water": parallel_high,
                              "reference_live_high_water": sequential_high,
                              "reference_wall_seconds": sequential_wall,
                              "parallel_wall_seconds": parallel_wall,
                              "provider_calls": args.workers, "actual_microusd": 5000 * args.workers,
                              "per_icp": measured}, sort_keys=True), flush=True)
            assert parallel_high == args.workers, "all requested sandboxes did not overlap"
            print("ARENA_PARALLEL_RUNSC_PROBE_PASSED", flush=True)
            return 0
        finally:
            try:
                if not args.dry_run:
                    subprocess.run(["umount", "--", str(rootfs / "usr")], check=True)
            finally:
                for directory in socket_dirs:
                    shutil.rmtree(directory)


if __name__ == "__main__":
    raise SystemExit(main())
