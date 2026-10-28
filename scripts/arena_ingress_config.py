#!/usr/bin/env python3
"""Render the fixed Caddy ingress for release-scoped Arena overlap."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


ARENA_EDGE_PORTS = (8890, 8891)
BACKENDS = {"legacy": 8000, "ready": 8001, "closed": None}
DATABASE_HANDLER = (
    "    @database path /rest/v1/*\n"
    "    handle @database {\n"
    "        reverse_proxy 127.0.0.1:8081\n"
    "    }\n"
)
BASELINE_CADDY_SHA256 = "feb10ab28f61a2bbcbfd0a75f6e16d53ecf7951bec5e94373f5958edc3d901dc"
BASELINE_CADDY = b"gateway.subnet71.com {\n\treverse_proxy localhost:8000\n}\n"
_SHA = re.compile(r"^[0-9a-f]{40}$")


def render(*, arena_edge_port: int, backend: str, direct: bool) -> str:
    if arena_edge_port not in ARENA_EDGE_PORTS:
        raise ValueError("Arena edge port is not an approved loopback port")
    if backend not in BACKENDS or (direct and backend == "legacy"):
        raise ValueError("gateway backend state is invalid for this ingress")

    def site(name: str) -> str:
        upstream = BACKENDS[backend]
        gateway_handler = (
            f"reverse_proxy 127.0.0.1:{upstream}"
            if upstream is not None
            else 'respond "gateway temporarily unavailable" 503'
        )
        return (
            f"{name} {{\n"
            f"{DATABASE_HANDLER if name == 'gateway.subnet71.com' else ''}"
            "    @arena path /arena /arena/* /testnet/arena /testnet/arena/*\n"
            "    handle @arena {\n"
            f"        reverse_proxy 127.0.0.1:{arena_edge_port}\n"
            "    }\n"
            "    handle {\n"
            f"        {gateway_handler}\n"
            "    }\n"
            "}\n"
        )

    config = site("gateway.subnet71.com")
    if direct:
        config += "\n" + site(":8000")
    return config


def slot_ports(slot: int) -> tuple[int, int]:
    if type(slot) is not int or slot not in (0, 1):
        raise ValueError("Arena ingress slot is invalid")
    return (8792, 8890) if slot == 0 else (8794, 8891)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def inspect(state_file: Path, caddy_file: Path) -> dict[str, object]:
    """Treat only the exact baseline or our own exact config as authority."""
    caddy_metadata = caddy_file.lstat()
    if not stat.S_ISREG(caddy_metadata.st_mode) or caddy_metadata.st_uid != os.getuid():
        raise ValueError("Caddy config is not an owned regular file")
    current = caddy_file.read_bytes()
    if not state_file.exists() and not state_file.is_symlink():
        if _sha256(current) != BASELINE_CADDY_SHA256:
            raise ValueError("Caddy config differs from the exact legacy baseline")
        return {"mode": "legacy", "slot": 0, "sha": "", "backend": "legacy"}
    metadata = state_file.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077 or metadata.st_size > 4096):
        raise ValueError("Arena ingress state is not an owned private file")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if (not isinstance(state, dict) or set(state) != {"version", "mode", "slot", "sha", "backend"}
            or type(state["version"]) is not int or state["version"] not in (1, 2)
            or state["mode"] not in ("domain", "dual")
            or type(state["slot"]) is not int or state["slot"] not in (0, 1)
            or not isinstance(state["sha"], str) or not _SHA.fullmatch(state["sha"])
            or state["backend"] not in BACKENDS
            or (state["mode"] == "domain" and state["backend"] != "legacy")
            or (state["mode"] == "dual" and state["backend"] == "legacy")):
        raise ValueError("Arena ingress state is invalid")
    expected = render(
        arena_edge_port=slot_ports(state["slot"])[1],
        backend=state["backend"], direct=state["mode"] == "dual",
    )
    if state["version"] == 1:
        # Version 1 predates the fixed HTTPS database route. Accept only its
        # exact former renderer, never an arbitrary pre-migration Caddy edit.
        expected = expected.replace(DATABASE_HANDLER, "", 1)
    expected = expected.encode("utf-8")
    if current != expected:
        raise ValueError("Caddy config differs from the recorded Arena ingress")
    return state


def commit_state(
    state_file: Path, caddy_file: Path, *, mode: str, slot: int,
    sha: str, backend: str,
) -> None:
    """Record a route only after Caddy's on-disk config exactly matches it."""
    if mode not in ("domain", "dual") or not _SHA.fullmatch(sha):
        raise ValueError("Arena ingress state is invalid")
    if (mode == "domain" and backend != "legacy") or (mode == "dual" and backend == "legacy"):
        raise ValueError("Arena ingress backend state is invalid")
    config = render(
        arena_edge_port=slot_ports(slot)[1], backend=backend,
        direct=mode == "dual",
    ).encode("utf-8")
    if caddy_file.read_bytes() != config:
        raise ValueError("Caddy config does not match the proposed Arena ingress")
    state = {"version": 2, "mode": mode, "slot": slot, "sha": sha, "backend": backend}
    state_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".arena-ingress.", dir=state_file.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, state_file)
        directory = os.open(state_file.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def activate(
    state_file: Path, caddy_file: Path, *, expected_mode: str,
    expected_slot: int, expected_sha: str, expected_backend: str,
    mode: str, slot: int, sha: str, backend: str,
) -> None:
    """Validate, reload, and record one exact Caddy routing transition."""
    lock_path = state_file.with_suffix(".lock")
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        previous = inspect(state_file, caddy_file)
        if previous != {
            "mode": expected_mode, "slot": expected_slot,
            "sha": "" if expected_sha == "-" else expected_sha,
            "backend": expected_backend,
            **({"version": previous.get("version")} if expected_mode != "legacy" else {}),
        }:
            raise ValueError("Arena ingress changed before routing transition")
        if mode == "legacy":
            if slot != 0 or sha != "-" or backend != "legacy":
                raise ValueError("legacy Caddy restoration is invalid")
            next_config = BASELINE_CADDY
        else:
            next_config = render(
                arena_edge_port=slot_ports(slot)[1], backend=backend,
                direct=mode == "dual",
            ).encode("utf-8")
        previous_config = caddy_file.read_bytes()
        previous_state = state_file.read_bytes() if state_file.exists() else None
        next_state = None if mode == "legacy" else (
            json.dumps({"version": 2, "mode": mode, "slot": slot, "sha": sha,
                        "backend": backend}, sort_keys=True) + "\n"
        ).encode("utf-8")
        with tempfile.NamedTemporaryFile(
            prefix=".arena-caddy.", dir=caddy_file.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(next_config)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary, 0o644)
            subprocess.run(
                ["caddy", "validate", "--config", str(temporary), "--adapter", "caddyfile"],
                check=True, capture_output=True, timeout=20,
            )
            if caddy_file.read_bytes() != previous_config:
                raise ValueError("Caddy config changed before routing transition")
            os.replace(temporary, caddy_file)
            try:
                subprocess.run(
                    ["caddy", "reload", "--config", str(caddy_file), "--adapter", "caddyfile"],
                    check=True, capture_output=True, timeout=20,
                )
                if mode == "legacy":
                    state_file.unlink()
                else:
                    commit_state(state_file, caddy_file, mode=mode, slot=slot, sha=sha, backend=backend)
            except BaseException:
                if caddy_file.read_bytes() != next_config:
                    raise ValueError("Caddy config changed during rollback; retaining live route")
                current_state = state_file.read_bytes() if state_file.exists() else None
                if current_state not in (previous_state, next_state):
                    raise ValueError("Arena ingress state changed during rollback; retaining live route")
                with tempfile.NamedTemporaryFile(
                    prefix=".arena-caddy-rollback.", dir=caddy_file.parent, delete=False,
                ) as rollback:
                    rollback_path = Path(rollback.name)
                    rollback.write(previous_config)
                    rollback.flush()
                    os.fsync(rollback.fileno())
                os.chmod(rollback_path, 0o644)
                os.replace(rollback_path, caddy_file)
                subprocess.run(
                    ["caddy", "reload", "--config", str(caddy_file), "--adapter", "caddyfile"],
                    check=True, capture_output=True, timeout=20,
                )
                if current_state != previous_state:
                    if previous_state is None:
                        state_file.unlink()
                    else:
                        with tempfile.NamedTemporaryFile(
                            prefix=".arena-state-rollback.", dir=state_file.parent,
                            delete=False,
                        ) as state_rollback:
                            state_rollback_path = Path(state_rollback.name)
                            state_rollback.write(previous_state)
                            state_rollback.flush()
                            os.fsync(state_rollback.fileno())
                        os.chmod(state_rollback_path, 0o600)
                        os.replace(state_rollback_path, state_file)
                raise
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-edge-port", type=int, choices=ARENA_EDGE_PORTS)
    parser.add_argument("--backend", choices=sorted(BACKENDS))
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--caddy-file", type=Path)
    parser.add_argument("--mode", choices=("legacy", "domain", "dual"))
    parser.add_argument("--slot", type=int, choices=(0, 1))
    parser.add_argument("--sha")
    parser.add_argument("--expect-mode")
    parser.add_argument("--expect-slot", type=int)
    parser.add_argument("--expect-sha")
    parser.add_argument("--expect-backend")
    args = parser.parse_args()
    if args.inspect:
        if not args.state_file or not args.caddy_file:
            parser.error("inspect requires state and Caddy files")
        state = inspect(args.state_file, args.caddy_file)
        print(state["mode"], state["slot"], state["sha"] or "-", state["backend"])
    elif args.activate:
        if not all((args.state_file, args.caddy_file, args.mode, args.sha,
                    args.backend, args.expect_mode, args.expect_sha,
                    args.expect_backend)) or args.slot is None or args.expect_slot is None:
            parser.error("activate requires exact old and new ingress states")
        activate(
            args.state_file, args.caddy_file,
            expected_mode=args.expect_mode, expected_slot=args.expect_slot,
            expected_sha=args.expect_sha, expected_backend=args.expect_backend,
            mode=args.mode, slot=args.slot, sha=args.sha, backend=args.backend,
        )
    elif args.commit:
        if not all((args.state_file, args.caddy_file, args.mode, args.sha, args.backend)) or args.slot is None:
            parser.error("commit requires exact ingress state")
        commit_state(
            args.state_file, args.caddy_file, mode=args.mode,
            slot=args.slot, sha=args.sha, backend=args.backend,
        )
    else:
        if args.output is None or args.arena_edge_port is None or args.backend is None:
            parser.error("render requires output, edge port, and backend")
        args.output.write_text(
            render(arena_edge_port=args.arena_edge_port, backend=args.backend, direct=args.direct),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
