#!/usr/bin/env python3
"""Run the exact-main canonical controller when its installed predecessor cannot prepare it."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile


FILES = {
    "gw_restart.sh": 0o700,
    "scripts/gateway_git_deploy.py": 0o600,
    "gateway/tee/host_memory_guard_v2.py": 0o600,
    "scripts/manage_owned_process_group.py": 0o600,
}
SELF_PATH = "scripts/bootstrap_gateway_restart_controller.py"
ORIGINS = {"https://github.com/leadpoet/leadpoet.git",
           "git@github.com:leadpoet/leadpoet.git",
           "ssh://git@github.com/leadpoet/leadpoet.git"}
SHA = re.compile(r"[0-9a-f]{40}")
INVOCATION = re.compile(r"gateway-([1-9][0-9]{0,11})-([1-9][0-9]{0,9})")


class BootstrapError(RuntimeError):
    """A redacted bootstrap refusal that is safe to show the operator."""


def _environment() -> dict[str, str]:
    # Never inherit restart phase, authority, invocation, Git/Python overrides,
    # proxy settings, or ambient provider/cloud credentials into this repair.
    return {key: value for key in ("HOME", "PATH", "USER", "LOGNAME", "LANG", "TERM")
            if (value := os.environ.get(key)) is not None} | {"PYTHONDONTWRITEBYTECODE": "1"}


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], env=_environment(),
                            capture_output=True, timeout=120)
    if result.returncode:
        # Git stderr can contain remote credentials. Keep failures redacted.
        raise BootstrapError(f"Git {args[0]} failed")
    return result.stdout


def _parents(path: Path, *, private_boundary: Path | None = None) -> None:
    if private_boundary is not None and private_boundary not in path.parents:
        raise BootstrapError("controller release boundary is not a parent")
    shielded = False
    for parent in reversed(path.parents):
        metadata = parent.lstat()
        if parent == private_boundary:
            if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or metadata.st_gid != os.getgid() or stat.S_IMODE(metadata.st_mode) != 0o700):
                raise BootstrapError("controller release boundary is not owned and private")
            shielded = True
        # The canonical installer's umask can leave owned 0775 subdirectories
        # inside its 0700 release. Only that verified private boundary permits
        # group-write; exposed parents and world-writable descendants still fail.
        protected_group_write = (shielded and metadata.st_uid == os.getuid()
                                 and metadata.st_gid == os.getgid()
                                 and not metadata.st_mode & 0o002)
        if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid not in (0, os.getuid())
                or (metadata.st_mode & 0o022 and not protected_group_write
                    and (shielded or not metadata.st_mode & stat.S_ISVTX))):
            raise BootstrapError("unsafe parent directory")


def _owned(path: Path, mode: int, *, directory: bool = False,
           private_boundary: Path | None = None) -> tuple[int, ...]:
    _parents(path, private_boundary=private_boundary)
    metadata = path.lstat()
    kind_ok = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if (not kind_ok or metadata.st_uid != os.getuid() or metadata.st_gid != os.getgid()
            or stat.S_IMODE(metadata.st_mode) != mode
            or (not directory and metadata.st_nlink != 1)):
        raise BootstrapError("controller path has unexpected ownership, mode, or type")
    return (metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns, metadata.st_size)


def _blobs(repo: Path, commit: str) -> dict[str, bytes]:
    return {name: _git(repo, "show", f"{commit}:{name}") for name in FILES}


def _installed(controller_root: Path, host_script: Path, commit: str,
               blobs: dict[str, bytes]) -> tuple[object, ...]:
    result: list[object] = [_owned(controller_root, 0o700, directory=True),
                            _owned(controller_root / "releases", 0o700, directory=True)]
    current = controller_root / "current"
    metadata = current.lstat()
    if (not stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid() or os.readlink(current) != f"releases/{commit}"):
        raise BootstrapError("installed controller does not match the expected commit")
    result.append((metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns))
    release = controller_root / "releases" / commit
    result.append(_owned(release, 0o700, directory=True))
    for name, mode in FILES.items():
        path = release / name
        result.append(_owned(path, mode, private_boundary=release))
        if path.read_bytes() != blobs[name]:
            raise BootstrapError("installed controller differs from its committed bytes")
    result.append(_owned(host_script, 0o700))
    if host_script.read_bytes() != blobs["gw_restart.sh"]:
        raise BootstrapError("host controller differs from its committed bytes")
    return tuple(result)


def _lock(stack: ExitStack, path: Path) -> None:
    _parents(path)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    stack.callback(os.close, descriptor)
    metadata = os.fstat(descriptor)
    _owned(path, 0o600)
    if (metadata.st_dev, metadata.st_ino) != (path.stat().st_dev, path.stat().st_ino):
        raise BootstrapError("restart lock changed while opening")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise BootstrapError("a canonical restart or recovery already owns the host lock") from None


def _archive(repo: Path, commit: str, root: Path) -> None:
    # Extract only ordinary Git archive entries under the private authority root.
    # No tar links, devices, absolute paths, or traversal are accepted.
    with tempfile.TemporaryFile() as payload:
        result = subprocess.run(["git", "-C", str(repo), "archive", commit],
                                env=_environment(), stdout=payload,
                                stderr=subprocess.PIPE, timeout=120)
        if result.returncode:
            raise BootstrapError("candidate Git archive failed")
        payload.seek(0)
        with tarfile.open(fileobj=payload) as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                    raise BootstrapError("candidate archive has an unsafe path")
                target = root.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(mode=0o700, parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    with target.open("xb") as destination:
                        source = archive.extractfile(member)
                        if source is None:
                            raise BootstrapError("candidate archive file is unreadable")
                        with source:
                            shutil.copyfileobj(source, destination)
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
                else:
                    raise BootstrapError("candidate archive has an unsupported entry")


def _resume_owner(*, repo: Path, source_commit: str, authority: Path,
                  invocation: str, timing_dir: Path, environment_file: Path) -> str:
    if not INVOCATION.fullmatch(invocation):
        raise BootstrapError("resume invocation is invalid")
    _owned(timing_dir, 0o700, directory=True)
    ledger = timing_dir / f"{invocation}.jsonl"
    before = _owned(ledger, 0o600)
    with os.fdopen(os.open(ledger, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if (metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns, metadata.st_size) != before:
            raise BootstrapError("resume timing ledger changed during read")
        raw = stream.read(1024 * 1024 + 1)
    if not raw or len(raw) > 1024 * 1024 or _owned(ledger, 0o600) != before:
        raise BootstrapError("resume timing ledger is unavailable or changed")
    try:
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        candidate = rows[0]["commit_sha"]
        previous_elapsed = -1.0
        for row in rows:
            elapsed = row["elapsed_seconds"]
            observed = datetime.fromisoformat(row["observed_at"].replace("Z", "+00:00"))
            if (set(row) != {"schema_version", "stage", "status", "commit_sha", "observed_at", "elapsed_seconds"}
                    or row["schema_version"] != "leadpoet.gateway_restart_timing.v1"
                    or not isinstance(candidate, str) or not SHA.fullmatch(candidate)
                    or row["commit_sha"] != candidate
                    or not isinstance(row["stage"], str)
                    or row["stage"] not in {
                        "offline_artifact_prepare_started", "offline_artifact_prepare_complete",
                        "local_release_ready", "dependency_preflight_complete",
                        "lab_arena_startup_preflight_complete", "pre_shutdown_disk_ready",
                        "pre_shutdown_checks_complete", "lab_arena_claim_drain",
                        "lab_arena_destructive_authorization",
                    }
                    or row["status"] not in ("reached", "passed", "failed")
                    or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed) or elapsed < 0 or elapsed < previous_elapsed
                    or observed.tzinfo is None
                    or observed.timestamp() < int(invocation.split("-")[1])):
                raise ValueError
            previous_elapsed = elapsed
        if (rows[-1]["status"] != "failed" or rows[-1]["stage"] not in
                {"lab_arena_claim_drain", "lab_arena_destructive_authorization"}):
            raise ValueError
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        raise BootstrapError("resume timing ledger does not show a supported failed pre-shutdown attempt") from None
    _git(repo, "merge-base", "--is-ancestor", candidate, source_commit)
    _owned(environment_file, 0o600)
    result = subprocess.run(
        [sys.executable, "-B", str(authority / "scripts/lab_arena_restart_claim_guard.py"),
         "owned", "--candidate", candidate, "--invocation", invocation,
         "--environment-file", str(environment_file)],
        env=_environment(), capture_output=True, timeout=30,
    )
    try:
        state = json.loads(result.stdout)
        drain = state["drain"]
        generation = state["guard_generation"]
        valid = (result.returncode == 0 and state["schema_version"] == "leadpoet.lab_arena.restart_guard_state.v1"
                 and state["guard_present"] is True and state["guard_active"] is True
                 and state["candidate_commit"] == candidate
                 and state["restart_scope"] == "gateway"
                 and state["restart_phase"] == "gateway_destructive"
                 and type(generation) is int and generation > 0
                 and drain["schema_version"] == "leadpoet.lab_arena.restart_drain_state.v1"
                 and drain["preserved"] is True)
    except (ValueError, KeyError, TypeError):
        valid = False
    if not valid or _owned(ledger, 0o600) != before:
        raise BootstrapError("resume guard ownership or phase could not be verified")
    return invocation


def bootstrap(*, repo: Path, source_commit: str, expected_controller_commit: str,
              controller_root: Path, host_script: Path, lock_file: Path,
              recovery_lock_file: Path, helper_path: Path,
              bootstrap_parent: Path = Path("/tmp"), resume_invocation: str = "",
              timing_dir: Path = Path("/home/ec2-user/.config/leadpoet/restart-timings"),
              environment_file: Path = Path("/home/ec2-user/.config/leadpoet/gateway.env")) -> int:
    if not SHA.fullmatch(source_commit) or not SHA.fullmatch(expected_controller_commit):
        raise BootstrapError("full source and installed controller commits are required")
    for path in (repo, controller_root, host_script, lock_file, recovery_lock_file, helper_path):
        if not path.is_absolute() or ".." in path.parts:
            raise BootstrapError("absolute paths without traversal are required")
        _parents(path)
    metadata = repo.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o022):
        raise BootstrapError("canonical repository path is unsafe")
    if _git(repo, "remote", "get-url", "origin").decode().strip() not in ORIGINS:
        raise BootstrapError("unknown repository origin")
    _git(repo, "fetch", "--no-tags", "origin")
    if _git(repo, "rev-parse", "origin/main").decode().strip() != source_commit:
        raise BootstrapError("source commit is not the current fetched origin/main")
    _git(repo, "merge-base", "--is-ancestor", expected_controller_commit, source_commit)
    helper_metadata = helper_path.lstat()
    if (not stat.S_ISREG(helper_metadata.st_mode) or helper_metadata.st_uid != os.getuid()
            or helper_metadata.st_mode & 0o022
            or helper_path.read_bytes() != _git(repo, "show", f"{source_commit}:{SELF_PATH}")):
        raise BootstrapError("bootstrap helper differs from exact main")
    old = _blobs(repo, expected_controller_commit)
    new = _blobs(repo, source_commit)
    if b'GATEWAY_GIT_DEPLOY_PROTOCOL="1"' not in new["gw_restart.sh"]:
        raise BootstrapError("candidate is not a canonical Git deployment controller")
    work = bootstrap_parent / f"gateway-restart-controller-bootstrap.{secrets.token_hex(12)}"
    work.mkdir(mode=0o700)
    authority = work / "authority"
    authority.mkdir(mode=0o700)
    canonical: subprocess.Popen | None = None
    try:
        # Match the canonical recovery -> restart lock order; never reclaim locks.
        with ExitStack() as locks:
            _lock(locks, recovery_lock_file)
            _lock(locks, lock_file)
            before = _installed(controller_root, host_script, expected_controller_commit, old)
            _archive(repo, source_commit, authority)
            for name, content in new.items():
                if (authority / name).read_bytes() != content:
                    raise BootstrapError("candidate controller archive differs from exact Git blobs")
            spec = importlib.util.spec_from_file_location("_bootstrap_git_verifier", authority / "scripts/gateway_git_deploy.py")
            if spec is None or spec.loader is None:
                raise BootstrapError("candidate tree verifier is unavailable")
            verifier = importlib.util.module_from_spec(spec)
            sys.dont_write_bytecode = True
            spec.loader.exec_module(verifier)
            proof = verifier.verify_materialized_tree(
                repo_root=repo, materialized_root=authority, target_sha=source_commit,
                strict_extras=True,
            )
            if resume_invocation:
                _resume_owner(repo=repo, source_commit=source_commit, authority=authority,
                              invocation=resume_invocation, timing_dir=timing_dir,
                              environment_file=environment_file)
            _git(repo, "fetch", "--no-tags", "origin")
            if _git(repo, "rev-parse", "origin/main").decode().strip() != source_commit:
                raise BootstrapError("origin/main advanced during controller preparation")
            if _installed(controller_root, host_script, expected_controller_commit, old) != before:
                raise BootstrapError("installed controller changed during preparation")
        # The full candidate controller must acquire the normal host lock itself.
        # No prepare/activation phase or destructive handoff can be inherited.
        environment = _environment() | {
            "LEADPOET_REPO_ROOT": str(repo),
            "GATEWAY_RESTART_CONTROLLER_ROOT": str(controller_root),
            "GATEWAY_HOST_RESTART_SCRIPT": str(host_script),
            "GATEWAY_RESTART_LOCK_FILE": str(lock_file),
            "GATEWAY_RESTART_RECOVERY_LOCK_FILE": str(recovery_lock_file),
            "GATEWAY_RESTART_AUTHORITY_ROOT": str(authority),
            "GATEWAY_RESTART_AUTHORITY_COMMIT": source_commit,
        }
        if resume_invocation:
            environment["GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID"] = resume_invocation
        canonical = subprocess.Popen(["/bin/bash", str(authority / "gw_restart.sh"),
                                      "--commit", source_commit], env=environment, cwd=repo)
        canonical_exit = canonical.wait()
        installed_state = "unknown"
        for commit, blobs, label in ((source_commit, new, "candidate"),
                                     (expected_controller_commit, old, "previous")):
            try:
                _installed(controller_root, host_script, commit, blobs)
                installed_state = label
                break
            except (OSError, RuntimeError):
                pass
        bootstrap_exit = (1 if canonical_exit == 0 and installed_state != "candidate"
                          else canonical_exit)
        print(json.dumps({"schema_version": "leadpoet.gateway_controller_bootstrap.v1",
                          "source_commit": source_commit,
                          "previous_controller_commit": expected_controller_commit,
                          "candidate_tree_hash": proof["tree_hash"],
                          "canonical_exit_code": canonical_exit,
                          "bootstrap_exit_code": bootstrap_exit,
                          "installed_controller": installed_state,
                          "runtime_deployment_verified": False}), flush=True)
        return bootstrap_exit
    finally:
        if canonical is None or canonical.poll() is not None:
            shutil.rmtree(work)
        else:
            # An interrupted observer must not kill the canonical controller or
            # delete the exact authority files it can still need.
            print(json.dumps({"schema_version": "leadpoet.gateway_controller_bootstrap.v1",
                              "source_commit": source_commit,
                              "canonical_pid": canonical.pid,
                              "canonical_exit_code": None,
                              "authority_retained": str(authority),
                              "runtime_deployment_verified": False}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-controller-commit", required=True)
    parser.add_argument("--resume-invocation", default="")
    parser.add_argument("--repo-root", type=Path, default=Path("/home/ec2-user/leadpoet_repo"))
    parser.add_argument("--controller-root", type=Path, default=Path("/home/ec2-user/.config/leadpoet/restart-controller/gateway"))
    parser.add_argument("--host-script", type=Path, default=Path("/home/ec2-user/gw_restart.sh"))
    parser.add_argument("--lock-file", type=Path, default=Path("/home/ec2-user/.config/leadpoet/gateway-restart.lock"))
    parser.add_argument("--recovery-lock-file", type=Path, default=Path("/home/ec2-user/.config/leadpoet/gateway-restart.lock.recovery"))
    args = parser.parse_args()
    try:
        return bootstrap(repo=args.repo_root, source_commit=args.source_commit,
                         expected_controller_commit=args.expected_controller_commit,
                         controller_root=args.controller_root, host_script=args.host_script,
                         lock_file=args.lock_file, recovery_lock_file=args.recovery_lock_file,
                         resume_invocation=args.resume_invocation,
                         helper_path=Path(__file__).absolute())
    except BootstrapError as exc:
        print(f"ERROR: gateway controller bootstrap refused: {exc}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, subprocess.SubprocessError, tarfile.TarError):
        print("ERROR: gateway controller bootstrap failed verification or I/O", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("ERROR: bootstrap observer interrupted; inspect any retained canonical controller", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
