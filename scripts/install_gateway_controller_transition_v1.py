#!/usr/bin/env python3
"""Install an exact current gateway restart controller before a breaking cutover."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FILES = {
    "gw_restart.sh": 0o700,
    "scripts/gateway_git_deploy.py": 0o600,
    "gateway/tee/host_memory_guard_v2.py": 0o600,
    "scripts/manage_owned_process_group.py": 0o600,
}
_UNSAFE_GIT_ENV = {
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_CONFIG",
    "GIT_CONFIG_COUNT", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_SYSTEM", "GIT_DIR", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_REPLACE_REF_BASE", "GIT_WORK_TREE",
}


class ControllerTransitionError(RuntimeError):
    pass


def _require_inherited_lock(descriptor: int, path: Path) -> None:
    try:
        info = os.fstat(descriptor)
        path_info = path.lstat()
        if (
            descriptor < 3
            or not stat.S_ISREG(info.st_mode)
            or not stat.S_ISREG(path_info.st_mode)
            or (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ControllerTransitionError("inherited gateway restart lock is invalid")
        # Reacquiring on the inherited open-file description proves the lock is
        # live. A different open description held by another process fails.
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as exc:
        raise ControllerTransitionError("inherited gateway restart lock is not held") from exc


def _git(repo: Path, *args: str, binary: bool = False):
    if any(os.environ.get(name) for name in _UNSAFE_GIT_ENV):
        raise ControllerTransitionError("Git environment overrides are prohibited")
    environment = {name: value for name, value in os.environ.items() if name not in _UNSAFE_GIT_ENV}
    environment.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_TERMINAL_PROMPT": "0"})
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=not binary,
        timeout=120,
        env=environment,
    )
    if result.returncode:
        raise ControllerTransitionError("exact Git authority is unavailable")
    return result.stdout if binary else result.stdout.strip()


def _require_exact_source(repo: Path, commit: str) -> None:
    if not _COMMIT_RE.fullmatch(commit):
        raise ControllerTransitionError("candidate commit is invalid")
    if _git(repo, "rev-parse", "HEAD") != commit:
        raise ControllerTransitionError("checkout differs from the candidate")
    if _git(repo, "rev-parse", "origin/main") != commit:
        raise ControllerTransitionError("candidate is not exact origin/main")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ControllerTransitionError("candidate checkout is dirty")
    if _git(repo, "for-each-ref", "--format=%(refname)", "refs/replace"):
        raise ControllerTransitionError("replacement refs are prohibited")


def install_transition(
    *, repo: Path, commit: str, controller_root: Path, host_restart: Path
) -> Path:
    repo = repo.resolve(strict=True)
    _require_exact_source(repo, commit)
    releases = controller_root / "releases"
    releases.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(controller_root, 0o700)
    os.chmod(releases, 0o700)
    destination = releases / commit
    temporary = Path(tempfile.mkdtemp(prefix=".transition.", dir=controller_root))
    try:
        os.chmod(temporary, 0o700)
        for relative, mode in _FILES.items():
            tree = _git(repo, "ls-tree", commit, "--", relative).split()
            expected_git_mode = "100755" if relative == "gw_restart.sh" else "100644"
            if len(tree) < 3 or tree[0] != expected_git_mode:
                raise ControllerTransitionError("candidate controller Git mode differs")
            payload = _git(repo, "show", f"{commit}:{relative}", binary=True)
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target.write_bytes(payload)
            target.chmod(mode)
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_dir():
                raise ControllerTransitionError("controller release path is unsafe")
            for relative, mode in _FILES.items():
                installed = destination / relative
                metadata = installed.lstat()
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or stat.S_IMODE(metadata.st_mode) != mode
                    or installed.read_bytes() != (temporary / relative).read_bytes()
                ):
                    raise ControllerTransitionError("existing controller release differs")
        else:
            os.replace(temporary, destination)
        link = controller_root / f".current.{os.getpid()}"
        link.symlink_to(f"releases/{commit}")
        os.replace(link, controller_root / "current")
        host_restart.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".gw_restart.", dir=host_restart.parent)
        os.close(descriptor)
        staged = Path(name)
        try:
            staged.write_bytes((destination / "gw_restart.sh").read_bytes())
            staged.chmod(0o700)
            os.replace(staged, host_restart)
        finally:
            staged.unlink(missing_ok=True)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--controller-root",
        type=Path,
        default=Path("/home/ec2-user/.config/leadpoet/restart-controller/gateway"),
    )
    parser.add_argument(
        "--host-restart", type=Path, default=Path("/home/ec2-user/gw_restart.sh")
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("/home/ec2-user/.config/leadpoet/gateway-restart.lock"),
    )
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    if args.lock_fd is not None:
        _require_inherited_lock(args.lock_fd, args.lock)
        destination = install_transition(
            repo=args.repo,
            commit=args.commit,
            controller_root=args.controller_root,
            host_restart=args.host_restart,
        )
    else:
        with args.lock.open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ControllerTransitionError("gateway restart is already active") from exc
            destination = install_transition(
                repo=args.repo,
                commit=args.commit,
                controller_root=args.controller_root,
                host_restart=args.host_restart,
            )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
