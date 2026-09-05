"""Verify and snapshot the exact installed gateway restart controller."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence


# These are minimum compatible ancestry floors, not an exhaustive release list.
SUPPORTED_CONTROLLER_COMMITS = frozenset(
    {"0dd3a385a23a3af0fa17210bfe02a39cc4023952"}
)
# Exact Git identity left in the production host-wrapper slot by the first
# controller cutover.  Keep this recovery allowance separate from the minimum
# compatible controller floors: it admits only these reviewed wrapper bytes.
RECOVERY_HOST_CONTROLLER_COMMITS = frozenset(
    {"ef0dfeaad19810d3ab2db137d397a2890830a574"}
)
CONTROLLER_FILES: Mapping[str, tuple[int, str]] = {
    "gw_restart.sh": (0o700, "100755"),
    "scripts/gateway_git_deploy.py": (0o600, "100644"),
    "Leadpoet/utils/exact_commit_restart_v2.py": (0o600, "100644"),
    "gateway/tee/host_memory_guard_v2.py": (0o600, "100644"),
}
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_UNSAFE_GIT_ENV_NAMES = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_WORK_TREE",
    }
)


class InstalledGatewayControllerError(RuntimeError):
    """The installed controller is not exact or safely snapshotable."""


def _safe_git_environment() -> dict[str, str]:
    if any(os.environ.get(name) for name in _UNSAFE_GIT_ENV_NAMES):
        raise InstalledGatewayControllerError(
            "installed controller Git environment contains overrides"
        )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _UNSAFE_GIT_ENV_NAMES
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _git(repo_root: Path, *arguments: str, binary: bool = False):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            capture_output=True,
            text=not binary,
            timeout=120,
            env=_safe_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise InstalledGatewayControllerError(
            "installed controller Git authority is unavailable"
        ) from exc
    if result.returncode != 0:
        raise InstalledGatewayControllerError(
            "installed controller Git authority is unavailable"
        )
    return result.stdout if binary else result.stdout.strip()


def _git_commit_exists(repo_root: Path, commit: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=_safe_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise InstalledGatewayControllerError(
            "installed controller Git authority is unavailable"
        ) from exc
    return result.returncode == 0


def _git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=_safe_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise InstalledGatewayControllerError(
            "installed controller Git authority is unavailable"
        ) from exc
    if result.returncode not in (0, 1):
        raise InstalledGatewayControllerError(
            "installed controller Git authority is unavailable"
        )
    return result.returncode == 0


def _require_unmodified_git_authority(repo_root: Path) -> None:
    if _git(
        repo_root,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace",
    ):
        raise InstalledGatewayControllerError(
            "installed controller repository contains replacement refs"
        )
    for relative in ("info/grafts", "objects/info/alternates"):
        candidate = Path(_git(repo_root, "rev-parse", "--git-path", relative))
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_size != 0
        ):
            raise InstalledGatewayControllerError(
                "installed controller repository contains graft authority"
            )


@contextmanager
def _open_parent_fd(
    path: Path,
    *,
    allowed_group_writable_paths: frozenset[Path] = frozenset(),
) -> Iterator[tuple[int, str]]:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.name in {"", ".", ".."}:
        raise InstalledGatewayControllerError(
            "installed controller file path is invalid"
        )
    allowed = {value.absolute() for value in allowed_group_writable_paths}
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[
        tuple[int, int | None, str | None, tuple[int, int]]
    ] = []
    try:
        root = os.open(candidate.parts[0], flags)
        root_metadata = os.fstat(root)
        descriptors.append(
            (root, None, None, (root_metadata.st_dev, root_metadata.st_ino))
        )
        current_path = Path(candidate.parts[0])
        for part in candidate.parts[1:-1]:
            parent = descriptors[-1][0]
            descriptor = os.open(part, flags, dir_fd=parent)
            metadata = os.fstat(descriptor)
            current_path /= part
            mode = stat.S_IMODE(metadata.st_mode)
            sticky_root_directory = bool(
                metadata.st_uid == 0 and metadata.st_mode & stat.S_ISVTX
            )
            reviewed_live_mode = bool(
                current_path in allowed
                and metadata.st_uid == os.geteuid()
                and metadata.st_gid == os.getegid()
                and mode == 0o775
            )
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid not in {0, os.geteuid()}
                or (
                    metadata.st_mode & 0o022
                    and not sticky_root_directory
                    and not reviewed_live_mode
                )
            ):
                os.close(descriptor)
                raise InstalledGatewayControllerError(
                    "installed controller file ancestry is unsafe"
                )
            if reviewed_live_mode:
                try:
                    os.fchmod(descriptor, 0o700)
                    os.fsync(descriptor)
                    hardened = os.fstat(descriptor)
                except OSError:
                    os.close(descriptor)
                    raise
                if (
                    not stat.S_ISDIR(hardened.st_mode)
                    or hardened.st_dev != metadata.st_dev
                    or hardened.st_ino != metadata.st_ino
                    or hardened.st_uid != os.geteuid()
                    or hardened.st_gid != os.getegid()
                    or stat.S_IMODE(hardened.st_mode) != 0o700
                ):
                    os.close(descriptor)
                    raise InstalledGatewayControllerError(
                        "installed controller file ancestry could not be hardened"
                    )
                metadata = hardened
            descriptors.append(
                (
                    descriptor,
                    parent,
                    part,
                    (metadata.st_dev, metadata.st_ino),
                )
            )
        yield descriptors[-1][0], candidate.name
        for descriptor, parent, name, identity in descriptors[1:]:
            assert parent is not None and name is not None
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != identity
                or (opened.st_dev, opened.st_ino) != identity
            ):
                raise InstalledGatewayControllerError(
                    "installed controller file ancestry changed"
                )
    except InstalledGatewayControllerError:
        raise
    except OSError as exc:
        raise InstalledGatewayControllerError(
            "installed controller file ancestry is unavailable"
        ) from exc
    finally:
        for descriptor, _parent, _name, _identity in reversed(descriptors):
            os.close(descriptor)


def _read_exact_file(
    path: Path,
    *,
    expected_mode: int,
    allowed_group_writable_paths: frozenset[Path] = frozenset(),
) -> bytes:
    descriptor: int | None = None
    try:
        with _open_parent_fd(
            path,
            allowed_group_writable_paths=allowed_group_writable_paths,
        ) as (parent_fd, leaf_name):
            descriptor = os.open(
                leaf_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            opened = os.fstat(descriptor)
            current = os.stat(
                leaf_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or stat.S_IMODE(opened.st_mode) != expected_mode
                or opened.st_dev != current.st_dev
                or opened.st_ino != current.st_ino
                or not 2 <= opened.st_size <= 4 * 1024 * 1024
            ):
                raise InstalledGatewayControllerError(
                    "installed controller file identity is unsafe"
                )
            chunks: list[bytes] = []
            observed_size = 0
            maximum_size = 4 * 1024 * 1024
            while True:
                chunk = os.read(
                    descriptor,
                    min(65536, maximum_size + 1 - observed_size),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                observed_size += len(chunk)
                if observed_size > maximum_size:
                    raise InstalledGatewayControllerError(
                        "installed controller file is too large"
                    )
            payload = b"".join(chunks)
            final = os.fstat(descriptor)
            final_path = os.stat(
                leaf_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                len(payload) != opened.st_size
                or final.st_dev != opened.st_dev
                or final.st_ino != opened.st_ino
                or final.st_size != opened.st_size
                or final_path.st_dev != opened.st_dev
                or final_path.st_ino != opened.st_ino
            ):
                raise InstalledGatewayControllerError(
                    "installed controller file changed while reading"
                )
            return payload
    except InstalledGatewayControllerError:
        raise
    except OSError as exc:
        raise InstalledGatewayControllerError(
            "installed controller file is unavailable"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verify_directory(path: Path, *, modes: frozenset[int]) -> tuple[int, int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        path_metadata = path.lstat()
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(path_metadata.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_gid != os.getegid()
            or stat.S_IMODE(opened.st_mode) not in modes
            or opened.st_dev != path_metadata.st_dev
            or opened.st_ino != path_metadata.st_ino
        ):
            raise InstalledGatewayControllerError(
                "installed controller directory identity is unsafe"
            )
        return opened.st_dev, opened.st_ino
    except OSError as exc:
        raise InstalledGatewayControllerError(
            "installed controller directory is unavailable"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _reviewed_controller_parent_paths(release_dir: Path) -> frozenset[Path]:
    release = Path(release_dir)
    return frozenset(
        release / parent
        for relative_path in CONTROLLER_FILES
        for parent in Path(relative_path).parents
        if parent != Path(".")
    )


def verify_candidate_bound_controller_lineage(
    *,
    repo_root: Path,
    expected_controller_commit: str,
    expected_candidate_commit: str,
) -> str:
    repository = Path(repo_root).expanduser().resolve()
    controller_commit = str(expected_controller_commit).lower()
    candidate_commit = str(expected_candidate_commit).lower()
    if not _COMMIT_RE.fullmatch(controller_commit) or not _COMMIT_RE.fullmatch(
        candidate_commit
    ):
        raise InstalledGatewayControllerError(
            "installed controller lineage identity is invalid"
        )
    _require_unmodified_git_authority(repository)
    if not _git_commit_exists(repository, controller_commit) or not _git_commit_exists(
        repository, candidate_commit
    ):
        raise InstalledGatewayControllerError(
            "installed controller lineage Git object is unavailable"
        )
    if not any(
        _git_is_ancestor(repository, floor, controller_commit)
        for floor in SUPPORTED_CONTROLLER_COMMITS
    ) or not _git_is_ancestor(repository, controller_commit, candidate_commit):
        raise InstalledGatewayControllerError(
            "installed controller lineage is incompatible with the candidate"
        )
    _require_unmodified_git_authority(repository)
    return controller_commit


def verify_installed_controller_bundle(
    *,
    repo_root: Path,
    controller_current: Path,
    host_restart_path: Path,
    expected_commit: str,
    expected_controller_commit: str,
) -> dict[str, Any]:
    repository = Path(repo_root).expanduser().resolve()
    commit = str(expected_commit).lower()
    expected_controller = str(expected_controller_commit).lower()
    if not _COMMIT_RE.fullmatch(commit) or not _COMMIT_RE.fullmatch(
        expected_controller
    ):
        raise InstalledGatewayControllerError("candidate commit is invalid")
    _require_unmodified_git_authority(repository)
    current = Path(controller_current)
    controller_root = current.parent
    releases_root = controller_root / "releases"
    ancestry = (controller_root.parent, controller_root, releases_root)
    ancestry_identities = [
        _verify_directory(path, modes=frozenset({0o700, 0o775}))
        for path in ancestry
    ]
    link_metadata = current.lstat()
    link_target = os.readlink(current)
    match = re.fullmatch(r"releases/([0-9a-f]{40})", link_target)
    if (
        not stat.S_ISLNK(link_metadata.st_mode)
        or link_metadata.st_uid != os.geteuid()
        or match is None
    ):
        raise InstalledGatewayControllerError(
            "installed controller link identity is unsafe"
        )
    controller_commit = match.group(1)
    if controller_commit != expected_controller:
        raise InstalledGatewayControllerError(
            "installed controller differs from the candidate-bound expectation"
        )
    controller_object_present = _git_commit_exists(repository, controller_commit)
    if not controller_object_present:
        raise InstalledGatewayControllerError(
            "installed controller Git object is unavailable"
        )
    if not any(
        _git_is_ancestor(repository, floor, controller_commit)
        for floor in SUPPORTED_CONTROLLER_COMMITS
    ):
        raise InstalledGatewayControllerError(
            "installed controller commit is unsupported"
        )
    release_dir = releases_root / controller_commit
    release_identity = _verify_directory(release_dir, modes=frozenset({0o700}))
    allowed_group_writable_paths = frozenset(ancestry) | (
        _reviewed_controller_parent_paths(release_dir)
    )
    observed: dict[str, bytes] = {}
    for relative_path, (installed_mode, git_mode) in CONTROLLER_FILES.items():
        payload = _read_exact_file(
            release_dir / relative_path,
            expected_mode=installed_mode,
            allowed_group_writable_paths=allowed_group_writable_paths,
        )
        tree_row = _git(
            repository,
            "ls-tree",
            controller_commit,
            "--",
            relative_path,
        ).split()
        if len(tree_row) < 3 or tree_row[0] != git_mode:
            raise InstalledGatewayControllerError(
                "installed controller Git mode differs"
            )
        authority = _git(
            repository,
            "show",
            f"{controller_commit}:{relative_path}",
            binary=True,
        )
        if payload != authority:
            raise InstalledGatewayControllerError(
                "installed controller bytes differ from Git authority"
        )
        observed[relative_path] = payload
    host_wrapper = _read_exact_file(Path(host_restart_path), expected_mode=0o700)
    host_candidates = (
        set(SUPPORTED_CONTROLLER_COMMITS)
        | set(RECOVERY_HOST_CONTROLLER_COMMITS)
        | {controller_commit}
    )
    host_controller_commits = {
        candidate
        for candidate in host_candidates
        if _git_commit_exists(repository, candidate)
        and host_wrapper
        == _git(
            repository,
            "show",
            f"{candidate}:gw_restart.sh",
            binary=True,
        )
    }
    if not host_controller_commits:
        raise InstalledGatewayControllerError(
            "installed gateway host wrapper differs from Git authority"
        )
    if (
        [
            _verify_directory(path, modes=frozenset({0o700, 0o775}))
            for path in ancestry
        ]
        != ancestry_identities
        or _verify_directory(release_dir, modes=frozenset({0o700}))
        != release_identity
        or current.lstat().st_dev != link_metadata.st_dev
        or current.lstat().st_ino != link_metadata.st_ino
        or current.lstat().st_uid != link_metadata.st_uid
        or current.lstat().st_mode != link_metadata.st_mode
        or os.readlink(current) != link_target
    ):
        raise InstalledGatewayControllerError(
            "installed controller changed while snapshotting"
        )
    _require_unmodified_git_authority(repository)
    return {
        "status": "verified",
        "controller_commit": controller_commit,
        "host_controller_commits": sorted(host_controller_commits),
        "payloads": observed,
    }


def _exec_verified_helper(
    *,
    bundle: Mapping[str, Any],
    relative_path: str,
    arguments: Sequence[str],
) -> None:
    if relative_path != "scripts/gateway_git_deploy.py" or not arguments:
        raise InstalledGatewayControllerError(
            "installed controller helper execution request is invalid"
        )
    payloads = bundle.get("payloads")
    payload = payloads.get(relative_path) if isinstance(payloads, Mapping) else None
    if not isinstance(payload, bytes) or not hasattr(os, "memfd_create"):
        raise InstalledGatewayControllerError(
            "installed controller helper cannot be sealed"
        )
    required_seals = sum(
        int(getattr(fcntl, name))
        for name in ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
    )
    descriptor = os.memfd_create(
        "leadpoet-verified-gateway-git-helper",
        flags=int(getattr(os, "MFD_ALLOW_SEALING", 0x0002)),
    )
    try:
        os.fchmod(descriptor, 0o400)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise InstalledGatewayControllerError(
                    "installed controller helper snapshot is incomplete"
                )
            written += count
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required_seals)
        if (
            int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS))
            & required_seals
            != required_seals
        ):
            raise InstalledGatewayControllerError(
                "installed controller helper snapshot is not sealed"
            )
        os.set_inheritable(descriptor, True)
        environment = _safe_git_environment()
        for name in (
            "PYTHONHOME",
            "PYTHONINSPECT",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
        ):
            environment.pop(name, None)
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        os.execve(
            sys.executable,
            [
                sys.executable,
                "-I",
                "-S",
                f"/proc/self/fd/{descriptor}",
                *[str(value) for value in arguments],
            ],
            environment,
        )
    finally:
        os.close(descriptor)


def _recover_exact_controller_checkout_drift(
    *,
    repo_root: Path,
    bundle: Mapping[str, Any],
    helper_arguments: Sequence[str],
) -> bool:
    """Remove only the exact controller copy left by an older pinned restart."""

    if list(helper_arguments[:1]) != ["prepare"]:
        return False
    repository = Path(repo_root).expanduser().resolve()
    head_commit = _git(repository, "rev-parse", "HEAD^{commit}").lower()
    if not _COMMIT_RE.fullmatch(head_commit):
        raise InstalledGatewayControllerError(
            "gateway Git checkout HEAD is invalid"
        )
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if not status:
        return False
    # _git strips outer whitespace, so an unstaged porcelain row is rendered
    # as one status letter followed by one space and the path.
    if status != "M gw_restart.sh":
        return False
    payloads = bundle.get("payloads")
    controller_payload = (
        payloads.get("gw_restart.sh") if isinstance(payloads, Mapping) else None
    )
    if not isinstance(controller_payload, bytes):
        raise InstalledGatewayControllerError(
            "installed controller recovery payload is unavailable"
        )
    checkout_path = repository / "gw_restart.sh"
    checkout_parent = frozenset({repository})
    if (
        _read_exact_file(
            checkout_path,
            expected_mode=0o700,
            allowed_group_writable_paths=checkout_parent,
        )
        != controller_payload
    ):
        raise InstalledGatewayControllerError(
            "gateway Git checkout has unrecognized controller drift"
        )
    _require_unmodified_git_authority(repository)
    if (
        _git(repository, "rev-parse", "HEAD^{commit}").lower() != head_commit
        or _git(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        != "M gw_restart.sh"
        or _read_exact_file(
            checkout_path,
            expected_mode=0o700,
            allowed_group_writable_paths=checkout_parent,
        )
        != controller_payload
    ):
        raise InstalledGatewayControllerError(
            "gateway Git checkout controller drift changed before recovery"
        )
    _git(
        repository,
        "restore",
        f"--source={head_commit}",
        "--worktree",
        "--",
        "gw_restart.sh",
    )
    if (
        _git(repository, "rev-parse", "HEAD^{commit}").lower() != head_commit
        or _git(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
    ):
        raise InstalledGatewayControllerError(
            "gateway Git checkout controller recovery was incomplete"
        )
    _require_unmodified_git_authority(repository)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--controller-current", type=Path)
    parser.add_argument("--host-restart-path", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-controller-commit", required=True)
    parser.add_argument("--verify-lineage-only", action="store_true")
    parser.add_argument(
        "--exec-helper",
        choices=("scripts/gateway_git_deploy.py",),
    )
    parser.add_argument("helper_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.verify_lineage_only:
            if (
                args.controller_current is not None
                or args.host_restart_path is not None
                or args.exec_helper is not None
                or args.helper_arguments
            ):
                raise InstalledGatewayControllerError(
                    "controller lineage-only request is invalid"
                )
            verify_candidate_bound_controller_lineage(
                repo_root=args.repo_root,
                expected_controller_commit=args.expected_controller_commit,
                expected_candidate_commit=args.expected_commit,
            )
            return 0
        if (
            args.controller_current is None
            or args.host_restart_path is None
            or args.exec_helper is None
        ):
            raise InstalledGatewayControllerError(
                "installed controller verification request is incomplete"
            )
        helper_arguments = list(args.helper_arguments)
        if helper_arguments[:1] == ["--"]:
            helper_arguments = helper_arguments[1:]
        result = verify_installed_controller_bundle(
            repo_root=args.repo_root,
            controller_current=args.controller_current,
            host_restart_path=args.host_restart_path,
            expected_commit=args.expected_commit,
            expected_controller_commit=args.expected_controller_commit,
        )
        _recover_exact_controller_checkout_drift(
            repo_root=args.repo_root,
            bundle=result,
            helper_arguments=helper_arguments,
        )
        _exec_verified_helper(
            bundle=result,
            relative_path=args.exec_helper,
            arguments=helper_arguments,
        )
        raise InstalledGatewayControllerError(
            "installed controller helper exec returned unexpectedly"
        )
    except InstalledGatewayControllerError as exc:
        print(
            json.dumps(
                {"status": "failed_closed", "error": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
