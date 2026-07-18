#!/usr/bin/env python3
"""Prepare, activate, and record one exact gateway Git deployment."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


SCHEMA_VERSION = "leadpoet.gateway_git_deployment.v1"
DEFAULT_REPO_URL = "https://github.com/leadpoet/leadpoet.git"
DEFAULT_BRANCH = "main"
RESTART_PROTOCOL_MARKER = 'GATEWAY_GIT_DEPLOY_PROTOCOL="1"'

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")


class GatewayGitDeployError(RuntimeError):
    """Raised when a gateway Git deployment cannot proceed safely."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _run_git(repo_root: Path, *args: str, timeout: float = 120.0) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GatewayGitDeployError(f"git {args[0]} could not run: {type(exc).__name__}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown git error").strip()
        if len(detail) > 500:
            detail = detail[:500] + "..."
        raise GatewayGitDeployError(f"git {args[0]} failed: {detail}")
    return result.stdout.strip()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GatewayGitDeployError(f"deployment plan is unreadable: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise GatewayGitDeployError("deployment plan schema is invalid")
    return value


def _read_env_file(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return {str(key): "" if value is None else str(value) for key, value in parsed.items()}

    values: dict[str, str] = {}
    for raw_line in raw.replace("\x00", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            parts = [line]
        candidate = parts[0] if len(parts) == 1 else line
        if "=" not in candidate:
            continue
        key, value = candidate.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            values[key] = value
    return values


def _configured_value(
    name: str,
    explicit: str | None,
    env_file_values: Mapping[str, str],
    default: str,
) -> str:
    return str(explicit or os.getenv(name) or env_file_values.get(name) or default).strip()


def _operator_only_value(name: str, explicit: str | None, default: str) -> str:
    """Resolve one-invocation controls without consulting persistent runtime config."""

    return str(explicit or os.getenv(name) or default).strip()


def _sanitize_remote(remote: str) -> str:
    value = str(remote or "").strip()
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or not parsed.netloc:
        scp_style = re.fullmatch(r"[^@\s]+@([^:\s]+):(.+)", value)
        if scp_style:
            return f"{scp_style.group(1).lower()}:{scp_style.group(2)}"
        return value
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme.lower(), host.lower(), parsed.path, "", ""))


def _canonical_remote(remote: str) -> str:
    value = _sanitize_remote(remote).rstrip("/")
    return value[:-4] if value.endswith(".git") else value


def _validate_branch(repo_root: Path, branch: str) -> None:
    if not branch or branch.startswith("-"):
        raise GatewayGitDeployError("configured Git branch is invalid")
    _run_git(repo_root, "check-ref-format", "--branch", branch, timeout=10)


def _validate_checkout(repo_root: Path, expected_remote: str) -> None:
    if not repo_root.is_dir():
        raise GatewayGitDeployError(f"gateway Git checkout is missing: {repo_root}")
    actual_root = Path(_run_git(repo_root, "rev-parse", "--show-toplevel")).resolve()
    if actual_root != repo_root.resolve():
        raise GatewayGitDeployError("configured gateway repository root is not the Git toplevel")
    actual_remote = _run_git(repo_root, "remote", "get-url", "origin")
    if _canonical_remote(actual_remote) != _canonical_remote(expected_remote):
        raise GatewayGitDeployError("gateway Git origin does not match GITHUB_REPO_URL")


def _require_clean_checkout(repo_root: Path) -> None:
    status = _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        first_path = status.splitlines()[0][3:]
        raise GatewayGitDeployError(f"gateway Git checkout is dirty: {first_path}")


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode not in (0, 1):
        raise GatewayGitDeployError("git merge-base failed while validating deployment history")
    return result.returncode == 0


def _validate_target_restart_protocol(repo_root: Path, target_sha: str) -> None:
    restart_script = _run_git(repo_root, "show", f"{target_sha}:gw_restart.sh", timeout=30)
    if RESTART_PROTOCOL_MARKER not in restart_script:
        raise GatewayGitDeployError(
            "selected commit does not support the gateway Git restart handoff protocol"
        )
    _run_git(
        repo_root,
        "cat-file",
        "-e",
        f"{target_sha}:scripts/gateway_git_deploy.py",
        timeout=30,
    )


def prepare_deployment(
    *,
    repo_root: Path,
    repo_url: str,
    branch: str,
    plan_file: Path,
    manifest_file: Path,
    last_good_file: Path,
    deploy_commit: str = "",
) -> dict[str, Any]:
    """Fetch one branch and persist an immutable deployment decision."""

    repo_root = repo_root.expanduser().resolve()
    _validate_checkout(repo_root, repo_url)
    _validate_branch(repo_root, branch)
    _require_clean_checkout(repo_root)

    previous_sha = _run_git(repo_root, "rev-parse", "HEAD").lower()
    _run_git(
        repo_root,
        "fetch",
        "--prune",
        "origin",
        f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
    )
    branch_head_sha = _run_git(
        repo_root,
        "rev-parse",
        f"refs/remotes/origin/{branch}^{{commit}}",
    ).lower()
    if not _FULL_SHA_RE.fullmatch(branch_head_sha):
        raise GatewayGitDeployError("fetched branch did not resolve to a full Git commit")

    requested_sha = str(deploy_commit or "").strip().lower()
    if requested_sha:
        if not _FULL_SHA_RE.fullmatch(requested_sha):
            raise GatewayGitDeployError("GATEWAY_DEPLOY_COMMIT must be a full 40-character SHA")
        _run_git(repo_root, "cat-file", "-e", f"{requested_sha}^{{commit}}", timeout=30)
        if not _is_ancestor(repo_root, requested_sha, branch_head_sha):
            raise GatewayGitDeployError(
                "GATEWAY_DEPLOY_COMMIT is not reachable from the configured branch"
            )
        target_sha = requested_sha
        mode = "pinned"
    else:
        target_sha = branch_head_sha
        mode = "fast_forward"
        if not _is_ancestor(repo_root, previous_sha, target_sha):
            raise GatewayGitDeployError("configured branch is not a fast-forward from deployed HEAD")

    _validate_target_restart_protocol(repo_root, target_sha)
    tree_hash = _run_git(repo_root, "rev-parse", f"{target_sha}^{{tree}}").lower()
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": "github",
        "status": "prepared",
        "stage": "git_prepare",
        "mode": mode,
        "repo_root": str(repo_root),
        "remote_url": _sanitize_remote(repo_url),
        "branch": branch,
        "previous_sha": previous_sha,
        "branch_head_sha": branch_head_sha,
        "target_sha": target_sha,
        "tree_hash": tree_hash,
        "prepared_at": _utc_now(),
        "manifest_file": str(manifest_file.expanduser().resolve()),
        "last_good_file": str(last_good_file.expanduser().resolve()),
    }
    _atomic_write_json(plan_file, document)
    _atomic_write_json(Path(document["manifest_file"]), document)
    return document


def activate_deployment(*, plan_file: Path) -> dict[str, Any]:
    """Move the stopped gateway checkout to the commit selected by prepare."""

    document = _read_json(plan_file)
    repo_root = Path(str(document["repo_root"])).resolve()
    _validate_checkout(repo_root, str(document["remote_url"]))
    _require_clean_checkout(repo_root)

    current_sha = _run_git(repo_root, "rev-parse", "HEAD").lower()
    if current_sha != document.get("previous_sha"):
        raise GatewayGitDeployError("gateway checkout HEAD changed after deployment prepare")

    target_sha = str(document.get("target_sha") or "").lower()
    branch = str(document.get("branch") or "")
    if not _FULL_SHA_RE.fullmatch(target_sha):
        raise GatewayGitDeployError("deployment target SHA is invalid")

    if document.get("mode") == "pinned":
        _run_git(repo_root, "checkout", "--detach", target_sha)
    elif document.get("mode") == "fast_forward":
        remote_head = _run_git(
            repo_root,
            "rev-parse",
            f"refs/remotes/origin/{branch}^{{commit}}",
        ).lower()
        if remote_head != target_sha:
            raise GatewayGitDeployError("prepared remote branch changed before activation")
        local_ref = f"refs/heads/{branch}"
        local_exists = subprocess.run(
            ["git", "-C", str(repo_root), "show-ref", "--verify", "--quiet", local_ref],
            check=False,
            timeout=10,
        ).returncode == 0
        if local_exists:
            local_sha = _run_git(repo_root, "rev-parse", local_ref).lower()
            if not _is_ancestor(repo_root, local_sha, target_sha):
                raise GatewayGitDeployError("local deployment branch cannot fast-forward to target")
            _run_git(repo_root, "checkout", branch)
            _run_git(repo_root, "merge", "--ff-only", target_sha)
        else:
            _run_git(repo_root, "checkout", "-b", branch, "--track", f"origin/{branch}")
    else:
        raise GatewayGitDeployError("deployment activation mode is invalid")

    activated_sha = _run_git(repo_root, "rev-parse", "HEAD").lower()
    if activated_sha != target_sha:
        raise GatewayGitDeployError("activated gateway commit does not match prepared target")
    _require_clean_checkout(repo_root)

    document.update(
        {
            "status": "activated",
            "stage": "git_activate",
            "activated_at": _utc_now(),
        }
    )
    _atomic_write_json(plan_file, document)
    _atomic_write_json(Path(str(document["manifest_file"])), document)
    return document


def _find_pcr0(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).upper() == "PCR0":
                candidate = str(nested or "").strip().lower()
                if _PCR0_RE.fullmatch(candidate):
                    return candidate
            found = _find_pcr0(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_pcr0(nested)
            if found:
                return found
    return None


def collect_role_pcr0s(eif_root: Path) -> dict[str, str]:
    role_pcr0s: dict[str, str] = {}
    if not eif_root.is_dir():
        return role_pcr0s
    for path in sorted(eif_root.glob("enclave-build-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pcr0 = _find_pcr0(payload)
        if pcr0:
            role = path.stem.removeprefix("enclave-build-")
            role_pcr0s[role] = pcr0
    return role_pcr0s


def finalize_deployment(
    *,
    plan_file: Path,
    status: str,
    stage: str,
    eif_root: Path,
) -> dict[str, Any]:
    if status not in {"succeeded", "failed"}:
        raise GatewayGitDeployError("deployment final status is invalid")
    document = _read_json(plan_file)
    document.update(
        {
            "status": status,
            "stage": str(stage or "unknown"),
            "completed_at": _utc_now(),
            "role_pcr0s": collect_role_pcr0s(eif_root) if status == "succeeded" else {},
        }
    )
    _atomic_write_json(plan_file, document)
    _atomic_write_json(Path(str(document["manifest_file"])), document)
    if status == "succeeded":
        _atomic_write_json(Path(str(document["last_good_file"])), document)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", required=True, type=Path)
    prepare.add_argument("--env-file", type=Path)
    prepare.add_argument("--repo-url")
    prepare.add_argument("--branch")
    prepare.add_argument("--deploy-commit")
    prepare.add_argument("--plan-file", required=True, type=Path)
    prepare.add_argument("--manifest-file", required=True, type=Path)
    prepare.add_argument("--last-good-file", required=True, type=Path)

    activate = subparsers.add_parser("activate")
    activate.add_argument("--plan-file", required=True, type=Path)

    field = subparsers.add_parser("field")
    field.add_argument("--plan-file", required=True, type=Path)
    field.add_argument(
        "--name",
        required=True,
        choices=("repo_root", "remote_url", "branch", "previous_sha", "target_sha", "tree_hash"),
    )

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--plan-file", required=True, type=Path)
    finalize.add_argument("--status", required=True, choices=("succeeded", "failed"))
    finalize.add_argument("--stage", required=True)
    finalize.add_argument("--eif-root", type=Path, default=Path("/home/ec2-user/tee"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            env_values = _read_env_file(args.env_file)
            document = prepare_deployment(
                repo_root=args.repo_root,
                repo_url=_configured_value(
                    "GITHUB_REPO_URL", args.repo_url, env_values, DEFAULT_REPO_URL
                ),
                branch=_configured_value("GITHUB_BRANCH", args.branch, env_values, DEFAULT_BRANCH),
                plan_file=args.plan_file,
                manifest_file=args.manifest_file,
                last_good_file=args.last_good_file,
                deploy_commit=_operator_only_value(
                    "GATEWAY_DEPLOY_COMMIT", args.deploy_commit, ""
                ),
            )
            print(document["target_sha"])
            return 0
        if args.command == "activate":
            document = activate_deployment(plan_file=args.plan_file)
            print(document["target_sha"])
            return 0
        if args.command == "field":
            document = _read_json(args.plan_file)
            print(document[args.name])
            return 0
        if args.command == "finalize":
            document = finalize_deployment(
                plan_file=args.plan_file,
                status=args.status,
                stage=args.stage,
                eif_root=args.eif_root,
            )
            print(document["status"])
            return 0
    except (GatewayGitDeployError, KeyError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
