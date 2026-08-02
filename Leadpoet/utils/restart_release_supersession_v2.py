"""Resolve the current forward-release authority during restart acquisition."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
from typing import Sequence


_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class RestartReleaseSupersessionV2Error(RuntimeError):
    """Raised when the forward release authority cannot be resolved safely."""


def _run_git(repo_root: Path, *args: str, timeout: int = 30) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RestartReleaseSupersessionV2Error(
            f"git {' '.join(args)} failed while resolving restart authority"
        ) from exc
    return completed.stdout.strip()


def resolve_forward_release_head(
    *,
    repo_root: Path,
    expected_commit: str,
    branch: str = "main",
) -> str:
    """Return the fetched branch head when it remains a fast-forward authority."""

    root = repo_root.expanduser().resolve()
    expected = str(expected_commit or "").strip().lower()
    if not root.is_dir():
        raise RestartReleaseSupersessionV2Error("restart repository is unavailable")
    if not _FULL_SHA_RE.fullmatch(expected):
        raise RestartReleaseSupersessionV2Error(
            "expected restart commit must be a lowercase full SHA"
        )
    if not branch or branch.startswith("-"):
        raise RestartReleaseSupersessionV2Error("restart authority branch is invalid")

    _run_git(root, "check-ref-format", "--branch", branch, timeout=10)
    remote_ref = f"refs/heads/{branch}"
    tracking_ref = f"refs/remotes/origin/{branch}"
    remote_rows = _run_git(
        root,
        "ls-remote",
        "--exit-code",
        "origin",
        remote_ref,
    ).splitlines()
    if len(remote_rows) != 1:
        raise RestartReleaseSupersessionV2Error(
            "restart authority did not resolve to one remote branch head"
        )
    remote_sha, separator, observed_ref = remote_rows[0].partition("\t")
    remote_sha = remote_sha.strip().lower()
    if separator != "\t" or observed_ref.strip() != remote_ref:
        raise RestartReleaseSupersessionV2Error(
            "restart authority remote response is malformed"
        )
    if not _FULL_SHA_RE.fullmatch(remote_sha):
        raise RestartReleaseSupersessionV2Error(
            "restart authority did not resolve to a full Git commit"
        )
    if remote_sha == expected:
        return expected

    _run_git(
        root,
        "fetch",
        "--prune",
        "origin",
        f"+{remote_ref}:{tracking_ref}",
        timeout=120,
    )
    fetched_sha = _run_git(root, "rev-parse", f"{tracking_ref}^{{commit}}").lower()
    if not _FULL_SHA_RE.fullmatch(fetched_sha):
        raise RestartReleaseSupersessionV2Error(
            "fetched restart authority is not a full Git commit"
        )
    ancestry = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", expected, fetched_sha],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if ancestry.returncode == 1:
        raise RestartReleaseSupersessionV2Error(
            "restart authority changed by a non-fast-forward update"
        )
    if ancestry.returncode != 0:
        raise RestartReleaseSupersessionV2Error(
            "restart authority ancestry could not be verified"
        )
    return fetched_sha


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--branch", default="main")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        resolved = resolve_forward_release_head(
            repo_root=args.repo_root,
            expected_commit=args.expected_commit,
            branch=args.branch,
        )
    except RestartReleaseSupersessionV2Error as exc:
        raise SystemExit(str(exc)) from exc
    print(resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
