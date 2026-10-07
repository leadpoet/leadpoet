"""Best-effort source identity for private Arena runtime audit records."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict


_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_UNKNOWN = {
    "validator_source_commit": "unknown",
    "validator_source_origin": "unknown",
    "validator_source_dirty": "unknown",
}


def source_metadata(module_file: str) -> Dict[str, str]:
    """Identify the checkout that loaded the runner, never the current directory.

    A release archive carries its exact committed source in `.release-commit`.
    A developer checkout uses its HEAD and records whether local changes mean
    those source bytes may differ. Failure to inspect is audit-only.
    """

    try:
        root = Path(module_file).resolve().parent.parent
        marker = root / ".release-commit"
        if marker.exists() or marker.is_symlink():
            if not marker.is_file() or marker.is_symlink() or marker.stat().st_size > 128:
                return dict(_UNKNOWN)
            commit = marker.read_text(encoding="ascii").strip()
            if not _COMMIT_RE.fullmatch(commit):
                return dict(_UNKNOWN)
            return {
                "validator_source_commit": commit,
                "validator_source_origin": "release_marker",
                "validator_source_dirty": "unknown",
            }

        # Ignore inherited Git routing variables so -C remains authoritative.
        git_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}

        def git(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", "--no-optional-locks", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
                env=git_env,
            )

        top_level = git("rev-parse", "--show-toplevel")
        if top_level.returncode or Path(top_level.stdout.strip()).resolve() != root:
            return dict(_UNKNOWN)
        head = git("rev-parse", "--verify", "HEAD^{commit}")
        commit = head.stdout.strip()
        if head.returncode or not _COMMIT_RE.fullmatch(commit):
            return dict(_UNKNOWN)
        try:
            status = git("status", "--porcelain=v1", "--untracked-files=normal")
        except (OSError, UnicodeError, ValueError, subprocess.TimeoutExpired):
            dirty = "unknown"
        else:
            dirty = (
                "unknown" if status.returncode
                else "dirty" if status.stdout else "clean"
            )
        return {
            "validator_source_commit": commit,
            "validator_source_origin": "git_checkout",
            "validator_source_dirty": dirty,
        }
    except (OSError, UnicodeError, ValueError, subprocess.TimeoutExpired):
        return dict(_UNKNOWN)


def commit_or_unknown(value: object) -> str:
    """Project an optional audit SHA without trusting arbitrary lease text."""

    return value if isinstance(value, str) and _COMMIT_RE.fullmatch(value) else "unknown"


# One process snapshot. A later checkout update cannot rewrite this process's
# audit identity; a release process takes a fresh snapshot on import.
SOURCE_METADATA = source_metadata(__file__)
