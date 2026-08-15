#!/usr/bin/env python3
"""Resolve the exact deployed gateway SHA through its public build identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_BUILD_INFO_BYTES = 64 * 1024


class DeployedShaResolutionError(RuntimeError):
    """The production runtime identity cannot safely select an N-1 source."""


def _build_info_url(gateway_url: str) -> str:
    parsed = urlsplit(str(gateway_url or "").strip())
    try:
        port = parsed.port
    except ValueError as exc:
        raise DeployedShaResolutionError("production gateway URL port is invalid") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise DeployedShaResolutionError(
            "production gateway URL is outside the HTTPS build-info boundary"
        )
    return urlunsplit(("https", parsed.netloc, "/build-info", "", ""))


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def resolve_deployed_sha(
    *,
    root: Path,
    gateway_url: str,
    candidate_sha: str,
    require_distinct: bool = False,
    opener: Callable[..., Any] = urlopen,
) -> str:
    candidate = str(candidate_sha or "").strip().lower()
    if not SHA_RE.fullmatch(candidate):
        raise DeployedShaResolutionError("candidate SHA is invalid")
    request = Request(
        _build_info_url(gateway_url),
        headers={"Accept": "application/json", "User-Agent": "leadpoet-parity-v1"},
    )
    try:
        with opener(request, timeout=10) as response:
            if int(response.status) != 200:
                raise DeployedShaResolutionError(
                    "production gateway build-info did not return HTTP 200"
                )
            payload = response.read(MAX_BUILD_INFO_BYTES + 1)
    except DeployedShaResolutionError:
        raise
    except Exception as exc:
        raise DeployedShaResolutionError(
            "production gateway build identity is unavailable"
        ) from exc
    if len(payload) > MAX_BUILD_INFO_BYTES:
        raise DeployedShaResolutionError(
            "production gateway build identity exceeds its bound"
        )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise DeployedShaResolutionError(
            "production gateway build identity is invalid"
        ) from exc
    if not isinstance(value, Mapping) or value.get("is_commit_known") is not True:
        raise DeployedShaResolutionError(
            "production gateway does not expose an exact build identity"
        )
    deployed = str(value.get("git_commit") or "").strip().lower()
    if not SHA_RE.fullmatch(deployed):
        raise DeployedShaResolutionError("deployed gateway SHA is invalid")
    if require_distinct and deployed == candidate:
        raise DeployedShaResolutionError(
            "deployed gateway already equals the candidate; explicit N-1 is required"
        )
    if _git(root, "cat-file", "-e", f"{deployed}^{{commit}}").returncode != 0:
        raise DeployedShaResolutionError(
            "deployed gateway SHA is absent from candidate history"
        )
    if _git(root, "merge-base", "--is-ancestor", deployed, candidate).returncode != 0:
        raise DeployedShaResolutionError(
            "deployed gateway SHA is not an ancestor of the candidate"
        )
    return deployed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-distinct", action="store_true")
    args = parser.parse_args(argv)
    try:
        deployed = resolve_deployed_sha(
            root=args.root.resolve(),
            gateway_url=args.gateway_url,
            candidate_sha=args.candidate_sha,
            require_distinct=bool(args.require_distinct),
        )
    except (OSError, ValueError, DeployedShaResolutionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(deployed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
