#!/usr/bin/env python3
"""Build and run the exact dashboard main against disposable parity state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

import boto3
from botocore.exceptions import BotoCoreError, ClientError


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
DOMAIN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
SECRET_RE = re.compile(
    r"^leadpoet/staging/production-parity/[a-z0-9-]{6,40}/dashboard$"
)
IMAGE_RE = re.compile(r"^[A-Za-z0-9._/:@-]+@sha256:[0-9a-f]{64}$")
ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DashboardRuntimeError(RuntimeError):
    pass


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _require(result: subprocess.CompletedProcess[str], *, stage: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-800:]
        raise DashboardRuntimeError(f"{stage} failed: {detail}")
    return result.stdout


def _secret_environment(secret_id: str) -> dict[str, str]:
    try:
        response = boto3.client("secretsmanager").get_secret_value(
            SecretId=secret_id
        )
        value = json.loads(str(response.get("SecretString") or ""))
    except (BotoCoreError, ClientError, ValueError) as exc:
        raise DashboardRuntimeError(
            "run-scoped dashboard secret is unavailable"
        ) from exc
    if not isinstance(value, Mapping) or not value:
        raise DashboardRuntimeError("run-scoped dashboard secret is invalid")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "")
        if not ENV_RE.fullmatch(key) or isinstance(raw_value, (dict, list)):
            raise DashboardRuntimeError("dashboard environment is invalid")
        result[key] = "" if raw_value is None else str(raw_value)
    return result


def _validate_environment(
    values: Mapping[str, str], *, dashboard_domain: str
) -> None:
    supabase = urlparse(values.get("NEXT_PUBLIC_SUPABASE_URL", ""))
    site = urlparse(values.get("NEXT_PUBLIC_SITE_URL", ""))
    gateway = urlparse(values.get("GATEWAY_URL", ""))
    if (
        supabase.scheme != "https"
        or not supabase.hostname
        or not supabase.hostname.startswith("database-")
        or site.scheme != "https"
        or site.hostname != dashboard_domain
        or gateway.scheme != "https"
        or not gateway.hostname
        or not gateway.hostname.startswith("gateway-")
        or not values.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        or not values.get("SUPABASE_SECRET_KEY")
        or values.get("RESEARCH_LAB_ALERT_MONITOR_ENABLED") != "false"
        or values.get("RESEARCH_LAB_EVENT_MONITOR_ENABLED") != "false"
    ):
        raise DashboardRuntimeError("dashboard staging boundary is invalid")


def _verify_checkout(repo_root: Path, source_sha: str) -> None:
    head = _require(
        _run(["git", "rev-parse", "HEAD"], cwd=repo_root, timeout=20),
        stage="dashboard source identity",
    ).strip()
    dirty = _require(
        _run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            timeout=20,
        ),
        stage="dashboard source cleanliness",
    ).strip()
    if head != source_sha or dirty:
        raise DashboardRuntimeError("dashboard checkout differs from frozen main")


def build(
    *,
    run_id: str,
    source_sha: str,
    repo_root: Path,
    secret_id: str,
    dashboard_domain: str,
    caddy_image: str,
) -> dict[str, Any]:
    if (
        not RUN_RE.fullmatch(run_id)
        or not SHA_RE.fullmatch(source_sha)
        or not SECRET_RE.fullmatch(secret_id)
        or not DOMAIN_RE.fullmatch(dashboard_domain)
        or not IMAGE_RE.fullmatch(caddy_image)
    ):
        raise DashboardRuntimeError("dashboard run identity is invalid")
    _verify_checkout(repo_root, source_sha)
    values = _secret_environment(secret_id)
    _validate_environment(values, dashboard_domain=dashboard_domain)
    environment = os.environ.copy()
    environment.update(values)
    environment["NODE_ENV"] = "production"
    lockfile = repo_root / "package-lock.json"
    if not lockfile.is_file():
        raise DashboardRuntimeError("dashboard package lock is unavailable")
    _require(
        _run(["npm", "ci", "--ignore-scripts=false"], cwd=repo_root, env=environment, timeout=1200),
        stage="dashboard dependency install",
    )
    _require(
        _run(["npm", "run", "build"], cwd=repo_root, env=environment, timeout=1200),
        stage="dashboard production build",
    )
    runtime_root = Path("/run/leadpoet-production-parity") / run_id / "dashboard"
    runtime_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    caddyfile = runtime_root / "Caddyfile"
    caddyfile.write_text(
        f"{dashboard_domain} {{\n  reverse_proxy 127.0.0.1:3000\n}}\n",
        encoding="utf-8",
    )
    caddyfile.chmod(0o600)
    caddy_name = f"leadpoet-parity-{run_id}-dashboard-caddy"
    caddy_volume = f"leadpoet-parity-{run_id}-dashboard-caddy-data"
    _require(
        _run(["docker", "volume", "create", caddy_volume], timeout=30),
        stage="dashboard Caddy volume",
    )
    _require(
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                caddy_name,
                "--network",
                "host",
                "-v",
                f"{caddyfile}:/etc/caddy/Caddyfile:ro",
                "-v",
                f"{caddy_volume}:/data",
                "--label",
                f"leadpoet.parity.run={run_id}",
                "--label",
                f"leadpoet.dashboard.sha={source_sha}",
                caddy_image,
            ],
            timeout=60,
        ),
        stage="dashboard TLS proxy",
    )
    return {
        "run_id": run_id,
        "source_sha": source_sha,
        "dashboard_domain": dashboard_domain,
        "caddy_container": caddy_name,
        "status": "built",
    }


def serve(
    *, repo_root: Path, source_sha: str, secret_id: str, dashboard_domain: str
) -> None:
    _verify_checkout(repo_root, source_sha)
    values = _secret_environment(secret_id)
    _validate_environment(values, dashboard_domain=dashboard_domain)
    environment = os.environ.copy()
    environment.update(values)
    environment["NODE_ENV"] = "production"
    npm = shutil.which("npm")
    if not npm:
        raise DashboardRuntimeError("dashboard npm executable is unavailable")
    os.execve(
        npm,
        [npm, "start", "--", "--hostname", "127.0.0.1", "--port", "3000"],
        environment,
    )


def wait_ready(*, dashboard_domain: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "dashboard endpoint pending"
    while time.monotonic() < deadline:
        try:
            with urlopen(
                f"https://{dashboard_domain}/api/research-lab", timeout=15
            ) as response:
                value = json.load(response)
                if (
                    int(response.status) == 200
                    and isinstance(value, Mapping)
                    and value.get("success") is True
                    and isinstance(value.get("data"), Mapping)
                ):
                    return
                last_error = "dashboard API returned an invalid document"
        except Exception as exc:
            last_error = type(exc).__name__
        time.sleep(3)
    raise DashboardRuntimeError(f"dashboard did not become ready: {last_error}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "serve"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--source-sha", required=True)
        sub.add_argument("--repo-root", type=Path, required=True)
        sub.add_argument("--secret-id", required=True)
        sub.add_argument("--dashboard-domain", required=True)
        if command == "build":
            sub.add_argument("--run-id", required=True)
            sub.add_argument("--caddy-image", required=True)
    wait = subparsers.add_parser("wait-ready")
    wait.add_argument("--dashboard-domain", required=True)
    wait.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build(
                run_id=args.run_id,
                source_sha=str(args.source_sha).lower(),
                repo_root=args.repo_root,
                secret_id=args.secret_id,
                dashboard_domain=args.dashboard_domain,
                caddy_image=args.caddy_image,
            )
            print(json.dumps(result, sort_keys=True))
        elif args.command == "serve":
            serve(
                repo_root=args.repo_root,
                source_sha=str(args.source_sha).lower(),
                secret_id=args.secret_id,
                dashboard_domain=args.dashboard_domain,
            )
        else:
            wait_ready(
                dashboard_domain=str(args.dashboard_domain),
                timeout_seconds=int(args.timeout_seconds),
            )
            print(json.dumps({"status": "ready"}, sort_keys=True))
    except (
        OSError,
        ValueError,
        BotoCoreError,
        ClientError,
        DashboardRuntimeError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
