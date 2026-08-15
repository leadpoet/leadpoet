#!/usr/bin/env python3
"""Launch the exact candidate auditor from one run-scoped staging secret."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping
from urllib.parse import urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SECRET_RE = re.compile(r"^leadpoet/staging/production-parity/[a-z0-9-]{6,40}/auditor-[ab]$")
ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PRODUCTION_HOSTS = {"52.91.135.79", "100.59.201.156"}


class AuditorLaunchError(RuntimeError):
    pass


def _secret_environment(secret_id: str) -> dict[str, str]:
    try:
        response = boto3.client("secretsmanager").get_secret_value(SecretId=secret_id)
        value = json.loads(str(response.get("SecretString") or ""))
    except (BotoCoreError, ClientError, ValueError) as exc:
        raise AuditorLaunchError("run-scoped auditor secret is unavailable") from exc
    if not isinstance(value, Mapping) or not value:
        raise AuditorLaunchError("run-scoped auditor secret is invalid")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "")
        if not ENV_RE.fullmatch(key) or isinstance(raw_value, (dict, list)):
            raise AuditorLaunchError("run-scoped auditor environment is invalid")
        result[key] = "" if raw_value is None else str(raw_value)
    return result


def _require_candidate_checkout(root: Path, candidate_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if (
        head.returncode != 0
        or head.stdout.strip().lower() != candidate_sha
        or dirty.returncode != 0
        or dirty.stdout.strip()
    ):
        raise AuditorLaunchError("auditor checkout differs from the exact candidate")


def launch() -> None:
    secret_id = str(os.environ.get("LEADPOET_PARITY_AUDITOR_SECRET_ID") or "")
    candidate_sha = str(os.environ.get("LEADPOET_PARITY_CANDIDATE_SHA") or "").lower()
    if not SECRET_RE.fullmatch(secret_id) or not SHA_RE.fullmatch(candidate_sha):
        raise AuditorLaunchError("auditor staging identity is invalid")
    root = Path(__file__).resolve().parents[1]
    _require_candidate_checkout(root, candidate_sha)
    values = _secret_environment(secret_id)
    gateway_url = values.get("GATEWAY_URL", "")
    parsed_gateway = urlparse(gateway_url)
    required = {
        "AUDITOR_WEIGHT_PROTOCOL": "authoritative_v2",
        "BITTENSOR_NETWORK": "test",
        "BT_SUBTENSOR_NETWORK": "test",
        "LEADPOET_PARITY_CANDIDATE_SHA": candidate_sha,
    }
    if (
        any(values.get(key) != expected for key, expected in required.items())
        or parsed_gateway.scheme != "https"
        or not parsed_gateway.hostname
        or parsed_gateway.hostname in PRODUCTION_HOSTS
        or parsed_gateway.username is not None
        or parsed_gateway.password is not None
    ):
        raise AuditorLaunchError("auditor staging trust boundary is invalid")
    for key in ("BT_WALLET_NAME", "BT_WALLET_HOTKEY", "BT_WALLET_PATH"):
        if not values.get(key):
            raise AuditorLaunchError(f"auditor staging wallet setting is missing: {key}")
    try:
        netuid = str(int(values["BITTENSOR_NETUID"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditorLaunchError("auditor staging netuid is invalid") from exc
    environment = os.environ.copy()
    environment.update(values)
    environment.update(
        {
            "GIT_COMMIT": candidate_sha,
            "LEADPOET_AUDITOR_WRAPPER_ACTIVE": "1",
            "LEADPOET_SENTRY_RELEASE": candidate_sha,
        }
    )
    argv = [
        sys.executable,
        str(root / "neurons" / "auditor_validator.py"),
        "--netuid",
        netuid,
        "--gateway-url",
        gateway_url,
        "--wallet.name",
        values["BT_WALLET_NAME"],
        "--wallet.hotkey",
        values["BT_WALLET_HOTKEY"],
        "--wallet.path",
        values["BT_WALLET_PATH"],
        "--subtensor.network",
        "test",
    ]
    os.execve(sys.executable, argv, environment)


if __name__ == "__main__":
    try:
        launch()
    except (OSError, AuditorLaunchError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
