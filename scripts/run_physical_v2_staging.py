#!/usr/bin/env python3
"""Run the attested V2 release on isolated physical staging infrastructure.

This controller deliberately owns no scoring, settlement, or weight business
logic. It invokes the repository's production restart operator and accepts a
release only after the real gateway, primary validator, and independently
running auditors report one identical finalized authority on Bittensor testnet.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "leadpoet.physical_v2_staging_config.v1"
LEDGER_SCHEMA_VERSION = "leadpoet.physical_v2_staging_ledger.v1"
PRODUCTION_ADDRESSES = {"52.91.135.79", "100.59.201.156"}
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")
SAFE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")
SAFE_SECRET_RE = re.compile(r"^[A-Za-z0-9/_+=.@-]+$")
SSH_HOST_RE = re.compile(r"^[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+$")


class PhysicalStagingError(RuntimeError):
    """Raised when physical staging cannot prove the release contract."""


@dataclass(frozen=True)
class RestartHost:
    ssh_host: str
    ssh_key: Path
    restart_path: str
    secret_id: str
    repo_root: str = ""
    container_name: str = ""


@dataclass(frozen=True)
class AuditorHost:
    ssh_host: str
    ssh_key: Path
    repo_root: str
    unit_name: str
    expected_hotkey: str


@dataclass(frozen=True)
class PhysicalStagingConfig:
    network: str
    netuid: int
    gateway_public_url: str
    gateway: RestartHost
    primary_validator: RestartHost
    auditors: tuple[AuditorHost, ...]
    timeout_seconds: int
    poll_seconds: int


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicalStagingError(f"{field} must be an object")
    return value


def _require_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise PhysicalStagingError(f"{field} is required")
    return text


def _address(ssh_host: str) -> str:
    return ssh_host.rsplit("@", 1)[-1].strip("[]")


def _validate_nonproduction_host(ssh_host: str, *, field: str) -> str:
    value = _require_text(ssh_host, field=field)
    if not SSH_HOST_RE.fullmatch(value):
        raise PhysicalStagingError(f"{field} is not a bounded SSH host")
    if _address(value) in PRODUCTION_ADDRESSES:
        raise PhysicalStagingError(f"{field} points at a production host")
    return value


def _validate_path(value: Any, *, field: str, must_exist: bool = False) -> str:
    text = _require_text(value, field=field)
    if not SAFE_PATH_RE.fullmatch(text) or ".." in Path(text).parts:
        raise PhysicalStagingError(f"{field} is not a safe absolute path")
    if must_exist and not Path(text).is_file():
        raise PhysicalStagingError(f"{field} is unavailable")
    return text


def _load_restart_host(
    value: Any,
    *,
    field: str,
    require_repo: bool,
) -> RestartHost:
    item = _require_mapping(value, field=field)
    secret_id = _require_text(item.get("secret_id"), field=f"{field}.secret_id")
    lowered = secret_id.lower()
    if (
        not SAFE_SECRET_RE.fullmatch(secret_id)
        or "/prod/" in lowered
        or "staging" not in lowered
    ):
        raise PhysicalStagingError(
            f"{field}.secret_id must name a non-production staging secret"
        )
    repo_root = ""
    if require_repo:
        repo_root = _validate_path(
            item.get("repo_root"), field=f"{field}.repo_root"
        )
    container_name = str(item.get("container_name") or "").strip()
    if container_name and not SAFE_NAME_RE.fullmatch(container_name):
        raise PhysicalStagingError(f"{field}.container_name is invalid")
    return RestartHost(
        ssh_host=_validate_nonproduction_host(
            item.get("ssh_host"), field=f"{field}.ssh_host"
        ),
        ssh_key=Path(
            _validate_path(
                item.get("ssh_key"),
                field=f"{field}.ssh_key",
                must_exist=True,
            )
        ),
        restart_path=_validate_path(
            item.get("restart_path"), field=f"{field}.restart_path"
        ),
        secret_id=secret_id,
        repo_root=repo_root,
        container_name=container_name,
    )


def load_config(path: Path) -> PhysicalStagingConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhysicalStagingError("physical staging config is unreadable") from exc
    doc = _require_mapping(raw, field="config")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise PhysicalStagingError("physical staging config schema differs")
    if doc.get("environment") != "physical-v2-staging":
        raise PhysicalStagingError("physical staging environment is invalid")
    network = _require_text(doc.get("network"), field="network").lower()
    if network != "test":
        raise PhysicalStagingError("physical staging must use Bittensor testnet")
    netuid = doc.get("netuid")
    if not isinstance(netuid, int) or isinstance(netuid, bool) or netuid <= 0:
        raise PhysicalStagingError("netuid must be a positive integer")
    gateway_public_url = _require_text(
        doc.get("gateway_public_url"), field="gateway_public_url"
    ).rstrip("/")
    parsed_gateway = urlparse(gateway_public_url)
    if (
        parsed_gateway.scheme not in {"http", "https"}
        or not parsed_gateway.hostname
        or parsed_gateway.username is not None
        or parsed_gateway.password is not None
        or parsed_gateway.query
        or parsed_gateway.fragment
        or parsed_gateway.hostname in PRODUCTION_ADDRESSES
    ):
        raise PhysicalStagingError("gateway_public_url is not an isolated URL")

    gateway = _load_restart_host(
        doc.get("gateway"), field="gateway", require_repo=False
    )
    primary = _load_restart_host(
        doc.get("primary_validator"),
        field="primary_validator",
        require_repo=True,
    )
    auditor_docs = doc.get("audit_validators")
    if not isinstance(auditor_docs, list) or len(auditor_docs) < 2:
        raise PhysicalStagingError(
            "at least two independent audit validators are required"
        )
    auditors = []
    for index, raw_auditor in enumerate(auditor_docs):
        field = f"audit_validators[{index}]"
        item = _require_mapping(raw_auditor, field=field)
        unit_name = _require_text(
            item.get("unit_name"), field=f"{field}.unit_name"
        )
        if not SAFE_NAME_RE.fullmatch(unit_name):
            raise PhysicalStagingError(f"{field}.unit_name is invalid")
        hotkey = _require_text(
            item.get("expected_hotkey"), field=f"{field}.expected_hotkey"
        )
        if not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{40,64}", hotkey):
            raise PhysicalStagingError(f"{field}.expected_hotkey is invalid")
        auditors.append(
            AuditorHost(
                ssh_host=_validate_nonproduction_host(
                    item.get("ssh_host"), field=f"{field}.ssh_host"
                ),
                ssh_key=Path(
                    _validate_path(
                        item.get("ssh_key"),
                        field=f"{field}.ssh_key",
                        must_exist=True,
                    )
                ),
                repo_root=_validate_path(
                    item.get("repo_root"), field=f"{field}.repo_root"
                ),
                unit_name=unit_name,
                expected_hotkey=hotkey,
            )
        )
    all_hosts = [gateway.ssh_host, primary.ssh_host] + [
        item.ssh_host for item in auditors
    ]
    if len(set(all_hosts)) != len(all_hosts):
        raise PhysicalStagingError(
            "gateway, primary, and audit validators require distinct hosts"
        )
    timeout_seconds = int(doc.get("timeout_seconds") or 7200)
    poll_seconds = int(doc.get("poll_seconds") or 10)
    if timeout_seconds < 300 or timeout_seconds > 14_400:
        raise PhysicalStagingError("timeout_seconds is outside the bounded range")
    if poll_seconds < 2 or poll_seconds > 60:
        raise PhysicalStagingError("poll_seconds is outside the bounded range")
    return PhysicalStagingConfig(
        network=network,
        netuid=netuid,
        gateway_public_url=gateway_public_url,
        gateway=gateway,
        primary_validator=primary,
        auditors=tuple(auditors),
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )


def _run(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        env=dict(env) if env is not None else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _ssh(host: str, key: Path, remote_command: str, *, timeout: int) -> str:
    result = _run(
        [
            "ssh",
            "-n",
            "-i",
            str(key),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=20",
            host,
            remote_command,
        ],
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise PhysicalStagingError(f"staging SSH command failed: {detail}")
    return result.stdout


def _gateway_json(
    config: PhysicalStagingConfig,
    path: str,
    *,
    allow_not_found: bool = False,
) -> dict[str, Any] | None:
    if not re.fullmatch(r"/[A-Za-z0-9_./:-]+", path):
        raise PhysicalStagingError("gateway probe path is invalid")
    probe = f"""
import json
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen(
        {('http://127.0.0.1:8000' + path)!r}, timeout=35
    ) as response:
        payload = json.load(response)
        status = int(response.status)
except urllib.error.HTTPError as exc:
    payload = None
    status = int(exc.code)
print(json.dumps({{"status": status, "payload": payload}}, sort_keys=True))
"""
    encoded = base64.b64encode(probe.encode("utf-8")).decode("ascii")
    remote = (
        "python3 -c \"import base64;"
        f"exec(base64.b64decode('{encoded}'))\""
    )
    output = _ssh(
        config.gateway.ssh_host,
        config.gateway.ssh_key,
        remote,
        timeout=60,
    )
    try:
        envelope = json.loads(output)
    except json.JSONDecodeError as exc:
        raise PhysicalStagingError("gateway probe did not return JSON") from exc
    if not isinstance(envelope, dict):
        raise PhysicalStagingError("gateway probe did not return an object")
    status = envelope.get("status")
    if status == 404 and allow_not_found:
        return None
    if status != 200:
        raise PhysicalStagingError(f"gateway probe returned HTTP {status}")
    value = envelope.get("payload")
    if not isinstance(value, dict):
        raise PhysicalStagingError("gateway probe payload is not an object")
    return value


def _restart_exact_release(config: PhysicalStagingConfig, commit: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "LEADPOET_GATEWAY_SSH_KEY": str(config.gateway.ssh_key),
            "LEADPOET_VALIDATOR_SSH_KEY": str(config.primary_validator.ssh_key),
            "LEADPOET_GATEWAY_SSH_HOST": config.gateway.ssh_host,
            "LEADPOET_VALIDATOR_SSH_HOST": config.primary_validator.ssh_host,
            "LEADPOET_GATEWAY_RESTART_PATH": config.gateway.restart_path,
            "LEADPOET_VALIDATOR_RESTART_PATH": config.primary_validator.restart_path,
            "LEADPOET_VALIDATOR_REPO_ROOT": config.primary_validator.repo_root,
            "LEADPOET_GATEWAY_ENV_SECRET_ID": config.gateway.secret_id,
            "LEADPOET_VALIDATOR_ENV_SECRET_ID": config.primary_validator.secret_id,
        }
    )
    result = _run(
        [
            "bash",
            "scripts/restart_attested_release_local.sh",
            "--commit",
            commit,
            "--component",
            "all",
        ],
        env=env,
        timeout=config.timeout_seconds,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise PhysicalStagingError(f"canonical paired restart failed: {detail}")


def _restart_auditor(auditor: AuditorHost, commit: str) -> None:
    command = "\n".join(
        [
            "set -Eeuo pipefail",
            f"repo='{auditor.repo_root}'",
            "test -z \"$(git -C \"$repo\" status --porcelain --untracked-files=no)\"",
            "git -C \"$repo\" fetch origin main",
            f"test \"$(git -C \"$repo\" rev-parse origin/main)\" = '{commit}'",
            "git -C \"$repo\" checkout main",
            "git -C \"$repo\" merge --ff-only origin/main",
            f"test \"$(git -C \"$repo\" rev-parse HEAD)\" = '{commit}'",
            "git -C \"$repo\" diff --quiet HEAD --",
            f"sudo systemctl restart '{auditor.unit_name}'",
            f"sudo systemctl is-active --quiet '{auditor.unit_name}'",
        ]
    )
    _ssh(auditor.ssh_host, auditor.ssh_key, command, timeout=180)


def _auditor_events(
    auditor: AuditorHost,
    *,
    since_epoch_seconds: int,
) -> list[dict[str, Any]]:
    output = _ssh(
        auditor.ssh_host,
        auditor.ssh_key,
        (
            f"sudo journalctl -u '{auditor.unit_name}' "
            f"--since '@{since_epoch_seconds}' --no-pager -o cat"
        ),
        timeout=60,
    )
    events = []
    for line in output.splitlines():
        marker = "auditor_event "
        if marker not in line:
            continue
        try:
            value = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _primary_finalized(
    config: PhysicalStagingConfig,
    *,
    since_epoch_seconds: int,
) -> bool:
    container = config.primary_validator.container_name or "leadpoet-validator-main"
    output = _ssh(
        config.primary_validator.ssh_host,
        config.primary_validator.ssh_key,
        f"docker logs --since '{since_epoch_seconds}' '{container}' 2>&1",
        timeout=60,
    )
    return (
        "Authoritative V2 gateway bundle persisted:" in output
        and "Authoritative V2 finalized chain state persisted:" in output
    )


def _authority_identity(authority: Mapping[str, Any]) -> tuple[int, str, str]:
    if authority.get("authority_stage") != "finalized":
        raise PhysicalStagingError("canonical authority is not finalized")
    epoch_id = authority.get("epoch_id")
    if not isinstance(epoch_id, int) or isinstance(epoch_id, bool):
        submission = authority.get("compact_submission")
        if isinstance(submission, Mapping):
            weight_result = submission.get("weight_result")
            if isinstance(weight_result, Mapping):
                epoch_id = weight_result.get("epoch_id")
    bundle_hash = str(authority.get("bundle_hash") or "")
    submission = authority.get("compact_submission")
    weight_result = (
        submission.get("weight_result")
        if isinstance(submission, Mapping)
        else None
    )
    weights_hash = str(
        weight_result.get("weights_hash")
        if isinstance(weight_result, Mapping)
        else ""
    )
    if (
        not isinstance(epoch_id, int)
        or isinstance(epoch_id, bool)
        or epoch_id < 0
        or not HASH_RE.fullmatch(bundle_hash)
        or not HASH_RE.fullmatch(weights_hash)
    ):
        raise PhysicalStagingError("canonical authority identity is invalid")
    return epoch_id, bundle_hash, weights_hash


def _matching_auditor_success(
    events: Sequence[Mapping[str, Any]],
    *,
    netuid: int,
    epoch_id: int,
    bundle_hash: str,
    weights_hash: str,
) -> bool:
    for event in events:
        try:
            matched = (
                event.get("event") == "submission_success"
                and int(event.get("netuid", -1)) == netuid
                and int(event.get("epoch", -1)) == epoch_id
                and event.get("bundle_hash") == bundle_hash
                and event.get("weights_hash") == weights_hash
                and event.get("confirmation_stage")
                == "timelocked_commit_finalized"
            )
        except (TypeError, ValueError):
            matched = False
        if matched:
            return True
    return False


def _matching_auditor_startup(
    events: Sequence[Mapping[str, Any]],
    *,
    commit: str,
    netuid: int,
    hotkey: str,
    gateway_public_url: str,
) -> bool:
    for event in events:
        try:
            matched = (
                event.get("event") == "startup_ready"
                and str(event.get("commit") or "").lower() == commit
                and int(event.get("netuid", -1)) == netuid
                and event.get("hotkey") == hotkey
                and str(event.get("gateway_endpoint") or "").rstrip("/")
                == gateway_public_url
                and event.get("weight_protocol") == "authoritative_v2"
            )
        except (TypeError, ValueError):
            matched = False
        if matched:
            return True
    return False


def _wait_for_canonical_acceptance(
    config: PhysicalStagingConfig,
    *,
    commit: str,
    started_at: int,
) -> dict[str, Any]:
    epoch_state = _gateway_json(config, "/epoch/current")
    assert epoch_state is not None
    first_epoch = int(epoch_state["current_epoch_id"])
    deadline = time.monotonic() + config.timeout_seconds
    last_reason = "no authority observed"
    while time.monotonic() < deadline:
        epoch_state = _gateway_json(config, "/epoch/current")
        assert epoch_state is not None
        current_epoch = int(epoch_state["current_epoch_id"])
        for epoch_id in range(first_epoch, current_epoch + 1):
            try:
                authority = _gateway_json(
                    config,
                    f"/weights/v2/published-compact/{config.netuid}/{epoch_id}",
                    allow_not_found=True,
                )
                if authority is None:
                    last_reason = "canonical authority pending"
                    continue
                observed_epoch, bundle_hash, weights_hash = _authority_identity(
                    authority
                )
            except PhysicalStagingError:
                raise
            if observed_epoch != epoch_id:
                raise PhysicalStagingError("canonical authority epoch differs")
            if not _primary_finalized(
                config, since_epoch_seconds=started_at
            ):
                last_reason = "primary finalization log pending"
                continue
            auditor_results = {}
            for auditor in config.auditors:
                events = _auditor_events(
                    auditor, since_epoch_seconds=started_at
                )
                matched = _matching_auditor_startup(
                    events,
                    commit=commit,
                    netuid=config.netuid,
                    hotkey=auditor.expected_hotkey,
                    gateway_public_url=config.gateway_public_url,
                ) and _matching_auditor_success(
                    events,
                    netuid=config.netuid,
                    epoch_id=epoch_id,
                    bundle_hash=bundle_hash,
                    weights_hash=weights_hash,
                )
                auditor_results[auditor.expected_hotkey] = matched
            if all(auditor_results.values()):
                return {
                    "epoch_id": epoch_id,
                    "bundle_hash": bundle_hash,
                    "weights_hash": weights_hash,
                    "authority_stage": "finalized",
                    "primary_finalized": True,
                    "auditors": auditor_results,
                }
            last_reason = "one or more auditor finalizations are pending"
        time.sleep(config.poll_seconds)
    raise PhysicalStagingError(
        f"physical V2 acceptance timed out: {last_reason}"
    )


def run(config: PhysicalStagingConfig, *, commit: str) -> dict[str, Any]:
    if not SHA_RE.fullmatch(commit):
        raise PhysicalStagingError("candidate commit must be a full lowercase SHA")
    fetch = _run(["git", "fetch", "origin", "main"], timeout=120)
    if fetch.returncode != 0:
        raise PhysicalStagingError("could not fetch candidate authority")
    remote = _run(["git", "rev-parse", "origin/main"], timeout=30)
    if remote.returncode != 0 or remote.stdout.strip() != commit:
        raise PhysicalStagingError("candidate is not the exact origin/main tip")

    started_at = int(time.time())
    _restart_exact_release(config, commit)
    health = _gateway_json(config, "/health/v2-authority")
    build = _gateway_json(config, "/build-info")
    if (
        health.get("status") != "ready"
        or str(health.get("commit_sha") or "").lower() != commit
        or str(build.get("git_commit") or "").lower() != commit
    ):
        raise PhysicalStagingError("staging gateway exact release is not ready")
    for auditor in config.auditors:
        _restart_auditor(auditor, commit)
    evidence = _wait_for_canonical_acceptance(
        config,
        commit=commit,
        started_at=started_at,
    )
    final_fetch = _run(["git", "fetch", "origin", "main"], timeout=120)
    final_remote = _run(["git", "rev-parse", "origin/main"], timeout=30)
    if (
        final_fetch.returncode != 0
        or final_remote.returncode != 0
        or final_remote.stdout.strip() != commit
    ):
        raise PhysicalStagingError("candidate was superseded during physical staging")
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "status": "passed",
        "candidate_sha": commit,
        "environment": "physical-v2-staging",
        "network": config.network,
        "netuid": config.netuid,
        "started_at": datetime.fromtimestamp(
            started_at, tz=timezone.utc
        ).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "evidence": evidence,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        ledger = run(config, commit=str(args.candidate_sha).lower())
    except (PhysicalStagingError, KeyError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(
        json.dumps(ledger, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(ledger, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
