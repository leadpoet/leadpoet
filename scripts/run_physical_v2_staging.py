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
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_canonical.production_parity import (
    ProductionParityError,
    StageLedger,
    required_oracle_stage_ids,
    sha256_json,
    validate_contract,
    validate_historical_oracle,
    validate_snapshot_manifest,
    verify_contract_checkout,
    production_database_host_hash,
)
from leadpoet_canonical.weights import weights_within_tolerance

SCHEMA_VERSION = "leadpoet.physical_v2_staging_config.v2"
PRODUCTION_ADDRESSES = {"52.91.135.79", "100.59.201.156"}
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")
SAFE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")
SAFE_SECRET_RE = re.compile(r"^[A-Za-z0-9/_+=.@-]+$")
SSH_HOST_RE = re.compile(r"^[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+$")
FULL_CRITICAL_STAGES = (
    "candidate-and-snapshot-identity",
    "exact-paired-restart",
    "staging-control-boundary",
    "full-rebenchmark-and-assignment",
    "dashboard-score-readback",
    "canonical-weight-bundles",
    "primary-finalization",
    "audit-finalization",
    "consecutive-epoch-readback",
    "candidate-not-superseded",
)


class PhysicalStagingError(RuntimeError):
    """Raised when physical staging cannot prove the release contract."""


@dataclass(frozen=True)
class RestartHost:
    ssh_host: str
    ssh_key: Path
    restart_path: str
    secret_id: str
    repo_root: str = ""
    python_bin: str = ""
    container_name: str = ""
    expected_hotkey: str = ""


@dataclass(frozen=True)
class AuditorHost:
    ssh_host: str
    ssh_key: Path
    repo_root: str
    unit_name: str
    expected_hotkey: str
    secret_id: str


@dataclass(frozen=True)
class DashboardHost:
    ssh_host: str
    ssh_key: Path
    repo_root: str
    unit_name: str
    source_sha: str


@dataclass(frozen=True)
class PhysicalStagingConfig:
    network: str
    netuid: int
    chain_endpoint: str
    network_genesis_hash: str
    gateway_public_url: str
    gateway: RestartHost
    primary_validator: RestartHost
    auditors: tuple[AuditorHost, ...]
    timeout_seconds: int
    poll_seconds: int
    required_consecutive_epochs: int
    rebenchmark_timeout_seconds: int
    dashboard_report_url: str
    dashboard: DashboardHost
    ephemeral_stack_id: str


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
    require_python: bool = False,
    require_hotkey: bool = False,
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
    python_bin = str(item.get("python_bin") or "").strip()
    if require_python:
        python_bin = _validate_path(
            python_bin, field=f"{field}.python_bin"
        )
    elif python_bin:
        python_bin = _validate_path(
            python_bin, field=f"{field}.python_bin"
        )
    container_name = str(item.get("container_name") or "").strip()
    if container_name and not SAFE_NAME_RE.fullmatch(container_name):
        raise PhysicalStagingError(f"{field}.container_name is invalid")
    expected_hotkey = str(item.get("expected_hotkey") or "").strip()
    if require_hotkey and not re.fullmatch(
        r"[1-9A-HJ-NP-Za-km-z]{40,64}", expected_hotkey
    ):
        raise PhysicalStagingError(f"{field}.expected_hotkey is invalid")
    if not require_hotkey and expected_hotkey:
        raise PhysicalStagingError(f"{field}.expected_hotkey is unexpected")
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
        python_bin=python_bin,
        container_name=container_name,
        expected_hotkey=expected_hotkey,
    )


def load_config(path: Path) -> PhysicalStagingConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhysicalStagingError("physical staging config is unreadable") from exc
    doc = _require_mapping(raw, field="config")
    schema_version = str(doc.get("schema_version") or "")
    if schema_version != SCHEMA_VERSION:
        raise PhysicalStagingError("physical staging config schema differs")
    if doc.get("environment") != "production-parity-ephemeral":
        raise PhysicalStagingError("physical staging environment is invalid")
    network = _require_text(doc.get("network"), field="network").lower()
    if network != "test":
        raise PhysicalStagingError("physical staging must use Bittensor testnet")
    netuid = doc.get("netuid")
    if not isinstance(netuid, int) or isinstance(netuid, bool) or netuid <= 0:
        raise PhysicalStagingError("netuid must be a positive integer")
    chain_endpoint = _require_text(
        doc.get("chain_endpoint"), field="chain_endpoint"
    )
    parsed_chain = urlparse(chain_endpoint)
    network_genesis_hash = str(
        doc.get("network_genesis_hash") or ""
    ).lower()
    if (
        parsed_chain.scheme != "wss"
        or parsed_chain.hostname != "test.finney.opentensor.ai"
        or parsed_chain.port not in (None, 443)
        or parsed_chain.path not in ("", "/")
        or parsed_chain.query
        or parsed_chain.fragment
        or parsed_chain.username is not None
        or parsed_chain.password is not None
        or not re.fullmatch(r"0x[0-9a-f]{64}", network_genesis_hash)
    ):
        raise PhysicalStagingError("testnet chain authority is invalid")
    gateway_public_url = _require_text(
        doc.get("gateway_public_url"), field="gateway_public_url"
    ).rstrip("/")
    parsed_gateway = urlparse(gateway_public_url)
    if (
        parsed_gateway.scheme != "https"
        or not parsed_gateway.hostname
        or parsed_gateway.username is not None
        or parsed_gateway.password is not None
        or parsed_gateway.query
        or parsed_gateway.fragment
        or parsed_gateway.hostname in PRODUCTION_ADDRESSES
    ):
        raise PhysicalStagingError("gateway_public_url is not an isolated URL")

    gateway = _load_restart_host(
        doc.get("gateway"),
        field="gateway",
        require_repo=True,
        require_python=True,
    )
    primary = _load_restart_host(
        doc.get("primary_validator"),
        field="primary_validator",
        require_repo=True,
        require_hotkey=True,
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
        secret_id = _require_text(
            item.get("secret_id"), field=f"{field}.secret_id"
        )
        if (
            not SAFE_SECRET_RE.fullmatch(secret_id)
            or not secret_id.startswith("leadpoet/staging/production-parity/")
        ):
            raise PhysicalStagingError(f"{field}.secret_id is invalid")
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
                secret_id=secret_id,
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
    required_consecutive_epochs = int(
        doc.get("required_consecutive_epochs") or 3
    )
    if required_consecutive_epochs < 1 or required_consecutive_epochs > 3:
        raise PhysicalStagingError(
            "required_consecutive_epochs is outside the bounded range"
        )
    rebenchmark_timeout_seconds = int(
        doc.get("rebenchmark_timeout_seconds") or timeout_seconds
    )
    if rebenchmark_timeout_seconds < 300 or rebenchmark_timeout_seconds > 14_400:
        raise PhysicalStagingError(
            "rebenchmark_timeout_seconds is outside the bounded range"
        )
    dashboard_report_url = str(doc.get("dashboard_report_url") or "").strip()
    dashboard_source_sha = str(doc.get("dashboard_source_sha") or "").lower()
    ephemeral_stack_id = str(doc.get("ephemeral_stack_id") or "").strip()
    parsed_dashboard = urlparse(dashboard_report_url)
    if (
        not ephemeral_stack_id
        or not SAFE_NAME_RE.fullmatch(ephemeral_stack_id)
        or parsed_dashboard.scheme != "https"
        or not parsed_dashboard.hostname
        or parsed_dashboard.hostname in PRODUCTION_ADDRESSES
        or parsed_dashboard.username is not None
        or parsed_dashboard.password is not None
        or parsed_dashboard.fragment
        or not SHA_RE.fullmatch(dashboard_source_sha)
    ):
        raise PhysicalStagingError(
            "ephemeral staging identity or dashboard URL is invalid"
        )
    dashboard_doc = _require_mapping(doc.get("dashboard"), field="dashboard")
    dashboard = DashboardHost(
        ssh_host=_validate_nonproduction_host(
            dashboard_doc.get("ssh_host"), field="dashboard.ssh_host"
        ),
        ssh_key=Path(
            _validate_path(
                dashboard_doc.get("ssh_key"),
                field="dashboard.ssh_key",
                must_exist=True,
            )
        ),
        repo_root=_validate_path(
            dashboard_doc.get("repo_root"), field="dashboard.repo_root"
        ),
        unit_name=_require_text(
            dashboard_doc.get("unit_name"), field="dashboard.unit_name"
        ),
        source_sha=_require_text(
            dashboard_doc.get("source_sha"), field="dashboard.source_sha"
        ).lower(),
    )
    if (
        not SAFE_NAME_RE.fullmatch(dashboard.unit_name)
        or not SHA_RE.fullmatch(dashboard.source_sha)
        or dashboard.source_sha != dashboard_source_sha
        or _address(dashboard.ssh_host) in {
            _address(gateway.ssh_host),
            _address(primary.ssh_host),
            *(_address(item.ssh_host) for item in auditors),
        }
    ):
        raise PhysicalStagingError("dashboard staging identity is invalid")
    return PhysicalStagingConfig(
        network=network,
        netuid=netuid,
        chain_endpoint=chain_endpoint,
        network_genesis_hash=network_genesis_hash,
        gateway_public_url=gateway_public_url,
        gateway=gateway,
        primary_validator=primary,
        auditors=tuple(auditors),
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
        required_consecutive_epochs=required_consecutive_epochs,
        rebenchmark_timeout_seconds=rebenchmark_timeout_seconds,
        dashboard_report_url=dashboard_report_url,
        dashboard=dashboard,
        ephemeral_stack_id=ephemeral_stack_id,
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


def _restart_exact_release(
    config: PhysicalStagingConfig,
    commit: str,
    *,
    release_prefix: str,
) -> None:
    if release_prefix not in {
        "attested-v2/candidates",
        "attested-v2/releases",
    }:
        raise PhysicalStagingError("physical staging release prefix is invalid")
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
            "LEADPOET_RELEASE_PREFIX": release_prefix,
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
            "previous_commit=\"$(git -C \"$repo\" rev-parse HEAD)\"",
            "git -C \"$repo\" fetch origin main",
            f"test \"$(git -C \"$repo\" rev-parse origin/main)\" = '{commit}'",
            "git -C \"$repo\" checkout main",
            "git -C \"$repo\" merge --ff-only origin/main",
            f"test \"$(git -C \"$repo\" rev-parse HEAD)\" = '{commit}'",
            "git -C \"$repo\" diff --quiet HEAD --",
            (
                "if [ \"$previous_commit\" != '"
                + commit
                + "' ] && git -C \"$repo\" diff --name-only "
                "\"$previous_commit\" '"
                + commit
                + "' -- requirements.txt | grep -qx requirements.txt; then "
                "python3 -m pip install -r \"$repo/requirements.txt\" --quiet; fi"
            ),
            f"dropin='/etc/systemd/system/{auditor.unit_name}.d'",
            "sudo install -d -m 0755 \"$dropin\"",
            (
                "printf '%s\\n' '[Service]' "
                f"'Environment=LEADPOET_PARITY_AUDITOR_SECRET_ID={auditor.secret_id}' "
                f"'Environment=LEADPOET_PARITY_CANDIDATE_SHA={commit}' "
                "| sudo tee \"$dropin/10-run-identity.conf\" >/dev/null"
            ),
            "sudo systemctl daemon-reload",
            f"sudo systemctl restart '{auditor.unit_name}'",
            f"sudo systemctl is-active --quiet '{auditor.unit_name}'",
        ]
    )
    _ssh(auditor.ssh_host, auditor.ssh_key, command, timeout=180)


def _dashboard_release_evidence(config: PhysicalStagingConfig) -> dict[str, Any]:
    dashboard = config.dashboard
    output = _ssh(
        dashboard.ssh_host,
        dashboard.ssh_key,
        "\n".join(
            [
                "set -Eeuo pipefail",
                f"repo='{dashboard.repo_root}'",
                f"unit='{dashboard.unit_name}'",
                "test -z \"$(git -C \"$repo\" status --porcelain --untracked-files=no)\"",
                "git -C \"$repo\" fetch origin main",
                "git -C \"$repo\" rev-parse HEAD",
                "git -C \"$repo\" rev-parse origin/main",
                "sudo systemctl is-active \"$unit\"",
            ]
        ),
        timeout=60,
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines != [dashboard.source_sha, dashboard.source_sha, "active"]:
        raise PhysicalStagingError("staging dashboard exact source is not active")
    return {
        "source_sha": dashboard.source_sha,
        "unit": dashboard.unit_name,
        "status": "active",
    }


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
    epoch_id: int,
    bundle_hash: str,
    weights_hash: str,
    finalized_block: int,
) -> dict[str, Any] | None:
    journal_path = (
        Path(config.primary_validator.repo_root)
        / "validator_weights"
        / "authoritative_weight_publication_v2.json"
    )
    raw = _ssh(
        config.primary_validator.ssh_host,
        config.primary_validator.ssh_key,
        (
            f"if test -r '{journal_path}'; then cat '{journal_path}'; "
            "else printf '{}'; fi"
        ),
        timeout=60,
    )
    try:
        journal = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(journal, Mapping):
        return None
    body = {key: value for key, value in journal.items() if key != "journal_hash"}
    publication = journal.get("publication")
    signatures = journal.get("extrinsic_signature_results")
    compact = journal.get("compact_submission")
    if not isinstance(compact, Mapping):
        compact = journal.get("published_bundle")
    weight_result = compact.get("weight_result") if isinstance(compact, Mapping) else None
    observed_bundle_hash = (
        compact.get("bundle_hash") if isinstance(compact, Mapping) else None
    )
    if observed_bundle_hash is None and isinstance(weight_result, Mapping):
        observed_bundle_hash = weight_result.get("bundle_hash")
    observed_weights_hash = (
        weight_result.get("weights_hash")
        if isinstance(weight_result, Mapping)
        else None
    )
    observed_epoch = (
        weight_result.get("epoch_id")
        if isinstance(weight_result, Mapping)
        else None
    )
    try:
        identity_matches = (
            int(observed_epoch if observed_epoch is not None else -1) == epoch_id
            and int(publication.get("epoch_id", -1)) == epoch_id
        )
    except (TypeError, ValueError, AttributeError):
        identity_matches = False
    if (
        journal.get("state") != "signed"
        or journal.get("journal_hash") != sha256_json(body)
        or not isinstance(publication, Mapping)
        or publication.get("success") is not True
        or not identity_matches
        or observed_bundle_hash != bundle_hash
        or observed_weights_hash != weights_hash
        or publication.get("weights_hash") != weights_hash
        or not isinstance(signatures, list)
        or not signatures
    ):
        return None
    extrinsic_hashes = [
        str(item.get("extrinsic_hash") or "")
        for item in signatures
        if isinstance(item, Mapping)
        and item.get("validator_hotkey")
        == config.primary_validator.expected_hotkey
    ]
    if len(extrinsic_hashes) != len(signatures) or any(
        not re.fullmatch(r"0x[0-9a-f]{64}", item) for item in extrinsic_hashes
    ):
        return None
    container = config.primary_validator.container_name or "leadpoet-validator-main"
    output = _ssh(
        config.primary_validator.ssh_host,
        config.primary_validator.ssh_key,
        f"docker logs --since '{since_epoch_seconds}' '{container}' 2>&1",
        timeout=60,
    )
    if (
        "Authoritative V2 gateway bundle persisted:" not in output
        or "Authoritative V2 finalized chain state persisted:" not in output
    ):
        return None
    return {
        "epoch_id": epoch_id,
        "bundle_hash": bundle_hash,
        "weights_hash": weights_hash,
        "weight_submission_event_hash": publication.get(
            "weight_submission_event_hash"
        ),
        "extrinsic_hashes": extrinsic_hashes,
        "finalized_block": finalized_block,
        "journal_hash": journal.get("journal_hash"),
    }


def _configure_staging_controls(
    config: PhysicalStagingConfig,
) -> dict[str, Any]:
    command = "\n".join(
        [
            "set -Eeuo pipefail",
            "admin=/home/ec2-user/bin/research-lab-admin",
            "test -x \"$admin\"",
            (
                "\"$admin\" pause-autoresearch "
                "--reason production_parity_full_lane "
                "--actor-ref operator:production-parity-full"
            ),
            (
                "\"$admin\" resume-scoring "
                "--reason production_parity_full_lane "
                "--actor-ref operator:production-parity-full"
            ),
            "\"$admin\" status",
        ]
    )
    output = _ssh(
        config.gateway.ssh_host,
        config.gateway.ssh_key,
        command,
        timeout=180,
    )
    documents: list[Mapping[str, Any]] = []
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            documents.append(value)
    if not documents:
        raise PhysicalStagingError("staging maintenance status is unavailable")
    status = documents[-1]
    autoresearch = status.get("state")
    scoring = status.get("scoring_state")
    if (
        status.get("ok") is not True
        or not isinstance(autoresearch, Mapping)
        or autoresearch.get("paused") is not True
        or not isinstance(scoring, Mapping)
        or scoring.get("paused") is not False
    ):
        raise PhysicalStagingError("staging maintenance controls differ")
    return {
        "autoresearch_paused": True,
        "scoring_paused": False,
        "autoresearch_reason": str(autoresearch.get("reason") or ""),
        "scoring_reason": str(scoring.get("reason") or ""),
    }


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


def _authority_chain_expectation(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    submission = authority.get("compact_submission")
    weight_result = (
        submission.get("weight_result")
        if isinstance(submission, Mapping)
        else None
    )
    finalization_section = authority.get("finalization")
    finalization_submission = (
        finalization_section.get("compact_submission")
        if isinstance(finalization_section, Mapping)
        else None
    )
    finalization = (
        finalization_submission.get("finalization")
        if isinstance(finalization_submission, Mapping)
        else None
    )
    uids = (
        weight_result.get("sparse_uids")
        if isinstance(weight_result, Mapping)
        else None
    )
    weights = (
        weight_result.get("sparse_weights_u16")
        if isinstance(weight_result, Mapping)
        else None
    )
    if uids is None and isinstance(weight_result, Mapping):
        uids = weight_result.get("uids")
    if weights is None and isinstance(weight_result, Mapping):
        weights = weight_result.get("weights_u16")
    try:
        pairs = sorted(
            (int(uid), int(weight)) for uid, weight in zip(uids, weights)
        )
        finalized_block = int(finalization["finalized_block"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PhysicalStagingError(
            "canonical authority chain expectation is incomplete"
        ) from exc
    if (
        not pairs
        or len(pairs) != len(uids)
        or len(pairs) != len(weights)
        or len({uid for uid, _weight in pairs}) != len(pairs)
        or any(uid < 0 or not 0 <= weight <= 65535 for uid, weight in pairs)
        or finalized_block < 0
    ):
        raise PhysicalStagingError(
            "canonical authority chain expectation is invalid"
        )
    return {"weights": pairs, "finalized_block": finalized_block}


def _matching_auditor_chain_confirmation(
    events: Sequence[Mapping[str, Any]], *, netuid: int, epoch_id: int
) -> dict[str, Any] | None:
    matches = []
    for event in events:
        try:
            if (
                event.get("event") == "submission_chain_confirmation"
                and int(event.get("netuid", -1)) == netuid
                and int(event.get("epoch", -1)) == epoch_id
                and event.get("confirmation_stage")
                == "timelocked_commit_finalized"
                and int(event.get("observed_last_update", -1)) >= 0
                and re.fullmatch(
                    r"0x[0-9a-f]{64}",
                    str(event.get("finalized_block_hash") or "").lower(),
                )
            ):
                matches.append(
                    {
                        "observed_last_update": int(
                            event["observed_last_update"]
                        ),
                        "finalized_block_hash": str(
                            event["finalized_block_hash"]
                        ).lower(),
                    }
                )
        except (TypeError, ValueError):
            continue
    return matches[-1] if matches else None


def _read_independent_chain_state(
    config: PhysicalStagingConfig,
) -> dict[str, Any]:
    reader = config.auditors[0]
    hotkeys = [
        config.primary_validator.expected_hotkey,
        *(item.expected_hotkey for item in config.auditors),
    ]
    command = [
        "python3",
        "scripts/read_production_parity_chain_state.py",
        "--endpoint",
        config.chain_endpoint,
        "--netuid",
        str(config.netuid),
        "--expected-genesis-hash",
        config.network_genesis_hash,
    ]
    for hotkey in hotkeys:
        command.extend(("--hotkey", hotkey))
    raw = _ssh(
        reader.ssh_host,
        reader.ssh_key,
        "cd "
        + shlex.quote(reader.repo_root)
        + " && "
        + " ".join(shlex.quote(item) for item in command),
        timeout=120,
    )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PhysicalStagingError(
            "independent finalized-chain readback is invalid"
        ) from exc
    body = {
        key: item for key, item in value.items() if key != "readback_hash"
    } if isinstance(value, Mapping) else {}
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version")
        != "leadpoet.production_parity_chain_readback.v1"
        or value.get("network") != "test"
        or value.get("network_genesis_hash") != config.network_genesis_hash
        or int(value.get("netuid", -1)) != config.netuid
        or value.get("readback_hash") != sha256_json(body)
    ):
        raise PhysicalStagingError(
            "independent finalized-chain readback identity differs"
        )
    return dict(value)


def _verify_independent_chain_acceptance(
    config: PhysicalStagingConfig,
    accepted: Sequence[Mapping[str, Any]],
    readback: Mapping[str, Any],
) -> dict[str, Any]:
    validators = readback.get("validators")
    if not isinstance(validators, list):
        raise PhysicalStagingError(
            "independent finalized-chain validator set is missing"
        )
    by_hotkey = {
        str(item.get("hotkey") or ""): item
        for item in validators
        if isinstance(item, Mapping)
    }
    expected_hotkeys = {
        config.primary_validator.expected_hotkey,
        *(item.expected_hotkey for item in config.auditors),
    }
    expected_vectors = [
        [(int(uid), int(weight)) for uid, weight in item["weights"]]
        for item in accepted
    ]
    if set(by_hotkey) != expected_hotkeys or not expected_vectors:
        raise PhysicalStagingError(
            "independent finalized-chain validator identity differs"
        )
    minima = {
        config.primary_validator.expected_hotkey: max(
            int(item["primary_finalized"]["finalized_block"])
            for item in accepted
        )
    }
    for auditor in config.auditors:
        minima[auditor.expected_hotkey] = max(
            int(item["auditors"][auditor.expected_hotkey]["observed_last_update"])
            for item in accepted
        )
    matches: dict[str, int] = {}
    for hotkey, minimum_last_update in minima.items():
        observed = by_hotkey[hotkey]
        try:
            last_update = int(observed["last_update"])
            actual = [
                (int(uid), int(weight))
                for uid, weight in observed["weights"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise PhysicalStagingError(
                "independent finalized-chain validator state is invalid"
            ) from exc
        if last_update < minimum_last_update:
            raise PhysicalStagingError(
                "independent finalized LastUpdate has not reached submission"
            )
        matched_epoch = next(
            (
                int(item["epoch_id"])
                for item, expected in zip(accepted, expected_vectors)
                if weights_within_tolerance(expected, actual)
            ),
            None,
        )
        if matched_epoch is None:
            raise PhysicalStagingError(
                "independent finalized weight vector differs from canonical authority"
            )
        matches[hotkey] = matched_epoch
    return {
        "readback_hash": readback["readback_hash"],
        "finalized_block": int(readback["finalized_block"]),
        "finalized_block_hash": str(readback["finalized_block_hash"]),
        "last_update_minima": minima,
        "visible_vector_epoch_by_hotkey": matches,
    }


def _report_document(value: Mapping[str, Any]) -> Mapping[str, Any]:
    report = value.get("report_doc")
    return report if isinstance(report, Mapping) else value


def _rebenchmark_identity(
    value: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    report = _report_document(value)
    expected_counts = {
        "public": int(policy.get("public_total_icps") or 0),
        "private": int(policy.get("private_total_icps") or 0),
        "conditional": int(policy.get("conditional_total_icps") or 0),
    }
    expected_total = sum(expected_counts.values())
    public_weak = int(policy.get("public_weak_total") or 0)
    private_weak = int(policy.get("private_weak_total") or 0)
    if (
        any(count <= 0 for count in expected_counts.values())
        or public_weak < 0
        or public_weak > expected_counts["public"]
        or private_weak < 0
        or private_weak > expected_counts["private"]
    ):
        raise PhysicalStagingError("candidate ICP policy is invalid")
    split = report.get("visibility_split")
    if not isinstance(split, Mapping):
        raise PhysicalStagingError("published rebenchmark assignment is missing")
    observed_counts = {
        "public": int(split.get("public_count") or 0),
        "private": int(split.get("private_count") or 0),
        "conditional": int(split.get("conditional_count") or 0),
    }
    try:
        aggregate_score = float(
            value.get("aggregate_score")
            if value.get("aggregate_score") is not None
            else report.get("aggregate_score")
        )
    except (TypeError, ValueError) as exc:
        raise PhysicalStagingError("published rebenchmark score is invalid") from exc
    public_strength = split.get("public_strength_counts")
    private_strength = split.get("private_strength_counts")
    expected_public_strength = {
        "strong": expected_counts["public"]
        - public_weak,
        "weak": public_weak,
    }
    expected_private_strength = {
        "strong": expected_counts["private"]
        - private_weak,
        "weak": private_weak,
    }
    expected_public_strength = {
        key: value for key, value in expected_public_strength.items() if value > 0
    }
    expected_private_strength = {
        key: value for key, value in expected_private_strength.items() if value > 0
    }
    report_hash = str(report.get("report_public_hash") or "")
    report_without_hash = {
        key: item for key, item in report.items() if key != "report_public_hash"
    }
    if (
        value.get("current_report_status") != "published"
        or value.get("benchmark_quality") != "passed"
        or report.get("report_type") != "research_lab_public_daily_benchmark"
        or int(report.get("item_count") or 0) != expected_total
        or observed_counts != expected_counts
        or dict(public_strength or {}) != expected_public_strength
        or dict(private_strength or {}) != expected_private_strength
        or str(split.get("split_policy") or "")
        != str(policy.get("selection_policy") or "")
        or str(split.get("rolling_window_hash") or "")
        != str(report.get("rolling_window_hash") or "")
        or not 0.0 <= aggregate_score <= 100.0
        or not HASH_RE.fullmatch(report_hash)
        or report_hash != sha256_json(report_without_hash)
    ):
        raise PhysicalStagingError(
            "published rebenchmark score or assignment differs from candidate policy"
        )
    identity = {
        "report_id": str(value.get("report_id") or ""),
        "benchmark_bundle_id": str(value.get("benchmark_bundle_id") or ""),
        "benchmark_date": str(value.get("benchmark_date") or ""),
        "rolling_window_hash": str(value.get("rolling_window_hash") or ""),
        "private_model_artifact_hash": str(
            value.get("private_model_artifact_hash") or ""
        ),
        "private_model_manifest_hash": str(
            value.get("private_model_manifest_hash") or ""
        ),
        "aggregate_score": aggregate_score,
        "item_count": expected_total,
        "report_public_hash": report_hash,
        "category_counts": observed_counts,
        "public_strength_counts": dict(public_strength),
        "private_strength_counts": dict(private_strength),
    }
    if (
        not identity["report_id"]
        or not identity["benchmark_bundle_id"]
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", identity["benchmark_date"])
        or not HASH_RE.fullmatch(identity["rolling_window_hash"])
        or not HASH_RE.fullmatch(identity["private_model_artifact_hash"])
        or not HASH_RE.fullmatch(identity["private_model_manifest_hash"])
    ):
        raise PhysicalStagingError("published rebenchmark identity is incomplete")
    return identity


def _contains_dashboard_identity(value: Any, identity: Mapping[str, Any]) -> bool:
    """Match the real subnet-dashboard public API, not an internal DB shape."""

    if not isinstance(value, Mapping) or value.get("success") is not True:
        return False
    data = value.get("data")
    benchmark = data.get("benchmark") if isinstance(data, Mapping) else None
    if not isinstance(benchmark, Mapping):
        return False
    try:
        score_matches = float(benchmark.get("aggregateScore")) == float(
            identity["aggregate_score"]
        )
        count_matches = int(benchmark.get("itemCount")) == int(
            identity["item_count"]
        )
    except (TypeError, ValueError):
        return False
    return (
        str(benchmark.get("reportId") or "") == identity["report_id"]
        and str(benchmark.get("benchmarkDate") or "")
        == identity["benchmark_date"]
        and str(benchmark.get("rollingWindowHash") or "")
        == identity["rolling_window_hash"]
        and score_matches
        and count_matches
    )


def _candidate_rebenchmark_readiness(
    config: PhysicalStagingConfig,
    *,
    candidate_sha: str,
) -> dict[str, Any]:
    gateway = config.gateway
    command = (
        f"cd '{gateway.repo_root}' && "
        f"'{gateway.python_bin}' scripts/check_production_parity_rebenchmark.py "
        f"--candidate-sha '{candidate_sha}' --secret-id '{gateway.secret_id}'"
    )
    try:
        value = json.loads(
            _ssh(
                gateway.ssh_host,
                gateway.ssh_key,
                command,
                timeout=120,
            )
        )
    except (ValueError, PhysicalStagingError) as exc:
        raise PhysicalStagingError(
            "candidate rebenchmark readiness probe failed"
        ) from exc
    if not isinstance(value, dict):
        raise PhysicalStagingError(
            "candidate rebenchmark readiness evidence is invalid"
        )
    return value


def _dashboard_json(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=35) as response:
            status = int(response.status)
            value = json.load(response)
    except Exception as exc:
        raise PhysicalStagingError("staging dashboard readback failed") from exc
    if status != 200 or not isinstance(value, dict):
        raise PhysicalStagingError("staging dashboard readback is invalid")
    return value


def _wait_for_rebenchmark_acceptance(
    config: PhysicalStagingConfig,
    *,
    candidate_sha: str,
    policy: Mapping[str, Any],
    expected_benchmark_date: str,
    started_at: int,
) -> dict[str, Any]:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", expected_benchmark_date):
        raise PhysicalStagingError("expected rebenchmark date is invalid")
    deadline = time.monotonic() + config.rebenchmark_timeout_seconds
    last_reason = "no published rebenchmark observed"
    while time.monotonic() < deadline:
        report = _gateway_json(
            config,
            "/research-lab/benchmarks/public/latest",
            allow_not_found=True,
        )
        if report is None:
            last_reason = "public rebenchmark report pending"
            time.sleep(config.poll_seconds)
            continue
        created_at = str(report.get("created_at") or "")
        try:
            created = datetime.fromisoformat(created_at).timestamp()
        except ValueError:
            last_reason = "public rebenchmark report creation time is invalid"
            time.sleep(config.poll_seconds)
            continue
        if created < started_at:
            last_reason = "public rebenchmark report predates this staging run"
            time.sleep(config.poll_seconds)
            continue
        try:
            identity = _rebenchmark_identity(report, policy=policy)
            readiness = _candidate_rebenchmark_readiness(
                config,
                candidate_sha=candidate_sha,
            )
            dashboard = _dashboard_json(config.dashboard_report_url)
        except PhysicalStagingError as exc:
            last_reason = str(exc)
            time.sleep(config.poll_seconds)
            continue
        if identity["benchmark_date"] != expected_benchmark_date:
            last_reason = "published rebenchmark date differs from the snapshot frontier"
            time.sleep(config.poll_seconds)
            continue
        if (
            readiness.get("available") is not True
            or readiness.get("reason") != "daily_baseline_published"
            or readiness.get("benchmark_date") != identity["benchmark_date"]
            or readiness.get("report_id") != identity["report_id"]
            or readiness.get("benchmark_bundle_id")
            != identity["benchmark_bundle_id"]
            or readiness.get("rolling_window_hash")
            != identity["rolling_window_hash"]
        ):
            last_reason = (
                "candidate daily-baseline readiness has not proven the complete "
                "private ICP assignment"
            )
            time.sleep(config.poll_seconds)
            continue
        commitments = readiness.get("completion_commitments")
        expected_counts = {
            "public": int(policy.get("public_total_icps") or 0),
            "private": int(policy.get("private_total_icps") or 0),
            "conditional": int(policy.get("conditional_total_icps") or 0),
        }
        expected_strength_counts = {
            "public": {
                key: value
                for key, value in {
                    "weak": int(policy.get("public_weak_total") or 0),
                    "strong": int(policy.get("public_strong_total") or 0),
                }.items()
                if value > 0
            },
            "private": {
                key: value
                for key, value in {
                    "weak": int(policy.get("private_weak_total") or 0),
                    "strong": int(policy.get("private_strong_total") or 0),
                }.items()
                if value > 0
            },
            "conditional": {"center": expected_counts["conditional"]},
        }
        if (
            not isinstance(commitments, Mapping)
            or int(commitments.get("all_icp_count") or 0)
            != sum(expected_counts.values())
            or dict(commitments.get("category_counts") or {}) != expected_counts
            or dict(commitments.get("category_strength_counts") or {})
            != expected_strength_counts
            or str(commitments.get("conditional_policy_hash") or "")
            != str(policy.get("policy_hash") or "")
            or not HASH_RE.fullmatch(
                str(commitments.get("per_icp_summaries_hash") or "")
            )
            or not HASH_RE.fullmatch(
                str(commitments.get("category_assignment_hash") or "")
            )
        ):
            last_reason = (
                "candidate readiness has not proven every configured ICP and "
                "the exact tail/center assignment"
            )
            time.sleep(config.poll_seconds)
            continue
        if not _contains_dashboard_identity(dashboard, identity):
            last_reason = "dashboard has not read back the exact published score"
            time.sleep(config.poll_seconds)
            continue
        return {
            **identity,
            "per_icp_summaries_hash": commitments["per_icp_summaries_hash"],
            "category_assignment_hash": commitments[
                "category_assignment_hash"
            ],
            "conditional_policy_hash": commitments[
                "conditional_policy_hash"
            ],
            "minimum_icp_score": commitments["minimum_icp_score"],
            "maximum_icp_score": commitments["maximum_icp_score"],
            "candidate_readiness": True,
            "dashboard_readback": True,
        }
    raise PhysicalStagingError(
        f"physical rebenchmark acceptance timed out: {last_reason}"
    )


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
) -> list[dict[str, Any]]:
    epoch_state = _gateway_json(config, "/epoch/current")
    assert epoch_state is not None
    first_epoch = int(epoch_state["current_epoch_id"])
    deadline = time.monotonic() + config.timeout_seconds
    last_reason = "no authority observed"
    accepted: dict[int, dict[str, Any]] = {}
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
                chain_expectation = _authority_chain_expectation(authority)
            except PhysicalStagingError:
                raise
            if observed_epoch != epoch_id:
                raise PhysicalStagingError("canonical authority epoch differs")
            primary = _primary_finalized(
                config,
                since_epoch_seconds=started_at,
                epoch_id=epoch_id,
                bundle_hash=bundle_hash,
                weights_hash=weights_hash,
                finalized_block=chain_expectation["finalized_block"],
            )
            if primary is None:
                last_reason = "primary finalization log pending"
                continue
            auditor_results: dict[str, dict[str, Any]] = {}
            for auditor in config.auditors:
                events = _auditor_events(
                    auditor, since_epoch_seconds=started_at
                )
                matched_success = _matching_auditor_startup(
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
                confirmation = _matching_auditor_chain_confirmation(
                    events,
                    netuid=config.netuid,
                    epoch_id=epoch_id,
                )
                if matched_success and confirmation is not None:
                    auditor_results[auditor.expected_hotkey] = {
                        "success": True,
                        **confirmation,
                    }
            if len(auditor_results) == len(config.auditors):
                accepted[epoch_id] = {
                    "epoch_id": epoch_id,
                    "bundle_hash": bundle_hash,
                    "weights_hash": weights_hash,
                    "weights": chain_expectation["weights"],
                    "authority_stage": "finalized",
                    "primary_finalized": primary,
                    "auditors": auditor_results,
                }
                ordered = [accepted[key] for key in sorted(accepted)]
                for start in range(len(ordered)):
                    candidate = ordered[start : start + config.required_consecutive_epochs]
                    if (
                        len(candidate) == config.required_consecutive_epochs
                        and all(
                            candidate[index]["epoch_id"]
                            == candidate[0]["epoch_id"] + index
                            for index in range(len(candidate))
                        )
                    ):
                        try:
                            chain_readback = _verify_independent_chain_acceptance(
                                config,
                                candidate,
                                _read_independent_chain_state(config),
                            )
                        except PhysicalStagingError as exc:
                            last_reason = str(exc)
                            continue
                        return [
                            {**item, "chain_readback": chain_readback}
                            for item in candidate
                        ]
            last_reason = "one or more auditor finalizations are pending"
        time.sleep(config.poll_seconds)
    raise PhysicalStagingError(
        f"physical V2 acceptance timed out: {last_reason}"
    )


def run(
    config: PhysicalStagingConfig,
    *,
    commit: str,
    contract: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    production_db_host: str,
    release_prefix: str = "attested-v2/candidates",
) -> dict[str, Any]:
    if not SHA_RE.fullmatch(commit):
        raise PhysicalStagingError("candidate commit must be a full lowercase SHA")
    normalized_contract = verify_contract_checkout(ROOT, contract)
    normalized_snapshot = validate_snapshot_manifest(snapshot)
    if normalized_snapshot["source_host_hash"] != production_database_host_hash(
        production_db_host
    ):
        raise PhysicalStagingError(
            "physical staging snapshot source differs from production"
        )
    if normalized_contract["candidate_sha"] != commit:
        raise PhysicalStagingError("physical staging contract commit differs")
    try:
        oracle = validate_historical_oracle(
            json.loads(
                (
                    ROOT
                    / "tests/restart_rehearsal/fixtures/august_9_known_good_v2.json"
                ).read_text(encoding="utf-8")
            )
        )
    except (OSError, ValueError, ProductionParityError) as exc:
        raise PhysicalStagingError(
            "historical production behavior oracle is unavailable"
        ) from exc
    if (
        sha256_json(oracle) != normalized_contract["historical_oracle_hash"]
        or not set(required_oracle_stage_ids(oracle, lane="full")).issubset(
            FULL_CRITICAL_STAGES
        )
    ):
        raise PhysicalStagingError(
            "full lane does not cover the historical production behavior oracle"
        )
    ledger = StageLedger(
        lane="full",
        candidate_sha=commit,
        contract_hash=normalized_contract["contract_hash"],
        snapshot_hash=normalized_snapshot["manifest_hash"],
        critical_stage_ids=FULL_CRITICAL_STAGES,
    )
    fetch = _run(["git", "fetch", "origin", "main"], timeout=120)
    remote = _run(["git", "rev-parse", "origin/main"], timeout=30)
    source_ancestry = _run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            normalized_snapshot["source_sha"],
            normalized_snapshot["capture_sha"],
        ],
        timeout=30,
    )
    capture_ancestry = _run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            normalized_snapshot["capture_sha"],
            commit,
        ],
        timeout=30,
    )
    if (
        fetch.returncode != 0
        or remote.returncode != 0
        or remote.stdout.strip() != commit
        or source_ancestry.returncode != 0
        or capture_ancestry.returncode != 0
    ):
        raise PhysicalStagingError(
            "candidate tip or production snapshot ancestry differs"
        )
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_database = normalized_snapshot["database"]
    latest_completed = snapshot_database["latest_completed_benchmark_date"]
    if (
        snapshot_database["target_rebenchmark_date"] != today
        or latest_completed == today
    ):
        raise PhysicalStagingError(
            "snapshot is not eligible to execute today's fresh rebenchmark"
        )
    ledger.record(
        "candidate-and-snapshot-identity",
        status="passed",
        duration_seconds=0,
        evidence={
            "candidate_sha": commit,
            "snapshot_source_sha": normalized_snapshot["source_sha"],
            "snapshot_capture_sha": normalized_snapshot["capture_sha"],
            "ephemeral_stack_id": config.ephemeral_stack_id,
            "snapshot_capture_utc_date": snapshot_database["capture_utc_date"],
            "target_rebenchmark_date": snapshot_database[
                "target_rebenchmark_date"
            ],
            "latest_completed_benchmark_date": latest_completed,
            "current_day_rebenchmark_run_count": snapshot_database[
                "current_day_rebenchmark_run_count"
            ],
            "current_day_benchmark_bundle_count": snapshot_database[
                "current_day_benchmark_bundle_count"
            ],
        },
    )

    restart_started = time.monotonic()
    _restart_exact_release(config, commit, release_prefix=release_prefix)
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
    dashboard_evidence = _dashboard_release_evidence(config)
    ledger.record(
        "exact-paired-restart",
        status="passed",
        duration_seconds=time.monotonic() - restart_started,
        evidence={
            "gateway_commit": str(build.get("git_commit") or "").lower(),
            "authority_commit": str(health.get("commit_sha") or "").lower(),
            "auditor_count": len(config.auditors),
            "dashboard": dashboard_evidence,
        },
    )

    controls_started = time.monotonic()
    controls = _configure_staging_controls(config)
    validation_started_at = int(time.time())
    ledger.record(
        "staging-control-boundary",
        status="passed",
        duration_seconds=time.monotonic() - controls_started,
        evidence=controls,
    )

    policy = normalized_contract["policy_commitments"]["conditional_icp"]
    parallel_started = {
        "rebenchmark": time.monotonic(),
        "weights": time.monotonic(),
    }
    parallel_results: dict[str, Any] = {}
    parallel_failures: dict[str, str] = {}
    parallel_durations: dict[str, float] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(
                _wait_for_rebenchmark_acceptance,
                config,
                candidate_sha=commit,
                policy=policy,
                expected_benchmark_date=snapshot_database[
                    "target_rebenchmark_date"
                ],
                started_at=validation_started_at,
            ): "rebenchmark",
            pool.submit(
                _wait_for_canonical_acceptance,
                config,
                commit=commit,
                started_at=validation_started_at,
            ): "weights",
        }
        for future in concurrent.futures.as_completed(futures):
            lane = futures[future]
            parallel_durations[lane] = (
                time.monotonic() - parallel_started[lane]
            )
            try:
                parallel_results[lane] = future.result()
            except Exception as exc:  # Preserve the independent lane's failure.
                parallel_failures[lane] = f"{type(exc).__name__}: {exc}"

    rebenchmark = parallel_results.get("rebenchmark")
    if rebenchmark is not None:
        ledger.record(
            "full-rebenchmark-and-assignment",
            status="passed",
            duration_seconds=parallel_durations["rebenchmark"],
            evidence={
                key: value
                for key, value in rebenchmark.items()
                if key != "dashboard_readback"
            },
        )
        ledger.record(
            "dashboard-score-readback",
            status="passed",
            duration_seconds=parallel_durations["rebenchmark"],
            evidence={
                "report_id": rebenchmark["report_id"],
                "report_public_hash": rebenchmark["report_public_hash"],
                "aggregate_score": rebenchmark["aggregate_score"],
            },
        )
    else:
        reason = parallel_failures.get(
            "rebenchmark", "rebenchmark acceptance returned no result"
        )
        ledger.record(
            "full-rebenchmark-and-assignment",
            status="failed",
            duration_seconds=parallel_durations.get("rebenchmark", 0),
            reason=reason,
        )
        ledger.record(
            "dashboard-score-readback",
            status="unexercised",
            duration_seconds=0,
            reason=f"blocked by full-rebenchmark-and-assignment: {reason}",
        )

    weights = parallel_results.get("weights")
    if weights is not None:
        ledger.record(
            "canonical-weight-bundles",
            status="passed",
            duration_seconds=parallel_durations["weights"],
            evidence={
                "epochs": [item["epoch_id"] for item in weights],
                "bundle_hashes": [item["bundle_hash"] for item in weights],
                "weights_hashes": [item["weights_hash"] for item in weights],
            },
        )
        ledger.record(
            "primary-finalization",
            status="passed",
            duration_seconds=parallel_durations["weights"],
            evidence={
                "epochs": [item["epoch_id"] for item in weights],
                "journals": [item["primary_finalized"] for item in weights],
            },
        )
        ledger.record(
            "audit-finalization",
            status="passed",
            duration_seconds=parallel_durations["weights"],
            evidence={
                "auditors": {
                    item.expected_hotkey: True for item in config.auditors
                },
                "epochs": [item["epoch_id"] for item in weights],
            },
        )
        ledger.record(
            "consecutive-epoch-readback",
            status="passed",
            duration_seconds=parallel_durations["weights"],
            evidence={
                "required": config.required_consecutive_epochs,
                "observed": len(weights),
                "first_epoch": weights[0]["epoch_id"],
                "last_epoch": weights[-1]["epoch_id"],
                "independent_chain": weights[-1]["chain_readback"],
            },
        )
    else:
        reason = parallel_failures.get(
            "weights", "canonical weight acceptance returned no result"
        )
        ledger.record(
            "canonical-weight-bundles",
            status="failed",
            duration_seconds=parallel_durations.get("weights", 0),
            reason=reason,
        )
        for stage_id in (
            "primary-finalization",
            "audit-finalization",
            "consecutive-epoch-readback",
        ):
            ledger.record(
                stage_id,
                status="unexercised",
                duration_seconds=0,
                reason=f"blocked by canonical-weight-bundles: {reason}",
            )
    final_fetch = _run(["git", "fetch", "origin", "main"], timeout=120)
    final_remote = _run(["git", "rev-parse", "origin/main"], timeout=30)
    final_dashboard: dict[str, Any] | None = None
    dashboard_failure = ""
    try:
        final_dashboard = _dashboard_release_evidence(config)
    except Exception as exc:
        dashboard_failure = f"{type(exc).__name__}: {exc}"
    if (
        final_fetch.returncode != 0
        or final_remote.returncode != 0
        or final_remote.stdout.strip() != commit
        or final_dashboard is None
    ):
        ledger.record(
            "candidate-not-superseded",
            status="failed",
            duration_seconds=0,
            reason=(
                "candidate or dashboard was superseded during physical staging"
                + (f": {dashboard_failure}" if dashboard_failure else "")
            ),
            evidence={
                "origin_main": final_remote.stdout.strip(),
                "dashboard": dict(final_dashboard or {}),
            },
        )
    else:
        ledger.record(
            "candidate-not-superseded",
            status="passed",
            duration_seconds=0,
            evidence={"origin_main": commit, "dashboard": final_dashboard},
        )
    return ledger.finalize()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--production-db-host", required=True)
    parser.add_argument(
        "--release-prefix",
        choices=("attested-v2/candidates", "attested-v2/releases"),
        required=True,
    )
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    contract: Mapping[str, Any] | None = None
    snapshot: Mapping[str, Any] | None = None
    try:
        config = load_config(args.config)
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        snapshot = json.loads(args.snapshot_manifest.read_text(encoding="utf-8"))
        ledger = run(
            config,
            commit=str(args.candidate_sha).lower(),
            contract=contract,
            snapshot=snapshot,
            production_db_host=args.production_db_host,
            release_prefix=args.release_prefix,
        )
    except (
        OSError,
        ProductionParityError,
        PhysicalStagingError,
        KeyError,
        ValueError,
        subprocess.TimeoutExpired,
    ) as exc:
        try:
            normalized_contract = validate_contract(contract or {})
            normalized_snapshot = validate_snapshot_manifest(snapshot or {})
            failed = StageLedger(
                lane="full",
                candidate_sha=str(args.candidate_sha).lower(),
                contract_hash=normalized_contract["contract_hash"],
                snapshot_hash=normalized_snapshot["manifest_hash"],
                critical_stage_ids=FULL_CRITICAL_STAGES,
            )
            reason = f"{type(exc).__name__}: {exc}"
            for stage_id in FULL_CRITICAL_STAGES:
                failed.record(
                    stage_id,
                    status="unexercised",
                    duration_seconds=0,
                    reason=reason,
                )
            failed.record(
                "controller-failure",
                status="failed",
                duration_seconds=0,
                reason=reason,
            )
            args.ledger.parent.mkdir(parents=True, exist_ok=True)
            args.ledger.write_text(
                json.dumps(failed.finalize(), sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(
        json.dumps(ledger, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(ledger, sort_keys=True))
    return 0 if ledger["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
