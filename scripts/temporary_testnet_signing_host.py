#!/usr/bin/env python3
"""Provision and remove one short-lived Nitro host for testnet signing proof."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
from pathlib import Path
import re
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# The workflow invokes this file directly, so sibling repository modules must
# remain importable without a runner-specific PYTHONPATH.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ACCOUNT_ID = "493765492819"
REGION = "us-east-1"
AMI_ID = "ami-0cae6d6fe6048ca2c"
INSTANCE_TYPE = "r7i.4xlarge"
SUBNET_ID = "subnet-025170c1eff61494d"
VPC_ID = "vpc-0c975a643bc1e0e79"
INSTANCE_PROFILE = "leadpoet-production-parity-runner"
VOLUME_GIB = 512
FUNCTION = "testnet401"
COMBINED_ALLOCATOR_CPUS = 10
COMBINED_ALLOCATOR_MEMORY_MIB = 66_560
SCHEMA_VERSION = "leadpoet.temporary_testnet_signing_host.v1"
RUN_RE = re.compile(r"^pp-[0-9]{1,20}-[0-9]{1,6}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
INSTANCE_ID_RE = re.compile(r"^i-(?:[0-9a-f]{8}|[0-9a-f]{17})$")
SECURITY_GROUP_ID_RE = re.compile(r"^sg-(?:[0-9a-f]{8}|[0-9a-f]{17})$")
VOLUME_ID_RE = re.compile(r"^vol-(?:[0-9a-f]{8}|[0-9a-f]{17})$")
NETWORK_INTERFACE_ID_RE = re.compile(r"^eni-(?:[0-9a-f]{8}|[0-9a-f]{17})$")
SSM_COMMAND_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
NAME_RE = re.compile(
    r"^leadpoet-parity-(?P<run>pp-[0-9]{1,20}-[0-9]{1,6})-"
    r"testnet401-exp-(?P<expiry>[0-9]{10})$"
)
TAG_RUN = "leadpoet:parity-run"
TAG_SHA = "leadpoet:candidate-sha"
TAG_EPHEMERAL = "leadpoet:ephemeral"
MIN_TTL_HOURS = 1
MAX_TTL_HOURS = 12
DEFAULT_TTL_HOURS = 6
ASSET_PREFIX_TEMPLATE = "production-parity/runs/{run_id}/testnet401"
SOURCE_BOOTSTRAP_SCHEMA_VERSION = (
    "leadpoet.temporary_testnet401_source_bootstrap.v1"
)
NATIVE_RECEIPT_SCHEMA_VERSION = (
    "leadpoet.temporary_testnet401_native_bootstrap_receipt.v1"
)
NATIVE_STAGES = frozenset({"preflight", "launch", "status", "cleanup"})
GATEWAY_NETWORK_RESTART_CANDIDATE_SHA = (
    "b056b2989f019ddc373a9a8fa1ba7bb0c94feb22"
)
GATEWAY_NETWORK_RESTART_SCHEMA_VERSION = (
    "leadpoet.temporary_testnet401_gateway_network_restart.v1"
)
NATIVE_STAGE_TIMEOUTS = {
    "preflight": 900,
    "launch": 7200,
    "status": 900,
    "cleanup": 900,
}
RUNTIME_ROOT = "/run/leadpoet-testnet401"
SOURCE_REPOSITORY = "/home/ec2-user/leadpoet/leadpoet"
SOURCE_VENV = "/home/ec2-user/venv311"
NATIVE_CONFIG = f"{RUNTIME_ROOT}/config.json"
RUNTIME_CALLBACK_ERROR_TYPES = (
    "AttributeError", "HTTPException", "ImportError", "KeyError", "NameError",
    "ResearchLabV2AuthorityError", "RuntimeError", "TypeError",
    "UnboundLocalError", "ValueError", "unclassified",
)
RUNTIME_CALLBACK_TOKEN_VOCAB = (
    "_fresh_testnet401_empty_origin", "activation", "allocation",
    "allocation_inputs", "allocation_sequence", "ambiguous",
    "absent", "argument", "authority", "available", "boot",
    "build_allocation_v2", "callback",
    "champion_v2_cutover_readiness", "checkpoint", "column", "connection",
    "context", "coordinator", "coordinatorallocationsourcev2",
    "coordinatorallocationfrontierbootstrapv2error", "coordinatorchainsourcev2",
    "coordinatorlegacysettlementv2error", "coordinatorrewardsourcev2error",
    "coordinatorweightsourcev2error", "database", "declared", "defined", "differs",
    "document", "down", "duplicate", "durable", "empty",
    "execute_coordinator_v2", "execution", "execution_championsettlementv2error",
    "execution_coordinatorallocationsourcev2error",
    "execution_coordinatorchainsourcev2error", "execution_executionjobv2error",
    "execution_runtimeerror", "execution_supabasesourcev2error",
    "execution_valueerror",
    "ensure_chain_realized_settlements_v1", "epoch", "executioncontextv2",
    "execution_receipt", "field", "graph",
    "exhausted", "failed", "failure", "finalized", "frontier", "function",
    "generations", "hash", "history",
    "http", "httpexception", "identity", "import", "importerror", "invalid",
    "job", "keyerror",
    "keyword", "lineage", "load_allocation_parent_graphs", "missing",
    "load", "nameerror", "network", "none", "not", "origin", "parameter",
    "parent", "parent_graphs", "parent_receipt_hashes", "payload", "policy",
    "postgrest", "provider", "query", "read", "receipt_graph",
    "receipt", "release", "manifest", "required", "response", "result", "retry", "root", "rpc",
    "research_lab_allocation_settlement_frontier_activation_v2",
    "research_lab_allocation_settlement_frontiers_v2",
    "research_lab_chain_realized_settlement_activation_v1",
    "research_lab_stateful_subnet_epoch_cutovers_v1", "runtimeerror", "schema",
    "cooling", "scope", "select", "settlement", "source", "source_state",
    "source_state_hash", "state", "status", "storage",
    "supabase", "table", "temporarytestnet401firstallocationerror", "timeout",
    "typeerror", "unexpected", "unavailable", "unboundlocalerror", "validation",
    "valueerror", "verify",
    "validate_testnet401_cutover_parent_v1",
)
RUNTIME_WEIGHT_INPUT_HTTP400_REASONS = (
    (b"Invalid V2 weight input request:", "weight_input_request_invalid"),
    (b"Invalid netuid:", "weight_input_netuid_invalid"),
    (
        b"V2 weight input request does not bind the calculation snapshot",
        "weight_input_calculation_snapshot_unbound",
    ),
    (
        b"V2 weight input request differs from snapshot at",
        "weight_input_calculation_scope_differs",
    ),
    (
        b"V2 weight input request differs from the Research Lab allocation",
        "weight_input_allocation_differs",
    ),
    (b"block drift is too large", "weight_input_block_drift_too_large"),
    (
        b"does not map to official subnet epoch",
        "weight_input_settlement_epoch_mapping_differs",
    ),
    (
        b"weight submission snapshot is not in the live official subnet epoch",
        "weight_input_snapshot_not_in_live_epoch",
    ),
    (
        b"weight submission is outside the live official subnet epoch window",
        "weight_input_outside_live_epoch_window",
    ),
    (
        b"finalized weight snapshot is outside the permitted official subnet epoch lag buffer",
        "weight_input_finalized_snapshot_outside_lag_buffer",
    ),
)
PRIVATE_ASSET_NAMES = (
    "validator.env",
    "testnet-hotkey-config.json",
    "testnet-hotkey-envelope.json",
    "testnet401-epoch-cutover.json",
)
SOURCE_ASSET_NAMES = ("candidate.bundle", "candidate-bundle-binding.json")
PUBLIC_RELEASE_ASSET_NAMES = (
    "prior-release-channel-v2.json",
    "prior-release-lineage-v1.json",
    "prior-release-channels-v2.json",
)
TEMPORARY_TESTNET401_RECOVERY_RELEASE_SOURCES = {
    "f92748d00ced815e710e4e42ba8f7f17207507d7": (
        "leadpoet-parity-493765492819-880be82085a01367",
        "pp-34227587131-1",
    ),
    "b056b2989f019ddc373a9a8fa1ba7bb0c94feb22": (
        "leadpoet-parity-493765492819-bcaa5d9dbe8a99bc",
        "pp-34257145122-1",
    ),
    "d2d82773815b32e66410428cbec6b0c72514307c": (
        "leadpoet-parity-493765492819-25ed272de7addf14",
        "pp-34275942256-1",
    ),
    "11e4e824f8400bcef728dd76c3fabb6264ce498b": (
        "leadpoet-parity-493765492819-ac34db86a1deb0a8",
        "pp-34279790175-1",
    ),
}
MAX_CANDIDATE_BUNDLE_BYTES = 512 * 1024 * 1024
EARLY_BOOT_ISOLATION = """#cloud-boothook
#!/bin/bash
set -eu
for unit in $(systemctl list-unit-files --no-legend 2>/dev/null \
  | awk '$1 ~ /(leadpoet|research-lab|gateway|validator)/ {print $1}'); do
  systemctl mask --now "$unit" >/dev/null 2>&1 || true
done
install -d -m 0700 /run/leadpoet-testnet401
printf '%s\n' isolated >/run/leadpoet-testnet401/early-boot-isolated
printf '%s\n' '{expiry_epoch}' >/run/leadpoet-testnet401/expires-epoch
chmod 600 /run/leadpoet-testnet401/expires-epoch
cat >/etc/systemd/system/leadpoet-testnet401-expiry.service <<'EOF'
[Unit]
Description=Terminate temporary Leadpoet testnet401 host

[Service]
Type=oneshot
ExecStart=/sbin/shutdown -h now
EOF
cat >/etc/systemd/system/leadpoet-testnet401-expiry.timer <<'EOF'
[Unit]
Description=Bound temporary Leadpoet testnet401 host lifetime

[Timer]
OnCalendar=@{expiry_epoch}
Persistent=true
Unit=leadpoet-testnet401-expiry.service

[Install]
WantedBy=timers.target
EOF
chmod 644 \
  /etc/systemd/system/leadpoet-testnet401-expiry.service \
  /etc/systemd/system/leadpoet-testnet401-expiry.timer
systemctl daemon-reload
systemctl enable --now leadpoet-testnet401-expiry.timer
"""


class TemporaryHostError(RuntimeError):
    """The temporary host operation was not safely bounded."""


class TemporaryHostCreateError(TemporaryHostError):
    """Provisioning failed and has a public cleanup receipt."""

    def __init__(self, receipt: Mapping[str, Any]):
        super().__init__("temporary host provisioning failed")
        self.receipt = dict(receipt)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise TemporaryHostError("time must include a timezone")
    return value.astimezone(timezone.utc)


def _tag_map(tags: Any) -> dict[str, str]:
    if not isinstance(tags, list):
        return {}
    pairs = [
        (str(item.get("Key") or ""), str(item.get("Value") or ""))
        for item in tags
        if isinstance(item, Mapping) and item.get("Key")
    ]
    if len(pairs) != len(tags) or len({key for key, _value in pairs}) != len(pairs):
        return {}
    return dict(pairs)


def _error_label(operation: str, exc: BaseException) -> str:
    code = ""
    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code") or "")
    return f"{operation}:{code or type(exc).__name__}"


def _name(*, run_id: str, expires_at: datetime) -> str:
    return (
        f"leadpoet-parity-{run_id}-testnet401-exp-"
        f"{int(_utc(expires_at).timestamp())}"
    )


def _tags(
    *, run_id: str, candidate_sha: str, expires_at: datetime
) -> list[dict[str, str]]:
    if RUN_RE.fullmatch(run_id) is None or SHA_RE.fullmatch(candidate_sha) is None:
        raise TemporaryHostError("temporary host ownership inputs are invalid")
    name = _name(run_id=run_id, expires_at=expires_at)
    if NAME_RE.fullmatch(name) is None:
        raise TemporaryHostError("temporary host expiry marker is invalid")
    # The existing parity OIDC policy permits exactly these four launch tags.
    # The protected Name tag carries the expiry without widening IAM.
    return [
        {"Key": TAG_RUN, "Value": run_id},
        {"Key": TAG_SHA, "Value": candidate_sha},
        {"Key": TAG_EPHEMERAL, "Value": "true"},
        {"Key": "Name", "Value": name},
    ]


def _owned_identity(tags: Any) -> tuple[str, str, datetime] | None:
    values = _tag_map(tags)
    match = NAME_RE.fullmatch(values.get("Name", ""))
    run_id = values.get(TAG_RUN, "")
    candidate_sha = values.get(TAG_SHA, "")
    if (
        match is None
        or match.group("run") != run_id
        or RUN_RE.fullmatch(run_id) is None
        or SHA_RE.fullmatch(candidate_sha) is None
        or values.get(TAG_EPHEMERAL) != "true"
        or set(values) != {TAG_RUN, TAG_SHA, TAG_EPHEMERAL, "Name"}
    ):
        return None
    expiry = datetime.fromtimestamp(int(match.group("expiry")), tz=timezone.utc)
    return run_id, candidate_sha, expiry


def _wait_ssm_online(
    ssm: Any, instance_id: str, *, timeout_seconds: int = 600
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
        )
        values = response.get("InstanceInformationList", [])
        if len(values) == 1 and values[0].get("PingStatus") == "Online":
            return
        time.sleep(10)
    raise TemporaryHostError("temporary Nitro host did not become SSM-online")


def _verify_expiry_timer(
    ssm: Any,
    *,
    instance_id: str,
    expiry_epoch: int,
    timeout_seconds: int = 180,
) -> None:
    command = "\n".join([
        "set -u",
        "for attempt in $(seq 1 60); do",
        "  if test -f /run/leadpoet-testnet401/expires-epoch &&",
        f"     test \"$(cat /run/leadpoet-testnet401/expires-epoch)\" = {expiry_epoch} &&",
        "     systemctl is-enabled --quiet leadpoet-testnet401-expiry.timer &&",
        "     systemctl is-active --quiet leadpoet-testnet401-expiry.timer &&",
        (
            "     grep -Fx 'OnCalendar=@%s' "
            "/etc/systemd/system/leadpoet-testnet401-expiry.timer >/dev/null; then"
        ) % expiry_epoch,
        "    printf '%s\\n' expiry_timer_ready",
        "    exit 0",
        "  fi",
        "  sleep 2",
        "done",
        "exit 1",
    ])
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={
            "commands": [command],
            "executionTimeout": ["150"],
        },
        TimeoutSeconds=150,
    )
    command_id = str(response.get("Command", {}).get("CommandId") or "")
    if re.fullmatch(r"[0-9a-f-]{36}", command_id) is None:
        raise TemporaryHostError("expiry-timer SSM command identity is invalid")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            result = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code") or "") == (
                "InvocationDoesNotExist"
            ):
                time.sleep(2)
                continue
            raise
        status = str(result.get("Status") or "")
        if status in {"Pending", "InProgress", "Delayed"}:
            time.sleep(2)
            continue
        if (
            status != "Success"
            or int(result.get("ResponseCode", -1)) != 0
            or result.get("StandardOutputContent") != "expiry_timer_ready\n"
            or result.get("StandardErrorContent") not in (None, "")
        ):
            raise TemporaryHostError("expiry-timer SSM verification failed")
        return
    raise TemporaryHostError("expiry-timer SSM verification timed out")


def _artifact_bucket_name(*, run_id: str, candidate_sha: str) -> str:
    if RUN_RE.fullmatch(run_id) is None or SHA_RE.fullmatch(candidate_sha) is None:
        raise TemporaryHostError("temporary asset ownership inputs are invalid")
    suffix = hashlib.sha256(
        f"{ACCOUNT_ID}:{run_id}:{candidate_sha}".encode("ascii")
    ).hexdigest()[:16]
    return f"leadpoet-parity-{ACCOUNT_ID}-{suffix}"


def _read_public_release_object(s3: Any, *, bucket: str, key: str) -> Any:
    response = s3.get_object(Bucket=bucket, Key=key)
    payload = response["Body"].read(4 * 1024 * 1024 + 1)
    if not 0 < len(payload) <= 4 * 1024 * 1024:
        raise TemporaryHostError("prior release document is unbounded")
    return json.loads(payload)


def _recover_fixed_public_release_object(
    s3: Any, *, bucket: str, key: str,
) -> Any:
    """Read behind the exact latest delete marker, then immediately re-hide it."""

    page = s3.list_object_versions(Bucket=bucket, Prefix=key, MaxKeys=32)
    if page.get("IsTruncated"):
        raise TemporaryHostError("release recovery versions are unbounded")
    if any(item.get("Key") != key for name in ("Versions", "DeleteMarkers")
           for item in page.get(name, [])):
        raise TemporaryHostError("release recovery object scope differs")
    latest = [
        item for item in page.get("DeleteMarkers", [])
        if item.get("Key") == key and item.get("IsLatest") is True
    ]
    if len(latest) != 1 or any(
        item.get("Key") == key and item.get("IsLatest") is True
        for item in page.get("Versions", [])
    ):
        raise TemporaryHostError("release recovery delete marker differs")
    version_id = str(latest[0].get("VersionId") or "")
    if not 0 < len(version_id.encode("utf-8")) <= 1024:
        raise TemporaryHostError("release recovery delete marker differs")
    s3.delete_object(Bucket=bucket, Key=key, VersionId=version_id)
    try:
        return _read_public_release_object(s3, bucket=bucket, key=key)
    finally:
        hidden = s3.delete_object(Bucket=bucket, Key=key)
        if hidden.get("DeleteMarker") is not True or not str(
            hidden.get("VersionId") or ""
        ):
            raise TemporaryHostError("release recovery re-hide differs")


def _validated_release_channel_store(
    *, value: Any, lineage: Mapping[str, Any], current_channel: Mapping[str, Any],
    current_commit: str,
) -> dict[str, dict[str, Any]]:
    from gateway.tee.release_channel_v2 import (
        build_release_lineage_v2,
        validate_prior_release_channel_v2,
    )

    if not isinstance(value, Mapping) or set(value) != set(lineage["releases"]):
        raise TemporaryHostError("prior release channel set differs")
    channels = {
        commit: validate_prior_release_channel_v2(channel, expected_commit=commit)
        for commit, channel in sorted(value.items())
    }
    if channels.get(current_commit) != current_channel:
        raise TemporaryHostError("prior current release channel differs")
    expected = build_release_lineage_v2(
        list(channels.values()), current_commit=current_commit
    )
    if lineage != expected:
        raise TemporaryHostError("prior release documents differ")
    return channels


def create_asset_bucket(
    *,
    s3: Any,
    account_id: str,
    region: str,
    run_id: str,
    candidate_sha: str,
    bundle_path: Path,
    binding_path: Path,
    prior_release_run_id: str | None = None,
    prior_release_commit: str | None = None,
) -> dict[str, Any]:
    if (
        account_id != ACCOUNT_ID
        or region != REGION
        or RUN_RE.fullmatch(run_id) is None
        or SHA_RE.fullmatch(candidate_sha) is None
    ):
        raise TemporaryHostError("temporary asset-bucket AWS identity differs")
    try:
        bundle_metadata = bundle_path.lstat()
        binding_metadata = binding_path.lstat()
        if (
            not bundle_path.is_file()
            or bundle_path.is_symlink()
            or not 0 < bundle_metadata.st_size <= MAX_CANDIDATE_BUNDLE_BYTES
            or not binding_path.is_file()
            or binding_path.is_symlink()
            or not 0 < binding_metadata.st_size <= 4096
        ):
            raise TemporaryHostError("temporary candidate bundle is unavailable")
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256()
        with bundle_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        bundle_size = bundle_metadata.st_size
        bundle_hash = digest.hexdigest()
    except TemporaryHostError:
        raise
    except (OSError, ValueError) as exc:
        raise TemporaryHostError("temporary candidate bundle is unavailable") from exc
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"candidate-sha", "bundle-sha256", "bundle-size-bytes"}
        or binding.get("candidate-sha") != candidate_sha
        or binding.get("bundle-sha256") != bundle_hash
        or str(binding.get("bundle-size-bytes")) != str(bundle_size)
        or bundle_size <= 0
    ):
        raise TemporaryHostError("temporary candidate bundle identity differs")
    from scripts.provision_production_parity_staging import _create_artifact_bucket

    bucket = _create_artifact_bucket(
        s3,
        region=region,
        account_id=account_id,
        run_id=run_id,
        candidate_sha=candidate_sha,
    )
    prefix = ASSET_PREFIX_TEMPLATE.format(run_id=run_id)
    public_release_objects: list[str] = []
    recovered_release_commits: list[str] = []
    try:
        for path, name in (
            (bundle_path, SOURCE_ASSET_NAMES[0]),
            (binding_path, SOURCE_ASSET_NAMES[1]),
        ):
            s3.upload_file(
                str(path),
                bucket,
                f"{prefix}/{name}",
                ExtraArgs={"ServerSideEncryption": "AES256"},
            )
        if (prior_release_run_id is None) != (prior_release_commit is None):
            raise TemporaryHostError("prior release identity is incomplete")
        if prior_release_run_id is not None:
            prior_bucket = _artifact_bucket_name(
                run_id=prior_release_run_id,
                candidate_sha=str(prior_release_commit),
            )
            prior_prefix = ASSET_PREFIX_TEMPLATE.format(run_id=prior_release_run_id)
            values = [
                _read_public_release_object(
                    s3, bucket=prior_bucket, key=f"{prior_prefix}/{name}"
                )
                for name in PUBLIC_RELEASE_ASSET_NAMES[:2]
            ]
            from gateway.tee.release_channel_v2 import (
                validate_prior_release_channel_v2,
            )
            from gateway.tee.release_lineage_v2 import (
                validate_prior_compact_release_lineage_v2,
            )

            prior_channel = validate_prior_release_channel_v2(
                values[0], expected_commit=str(prior_release_commit)
            )
            prior_lineage = validate_prior_compact_release_lineage_v2(
                values[1], expected_current_commit=str(prior_release_commit)
            )
            try:
                channel_store = _read_public_release_object(
                    s3,
                    bucket=prior_bucket,
                    key=f"{prior_prefix}/{PUBLIC_RELEASE_ASSET_NAMES[2]}",
                )
            except ClientError as exc:
                if str(exc.response.get("Error", {}).get("Code") or "") not in {
                    "404", "NoSuchKey",
                }:
                    raise
                channel_store = {str(prior_release_commit): prior_channel}
                for commit in sorted(set(prior_lineage["releases"]) - set(channel_store)):
                    source = TEMPORARY_TESTNET401_RECOVERY_RELEASE_SOURCES.get(commit)
                    if source is None:
                        raise TemporaryHostError(
                            "prior release channel recovery source is unavailable"
                        )
                    source_bucket, source_run = source
                    source_key = (
                        f"production-parity/runs/{source_run}/testnet401/"
                        "prior-release-channel-v2.json"
                    )
                    try:
                        recovered = _read_public_release_object(
                            s3, bucket=source_bucket, key=source_key
                        )
                    except ClientError as exc:
                        if str(exc.response.get("Error", {}).get("Code") or "") not in {
                            "404", "NoSuchKey",
                        }:
                            raise
                        recovered = _recover_fixed_public_release_object(
                            s3, bucket=source_bucket, key=source_key
                        )
                        recovered_release_commits.append(commit)
                    channel_store[commit] = recovered
            channel_store = _validated_release_channel_store(
                value=channel_store,
                lineage=prior_lineage,
                current_channel=prior_channel,
                current_commit=str(prior_release_commit),
            )
            values.append(channel_store)
            for name, value in zip(PUBLIC_RELEASE_ASSET_NAMES, values):
                key = f"{prefix}/{name}"
                s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"),
                    ServerSideEncryption="AES256",
                )
                public_release_objects.append(key)
    except Exception as exc:
        raise TemporaryHostError(
            "temporary asset bucket created but source upload failed"
        ) from exc
    return {
        "schema_version": "leadpoet.temporary_testnet401_asset_bucket.v1",
        "status": "source_ready",
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "bucket": bucket,
        "prefix": prefix,
        "source_objects": [f"{prefix}/{name}" for name in SOURCE_ASSET_NAMES],
        "public_release_objects": public_release_objects,
        "temporarily_recovered_release_commits": sorted(
            recovered_release_commits
        ),
        "private_inputs_pending": [
            f"{prefix}/{name}" for name in PRIVATE_ASSET_NAMES
        ],
        "compliance_retention_days": 1,
    }


def wait_for_private_assets(
    *,
    s3: Any,
    account_id: str,
    region: str,
    run_id: str,
    candidate_sha: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    if (
        account_id != ACCOUNT_ID
        or region != REGION
        or timeout_seconds not in range(1, 1801)
    ):
        raise TemporaryHostError("temporary private-asset wait inputs differ")
    bucket = _artifact_bucket_name(run_id=run_id, candidate_sha=candidate_sha)
    prefix = ASSET_PREFIX_TEMPLATE.format(run_id=run_id)
    tags = _tag_map(s3.get_bucket_tagging(Bucket=bucket).get("TagSet", []))
    versioning = s3.get_bucket_versioning(Bucket=bucket)
    object_lock = s3.get_object_lock_configuration(Bucket=bucket).get(
        "ObjectLockConfiguration", {}
    )
    retention = object_lock.get("Rule", {}).get("DefaultRetention", {})
    public_access = s3.get_public_access_block(Bucket=bucket).get(
        "PublicAccessBlockConfiguration", {}
    )
    encryption = s3.get_bucket_encryption(Bucket=bucket).get(
        "ServerSideEncryptionConfiguration", {}
    )
    rules = encryption.get("Rules", [])
    if (
        tags
        != {
            TAG_RUN: run_id,
            TAG_SHA: candidate_sha,
            TAG_EPHEMERAL: "true",
            "Name": f"leadpoet-parity-{run_id}",
        }
        or versioning.get("Status") != "Enabled"
        or object_lock.get("ObjectLockEnabled") != "Enabled"
        or retention != {"Mode": "COMPLIANCE", "Days": 1}
        or set(public_access.values()) != {True}
        or len(public_access) != 4
        or len(rules) != 1
        or rules[0].get("ApplyServerSideEncryptionByDefault", {}).get(
            "SSEAlgorithm"
        )
        != "AES256"
    ):
        raise TemporaryHostError("temporary private asset bucket differs")
    deadline = time.monotonic() + timeout_seconds
    observed: dict[str, int] = {}
    while time.monotonic() < deadline:
        observed = {}
        for name in PRIVATE_ASSET_NAMES:
            try:
                value = s3.head_object(Bucket=bucket, Key=f"{prefix}/{name}")
            except ClientError as exc:
                if str(exc.response.get("Error", {}).get("Code") or "") in {
                    "404",
                    "NoSuchKey",
                    "NotFound",
                }:
                    continue
                raise
            size = int(value.get("ContentLength") or 0)
            if (
                not 0 < size <= 32 * 1024 * 1024
                or value.get("ServerSideEncryption") != "AES256"
            ):
                raise TemporaryHostError("temporary private asset metadata differs")
            observed[name] = size
        if set(observed) == set(PRIVATE_ASSET_NAMES):
            return {
                "schema_version": "leadpoet.temporary_testnet401_private_assets.v1",
                "status": "ready",
                "run_id": run_id,
                "candidate_sha": candidate_sha,
                "bucket": bucket,
                "prefix": prefix,
                "objects": {
                    name: observed[name] for name in PRIVATE_ASSET_NAMES
                },
            }
        time.sleep(min(15, max(1, timeout_seconds)))
    raise TemporaryHostError("temporary private assets did not arrive before timeout")


def remove_staging_asset_heads(
    *,
    s3: Any,
    account_id: str,
    region: str,
    run_id: str,
    candidate_sha: str,
) -> dict[str, Any]:
    if account_id != ACCOUNT_ID or region != REGION:
        raise TemporaryHostError("temporary staging cleanup AWS identity differs")
    bucket = _artifact_bucket_name(run_id=run_id, candidate_sha=candidate_sha)
    tags = s3.get_bucket_tagging(Bucket=bucket).get("TagSet", [])
    values = _tag_map(tags)
    if values != {
        TAG_RUN: run_id,
        TAG_SHA: candidate_sha,
        TAG_EPHEMERAL: "true",
        "Name": f"leadpoet-parity-{run_id}",
    }:
        raise TemporaryHostError("temporary asset-bucket ownership differs")
    prefix = ASSET_PREFIX_TEMPLATE.format(run_id=run_id)
    deleted: list[str] = []
    for name in (
        *SOURCE_ASSET_NAMES,
        *PRIVATE_ASSET_NAMES,
        *PUBLIC_RELEASE_ASSET_NAMES,
    ):
        key = f"{prefix}/{name}"
        s3.delete_object(Bucket=bucket, Key=key)
        deleted.append(key)
    return {
        "schema_version": "leadpoet.temporary_testnet401_staging_cleanup.v1",
        "status": "delete_markers_created",
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "bucket": bucket,
        "staging_object_heads_removed": deleted,
        "locked_versions_retained_until_compliance_expiry": True,
        "proof_prefix_preserved": f"{prefix}/evidence/",
    }


def _require_live_host(
    ec2: Any,
    *,
    instance_id: str,
    run_id: str,
    candidate_sha: str,
    now: datetime,
) -> Mapping[str, Any]:
    if (
        INSTANCE_ID_RE.fullmatch(instance_id) is None
        or RUN_RE.fullmatch(run_id) is None
        or SHA_RE.fullmatch(candidate_sha) is None
    ):
        raise TemporaryHostError("temporary SSM host identity is invalid")
    host = _instance_by_id(ec2, instance_id)
    identity = _owned_identity(host.get("Tags")) if host is not None else None
    if (
        host is None
        or identity is None
        or identity[:2] != (run_id, candidate_sha)
        or identity[2] <= _utc(now)
        or host.get("State", {}).get("Name") != "running"
        or host.get("ImageId") != AMI_ID
        or host.get("InstanceType") != INSTANCE_TYPE
        or host.get("SubnetId") != SUBNET_ID
        or host.get("VpcId") != VPC_ID
        or host.get("EnclaveOptions", {}).get("Enabled") is not True
    ):
        raise TemporaryHostError("temporary SSM host authority differs")
    _child_ids(host)
    return host


def _wait_ssm_command(
    ssm: Any,
    *,
    instance_id: str,
    command_id: str,
    timeout_seconds: int,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            result = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code") or "") == (
                "InvocationDoesNotExist"
            ):
                time.sleep(2)
                continue
            raise
        status = str(result.get("Status") or "")
        if status in {"Pending", "InProgress", "Delayed"}:
            time.sleep(5)
            continue
        stdout = str(result.get("StandardOutputContent") or "")
        stderr = str(result.get("StandardErrorContent") or "")
        if (
            status != "Success"
            or int(result.get("ResponseCode", -1)) != 0
            or stderr
            or len(stdout.encode("utf-8")) > 1024 * 1024
        ):
            evidence = _redacted_ssm_failure(
                command_id=command_id,
                status=status,
                response_code=result.get("ResponseCode", -1),
                stdout=stdout,
                stderr=stderr,
            )
            raise TemporaryHostError(
                "fixed temporary SSM stage failed "
                + json.dumps(evidence, sort_keys=True, separators=(",", ":"))
            )
        return stdout
    raise TemporaryHostError("fixed temporary SSM stage timed out")


def _redacted_ssm_failure(
    *, command_id: str, status: str, response_code: Any,
    stdout: str, stderr: str,
) -> dict[str, Any]:
    """Keep fixed failure identity without returning remote output."""
    statuses = {
        "Success", "Failed", "Cancelled", "TimedOut", "Cancelling",
        "Undeliverable", "Terminated",
    }
    categories = (
        "AccessDenied", "NoSuchKey", "ModuleNotFoundError", "ImportError",
        "FileNotFoundError", "PermissionError", "No space left on device",
        "AssertionError", "RuntimeError", "ValueError", "command not found",
        "unbound variable", "Killed", "Terminated", "Segmentation fault",
        "timed out",
    )
    tail = (stdout[-65536:] + "\n" + stderr[-65536:])
    try:
        code = int(response_code)
    except (TypeError, ValueError):
        code = -1
    if code < -1 or code > 255:
        code = -1
    identity: dict[str, Any] = {
        "schema_version": "leadpoet.temporary_testnet401_ssm_failure.v1",
        "ssm_command_id": (
            command_id if SSM_COMMAND_ID_RE.fullmatch(command_id) else "invalid"
        ),
        "ssm_status": status if status in statuses else "Unknown",
        "response_code": code,
        "error_categories": [item for item in categories if item in tail],
        "source_locations": [
            {"file": name, "line": int(line)}
            for name, line in re.findall(
                r'File "(?:[^"\n]*/)?([A-Za-z0-9_]+\.py)", line ([0-9]{1,6})',
                tail,
            )[-8:]
        ],
        "shell_locations": [
            {"file": name, "line": int(line)}
            for name, line in re.findall(
                r'([A-Za-z0-9_]+\.sh): line ([0-9]{1,6}):', tail
            )[-8:]
        ],
    }
    for line in stdout[-65536:].splitlines()[-20:]:
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, Mapping) or value.get("status") != "failed":
            continue
        safe: dict[str, str] = {}
        for name, pattern in {
            "error_type": r"[A-Za-z_][A-Za-z0-9_]{0,79}",
            "operation": r"[A-Za-z_][A-Za-z0-9_.:-]{0,79}",
            "code": r"[A-Za-z][A-Za-z0-9_.:-]{0,79}",
            "location": r"(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_]+\.py:[0-9]{1,6}",
        }.items():
            item = str(value.get(name) or "")
            if re.fullmatch(pattern, item):
                safe[name] = item
        if safe:
            identity["native_failure"] = safe
    return identity


def _send_fixed_ssm(
    ssm: Any,
    *,
    instance_id: str,
    command: str,
    timeout_seconds: int,
) -> tuple[str, str]:
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={
            "commands": [command],
            "executionTimeout": [str(timeout_seconds)],
        },
        TimeoutSeconds=timeout_seconds,
    )
    command_id = str(response.get("Command", {}).get("CommandId") or "")
    if SSM_COMMAND_ID_RE.fullmatch(command_id) is None:
        raise TemporaryHostError("fixed temporary SSM command identity is invalid")
    return command_id, _wait_ssm_command(
        ssm,
        instance_id=instance_id,
        command_id=command_id,
        timeout_seconds=timeout_seconds + 120,
    )


def source_bootstrap_command(
    *,
    run_id: str,
    candidate_sha: str,
    instance_id: str,
    assets_bucket: str,
    assets_prefix: str,
    prior_release_commit: str | None = None,
) -> str:
    expected_bucket = _artifact_bucket_name(
        run_id=run_id,
        candidate_sha=candidate_sha,
    )
    expected_prefix = ASSET_PREFIX_TEMPLATE.format(run_id=run_id)
    if (
        INSTANCE_ID_RE.fullmatch(instance_id) is None
        or assets_bucket != expected_bucket
        or assets_prefix != expected_prefix
    ):
        raise TemporaryHostError("temporary source-bootstrap assets differ")
    q = shlex.quote
    bundle = f"{RUNTIME_ROOT}/candidate.bundle"
    binding = f"{RUNTIME_ROOT}/candidate-bundle-binding.json"
    requirements = f"{RUNTIME_ROOT}/requirements.txt"
    stage_receipt = f"{RUNTIME_ROOT}/source-stage.json"
    config_probe = (
        "import json,os,stat,sys; p=sys.argv[1]; s=os.lstat(p); "
        "assert stat.S_ISREG(s.st_mode) and not stat.S_ISLNK(s.st_mode) "
        "and not s.st_mode & 0o077; v=json.load(open(p, encoding='utf-8')); "
        "assert v['run_id']==sys.argv[2] and v['candidate_sha']==sys.argv[3] "
        "and v['expected_instance_id']==sys.argv[4]"
    )
    binding_probe = (
        "import hashlib,json,os,sys; b=json.load(open(sys.argv[1], encoding='utf-8')); "
        "assert set(b)=={'candidate-sha','bundle-sha256','bundle-size-bytes'}; "
        "assert b['candidate-sha']==sys.argv[3]; p=sys.argv[2]; "
        "assert os.path.getsize(p)==int(b['bundle-size-bytes']); "
        "assert hashlib.sha256(open(p,'rb').read()).hexdigest()==b['bundle-sha256']"
    )
    lines = [
        "set -Eeuo pipefail",
        "umask 077",
        f"test -f {q(RUNTIME_ROOT + '/early-boot-isolated')}",
        f"test \"$(cat {q(RUNTIME_ROOT + '/early-boot-isolated')})\" = isolated",
        f"test ! -e {q(SOURCE_REPOSITORY)}",
        f"test ! -e {q(SOURCE_VENV)}",
        f"aws s3api get-object --region {q(REGION)} --bucket {q(assets_bucket)} "
        f"--key {q(assets_prefix + '/candidate.bundle')} {q(bundle)} >/dev/null 2>&1",
        f"aws s3api get-object --region {q(REGION)} --bucket {q(assets_bucket)} "
        f"--key {q(assets_prefix + '/candidate-bundle-binding.json')} {q(binding)} >/dev/null 2>&1",
        f"/usr/bin/python3 -I -c {q(binding_probe)} {q(binding)} {q(bundle)} {q(candidate_sha)}",
        "/usr/bin/dnf -q -y install aws-nitro-enclaves-cli "
        "aws-nitro-enclaves-cli-devel docker rsync jq tar gzip >/dev/null 2>&1",
        "test -x /usr/bin/nitro-cli",
        "/usr/bin/rpm -q aws-nitro-enclaves-cli aws-nitro-enclaves-cli-devel "
        "docker rsync jq tar gzip >/dev/null",
        "test -x /usr/bin/curl",
        "/usr/bin/systemctl enable --now docker.service >/dev/null 2>&1",
        "/usr/bin/systemctl is-active --quiet docker.service",
        "/usr/bin/docker info >/dev/null 2>&1",
        "if [ ! -x /usr/bin/git ]; then /usr/bin/dnf -q -y install git-core >/dev/null 2>&1; fi",
        f"install -d -m 0700 {q(str(Path(SOURCE_REPOSITORY).parent))} {q(SOURCE_REPOSITORY)}",
        f"/usr/bin/git -C {q(SOURCE_REPOSITORY)} init >/dev/null 2>&1",
        f"/usr/bin/git -C {q(SOURCE_REPOSITORY)} fetch --no-tags {q(bundle)} HEAD >/dev/null 2>&1",
        f"test \"$(/usr/bin/git -C {q(SOURCE_REPOSITORY)} rev-parse FETCH_HEAD)\" = {q(candidate_sha)}",
        f"/usr/bin/git -C {q(SOURCE_REPOSITORY)} checkout --detach {q(candidate_sha)} >/dev/null 2>&1",
        f"/usr/bin/git -C {q(SOURCE_REPOSITORY)} remote add origin https://github.com/leadpoet/leadpoet.git",
        f"/usr/bin/git -C {q(SOURCE_REPOSITORY)} update-ref refs/remotes/origin/main {q(candidate_sha)}",
        f"test \"$(/usr/bin/git -C {q(SOURCE_REPOSITORY)} rev-parse origin/main)\" = {q(candidate_sha)}",
        f"test -z \"$(/usr/bin/git -C {q(SOURCE_REPOSITORY)} status --porcelain --untracked-files=all)\"",
        "/usr/bin/dnf -q -y install python3.11-pip >/dev/null 2>&1",
        f"/usr/bin/python3.11 -I -m venv {q(SOURCE_VENV)}",
        f"/usr/bin/python3.11 {q(SOURCE_REPOSITORY + '/scripts/resolve_production_parity_controller_requirements.py')} "
        f"--requirements {q(SOURCE_REPOSITORY + '/requirements.txt')} --output {q(requirements)}",
        f"PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1 {q(SOURCE_VENV + '/bin/python3')} -m pip install "
        f"--disable-pip-version-check --no-input --no-cache-dir --requirement {q(requirements)} >/dev/null 2>&1",
        f"PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1 {q(SOURCE_VENV + '/bin/python3')} -m pip check >/dev/null 2>&1",
        f"test -f {q(SOURCE_REPOSITORY + '/scripts/stage_temporary_testnet_weights_host.py')}",
        f"cd {q(SOURCE_REPOSITORY)}",
        f"PYTHONPATH={q(SOURCE_REPOSITORY)} {q(SOURCE_VENV + '/bin/python3')} "
        "-m scripts.stage_temporary_testnet_weights_host "
        f"--candidate-sha {q(candidate_sha)} --run-id {q(run_id)} "
        f"--instance-id {q(instance_id)} --assets-bucket {q(assets_bucket)} "
        f"--assets-prefix {q(assets_prefix)} >{q(stage_receipt)}",
        f"chmod 600 {q(stage_receipt)} {q(NATIVE_CONFIG)}",
        f"/usr/bin/python3 -I -c {q(config_probe)} {q(NATIVE_CONFIG)} {q(run_id)} {q(candidate_sha)} {q(instance_id)}",
        f"rm -f {q(bundle)} {q(binding)}",
        "printf '%s\\n' temporary_testnet401_source_ready",
    ]
    if prior_release_commit is not None:
        if SHA_RE.fullmatch(prior_release_commit) is None or prior_release_commit == candidate_sha:
            raise TemporaryHostError("prior release commit is invalid")
        channel = f"{RUNTIME_ROOT}/{PUBLIC_RELEASE_ASSET_NAMES[0]}"
        lineage = f"{RUNTIME_ROOT}/{PUBLIC_RELEASE_ASSET_NAMES[1]}"
        insertion = 8
        channels = f"{RUNTIME_ROOT}/{PUBLIC_RELEASE_ASSET_NAMES[2]}"
        lines[insertion:insertion] = [
            f"aws s3api get-object --region {q(REGION)} --bucket {q(assets_bucket)} "
            f"--key {q(assets_prefix + '/' + name)} {q(path)} >/dev/null 2>&1"
            for name, path in zip(
                PUBLIC_RELEASE_ASSET_NAMES, (channel, lineage, channels)
            )
        ]
        stage_index = next(
            index for index, line in enumerate(lines)
            if "-m scripts.stage_temporary_testnet_weights_host " in line
        )
        lines[stage_index] += f" --prior-release-commit {q(prior_release_commit)}"
    return "\n".join(lines)


def run_source_bootstrap(
    *,
    ec2: Any,
    ssm: Any,
    account_id: str,
    region: str,
    run_id: str,
    candidate_sha: str,
    instance_id: str,
    assets_bucket: str,
    assets_prefix: str,
    now: datetime,
    prior_release_commit: str | None = None,
) -> dict[str, Any]:
    if account_id != ACCOUNT_ID or region != REGION:
        raise TemporaryHostError("temporary source-bootstrap AWS identity differs")
    _require_live_host(
        ec2,
        instance_id=instance_id,
        run_id=run_id,
        candidate_sha=candidate_sha,
        now=now,
    )
    command = source_bootstrap_command(
        run_id=run_id,
        candidate_sha=candidate_sha,
        instance_id=instance_id,
        assets_bucket=assets_bucket,
        assets_prefix=assets_prefix,
        prior_release_commit=prior_release_commit,
    )
    command_id, stdout = _send_fixed_ssm(
        ssm,
        instance_id=instance_id,
        command=command,
        timeout_seconds=7200,
    )
    if stdout != "temporary_testnet401_source_ready\n":
        raise TemporaryHostError("temporary source-bootstrap receipt differs")
    return {
        "schema_version": SOURCE_BOOTSTRAP_SCHEMA_VERSION,
        "status": "ready",
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "instance_id": instance_id,
        "assets_bucket": assets_bucket,
        "assets_prefix": assets_prefix,
        "ssm_command_id": command_id,
    }


def export_public_release_documents(
    *, ec2: Any, ssm: Any, s3: Any, account_id: str, region: str, run_id: str,
    candidate_sha: str, instance_id: str, now: datetime,
) -> dict[str, Any]:
    """Store the retained host's validated public release pair in its task bucket."""

    if account_id != ACCOUNT_ID or region != REGION:
        raise TemporaryHostError("temporary release export AWS identity differs")
    _require_live_host(
        ec2, instance_id=instance_id, run_id=run_id,
        candidate_sha=candidate_sha, now=now,
    )
    bucket = _artifact_bucket_name(run_id=run_id, candidate_sha=candidate_sha)
    prefix = ASSET_PREFIX_TEMPLATE.format(run_id=run_id)
    q = shlex.quote
    channel_file = f"{RUNTIME_ROOT}/export-{PUBLIC_RELEASE_ASSET_NAMES[0]}"
    lineage_file = f"{RUNTIME_ROOT}/export-{PUBLIC_RELEASE_ASSET_NAMES[1]}"
    channels_file = f"{RUNTIME_ROOT}/export-{PUBLIC_RELEASE_ASSET_NAMES[2]}"
    program = public_release_export_program(
        candidate_sha=candidate_sha,
        repository=SOURCE_REPOSITORY,
        config_path=NATIVE_CONFIG,
        channel_output=channel_file,
        lineage_output=lineage_file,
        channels_output=channels_file,
        channels_path=f"{RUNTIME_ROOT}/release-channels-v2.json",
    )
    command = "\n".join((
        "set -Eeuo pipefail", "umask 077",
        f"test ! -e {q(channel_file)} && test ! -e {q(lineage_file)} && test ! -e {q(channels_file)}",
        f"cleanup_public_release_export() {{ rm -f -- {q(channel_file)} {q(lineage_file)} {q(channels_file)}; }}",
        "trap cleanup_public_release_export EXIT",
        f"test \"$(/usr/bin/git -C {q(SOURCE_REPOSITORY)} rev-parse HEAD)\" = {q(candidate_sha)}",
        f"test -z \"$(/usr/bin/git -C {q(SOURCE_REPOSITORY)} status --porcelain --untracked-files=no)\"",
        f"{q(SOURCE_VENV + '/bin/python3')} -I -c {q(program)}",
        f"aws s3api put-object --region {q(REGION)} --bucket {q(bucket)} --key {q(prefix + '/' + PUBLIC_RELEASE_ASSET_NAMES[0])} --server-side-encryption AES256 --body {q(channel_file)} >/dev/null",
        f"aws s3api put-object --region {q(REGION)} --bucket {q(bucket)} --key {q(prefix + '/' + PUBLIC_RELEASE_ASSET_NAMES[1])} --server-side-encryption AES256 --body {q(lineage_file)} >/dev/null",
        f"aws s3api put-object --region {q(REGION)} --bucket {q(bucket)} --key {q(prefix + '/' + PUBLIC_RELEASE_ASSET_NAMES[2])} --server-side-encryption AES256 --body {q(channels_file)} >/dev/null",
        "printf '%s\\n' temporary_public_release_export_ready",
    ))
    command_id, stdout = _send_fixed_ssm(
        ssm, instance_id=instance_id, command=command, timeout_seconds=120,
    )
    if stdout != "temporary_public_release_export_ready\n":
        raise TemporaryHostError("temporary public release export receipt differs")
    values = []
    for name in PUBLIC_RELEASE_ASSET_NAMES:
        response = s3.get_object(Bucket=bucket, Key=f"{prefix}/{name}")
        payload = response["Body"].read(4 * 1024 * 1024 + 1)
        if not 0 < len(payload) <= 4 * 1024 * 1024:
            raise TemporaryHostError("exported public release document is unbounded")
        values.append(json.loads(payload))
    from gateway.tee.release_channel_v2 import validate_release_channel_v2
    from gateway.tee.release_lineage_v2 import validate_compact_release_lineage_v2
    channel = validate_release_channel_v2(values[0], expected_commit=candidate_sha)
    lineage = validate_compact_release_lineage_v2(
        values[1], expected_current_commit=candidate_sha
    )
    _validated_release_channel_store(
        value=values[2], lineage=lineage, current_channel=channel,
        current_commit=candidate_sha,
    )
    return {
        "status": "ready", "run_id": run_id, "candidate_sha": candidate_sha,
        "instance_id": instance_id, "bucket": bucket,
        "objects": [f"{prefix}/{name}" for name in PUBLIC_RELEASE_ASSET_NAMES],
        "channel_hash": channel["channel_hash"],
        "lineage_hash": lineage["lineage_hash"],
        "ssm_command_id": command_id,
    }


def public_release_export_program(
    *, candidate_sha: str, repository: str, config_path: str,
    channel_output: str, lineage_output: str, channels_output: str,
    channels_path: str,
) -> str:
    """Build the fixed old-host reader from APIs already present at f927."""

    if SHA_RE.fullmatch(candidate_sha) is None:
        raise TemporaryHostError("temporary release export commit is invalid")
    return (
        "import json,sys; from pathlib import Path; "
        f"sys.path.insert(0,{repository!r}); "
        "from scripts import bootstrap_temporary_testnet_weights_host as n; "
        "from gateway.tee.release_channel_v2 import build_release_channel_v2 as bc,build_release_lineage_v2 as bl,validate_release_channel_v2 as vc; "
        "from gateway.tee.release_lineage_v2 import validate_compact_release_lineage_v2 as vl; "
        "from leadpoet_canonical.attested_v2 import canonical_json; "
        f"c=n.load_config(Path({config_path!r})); "
        "g=json.loads(Path(c['gateway']['release_manifest']).read_text()); "
        "v=json.loads(Path(c['validator']['release_manifest']).read_text()); "
        f"a=vc(bc(gateway_release_manifest=g,validator_release_manifest=v),expected_commit={candidate_sha!r}); "
        f"b=vl(json.loads(Path(c['gateway']['release_lineage']).read_text()),expected_current_commit={candidate_sha!r}); "
        f"d=json.loads(Path({channels_path!r}).read_text()); "
        "assert isinstance(d,dict) and set(d)==set(b['releases']); "
        "d={k:vc(v,expected_commit=k) for k,v in sorted(d.items())}; "
        f"assert d.get({candidate_sha!r})==a; "
        f"assert b==bl(list(d.values()),current_commit={candidate_sha!r}); "
        f"Path({channel_output!r}).write_text(canonical_json(a)+'\\n',encoding='ascii'); "
        f"Path({lineage_output!r}).write_text(canonical_json(b)+'\\n',encoding='ascii'); "
        f"Path({channels_output!r}).write_text(canonical_json(d)+'\\n',encoding='ascii')"
    )


def _gateway_network_restart_host(
    *, repository, config_path, expected_run_id, expected_candidate_sha,
    expected_instance_id, native_module=None, proc_root="/proc",
):
    """Replace only the exact task gateway app with the corrected SDK network."""

    import json as _json
    import os as _os
    from pathlib import Path as _Path
    import signal as _signal
    import subprocess as _subprocess
    import sys as _sys
    import time as _time

    if native_module is None:
        _sys.path.insert(0, repository)
        from scripts import bootstrap_temporary_testnet_weights_host as native_module
    n = native_module
    config = n.load_config(_Path(config_path))
    if (
        config["run_id"] != expected_run_id
        or config["candidate_sha"] != expected_candidate_sha
        or config["expected_instance_id"] != expected_instance_id
        or expected_candidate_sha != GATEWAY_NETWORK_RESTART_CANDIDATE_SHA
    ):
        raise RuntimeError("gateway restart identity differs")
    n.verify_host_authority(config)
    n.validate_static_inputs(config)
    root = _Path(config["runtime_root"])
    lock_fd = _os.open(
        root / "gateway-network-restart.lock",
        _os.O_CREAT | _os.O_RDWR | getattr(_os, "O_NOFOLLOW", 0), 0o600,
    )
    try:
        import fcntl as _fcntl
        _fcntl.flock(lock_fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        state = n._load_process_state(config)
        processes = [dict(item) for item in state["processes"]]
        names = [item.get("name") for item in processes]
        preserved_names = {
            "gateway_egress_relay", "gateway_inter_enclave_relay",
            "validator_chain_relay", "validator_application",
        }
        allowed_names = preserved_names | {"gateway_application"}
        observed_names = set(names)
        if (
            len(names) != len(observed_names)
            or (observed_names != preserved_names and observed_names != allowed_names)
        ):
            raise RuntimeError("gateway restart process set differs")
        preserved = [item for item in processes if item["name"] in preserved_names]
        if {item["name"] for item in preserved} != preserved_names:
            raise RuntimeError("gateway restart preserved process set differs")
        if any(not n._same_process(item) for item in preserved):
            raise RuntimeError("gateway restart process identity differs")
        gateway_rows = [item for item in processes if item["name"] == "gateway_application"]
        if gateway_rows:
            gateway_pid = int(gateway_rows[0]["pid"])
            if n._same_process(gateway_rows[0]):
                if _os.getpgid(gateway_pid) != gateway_pid:
                    raise RuntimeError("gateway restart process group differs")
            else:
                try:
                    _os.getpgid(gateway_pid)
                except ProcessLookupError:
                    gateway_rows = []
                else:
                    raise RuntimeError("gateway restart process identity differs")

        def _enclaves():
            completed = _subprocess.run(
                ["nitro-cli", "describe-enclaves"], check=True,
                capture_output=True, text=True,
            )
            rows = _json.loads(completed.stdout)
            selected = sorted(
                (dict(item) for item in rows if int(item.get("EnclaveCID") or -1) in {16, 17, 18}),
                key=lambda item: int(item["EnclaveCID"]),
            )
            if (
                len(rows) != 3
                or len(selected) != 3
                or [int(item["EnclaveCID"]) for item in selected] != [16, 17, 18]
                or any(item.get("State") != "RUNNING" for item in selected)
                or selected[2].get("EnclaveName") != "leadpoet-testnet401-validator"
            ):
                raise RuntimeError("gateway restart enclave identity differs")
            return selected

        enclaves_before = _enclaves()
        gateway = config["gateway"]
        gateway_env_path = root / "gateway.env"
        n._private_regular_file(gateway_env_path, "scrubbed gateway environment")
        env = n._runtime_environment(
            gateway_env_path,
            overrides={
                **n._gateway_runtime_overrides(config),
                "GATEWAY_ENV_FILE": str(gateway_env_path),
                "GATEWAY_TEE_EIF_ROOT": gateway["eif_root"],
                "GATEWAY_V2_CONFIG_DIR": gateway["config_dir"],
                "GATEWAY_V2_RELEASE_MANIFEST": gateway["release_manifest"],
                "BT_SUBTENSOR_NETWORK": "test",
                "BT_SUBTENSOR_CHAIN_ENDPOINT": n.CHAIN_ENDPOINT,
            },
            repo_root=_Path(config["repo_root"]),
            candidate_sha=config["candidate_sha"],
        )
        runner = n.NativeRunner(config)
        runner.processes = preserved
        if gateway_rows:
            n._stop_owned_processes(gateway_rows)
            if n._same_process(gateway_rows[0]):
                raise RuntimeError("gateway restart old process remained live")
        runner.persist()
        if any(not n._same_process(item) for item in preserved):
            raise RuntimeError("gateway restart changed a preserved process")
        process = None
        new_record = None
        try:
            log_path = runner._log_path("gateway_application")
            with log_path.open("ab") as handle:
                process = _subprocess.Popen(
                    [config["python_bin"], "-u", "-m", "gateway.main"],
                    cwd=config["repo_root"], env=env, stdout=handle,
                    stderr=_subprocess.STDOUT, stdin=_subprocess.DEVNULL,
                    start_new_session=True,
                )
            _time.sleep(2)
            if process.poll() is not None or _os.getpgid(process.pid) != process.pid:
                raise RuntimeError("gateway restart replacement exited")
            new_record = n._process_identity(process.pid, name="gateway_application")
            runner.processes = preserved + [new_record]
            runner.persist()
            raw_environment = (_Path(proc_root) / str(process.pid) / "environ").read_bytes().split(b"\0")
            expected_aliases = {
                b"BT_SUBTENSOR_NETWORK=test",
                ("BT_SUBTENSOR_CHAIN_ENDPOINT=" + n.CHAIN_ENDPOINT).encode("ascii"),
            }
            if not expected_aliases.issubset(set(raw_environment)):
                raise RuntimeError("gateway restart network aliases differ")
            readiness = n._wait_gateway(config, runner)
            if not n._same_process(new_record):
                raise RuntimeError("gateway restart replacement is not live")
            if any(not n._same_process(item) for item in preserved):
                raise RuntimeError("gateway restart changed a preserved process")
            if _enclaves() != enclaves_before:
                raise RuntimeError("gateway restart changed enclave identity")
        except Exception:
            if new_record is not None:
                n._stop_owned_processes([new_record])
                if n._same_process(new_record):
                    runner.processes = preserved + [new_record]
                    runner.persist()
                    raise RuntimeError("gateway restart replacement remained live")
            elif process is not None and process.poll() is None:
                try:
                    if _os.getpgid(process.pid) == process.pid:
                        _os.killpg(process.pid, _signal.SIGTERM)
                        process.wait(timeout=20)
                except Exception:
                    try:
                        if _os.getpgid(process.pid) == process.pid:
                            _os.killpg(process.pid, _signal.SIGKILL)
                            process.wait(timeout=20)
                    except Exception:
                        pass
            runner.processes = preserved
            runner.persist()
            raise
        return {
            "schema_version": GATEWAY_NETWORK_RESTART_SCHEMA_VERSION,
            "status": "ready",
            "run_id": config["run_id"],
            "candidate_sha": config["candidate_sha"],
            "instance_id": config["expected_instance_id"],
            "old_gateway_pid": int(gateway_rows[0]["pid"]) if gateway_rows else None,
            "new_gateway_pid": int(new_record["pid"]),
            "preserved_process_names": sorted(preserved_names),
            "enclave_cids_unchanged": [16, 17, 18],
            "network_alias_names": ["BT_SUBTENSOR_CHAIN_ENDPOINT", "BT_SUBTENSOR_NETWORK"],
            "runtime_result_mutation_performed": False,
            "gateway_readiness": readiness,
        }
    finally:
        _os.close(lock_fd)


def gateway_network_restart_program(
    *, run_id: str, candidate_sha: str, instance_id: str,
) -> str:
    if candidate_sha != GATEWAY_NETWORK_RESTART_CANDIDATE_SHA:
        raise TemporaryHostError("gateway restart is not for the frozen candidate")
    source = inspect.getsource(_gateway_network_restart_host)
    return (
        "import json\n"
        f"GATEWAY_NETWORK_RESTART_CANDIDATE_SHA={GATEWAY_NETWORK_RESTART_CANDIDATE_SHA!r}\n"
        f"GATEWAY_NETWORK_RESTART_SCHEMA_VERSION={GATEWAY_NETWORK_RESTART_SCHEMA_VERSION!r}\n"
        + source
        + "\nprint(json.dumps(_gateway_network_restart_host("
        f"repository={SOURCE_REPOSITORY!r},config_path={NATIVE_CONFIG!r},"
        f"expected_run_id={run_id!r},expected_candidate_sha={candidate_sha!r},"
        f"expected_instance_id={instance_id!r}),sort_keys=True))\n"
    )


def run_gateway_network_restart(
    *, ec2: Any, ssm: Any, account_id: str, region: str, run_id: str,
    candidate_sha: str, instance_id: str, now: datetime,
) -> dict[str, Any]:
    if account_id != ACCOUNT_ID or region != REGION:
        raise TemporaryHostError("gateway restart AWS scope differs")
    _require_live_host(
        ec2, instance_id=instance_id, run_id=run_id,
        candidate_sha=candidate_sha, now=now,
    )
    program = gateway_network_restart_program(
        run_id=run_id, candidate_sha=candidate_sha, instance_id=instance_id,
    )
    command_id, stdout = _send_fixed_ssm(
        ssm, instance_id=instance_id,
        command=(
            "set -Eeuo pipefail\nexec "
            f"{shlex.quote(SOURCE_VENV + '/bin/python3')} -I -c "
            f"{shlex.quote(program)}"
        ),
        timeout_seconds=900,
    )
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise TemporaryHostError("gateway restart receipt is invalid") from exc
    expected_fields = {
        "schema_version", "status", "run_id", "candidate_sha", "instance_id",
        "old_gateway_pid", "new_gateway_pid", "preserved_process_names",
        "enclave_cids_unchanged", "network_alias_names",
        "runtime_result_mutation_performed", "gateway_readiness",
    }
    readiness = result.get("gateway_readiness") if isinstance(result, Mapping) else None
    if (
        not isinstance(result, Mapping)
        or set(result) != expected_fields
        or result.get("schema_version") != GATEWAY_NETWORK_RESTART_SCHEMA_VERSION
        or result.get("status") != "ready"
        or result.get("run_id") != run_id
        or result.get("candidate_sha") != candidate_sha
        or result.get("instance_id") != instance_id
        or result.get("runtime_result_mutation_performed") is not False
        or result.get("preserved_process_names") != [
            "gateway_egress_relay", "gateway_inter_enclave_relay",
            "validator_application", "validator_chain_relay",
        ]
        or result.get("enclave_cids_unchanged") != [16, 17, 18]
        or result.get("network_alias_names") != [
            "BT_SUBTENSOR_CHAIN_ENDPOINT", "BT_SUBTENSOR_NETWORK",
        ]
        or not isinstance(readiness, Mapping)
        or readiness.get("status") != "ready"
        or readiness.get("commit_sha") != candidate_sha
    ):
        raise TemporaryHostError("gateway restart receipt differs")
    return {key: result[key] for key in sorted(expected_fields)} | {
        "ssm_command_id": command_id
    }


def run_native_stage(
    *,
    ec2: Any,
    ssm: Any,
    account_id: str,
    region: str,
    run_id: str,
    candidate_sha: str,
    instance_id: str,
    stage: str,
    now: datetime,
) -> dict[str, Any]:
    if account_id != ACCOUNT_ID or region != REGION or stage not in NATIVE_STAGES:
        raise TemporaryHostError("temporary native-stage inputs differ")
    _require_live_host(
        ec2,
        instance_id=instance_id,
        run_id=run_id,
        candidate_sha=candidate_sha,
        now=now,
    )
    argv = [
        f"{SOURCE_VENV}/bin/python3",
        "-m",
        "scripts.bootstrap_temporary_testnet_weights_host",
        stage,
        "--config",
        NATIVE_CONFIG,
    ]
    if stage in {"launch", "cleanup"}:
        argv.extend(("--confirm-instance-id", instance_id))
    command = (
        "set -Eeuo pipefail\n"
        f"cd {shlex.quote(SOURCE_REPOSITORY)}\n"
        f"export PYTHONPATH={shlex.quote(SOURCE_REPOSITORY)}\n"
        "exec "
    ) + " ".join(
        shlex.quote(value) for value in argv
    )
    if stage == "status":
        # Failed source setup may not have installed Python or written config.
        # Inspect only fixed task paths; never return log lines or private data.
        probe = staging_diagnostic_program(
            run_id=run_id, candidate_sha=candidate_sha, instance_id=instance_id
        )
        launch_receipt = RUNTIME_ROOT + "/evidence/launch.json"
        failed_launch_probe = (
            "import json,sys; v=json.load(open(sys.argv[1], encoding='utf-8')); "
            "raise SystemExit(0 if v.get('stage') == 'launch' and "
            "v.get('status') == 'failed' else 1)"
        )
        command = (
            "set -Eeuo pipefail\n"
            f"if [ ! -f {shlex.quote(NATIVE_CONFIG)} ] || "
            f"[ ! -f {shlex.quote(RUNTIME_ROOT + '/processes.json')} ] || "
            f"( [ -f {shlex.quote(launch_receipt)} ] && "
            f"/usr/bin/python3 -I -c {shlex.quote(failed_launch_probe)} "
            f"{shlex.quote(launch_receipt)} ); then\n"
            f"  exec /usr/bin/python3 -I -c {shlex.quote(probe)}\n"
            "fi\n"
        ) + command
    command_id, stdout = _send_fixed_ssm(
        ssm,
        instance_id=instance_id,
        command=command,
        timeout_seconds=NATIVE_STAGE_TIMEOUTS[stage],
    )
    try:
        receipt = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise TemporaryHostError("temporary native-stage receipt is invalid") from exc
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != NATIVE_RECEIPT_SCHEMA_VERSION
        or receipt.get("stage") != stage
        or receipt.get("run_id") != run_id
        or receipt.get("candidate_sha") != candidate_sha
        or receipt.get("instance_id") != instance_id
    ):
        raise TemporaryHostError("temporary native-stage receipt differs")
    result = {
        "schema_version": "leadpoet.temporary_testnet401_native_ssm.v1",
        "stage": stage,
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "instance_id": instance_id,
        "ssm_command_id": command_id,
        "receipt": dict(receipt),
    }
    if stage == "status" and receipt.get("status") != "staging_incomplete":
        probe = runtime_log_diagnostic_program(
            run_id=run_id, candidate_sha=candidate_sha, instance_id=instance_id,
        )
        diagnostic_id, diagnostic_stdout = _send_fixed_ssm(
            ssm, instance_id=instance_id,
            command=(
                "set -Eeuo pipefail\nexec "
                f"{shlex.quote(SOURCE_VENV + '/bin/python3')} -I -c "
                f"{shlex.quote(probe)}"
            ),
            timeout_seconds=120,
        )
        try:
            diagnostics = json.loads(diagnostic_stdout)
        except json.JSONDecodeError as exc:
            raise TemporaryHostError("runtime log diagnostics are invalid") from exc
        _validate_runtime_log_diagnostics(diagnostics)
        result["runtime_log_ssm_command_id"] = diagnostic_id
        result["runtime_log_diagnostics"] = diagnostics
    return result


def _validate_runtime_log_diagnostics(value: Any) -> None:
    """Reject any remote projection that could carry arbitrary log content."""
    if not isinstance(value, Mapping) or set(value) != {"status", "logs"}:
        raise TemporaryHostError("runtime log diagnostics differ")
    if value["status"] != "ready" or not isinstance(value["logs"], list):
        raise TemporaryHostError("runtime log diagnostics differ")
    expected = ("gateway_application", "validator_application")
    if [item.get("process_name") for item in value["logs"]] != list(expected):
        raise TemporaryHostError("runtime log diagnostics differ")
    allowed = {
        "process_name", "process_live", "log_size_bytes", "tail_bytes_read",
        "progress_markers", "failure_markers", "exception_types",
        "source_locations", "latest_epoch_id", "latest_block",
        "reason_codes", "http_statuses", "allocation_build_callbacks",
        "latest_weight_input_http400_category",
    }
    progress_allowed = {
        "gateway_application": {
            "application_ready", "allocation_handoff_request", "weight_inputs_request",
            "compact_submission_request", "compact_finalization_request",
            "authority_health_request",
        },
        "validator_application": {
            "epoch_submission_started", "gateway_bundle_persisted",
            "finalized_state_persisted", "chain_submission_succeeded",
            "automatic_weight_tick",
        },
    }
    failure_allowed = {
        "automatic_weight_tick_failed", "pre_submission_guard_blocked",
        "submission_guard_blocked", "allocation_failed_closed",
        "leaderboard_snapshot_failed", "burn_submission_failed",
        "allocation_build_failed", "attested_allocation_not_ready",
    }
    exception_allowed = {
        "AssertionError", "AuthoritativeWeightFlowV2Error", "ConnectionError",
        "ChampionSettlementV2Error", "CoordinatorAllocationSourceV2Error",
        "CoordinatorChainSourceV2Error", "FileNotFoundError", "HTTPError",
        "HTTPException", "PermissionError", "ResearchLabV2AuthorityError",
        "RuntimeError", "TemporaryTestnet401FirstAllocationError",
        "TimeoutError", "ValueError",
    }
    reason_allowed = {
        "allocation_response_gzip_invalid", "allocation_response_gzip_truncated",
        "allocation_response_gzip_size_limit", "allocation_response_wire_size_limit",
        "allocation_response_encoding_unsupported", "allocation_fetch_exhausted",
        "allocation_policy_verification_failed", "weight_input_reconstruction_failed",
        "request_shape_invalid", "primary_validator_configuration_missing",
        "validator_hotkey_unauthorized", "netuid_unauthorized",
        "calculation_snapshot_hash_mismatch", "calculation_scope_mismatch",
        "allocation_hash_mismatch", "validator_signature_invalid",
        "epoch_authority_rejected", "compact_ancestry_unavailable",
        "champion_v2_cutover_blocked",
        "chain_realized_settlement_activation_unavailable_or_ambiguous",
        "chain_realized_settlement_activation_invalid",
        "v2_release_manifest_unavailable",
        "event_signer_reinitialization_tip_differs",
        "fresh_allocation_readiness_wrong_network",
        "fresh_testnet401_allocation_history_not_empty",
        "temporary_first_allocation_origin_unapproved",
        "temporary_first_allocation_cutover_root_differs",
        "temporary_first_allocation_cutover_receipt_unavailable",
        "temporary_first_allocation_cutover_receipt_differs",
        "testnet401_settlement_activation_ambiguous",
        "fresh_testnet401_cutover_authority_unavailable_or_ambiguous",
        "fresh_testnet401_cutover_authority_differs",
        "fresh_testnet401_cutover_receipt_differs",
        "fresh_testnet401_cutover_parent_duplicated",
        "temporary_first_allocation_cutover_parent_absent",
        "fresh_testnet401_allocation_origin_invalid",
        "fresh_testnet401_finalized_identities_differ",
        "fresh_testnet401_validator_last_update_absent",
        "fresh_testnet401_finalized_origin_invalid",
        "fresh_testnet401_finalized_origin_not_empty",
    }
    reason_allowed.update(label for _, label in RUNTIME_WEIGHT_INPUT_HTTP400_REASONS)
    endpoint_allowed = {
        "allocation_handoff", "weight_inputs", "compact_submission",
        "compact_finalization", "authority_health",
    }
    for item in value["logs"]:
        if not isinstance(item, Mapping) or not set(item) <= allowed:
            raise TemporaryHostError("runtime log diagnostics differ")
        if not isinstance(item.get("process_live"), bool):
            raise TemporaryHostError("runtime log diagnostics differ")
        for name in ("log_size_bytes", "tail_bytes_read"):
            if not isinstance(item.get(name), int) or not 0 <= item[name] <= 2**31:
                raise TemporaryHostError("runtime log diagnostics differ")
        progress_values = item.get("progress_markers")
        failure_values = item.get("failure_markers")
        if (
            not isinstance(progress_values, list)
            or len(progress_values) > len(progress_allowed[item["process_name"]])
            or any(not isinstance(value, str)
                   or value not in progress_allowed[item["process_name"]]
                   for value in progress_values)
            or not isinstance(failure_values, list)
            or len(failure_values) > len(failure_allowed)
            or any(not isinstance(value, str) or value not in failure_allowed
                   for value in failure_values)
        ):
            raise TemporaryHostError("runtime log diagnostics differ")
        reasons = item.get("reason_codes")
        if (
            not isinstance(reasons, list) or len(reasons) > len(reason_allowed)
            or any(not isinstance(value, str) or value not in reason_allowed
                   for value in reasons)
        ):
            raise TemporaryHostError("runtime log diagnostics differ")
        statuses = item.get("http_statuses")
        if not isinstance(statuses, list) or len(statuses) > 8 or any(
            not isinstance(entry, Mapping)
            or set(entry) != {"endpoint", "status"}
            or entry["endpoint"] not in endpoint_allowed
            or not isinstance(entry["status"], int)
            or not 100 <= entry["status"] <= 599
            for entry in statuses
        ):
            raise TemporaryHostError("runtime log diagnostics differ")
        callbacks = item.get("allocation_build_callbacks")
        if not isinstance(callbacks, list) or len(callbacks) > 2 or any(
            not isinstance(entry, Mapping)
            or set(entry) != {"error_type", "tokens"}
            or entry["error_type"] not in RUNTIME_CALLBACK_ERROR_TYPES
            or not isinstance(entry["tokens"], list)
            or len(entry["tokens"]) > 40
            or any(
                token != "[redacted]" and token not in RUNTIME_CALLBACK_TOKEN_VOCAB
                for token in entry["tokens"]
            )
            for entry in callbacks
        ):
            raise TemporaryHostError("runtime log diagnostics differ")
        exceptions = item.get("exception_types")
        if (
            not isinstance(exceptions, list)
            or len(exceptions) > len(exception_allowed)
            or any(not isinstance(value, str) or value not in exception_allowed
                   for value in exceptions)
        ):
            raise TemporaryHostError("runtime log diagnostics differ")
        locations = item.get("source_locations")
        if not isinstance(locations, list) or len(locations) > 8:
            raise TemporaryHostError("runtime log diagnostics differ")
        if any(
            not isinstance(entry, Mapping)
            or set(entry) != {"file", "line"}
            or re.fullmatch(r"[A-Za-z0-9_]+\.py", str(entry["file"])) is None
            or not isinstance(entry["line"], int)
            or not 1 <= entry["line"] <= 999999
            for entry in locations
        ):
            raise TemporaryHostError("runtime log diagnostics differ")
        for name in ("latest_epoch_id", "latest_block"):
            if name in item and (
                not isinstance(item[name], int) or not 0 <= item[name] <= 10**20 - 1
            ):
                raise TemporaryHostError("runtime log diagnostics differ")
        latest_http400_category = item.get("latest_weight_input_http400_category")
        if latest_http400_category is not None and latest_http400_category not in {
            label for _, label in RUNTIME_WEIGHT_INPUT_HTTP400_REASONS
        }:
            raise TemporaryHostError("runtime log diagnostics differ")


def runtime_log_diagnostic_program(
    *, run_id: str, candidate_sha: str, instance_id: str,
    repository: str = SOURCE_REPOSITORY, config_path: str = NATIVE_CONFIG,
) -> str:
    """Build a fixed, bounded reader for the two live application logs."""
    identity = (run_id, candidate_sha, instance_id)
    return "\n".join([
        "import json,re,stat,sys",
        "from pathlib import Path",
        f"sys.path.insert(0, {repository!r})",
        "from scripts import bootstrap_temporary_testnet_weights_host as native",
        f"config = native.load_config(Path({config_path!r}))",
        f"expected = {identity!r}",
        "actual = (config['run_id'], config['candidate_sha'], config['expected_instance_id'])",
        "assert actual == expected, 'host identity differs'",
        "state = native._load_process_state(config)",
        "names = ('gateway_application', 'validator_application')",
        "progress = {'gateway_application': ((b'Application startup complete', 'application_ready'), (b'/research-lab/allocations/attested/', 'allocation_handoff_request'), (b'/weights/inputs/v2', 'weight_inputs_request'), (b'/weights/submit/compact/v2', 'compact_submission_request'), (b'/weights/finalize/compact/v2', 'compact_finalization_request'), (b'/health/v2-authority', 'authority_health_request')), 'validator_application': ((b'SUBMITTING WEIGHTS FOR EPOCH', 'epoch_submission_started'), (b'Authoritative V2 gateway bundle persisted:', 'gateway_bundle_persisted'), (b'Authoritative V2 finalized chain state persisted:', 'finalized_state_persisted'), (b'Successfully submitted weights to Bittensor chain', 'chain_submission_succeeded'), (b'\"event\": \"automatic_weight_tick\"', 'automatic_weight_tick'))}",
        "failures = ((b'\"event\": \"automatic_weight_tick_failed\"', 'automatic_weight_tick_failed'), (b'Research Lab pre-submission guard blocked weights', 'pre_submission_guard_blocked'), (b'weight_submission_blocked_by_guard', 'submission_guard_blocked'), (b'Authoritative V2 Research Lab allocation failed closed', 'allocation_failed_closed'), (b'leaderboard snapshot failed', 'leaderboard_snapshot_failed'), (b'Failed to submit burn weights', 'burn_submission_failed'), (b'research_lab_allocation_build_failed', 'allocation_build_failed'), (b'research_lab_attested_allocation_not_ready', 'attested_allocation_not_ready'))",
        "exceptions = tuple(name.encode() for name in ('AssertionError', 'AuthoritativeWeightFlowV2Error', 'ChampionSettlementV2Error', 'ConnectionError', 'CoordinatorAllocationSourceV2Error', 'CoordinatorChainSourceV2Error', 'FileNotFoundError', 'HTTPError', 'HTTPException', 'PermissionError', 'ResearchLabV2AuthorityError', 'RuntimeError', 'TemporaryTestnet401FirstAllocationError', 'TimeoutError', 'ValueError'))",
        "reasons = ((b'allocation response gzip is invalid', 'allocation_response_gzip_invalid'), (b'allocation response gzip is truncated', 'allocation_response_gzip_truncated'), (b'allocation response gzip exceeds size limit', 'allocation_response_gzip_size_limit'), (b'allocation response exceeds wire size limit', 'allocation_response_wire_size_limit'), (b'unsupported allocation response encoding', 'allocation_response_encoding_unsupported'), (b'allocation fetch exhausted without a response', 'allocation_fetch_exhausted'), (b'Research Lab allocation arithmetic or policy verification failed', 'allocation_policy_verification_failed'), (b'Authoritative V2 weight input reconstruction failed closed', 'weight_input_reconstruction_failed'), (b'champion V2 cutover blocked:', 'champion_v2_cutover_blocked'), (b'chain-realized settlement activation is unavailable or ambiguous', 'chain_realized_settlement_activation_unavailable_or_ambiguous'), (b'chain-realized settlement activation is invalid', 'chain_realized_settlement_activation_invalid'), (b'fresh allocation readiness is restricted to testnet401', 'fresh_allocation_readiness_wrong_network'), (b'fresh testnet401 allocation history is not empty', 'fresh_testnet401_allocation_history_not_empty'), (b'temporary first allocation is not the approved testnet401 origin', 'temporary_first_allocation_origin_unapproved'), (b'temporary first allocation cutover root differs', 'temporary_first_allocation_cutover_root_differs'), (b'temporary first allocation cutover receipt is unavailable', 'temporary_first_allocation_cutover_receipt_unavailable'), (b'temporary first allocation cutover receipt differs', 'temporary_first_allocation_cutover_receipt_differs'), (b'testnet401 settlement activation is ambiguous', 'testnet401_settlement_activation_ambiguous'), (b'fresh testnet401 cutover authority is unavailable or ambiguous', 'fresh_testnet401_cutover_authority_unavailable_or_ambiguous'), (b'fresh testnet401 cutover authority differs', 'fresh_testnet401_cutover_authority_differs'), (b'fresh testnet401 cutover receipt differs', 'fresh_testnet401_cutover_receipt_differs'), (b'fresh testnet401 cutover parent is duplicated', 'fresh_testnet401_cutover_parent_duplicated'), (b'temporary first allocation cutover parent is absent', 'temporary_first_allocation_cutover_parent_absent'), (b'fresh testnet401 allocation origin is invalid', 'fresh_testnet401_allocation_origin_invalid'), (b'fresh testnet401 finalized identities differ', 'fresh_testnet401_finalized_identities_differ'), (b'fresh testnet401 validator LastUpdate is absent', 'fresh_testnet401_validator_last_update_absent'), (b'fresh testnet401 finalized origin is invalid', 'fresh_testnet401_finalized_origin_invalid'), (b'fresh testnet401 finalized origin is not empty', 'fresh_testnet401_finalized_origin_not_empty'), (b'request_shape_invalid', 'request_shape_invalid'), (b'primary_validator_configuration_missing', 'primary_validator_configuration_missing'), (b'validator_hotkey_unauthorized', 'validator_hotkey_unauthorized'), (b'netuid_unauthorized', 'netuid_unauthorized'), (b'calculation_snapshot_hash_mismatch', 'calculation_snapshot_hash_mismatch'), (b'calculation_scope_mismatch', 'calculation_scope_mismatch'), (b'allocation_hash_mismatch', 'allocation_hash_mismatch'), (b'validator_signature_invalid', 'validator_signature_invalid'), (b'epoch_authority_rejected', 'epoch_authority_rejected'), (b'compact_ancestry_unavailable', 'compact_ancestry_unavailable'))",
        "reasons += ((b'V2 release manifest is unavailable', 'v2_release_manifest_unavailable'),)",
        "reasons += ((b'transparency signer is already initialized with another log tip', 'event_signer_reinitialization_tip_differs'),)",
        f"weight_input_400_reasons = {RUNTIME_WEIGHT_INPUT_HTTP400_REASONS!r}",
        "reasons += weight_input_400_reasons",
        f"callback_types = {RUNTIME_CALLBACK_ERROR_TYPES!r}",
        f"callback_vocab = {RUNTIME_CALLBACK_TOKEN_VOCAB!r}",
        "rows = []",
        "for name in names:",
        "    records = [p for p in state['processes'] if p.get('name') == name]",
        "    assert len(records) <= 1, 'process records differ'",
        "    process_live = bool(records and native._same_process(records[0]))",
        "    path = Path(config['runtime_root']) / 'logs' / (name + '.log')",
        "    meta = path.lstat()",
        "    assert stat.S_ISREG(meta.st_mode) and not stat.S_ISLNK(meta.st_mode), 'log differs'",
        "    size = meta.st_size; assert 0 <= size <= 2**31, 'log size differs'",
        "    with path.open('rb') as stream:",
        "        stream.seek(max(0, size - 262144)); data = stream.read(262144)",
        "    row = {'process_name': name, 'process_live': True, 'log_size_bytes': size, 'tail_bytes_read': len(data), 'progress_markers': [label for token,label in progress[name] if token in data], 'failure_markers': [label for token,label in failures if token in data], 'exception_types': [token.decode() for token in exceptions if token in data], 'reason_codes': [label for token,label in reasons if token in data], 'source_locations': [{'file': f.decode(), 'line': int(line)} for f,line in re.findall(rb'File \"(?:[^\"\\n]*/)?([A-Za-z0-9_]+\\.py)\", line ([0-9]{1,6})', data)[-8:]]}",
        "    row['process_live'] = process_live",
        "    callbacks = []",
        "    for line in data.splitlines():",
        "        if b'research_lab_allocation_build_failed ' not in line: continue",
        "        kind = re.search(rb'error_type=([A-Za-z_][A-Za-z0-9_]{0,79})', line)",
        "        kind = kind.group(1).decode() if kind and kind.group(1).decode() in callback_types else 'unclassified'",
        "        error = line.split(b' error=', 1)[1] if b' error=' in line else b''",
        "        error = re.sub(rb'(?:api[_-]?key|token|secret|credential|password|raw[_-]?payload)[=:][^\\s]+', b' ', error, flags=re.I)",
        "        error = re.sub(rb'https?://[^\\s]+', b' ', error, flags=re.I)",
        "        tokens = []",
        "        for raw in re.findall(rb'[A-Za-z_][A-Za-z0-9_]{0,127}', error):",
        "            token = raw.decode().lower()",
        "            projected = token if token in callback_vocab else '[redacted]'",
        "            if not tokens or projected != '[redacted]' or tokens[-1] != '[redacted]': tokens.append(projected)",
        "            if len(tokens) == 40: break",
        "        callbacks.append({'error_type': kind, 'tokens': tokens})",
        "    if callbacks:",
        "        first = callbacks[0]",
        "        latest = next((item for item in reversed(callbacks) if item != first), None)",
        "        row['allocation_build_callbacks'] = [first] + ([latest] if latest else [])",
        "    else: row['allocation_build_callbacks'] = []",
        "    endpoints = ((b'/research-lab/allocations/attested/', 'allocation_handoff'), (b'/weights/inputs/v2', 'weight_inputs'), (b'/weights/submit/compact/v2', 'compact_submission'), (b'/weights/finalize/compact/v2', 'compact_finalization'), (b'/health/v2-authority', 'authority_health'))",
        "    found_statuses = []",
        "    for line in data.splitlines():",
        "        status = re.search(rb'HTTP/[0-9.]+[\\\" ]+([1-5][0-9]{2})(?: |$)', line) or re.search(rb'HTTP Error ([1-5][0-9]{2})(?::| |$)', line)",
        "        if status:",
        "            for token,label in endpoints:",
        "                if token in line: found_statuses.append({'endpoint': label, 'status': int(status.group(1))}); break",
        "            else:",
        "                if name == 'validator_application' and b'Authoritative V2 Research Lab allocation failed closed' in line: found_statuses.append({'endpoint': 'allocation_handoff', 'status': int(status.group(1))})",
        "    row['http_statuses'] = found_statuses[-8:]",
        "    latest_weight_input_http400_category = None",
        "    for line in data.splitlines():",
        "        if b'gateway V2 weight input request failed with HTTP 400:' not in line: continue",
        "        matched = next((label for token,label in weight_input_400_reasons if token in line), None)",
        "        if matched is not None: latest_weight_input_http400_category = matched",
        "    if latest_weight_input_http400_category is not None: row['latest_weight_input_http400_category'] = latest_weight_input_http400_category",
        "    epochs = re.findall(rb'SUBMITTING WEIGHTS FOR EPOCH ([0-9]{1,20})', data)",
        "    blocks = re.findall(rb'Block: ([0-9]{1,20}) \\(block ', data)",
        "    if epochs: row['latest_epoch_id'] = int(epochs[-1])",
        "    if blocks: row['latest_block'] = int(blocks[-1])",
        "    rows.append(row)",
        "print(json.dumps({'status': 'ready', 'logs': rows}, sort_keys=True, separators=(',', ':')))",
    ])


def staging_diagnostic_program(*, run_id: str, candidate_sha: str,
                               instance_id: str) -> str:
    """Read-only, bounded pre-config diagnostics with no raw log output."""
    identity = {
        "schema_version": NATIVE_RECEIPT_SCHEMA_VERSION,
        "stage": "status", "status": "staging_incomplete",
        "run_id": run_id, "candidate_sha": candidate_sha,
        "instance_id": instance_id,
    }
    return "\n".join([
        "import json, pathlib, re, subprocess",
        f"result = {identity!r}",
        f"root = pathlib.Path({RUNTIME_ROOT!r})",
        "names = ('early-boot-isolated', 'expires-epoch', 'candidate.bundle', "
        "'candidate-bundle-binding.json', 'requirements.txt', 'source-stage.json', "
        "'config.json', 'processes.json', 'evidence/launch.json', 'inputs', "
        "'staging-logs', 'logs')",
        "result['paths_present'] = {name: (root / name).exists() for name in names}",
        f"result['repository_exists'] = pathlib.Path({SOURCE_REPOSITORY!r}).exists()",
        f"result['venv_exists'] = pathlib.Path({SOURCE_VENV!r}).exists()",
        "result['stage_states'] = []",
        "path = root / 'source-stage.json'",
        "if path.is_file() and not path.is_symlink():",
        "    for line in path.open().read(65536).splitlines()[-20:]:",
        "        try: value = json.loads(line)",
        "        except ValueError: continue",
        "        if isinstance(value, dict) and value.get('status') == 'failed':",
        "            safe = {}",
        "            for key, pattern in {'error_type':'[A-Za-z_][A-Za-z0-9_]{0,79}', 'operation':'[A-Za-z_][A-Za-z0-9_.:-]{0,79}', 'code':'[A-Za-z][A-Za-z0-9_.:-]{0,79}', 'location':'(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_]+\\.py:[0-9]{1,6}'}.items():",
        "                item = str(value.get(key, ''))",
        "                if re.fullmatch(pattern, item): safe[key] = item",
        "            if safe: result['staging_failure'] = safe",
        "        if isinstance(value, dict) and re.fullmatch('[a-z_]{1,64}', "
        "str(value.get('stage', ''))) and value.get('status') in ('running', 'passed'):",
        "            result['stage_states'].append({key: value[key] for key in ('stage', 'status')})",
        "launch = root / 'evidence' / 'launch.json'",
        "if launch.is_file() and not launch.is_symlink() and launch.stat().st_size <= 65536:",
        "    try: value = json.loads(launch.read_text())",
        "    except ValueError: value = {}",
        "    expected = {'schema_version':'leadpoet.temporary_testnet401_native_bootstrap_receipt.v1', 'stage':'launch', 'status':'failed', 'run_id':result['run_id'], 'candidate_sha':result['candidate_sha'], 'instance_id':result['instance_id']}",
        "    if isinstance(value, dict) and all(value.get(k) == v for k, v in expected.items()):",
        "        failure_type = str((value.get('evidence') or {}).get('failure_type', '')) if isinstance(value.get('evidence'), dict) else ''",
        "        if re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,79}', failure_type): result['launch_failure'] = {'status':'failed', 'failure_type':failure_type}",
        "patterns = ('AccessDenied', 'ModuleNotFoundError', 'ImportError', "
        "'PermissionError', 'NoSuchKey', 'No space left on device', 'AssertionError', "
        "'RuntimeError', 'ValueError', 'command not found', 'not found', 'fatal:', "
        "'ResolutionImpossible', 'No matching distribution', 'FileExistsError', "
        "'unbound variable', 'Killed', 'Terminated', 'Segmentation fault', "
        "'invalid PCR0', 'no live lock owner', 'protected workflow', 'timed out', "
        "'V2RuntimeReadinessError', 'runtime clients do not cover every role', "
        "'coordinator provider broker is not ready', "
        "'coordinator provider semantics authority is not ready', "
        "'gateway_coordinator execution manager is not ready', "
        "'gateway_scoring execution manager is not ready', 'workers_alive')",
        "result['log_diagnostics'] = []",
        f"ssm = pathlib.Path('/var/lib/amazon/ssm/{instance_id}/document/orchestration')",
        "paths = list(ssm.glob('*/awsrunShellScript/0.awsrunShellScript/stderr'))[-12:]",
        "paths += list((root / 'staging-logs').glob('*.log'))[:16]",
        "paths += list((root / 'logs').glob('*.log'))[:24]",
        "for path in paths:",
        "    if path.is_symlink() or not path.is_file(): continue",
        "    with path.open('rb') as stream:",
        "        stream.seek(max(0, path.stat().st_size - 65536))",
        "        data = stream.read(65536).decode('utf-8', 'replace')",
        "    result['log_diagnostics'].append({'file': path.name, "
        "'bytes': path.stat().st_size, 'categories': [p for p in patterns if p in data], "
        "'trace_locations': re.findall(r'File \"[^\"\\n]*/([a-zA-Z0-9_]+\\.py)\", line ([0-9]{1,6})', data)[-8:], "
        "'nitro_error_codes': sorted(set(re.findall(r'\\[\\s*(E[0-9]{1,3})\\s*\\]', data))), "
        "'child_exit_codes': re.findall(r'failed with exit code ([0-9]{1,3})', data)[-4:], "
        "'shell_locations': re.findall(r'([a-zA-Z0-9_]+\\.sh): line ([0-9]{1,6}):', data)[-8:], "
        "'build_milestones': [p for p in ('Building one local gateway identity', 'gateway_reproducible_pcr0_build', 'Building one local validator identity', 'Building Validator Nitro Enclave Image', 'Cleaning PCR0 Docker context', 'Verifying protected validator', 'Building pinned bittensor-drand', 'Writing validator V2 release metadata') if p in data], "
        "'system_error_categories': [p for p in ('Permission denied', 'No such file or directory', 'Cannot allocate memory', 'Invalid argument', 'Read-only file system', 'File exists', 'Out of memory') if p in data]})",
        "probe = " + repr("\n".join([
            "import json,sys,traceback",
            f"sys.path.insert(0, {SOURCE_REPOSITORY!r})",
            "from pathlib import Path",
            "from scripts import stage_temporary_testnet_weights_host as stage",
            "from scripts import bootstrap_temporary_testnet_weights_host as native",
            f"config = stage.build_config(repository=Path({SOURCE_REPOSITORY!r}), candidate={candidate_sha!r}, run_id={run_id!r}, instance_id={instance_id!r}, expiry=int(Path({(RUNTIME_ROOT + '/expires-epoch')!r}).read_text()))",
            "result = {}",
            "for name in ('verify_host_authority', 'verify_host_is_empty'):",
            "    try:",
            "        getattr(native, name)(config)",
            "        result[name] = {'status': 'passed'}",
            "    except Exception as exc:",
            "        response = getattr(exc, 'response', {})",
            "        result[name] = {'status': 'failed', 'type': type(exc).__name__, 'operation': getattr(exc, 'operation_name', ''), 'code': response.get('Error', {}).get('Code', ''), 'line': traceback.extract_tb(exc.__traceback__)[-1].lineno}",
            "print(json.dumps(result, sort_keys=True))",
        ])),
        f"if pathlib.Path({(SOURCE_VENV + '/bin/python3')!r}).is_file():",
        f"    checked = subprocess.run([{(SOURCE_VENV + '/bin/python3')!r}, '-I', '-c', probe], capture_output=True, text=True, timeout=90)",
        "    if checked.returncode == 0:",
        "        try: result['pre_input_checks'] = json.loads(checked.stdout)",
        "        except ValueError: result['pre_input_checks'] = {'output': 'invalid'}",
        "    else: result['pre_input_checks'] = {'process_exit': checked.returncode}",
        "print(json.dumps(result, sort_keys=True))",
    ])


def _delete_security_group(ec2: Any, group_id: str) -> None:
    for attempt in range(1, 7):
        try:
            ec2.delete_security_group(GroupId=group_id)
            return
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code") or "")
            if code == "InvalidGroup.NotFound":
                return
            if code != "DependencyViolation" or attempt == 6:
                raise
            time.sleep(5)


def _pages(call: Any, **request: Any) -> list[Mapping[str, Any]]:
    pages: list[Mapping[str, Any]] = []
    token: str | None = None
    seen: set[str] = set()
    for _ in range(1000):
        current = dict(request)
        if token is not None:
            current["NextToken"] = token
        response = call(**current)
        if not isinstance(response, Mapping):
            raise TemporaryHostError("AWS inventory page is invalid")
        pages.append(response)
        next_token = response.get("NextToken")
        if next_token in (None, ""):
            return pages
        if (
            not isinstance(next_token, str)
            or next_token in seen
            or next_token == token
        ):
            raise TemporaryHostError("AWS inventory pagination is invalid")
        seen.add(next_token)
        token = next_token
    raise TemporaryHostError("AWS inventory pagination exceeded its bound")


def _instance_by_id(ec2: Any, instance_id: str) -> Mapping[str, Any] | None:
    response = ec2.describe_instances(InstanceIds=[instance_id])
    values = [
        item
        for reservation in response.get("Reservations", [])
        for item in reservation.get("Instances", [])
    ]
    if not values:
        return None
    if len(values) != 1 or values[0].get("InstanceId") != instance_id:
        raise TemporaryHostError("temporary instance inventory is ambiguous")
    return values[0]


def _child_ids(host: Mapping[str, Any]) -> tuple[str, str]:
    volumes = host.get("BlockDeviceMappings", [])
    interfaces = host.get("NetworkInterfaces", [])
    if len(volumes) != 1 or len(interfaces) != 1:
        raise TemporaryHostError("temporary child-resource inventory differs")
    volume = volumes[0]
    interface = interfaces[0]
    volume_id = str(volume.get("Ebs", {}).get("VolumeId") or "")
    interface_id = str(interface.get("NetworkInterfaceId") or "")
    if (
        volume.get("DeviceName") != "/dev/xvda"
        or volume.get("Ebs", {}).get("DeleteOnTermination") is not True
        or VOLUME_ID_RE.fullmatch(volume_id) is None
        or NETWORK_INTERFACE_ID_RE.fullmatch(interface_id) is None
        or interface.get("Attachment", {}).get("DeleteOnTermination") is not True
    ):
        raise TemporaryHostError("temporary child-resource identity differs")
    return volume_id, interface_id


def _confirm_children_deleted(
    ec2: Any,
    *,
    volume_ids: Sequence[str],
    network_interface_ids: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    if volume_ids:
        try:
            ec2.get_waiter("volume_deleted").wait(
                VolumeIds=sorted(set(volume_ids)),
                WaiterConfig={"Delay": 5, "MaxAttempts": 24},
            )
        except Exception as exc:  # noqa: BLE001 - return bounded residue evidence
            errors.append(_error_label("volumes-delete-on-termination", exc))
    remaining = set(network_interface_ids)
    for _ in range(24):
        if not remaining:
            break
        try:
            response = ec2.describe_network_interfaces(
                NetworkInterfaceIds=sorted(remaining)
            )
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code") or "") == (
                "InvalidNetworkInterfaceID.NotFound"
            ):
                remaining.clear()
                break
            errors.append(_error_label("network-interfaces-delete", exc))
            break
        observed = {
            str(item.get("NetworkInterfaceId") or "")
            for item in response.get("NetworkInterfaces", [])
        }
        remaining &= observed
        if remaining:
            time.sleep(5)
    if remaining:
        errors.append("network-interfaces-delete:residue")
    return errors


def _rollback(
    ec2: Any,
    *,
    account_id: str,
    region: str,
    run_id: str,
    candidate_sha: str,
    journal: Mapping[str, str],
) -> dict[str, Any]:
    instance_id = str(journal.get("instance_id") or "")
    group_id = str(journal.get("security_group_id") or "")
    volume_id = str(journal.get("volume_id") or "")
    network_interface_id = str(journal.get("network_interface_id") or "")
    errors: list[str] = []
    if INSTANCE_ID_RE.fullmatch(instance_id):
        try:
            instance = _instance_by_id(ec2, instance_id)
            identity = (
                _owned_identity(instance.get("Tags"))
                if instance is not None
                else None
            )
            if instance is not None and (
                identity is None or identity[:2] != (run_id, candidate_sha)
            ):
                raise TemporaryHostError("rollback instance ownership changed")
            if instance is not None:
                volume_id, network_interface_id = _child_ids(instance)
            if instance is not None and instance.get("State", {}).get("Name") not in {
                "terminated",
                "shutting-down",
            }:
                ec2.terminate_instances(InstanceIds=[instance_id])
                ec2.get_waiter("instance_terminated").wait(
                    InstanceIds=[instance_id]
                )
        except Exception as exc:  # noqa: BLE001 - report bounded rollback failure
            errors.append(_error_label(f"instance:{instance_id}", exc))
    child_errors = _confirm_children_deleted(
        ec2,
        volume_ids=[volume_id] if VOLUME_ID_RE.fullmatch(volume_id) else [],
        network_interface_ids=(
            [network_interface_id]
            if NETWORK_INTERFACE_ID_RE.fullmatch(network_interface_id)
            else []
        ),
    )
    errors.extend(child_errors)
    if SECURITY_GROUP_ID_RE.fullmatch(group_id):
        try:
            values = ec2.describe_security_groups(GroupIds=[group_id]).get(
                "SecurityGroups", []
            )
            if len(values) > 1:
                raise TemporaryHostError("rollback security group is ambiguous")
            if values:
                identity = _owned_identity(values[0].get("Tags"))
                if identity is None or identity[:2] != (run_id, candidate_sha):
                    raise TemporaryHostError(
                        "rollback security-group ownership changed"
                    )
                _delete_security_group(ec2, group_id)
        except Exception as exc:  # noqa: BLE001 - report bounded rollback failure
            errors.append(_error_label(f"security-group:{group_id}", exc))
    residue: dict[str, list[str]] | dict[str, bool]
    try:
        instances, groups = _inventory(
            ec2,
            run_id=run_id,
            candidate_sha=candidate_sha,
        )
        residue = {
            key: values
            for key, values in {
                "instances": sorted(str(item["InstanceId"]) for item in instances),
                "security_groups": sorted(
                    str(item["GroupId"]) for item in groups
                ),
            }.items()
            if values
        }
        if child_errors:
            if VOLUME_ID_RE.fullmatch(volume_id):
                residue["volumes"] = [volume_id]
            if NETWORK_INTERFACE_ID_RE.fullmatch(network_interface_id):
                residue["network_interfaces"] = [network_interface_id]
    except Exception as exc:  # noqa: BLE001 - make unknown residue explicit
        errors.append(_error_label("rollback-inventory", exc))
        residue = {"inventory_unknown": True}
    return {
        "attempted": True,
        "cleanup_complete": not errors and not residue,
        "errors": sorted(set(errors)),
        "residue": residue,
        "instance_id": instance_id or None,
        "security_group_id": group_id or None,
        "volume_id": volume_id or None,
        "network_interface_id": network_interface_id or None,
        "delete_on_termination_confirmed": not child_errors,
        "account_id": account_id,
        "region": region,
    }


def create_host(
    *,
    ec2: Any,
    ssm: Any,
    account_id: str,
    region: str,
    run_id: str,
    candidate_sha: str,
    ttl_hours: int,
    now: datetime,
) -> dict[str, Any]:
    if (
        account_id != ACCOUNT_ID
        or region != REGION
        or RUN_RE.fullmatch(run_id) is None
        or SHA_RE.fullmatch(candidate_sha) is None
        or ttl_hours not in range(MIN_TTL_HOURS, MAX_TTL_HOURS + 1)
    ):
        raise TemporaryHostError("temporary host inputs are invalid")
    created_at = _utc(now)
    expires_at = created_at + timedelta(hours=ttl_hours)
    tags = _tags(
        run_id=run_id,
        candidate_sha=candidate_sha,
        expires_at=expires_at,
    )
    journal: dict[str, str] = {}
    try:
        security_group = ec2.create_security_group(
            GroupName=f"leadpoet-parity-{run_id}-testnet",
            Description="Temporary Leadpoet testnet signing host; no ingress",
            VpcId=VPC_ID,
            TagSpecifications=[{
                "ResourceType": "security-group",
                "Tags": tags,
            }],
        )
        group_id = str(security_group.get("GroupId") or "")
        journal["security_group_id"] = group_id
        if SECURITY_GROUP_ID_RE.fullmatch(group_id) is None:
            raise TemporaryHostError("temporary security-group identity is invalid")

        launched = ec2.run_instances(
            ImageId=AMI_ID,
            InstanceType=INSTANCE_TYPE,
            MinCount=1,
            MaxCount=1,
            NetworkInterfaces=[{
                "DeviceIndex": 0,
                "SubnetId": SUBNET_ID,
                "Groups": [group_id],
                "AssociatePublicIpAddress": True,
                "DeleteOnTermination": True,
            }],
            IamInstanceProfile={"Name": INSTANCE_PROFILE},
            EnclaveOptions={"Enabled": True},
            MetadataOptions={
                "HttpEndpoint": "enabled",
                "HttpTokens": "required",
                "HttpPutResponseHopLimit": 2,
                "InstanceMetadataTags": "enabled",
            },
            BlockDeviceMappings=[{
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "Encrypted": True,
                    "VolumeSize": VOLUME_GIB,
                    "VolumeType": "gp3",
                },
            }],
            InstanceInitiatedShutdownBehavior="terminate",
            UserData=EARLY_BOOT_ISOLATION.replace(
                "{expiry_epoch}", str(int(expires_at.timestamp()))
            ),
            TagSpecifications=[
                {"ResourceType": kind, "Tags": tags}
                for kind in ("instance", "volume", "network-interface")
            ],
        )
        values = launched.get("Instances", [])
        if len(values) != 1:
            raise TemporaryHostError("temporary launch returned invalid inventory")
        instance_id = str(values[0].get("InstanceId") or "")
        journal["instance_id"] = instance_id
        if INSTANCE_ID_RE.fullmatch(instance_id) is None:
            raise TemporaryHostError("temporary instance identity is invalid")
        ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
        response = ec2.describe_instances(InstanceIds=[instance_id])
        hosts = [
            item
            for reservation in response.get("Reservations", [])
            for item in reservation.get("Instances", [])
        ]
        if len(hosts) != 1:
            raise TemporaryHostError("temporary instance readback is invalid")
        host = hosts[0]
        if (
            host.get("InstanceId") != instance_id
            or host.get("ImageId") != AMI_ID
            or host.get("InstanceType") != INSTANCE_TYPE
            or host.get("SubnetId") != SUBNET_ID
            or host.get("VpcId") != VPC_ID
            or host.get("EnclaveOptions", {}).get("Enabled") is not True
            or host.get("MetadataOptions", {}).get("HttpTokens") != "required"
            or _owned_identity(host.get("Tags"))
            != (run_id, candidate_sha, expires_at.replace(microsecond=0))
        ):
            raise TemporaryHostError("temporary instance readback differs")
        volume_id, network_interface_id = _child_ids(host)
        journal["volume_id"] = volume_id
        journal["network_interface_id"] = network_interface_id
        volumes = ec2.describe_volumes(VolumeIds=[volume_id]).get("Volumes", [])
        if (
            len(volumes) != 1
            or volumes[0].get("VolumeId") != volume_id
            or volumes[0].get("Encrypted") is not True
            or volumes[0].get("Size") != VOLUME_GIB
            or volumes[0].get("VolumeType") != "gp3"
        ):
            raise TemporaryHostError("temporary root-volume readback differs")
        groups = ec2.describe_security_groups(GroupIds=[group_id]).get(
            "SecurityGroups", []
        )
        if (
            len(groups) != 1
            or groups[0].get("GroupId") != group_id
            or groups[0].get("VpcId") != VPC_ID
            or groups[0].get("IpPermissions") not in (None, [])
            or _owned_identity(groups[0].get("Tags"))
            != (run_id, candidate_sha, expires_at.replace(microsecond=0))
        ):
            raise TemporaryHostError("temporary security-group readback differs")
        _wait_ssm_online(ssm, instance_id)
        _verify_expiry_timer(
            ssm,
            instance_id=instance_id,
            expiry_epoch=int(expires_at.timestamp()),
        )
    except Exception as exc:
        rollback = _rollback(
            ec2,
            account_id=account_id,
            region=region,
            run_id=run_id,
            candidate_sha=candidate_sha,
            journal=journal,
        )
        raise TemporaryHostCreateError({
            "schema_version": SCHEMA_VERSION,
            "status": (
                "provision_failed_cleanup_complete"
                if rollback["cleanup_complete"]
                else "provision_failed_cleanup_incomplete"
            ),
            "account_id": account_id,
            "region": region,
            "run_id": run_id,
            "candidate_sha": candidate_sha,
            "function": FUNCTION,
            "created_at": created_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "instance_id": rollback["instance_id"],
            "security_group_id": rollback["security_group_id"],
            "volume_id": rollback["volume_id"],
            "network_interface_id": rollback["network_interface_id"],
            "failure": _error_label("provision", exc),
            "rollback": rollback,
        }) from exc

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "account_id": account_id,
        "region": region,
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "function": FUNCTION,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "instance_id": instance_id,
        "security_group_id": group_id,
        "volume_id": volume_id,
        "network_interface_id": network_interface_id,
        "instance_type": INSTANCE_TYPE,
        "volume_gib": VOLUME_GIB,
        "nitro_enclaves_enabled": True,
        "ssm_online": True,
        "inbound_rules": 0,
        "required_combined_allocator": {
            "cpu_count": COMBINED_ALLOCATOR_CPUS,
            "memory_mib": COMBINED_ALLOCATOR_MEMORY_MIB,
        },
    }


def _inventory(
    ec2: Any,
    *,
    run_id: str | None = None,
    candidate_sha: str | None = None,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    exact = run_id is not None or candidate_sha is not None
    if exact and (
        not isinstance(run_id, str)
        or RUN_RE.fullmatch(run_id) is None
        or not isinstance(candidate_sha, str)
        or SHA_RE.fullmatch(candidate_sha) is None
    ):
        raise TemporaryHostError("exact inventory inputs are invalid")
    filters = [{"Name": f"tag:{TAG_EPHEMERAL}", "Values": ["true"]}]
    if exact:
        filters.extend([
            {"Name": f"tag:{TAG_RUN}", "Values": [run_id]},
            {"Name": f"tag:{TAG_SHA}", "Values": [candidate_sha]},
        ])

    def owned(tags: Any) -> bool:
        identity = _owned_identity(tags)
        return identity is not None and (
            not exact or identity[:2] == (run_id, candidate_sha)
        )

    instance_pages = _pages(ec2.describe_instances, Filters=filters)
    instances = [
        item
        for page in instance_pages
        for reservation in page.get("Reservations", [])
        for item in reservation.get("Instances", [])
        if owned(item.get("Tags"))
        and item.get("State", {}).get("Name") not in {"terminated", "shutting-down"}
    ]
    group_pages = _pages(ec2.describe_security_groups, Filters=filters)
    groups = [
        item
        for page in group_pages
        for item in page.get("SecurityGroups", [])
        if owned(item.get("Tags"))
        and item.get("VpcId") == VPC_ID
    ]
    instance_ids = [str(item.get("InstanceId") or "") for item in instances]
    group_ids = [str(item.get("GroupId") or "") for item in groups]
    if (
        any(INSTANCE_ID_RE.fullmatch(value) is None for value in instance_ids)
        or len(instance_ids) != len(set(instance_ids))
        or any(
            SECURITY_GROUP_ID_RE.fullmatch(value) is None for value in group_ids
        )
        or len(group_ids) != len(set(group_ids))
    ):
        raise TemporaryHostError("temporary inventory identity is invalid")
    return instances, groups


def cleanup_hosts(
    *,
    ec2: Any,
    account_id: str,
    region: str,
    now: datetime,
    apply: bool,
    run_id: str | None = None,
    candidate_sha: str | None = None,
) -> dict[str, Any]:
    if account_id != ACCOUNT_ID or region != REGION:
        raise TemporaryHostError("temporary cleanup AWS identity is invalid")
    exact = run_id is not None or candidate_sha is not None
    if exact and (
        not isinstance(run_id, str)
        or RUN_RE.fullmatch(run_id) is None
        or not isinstance(candidate_sha, str)
        or SHA_RE.fullmatch(candidate_sha) is None
    ):
        raise TemporaryHostError("exact temporary cleanup inputs are invalid")
    current = _utc(now)
    instances, groups = _inventory(
        ec2,
        run_id=run_id if exact else None,
        candidate_sha=candidate_sha if exact else None,
    )

    def selected(resource: Mapping[str, Any]) -> bool:
        identity = _owned_identity(resource.get("Tags"))
        if identity is None:
            return False
        owner_run, owner_sha, expiry = identity
        if exact:
            return owner_run == run_id and owner_sha == candidate_sha
        return expiry <= current

    selected_instances = [item for item in instances if selected(item)]
    instance_ids = sorted(str(item["InstanceId"]) for item in selected_instances)
    group_ids = sorted(
        str(item["GroupId"]) for item in groups if selected(item)
    )
    errors: list[str] = []
    child_ids: dict[str, tuple[str, str]] = {}
    for item in selected_instances:
        instance_id = str(item.get("InstanceId") or "")
        try:
            child_ids[instance_id] = _child_ids(item)
        except Exception as exc:  # noqa: BLE001 - preserve cleanup receipt
            errors.append(_error_label(f"children:{instance_id}", exc))
    volume_ids = sorted({value[0] for value in child_ids.values()})
    network_interface_ids = sorted({value[1] for value in child_ids.values()})
    result = {
        "schema_version": "leadpoet.temporary_testnet_signing_cleanup.v1",
        "mode": "apply" if apply else "dry-run",
        "selection": "exact-run" if exact else "expired",
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "instances": instance_ids,
        "security_groups": group_ids,
        "volumes": volume_ids,
        "network_interfaces": network_interface_ids,
        "delete_on_termination_confirmed": False,
        "errors": errors,
        "residue": {},
    }
    if not apply:
        return result
    terminated: list[str] = []
    for instance_id in instance_ids:
        if instance_id not in child_ids:
            continue
        try:
            live = _instance_by_id(ec2, instance_id)
            if live is None:
                continue
            if not selected(live):
                raise TemporaryHostError("temporary instance ownership changed")
            if _child_ids(live) != child_ids[instance_id]:
                raise TemporaryHostError("temporary child-resource identity changed")
            if live.get("State", {}).get("Name") not in {
                "terminated",
                "shutting-down",
            }:
                ec2.terminate_instances(InstanceIds=[instance_id])
                terminated.append(instance_id)
        except Exception as exc:  # noqa: BLE001 - preserve cleanup receipt
            errors.append(_error_label(f"instance:{instance_id}", exc))
    if terminated:
        try:
            ec2.get_waiter("instance_terminated").wait(InstanceIds=terminated)
        except Exception as exc:  # noqa: BLE001 - preserve cleanup receipt
            errors.append(_error_label("instances-terminate", exc))
    child_errors = _confirm_children_deleted(
        ec2,
        volume_ids=[child_ids[item][0] for item in terminated],
        network_interface_ids=[child_ids[item][1] for item in terminated],
    )
    errors.extend(child_errors)
    result["delete_on_termination_confirmed"] = not child_errors
    for group_id in group_ids:
        try:
            live = ec2.describe_security_groups(GroupIds=[group_id]).get(
                "SecurityGroups", []
            )
            if not live:
                continue
            if len(live) != 1 or not selected(live[0]):
                raise TemporaryHostError(
                    "temporary security-group ownership changed"
                )
            _delete_security_group(ec2, group_id)
        except Exception as exc:  # noqa: BLE001 - preserve cleanup receipt
            errors.append(_error_label(f"security-group:{group_id}", exc))
    try:
        remaining_instances, remaining_groups = _inventory(
            ec2,
            run_id=run_id if exact else None,
            candidate_sha=candidate_sha if exact else None,
        )
        residue = {
            "instances": sorted(
                str(item["InstanceId"])
                for item in remaining_instances
                if selected(item)
            ),
            "security_groups": sorted(
                str(item["GroupId"])
                for item in remaining_groups
                if selected(item)
            ),
        }
    except Exception as exc:  # noqa: BLE001 - make unknown residue explicit
        errors.append(_error_label("cleanup-inventory", exc))
        residue = {"inventory_unknown": True}
    if child_errors:
        residue["volumes"] = [child_ids[item][0] for item in terminated]
        residue["network_interfaces"] = [
            child_ids[item][1] for item in terminated
        ]
    result["errors"] = sorted(set(errors))
    result["residue"] = {key: value for key, value in residue.items() if value}
    return result


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--run-id", required=True)
    create.add_argument("--candidate-sha", required=True)
    create.add_argument("--ttl-hours", type=int, default=DEFAULT_TTL_HOURS)
    create.add_argument("--state", type=Path, required=True)
    cleanup_run = commands.add_parser("cleanup-run")
    cleanup_run.add_argument("--run-id", required=True)
    cleanup_run.add_argument("--candidate-sha", required=True)
    cleanup_run.add_argument("--apply", action="store_true")
    cleanup_expired = commands.add_parser("cleanup-expired")
    cleanup_expired.add_argument("--apply", action="store_true")
    create_assets = commands.add_parser("create-assets")
    create_assets.add_argument("--run-id", required=True)
    create_assets.add_argument("--candidate-sha", required=True)
    create_assets.add_argument("--bundle", type=Path, required=True)
    create_assets.add_argument("--binding", type=Path, required=True)
    create_assets.add_argument("--prior-release-run-id")
    create_assets.add_argument("--prior-release-commit")
    create_assets.add_argument("--state", type=Path, required=True)
    wait_assets = commands.add_parser("wait-assets")
    wait_assets.add_argument("--run-id", required=True)
    wait_assets.add_argument("--candidate-sha", required=True)
    wait_assets.add_argument("--timeout-seconds", type=int, default=1800)
    wait_assets.add_argument("--state", type=Path, required=True)
    cleanup_assets = commands.add_parser("cleanup-assets")
    cleanup_assets.add_argument("--run-id", required=True)
    cleanup_assets.add_argument("--candidate-sha", required=True)
    cleanup_assets.add_argument("--state", type=Path, required=True)
    source_bootstrap = commands.add_parser("ssm-source-bootstrap")
    source_bootstrap.add_argument("--run-id", required=True)
    source_bootstrap.add_argument("--candidate-sha", required=True)
    source_bootstrap.add_argument("--instance-id", required=True)
    source_bootstrap.add_argument("--assets-bucket", required=True)
    source_bootstrap.add_argument("--assets-prefix", required=True)
    source_bootstrap.add_argument("--prior-release-commit")
    source_bootstrap.add_argument("--state", type=Path, required=True)
    export_release = commands.add_parser("ssm-export-public-release")
    export_release.add_argument("--run-id", required=True)
    export_release.add_argument("--candidate-sha", required=True)
    export_release.add_argument("--instance-id", required=True)
    export_release.add_argument("--state", type=Path, required=True)
    native_stage = commands.add_parser("ssm-native-stage")
    native_stage.add_argument("--run-id", required=True)
    native_stage.add_argument("--candidate-sha", required=True)
    native_stage.add_argument("--instance-id", required=True)
    native_stage.add_argument("--stage", choices=sorted(NATIVE_STAGES), required=True)
    native_stage.add_argument("--state", type=Path, required=True)
    restart_gateway = commands.add_parser("ssm-restart-testnet401-gateway-network")
    restart_gateway.add_argument("--run-id", required=True)
    restart_gateway.add_argument("--candidate-sha", required=True)
    restart_gateway.add_argument("--instance-id", required=True)
    restart_gateway.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        session = boto3.session.Session(region_name=args.region)
        account_id = str(session.client("sts").get_caller_identity()["Account"])
        ec2 = session.client("ec2")
        now = datetime.now(timezone.utc).replace(microsecond=0)
        candidate_sha = (
            args.candidate_sha.lower()
            if hasattr(args, "candidate_sha")
            else None
        )
        if args.command == "create-assets":
            result = create_asset_bucket(
                s3=session.client("s3"),
                account_id=account_id,
                region=args.region,
                run_id=args.run_id,
                candidate_sha=candidate_sha,
                bundle_path=args.bundle,
                binding_path=args.binding,
                prior_release_run_id=args.prior_release_run_id,
                prior_release_commit=args.prior_release_commit,
            )
            _write(args.state, result)
        elif args.command == "wait-assets":
            result = wait_for_private_assets(
                s3=session.client("s3"),
                account_id=account_id,
                region=args.region,
                run_id=args.run_id,
                candidate_sha=candidate_sha,
                timeout_seconds=args.timeout_seconds,
            )
            _write(args.state, result)
        elif args.command == "cleanup-assets":
            result = remove_staging_asset_heads(
                s3=session.client("s3"),
                account_id=account_id,
                region=args.region,
                run_id=args.run_id,
                candidate_sha=candidate_sha,
            )
            _write(args.state, result)
        elif args.command == "create":
            result = create_host(
                ec2=ec2,
                ssm=session.client("ssm"),
                account_id=account_id,
                region=args.region,
                run_id=args.run_id,
                candidate_sha=candidate_sha,
                ttl_hours=args.ttl_hours,
                now=now,
            )
            _write(args.state, result)
        elif args.command == "ssm-export-public-release":
            result = export_public_release_documents(
                ec2=ec2,
                ssm=session.client("ssm"),
                s3=session.client("s3"),
                account_id=account_id,
                region=args.region,
                run_id=args.run_id,
                candidate_sha=candidate_sha,
                instance_id=args.instance_id,
                now=now,
            )
            _write(args.state, result)
        elif args.command == "ssm-source-bootstrap":
            result = run_source_bootstrap(
                ec2=ec2,
                ssm=session.client("ssm"),
                account_id=account_id,
                region=args.region,
                run_id=args.run_id,
                candidate_sha=candidate_sha,
                instance_id=args.instance_id,
                assets_bucket=args.assets_bucket,
                assets_prefix=args.assets_prefix,
                now=now,
                prior_release_commit=args.prior_release_commit,
            )
            _write(args.state, result)
        elif args.command == "ssm-native-stage":
            result = run_native_stage(
                ec2=ec2,
                ssm=session.client("ssm"),
                account_id=account_id,
                region=args.region,
                run_id=args.run_id,
                candidate_sha=candidate_sha,
                instance_id=args.instance_id,
                stage=args.stage,
                now=now,
            )
            _write(args.state, result)
        elif args.command == "ssm-restart-testnet401-gateway-network":
            result = run_gateway_network_restart(
                ec2=ec2,
                ssm=session.client("ssm"),
                account_id=account_id,
                region=args.region,
                run_id=args.run_id,
                candidate_sha=candidate_sha,
                instance_id=args.instance_id,
                now=now,
            )
            _write(args.state, result)
        else:
            result = cleanup_hosts(
                ec2=ec2,
                account_id=account_id,
                region=args.region,
                now=now,
                apply=args.apply,
                run_id=args.run_id if args.command == "cleanup-run" else None,
                candidate_sha=(
                    candidate_sha
                    if args.command == "cleanup-run"
                    else None
                ),
            )
    except TemporaryHostCreateError as exc:
        _write(args.state, exc.receipt)
        print(json.dumps(exc.receipt, sort_keys=True))
        print("ERROR: temporary host provisioning failed; see receipt", file=sys.stderr)
        return 1
    except (BotoCoreError, ClientError, OSError, TemporaryHostError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    if result.get("errors") or result.get("residue"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
