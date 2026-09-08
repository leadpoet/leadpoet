#!/usr/bin/env python3
"""Provision and remove one short-lived Nitro host for testnet signing proof."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
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
PRIVATE_ASSET_NAMES = (
    "validator.env",
    "testnet-hotkey-config.json",
    "testnet-hotkey-envelope.json",
    "testnet401-epoch-cutover.json",
)
SOURCE_ASSET_NAMES = ("candidate.bundle", "candidate-bundle-binding.json")
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


def create_asset_bucket(
    *,
    s3: Any,
    account_id: str,
    region: str,
    run_id: str,
    candidate_sha: str,
    bundle_path: Path,
    binding_path: Path,
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
    for name in (*SOURCE_ASSET_NAMES, *PRIVATE_ASSET_NAMES):
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
            raise TemporaryHostError("fixed temporary SSM stage failed")
        return stdout
    raise TemporaryHostError("fixed temporary SSM stage timed out")


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
        "if [ ! -x /usr/bin/git ]; then /usr/bin/dnf -q -y install git-core >/dev/null 2>&1; fi",
        f"install -d -m 0700 {q(str(Path(SOURCE_REPOSITORY).parent))} {q(SOURCE_REPOSITORY)}",
        f"/usr/bin/git -C {q(SOURCE_REPOSITORY)} init >/dev/null 2>&1",
        f"/usr/bin/git -C {q(SOURCE_REPOSITORY)} fetch --no-tags {q(bundle)} HEAD >/dev/null 2>&1",
        f"test \"$(/usr/bin/git -C {q(SOURCE_REPOSITORY)} rev-parse FETCH_HEAD)\" = {q(candidate_sha)}",
        f"/usr/bin/git -C {q(SOURCE_REPOSITORY)} checkout --detach {q(candidate_sha)} >/dev/null 2>&1",
        f"/usr/bin/git -C {q(SOURCE_REPOSITORY)} remote add origin https://github.com/leadpoet/leadpoet.git",
        f"/usr/bin/git -C {q(SOURCE_REPOSITORY)} fetch --no-tags origin "
        "refs/heads/main:refs/remotes/origin/main >/dev/null 2>&1",
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
        command = (
            "set -Eeuo pipefail\n"
            f"if [ ! -f {shlex.quote(NATIVE_CONFIG)} ]; then\n"
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
    return {
        "schema_version": "leadpoet.temporary_testnet401_native_ssm.v1",
        "stage": stage,
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "instance_id": instance_id,
        "ssm_command_id": command_id,
        "receipt": dict(receipt),
    }


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
        "import json, pathlib, re",
        f"result = {identity!r}",
        f"root = pathlib.Path({RUNTIME_ROOT!r})",
        "names = ('early-boot-isolated', 'expires-epoch', 'candidate.bundle', "
        "'candidate-bundle-binding.json', 'requirements.txt', 'source-stage.json', "
        "'config.json', 'inputs', 'staging-logs')",
        "result['paths_present'] = {name: (root / name).exists() for name in names}",
        f"result['repository_exists'] = pathlib.Path({SOURCE_REPOSITORY!r}).exists()",
        f"result['venv_exists'] = pathlib.Path({SOURCE_VENV!r}).exists()",
        "result['stage_states'] = []",
        "path = root / 'source-stage.json'",
        "if path.is_file() and not path.is_symlink():",
        "    for line in path.open().read(65536).splitlines()[-20:]:",
        "        try: value = json.loads(line)",
        "        except ValueError: continue",
        "        if isinstance(value, dict) and re.fullmatch('[a-z_]{1,64}', "
        "str(value.get('stage', ''))) and value.get('status') in ('running', 'passed'):",
        "            result['stage_states'].append({key: value[key] for key in ('stage', 'status')})",
        "patterns = ('AccessDenied', 'ModuleNotFoundError', 'ImportError', "
        "'PermissionError', 'NoSuchKey', 'No space left on device', 'AssertionError', "
        "'RuntimeError', 'ValueError', 'command not found', 'not found', 'fatal:', "
        "'ResolutionImpossible', 'No matching distribution', 'FileExistsError')",
        "result['log_diagnostics'] = []",
        f"ssm = pathlib.Path('/var/lib/amazon/ssm/{instance_id}/document/orchestration')",
        "paths = list(ssm.glob('*/awsrunShellScript/0.awsrunShellScript/stderr'))[-12:]",
        "paths += list((root / 'staging-logs').glob('*.log'))[:16]",
        "for path in paths:",
        "    if path.is_symlink() or not path.is_file(): continue",
        "    with path.open('rb') as stream:",
        "        stream.seek(max(0, path.stat().st_size - 65536))",
        "        data = stream.read(65536).decode('utf-8', 'replace')",
        "    result['log_diagnostics'].append({'file': path.name, "
        "'bytes': path.stat().st_size, 'categories': [p for p in patterns if p in data], "
        "'trace_locations': re.findall(r'File \"[^\"\\n]*/([a-zA-Z0-9_]+\\.py)\", line ([0-9]{1,6})', data)[-8:]})",
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
    source_bootstrap.add_argument("--state", type=Path, required=True)
    native_stage = commands.add_parser("ssm-native-stage")
    native_stage.add_argument("--run-id", required=True)
    native_stage.add_argument("--candidate-sha", required=True)
    native_stage.add_argument("--instance-id", required=True)
    native_stage.add_argument("--stage", choices=sorted(NATIVE_STAGES), required=True)
    native_stage.add_argument("--state", type=Path, required=True)
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
