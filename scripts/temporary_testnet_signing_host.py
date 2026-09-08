#!/usr/bin/env python3
"""Provision and remove one short-lived Nitro host for testnet signing proof."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError


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
EARLY_BOOT_ISOLATION = """#cloud-boothook
#!/bin/bash
set -eu
for unit in $(systemctl list-unit-files --no-legend 2>/dev/null \
  | awk '$1 ~ /(leadpoet|research-lab|gateway|validator)/ {print $1}'); do
  systemctl mask --now "$unit" >/dev/null 2>&1 || true
done
install -d -m 0700 /run/leadpoet-temporary-testnet
printf '%s\n' isolated >/run/leadpoet-temporary-testnet/early-boot-isolated
"""


class TemporaryHostError(RuntimeError):
    """The temporary host operation was not safely bounded."""


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise TemporaryHostError("time must include a timezone")
    return value.astimezone(timezone.utc)


def _tag_map(tags: Any) -> dict[str, str]:
    if not isinstance(tags, list):
        return {}
    return {
        str(item.get("Key") or ""): str(item.get("Value") or "")
        for item in tags
        if isinstance(item, Mapping) and item.get("Key")
    }


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


def _rollback(ec2: Any, journal: Mapping[str, str]) -> None:
    instance_id = str(journal.get("instance_id") or "")
    group_id = str(journal.get("security_group_id") or "")
    if INSTANCE_ID_RE.fullmatch(instance_id):
        try:
            ec2.terminate_instances(InstanceIds=[instance_id])
            ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
        except Exception:  # noqa: BLE001 - preserve the provisioning failure
            pass
    if SECURITY_GROUP_ID_RE.fullmatch(group_id):
        try:
            _delete_security_group(ec2, group_id)
        except Exception:  # noqa: BLE001 - scheduled cleanup remains the backstop
            pass


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
            UserData=EARLY_BOOT_ISOLATION,
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
    except Exception:
        _rollback(ec2, journal)
        raise

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


def _inventory(ec2: Any) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    response = ec2.describe_instances(
        Filters=[{"Name": f"tag:{TAG_EPHEMERAL}", "Values": ["true"]}]
    )
    instances = [
        item
        for reservation in response.get("Reservations", [])
        for item in reservation.get("Instances", [])
        if _owned_identity(item.get("Tags")) is not None
        and item.get("State", {}).get("Name") not in {"terminated", "shutting-down"}
    ]
    groups = [
        item
        for item in ec2.describe_security_groups(
            Filters=[{"Name": f"tag:{TAG_EPHEMERAL}", "Values": ["true"]}]
        ).get("SecurityGroups", [])
        if _owned_identity(item.get("Tags")) is not None
        and item.get("VpcId") == VPC_ID
    ]
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
    instances, groups = _inventory(ec2)

    def selected(resource: Mapping[str, Any]) -> bool:
        identity = _owned_identity(resource.get("Tags"))
        assert identity is not None
        owner_run, owner_sha, expiry = identity
        if exact:
            return owner_run == run_id and owner_sha == candidate_sha
        return expiry <= current

    instance_ids = sorted(
        str(item["InstanceId"]) for item in instances if selected(item)
    )
    group_ids = sorted(
        str(item["GroupId"]) for item in groups if selected(item)
    )
    result = {
        "schema_version": "leadpoet.temporary_testnet_signing_cleanup.v1",
        "mode": "apply" if apply else "dry-run",
        "selection": "exact-run" if exact else "expired",
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "instances": instance_ids,
        "security_groups": group_ids,
        "residue": {},
    }
    if not apply:
        return result
    for instance_id in instance_ids:
        ec2.terminate_instances(InstanceIds=[instance_id])
    if instance_ids:
        ec2.get_waiter("instance_terminated").wait(InstanceIds=instance_ids)
    for group_id in group_ids:
        live = ec2.describe_security_groups(GroupIds=[group_id]).get(
            "SecurityGroups", []
        )
        if len(live) != 1 or not selected(live[0]):
            raise TemporaryHostError("temporary security-group ownership changed")
        _delete_security_group(ec2, group_id)
    remaining_instances, remaining_groups = _inventory(ec2)
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
    args = parser.parse_args(argv)
    try:
        session = boto3.session.Session(region_name=args.region)
        account_id = str(session.client("sts").get_caller_identity()["Account"])
        ec2 = session.client("ec2")
        now = datetime.now(timezone.utc).replace(microsecond=0)
        if args.command == "create":
            result = create_host(
                ec2=ec2,
                ssm=session.client("ssm"),
                account_id=account_id,
                region=args.region,
                run_id=args.run_id,
                candidate_sha=args.candidate_sha.lower(),
                ttl_hours=args.ttl_hours,
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
                    args.candidate_sha.lower()
                    if args.command == "cleanup-run"
                    else None
                ),
            )
    except (BotoCoreError, ClientError, OSError, TemporaryHostError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    if result.get("residue"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
