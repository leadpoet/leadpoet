#!/usr/bin/env python3
"""Provision one disposable Nitro host for full production-parity validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError


RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
INSTANCE_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,31}$")
IP_RE = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
SCHEMA_VERSION = "leadpoet.production_parity_ephemeral_stack.v2"
TAG_RUN = "leadpoet:parity-run"
TAG_SHA = "leadpoet:candidate-sha"
TAG_EPHEMERAL = "leadpoet:ephemeral"
ARTIFACT_RETENTION_DAYS = 1
EARLY_BOOT_ISOLATION = """#cloud-boothook
#!/bin/bash
set -eu
for unit in $(systemctl list-unit-files --no-legend 2>/dev/null \
  | awk '$1 ~ /(leadpoet|research-lab|gateway|validator)/ {print $1}'); do
  systemctl mask --now "$unit" >/dev/null 2>&1 || true
done
install -d -m 0700 /run/leadpoet-production-parity
printf '%s\n' isolated >/run/leadpoet-production-parity/early-boot-isolated
"""


class ProvisioningError(RuntimeError):
    """A transient stack could not be proven bounded to one candidate."""


def _tags(run_id: str, candidate_sha: str) -> list[dict[str, str]]:
    return [
        {"Key": TAG_RUN, "Value": run_id},
        {"Key": TAG_SHA, "Value": candidate_sha},
        {"Key": TAG_EPHEMERAL, "Value": "true"},
        {"Key": "Name", "Value": f"leadpoet-parity-{run_id}"},
    ]


def _artifact_bucket_name(
    *, account_id: str, run_id: str, candidate_sha: str
) -> str:
    if not re.fullmatch(r"^[0-9]{12}$", account_id):
        raise ProvisioningError("AWS account identity is invalid")
    suffix = hashlib.sha256(
        f"{account_id}:{run_id}:{candidate_sha}".encode("ascii")
    ).hexdigest()[:16]
    return f"leadpoet-parity-{account_id}-{suffix}"


def _create_artifact_bucket(
    s3: Any,
    *,
    region: str,
    account_id: str,
    run_id: str,
    candidate_sha: str,
) -> str:
    bucket = _artifact_bucket_name(
        account_id=account_id,
        run_id=run_id,
        candidate_sha=candidate_sha,
    )
    create: dict[str, Any] = {
        "Bucket": bucket,
        "ObjectLockEnabledForBucket": True,
    }
    if region != "us-east-1":
        create["CreateBucketConfiguration"] = {
            "LocationConstraint": region
        }
    created = False
    try:
        s3.create_bucket(**create)
        created = True
        s3.put_bucket_tagging(
            Bucket=bucket,
            Tagging={"TagSet": _tags(run_id, candidate_sha)},
        )
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        s3.put_bucket_encryption(
            Bucket=bucket,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        },
                        "BucketKeyEnabled": False,
                    }
                ]
            },
        )
        s3.put_bucket_versioning(
            Bucket=bucket,
            VersioningConfiguration={"Status": "Enabled"},
        )
        s3.put_object_lock_configuration(
            Bucket=bucket,
            ObjectLockConfiguration={
                "ObjectLockEnabled": "Enabled",
                "Rule": {
                    "DefaultRetention": {
                        "Mode": "COMPLIANCE",
                        "Days": ARTIFACT_RETENTION_DAYS,
                    }
                },
            },
        )
    except Exception:
        if created:
            try:
                s3.delete_bucket(Bucket=bucket)
            except Exception:
                pass
        raise
    return bucket


def _single_production_instance(ec2: Any, public_ip: str) -> Mapping[str, Any]:
    if not IP_RE.fullmatch(public_ip):
        raise ProvisioningError("production gateway address is invalid")
    response = ec2.describe_instances(
        Filters=[
            {"Name": "ip-address", "Values": [public_ip]},
            {"Name": "instance-state-name", "Values": ["running"]},
        ]
    )
    instances = [
        instance
        for reservation in response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    if len(instances) != 1:
        raise ProvisioningError(
            "production gateway address did not resolve one running instance"
        )
    instance = instances[0]
    if (
        instance.get("EnclaveOptions", {}).get("Enabled") is not True
        or instance.get("MetadataOptions", {}).get("HttpTokens") != "required"
        or not instance.get("ImageId")
        or not instance.get("SubnetId")
        or not instance.get("VpcId")
    ):
        raise ProvisioningError(
            "production gateway is not a complete Nitro/IMDSv2 reference"
        )
    return instance


def _cloudfront_prefix_list(ec2: Any) -> str:
    response = ec2.describe_managed_prefix_lists(
        Filters=[
            {
                "Name": "prefix-list-name",
                "Values": ["com.amazonaws.global.cloudfront.origin-facing"],
            }
        ]
    )
    values = [
        str(item.get("PrefixListId") or "")
        for item in response.get("PrefixLists", [])
        if item.get("State") in (None, "create-complete", "modify-complete")
    ]
    if len(values) != 1 or not values[0].startswith("pl-"):
        raise ProvisioningError("CloudFront origin prefix list is unavailable")
    return values[0]


def _managed_policy_id(client: Any, *, policy_type: str, name: str) -> str:
    paginator = client.get_paginator(f"list_{policy_type}_policies")
    key = "CachePolicyList" if policy_type == "cache" else "OriginRequestPolicyList"
    for page in paginator.paginate(Type="managed"):
        for item in page.get(key, {}).get("Items", []):
            policy = item.get(
                "CachePolicy" if policy_type == "cache" else "OriginRequestPolicy",
                {},
            )
            config = policy.get(
                "CachePolicyConfig"
                if policy_type == "cache"
                else "OriginRequestPolicyConfig",
                {},
            )
            if config.get("Name") == name:
                value = str(policy.get("Id") or "")
                if value:
                    return value
    raise ProvisioningError(f"CloudFront managed policy is unavailable: {name}")


def _wait_ssm_online(ssm: Any, instance_id: str, *, timeout_seconds: int = 600) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
        )
        values = response.get("InstanceInformationList", [])
        if len(values) == 1 and values[0].get("PingStatus") == "Online":
            return
        time.sleep(5)
    raise ProvisioningError("ephemeral parity host did not become SSM-online")


def _wait_distribution(
    cloudfront: Any, distribution_id: str, *, enabled: bool, timeout_seconds: int = 1200
) -> Mapping[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = cloudfront.get_distribution(Id=distribution_id)["Distribution"]
        observed_enabled = bool(value.get("DistributionConfig", {}).get("Enabled"))
        if value.get("Status") == "Deployed" and observed_enabled is enabled:
            return value
        time.sleep(10)
    raise ProvisioningError("CloudFront parity boundary did not converge")


def _create_stack_resources(
    *,
    ec2: Any,
    ssm: Any,
    cloudfront: Any,
    s3: Any,
    region: str,
    account_id: str,
    run_id: str,
    candidate_sha: str,
    production_gateway_ip: str,
    instance_profile_name: str,
    instance_type: str | None,
    volume_gib: int,
    journal: dict[str, str],
) -> dict[str, Any]:
    if (
        not RUN_RE.fullmatch(run_id)
        or not SHA_RE.fullmatch(candidate_sha)
        or not instance_profile_name
        or volume_gib != 512
    ):
        raise ProvisioningError("ephemeral stack inputs are invalid")
    reference = _single_production_instance(ec2, production_gateway_ip)
    selected_type = str(instance_type or reference.get("InstanceType") or "")
    if not INSTANCE_TYPE_RE.fullmatch(selected_type):
        raise ProvisioningError("ephemeral instance type is invalid")

    artifact_bucket = _create_artifact_bucket(
        s3,
        region=region,
        account_id=account_id,
        run_id=run_id,
        candidate_sha=candidate_sha,
    )
    journal["artifact_bucket"] = artifact_bucket
    security_group = ec2.create_security_group(
        GroupName=f"leadpoet-parity-{run_id}",
        Description=f"Disposable Leadpoet parity database origin {run_id}",
        VpcId=reference["VpcId"],
        TagSpecifications=[{"ResourceType": "security-group", "Tags": _tags(run_id, candidate_sha)}],
    )
    security_group_id = str(security_group["GroupId"])
    journal["security_group_id"] = security_group_id
    prefix_list_id = _cloudfront_prefix_list(ec2)
    ec2.authorize_security_group_ingress(
        GroupId=security_group_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 3000,
                "ToPort": 3000,
                "PrefixListIds": [{"PrefixListId": prefix_list_id}],
            }
        ],
    )

    launched = ec2.run_instances(
        ImageId=reference["ImageId"],
        InstanceType=selected_type,
        MinCount=1,
        MaxCount=1,
        NetworkInterfaces=[
            {
                "DeviceIndex": 0,
                "SubnetId": reference["SubnetId"],
                "Groups": [security_group_id],
                "AssociatePublicIpAddress": True,
                "DeleteOnTermination": True,
            }
        ],
        IamInstanceProfile={"Name": instance_profile_name},
        EnclaveOptions={"Enabled": True},
        MetadataOptions={
            "HttpEndpoint": "enabled",
            "HttpTokens": "required",
            "HttpPutResponseHopLimit": 2,
            "InstanceMetadataTags": "enabled",
        },
        BlockDeviceMappings=[
            {
                "DeviceName": str(reference.get("RootDeviceName") or "/dev/xvda"),
                "Ebs": {
                    "DeleteOnTermination": True,
                    "Encrypted": True,
                    "VolumeSize": volume_gib,
                    "VolumeType": "gp3",
                },
            }
        ],
        InstanceInitiatedShutdownBehavior="terminate",
        UserData=EARLY_BOOT_ISOLATION,
        TagSpecifications=[
            {"ResourceType": "instance", "Tags": _tags(run_id, candidate_sha)},
            {"ResourceType": "volume", "Tags": _tags(run_id, candidate_sha)},
        ],
    )
    values = launched.get("Instances", [])
    if len(values) != 1:
        raise ProvisioningError("ephemeral host launch returned an invalid inventory")
    instance_id = str(values[0].get("InstanceId") or "")
    journal["instance_id"] = instance_id
    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
    described = ec2.describe_instances(InstanceIds=[instance_id])
    host = described["Reservations"][0]["Instances"][0]
    public_dns = str(host.get("PublicDnsName") or "")
    if not public_dns:
        raise ProvisioningError("ephemeral parity host has no public origin address")
    _wait_ssm_online(ssm, instance_id)

    cache_policy_id = _managed_policy_id(
        cloudfront, policy_type="cache", name="Managed-CachingDisabled"
    )
    origin_policy_id = _managed_policy_id(
        cloudfront,
        policy_type="origin_request",
        name="Managed-AllViewerExceptHostHeader",
    )
    created = cloudfront.create_distribution(
        DistributionConfig={
            "CallerReference": f"{run_id}-{candidate_sha}",
            "Comment": f"Disposable Leadpoet parity {run_id}",
            "Enabled": True,
            "HttpVersion": "http2and3",
            "IsIPV6Enabled": True,
            "PriceClass": "PriceClass_100",
            "Origins": {
                "Quantity": 1,
                "Items": [
                    {
                        "Id": "parity-postgrest",
                        "DomainName": public_dns,
                        "ConnectionAttempts": 3,
                        "ConnectionTimeout": 10,
                        "CustomOriginConfig": {
                            "HTTPPort": 3000,
                            "HTTPSPort": 443,
                            "OriginProtocolPolicy": "http-only",
                            "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]},
                            "OriginReadTimeout": 60,
                            "OriginKeepaliveTimeout": 60,
                        },
                    }
                ],
            },
            "DefaultCacheBehavior": {
                "TargetOriginId": "parity-postgrest",
                "ViewerProtocolPolicy": "https-only",
                "AllowedMethods": {
                    "Quantity": 7,
                    "Items": ["GET", "HEAD", "OPTIONS", "PUT", "PATCH", "POST", "DELETE"],
                    "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]},
                },
                "Compress": False,
                "CachePolicyId": cache_policy_id,
                "OriginRequestPolicyId": origin_policy_id,
                "SmoothStreaming": False,
            },
            "ViewerCertificate": {"CloudFrontDefaultCertificate": True},
            "Restrictions": {
                "GeoRestriction": {"RestrictionType": "none", "Quantity": 0}
            },
        }
    )
    distribution = created["Distribution"]
    distribution_id = str(distribution["Id"])
    journal["cloudfront_distribution_id"] = distribution_id
    cloudfront.tag_resource(
        Resource=distribution["ARN"],
        Tags={"Items": _tags(run_id, candidate_sha)},
    )
    deployed = _wait_distribution(cloudfront, distribution_id, enabled=True)
    domain = str(deployed.get("DomainName") or "")
    if not domain.endswith(".cloudfront.net"):
        raise ProvisioningError("CloudFront parity domain is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "region": region,
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instance_id": instance_id,
        "security_group_id": security_group_id,
        "cloudfront_distribution_id": distribution_id,
        "cloudfront_distribution_arn": str(deployed.get("ARN") or ""),
        "supabase_origin": f"https://{domain}",
        "artifact_bucket": artifact_bucket,
        "artifact_retention_days": ARTIFACT_RETENTION_DAYS,
        "reference_image_id": str(reference["ImageId"]),
        "reference_instance_type": str(reference["InstanceType"]),
        "selected_instance_type": selected_type,
        "volume_gib": volume_gib,
    }


def _rollback_partial_stack(
    *,
    ec2: Any,
    cloudfront: Any,
    s3: Any,
    journal: Mapping[str, str],
) -> list[str]:
    """Best-effort immediate rollback; exact-tag cleanup remains the backstop."""

    errors: list[str] = []
    distribution_id = str(journal.get("cloudfront_distribution_id") or "")
    if distribution_id:
        try:
            current = cloudfront.get_distribution_config(Id=distribution_id)
            config = dict(current["DistributionConfig"])
            if config.get("Enabled") is True:
                config["Enabled"] = False
                cloudfront.update_distribution(
                    Id=distribution_id,
                    IfMatch=current["ETag"],
                    DistributionConfig=config,
                )
                _wait_distribution(cloudfront, distribution_id, enabled=False)
            current = cloudfront.get_distribution_config(Id=distribution_id)
            cloudfront.delete_distribution(
                Id=distribution_id, IfMatch=current["ETag"]
            )
        except Exception as exc:  # noqa: BLE001 - preserve the original failure
            errors.append(f"cloudfront:{type(exc).__name__}")
    instance_id = str(journal.get("instance_id") or "")
    if instance_id:
        try:
            ec2.terminate_instances(InstanceIds=[instance_id])
            ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"instance:{type(exc).__name__}")
    security_group_id = str(journal.get("security_group_id") or "")
    if security_group_id:
        try:
            deadline = time.monotonic() + 180
            while True:
                try:
                    ec2.delete_security_group(GroupId=security_group_id)
                    break
                except ClientError as exc:
                    if (
                        exc.response.get("Error", {}).get("Code")
                        != "DependencyViolation"
                        or time.monotonic() >= deadline
                    ):
                        raise
                    time.sleep(5)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"security-group:{type(exc).__name__}")
    artifact_bucket = str(journal.get("artifact_bucket") or "")
    if artifact_bucket:
        try:
            # Provisioning has not uploaded anything yet. Object Lock does not
            # prevent deletion of a still-empty bucket.
            s3.delete_bucket(Bucket=artifact_bucket)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"artifact-bucket:{type(exc).__name__}")
    return errors


def create_stack(
    *,
    ec2: Any,
    ssm: Any,
    cloudfront: Any,
    s3: Any,
    region: str,
    account_id: str,
    run_id: str,
    candidate_sha: str,
    production_gateway_ip: str,
    instance_profile_name: str,
    instance_type: str | None,
    volume_gib: int,
) -> dict[str, Any]:
    journal: dict[str, str] = {}
    try:
        return _create_stack_resources(
            ec2=ec2,
            ssm=ssm,
            cloudfront=cloudfront,
            s3=s3,
            region=region,
            account_id=account_id,
            run_id=run_id,
            candidate_sha=candidate_sha,
            production_gateway_ip=production_gateway_ip,
            instance_profile_name=instance_profile_name,
            instance_type=instance_type,
            volume_gib=volume_gib,
            journal=journal,
        )
    except Exception as exc:
        rollback_errors = _rollback_partial_stack(
            ec2=ec2, cloudfront=cloudfront, s3=s3, journal=journal
        )
        if rollback_errors:
            raise ProvisioningError(
                "ephemeral stack creation failed and immediate rollback was "
                "incomplete: " + ",".join(rollback_errors)
            ) from exc
        raise


def _load_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProvisioningError("ephemeral stack state is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ProvisioningError("ephemeral stack state schema differs")
    if not RUN_RE.fullmatch(str(value.get("run_id") or "")):
        raise ProvisioningError("ephemeral stack state run identity is invalid")
    return value


def delete_stack(*, ec2: Any, cloudfront: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    distribution_id = str(state.get("cloudfront_distribution_id") or "")
    instance_id = str(state.get("instance_id") or "")
    security_group_id = str(state.get("security_group_id") or "")
    artifact_bucket = str(state.get("artifact_bucket") or "")
    if distribution_id:
        current = cloudfront.get_distribution_config(Id=distribution_id)
        config = dict(current["DistributionConfig"])
        if config.get("Enabled") is True:
            config["Enabled"] = False
            cloudfront.update_distribution(
                Id=distribution_id,
                IfMatch=current["ETag"],
                DistributionConfig=config,
            )
            _wait_distribution(cloudfront, distribution_id, enabled=False)
        current = cloudfront.get_distribution_config(Id=distribution_id)
        cloudfront.delete_distribution(Id=distribution_id, IfMatch=current["ETag"])
    if instance_id:
        ec2.terminate_instances(InstanceIds=[instance_id])
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
    if security_group_id:
        deadline = time.monotonic() + 180
        while True:
            try:
                ec2.delete_security_group(GroupId=security_group_id)
                break
            except ClientError as exc:
                if (
                    exc.response.get("Error", {}).get("Code") != "DependencyViolation"
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(5)
    return {
        "run_id": state["run_id"],
        "instance_terminated": bool(instance_id),
        "distribution_deleted": bool(distribution_id),
        "security_group_deleted": bool(security_group_id),
        # COMPLIANCE retention is intentionally irreversible. The scheduled
        # exact-tag cleanup deletes versions and this bucket after one day.
        "artifact_bucket_retained_for_compliance": bool(artifact_bucket),
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--run-id", required=True)
    create_parser.add_argument("--candidate-sha", required=True)
    create_parser.add_argument("--production-gateway-ip", required=True)
    create_parser.add_argument("--instance-profile-name", required=True)
    create_parser.add_argument("--instance-type")
    create_parser.add_argument("--volume-gib", type=int, default=512)
    create_parser.add_argument("--state", type=Path, required=True)
    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        ec2 = boto3.client("ec2", region_name=args.region)
        ssm = boto3.client("ssm", region_name=args.region)
        cloudfront = boto3.client("cloudfront")
        s3 = boto3.client("s3", region_name=args.region)
        sts = boto3.client("sts", region_name=args.region)
        if args.command == "create":
            result = create_stack(
                ec2=ec2,
                ssm=ssm,
                cloudfront=cloudfront,
                s3=s3,
                region=args.region,
                account_id=str(sts.get_caller_identity()["Account"]),
                run_id=args.run_id,
                candidate_sha=args.candidate_sha.lower(),
                production_gateway_ip=args.production_gateway_ip,
                instance_profile_name=args.instance_profile_name,
                instance_type=args.instance_type,
                volume_gib=args.volume_gib,
            )
            _write(args.state, result)
        else:
            state = _load_state(args.state)
            result = delete_stack(ec2=ec2, cloudfront=cloudfront, state=state)
    except (BotoCoreError, ClientError, OSError, ValueError, ProvisioningError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
