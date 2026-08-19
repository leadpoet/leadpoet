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


RUN_RE = re.compile(r"^pp-[0-9]{1,20}-[0-9]{1,6}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
IP_RE = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
INSTANCE_ID_RE = re.compile(r"^i-(?:[0-9a-f]{8}|[0-9a-f]{17})$")
SECURITY_GROUP_ID_RE = re.compile(r"^sg-(?:[0-9a-f]{8}|[0-9a-f]{17})$")
CLOUDFRONT_DISTRIBUTION_ID_RE = re.compile(r"^E[A-Z0-9]{7,31}$")
SCHEMA_VERSION = "leadpoet.production_parity_ephemeral_stack.v3"
PRODUCTION_ACCOUNT_ID = "493765492819"
PRODUCTION_REGION = "us-east-1"
PRODUCTION_GATEWAY_INSTANCE_ID = "i-07e945bb2653c2e8f"
PRODUCTION_AMI_ID = "ami-0cae6d6fe6048ca2c"
PRODUCTION_INSTANCE_TYPE = "r7i.4xlarge"
PRODUCTION_SUBNET_ID = "subnet-025170c1eff61494d"
PRODUCTION_VPC_ID = "vpc-0c975a643bc1e0e79"
PRODUCTION_RUNNER_PROFILE = "leadpoet-production-parity-runner"
PARITY_VOLUME_GIB = 512
TAG_RUN = "leadpoet:parity-run"
TAG_SHA = "leadpoet:candidate-sha"
TAG_EPHEMERAL = "leadpoet:ephemeral"
ARTIFACT_RETENTION_DAYS = 1
CLOUDFRONT_MANAGED_POLICY_MAX_PAGES = 100
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


def _resource_arn(*, kind: str, resource_id: str) -> str:
    return (
        f"arn:aws:ec2:{PRODUCTION_REGION}:{PRODUCTION_ACCOUNT_ID}:"
        f"{kind}/{resource_id}"
    )


def _distribution_arn(distribution_id: str) -> str:
    return (
        f"arn:aws:cloudfront::{PRODUCTION_ACCOUNT_ID}:"
        f"distribution/{distribution_id}"
    )


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


def _client_error_code(exc: BaseException) -> str:
    if not isinstance(exc, ClientError):
        return ""
    return str(exc.response.get("Error", {}).get("Code") or "")


def _require_ownership_tags(
    tags: Sequence[Mapping[str, Any]], *, run_id: str, candidate_sha: str
) -> None:
    expected = {
        TAG_RUN: run_id,
        TAG_SHA: candidate_sha,
        TAG_EPHEMERAL: "true",
        "Name": f"leadpoet-parity-{run_id}",
    }
    for key, expected_value in expected.items():
        observed = [
            str(tag.get("Value") or "")
            for tag in tags
            if str(tag.get("Key") or "") == key
        ]
        if observed != [expected_value]:
            raise ProvisioningError(
                "ephemeral resource ownership tags do not match state"
            )


def _validate_state(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvisioningError("ephemeral stack state schema differs")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ProvisioningError("ephemeral stack state schema differs")
    if value.get("status") != "ready":
        raise ProvisioningError("ephemeral stack state is not ready")
    if value.get("account_id") != PRODUCTION_ACCOUNT_ID:
        raise ProvisioningError("ephemeral stack state account differs")
    if value.get("region") != PRODUCTION_REGION:
        raise ProvisioningError("ephemeral stack state region differs")

    run_id = str(value.get("run_id") or "")
    candidate_sha = str(value.get("candidate_sha") or "")
    instance_id = str(value.get("instance_id") or "")
    security_group_id = str(value.get("security_group_id") or "")
    distribution_id = str(value.get("cloudfront_distribution_id") or "")
    if not RUN_RE.fullmatch(run_id):
        raise ProvisioningError("ephemeral stack state run identity is invalid")
    if not SHA_RE.fullmatch(candidate_sha):
        raise ProvisioningError("ephemeral stack state candidate identity is invalid")
    if not INSTANCE_ID_RE.fullmatch(instance_id):
        raise ProvisioningError("ephemeral stack state instance identity is invalid")
    if not SECURITY_GROUP_ID_RE.fullmatch(security_group_id):
        raise ProvisioningError(
            "ephemeral stack state security-group identity is invalid"
        )
    if not CLOUDFRONT_DISTRIBUTION_ID_RE.fullmatch(distribution_id):
        raise ProvisioningError(
            "ephemeral stack state CloudFront identity is invalid"
        )

    expected_instance_arn = _resource_arn(
        kind="instance", resource_id=instance_id
    )
    expected_security_group_arn = _resource_arn(
        kind="security-group", resource_id=security_group_id
    )
    expected_distribution_arn = _distribution_arn(distribution_id)
    if value.get("instance_arn") != expected_instance_arn:
        raise ProvisioningError("ephemeral stack state instance ARN differs")
    if value.get("security_group_arn") != expected_security_group_arn:
        raise ProvisioningError("ephemeral stack state security-group ARN differs")
    if value.get("cloudfront_distribution_arn") != expected_distribution_arn:
        raise ProvisioningError("ephemeral stack state CloudFront ARN differs")

    expected_bucket = _artifact_bucket_name(
        account_id=PRODUCTION_ACCOUNT_ID,
        run_id=run_id,
        candidate_sha=candidate_sha,
    )
    if value.get("artifact_bucket") != expected_bucket:
        raise ProvisioningError("ephemeral stack state artifact bucket differs")
    if (
        type(value.get("volume_gib")) is not int
        or value.get("volume_gib") != PARITY_VOLUME_GIB
    ):
        raise ProvisioningError("ephemeral stack state volume differs")
    return dict(value)


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
        instance.get("InstanceId") != PRODUCTION_GATEWAY_INSTANCE_ID
        or instance.get("ImageId") != PRODUCTION_AMI_ID
        or instance.get("InstanceType") != PRODUCTION_INSTANCE_TYPE
        or instance.get("SubnetId") != PRODUCTION_SUBNET_ID
        or instance.get("VpcId") != PRODUCTION_VPC_ID
        or instance.get("EnclaveOptions", {}).get("Enabled") is not True
        or instance.get("MetadataOptions", {}).get("HttpTokens") != "required"
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
    policy_shapes = {
        "cache": (
            "list_cache_policies",
            "CachePolicyList",
            "CachePolicy",
            "CachePolicyConfig",
        ),
        "origin_request": (
            "list_origin_request_policies",
            "OriginRequestPolicyList",
            "OriginRequestPolicy",
            "OriginRequestPolicyConfig",
        ),
    }
    shape = policy_shapes.get(policy_type)
    if shape is None or not name:
        raise ProvisioningError("CloudFront managed policy lookup is invalid")
    operation, list_key, policy_key, config_key = shape
    marker: str | None = None
    seen_markers: set[str] = set()
    matches: list[str] = []

    for _page_number in range(CLOUDFRONT_MANAGED_POLICY_MAX_PAGES):
        request = {"Type": "managed"}
        if marker is not None:
            request["Marker"] = marker
        page = getattr(client, operation)(**request)
        if not isinstance(page, Mapping):
            raise ProvisioningError("CloudFront managed policy page is invalid")
        listing = page.get(list_key)
        if not isinstance(listing, Mapping):
            raise ProvisioningError("CloudFront managed policy page is invalid")
        items = listing.get("Items", [])
        if not isinstance(items, list):
            raise ProvisioningError("CloudFront managed policy page is invalid")
        for item in items:
            if not isinstance(item, Mapping):
                raise ProvisioningError("CloudFront managed policy page is invalid")
            policy = item.get(policy_key)
            if not isinstance(policy, Mapping):
                raise ProvisioningError("CloudFront managed policy page is invalid")
            config = policy.get(config_key)
            if not isinstance(config, Mapping):
                raise ProvisioningError("CloudFront managed policy page is invalid")
            if config.get("Name") == name:
                value = policy.get("Id")
                if not isinstance(value, str) or not value:
                    raise ProvisioningError("CloudFront managed policy page is invalid")
                matches.append(value)

        if "NextMarker" not in listing:
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ProvisioningError(
                    f"CloudFront managed policy is ambiguous: {name}"
                )
            raise ProvisioningError(
                f"CloudFront managed policy is unavailable: {name}"
            )
        next_marker = listing.get("NextMarker")
        if not isinstance(next_marker, str) or not next_marker.strip():
            raise ProvisioningError("CloudFront managed policy pagination is invalid")
        if next_marker == marker or next_marker in seen_markers:
            raise ProvisioningError("CloudFront managed policy pagination is invalid")
        seen_markers.add(next_marker)
        marker = next_marker

    raise ProvisioningError("CloudFront managed policy pagination limit exceeded")


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
        or instance_profile_name != PRODUCTION_RUNNER_PROFILE
        or account_id != PRODUCTION_ACCOUNT_ID
        or region != PRODUCTION_REGION
        or volume_gib != PARITY_VOLUME_GIB
    ):
        raise ProvisioningError("ephemeral stack inputs are invalid")
    reference = _single_production_instance(ec2, production_gateway_ip)
    selected_type = str(instance_type or PRODUCTION_INSTANCE_TYPE)
    if (
        selected_type != PRODUCTION_INSTANCE_TYPE
        or reference.get("InstanceType") != PRODUCTION_INSTANCE_TYPE
    ):
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
        TagSpecifications=[{
            "ResourceType": "security-group",
            "Tags": _tags(run_id, candidate_sha),
        }],
    )
    security_group_id = str(security_group["GroupId"])
    journal["security_group_id"] = security_group_id
    if not SECURITY_GROUP_ID_RE.fullmatch(security_group_id):
        raise ProvisioningError("ephemeral security-group identity is invalid")
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
            {
                "ResourceType": "network-interface",
                "Tags": _tags(run_id, candidate_sha),
            },
        ],
    )
    values = launched.get("Instances", [])
    if len(values) != 1:
        raise ProvisioningError("ephemeral host launch returned an invalid inventory")
    instance_id = str(values[0].get("InstanceId") or "")
    journal["instance_id"] = instance_id
    if not INSTANCE_ID_RE.fullmatch(instance_id):
        raise ProvisioningError("ephemeral instance identity is invalid")
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
    created = cloudfront.create_distribution_with_tags(
        DistributionConfigWithTags={
            "DistributionConfig": {
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
                                "OriginSslProtocols": {
                                    "Quantity": 1,
                                    "Items": ["TLSv1.2"],
                                },
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
                        "Items": [
                            "GET",
                            "HEAD",
                            "OPTIONS",
                            "PUT",
                            "PATCH",
                            "POST",
                            "DELETE",
                        ],
                        "CachedMethods": {
                            "Quantity": 2,
                            "Items": ["GET", "HEAD"],
                        },
                    },
                    "Compress": False,
                    "CachePolicyId": cache_policy_id,
                    "OriginRequestPolicyId": origin_policy_id,
                    "SmoothStreaming": False,
                },
                "ViewerCertificate": {
                    "CloudFrontDefaultCertificate": True
                },
                "Restrictions": {
                    "GeoRestriction": {
                        "RestrictionType": "none",
                        "Quantity": 0,
                    }
                },
            },
            "Tags": {"Items": _tags(run_id, candidate_sha)},
        },
    )
    distribution = created["Distribution"]
    distribution_id = str(distribution["Id"])
    journal["cloudfront_distribution_id"] = distribution_id
    if not CLOUDFRONT_DISTRIBUTION_ID_RE.fullmatch(distribution_id):
        raise ProvisioningError("ephemeral CloudFront identity is invalid")
    distribution_arn = str(distribution.get("ARN") or "")
    if distribution_arn != _distribution_arn(distribution_id):
        raise ProvisioningError("ephemeral CloudFront ARN is invalid")
    deployed = _wait_distribution(cloudfront, distribution_id, enabled=True)
    if deployed.get("ARN") != distribution_arn:
        raise ProvisioningError("deployed CloudFront ARN differs from creation")
    domain = str(deployed.get("DomainName") or "")
    if not domain.endswith(".cloudfront.net"):
        raise ProvisioningError("CloudFront parity domain is invalid")
    state = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "account_id": account_id,
        "region": region,
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instance_id": instance_id,
        "instance_arn": _resource_arn(kind="instance", resource_id=instance_id),
        "security_group_id": security_group_id,
        "security_group_arn": _resource_arn(
            kind="security-group", resource_id=security_group_id
        ),
        "cloudfront_distribution_id": distribution_id,
        "cloudfront_distribution_arn": distribution_arn,
        "supabase_origin": f"https://{domain}",
        "artifact_bucket": artifact_bucket,
        "artifact_retention_days": ARTIFACT_RETENTION_DAYS,
        "reference_image_id": str(reference["ImageId"]),
        "reference_instance_type": str(reference["InstanceType"]),
        "selected_instance_type": selected_type,
        "volume_gib": volume_gib,
    }
    return _validate_state(state)


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
    if not isinstance(value, dict):
        raise ProvisioningError("ephemeral stack state schema differs")
    return _validate_state(value)


def _read_owned_distribution(
    cloudfront: Any, state: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    distribution_id = str(state["cloudfront_distribution_id"])
    distribution_arn = str(state["cloudfront_distribution_arn"])
    try:
        response = cloudfront.get_distribution(Id=distribution_id)
    except ClientError as exc:
        if _client_error_code(exc) == "NoSuchDistribution":
            return None
        raise
    distribution = response.get("Distribution")
    if not isinstance(distribution, Mapping):
        raise ProvisioningError("CloudFront deletion identity is unavailable")
    if (
        distribution.get("Id") != distribution_id
        or distribution.get("ARN") != distribution_arn
        or response.get("ETag") in (None, "")
    ):
        raise ProvisioningError("CloudFront deletion identity differs from state")
    tags = cloudfront.list_tags_for_resource(Resource=distribution_arn)
    values = tags.get("Tags", {}).get("Items", [])
    if not isinstance(values, list):
        raise ProvisioningError("CloudFront ownership tags are unavailable")
    _require_ownership_tags(
        values,
        run_id=str(state["run_id"]),
        candidate_sha=str(state["candidate_sha"]),
    )
    return response


def _delete_owned_distribution(cloudfront: Any, state: Mapping[str, Any]) -> None:
    distribution_id = str(state["cloudfront_distribution_id"])
    current = _read_owned_distribution(cloudfront, state)
    if current is None:
        return
    distribution = current["Distribution"]
    config = dict(distribution.get("DistributionConfig") or {})
    if config.get("Enabled") is True:
        config["Enabled"] = False
        try:
            cloudfront.update_distribution(
                Id=distribution_id,
                IfMatch=current["ETag"],
                DistributionConfig=config,
            )
        except ClientError as exc:
            if _client_error_code(exc) == "NoSuchDistribution":
                return
            raise
        try:
            _wait_distribution(cloudfront, distribution_id, enabled=False)
        except ClientError as exc:
            if _client_error_code(exc) == "NoSuchDistribution":
                return
            raise
    elif distribution.get("Status") != "Deployed":
        try:
            _wait_distribution(cloudfront, distribution_id, enabled=False)
        except ClientError as exc:
            if _client_error_code(exc) == "NoSuchDistribution":
                return
            raise
    current = _read_owned_distribution(cloudfront, state)
    if current is None:
        return
    try:
        cloudfront.delete_distribution(
            Id=distribution_id, IfMatch=current["ETag"]
        )
    except ClientError as exc:
        if _client_error_code(exc) != "NoSuchDistribution":
            raise


def _read_owned_instance(
    ec2: Any, state: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    instance_id = str(state["instance_id"])
    try:
        response = ec2.describe_instances(InstanceIds=[instance_id])
    except ClientError as exc:
        if _client_error_code(exc) == "InvalidInstanceID.NotFound":
            return None
        raise
    instances = [
        instance
        for reservation in response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    if not instances:
        return None
    if len(instances) != 1 or instances[0].get("InstanceId") != instance_id:
        raise ProvisioningError("instance deletion identity differs from state")
    _require_ownership_tags(
        instances[0].get("Tags", []),
        run_id=str(state["run_id"]),
        candidate_sha=str(state["candidate_sha"]),
    )
    return instances[0]


def _delete_owned_instance(ec2: Any, state: Mapping[str, Any]) -> None:
    instance_id = str(state["instance_id"])
    instance = _read_owned_instance(ec2, state)
    if instance is None:
        return
    state_name = str(instance.get("State", {}).get("Name") or "")
    if state_name == "terminated":
        return
    if state_name != "shutting-down":
        try:
            ec2.terminate_instances(InstanceIds=[instance_id])
        except ClientError as exc:
            if _client_error_code(exc) == "InvalidInstanceID.NotFound":
                return
            raise
    try:
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
    except Exception:  # noqa: BLE001 - verify idempotent completion after races
        observed = _read_owned_instance(ec2, state)
        if observed is None or observed.get("State", {}).get("Name") == "terminated":
            return
        raise


def _read_owned_security_group(
    ec2: Any, state: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    security_group_id = str(state["security_group_id"])
    try:
        response = ec2.describe_security_groups(GroupIds=[security_group_id])
    except ClientError as exc:
        if _client_error_code(exc) == "InvalidGroup.NotFound":
            return None
        raise
    groups = response.get("SecurityGroups", [])
    if not groups:
        return None
    expected_arn = str(state["security_group_arn"])
    if (
        len(groups) != 1
        or groups[0].get("GroupId") != security_group_id
        or groups[0].get("OwnerId") != PRODUCTION_ACCOUNT_ID
        or groups[0].get("SecurityGroupArn") != expected_arn
    ):
        raise ProvisioningError(
            "security-group deletion identity differs from state"
        )
    _require_ownership_tags(
        groups[0].get("Tags", []),
        run_id=str(state["run_id"]),
        candidate_sha=str(state["candidate_sha"]),
    )
    return groups[0]


def _delete_owned_security_group(ec2: Any, state: Mapping[str, Any]) -> None:
    security_group_id = str(state["security_group_id"])
    deadline = time.monotonic() + 180
    while True:
        if _read_owned_security_group(ec2, state) is None:
            return
        try:
            ec2.delete_security_group(GroupId=security_group_id)
            return
        except ClientError as exc:
            code = _client_error_code(exc)
            if code == "InvalidGroup.NotFound":
                return
            if code != "DependencyViolation" or time.monotonic() >= deadline:
                raise
            time.sleep(5)


def delete_stack(
    *,
    ec2: Any,
    cloudfront: Any,
    state: Mapping[str, Any],
    region: str = PRODUCTION_REGION,
    account_id: str = PRODUCTION_ACCOUNT_ID,
) -> dict[str, Any]:
    state = _validate_state(state)
    if region != state["region"] or account_id != state["account_id"]:
        raise ProvisioningError("AWS deletion identity differs from state")

    errors: list[str] = []
    cleanup = (
        ("cloudfront", lambda: _delete_owned_distribution(cloudfront, state)),
        ("instance", lambda: _delete_owned_instance(ec2, state)),
        ("security-group", lambda: _delete_owned_security_group(ec2, state)),
    )
    for label, operation in cleanup:
        try:
            operation()
        except Exception as exc:  # noqa: BLE001 - report all cleanup failures
            errors.append(f"{label}:{type(exc).__name__}")
    if errors:
        raise ProvisioningError(
            "ephemeral stack deletion was incomplete: " + ",".join(errors)
        )

    return {
        "run_id": state["run_id"],
        "instance_terminated": True,
        "distribution_deleted": True,
        "security_group_deleted": True,
        # COMPLIANCE retention is intentionally irreversible. The scheduled
        # exact-tag cleanup deletes versions and this bucket after one day.
        "artifact_bucket_retained_for_compliance": True,
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
    create_parser.add_argument("--volume-gib", type=int, default=PARITY_VOLUME_GIB)
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
        caller_account_id = str(sts.get_caller_identity()["Account"])
        if args.command == "create":
            result = create_stack(
                ec2=ec2,
                ssm=ssm,
                cloudfront=cloudfront,
                s3=s3,
                region=args.region,
                account_id=caller_account_id,
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
            result = delete_stack(
                ec2=ec2,
                cloudfront=cloudfront,
                state=state,
                region=args.region,
                account_id=caller_account_id,
            )
    except (BotoCoreError, ClientError, OSError, ValueError, ProvisioningError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
