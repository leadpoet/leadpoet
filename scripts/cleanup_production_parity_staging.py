#!/usr/bin/env python3
"""Remove only stale resources carrying the exact parity ownership tags."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import re
import sys
import time
from typing import Any, Mapping, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError


RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
NEW_SECRET_RE = re.compile(
    r"^leadpoet/staging/production-parity/runs/"
    r"(?P<run>[a-z0-9-]{6,40})/gateway$"
)
LEGACY_SECRET_RE = re.compile(
    r"^leadpoet/staging/production-parity/"
    r"(?P<run>[a-z0-9-]{6,40})/gateway$"
)
BUCKET_RE = re.compile(r"^leadpoet-parity-[0-9]{12}-[0-9a-f]{16}$")
TAG_RUN = "leadpoet:parity-run"
TAG_SHA = "leadpoet:candidate-sha"
TAG_EPHEMERAL = "leadpoet:ephemeral"


class StagingCleanupError(RuntimeError):
    pass


def _tag_map(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        value = value.get("Items", [])
    if not isinstance(value, list):
        return {}
    return {
        str(item.get("Key") or ""): str(item.get("Value") or "")
        for item in value
        if isinstance(item, Mapping) and item.get("Key")
    }


def _owned_run(tags: Any) -> str | None:
    values = _tag_map(tags)
    run_id = values.get(TAG_RUN, "")
    if (
        values.get(TAG_EPHEMERAL) != "true"
        or RUN_RE.fullmatch(run_id) is None
        or SHA_RE.fullmatch(values.get(TAG_SHA, "")) is None
    ):
        return None
    return run_id


def _utc(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise StagingCleanupError("resource timestamp is unavailable")
    return value.astimezone(timezone.utc)


def _wait_distribution(
    client: Any, distribution_id: str, *, enabled: bool, timeout: int = 1800
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = client.get_distribution(Id=distribution_id)["Distribution"]
        config = value.get("DistributionConfig", {})
        if value.get("Status") == "Deployed" and bool(config.get("Enabled")) is enabled:
            return
        time.sleep(15)
    raise StagingCleanupError("stale CloudFront distribution did not converge")


def cleanup_stale(
    *,
    ec2: Any,
    cloudfront: Any,
    secretsmanager: Any,
    s3: Any,
    now: datetime,
    max_age_hours: int,
    apply: bool,
) -> dict[str, Any]:
    if max_age_hours < 24 or max_age_hours > 168:
        raise StagingCleanupError("cleanup inputs are invalid")
    cutoff = now.astimezone(timezone.utc) - timedelta(hours=max_age_hours)
    instances: list[str] = []
    distributions: list[str] = []
    secrets: list[str] = []
    stale_runs: set[str] = set()
    artifact_buckets: list[str] = []

    response = ec2.describe_instances(
        Filters=[{"Name": f"tag:{TAG_EPHEMERAL}", "Values": ["true"]}]
    )
    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            run_id = _owned_run(instance.get("Tags"))
            state = str(instance.get("State", {}).get("Name") or "")
            if (
                run_id is not None
                and state not in {"terminated", "shutting-down"}
                and _utc(instance.get("LaunchTime")) <= cutoff
            ):
                stale_runs.add(run_id)
                instances.append(str(instance["InstanceId"]))

    marker = None
    while True:
        page = cloudfront.list_distributions(**({"Marker": marker} if marker else {}))
        listing = page.get("DistributionList", {})
        for item in listing.get("Items", []):
            arn = str(item.get("ARN") or "")
            tags = cloudfront.list_tags_for_resource(Resource=arn).get("Tags", {})
            run_id = _owned_run(tags)
            if (
                run_id is not None
                and _utc(item.get("LastModifiedTime")) <= cutoff
            ):
                stale_runs.add(run_id)
                distributions.append(str(item["Id"]))
        if not listing.get("IsTruncated"):
            break
        marker = str(listing.get("NextMarker") or "")
        if not marker:
            raise StagingCleanupError("CloudFront pagination is invalid")

    paginator = secretsmanager.get_paginator("list_secrets")
    observed_secret_names: set[str] = set()
    for prefix, pattern in (
        ("leadpoet/staging/production-parity/runs/pp-", NEW_SECRET_RE),
        ("leadpoet/staging/production-parity/pp-", LEGACY_SECRET_RE),
    ):
        for page in paginator.paginate(
            Filters=[{"Key": "name", "Values": [prefix]}],
            IncludePlannedDeletion=False,
        ):
            for item in page.get("SecretList", []):
                name = str(item.get("Name") or "")
                if name in observed_secret_names:
                    continue
                observed_secret_names.add(name)
                match = pattern.fullmatch(name)
                run_id = _owned_run(item.get("Tags"))
                if (
                    match is not None
                    and run_id == match.group("run")
                    and _utc(item.get("CreatedDate")) <= cutoff
                ):
                    stale_runs.add(run_id)
                    secrets.append(name)

    for item in s3.list_buckets().get("Buckets", []):
        name = str(item.get("Name") or "")
        if (
            BUCKET_RE.fullmatch(name) is None
            or _utc(item.get("CreationDate")) > cutoff
        ):
            continue
        try:
            tags = s3.get_bucket_tagging(Bucket=name).get("TagSet", [])
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {
                "NoSuchTagSet",
                "NoSuchBucket",
            }:
                continue
            raise
        run_id = _owned_run(tags)
        if run_id is not None:
            stale_runs.add(run_id)
            artifact_buckets.append(name)

    security_groups: list[str] = []
    groups = ec2.describe_security_groups(
        Filters=[{"Name": f"tag:{TAG_EPHEMERAL}", "Values": ["true"]}]
    ).get("SecurityGroups", [])
    for group in groups:
        run_id = _owned_run(group.get("Tags"))
        if run_id in stale_runs:
            security_groups.append(str(group["GroupId"]))

    result = {
        "schema_version": "leadpoet.production_parity_stale_cleanup.v2",
        "mode": "apply" if apply else "dry-run",
        "cutoff": cutoff.isoformat(),
        "runs": sorted(stale_runs),
        "instances": sorted(set(instances)),
        "distributions": sorted(set(distributions)),
        "secrets": sorted(set(secrets)),
        "security_groups": sorted(set(security_groups)),
        "artifact_buckets": sorted(set(artifact_buckets)),
    }
    if not apply:
        return result

    if instances:
        ec2.terminate_instances(InstanceIds=sorted(set(instances)))
        ec2.get_waiter("instance_terminated").wait(
            InstanceIds=sorted(set(instances))
        )
    for distribution_id in sorted(set(distributions)):
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
    for name in sorted(set(secrets)):
        secretsmanager.delete_secret(
            SecretId=name, ForceDeleteWithoutRecovery=True
        )
    for bucket in sorted(set(artifact_buckets)):
        paginator = s3.get_paginator("list_object_versions")
        for page in paginator.paginate(Bucket=bucket):
            objects = [
                {"Key": item["Key"], "VersionId": item["VersionId"]}
                for field in ("Versions", "DeleteMarkers")
                for item in page.get(field, [])
                if isinstance(item, Mapping)
                and item.get("Key")
                and item.get("VersionId")
            ]
            for offset in range(0, len(objects), 1000):
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": objects[offset : offset + 1000], "Quiet": True},
                )
        s3.delete_bucket(Bucket=bucket)
    for group_id in sorted(set(security_groups)):
        deadline = time.monotonic() + 300
        while True:
            try:
                ec2.delete_security_group(GroupId=group_id)
                break
            except ClientError as exc:
                if (
                    exc.response.get("Error", {}).get("Code") != "DependencyViolation"
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(10)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument("--max-age-hours", type=int, default=30)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        session = boto3.session.Session(region_name=args.region)
        result = cleanup_stale(
            ec2=session.client("ec2"),
            cloudfront=session.client("cloudfront"),
            secretsmanager=session.client("secretsmanager"),
            s3=session.client("s3"),
            now=datetime.now(timezone.utc),
            max_age_hours=args.max_age_hours,
            apply=args.apply,
        )
    except (BotoCoreError, ClientError, StagingCleanupError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
