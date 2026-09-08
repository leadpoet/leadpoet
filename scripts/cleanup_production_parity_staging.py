#!/usr/bin/env python3
"""Remove only stale resources carrying the exact parity ownership tags."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
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
EXACT_RUN_RE = re.compile(r"^pp-[0-9]{1,20}-[0-9]{1,6}$")
PRODUCTION_ACCOUNT_ID = "493765492819"
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


def _is_exact_owner(tags: Any, *, run_id: str, candidate_sha: str) -> bool:
    values = _tag_map(tags)
    return (
        values.get(TAG_RUN) == run_id
        and values.get(TAG_SHA) == candidate_sha
        and values.get(TAG_EPHEMERAL) == "true"
    )


def _artifact_bucket_name(*, run_id: str, candidate_sha: str) -> str:
    suffix = hashlib.sha256(
        f"{PRODUCTION_ACCOUNT_ID}:{run_id}:{candidate_sha}".encode("ascii")
    ).hexdigest()[:16]
    return f"leadpoet-parity-{PRODUCTION_ACCOUNT_ID}-{suffix}"


def _error_label(resource: str, exc: BaseException) -> str:
    code = ""
    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code") or "")
    return f"{resource}:{code or type(exc).__name__}"


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


def _bucket_has_objects(s3: Any, bucket: str) -> bool:
    for page in s3.get_paginator("list_object_versions").paginate(
        Bucket=bucket
    ):
        if page.get("Versions") or page.get("DeleteMarkers"):
            return True
    return bool(
        s3.list_objects_v2(Bucket=bucket, MaxKeys=1).get("Contents", [])
    )


def cleanup_stale(
    *,
    ec2: Any,
    cloudfront: Any,
    secretsmanager: Any,
    s3: Any,
    now: datetime,
    max_age_hours: int,
    apply: bool,
    run_id: str | None = None,
    candidate_sha: str | None = None,
) -> dict[str, Any]:
    if max_age_hours < 24 or max_age_hours > 168:
        raise StagingCleanupError("cleanup inputs are invalid")
    exact = run_id is not None or candidate_sha is not None
    if exact and (
        not isinstance(run_id, str)
        or EXACT_RUN_RE.fullmatch(run_id) is None
        or not isinstance(candidate_sha, str)
        or SHA_RE.fullmatch(candidate_sha) is None
    ):
        raise StagingCleanupError("exact-run cleanup inputs are invalid")
    cutoff = now.astimezone(timezone.utc) - timedelta(hours=max_age_hours)
    instances: list[str] = []
    distributions: list[str] = []
    secrets: list[str] = []
    stale_runs: set[str] = set()
    artifact_buckets: list[str] = []
    unowned_artifact_buckets: list[str] = []
    errors: list[str] = []

    def selected(tags: Any, created: Any) -> bool:
        if exact:
            return _is_exact_owner(
                tags, run_id=run_id, candidate_sha=candidate_sha
            )
        return _owned_run(tags) is not None and _utc(created) <= cutoff

    response = ec2.describe_instances(
        Filters=[{"Name": f"tag:{TAG_EPHEMERAL}", "Values": ["true"]}]
    )
    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            owned_run = _owned_run(instance.get("Tags"))
            state = str(instance.get("State", {}).get("Name") or "")
            if (
                selected(instance.get("Tags"), instance.get("LaunchTime"))
                and state not in {"terminated", "shutting-down"}
            ):
                stale_runs.add(str(owned_run))
                instances.append(str(instance["InstanceId"]))

    marker = None
    while True:
        page = cloudfront.list_distributions(**({"Marker": marker} if marker else {}))
        listing = page.get("DistributionList", {})
        for item in listing.get("Items", []):
            arn = str(item.get("ARN") or "")
            tags = cloudfront.list_tags_for_resource(Resource=arn).get("Tags", {})
            owned_run = _owned_run(tags)
            if selected(tags, item.get("LastModifiedTime")):
                stale_runs.add(str(owned_run))
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
                owned_run = _owned_run(item.get("Tags"))
                if (
                    match is not None
                    and owned_run == match.group("run")
                    and selected(item.get("Tags"), item.get("CreatedDate"))
                ):
                    stale_runs.add(str(owned_run))
                    secrets.append(name)

    expected_bucket = (
        _artifact_bucket_name(run_id=run_id, candidate_sha=candidate_sha)
        if exact
        else None
    )
    for item in s3.list_buckets().get("Buckets", []):
        name = str(item.get("Name") or "")
        if (
            BUCKET_RE.fullmatch(name) is None
            or (exact and name != expected_bucket)
            or (not exact and _utc(item.get("CreationDate")) > cutoff)
        ):
            continue
        try:
            tags = s3.get_bucket_tagging(Bucket=name).get("TagSet", [])
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {
                "NoSuchTagSet",
                "NoSuchBucket",
            }:
                if exact:
                    unowned_artifact_buckets.append(name)
                continue
            raise
        owned_run = _owned_run(tags)
        if selected(tags, item.get("CreationDate")):
            stale_runs.add(str(owned_run))
            artifact_buckets.append(name)
        elif exact:
            unowned_artifact_buckets.append(name)

    security_groups: list[str] = []
    groups = ec2.describe_security_groups(
        Filters=[{"Name": f"tag:{TAG_EPHEMERAL}", "Values": ["true"]}]
    ).get("SecurityGroups", [])
    for group in groups:
        owned_run = _owned_run(group.get("Tags"))
        if (
            exact
            and _is_exact_owner(
                group.get("Tags"),
                run_id=run_id,
                candidate_sha=candidate_sha,
            )
        ) or (not exact and owned_run in stale_runs):
            stale_runs.add(str(owned_run))
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
    if exact:
        result.update({
            "run_id": run_id,
            "candidate_sha": candidate_sha,
            "unowned_artifact_buckets": sorted(
                set(unowned_artifact_buckets)
            ),
            "retained_artifact_buckets": [],
            "errors": errors,
        })
    if not apply:
        return result

    if instances and not exact:
        ec2.terminate_instances(InstanceIds=sorted(set(instances)))
        ec2.get_waiter("instance_terminated").wait(
            InstanceIds=sorted(set(instances))
        )
    elif instances:
        for instance_id in sorted(set(instances)):
            try:
                response = ec2.describe_instances(InstanceIds=[instance_id])
                values = [
                    instance
                    for reservation in response.get("Reservations", [])
                    for instance in reservation.get("Instances", [])
                ]
                if len(values) != 1 or not _is_exact_owner(
                    values[0].get("Tags"),
                    run_id=run_id,
                    candidate_sha=candidate_sha,
                ):
                    raise StagingCleanupError("instance ownership changed")
                ec2.terminate_instances(InstanceIds=[instance_id])
                ec2.get_waiter("instance_terminated").wait(
                    InstanceIds=[instance_id]
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(_error_label(f"instance:{instance_id}", exc))
    for distribution_id in sorted(set(distributions)):
        try:
            if exact:
                distribution = cloudfront.get_distribution(Id=distribution_id)[
                    "Distribution"
                ]
                tags = cloudfront.list_tags_for_resource(
                    Resource=str(distribution.get("ARN") or "")
                ).get("Tags", {})
                if not _is_exact_owner(
                    tags, run_id=run_id, candidate_sha=candidate_sha
                ):
                    raise StagingCleanupError("distribution ownership changed")
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
            if exact:
                distribution = cloudfront.get_distribution(Id=distribution_id)[
                    "Distribution"
                ]
                tags = cloudfront.list_tags_for_resource(
                    Resource=str(distribution.get("ARN") or "")
                ).get("Tags", {})
                if not _is_exact_owner(
                    tags, run_id=run_id, candidate_sha=candidate_sha
                ):
                    raise StagingCleanupError("distribution ownership changed")
            current = cloudfront.get_distribution_config(Id=distribution_id)
            cloudfront.delete_distribution(
                Id=distribution_id, IfMatch=current["ETag"]
            )
        except Exception as exc:  # noqa: BLE001
            if not exact:
                raise
            errors.append(_error_label(f"distribution:{distribution_id}", exc))
    for name in sorted(set(secrets)):
        try:
            if exact:
                secret = secretsmanager.describe_secret(SecretId=name)
                if not _is_exact_owner(
                    secret.get("Tags"),
                    run_id=run_id,
                    candidate_sha=candidate_sha,
                ):
                    raise StagingCleanupError("secret ownership changed")
            secretsmanager.delete_secret(
                SecretId=name, ForceDeleteWithoutRecovery=True
            )
        except Exception as exc:  # noqa: BLE001
            if not exact:
                raise
            errors.append(_error_label(f"secret:{name}", exc))
    for bucket in sorted(set(artifact_buckets)):
        try:
            if exact:
                tags = s3.get_bucket_tagging(Bucket=bucket).get("TagSet", [])
                if not _is_exact_owner(
                    tags, run_id=run_id, candidate_sha=candidate_sha
                ):
                    raise StagingCleanupError("artifact bucket ownership changed")
                if _bucket_has_objects(s3, bucket):
                    result["retained_artifact_buckets"].append(bucket)
                    continue
            else:
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
        except Exception as exc:  # noqa: BLE001
            if not exact:
                raise
            errors.append(_error_label(f"artifact-bucket:{bucket}", exc))
    for group_id in sorted(set(security_groups)):
        try:
            if exact:
                groups = ec2.describe_security_groups(GroupIds=[group_id]).get(
                    "SecurityGroups", []
                )
                if len(groups) != 1 or not _is_exact_owner(
                    groups[0].get("Tags"),
                    run_id=run_id,
                    candidate_sha=candidate_sha,
                ):
                    raise StagingCleanupError("security group ownership changed")
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
        except Exception as exc:  # noqa: BLE001
            if not exact:
                raise
            errors.append(_error_label(f"security-group:{group_id}", exc))
    if exact:
        result["errors"] = sorted(set(errors))
    return result


def cleanup_exact_run(
    *,
    ec2: Any,
    cloudfront: Any,
    secretsmanager: Any,
    s3: Any,
    run_id: str,
    candidate_sha: str,
    apply: bool,
    max_attempts: int = 3,
    retry_delay_seconds: float = 5,
) -> dict[str, Any]:
    if (
        EXACT_RUN_RE.fullmatch(run_id) is None
        or SHA_RE.fullmatch(candidate_sha) is None
        or max_attempts not in range(1, 6)
        or retry_delay_seconds < 0
        or retry_delay_seconds > 30
    ):
        raise StagingCleanupError("exact-run cleanup inputs are invalid")

    clients = {
        "ec2": ec2,
        "cloudfront": cloudfront,
        "secretsmanager": secretsmanager,
        "s3": s3,
    }
    resource_keys = (
        "instances",
        "distributions",
        "secrets",
        "security_groups",
        "artifact_buckets",
        "unowned_artifact_buckets",
    )
    seen = {key: set() for key in resource_keys}
    retained: set[str] = set()
    errors: list[str] = []
    attempts = 1
    for attempts in range(1, (max_attempts if apply else 1) + 1):
        try:
            result = cleanup_stale(
                **clients,
                now=datetime.now(timezone.utc),
                max_age_hours=30,
                apply=apply,
                run_id=run_id,
                candidate_sha=candidate_sha,
            )
        except Exception as exc:  # noqa: BLE001 - retry bounded inventory errors
            errors = [_error_label("exact-run-inventory", exc)]
            if attempts < max_attempts:
                time.sleep(retry_delay_seconds)
            continue
        for key in resource_keys:
            seen[key].update(result[key])
        retained.update(result["retained_artifact_buckets"])
        errors = list(result["errors"])
        if apply and attempts < max_attempts:
            time.sleep(retry_delay_seconds)

    try:
        final = cleanup_stale(
            **clients,
            now=datetime.now(timezone.utc),
            max_age_hours=30,
            apply=False,
            run_id=run_id,
            candidate_sha=candidate_sha,
        )
    except Exception as exc:  # noqa: BLE001 - return machine-readable failure
        return {
            "schema_version": "leadpoet.production_parity_stale_cleanup.v2",
            "mode": "exact-run-apply" if apply else "exact-run-dry-run",
            "run_id": run_id,
            "candidate_sha": candidate_sha,
            "attempts": attempts,
            **{key: sorted(values) for key, values in seen.items()},
            "retained_artifact_buckets": sorted(retained),
            "residue": {},
            "errors": sorted(
                set(errors) | {_error_label("exact-run-inventory", exc)}
            ),
        }
    for key in resource_keys:
        seen[key].update(final[key])
    if not apply:
        return {
            "schema_version": "leadpoet.production_parity_stale_cleanup.v2",
            "mode": "exact-run-dry-run",
            "run_id": run_id,
            "candidate_sha": candidate_sha,
            "attempts": attempts,
            **{key: sorted(values) for key, values in seen.items()},
            "retained_artifact_buckets": [],
            "residue": {},
            "errors": [],
        }
    residue = {
        key: final[key]
        for key in resource_keys
        if final[key] and key != "artifact_buckets"
    }
    unresolved_buckets = sorted(
        set(final["artifact_buckets"]) - retained
    )
    if unresolved_buckets:
        residue["artifact_buckets"] = unresolved_buckets
    errors = sorted(
        set(errors)
        | {
            f"residue:{kind}:{resource}"
            for kind, resources in residue.items()
            for resource in resources
        }
    )
    return {
        "schema_version": "leadpoet.production_parity_stale_cleanup.v2",
        "mode": "exact-run-apply" if apply else "exact-run-dry-run",
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "attempts": attempts,
        **{key: sorted(values) for key, values in seen.items()},
        "retained_artifact_buckets": sorted(retained),
        "residue": residue,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument("--max-age-hours", type=int, default=30)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--candidate-sha")
    args = parser.parse_args(argv)
    try:
        if bool(args.run_id) != bool(args.candidate_sha):
            raise StagingCleanupError(
                "exact-run cleanup requires run ID and candidate SHA together"
            )
        session = boto3.session.Session(region_name=args.region)
        clients = {
            "ec2": session.client("ec2"),
            "cloudfront": session.client("cloudfront"),
            "secretsmanager": session.client("secretsmanager"),
            "s3": session.client("s3"),
        }
        if args.run_id:
            result = cleanup_exact_run(
                **clients,
                run_id=args.run_id,
                candidate_sha=args.candidate_sha,
                apply=args.apply,
            )
        else:
            result = cleanup_stale(
                **clients,
                now=datetime.now(timezone.utc),
                max_age_hours=args.max_age_hours,
                apply=args.apply,
            )
    except (BotoCoreError, ClientError, StagingCleanupError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    if args.run_id and (result["errors"] or result["residue"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
