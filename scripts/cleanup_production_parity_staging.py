#!/usr/bin/env python3
"""Remove only stale, exactly tagged production-parity staging resources."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import re
from typing import Any, Iterable, Mapping, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError


RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
STACK_RE = re.compile(r"^leadpoet-parity-(?P<run>[a-z0-9-]{6,40})$")
SECRET_RE = re.compile(
    r"^leadpoet/staging/production-parity/(?P<run>[a-z0-9-]{6,40})/"
    r"(?:gateway|validator|database|auditor-a|auditor-b|dashboard)$"
)
ACTIVE_STACK_STATUSES = (
    "CREATE_IN_PROGRESS",
    "CREATE_FAILED",
    "CREATE_COMPLETE",
    "ROLLBACK_IN_PROGRESS",
    "ROLLBACK_FAILED",
    "ROLLBACK_COMPLETE",
    "DELETE_FAILED",
    "UPDATE_IN_PROGRESS",
    "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS",
    "UPDATE_COMPLETE",
    "UPDATE_FAILED",
    "UPDATE_ROLLBACK_IN_PROGRESS",
    "UPDATE_ROLLBACK_FAILED",
    "UPDATE_ROLLBACK_COMPLETE_CLEANUP_IN_PROGRESS",
    "UPDATE_ROLLBACK_COMPLETE",
    "REVIEW_IN_PROGRESS",
    "IMPORT_IN_PROGRESS",
    "IMPORT_COMPLETE",
    "IMPORT_ROLLBACK_IN_PROGRESS",
    "IMPORT_ROLLBACK_FAILED",
    "IMPORT_ROLLBACK_COMPLETE",
)


class StagingCleanupError(RuntimeError):
    pass


def _tags(value: Any) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    return {
        str(item.get("Key") or ""): str(item.get("Value") or "")
        for item in value
        if isinstance(item, Mapping) and item.get("Key")
    }


def _is_stale(value: Any, *, cutoff: datetime) -> bool:
    return (
        isinstance(value, datetime)
        and value.astimezone(timezone.utc) <= cutoff
    )


def _tagged_run(value: Any, *, run_id: str) -> bool:
    tags = _tags(value)
    return (
        tags.get("leadpoet:parity-run") == run_id
        and SHA_RE.fullmatch(tags.get("leadpoet:candidate-sha", "")) is not None
    )


def _pages(client: Any, operation: str, result_key: str, **kwargs: Any) -> Iterable[Any]:
    paginator = client.get_paginator(operation)
    for page in paginator.paginate(**kwargs):
        values = page.get(result_key, [])
        if not isinstance(values, list):
            raise StagingCleanupError(f"{operation} returned an invalid page")
        yield from values


def cleanup_stale(
    *,
    cloudformation: Any,
    secretsmanager: Any,
    ec2: Any,
    now: datetime,
    max_age_hours: int,
    apply: bool,
) -> dict[str, Any]:
    if max_age_hours < 8 or max_age_hours > 168:
        raise StagingCleanupError("cleanup age must be between 8 and 168 hours")
    cutoff = now.astimezone(timezone.utc) - timedelta(hours=max_age_hours)
    stacks: list[str] = []
    secrets: list[str] = []
    key_pairs: list[str] = []

    for summary in _pages(
        cloudformation,
        "list_stacks",
        "StackSummaries",
        StackStatusFilter=list(ACTIVE_STACK_STATUSES),
    ):
        if not isinstance(summary, Mapping):
            raise StagingCleanupError("stack inventory is invalid")
        name = str(summary.get("StackName") or "")
        match = STACK_RE.fullmatch(name)
        if match is None or not _is_stale(summary.get("CreationTime"), cutoff=cutoff):
            continue
        detail = cloudformation.describe_stacks(StackName=name).get("Stacks", [])
        if (
            not isinstance(detail, list)
            or len(detail) != 1
            or not _tagged_run(detail[0].get("Tags"), run_id=match.group("run"))
        ):
            raise StagingCleanupError(f"stale stack tags differ: {name}")
        stacks.append(name)

    for item in _pages(
        secretsmanager,
        "list_secrets",
        "SecretList",
        Filters=[{"Key": "tag-key", "Values": ["leadpoet:parity-run"]}],
        IncludePlannedDeletion=False,
    ):
        if not isinstance(item, Mapping):
            raise StagingCleanupError("secret inventory is invalid")
        name = str(item.get("Name") or "")
        match = SECRET_RE.fullmatch(name)
        if match is None or not _is_stale(item.get("CreatedDate"), cutoff=cutoff):
            continue
        if not _tagged_run(item.get("Tags"), run_id=match.group("run")):
            raise StagingCleanupError(f"stale secret tags differ: {name}")
        secrets.append(name)

    response = ec2.describe_key_pairs(
        Filters=[{"Name": "tag-key", "Values": ["leadpoet:parity-run"]}]
    )
    values = response.get("KeyPairs", [])
    if not isinstance(values, list):
        raise StagingCleanupError("key-pair inventory is invalid")
    for item in values:
        if not isinstance(item, Mapping):
            raise StagingCleanupError("key-pair inventory is invalid")
        name = str(item.get("KeyName") or "")
        match = STACK_RE.fullmatch(name)
        if match is None or not _is_stale(item.get("CreateTime"), cutoff=cutoff):
            continue
        if not _tagged_run(item.get("Tags"), run_id=match.group("run")):
            raise StagingCleanupError(f"stale key-pair tags differ: {name}")
        key_pairs.append(name)

    stacks = sorted(set(stacks))
    secrets = sorted(set(secrets))
    key_pairs = sorted(set(key_pairs))
    if apply:
        for name in secrets:
            secretsmanager.delete_secret(
                SecretId=name, ForceDeleteWithoutRecovery=True
            )
        for name in stacks:
            cloudformation.delete_stack(StackName=name)
        for name in key_pairs:
            ec2.delete_key_pair(KeyName=name)
    return {
        "schema_version": "leadpoet.production_parity_stale_cleanup.v1",
        "mode": "apply" if apply else "dry-run",
        "cutoff": cutoff.isoformat(),
        "stack_count": len(stacks),
        "secret_count": len(secrets),
        "key_pair_count": len(key_pairs),
        "stacks": stacks,
        "secrets": secrets,
        "key_pairs": key_pairs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument("--max-age-hours", type=int, default=12)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[a-z]{2}-[a-z]+-[0-9]", args.region):
            raise StagingCleanupError("AWS region is invalid")
        session = boto3.session.Session(region_name=args.region)
        result = cleanup_stale(
            cloudformation=session.client("cloudformation"),
            secretsmanager=session.client("secretsmanager"),
            ec2=session.client("ec2"),
            now=datetime.now(timezone.utc),
            max_age_hours=args.max_age_hours,
            apply=args.apply,
        )
    except (BotoCoreError, ClientError, StagingCleanupError, ValueError) as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
