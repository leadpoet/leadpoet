#!/usr/bin/env python3
"""Prove full rebenchmark readiness with the exact candidate gateway code."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError


SCHEMA_VERSION = "leadpoet.production_parity_rebenchmark_readiness.v1"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SECRET_RE = re.compile(
    r"^leadpoet/staging/production-parity/[a-z0-9-]{6,40}/gateway$"
)
ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class RebenchmarkReadinessError(RuntimeError):
    pass


def _checkout_identity(root: Path, candidate_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if (
        head.returncode != 0
        or dirty.returncode != 0
        or head.stdout.strip().lower() != candidate_sha
        or dirty.stdout.strip()
    ):
        raise RebenchmarkReadinessError(
            "gateway checkout differs from the exact staging candidate"
        )


def _secret_environment(secret_id: str, candidate_sha: str) -> dict[str, str]:
    if not SECRET_RE.fullmatch(secret_id):
        raise RebenchmarkReadinessError("gateway staging secret identity is invalid")
    try:
        response = boto3.client("secretsmanager").get_secret_value(
            SecretId=secret_id
        )
        value = json.loads(str(response.get("SecretString") or ""))
    except (BotoCoreError, ClientError, ValueError) as exc:
        raise RebenchmarkReadinessError(
            "gateway staging secret is unavailable"
        ) from exc
    if not isinstance(value, Mapping) or not value:
        raise RebenchmarkReadinessError("gateway staging secret is invalid")
    environment: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "")
        if not ENV_RE.fullmatch(key) or isinstance(raw_value, (dict, list)):
            raise RebenchmarkReadinessError(
                "gateway staging environment is invalid"
            )
        environment[key] = "" if raw_value is None else str(raw_value)
    if environment.get("LEADPOET_PARITY_CANDIDATE_SHA") != candidate_sha:
        raise RebenchmarkReadinessError(
            "gateway staging secret is bound to another candidate"
        )
    from leadpoet_canonical.production_parity_boundary_v2 import (
        validate_production_parity_boundary_document_v2,
    )

    boundary = validate_production_parity_boundary_document_v2(
        environment, network="finney", netuid=71
    )
    if boundary.get("mode") != "production-parity":
        raise RebenchmarkReadinessError(
            "gateway readiness probe is not bound to disposable state"
        )
    return environment


@contextmanager
def _patched_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _sanitize(value: Mapping[str, Any], *, candidate_sha: str) -> dict[str, Any]:
    available = value.get("available") is True
    reason = str(value.get("reason") or "")
    benchmark_date = str(value.get("benchmark_date") or "")
    report_id = str(value.get("report_id") or "")
    bundle_id = str(value.get("benchmark_bundle_id") or "")
    window_hash = str(value.get("rolling_window_hash") or "")
    commitments = value.get("completion_commitments")
    if not isinstance(commitments, Mapping):
        commitments = {}
    category_counts = commitments.get("category_counts")
    strength_counts = commitments.get("category_strength_counts")
    try:
        all_icp_count = int(commitments.get("all_icp_count") or 0)
        minimum_score = float(commitments.get("minimum_icp_score"))
        maximum_score = float(commitments.get("maximum_icp_score"))
        normalized_category_counts = {
            name: int(category_counts.get(name) or 0)
            for name in ("public", "private", "conditional")
        }
        normalized_strength_counts = {
            name: {
                str(label): int(count)
                for label, count in dict(strength_counts.get(name) or {}).items()
            }
            for name in ("public", "private", "conditional")
        }
    except (AttributeError, TypeError, ValueError):
        all_icp_count = 0
        minimum_score = math.nan
        maximum_score = math.nan
        normalized_category_counts = {}
        normalized_strength_counts = {}
    if (
        not available
        or reason != "daily_baseline_published"
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", benchmark_date)
        or not report_id
        or not bundle_id
        or not HASH_RE.fullmatch(window_hash)
        or all_icp_count <= 0
        or not isinstance(category_counts, Mapping)
        or sum(normalized_category_counts.values()) != all_icp_count
        or not isinstance(strength_counts, Mapping)
        or not HASH_RE.fullmatch(
            str(commitments.get("per_icp_summaries_hash") or "")
        )
        or not HASH_RE.fullmatch(
            str(commitments.get("category_assignment_hash") or "")
        )
        or not HASH_RE.fullmatch(
            str(commitments.get("conditional_policy_hash") or "")
        )
        or not math.isfinite(minimum_score)
        or not math.isfinite(maximum_score)
        or not 0.0 <= minimum_score <= maximum_score <= 100.0
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_sha": candidate_sha,
            "available": False,
            "reason": reason or "daily_baseline_not_published",
            "benchmark_date": benchmark_date,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_sha": candidate_sha,
        "available": True,
        "reason": reason,
        "benchmark_date": benchmark_date,
        "report_id": report_id,
        "benchmark_bundle_id": bundle_id,
        "rolling_window_hash": window_hash,
        "completion_commitments": {
            "all_icp_count": all_icp_count,
            "per_icp_summaries_hash": str(
                commitments["per_icp_summaries_hash"]
            ),
            "category_assignment_hash": str(
                commitments["category_assignment_hash"]
            ),
            "conditional_policy_hash": str(
                commitments["conditional_policy_hash"]
            ),
            "category_counts": normalized_category_counts,
            "category_strength_counts": normalized_strength_counts,
            "minimum_icp_score": minimum_score,
            "maximum_icp_score": maximum_score,
        },
    }


async def check(
    *, root: Path, candidate_sha: str, secret_id: str
) -> dict[str, Any]:
    if not SHA_RE.fullmatch(candidate_sha):
        raise RebenchmarkReadinessError("candidate SHA is invalid")
    root = root.resolve()
    _checkout_identity(root, candidate_sha)
    environment = _secret_environment(secret_id, candidate_sha)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    with _patched_environment(environment):
        from gateway.research_lab.config import ResearchLabGatewayConfig
        from gateway.research_lab.daily_baseline_readiness import (
            autoresearch_daily_baseline_readiness,
        )

        readiness = await autoresearch_daily_baseline_readiness(
            ResearchLabGatewayConfig.from_env(),
            include_commitments=True,
        )
    return _sanitize(readiness, candidate_sha=candidate_sha)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--secret-id", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    try:
        result = asyncio.run(
            check(
                root=args.root,
                candidate_sha=str(args.candidate_sha).lower(),
                secret_id=str(args.secret_id),
            )
        )
    except (OSError, ValueError, BotoCoreError, ClientError, RebenchmarkReadinessError):
        print("ERROR: candidate rebenchmark readiness could not be proven", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("available") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
