#!/usr/bin/env python3
"""Create one disposable gateway secret for production-parity validation.

The run keeps production provider and model-read configuration, but replaces
every mutable state boundary with the disposable database and disables miner,
autoresearch, promotion, telemetry, and credential-management paths. Secret
values are never written to stdout or to the evidence state file.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_canonical.production_parity_boundary_v2 import (  # noqa: E402
    validate_production_parity_boundary_document_v2,
)


RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SECRET_PREFIX = "leadpoet/staging/production-parity"

_DROP_EXACT = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "LEADPOET_SENTRY_API_TOKEN",
    "LEADPOET_SENTRY_DSN",
    "GATEWAY_OTEL_TOKEN",
    "RESEARCH_LAB_OPENROUTER_MANAGEMENT_KEY",
    "OPENROUTER_MANAGEMENT_KEY",
    "OPENROUTER_API_MANAGEMENT_KEY",
    "OR_MANAGEMENT_KEY",
}
_DROP_NAME_RE = re.compile(
    r"(?:^|_)(?:MNEMONIC|PRIVATE_KEY|SECRET_SEED|SEED_PHRASE|WALLET_PASSWORD)(?:_|$)"
)
_FORCED_KEYS = {
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "BITTENSOR_NETWORK",
    "BITTENSOR_NETUID",
    "SUBTENSOR_NETWORK",
    "NETUID",
    "EXPECTED_CHAIN",
    "VALIDATOR_SUBTENSOR_NETWORK",
    "VALIDATOR_NETUID",
    "GATEWAY_URL",
    "VALIDATOR_V2_GATEWAY_URL",
    "LEADPOET_PARITY_CANDIDATE_SHA",
    "LEADPOET_PRODUCTION_PARITY_MODE",
    "LEADPOET_PRODUCTION_PARITY_RUN_ID",
    "LEADPOET_PRODUCTION_PARITY_SUPABASE_ORIGIN",
    "LEADPOET_PRODUCTION_PARITY_BENCHMARK_DATE",
    "RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET",
    "RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED",
    "ENABLE_FULFILLMENT",
    "RESEARCH_LAB_AUTO_START_WORKERS",
    "RESEARCH_LAB_AUTO_START_HOSTED_WORKERS",
    "RESEARCH_LAB_AUTO_START_SCORING_WORKERS",
    "RESEARCH_LAB_HOSTED_RUNS_ENABLED",
    "RESEARCH_LAB_HOSTED_WORKER_ENABLED",
    "RESEARCH_LAB_HOSTED_WORKER_DRY_RUN",
    "RESEARCH_LAB_HOSTED_WORKER_MAX_RUNS",
    "RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED",
}


class SecretMaterializationError(RuntimeError):
    """A source secret or run-scoped replacement is incomplete or unsafe."""


def _parse_environment_document(raw: str, *, field: str) -> dict[str, str]:
    """Parse the JSON or shell-assignment format used by production secrets."""

    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    rows: Sequence[tuple[object, object]]
    if isinstance(parsed, Mapping):
        rows = list(parsed.items())
    else:
        shell_rows: list[tuple[object, object]] = []
        for raw_line in raw.replace("\x00", "\n").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                raise SecretMaterializationError(
                    f"{field} contains a non-assignment"
                )
            key, value = line.split("=", 1)
            try:
                tokens = shlex.split(value.strip(), comments=False, posix=True)
            except ValueError as exc:
                raise SecretMaterializationError(
                    f"{field} contains invalid shell quoting"
                ) from exc
            if not tokens and not value.strip():
                normalized = ""
            elif len(tokens) == 1:
                normalized = tokens[0]
            else:
                raise SecretMaterializationError(
                    f"{field} contains an unquoted multi-token value"
                )
            shell_rows.append((key.strip(), normalized))
        rows = shell_rows

    values: dict[str, str] = {}
    for raw_key, raw_value in rows:
        key = str(raw_key or "").strip()
        if not ENV_KEY_RE.fullmatch(key) or key in values:
            raise SecretMaterializationError(f"{field} contains an invalid key")
        if isinstance(raw_value, (dict, list)):
            value = json.dumps(raw_value, sort_keys=True, separators=(",", ":"))
        else:
            value = "" if raw_value is None else str(raw_value)
        values[key] = value
    if not values:
        raise SecretMaterializationError(f"{field} is empty")
    return values


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _jwt(secret: str, role: str) -> str:
    now = int(time.time())
    header = _b64url(b'{"alg":"HS256","typ":"JWT"}')
    payload = _b64url(
        json.dumps(
            {
                "aud": "authenticated",
                "exp": now + 86400 * 2,
                "iat": now - 5,
                "iss": "leadpoet-production-parity",
                "role": role,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = hmac.new(
        secret.encode("ascii"), signing_input, hashlib.sha256
    ).digest()
    return f"{header}.{payload}.{_b64url(signature)}"


def _secret_string(client: Any, secret_id: str) -> str:
    try:
        value = client.get_secret_value(SecretId=secret_id).get("SecretString")
    except (BotoCoreError, ClientError) as exc:
        raise SecretMaterializationError(
            "production gateway environment is unavailable"
        ) from exc
    if not isinstance(value, str) or not value:
        raise SecretMaterializationError(
            "production gateway environment has no string value"
        )
    return value


def secret_name(run_id: str) -> str:
    if not RUN_RE.fullmatch(run_id):
        raise SecretMaterializationError("parity run identity is invalid")
    return f"{SECRET_PREFIX}/{run_id}/gateway"


def build_gateway_environment(
    source: Mapping[str, str],
    *,
    run_id: str,
    candidate_sha: str,
    supabase_origin: str,
    artifact_bucket: str,
    benchmark_date: str,
    jwt_secret: str,
) -> dict[str, str]:
    if not RUN_RE.fullmatch(run_id) or not SHA_RE.fullmatch(candidate_sha):
        raise SecretMaterializationError("parity run or candidate identity is invalid")
    if not DATE_RE.fullmatch(benchmark_date):
        raise SecretMaterializationError("parity benchmark date is invalid")
    if not re.fullmatch(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", artifact_bucket):
        raise SecretMaterializationError("parity artifact bucket is invalid")
    boundary_environment = {
        "LEADPOET_PRODUCTION_PARITY_MODE": "enabled",
        "LEADPOET_PRODUCTION_PARITY_RUN_ID": run_id,
        "LEADPOET_PRODUCTION_PARITY_SUPABASE_ORIGIN": supabase_origin,
        "LEADPOET_PRODUCTION_PARITY_BENCHMARK_DATE": benchmark_date,
    }
    validate_production_parity_boundary_document_v2(
        boundary_environment,
        network="finney",
        netuid=71,
    )

    result = {
        str(key): str(value)
        for key, value in source.items()
        if key not in _DROP_EXACT
        and not _DROP_NAME_RE.search(key)
        and key not in _FORCED_KEYS
    }
    generated = {
        "SUPABASE_URL": supabase_origin.rstrip("/"),
        "SUPABASE_ANON_KEY": _jwt(jwt_secret, "anon"),
        "SUPABASE_SERVICE_ROLE_KEY": _jwt(jwt_secret, "service_role"),
        "BITTENSOR_NETWORK": "finney",
        "BITTENSOR_NETUID": "71",
        "SUBTENSOR_NETWORK": "finney",
        "NETUID": "71",
        "EXPECTED_CHAIN": "wss://entrypoint-finney.opentensor.ai:443",
        "VALIDATOR_SUBTENSOR_NETWORK": "finney",
        "VALIDATOR_NETUID": "71",
        "GATEWAY_URL": "http://127.0.0.1:8000",
        "VALIDATOR_V2_GATEWAY_URL": "http://127.0.0.1:8000",
        "LEADPOET_PARITY_CANDIDATE_SHA": candidate_sha,
        **boundary_environment,
        "RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET": artifact_bucket,
        "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true",
        "LEADPOET_SENTRY_ENABLED": "0",
        "GATEWAY_OTEL_ENABLED": "0",
        "GATEWAY_OTEL_ENDPOINT": "",
        "GATEWAY_OTEL_METRICS_ENDPOINT": "",
        # The clone must use real persistence and scoring code, but no path may
        # accept miners, create loops, mutate Git/model pointers, or promote.
        "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED": "true",
        "RESEARCH_LAB_GATEWAY_API_ENABLED": "true",
        "RESEARCH_LAB_AUTO_START_WORKERS": "true",
        "RESEARCH_LAB_AUTO_START_HOSTED_WORKERS": "true",
        "RESEARCH_LAB_AUTO_START_SCORING_WORKERS": "true",
        "RESEARCH_LAB_SCORING_WORKER_ENABLED": "true",
        "RESEARCH_LAB_PRIVATE_BASELINE_REBENCHMARK_ENABLED": "true",
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
        "RESEARCH_LAB_PAID_LOOPS_ENABLED": "false",
        "RESEARCH_LAB_LOOP_TOPUPS_ENABLED": "false",
        # Full V2 startup requires the production fleet topology. Hosted
        # processes start in dry-run mode so copied queue rows are never
        # claimed or executed.
        "RESEARCH_LAB_HOSTED_RUNS_ENABLED": "true",
        "RESEARCH_LAB_HOSTED_WORKER_ENABLED": "true",
        "RESEARCH_LAB_HOSTED_WORKER_DRY_RUN": "true",
        "RESEARCH_LAB_HOSTED_WORKER_MAX_RUNS": "0",
        # Intake is exercised explicitly after the rebenchmark and weight
        # proofs. Nothing may claim its queued provenance work in this run.
        "RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED": "false",
        "RESEARCH_LAB_AUTO_PROMOTION_ENABLED": "false",
        "RESEARCH_LAB_AUTO_COMMIT_ENABLED": "false",
        "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": "true",
        "RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED": "false",
        "RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED": "false",
        "ENABLE_FULFILLMENT": "false",
    }
    result.update(generated)
    if any(not ENV_KEY_RE.fullmatch(key) for key in result):
        raise SecretMaterializationError("parity gateway environment has an invalid key")
    if any(key in result for key in _DROP_EXACT):
        raise SecretMaterializationError("parity gateway retained a forbidden credential")
    return dict(sorted(result.items()))


def create(
    *,
    client: Any,
    source_secret_id: str,
    run_id: str,
    candidate_sha: str,
    supabase_origin: str,
    artifact_bucket: str,
    benchmark_date: str,
    jwt_secret: str | None = None,
) -> dict[str, Any]:
    source = _parse_environment_document(
        _secret_string(client, source_secret_id),
        field="production gateway environment",
    )
    jwt_secret = jwt_secret or secrets.token_urlsafe(48)
    environment = build_gateway_environment(
        source,
        run_id=run_id,
        candidate_sha=candidate_sha,
        supabase_origin=supabase_origin,
        artifact_bucket=artifact_bucket,
        benchmark_date=benchmark_date,
        jwt_secret=jwt_secret,
    )
    name = secret_name(run_id)
    try:
        client.create_secret(
            Name=name,
            Description=f"Disposable Leadpoet production-parity gateway {run_id}",
            SecretString=json.dumps(environment, sort_keys=True, separators=(",", ":")),
            Tags=[
                {"Key": "leadpoet:parity-run", "Value": run_id},
                {"Key": "leadpoet:candidate-sha", "Value": candidate_sha},
                {"Key": "leadpoet:ephemeral", "Value": "true"},
            ],
        )
    except (BotoCoreError, ClientError) as exc:
        raise SecretMaterializationError("run-scoped gateway secret creation failed") from exc
    return {
        "schema_version": "leadpoet.production_parity_secret_state.v2",
        "candidate_sha": candidate_sha,
        "run_id": run_id,
        "secret_id": name,
        "benchmark_date": benchmark_date,
        "environment_key_count": len(environment),
        "source_key_count": len(source),
        # Passed in memory to the local PostgREST process; never persisted in
        # the redacted state document emitted by the CLI.
        "_jwt_secret": jwt_secret,
    }


def delete(*, client: Any, run_id: str) -> dict[str, Any]:
    name = secret_name(run_id)
    try:
        client.delete_secret(SecretId=name, ForceDeleteWithoutRecovery=True)
        deleted = True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            deleted = False
        else:
            raise SecretMaterializationError(
                "run-scoped gateway secret deletion failed"
            ) from exc
    except BotoCoreError as exc:
        raise SecretMaterializationError(
            "run-scoped gateway secret deletion failed"
        ) from exc
    return {"run_id": run_id, "secret_id": name, "deleted": deleted}


def _write_redacted(path: Path, value: Mapping[str, Any]) -> None:
    document = {key: item for key, item in value.items() if not key.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o600)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--source-secret-id", required=True)
    create_parser.add_argument("--run-id", required=True)
    create_parser.add_argument("--candidate-sha", required=True)
    create_parser.add_argument("--supabase-origin", required=True)
    create_parser.add_argument("--artifact-bucket", required=True)
    create_parser.add_argument("--benchmark-date", required=True)
    create_parser.add_argument("--state", type=Path, required=True)
    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("--run-id", required=True)
    delete_parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    client = boto3.client("secretsmanager", region_name=args.region)
    try:
        if args.command == "create":
            result = create(
                client=client,
                source_secret_id=args.source_secret_id,
                run_id=args.run_id,
                candidate_sha=args.candidate_sha.lower(),
                supabase_origin=args.supabase_origin,
                artifact_bucket=args.artifact_bucket,
                benchmark_date=args.benchmark_date,
            )
        else:
            result = delete(client=client, run_id=args.run_id)
        _write_redacted(args.state, result)
    except (OSError, ValueError, SecretMaterializationError):
        print("ERROR: production-parity secret operation failed", file=sys.stderr)
        return 1
    print(json.dumps({key: value for key, value in result.items() if not key.startswith("_")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
