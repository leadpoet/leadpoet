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
RUN_SECRET_RE = re.compile(r"^pp-[1-9][0-9]*-[1-9][0-9]*$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SECRET_PREFIX = "leadpoet/staging/production-parity/runs"

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
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_BASE_URL",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MINIO_ENDPOINT",
    "MINIO_BUCKET",
}
_DROP_NAME_RE = re.compile(
    r"(?:^|_)(?:MNEMONIC|PRIVATE_KEY|SECRET_SEED|SEED_PHRASE|WALLET_PASSWORD)(?:_|$)"
)
_DROP_PROCESS_CONTROL_EXACT = frozenset(
    {
        "ALL_PROXY",
        "AWS_ACCESS_KEY_ID",
        "AWS_CA_BUNDLE",
        "AWS_CONFIG_FILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_DEFAULT_REGION",
        "AWS_EC2_METADATA_DISABLED",
        "AWS_ENDPOINT_URL",
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SECURITY_TOKEN",
        "AWS_SESSION_TOKEN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "BASH_ENV",
        "BOTO_CONFIG",
        "CDPATH",
        "CURL_CA_BUNDLE",
        "ENV",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_WORK_TREE",
        "GATEWAY_DEPLOY_COMMIT",
        "GATEWAY_DEPLOY_PLAN_FILE",
        "GATEWAY_DEPLOYMENT_DIR",
        "GATEWAY_DEPLOYMENT_MANIFEST",
        "GATEWAY_ENV_FILE",
        "GATEWAY_EXACT_COMMIT_HELPER",
        "GATEWAY_GIT_HELPER",
        "GATEWAY_HOST_MEMORY_GUARD_PATH",
        "GATEWAY_HOST_RESTART_SCRIPT",
        "GATEWAY_LAST_GOOD_MANIFEST",
        "GATEWAY_LOG_FILE",
        "GATEWAY_LOG_ROOT",
        "GATEWAY_PRIVATE_KEY_PATH",
        "ARWEAVE_KEYFILE_PATH",
        "GATEWAY_PREPARED_V2_RELEASE_LINEAGE",
        "GATEWAY_PREPARED_V2_RELEASE_MANIFEST",
        "GATEWAY_PYTHON_BIN",
        "GATEWAY_RESTART_CONTROLLER_ROOT",
        "GATEWAY_RESTART_GIT_SSH_COMMAND",
        "GATEWAY_RESTART_LOCK_FILE",
        "GATEWAY_RESTART_RECOVERY_LOCK_FILE",
        "GATEWAY_RESTART_TIMING_DIR",
        "GATEWAY_RESTART_TIMING_FILE",
        "GATEWAY_ROOT",
        "GATEWAY_STATEFUL_CUTOVER_MANIFEST",
        "GATEWAY_STATEFUL_CUTOVER_VALIDATOR_RELEASE_MANIFEST",
        "GATEWAY_TEE_EIF_ROOT",
        "GATEWAY_V2_ACCEPTANCE_CORPUS_MANIFEST",
        "GATEWAY_V2_ACCEPTANCE_CORPUS_ROOT",
        "GATEWAY_V2_ARTIFACT_POLICY",
        "GATEWAY_V2_CONFIG_DIR",
        "GATEWAY_V2_KMS_KEY_ID",
        "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT",
        "GATEWAY_V2_RELEASE_ARCHIVE_ROOT",
        "GATEWAY_V2_RELEASE_BUCKET",
        "GATEWAY_V2_RELEASE_LINEAGE",
        "GATEWAY_V2_RELEASE_MANIFEST",
        "GATEWAY_V2_RELEASE_PREFIX",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "IFS",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
        "LEADPOET_GATEWAY_ENV_SECRET_ID",
        "LEADPOET_REPO_ROOT",
        "LEADPOET_RESTART_INVOCATION_ID",
        "LEADPOET_RESTART_START_PATH",
        "LEADPOET_SUBNET_EPOCH_CUTOVER_JSON",
        "LEADPOET_SUBNET_EPOCH_CUTOVER_PATH",
        "LOGNAME",
        "NO_PROXY",
        "PATH",
        "PYTHONBREAKPOINT",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONWARNINGS",
        "REQUESTS_CA_BUNDLE",
        "SHELL",
        "SHELLOPTS",
        "SSH_AUTH_SOCK",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USER",
        "VIRTUAL_ENV",
        "VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT",
        "all_proxy",
        "https_proxy",
        "http_proxy",
        "no_proxy",
    }
)
_DROP_PROCESS_CONTROL_PREFIXES = (
    "AWS_ENDPOINT_URL_",
    "DYLD_",
    "GIT_CONFIG_KEY_",
    "GIT_CONFIG_VALUE_",
)
_FORCED_KEYS = {
    "AWS_S3_BUCKET",
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
    "GATEWAY_PUBLIC_KEY",
    "GATEWAY_PRIVATE_KEY_PASSWORD",
    "VALIDATOR_V2_GATEWAY_URL",
    "DISABLE_BACKGROUND_TASKS",
    "GATEWAY_STATEFUL_CUTOVER_CEREMONY",
    "GATEWAY_TEE_TOPOLOGY_MODE",
    "LANGFUSE_ENABLED",
    "LAB_ARENA_MODE",
    "LAB_ARENA_SUPABASE_URL",
    "LAB_ARENA_SUPABASE_ANON_KEY",
    "LAB_ARENA_SERVICE_JWT",
    "LAB_ARENA_BUCKET",
    "LEADPOET_PARITY_CANDIDATE_SHA",
    "LEADPOET_PRODUCTION_PARITY_MODE",
    "LEADPOET_PRODUCTION_PARITY_RUN_ID",
    "LEADPOET_PRODUCTION_PARITY_SUPABASE_ORIGIN",
    "LEADPOET_PRODUCTION_PARITY_BENCHMARK_DATE",
    "RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET",
    "RESEARCH_LAB_CORPUS_EXPORT_ENABLED",
    "RESEARCH_LAB_CORPUS_EXPORT_S3_PREFIX",
    "RESEARCH_LAB_EVIDENCE_PROXY_URL",
    "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR",
    "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH",
    "RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH",
    "RESEARCH_LAB_SCORE_BUNDLE_SIGNATURE_URI_PREFIX",
    "RESEARCH_LAB_SCORING_CACHE_DIR",
    "RESEARCH_LAB_RAW_TRACE_S3_PREFIX",
    "RESEARCH_LAB_SCORER_TRACE_S3_PREFIX",
    "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX",
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


def production_parity_trace_prefixes(
    *, artifact_bucket: str, run_id: str
) -> dict[str, str]:
    """Return the only trace destinations permitted inside a parity run."""

    if not RUN_RE.fullmatch(run_id):
        raise SecretMaterializationError("parity run identity is invalid")
    if not re.fullmatch(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", artifact_bucket):
        raise SecretMaterializationError("parity artifact bucket is invalid")
    root = f"s3://{artifact_bucket}/production-parity/runs/{run_id}/traces"
    return {
        "RESEARCH_LAB_RAW_TRACE_S3_PREFIX": f"{root}/raw",
        "RESEARCH_LAB_SCORER_TRACE_S3_PREFIX": f"{root}/scorer",
        "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX": f"{root}/incontainer",
    }


def production_parity_scoring_cache_dir(*, run_id: str) -> str:
    """Return the disposable local scoring cache owned by one parity run."""

    if not RUN_RE.fullmatch(run_id):
        raise SecretMaterializationError("parity run identity is invalid")
    return f"/opt/leadpoet-production-parity/{run_id}/runtime/scoring-cache"


def is_process_control_environment_key(key: str) -> bool:
    return key in _DROP_PROCESS_CONTROL_EXACT or key.startswith(
        _DROP_PROCESS_CONTROL_PREFIXES
    )


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
            try:
                tokens = shlex.split(line, comments=False, posix=True)
            except ValueError:
                tokens = [line]
            assignment = tokens[0] if len(tokens) == 1 else line
            if "=" not in assignment:
                raise SecretMaterializationError(
                    f"{field} contains a non-assignment"
                )
            key, value = assignment.split("=", 1)
            shell_rows.append((key.strip(), value))
        rows = shell_rows

    values: dict[str, str] = {}
    for raw_key, raw_value in rows:
        key = str(raw_key or "").strip()
        if not ENV_KEY_RE.fullmatch(key):
            raise SecretMaterializationError(f"{field} contains an invalid key")
        if isinstance(raw_value, (dict, list)):
            value = json.dumps(raw_value, sort_keys=True, separators=(",", ":"))
        else:
            value = "" if raw_value is None else str(raw_value)
        if key in values:
            if values[key] != value:
                raise SecretMaterializationError(
                    f"{field} contains a conflicting duplicate assignment"
                )
            continue
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
    if not RUN_SECRET_RE.fullmatch(run_id):
        raise SecretMaterializationError("parity run identity is invalid")
    return f"{SECRET_PREFIX}/{run_id}/gateway"


def build_gateway_environment(
    source: Mapping[str, str],
    *,
    run_id: str,
    candidate_sha: str,
    gateway_public_key: str,
    supabase_origin: str,
    artifact_bucket: str,
    benchmark_date: str,
    jwt_secret: str,
) -> dict[str, str]:
    if not RUN_RE.fullmatch(run_id) or not SHA_RE.fullmatch(candidate_sha):
        raise SecretMaterializationError("parity run or candidate identity is invalid")
    if not DATE_RE.fullmatch(benchmark_date):
        raise SecretMaterializationError("parity benchmark date is invalid")
    normalized_gateway_public_key = str(gateway_public_key or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized_gateway_public_key) is None:
        raise SecretMaterializationError("parity gateway public key is invalid")
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
        and not is_process_control_environment_key(str(key))
        and key not in _FORCED_KEYS
    }
    generated = {
        "AWS_S3_BUCKET": artifact_bucket,
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
        "GATEWAY_PUBLIC_KEY": normalized_gateway_public_key,
        "GATEWAY_PRIVATE_KEY_PASSWORD": "",
        "VALIDATOR_V2_GATEWAY_URL": "http://127.0.0.1:8000",
        "DISABLE_BACKGROUND_TASKS": "true",
        "GATEWAY_STATEFUL_CUTOVER_CEREMONY": "0",
        "GATEWAY_TEE_TOPOLOGY_MODE": "full",
        "LANGFUSE_ENABLED": "false",
        # The restarted gateway must not inherit a production Arena process.
        # The full-parity runner starts one isolated shadow service explicitly
        # after the exact-candidate gateway is ready.
        "LAB_ARENA_MODE": "off",
        "LAB_ARENA_SUPABASE_URL": supabase_origin.rstrip("/"),
        "LAB_ARENA_SUPABASE_ANON_KEY": _jwt(jwt_secret, "anon"),
        "LAB_ARENA_SERVICE_JWT": _jwt(jwt_secret, "lab_arena_service"),
        "LAB_ARENA_BUCKET": artifact_bucket,
        "LEADPOET_PARITY_CANDIDATE_SHA": candidate_sha,
        **boundary_environment,
        "RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET": artifact_bucket,
        "RESEARCH_LAB_CORPUS_EXPORT_ENABLED": "false",
        "RESEARCH_LAB_CORPUS_EXPORT_S3_PREFIX": "",
        "RESEARCH_LAB_EVIDENCE_PROXY_URL": "",
        "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR": "",
        "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH": "",
        "RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH": "",
        "RESEARCH_LAB_SCORE_BUNDLE_SIGNATURE_URI_PREFIX": "",
        "RESEARCH_LAB_SCORING_CACHE_DIR": production_parity_scoring_cache_dir(
            run_id=run_id
        ),
        **production_parity_trace_prefixes(
            artifact_bucket=artifact_bucket,
            run_id=run_id,
        ),
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
    gateway_public_key: str,
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
        gateway_public_key=gateway_public_key,
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
    create_parser.add_argument("--gateway-public-key", required=True)
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
                gateway_public_key=args.gateway_public_key,
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
