#!/usr/bin/env python3
"""Install two fixed parity secrets through a 15-minute validator bootstrap role.

The read-only DSN arrives only on an already-open anonymous descriptor.  The
BuiltWith credential is selected by exact name from the running validator
container without rendering or returning the rest of its environment.
Neither credential is accepted through argv, environment variables, or files.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, unquote, urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError


EXPECTED_ACCOUNT_ID = "493765492819"
EXPECTED_PROJECT_REF = "qplwoislplkcegvdmbim"
EXPECTED_REGION = "us-east-1"
EXPECTED_POOLER_HOST = "aws-0-us-east-1.pooler.supabase.com"
VALIDATOR_ROLE = "leadpoet-validator-s3-cloudwatch-role"
BOOTSTRAP_ROLE = "leadpoet-production-parity-static-bootstrap"
DEFAULT_READONLY_SECRET_ID = "leadpoet/staging/production-parity/readonly-dsn"
DEFAULT_MINER_INTAKE_SECRET_ID = (
    "leadpoet/staging/production-parity-miner-intake"
)
STATIC_DESCRIPTIONS = {
    DEFAULT_READONLY_SECRET_ID: (
        "Leadpoet production-parity read-only source DSN"
    ),
    DEFAULT_MINER_INTAKE_SECRET_ID: (
        "Leadpoet production-parity miner-intake provider credential"
    ),
}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
SECRET_RE = re.compile(r"^[A-Za-z0-9/_+=.@-]{6,512}$")
ROLE_RE = re.compile(r"^[A-Za-z0-9+=,.@_-]{1,64}$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class StaticSecretInstallError(RuntimeError):
    """A fixed identity, authority, or secret contract failed."""


def _require_anonymous_fd(descriptor: int, *, description: str) -> None:
    if descriptor < 3:
        raise StaticSecretInstallError(
            f"{description} descriptor must be an inherited pipe or socket"
        )
    mode = os.fstat(descriptor).st_mode
    if not (stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)) or os.isatty(descriptor):
        raise StaticSecretInstallError(
            f"{description} descriptor must be an inherited pipe or socket"
        )


def _write_all(descriptor: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise StaticSecretInstallError("anonymous descriptor write failed")
        view = view[written:]


def _read_request(descriptor: int) -> dict[str, Any]:
    _require_anonymous_fd(descriptor, description="request")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, 16384)
        if not chunk:
            break
        total += len(chunk)
        if total > 16384:
            raise StaticSecretInstallError("static bootstrap request is too large")
        chunks.append(chunk)
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise StaticSecretInstallError("static bootstrap request is invalid") from exc
    if not isinstance(value, dict):
        raise StaticSecretInstallError("static bootstrap request is invalid")
    return value


def _validate_builtwith_key(value: str) -> str:
    normalized = str(value or "").strip()
    if (
        not 8 <= len(normalized) <= 512
        or any(character.isspace() for character in normalized)
        or "\x00" in normalized
    ):
        raise StaticSecretInstallError("BuiltWith credential is invalid")
    return normalized


def _builtwith_from_validator_container(container_name: str) -> str:
    if container_name != "leadpoet-validator-main":
        raise StaticSecretInstallError("validator container identity differs")
    template = (
        '{{range .Config.Env}}{{if eq (index (split . "=") 0) '
        '"BUILTWITH_API_KEY"}}{{println .}}{{end}}{{end}}'
    )
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", template, container_name],
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StaticSecretInstallError(
            "validator BuiltWith credential read failed"
        ) from exc
    prefix = "BUILTWITH_API_KEY="
    lines = [line for line in result.stdout.splitlines() if line]
    if (
        result.returncode != 0
        or len(lines) != 1
        or not lines[0].startswith(prefix)
    ):
        raise StaticSecretInstallError(
            "validator BuiltWith credential is unavailable"
        )
    return _validate_builtwith_key(lines[0][len(prefix) :])


def _validate_readonly_dsn(value: str) -> str:
    dsn = str(value or "").strip()
    try:
        parsed = urlparse(dsn)
        query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
        port = parsed.port or 5432
    except ValueError as exc:
        raise StaticSecretInstallError("read-only parity DSN is invalid") from exc
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or str(parsed.hostname or "").lower() != EXPECTED_POOLER_HOST
        or port != 5432
        or unquote(str(parsed.username or ""))
        != f"leadpoet_parity_reader.{EXPECTED_PROJECT_REF}"
        or not HASH_RE.fullmatch(unquote(str(parsed.password or "")))
        or unquote(parsed.path.lstrip("/")) != "postgres"
        or query != [("sslmode", "require")]
        or parsed.fragment
    ):
        raise StaticSecretInstallError("read-only parity DSN is invalid")
    return dsn


def _secret_value(client: Any, secret_id: str) -> str | None:
    try:
        value = client.get_secret_value(SecretId=secret_id).get("SecretString")
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            return None
        raise StaticSecretInstallError("required Secrets Manager read failed") from exc
    except BotoCoreError as exc:
        raise StaticSecretInstallError("required Secrets Manager read failed") from exc
    if not isinstance(value, str) or not value:
        raise StaticSecretInstallError("required secret has no string value")
    return value


def _static_secret_value(client: Any, secret_id: str) -> str | None:
    value = _secret_value(client, secret_id)
    if value is None:
        return None
    try:
        description = client.describe_secret(SecretId=secret_id)
    except (BotoCoreError, ClientError) as exc:
        raise StaticSecretInstallError("static secret metadata read failed") from exc
    tags = {
        str(item.get("Key") or ""): str(item.get("Value") or "")
        for item in description.get("Tags", [])
        if isinstance(item, Mapping)
    }
    try:
        resource_policy = client.get_resource_policy(SecretId=secret_id).get(
            "ResourcePolicy"
        )
    except (BotoCoreError, ClientError) as exc:
        raise StaticSecretInstallError("static secret policy read failed") from exc
    versions: list[Mapping[str, Any]] = []
    next_token: str | None = None
    seen_tokens: set[str] = set()
    try:
        while True:
            request: dict[str, Any] = {
                "SecretId": secret_id,
                "IncludeDeprecated": True,
                "MaxResults": 100,
            }
            if next_token:
                request["NextToken"] = next_token
            response = client.list_secret_version_ids(**request)
            page = response.get("Versions", [])
            if not isinstance(page, list) or not all(
                isinstance(item, Mapping) for item in page
            ):
                raise StaticSecretInstallError(
                    "static secret version inventory differs"
                )
            versions.extend(page)
            next_value = str(response.get("NextToken") or "")
            if not next_value:
                break
            if next_value in seen_tokens:
                raise StaticSecretInstallError(
                    "static secret version inventory differs"
                )
            seen_tokens.add(next_value)
            next_token = next_value
    except (BotoCoreError, ClientError) as exc:
        raise StaticSecretInstallError("static secret version read failed") from exc
    expected_arn = re.compile(
        rf"^arn:aws:secretsmanager:{re.escape(EXPECTED_REGION)}:"
        rf"{EXPECTED_ACCOUNT_ID}:secret:{re.escape(secret_id)}-[A-Za-z0-9]{{6}}$"
    )
    if (
        description.get("Name") != secret_id
        or description.get("Description") != STATIC_DESCRIPTIONS.get(secret_id)
        or expected_arn.fullmatch(str(description.get("ARN") or "")) is None
        or set(tags) != {
            "leadpoet:purpose",
            "leadpoet:parity-static-bootstrap",
            "leadpoet:candidate-sha",
        }
        or tags.get("leadpoet:purpose") != "production-parity-static"
        or tags.get("leadpoet:parity-static-bootstrap") != "true"
        or SHA_RE.fullmatch(tags.get("leadpoet:candidate-sha", "")) is None
        or description.get("RotationEnabled") is True
        or description.get("RotationLambdaARN")
        or description.get("LastRotatedDate")
        or description.get("DeletedDate")
        or description.get("OwningService")
        or description.get("PrimaryRegion")
        or description.get("ReplicationStatus")
        or description.get("KmsKeyId") not in (None, "", "alias/aws/secretsmanager")
        or resource_policy
        or len(versions) != 1
        or set(versions[0].get("VersionStages") or []) != {"AWSCURRENT"}
        or len(versions[0].get("VersionStages") or []) != 1
    ):
        raise StaticSecretInstallError("existing static secret ownership differs")
    return value


def _instance_bootstrap_client(
    *, validator_role: str
) -> tuple[Any, str, str]:
    forbidden = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
    }
    if any(os.environ.get(name) for name in forbidden):
        raise StaticSecretInstallError(
            "installer requires the validator EC2 instance role"
        )
    session = boto3.session.Session(region_name=EXPECTED_REGION)
    credentials = session.get_credentials()
    if credentials is None or credentials.method != "iam-role":
        raise StaticSecretInstallError(
            "validator EC2 instance-role credentials are unavailable"
        )
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    instance_arn = str(identity.get("Arn") or "")
    expected_instance = (
        f"arn:aws:sts::{EXPECTED_ACCOUNT_ID}:assumed-role/{validator_role}/"
    )
    if (
        str(identity.get("Account") or "") != EXPECTED_ACCOUNT_ID
        or not instance_arn.startswith(expected_instance)
    ):
        raise StaticSecretInstallError("validator instance-role identity differs")
    bootstrap_role_arn = (
        f"arn:aws:iam::{EXPECTED_ACCOUNT_ID}:role/{BOOTSTRAP_ROLE}"
    )
    try:
        assumed = sts.assume_role(
            RoleArn=bootstrap_role_arn,
            RoleSessionName="leadpoet-parity-static-bootstrap",
            DurationSeconds=900,
        )
        temporary = assumed["Credentials"]
        bootstrap_session = boto3.session.Session(
            aws_access_key_id=temporary["AccessKeyId"],
            aws_secret_access_key=temporary["SecretAccessKey"],
            aws_session_token=temporary["SessionToken"],
            region_name=EXPECTED_REGION,
        )
        bootstrap_identity = bootstrap_session.client("sts").get_caller_identity()
    except (BotoCoreError, ClientError, KeyError) as exc:
        raise StaticSecretInstallError(
            "temporary static-bootstrap role could not be assumed"
        ) from exc
    bootstrap_arn = str(bootstrap_identity.get("Arn") or "")
    expected_bootstrap = (
        f"arn:aws:sts::{EXPECTED_ACCOUNT_ID}:assumed-role/{BOOTSTRAP_ROLE}/"
    )
    if (
        str(bootstrap_identity.get("Account") or "") != EXPECTED_ACCOUNT_ID
        or not bootstrap_arn.startswith(expected_bootstrap)
    ):
        raise StaticSecretInstallError("temporary bootstrap identity differs")
    return bootstrap_session.client("secretsmanager"), instance_arn, bootstrap_arn


def _create_static_secret(
    client: Any,
    *,
    name: str,
    description: str,
    value: str,
    commit: str,
) -> None:
    try:
        client.create_secret(
            Name=name,
            Description=description,
            SecretString=value,
            Tags=[
                {"Key": "leadpoet:purpose", "Value": "production-parity-static"},
                {"Key": "leadpoet:parity-static-bootstrap", "Value": "true"},
                {"Key": "leadpoet:candidate-sha", "Value": commit},
            ],
        )
    except (BotoCoreError, ClientError) as exc:
        raise StaticSecretInstallError("static parity secret creation failed") from exc


def install(
    args: argparse.Namespace, request: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = str(request.get("mode") or "")
    if (
        not SHA_RE.fullmatch(args.commit)
        or not HASH_RE.fullmatch(args.migration_sha256)
        or any(
            SECRET_RE.fullmatch(value) is None
            for value in (
                args.readonly_dsn_secret_id,
                args.miner_intake_secret_id,
            )
        )
        or args.readonly_dsn_secret_id == args.miner_intake_secret_id
        or args.readonly_dsn_secret_id != DEFAULT_READONLY_SECRET_ID
        or args.miner_intake_secret_id != DEFAULT_MINER_INTAKE_SECRET_ID
        or request.get("migration_sha256") != args.migration_sha256
        or mode not in {"probe", "ensure"}
    ):
        raise StaticSecretInstallError("static installer inputs are invalid")
    requested_dsn = str(request.get("readonly_dsn") or "")
    readonly_dsn = (
        _validate_readonly_dsn(requested_dsn) if requested_dsn else ""
    )
    if mode == "ensure" and not readonly_dsn:
        raise StaticSecretInstallError("ensure request lacks the read-only DSN")
    client, instance_arn, bootstrap_arn = _instance_bootstrap_client(
        validator_role=VALIDATOR_ROLE
    )
    builtwith_key = _builtwith_from_validator_container(args.validator_container)

    readonly_raw = _static_secret_value(client, args.readonly_dsn_secret_id)
    intake_raw = _static_secret_value(client, args.miner_intake_secret_id)
    existing_dsn = ""
    if readonly_raw is not None:
        try:
            readonly_document = json.loads(readonly_raw)
        except ValueError as exc:
            raise StaticSecretInstallError(
                "existing read-only parity secret is invalid"
            ) from exc
        if not isinstance(readonly_document, Mapping):
            raise StaticSecretInstallError(
                "existing read-only parity secret is invalid"
            )
        existing_dsn = _validate_readonly_dsn(
            str(readonly_document.get("readonly_dsn") or "")
        )
    effective_dsn = readonly_dsn or existing_dsn
    expected_readonly = (
        json.dumps(
            {"readonly_dsn": effective_dsn},
            sort_keys=True,
            separators=(",", ":"),
        )
        if effective_dsn
        else ""
    )
    expected_intake = json.dumps(
        {"builtwith_api_key": builtwith_key},
        sort_keys=True,
        separators=(",", ":"),
    )
    if (
        readonly_raw is not None
        and readonly_dsn
        and not hmac.compare_digest(existing_dsn, readonly_dsn)
    ):
        raise StaticSecretInstallError("existing read-only parity secret differs")
    if intake_raw is not None and not hmac.compare_digest(
        intake_raw, expected_intake
    ):
        raise StaticSecretInstallError("existing miner-intake parity secret differs")
    if mode == "ensure" and readonly_raw is None:
        _create_static_secret(
            client,
            name=args.readonly_dsn_secret_id,
            description=STATIC_DESCRIPTIONS[DEFAULT_READONLY_SECRET_ID],
            value=expected_readonly,
            commit=args.commit,
        )
    if mode == "ensure" and intake_raw is None:
        _create_static_secret(
            client,
            name=args.miner_intake_secret_id,
            description=STATIC_DESCRIPTIONS[DEFAULT_MINER_INTAKE_SECRET_ID],
            value=expected_intake,
            commit=args.commit,
        )
    if mode == "ensure":
        readonly_readback = _static_secret_value(
            client, args.readonly_dsn_secret_id
        )
        intake_readback = _static_secret_value(
            client, args.miner_intake_secret_id
        )
        if (
            not isinstance(readonly_readback, str)
            or not isinstance(intake_readback, str)
            or not hmac.compare_digest(readonly_readback, expected_readonly)
            or not hmac.compare_digest(intake_readback, expected_intake)
        ):
            raise StaticSecretInstallError("static parity secret readback differs")
    receipt = {
        "schema_version": "leadpoet.production_parity_static_bootstrap.v1",
        "status": "installed" if mode == "ensure" else "probed",
        "commit": args.commit,
        "migration": "scripts/156-production-parity-readonly-role.sql",
        "migration_sha256": args.migration_sha256,
        "account_id": EXPECTED_ACCOUNT_ID,
        "instance_role": instance_arn,
        "bootstrap_role": bootstrap_arn,
        "readonly_secret_id": args.readonly_dsn_secret_id,
        "miner_intake_secret_id": args.miner_intake_secret_id,
        "reader_role": "leadpoet_parity_reader",
        "readonly_secret_present": readonly_raw is not None or mode == "ensure",
        "miner_intake_secret_present": intake_raw is not None or mode == "ensure",
        "secret_values_printed": False,
    }
    secret_response = {
        "readonly_dsn_available": bool(effective_dsn),
        "readonly_dsn": effective_dsn,
    }
    return receipt, secret_response


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--migration-sha256", required=True)
    parser.add_argument(
        "--validator-container", default="leadpoet-validator-main"
    )
    parser.add_argument(
        "--readonly-dsn-secret-id", default=DEFAULT_READONLY_SECRET_ID
    )
    parser.add_argument(
        "--miner-intake-secret-id", default=DEFAULT_MINER_INTAKE_SECRET_ID
    )
    parser.add_argument("--request-fd", type=int, required=True)
    parser.add_argument("--receipt-fd", type=int, required=True)
    parser.add_argument("--secret-response-fd", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if (
            args.request_fd < 3
            or args.receipt_fd < 3
            or args.secret_response_fd < 3
            or len(
                {args.request_fd, args.receipt_fd, args.secret_response_fd}
            )
            != 3
        ):
            raise StaticSecretInstallError("bootstrap descriptors must be distinct")
        _require_anonymous_fd(args.request_fd, description="request")
        _require_anonymous_fd(args.receipt_fd, description="receipt")
        _require_anonymous_fd(
            args.secret_response_fd, description="secret response"
        )
        request = _read_request(args.request_fd)
        result, secret_response = install(args, request)
        _write_all(
            args.receipt_fd,
            (
                json.dumps(result, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8"),
        )
        _write_all(
            args.secret_response_fd,
            json.dumps(
                secret_response, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        )
        return 0
    except (
        BotoCoreError,
        ClientError,
        OSError,
        StaticSecretInstallError,
        ValueError,
    ) as exc:
        print(f"STATIC_PARITY_BOOTSTRAP_ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
