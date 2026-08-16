#!/usr/bin/env python3
"""Idempotently install the small production-parity control plane."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen

import boto3
from botocore.exceptions import BotoCoreError, ClientError


REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
REGION_RE = re.compile(r"^[a-z]{2}-[a-z]+-[0-9]$")
IP_RE = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
SECRET_RE = re.compile(r"^[A-Za-z0-9/_+=.@-]{6,512}$")
PINNED_IMAGE_RE = re.compile(r"^[A-Za-z0-9._/-]+@sha256:[0-9a-f]{64}$")
CONTROLLER_ROLE = "leadpoet-production-parity-controller"
RUNNER_ROLE = "leadpoet-production-parity-runner"
RUNNER_PROFILE = "leadpoet-production-parity-runner"
OIDC_URL = "https://token.actions.githubusercontent.com"
DEFAULT_REPOSITORY = "leadpoet/leadpoet"


class SetupError(RuntimeError):
    pass


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _secret_dsn(value: str) -> str:
    try:
        document = json.loads(value)
    except ValueError:
        document = value
    if isinstance(document, Mapping):
        document = (
            document.get("readonly_dsn")
            or document.get("dsn")
            or document.get("url")
            or ""
        )
    dsn = str(document or "").strip()
    parsed = urlparse(dsn)
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or not parsed.hostname
        or not parsed.username
        or not parsed.password
        or not parsed.path.strip("/")
        or parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    ):
        raise SetupError("the read-only PostgreSQL DSN is invalid")
    return dsn


def _verify_readonly_dsn(dsn: str) -> None:
    query = """
SELECT json_build_object(
  'read_only', current_setting('transaction_read_only') = 'on',
  'superuser', rolsuper,
  'replication', rolreplication,
  'table_write_capable', EXISTS (
    SELECT 1
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public'
      AND c.relkind IN ('r', 'p')
      AND has_table_privilege(
        current_user, c.oid, 'INSERT,UPDATE,DELETE,TRUNCATE,TRIGGER'
      )
  ),
  'public_relation_count', (
    SELECT COUNT(*)
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relkind IN ('r', 'm', 'p')
  )
)
FROM pg_roles
WHERE rolname = current_user;
"""
    env = os.environ.copy()
    env["PGOPTIONS"] = "-c default_transaction_read_only=on -c statement_timeout=60000"
    parsed = urlparse(dsn)
    query_options = parse_qs(parsed.query, keep_blank_values=True)
    supported_query = {"sslmode", "connect_timeout"}
    unsupported_query = sorted(set(query_options) - supported_query)
    if unsupported_query:
        raise SetupError(
            "the read-only PostgreSQL DSN contains unsupported options: "
            + ", ".join(unsupported_query)
        )
    env.update(
        {
            "PGHOST": str(parsed.hostname or ""),
            "PGPORT": str(parsed.port or 5432),
            "PGUSER": unquote(str(parsed.username or "")),
            "PGPASSWORD": unquote(str(parsed.password or "")),
            "PGDATABASE": unquote(parsed.path.lstrip("/")),
            "PGSSLMODE": str(
                query_options.get("sslmode", ["require"])[-1]
            ),
            "PGCONNECT_TIMEOUT": str(
                query_options.get("connect_timeout", ["15"])[-1]
            ),
        }
    )
    result = subprocess.run(
        ["psql", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-c", query],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    if result.returncode != 0:
        raise SetupError(
            "the read-only PostgreSQL connection check failed: "
            + (result.stderr or "").strip()[-300:]
        )
    try:
        value = json.loads(result.stdout.strip())
    except ValueError as exc:
        raise SetupError("the read-only PostgreSQL check returned invalid data") from exc
    if (
        value.get("read_only") is not True
        or value.get("superuser") is not False
        or value.get("replication") is not False
        or value.get("table_write_capable") is not False
        or int(value.get("public_relation_count") or 0) <= 0
    ):
        raise SetupError(
            "the PostgreSQL credential is not a dedicated read-only production role"
        )


def _docker_hub_image(repository: str, tag: str, display: str) -> str:
    scope = quote(f"repository:{repository}:pull", safe=":")
    with urlopen(
        f"https://auth.docker.io/token?service=registry.docker.io&scope={scope}",
        timeout=30,
    ) as response:
        token = json.load(response).get("token")
    if not isinstance(token, str) or not token:
        raise SetupError(f"could not resolve the {display} image token")
    request = Request(
        f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.oci.image.index.v1+json,"
                "application/vnd.docker.distribution.manifest.list.v2+json,"
                "application/vnd.oci.image.manifest.v1+json,"
                "application/vnd.docker.distribution.manifest.v2+json"
            ),
        },
    )
    with urlopen(request, timeout=30) as response:
        response.read(1)
        digest = str(response.headers.get("Docker-Content-Digest") or "")
    pinned = f"{display}@{digest}"
    if PINNED_IMAGE_RE.fullmatch(pinned) is None:
        raise SetupError(f"could not resolve an immutable {display} image")
    return pinned


def _ensure_oidc_provider(iam: Any, account_id: str) -> str:
    arn = f"arn:aws:iam::{account_id}:oidc-provider/token.actions.githubusercontent.com"
    existing = {
        str(item.get("Arn") or "")
        for item in iam.list_open_id_connect_providers().get(
            "OpenIDConnectProviderList", []
        )
    }
    if arn not in existing:
        iam.create_open_id_connect_provider(
            Url=OIDC_URL,
            ClientIDList=["sts.amazonaws.com"],
            Tags=[{"Key": "leadpoet:purpose", "Value": "production-parity"}],
        )
    return arn


def _ensure_role(iam: Any, *, name: str, trust: Mapping[str, Any]) -> str:
    try:
        role = iam.get_role(RoleName=name)["Role"]
        iam.update_assume_role_policy(
            RoleName=name, PolicyDocument=_json(trust)
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise
        role = iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=_json(trust),
            Description="Leadpoet disposable production-parity validation",
            MaxSessionDuration=43200,
            Tags=[{"Key": "leadpoet:purpose", "Value": "production-parity"}],
        )["Role"]
    return str(role["Arn"])


def _put_policy(iam: Any, *, role: str, name: str, document: Mapping[str, Any]) -> None:
    iam.put_role_policy(
        RoleName=role,
        PolicyName=name,
        PolicyDocument=_json(document),
    )


def _ensure_instance_profile(iam: Any) -> None:
    try:
        profile = iam.get_instance_profile(
            InstanceProfileName=RUNNER_PROFILE
        )["InstanceProfile"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise
        profile = iam.create_instance_profile(
            InstanceProfileName=RUNNER_PROFILE,
            Tags=[{"Key": "leadpoet:purpose", "Value": "production-parity"}],
        )["InstanceProfile"]
    roles = {str(item.get("RoleName") or "") for item in profile.get("Roles", [])}
    if RUNNER_ROLE not in roles:
        iam.add_role_to_instance_profile(
            InstanceProfileName=RUNNER_PROFILE,
            RoleName=RUNNER_ROLE,
        )


def _controller_policy(
    *,
    account_id: str,
    region: str,
    production_secret_id: str,
    readonly_secret_id: str,
    runner_arn: str,
) -> dict[str, Any]:
    run_secret = (
        f"arn:aws:secretsmanager:{region}:{account_id}:secret:"
        "leadpoet/staging/production-parity/*"
    )
    parity_bucket = f"arn:aws:s3:::leadpoet-parity-{account_id}-*"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ec2:Describe*",
                    "ec2:RunInstances",
                    "ec2:TerminateInstances",
                    "ec2:CreateSecurityGroup",
                    "ec2:DeleteSecurityGroup",
                    "ec2:AuthorizeSecurityGroupIngress",
                    "ec2:CreateTags",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "cloudfront:CreateDistribution",
                    "cloudfront:GetDistribution",
                    "cloudfront:GetDistributionConfig",
                    "cloudfront:ListDistributions",
                    "cloudfront:ListTagsForResource",
                    "cloudfront:ListCachePolicies",
                    "cloudfront:ListOriginRequestPolicies",
                    "cloudfront:TagResource",
                    "cloudfront:UpdateDistribution",
                    "cloudfront:DeleteDistribution",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ssm:SendCommand",
                    "ssm:GetCommandInvocation",
                    "ssm:DescribeInstanceInformation",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": ["iam:PassRole"],
                "Resource": runner_arn,
            },
            {
                "Effect": "Allow",
                "Action": ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"],
                "Resource": [
                    f"arn:aws:secretsmanager:{region}:{account_id}:secret:{production_secret_id}*",
                    f"arn:aws:secretsmanager:{region}:{account_id}:secret:{readonly_secret_id}*",
                    run_secret,
                ],
            },
            {
                "Effect": "Allow",
                "Action": "secretsmanager:CreateSecret",
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "aws:RequestTag/leadpoet:ephemeral": "true"
                    },
                    "StringLike": {
                        "aws:RequestTag/leadpoet:parity-run": "????????*",
                        "aws:RequestTag/leadpoet:candidate-sha": "????????????????????????????????????????",
                    },
                },
            },
            {
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:PutSecretValue",
                    "secretsmanager:TagResource",
                    "secretsmanager:DeleteSecret",
                ],
                "Resource": run_secret,
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:CreateBucket",
                    "s3:DeleteBucket",
                    "s3:GetBucketTagging",
                    "s3:GetBucketLocation",
                    "s3:PutBucketTagging",
                    "s3:GetBucketVersioning",
                    "s3:PutBucketVersioning",
                    "s3:GetObjectLockConfiguration",
                    "s3:PutObjectLockConfiguration",
                    "s3:GetEncryptionConfiguration",
                    "s3:PutEncryptionConfiguration",
                    "s3:GetBucketPublicAccessBlock",
                    "s3:PutBucketPublicAccessBlock",
                    "s3:ListBucket",
                    "s3:ListBucketVersions",
                ],
                "Resource": parity_bucket,
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:DeleteObjectVersion",
                ],
                "Resource": parity_bucket + "/*",
            },
            {
                "Effect": "Allow",
                "Action": ["s3:ListAllMyBuckets", "secretsmanager:ListSecrets"],
                "Resource": "*",
            },
        ],
    }


def _runner_policy(
    *,
    account_id: str,
    region: str,
    production_secret_id: str,
    readonly_secret_id: str,
) -> dict[str, Any]:
    run_secret = (
        f"arn:aws:secretsmanager:{region}:{account_id}:secret:"
        "leadpoet/staging/production-parity/*"
    )
    parity_bucket = f"arn:aws:s3:::leadpoet-parity-{account_id}-*"
    run_objects = parity_bucket + "/*"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["sts:GetCallerIdentity", "ec2:Describe*"],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:DescribeImages",
                    "ecr:DescribeRepositories",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": [
                    parity_bucket,
                    "arn:aws:s3:::leadpoet-private-model-artifacts-*",
                    "arn:aws:s3:::leadpoet-attested-v2-artifacts-*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": [
                    run_objects,
                    "arn:aws:s3:::leadpoet-private-model-artifacts-*/*",
                    "arn:aws:s3:::leadpoet-attested-v2-artifacts-*/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:PutObject",
                    "s3:PutObjectRetention",
                    "s3:DeleteObject",
                ],
                "Resource": run_objects,
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetBucketVersioning",
                    "s3:GetObjectLockConfiguration",
                    "s3:GetBucketLocation",
                ],
                "Resource": parity_bucket,
            },
            {
                "Effect": "Allow",
                "Action": ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"],
                "Resource": [
                    f"arn:aws:secretsmanager:{region}:{account_id}:secret:{production_secret_id}*",
                    f"arn:aws:secretsmanager:{region}:{account_id}:secret:{readonly_secret_id}*",
                    run_secret,
                ],
            },
            {
                "Effect": "Allow",
                "Action": "secretsmanager:CreateSecret",
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "aws:RequestTag/leadpoet:ephemeral": "true"
                    },
                    "StringLike": {
                        "aws:RequestTag/leadpoet:parity-run": "????????*",
                        "aws:RequestTag/leadpoet:candidate-sha": "????????????????????????????????????????",
                    },
                },
            },
            {
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:PutSecretValue",
                    "secretsmanager:TagResource",
                    "secretsmanager:DeleteSecret",
                ],
                "Resource": run_secret,
            },
            {
                "Effect": "Allow",
                "Action": [
                    "kms:DescribeKey",
                    "kms:GetPublicKey",
                    "kms:Encrypt",
                    "kms:Decrypt",
                    "kms:GenerateDataKey",
                    "kms:Sign",
                    "kms:Verify",
                ],
                "Resource": f"arn:aws:kms:{region}:{account_id}:key/*",
            },
            {
                "Effect": "Deny",
                "Action": [
                    "ec2:RunInstances",
                    "ec2:TerminateInstances",
                    "iam:*",
                    "cloudfront:Create*",
                    "cloudfront:Update*",
                    "cloudfront:Delete*",
                    "ssm:SendCommand",
                    "kms:Create*",
                    "kms:ScheduleKeyDeletion",
                    "kms:DisableKey",
                    "kms:PutKeyPolicy",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Deny",
                "Action": [
                    "s3:PutObject",
                    "s3:PutObjectRetention",
                    "s3:DeleteObject",
                ],
                "NotResource": run_objects,
            },
            {
                "Effect": "Deny",
                "Action": [
                    "secretsmanager:PutSecretValue",
                    "secretsmanager:UpdateSecret",
                    "secretsmanager:DeleteSecret",
                ],
                "NotResource": run_secret,
            },
        ],
    }


def _gh_variable(repository: str, name: str, value: str) -> None:
    result = subprocess.run(
        ["gh", "variable", "set", name, "--repo", repository, "--body", value],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise SetupError(f"GitHub variable {name} could not be set")


def _validate_inputs(args: argparse.Namespace) -> None:
    if (
        REPOSITORY_RE.fullmatch(args.repository) is None
        or REGION_RE.fullmatch(args.region) is None
        or IP_RE.fullmatch(args.production_gateway_ip) is None
        or SECRET_RE.fullmatch(args.production_gateway_secret_id) is None
        or SECRET_RE.fullmatch(args.readonly_dsn_secret_id) is None
    ):
        raise SetupError("setup inputs are invalid")
    parsed = urlparse(args.production_gateway_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise SetupError("production gateway URL must be HTTPS")
    with urlopen(
        args.production_gateway_url.rstrip("/") + "/build-info", timeout=20
    ) as response:
        value = json.load(response)
    commit = str(value.get("git_commit") or value.get("commit_sha") or "")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SetupError("production gateway build identity is unavailable")


def setup(args: argparse.Namespace) -> dict[str, Any]:
    _validate_inputs(args)
    session = boto3.session.Session(region_name=args.region)
    sts = session.client("sts")
    iam = session.client("iam")
    secrets = session.client("secretsmanager")
    account_id = str(sts.get_caller_identity()["Account"])

    try:
        current_secret = secrets.get_secret_value(
            SecretId=args.readonly_dsn_secret_id
        ).get("SecretString")
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise
        current_secret = None
    if current_secret is None or args.replace_readonly_dsn:
        entered = getpass.getpass(
            "Paste the dedicated READ-ONLY production PostgreSQL DSN, then press Return: "
        )
        dsn = _secret_dsn(entered)
    else:
        dsn = _secret_dsn(str(current_secret))
    _verify_readonly_dsn(dsn)

    if current_secret is None:
        secrets.create_secret(
            Name=args.readonly_dsn_secret_id,
            Description="Leadpoet production-parity read-only source DSN",
            SecretString=_json({"readonly_dsn": dsn}),
            Tags=[{"Key": "leadpoet:purpose", "Value": "production-parity"}],
        )
    elif args.replace_readonly_dsn:
        secrets.put_secret_value(
            SecretId=args.readonly_dsn_secret_id,
            SecretString=_json({"readonly_dsn": dsn}),
        )

    oidc_arn = _ensure_oidc_provider(iam, account_id)
    runner_trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }
    runner_arn = _ensure_role(iam, name=RUNNER_ROLE, trust=runner_trust)
    _put_policy(
        iam,
        role=RUNNER_ROLE,
        name="LeadpoetProductionParityRunner",
        document=_runner_policy(
            account_id=account_id,
            region=args.region,
            production_secret_id=args.production_gateway_secret_id,
            readonly_secret_id=args.readonly_dsn_secret_id,
        ),
    )
    iam.attach_role_policy(
        RoleName=RUNNER_ROLE,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
    )
    _ensure_instance_profile(iam)

    owner, repository_name = args.repository.split("/", 1)
    controller_trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Federated": oidc_arn},
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                },
                "StringLike": {
                    "token.actions.githubusercontent.com:sub": (
                        f"repo:{owner}/{repository_name}:ref:refs/heads/main"
                    )
                },
            },
        }],
    }
    controller_arn = _ensure_role(
        iam, name=CONTROLLER_ROLE, trust=controller_trust
    )
    _put_policy(
        iam,
        role=CONTROLLER_ROLE,
        name="LeadpoetProductionParityController",
        document=_controller_policy(
            account_id=account_id,
            region=args.region,
            production_secret_id=args.production_gateway_secret_id,
            readonly_secret_id=args.readonly_dsn_secret_id,
            runner_arn=runner_arn,
        ),
    )

    postgres = args.postgres_image or _docker_hub_image(
        "library/postgres", "15", "postgres"
    )
    postgrest = args.postgrest_image or _docker_hub_image(
        "postgrest/postgrest", "v12.2.3", "postgrest/postgrest"
    )
    if (
        PINNED_IMAGE_RE.fullmatch(postgres) is None
        or PINNED_IMAGE_RE.fullmatch(postgrest) is None
    ):
        raise SetupError("container images must be immutable digest references")

    variables = {
        "LEADPOET_PARITY_ENABLED": "true" if args.enable else "false",
        "LEADPOET_PARITY_AWS_ROLE_ARN": controller_arn,
        "LEADPOET_PARITY_AWS_REGION": args.region,
        "LEADPOET_PARITY_PRODUCTION_GATEWAY_IP": args.production_gateway_ip,
        "LEADPOET_PARITY_PRODUCTION_GATEWAY_URL": args.production_gateway_url.rstrip("/"),
        "LEADPOET_PARITY_PRODUCTION_GATEWAY_SECRET_ID": args.production_gateway_secret_id,
        "LEADPOET_PARITY_READONLY_DSN_SECRET_ID": args.readonly_dsn_secret_id,
        "LEADPOET_PARITY_RUNNER_INSTANCE_PROFILE": RUNNER_PROFILE,
        "LEADPOET_PARITY_POSTGRES_IMAGE": postgres,
        "LEADPOET_PARITY_POSTGREST_IMAGE": postgrest,
        "LEADPOET_PARITY_VOLUME_GIB": str(args.volume_gib),
    }
    for name, value in variables.items():
        _gh_variable(args.repository, name, value)
    return {
        "status": "configured",
        "repository": args.repository,
        "account_id": account_id,
        "controller_role_arn": controller_arn,
        "runner_role_arn": runner_arn,
        "runner_instance_profile": RUNNER_PROFILE,
        "readonly_secret_id": args.readonly_dsn_secret_id,
        "enabled": args.enable,
        "secret_values_printed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("apply",))
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--production-gateway-ip", default="52.91.135.79")
    parser.add_argument("--production-gateway-url", required=True)
    parser.add_argument(
        "--production-gateway-secret-id", default="leadpoet/prod/gateway/env"
    )
    parser.add_argument(
        "--readonly-dsn-secret-id",
        default="leadpoet/staging/production-parity/readonly-dsn",
    )
    parser.add_argument("--postgres-image")
    parser.add_argument("--postgrest-image")
    parser.add_argument("--volume-gib", type=int, default=200)
    parser.add_argument("--replace-readonly-dsn", action="store_true")
    parser.add_argument("--enable", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = setup(args)
    except (
        BotoCoreError,
        ClientError,
        OSError,
        SetupError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
