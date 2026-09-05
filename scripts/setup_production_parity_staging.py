#!/usr/bin/env python3
"""Idempotently install the small production-parity control plane."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
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
EXPECTED_REPOSITORY_ID = "1075412927"
EXPECTED_ACCOUNT_ID = "493765492819"
EXPECTED_REGION = "us-east-1"
PRODUCTION_GATEWAY_IP = "52.91.135.79"
PRODUCTION_GATEWAY_URL = "https://gateway.subnet71.com"
PRODUCTION_GATEWAY_SECRET_ID = "leadpoet/prod/gateway/env"
PRODUCTION_AMI_ID = "ami-0cae6d6fe6048ca2c"
PRODUCTION_INSTANCE_TYPE = "r7i.4xlarge"
PRODUCTION_SUBNET_ID = "subnet-025170c1eff61494d"
PRODUCTION_VPC_ID = "vpc-0c975a643bc1e0e79"
READONLY_DSN_SECRET_ID = "leadpoet/staging/production-parity/readonly-dsn"
PRODUCTION_POOLER_HOST = "aws-0-us-east-1.pooler.supabase.com"
PRODUCTION_READER_USER = "leadpoet_parity_reader.qplwoislplkcegvdmbim"
LOCAL_PSQL = Path("/opt/homebrew/opt/libpq/bin/psql")
VALIDATOR_ROLE = "leadpoet-validator-s3-cloudwatch-role"
STATIC_BOOTSTRAP_ROLE = "leadpoet-production-parity-static-bootstrap"
STATIC_BOOTSTRAP_POLICY = "LeadpoetProductionParityStaticBootstrap"
MINIMUM_VOLUME_GIB = 512
DEFAULT_VOLUME_GIB = 512
GATEWAY_IAM_CACHE = Path("/home/ec2-user/.config/leadpoet/gateway.env")
DEFAULT_MINER_INTAKE_SECRET_ID = (
    "leadpoet/staging/production-parity-miner-intake"
)
CONTROLLER_WORKFLOWS = (
    "Production Parity Full",
    "Production Parity Fast",
    "Production Parity Cleanup",
)
CONTROLLER_POLICY_PATH = "/leadpoet/production-parity/"
CONTROLLER_REVOKE_POLICY = "LeadpoetProductionParityRevokeOlderSessions"
RUNNER_REVOKE_POLICY = "LeadpoetProductionParityRunnerRevokeOlderSessions"
CONTROLLER_POLICY_DESCRIPTION = (
    "Leadpoet production-parity controller boundary"
)
CONTROLLER_POLICY_NAMES = {
    "ec2_launch": "LeadpoetParityControllerEc2Launch",
    "lifecycle": "LeadpoetParityControllerLifecycle",
    "cloudfront": "LeadpoetParityControllerCloudFront",
    "data": "LeadpoetParityControllerData",
}


class SetupError(RuntimeError):
    pass


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _require_anonymous_fd(descriptor: int, *, description: str) -> None:
    if descriptor < 3:
        raise SetupError(f"{description} descriptor must be an inherited pipe or socket")
    mode = os.fstat(descriptor).st_mode
    if not (stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)) or os.isatty(descriptor):
        raise SetupError(f"{description} descriptor must be an inherited pipe or socket")


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
  'role_name', current_user,
  'read_only', current_setting('transaction_read_only') = 'on',
  'superuser', rolsuper,
  'bypass_rls', rolbypassrls,
  'createdb', rolcreatedb,
  'createrole', rolcreaterole,
  'inherit', rolinherit,
  'replication', rolreplication,
  'connection_limit', rolconnlimit,
  'table_write_capable', EXISTS (
    SELECT 1
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public'
      AND c.relkind IN ('r', 'p')
      AND has_table_privilege(
        current_user, c.oid,
        'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'
      )
  ),
  'sequence_write_capable', EXISTS (
    SELECT 1
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public'
      AND CASE WHEN c.relkind = 'S' THEN
        has_sequence_privilege(current_user, c.oid, 'USAGE,UPDATE')
      ELSE false END
  ),
  'schema_create_capable', has_schema_privilege(current_user, 'public', 'CREATE'),
  'membership_count', (
    SELECT COUNT(*)
    FROM pg_auth_members member
    JOIN pg_roles recipient ON recipient.oid = member.member
    WHERE recipient.rolname = current_user
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
    parsed = urlparse(dsn)
    query_options = parse_qs(parsed.query, keep_blank_values=True)
    supported_query = {"sslmode", "connect_timeout"}
    unsupported_query = sorted(set(query_options) - supported_query)
    if unsupported_query:
        raise SetupError(
            "the read-only PostgreSQL DSN contains unsupported options: "
            + ", ".join(unsupported_query)
        )
    if (
        str(parsed.hostname or "").lower() != PRODUCTION_POOLER_HOST
        or (parsed.port or 5432) != 5432
        or unquote(str(parsed.username or "")) != PRODUCTION_READER_USER
        or re.fullmatch(r"[0-9a-f]{64}", unquote(str(parsed.password or "")))
        is None
        or unquote(parsed.path.lstrip("/")) != "postgres"
        or query_options.get("sslmode") != ["require"]
        or parsed.fragment
    ):
        raise SetupError("the read-only PostgreSQL DSN identity differs")
    # Build a small child environment so inherited libpq service/password/URI
    # routes (including HOME/.pgpass) cannot override this fixed connection.
    env = {
        name: os.environ[name]
        for name in ("PATH", "LANG", "LC_ALL")
        if os.environ.get(name)
    }
    env.update({"PGSSLMODE": "require", "PGCONNECT_TIMEOUT": "15"})
    password = unquote(str(parsed.password or ""))
    try:
        psql_path = LOCAL_PSQL.resolve(strict=True)
        psql_stat = psql_path.stat()
    except OSError as exc:
        raise SetupError("the fixed local psql executable is unavailable") from exc
    if (
        not stat.S_ISREG(psql_stat.st_mode)
        or psql_stat.st_uid not in {0, os.geteuid()}
        or not os.access(psql_path, os.X_OK)
        or not str(psql_path).startswith("/opt/homebrew/Cellar/libpq/")
    ):
        raise SetupError("the fixed local psql executable identity differs")
    result = subprocess.run(
        [
            str(psql_path),
            "-X",
            "-A",
            "-t",
            "-W",
            "-v",
            "ON_ERROR_STOP=1",
            "-h",
            PRODUCTION_POOLER_HOST,
            "-p",
            "5432",
            "-U",
            PRODUCTION_READER_USER,
            "-d",
            "postgres",
            "-c",
            query,
        ],
        env=env,
        input=password + "\n",
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
        start_new_session=True,
    )
    password = ""
    if result.returncode != 0:
        raise SetupError("the read-only PostgreSQL connection check failed")
    try:
        rows = [line for line in result.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError("unexpected row count")
        value = json.loads(rows[0])
    except ValueError as exc:
        raise SetupError("the read-only PostgreSQL check returned invalid data") from exc
    if (
        value.get("role_name") != "leadpoet_parity_reader"
        or value.get("read_only") is not True
        or value.get("superuser") is not False
        or value.get("bypass_rls") is not True
        or value.get("createdb") is not False
        or value.get("createrole") is not False
        or value.get("inherit") is not False
        or value.get("replication") is not False
        or not 0 < int(value.get("connection_limit") or 0) <= 2
        or value.get("table_write_capable") is not False
        or value.get("sequence_write_capable") is not False
        or value.get("schema_create_capable") is not False
        or int(value.get("membership_count") or 0) != 0
        or int(value.get("public_relation_count") or 0) <= 0
    ):
        raise SetupError(
            "the PostgreSQL credential is not the bounded read-only parity role"
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
    existing = [
        str(item.get("Arn") or "")
        for item in iam.list_open_id_connect_providers().get(
            "OpenIDConnectProviderList", []
        )
    ]
    if existing.count(arn) > 1:
        raise SetupError("GitHub OIDC provider inventory is ambiguous")
    if arn not in existing:
        iam.create_open_id_connect_provider(
            Url=OIDC_URL,
            ClientIDList=["sts.amazonaws.com"],
        )
    provider = iam.get_open_id_connect_provider(
        OpenIDConnectProviderArn=arn
    )
    tags = {
        str(item.get("Key") or ""): str(item.get("Value") or "")
        for item in provider.get("Tags", [])
    }
    thumbprints = provider.get("ThumbprintList", [])
    if (
        str(provider.get("Url") or "")
        != "token.actions.githubusercontent.com"
        or provider.get("ClientIDList") != ["sts.amazonaws.com"]
        or not isinstance(thumbprints, list)
        or len(thumbprints) != 1
        or re.fullmatch(r"[0-9a-f]{40}", str(thumbprints[0] or "")) is None
        or tags
    ):
        raise SetupError("GitHub OIDC provider configuration differs")
    readback = [
        str(item.get("Arn") or "")
        for item in iam.list_open_id_connect_providers().get(
            "OpenIDConnectProviderList", []
        )
    ]
    if readback.count(arn) != 1:
        raise SetupError("GitHub OIDC provider ARN readback differs")
    return arn


def _ensure_role(
    iam: Any,
    *,
    account_id: str,
    name: str,
    trust: Mapping[str, Any],
    expected_inline_policies: set[str],
    expected_attached_policies: set[str],
    max_session_duration: int = 43200,
    revoke_policy_name: str | None = None,
    revocation_receipt: dict[str, Any] | None = None,
) -> str:
    try:
        role = iam.get_role(RoleName=name)["Role"]
        tags = {
            str(item.get("Key") or ""): str(item.get("Value") or "")
            for item in iam.list_role_tags(RoleName=name).get("Tags", [])
        }
        if (
            str(role.get("Arn") or "")
            != f"arn:aws:iam::{account_id}:role/{name}"
            or str(role.get("Path") or "/") != "/"
            or tags != {"leadpoet:purpose": "production-parity"}
            or role.get("PermissionsBoundary")
        ):
            raise SetupError(f"existing IAM role {name} is not owned by parity")
        inline = set(
            iam.list_role_policies(RoleName=name).get("PolicyNames", [])
        )
        attached = {
            str(item.get("PolicyArn") or "")
            for item in iam.list_attached_role_policies(RoleName=name).get(
                "AttachedPolicies", []
            )
        }
        if (
            not inline.issubset(expected_inline_policies)
            or not attached.issubset(expected_attached_policies)
        ):
            raise SetupError(f"existing IAM role {name} policy inventory differs")
        iam.update_assume_role_policy(
            RoleName=name, PolicyDocument=_json(trust)
        )
        iam.update_role(
            RoleName=name,
            MaxSessionDuration=max_session_duration,
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise
        role = iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=_json(trust),
            Description="Leadpoet disposable production-parity validation",
            MaxSessionDuration=max_session_duration,
            Tags=[{"Key": "leadpoet:purpose", "Value": "production-parity"}],
        )["Role"]
        inline = set()
        attached = set()
    # Trust is an authorization boundary, so converge and prove the inert
    # trust before any caller replaces existing inline policy contents.
    role = iam.get_role(RoleName=name)["Role"]
    tags = {
        str(item.get("Key") or ""): str(item.get("Value") or "")
        for item in iam.list_role_tags(RoleName=name).get("Tags", [])
    }
    if (
        str(role.get("Arn") or "")
        != f"arn:aws:iam::{account_id}:role/{name}"
        or str(role.get("Path") or "/") != "/"
        or tags != {"leadpoet:purpose": "production-parity"}
        or role.get("PermissionsBoundary")
        or int(role.get("MaxSessionDuration") or 0) != max_session_duration
        or _json(_policy_document(role.get("AssumeRolePolicyDocument")))
        != _json(trust)
    ):
        raise SetupError(f"IAM role {name} trust readback differs")
    if revoke_policy_name is not None:
        if _json(trust) != _json(_inert_trust()) or revocation_receipt is None:
            raise SetupError("IAM role session revocation contract is invalid")
        cutoff = _new_session_cutoff()
        revoke_document = _revoke_older_sessions_policy(cutoff=cutoff)
        _put_policy(
            iam,
            role=name,
            name=revoke_policy_name,
            document=revoke_document,
        )
        actual_revoke = iam.get_role_policy(
            RoleName=name,
            PolicyName=revoke_policy_name,
        ).get("PolicyDocument")
        if _json(_policy_document(actual_revoke)) != _json(revoke_document):
            raise SetupError(f"IAM role {name} session revocation readback differs")
        revocation_receipt.update({
            "cutoff": cutoff,
            "document": revoke_document,
        })
    inline = set(
        iam.list_role_policies(RoleName=name).get("PolicyNames", [])
    )
    attached = {
        str(item.get("PolicyArn") or "")
        for item in iam.list_attached_role_policies(RoleName=name).get(
            "AttachedPolicies", []
        )
    }
    if (
        not inline.issubset(expected_inline_policies)
        or not attached.issubset(expected_attached_policies)
    ):
        raise SetupError(f"existing IAM role {name} policy inventory differs")
    return str(role["Arn"])


def _policy_document(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        decoded = json.loads(unquote(value))
        if isinstance(decoded, Mapping):
            return decoded
    raise SetupError("IAM policy document is invalid")


def _assert_role_configuration(
    iam: Any,
    *,
    account_id: str,
    name: str,
    trust: Mapping[str, Any],
    inline_policies: Mapping[str, Mapping[str, Any]],
    attached_policies: set[str],
    max_session_duration: int,
) -> None:
    role = iam.get_role(RoleName=name)["Role"]
    tags = {
        str(item.get("Key") or ""): str(item.get("Value") or "")
        for item in iam.list_role_tags(RoleName=name).get("Tags", [])
    }
    inline = set(iam.list_role_policies(RoleName=name).get("PolicyNames", []))
    attached = {
        str(item.get("PolicyArn") or "")
        for item in iam.list_attached_role_policies(RoleName=name).get(
            "AttachedPolicies", []
        )
    }
    if (
        str(role.get("Arn") or "") != f"arn:aws:iam::{account_id}:role/{name}"
        or str(role.get("Path") or "/") != "/"
        or tags != {"leadpoet:purpose": "production-parity"}
        or role.get("PermissionsBoundary")
        or int(role.get("MaxSessionDuration") or 0) != max_session_duration
        or _json(_policy_document(role.get("AssumeRolePolicyDocument")))
        != _json(trust)
        or inline != set(inline_policies)
        or attached != attached_policies
    ):
        raise SetupError(f"IAM role {name} readback differs")
    for policy_name, expected in inline_policies.items():
        actual = iam.get_role_policy(
            RoleName=name, PolicyName=policy_name
        ).get("PolicyDocument")
        if _json(_policy_document(actual)) != _json(expected):
            raise SetupError(f"IAM role {name} inline policy readback differs")


def _put_policy(iam: Any, *, role: str, name: str, document: Mapping[str, Any]) -> None:
    iam.put_role_policy(
        RoleName=role,
        PolicyName=name,
        PolicyDocument=_json(document),
    )


def _neutralize_controller_role(
    iam: Any,
    *,
    account_id: str,
    managed_policy_arns: set[str],
    legacy_inline_policy: str,
    revoke_document: Mapping[str, Any],
) -> None:
    inert_trust = _inert_trust()
    iam.update_assume_role_policy(
        RoleName=CONTROLLER_ROLE,
        PolicyDocument=_json(inert_trust),
    )
    iam.update_role(RoleName=CONTROLLER_ROLE, MaxSessionDuration=43200)
    _put_policy(
        iam,
        role=CONTROLLER_ROLE,
        name=CONTROLLER_REVOKE_POLICY,
        document=revoke_document,
    )
    revoke_readback = iam.get_role_policy(
        RoleName=CONTROLLER_ROLE,
        PolicyName=CONTROLLER_REVOKE_POLICY,
    ).get("PolicyDocument")
    if _json(_policy_document(revoke_readback)) != _json(revoke_document):
        raise SetupError("controller IAM session revocation readback differs")
    inline = set(
        iam.list_role_policies(RoleName=CONTROLLER_ROLE).get("PolicyNames", [])
    )
    attached = {
        str(item.get("PolicyArn") or "")
        for item in iam.list_attached_role_policies(
            RoleName=CONTROLLER_ROLE
        ).get("AttachedPolicies", [])
    }
    if not inline.issubset(
        {legacy_inline_policy, CONTROLLER_REVOKE_POLICY}
    ) or not attached.issubset(managed_policy_arns):
        raise SetupError("controller IAM policy inventory changed unexpectedly")
    if legacy_inline_policy in inline:
        iam.delete_role_policy(
            RoleName=CONTROLLER_ROLE,
            PolicyName=legacy_inline_policy,
        )
    for policy_arn in sorted(attached):
        iam.detach_role_policy(
            RoleName=CONTROLLER_ROLE,
            PolicyArn=policy_arn,
        )
    _assert_role_configuration(
        iam,
        account_id=account_id,
        name=CONTROLLER_ROLE,
        trust=inert_trust,
        inline_policies={CONTROLLER_REVOKE_POLICY: revoke_document},
        attached_policies=set(),
        max_session_duration=43200,
    )


def _managed_policy_arn(*, account_id: str, name: str) -> str:
    return (
        f"arn:aws:iam::{account_id}:policy/"
        f"{CONTROLLER_POLICY_PATH.strip('/')}/{name}"
    )


def _managed_policy_entities(
    iam: Any, *, policy_arn: str
) -> tuple[set[str], set[str], set[str]]:
    roles: set[str] = set()
    users: set[str] = set()
    groups: set[str] = set()
    marker: str | None = None
    while True:
        response = iam.list_entities_for_policy(
            PolicyArn=policy_arn,
            **({"Marker": marker} if marker else {}),
        )
        roles.update(
            str(item.get("RoleName") or "")
            for item in response.get("PolicyRoles", [])
        )
        users.update(
            str(item.get("UserName") or "")
            for item in response.get("PolicyUsers", [])
        )
        groups.update(
            str(item.get("GroupName") or "")
            for item in response.get("PolicyGroups", [])
        )
        if not response.get("IsTruncated"):
            break
        marker = str(response.get("Marker") or "")
        if not marker:
            raise SetupError("IAM managed policy entity pagination differs")
    return roles, users, groups


def _assert_managed_policy(
    iam: Any,
    *,
    account_id: str,
    name: str,
    document: Mapping[str, Any],
    expected_roles: set[str],
) -> str:
    arn = _managed_policy_arn(account_id=account_id, name=name)
    policy = iam.get_policy(PolicyArn=arn).get("Policy", {})
    tags = {
        str(item.get("Key") or ""): str(item.get("Value") or "")
        for item in iam.list_policy_tags(PolicyArn=arn).get("Tags", [])
    }
    versions = iam.list_policy_versions(PolicyArn=arn).get("Versions", [])
    roles, users, groups = _managed_policy_entities(iam, policy_arn=arn)
    if (
        policy.get("Arn") != arn
        or policy.get("PolicyName") != name
        or policy.get("Path") != CONTROLLER_POLICY_PATH
        or policy.get("Description") != CONTROLLER_POLICY_DESCRIPTION
        or tags != {"leadpoet:purpose": "production-parity"}
        or roles != expected_roles
        or users
        or groups
        or len(versions) != 1
        or versions[0].get("IsDefaultVersion") is not True
        or versions[0].get("VersionId") != policy.get("DefaultVersionId")
    ):
        raise SetupError(f"IAM managed policy {name} identity differs")
    actual = iam.get_policy_version(
        PolicyArn=arn,
        VersionId=str(policy["DefaultVersionId"]),
    ).get("PolicyVersion", {}).get("Document")
    if _json(_policy_document(actual)) != _json(document):
        raise SetupError(f"IAM managed policy {name} document differs")
    return arn


def _ensure_managed_policy(
    iam: Any,
    *,
    account_id: str,
    name: str,
    document: Mapping[str, Any],
) -> str:
    if len(_json(document)) > 6144:
        raise SetupError(f"IAM managed policy {name} exceeds the AWS quota")
    arn = _managed_policy_arn(account_id=account_id, name=name)
    try:
        policy = iam.get_policy(PolicyArn=arn).get("Policy", {})
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise
        iam.create_policy(
            PolicyName=name,
            Path=CONTROLLER_POLICY_PATH,
            PolicyDocument=_json(document),
            Description=CONTROLLER_POLICY_DESCRIPTION,
            Tags=[{"Key": "leadpoet:purpose", "Value": "production-parity"}],
        )
        return _assert_managed_policy(
            iam,
            account_id=account_id,
            name=name,
            document=document,
            expected_roles=set(),
        )

    tags = {
        str(item.get("Key") or ""): str(item.get("Value") or "")
        for item in iam.list_policy_tags(PolicyArn=arn).get("Tags", [])
    }
    roles, users, groups = _managed_policy_entities(iam, policy_arn=arn)
    if (
        policy.get("Arn") != arn
        or policy.get("PolicyName") != name
        or policy.get("Path") != CONTROLLER_POLICY_PATH
        or policy.get("Description") != CONTROLLER_POLICY_DESCRIPTION
        or tags != {"leadpoet:purpose": "production-parity"}
        or roles
        or users
        or groups
    ):
        raise SetupError(f"existing IAM managed policy {name} is not owned by parity")

    default_version = str(policy.get("DefaultVersionId") or "")
    current = iam.get_policy_version(
        PolicyArn=arn,
        VersionId=default_version,
    ).get("PolicyVersion", {}).get("Document")
    versions = iam.list_policy_versions(PolicyArn=arn).get("Versions", [])
    version_ids = [str(item.get("VersionId") or "") for item in versions]
    default_ids = [
        str(item.get("VersionId") or "")
        for item in versions
        if item.get("IsDefaultVersion") is True
    ]
    if (
        not default_version
        or not 1 <= len(versions) <= 5
        or any(not value for value in version_ids)
        or len(set(version_ids)) != len(version_ids)
        or default_ids != [default_version]
    ):
        raise SetupError(f"IAM managed policy {name} version inventory differs")
    for version in versions:
        version_id = str(version.get("VersionId") or "")
        if not version_id or version_id == default_version:
            continue
        iam.delete_policy_version(PolicyArn=arn, VersionId=version_id)
    if _json(_policy_document(current)) != _json(document):
        created = iam.create_policy_version(
            PolicyArn=arn,
            PolicyDocument=_json(document),
            SetAsDefault=True,
        ).get("PolicyVersion", {})
        replacement = str(created.get("VersionId") or "")
        if not replacement or replacement == default_version:
            raise SetupError(f"IAM managed policy {name} version creation differs")
        iam.delete_policy_version(
            PolicyArn=arn,
            VersionId=default_version,
        )
    return _assert_managed_policy(
        iam,
        account_id=account_id,
        name=name,
        document=document,
        expected_roles=set(),
    )


def _inert_trust() -> dict[str, Any]:
    # IAM role trust policies reject a bare wildcard principal.  Explicitly
    # deny the production account; with no Allow statement, every other
    # principal remains implicitly denied as well.
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Deny",
            "Principal": {
                "AWS": f"arn:aws:iam::{EXPECTED_ACCOUNT_ID}:root",
            },
            "Action": "sts:AssumeRole",
        }],
    }


def _controller_trust(*, oidc_arn: str) -> dict[str, Any]:
    claim = "token.actions.githubusercontent.com:"
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Federated": oidc_arn},
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    claim + "aud": "sts.amazonaws.com",
                    claim + "sub": (
                        "repo:leadpoet/leadpoet:ref:refs/heads/main"
                    ),
                    claim + "repository": DEFAULT_REPOSITORY,
                    claim + "repository_id": EXPECTED_REPOSITORY_ID,
                    claim + "ref": "refs/heads/main",
                    claim + "workflow": list(CONTROLLER_WORKFLOWS),
                },
            },
        }],
    }


def _revoke_older_sessions_policy(*, cutoff: str) -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Deny",
            "Action": "*",
            "Resource": "*",
            "Condition": {
                "DateLessThan": {"aws:TokenIssueTime": cutoff},
            },
        }],
    }


def _new_session_cutoff() -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def _wait_until_after_session_cutoffs(cutoffs: Sequence[str]) -> None:
    try:
        not_before = max(
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            for value in cutoffs
        )
    except (ValueError, TypeError) as exc:
        raise SetupError("IAM session revocation cutoff is invalid") from exc
    remaining = (not_before - datetime.now(timezone.utc)).total_seconds()
    if remaining > 35:
        raise SetupError("IAM session revocation cutoff is unexpectedly distant")
    if remaining >= 0:
        time.sleep(remaining + 0.1)
    if datetime.now(timezone.utc) <= not_before:
        raise SetupError("IAM session revocation cutoff has not elapsed")


def _ensure_instance_profile(iam: Any, *, account_id: str) -> None:
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
    tags = {
        str(item.get("Key") or ""): str(item.get("Value") or "")
        for item in iam.list_instance_profile_tags(
            InstanceProfileName=RUNNER_PROFILE
        ).get("Tags", [])
    }
    roles = {str(item.get("RoleName") or "") for item in profile.get("Roles", [])}
    if (
        str(profile.get("Arn") or "")
        != f"arn:aws:iam::{account_id}:instance-profile/{RUNNER_PROFILE}"
        or str(profile.get("Path") or "/") != "/"
        or tags != {"leadpoet:purpose": "production-parity"}
        or roles not in (set(), {RUNNER_ROLE})
    ):
        raise SetupError("runner instance profile identity differs")
    if RUNNER_ROLE not in roles:
        iam.add_role_to_instance_profile(
            InstanceProfileName=RUNNER_PROFILE,
            RoleName=RUNNER_ROLE,
        )
    profile = iam.get_instance_profile(
        InstanceProfileName=RUNNER_PROFILE
    )["InstanceProfile"]
    if {str(item.get("RoleName") or "") for item in profile.get("Roles", [])} != {
        RUNNER_ROLE
    }:
        raise SetupError("runner instance profile role readback differs")


def _controller_policy(
    *,
    account_id: str,
    region: str,
    production_secret_id: str,
    readonly_secret_id: str,
    miner_intake_secret_id: str,
    runner_arn: str,
) -> dict[str, Any]:
    run_secret = (
        f"arn:aws:secretsmanager:{region}:{account_id}:secret:"
        "leadpoet/staging/production-parity/runs/pp-*/gateway-??????"
    )
    legacy_run_secret = (
        f"arn:aws:secretsmanager:{region}:{account_id}:secret:"
        "leadpoet/staging/production-parity/pp-*/gateway-??????"
    )
    readonly_secret = (
        f"arn:aws:secretsmanager:{region}:{account_id}:secret:"
        f"{readonly_secret_id}-??????"
    )
    miner_intake_secret = (
        f"arn:aws:secretsmanager:{region}:{account_id}:secret:"
        f"{miner_intake_secret_id}-??????"
    )
    parity_bucket = f"arn:aws:s3:::leadpoet-parity-{account_id}-*"
    ec2_prefix = f"arn:aws:ec2:{region}:{account_id}"
    instance_arn = ec2_prefix + ":instance/*"
    volume_arn = ec2_prefix + ":volume/*"
    network_interface_arn = ec2_prefix + ":network-interface/*"
    security_group_arn = ec2_prefix + ":security-group/*"
    subnet_arn = ec2_prefix + f":subnet/{PRODUCTION_SUBNET_ID}"
    vpc_arn = ec2_prefix + f":vpc/{PRODUCTION_VPC_ID}"
    image_arn = f"arn:aws:ec2:{region}::image/{PRODUCTION_AMI_ID}"
    distribution_arn = (
        f"arn:aws:cloudfront::{account_id}:distribution/*"
    )
    parity_tag_keys = [
        "Name",
        "leadpoet:candidate-sha",
        "leadpoet:ephemeral",
        "leadpoet:parity-run",
    ]
    request_tag_condition = {
        "StringEquals": {
            "aws:RequestTag/leadpoet:ephemeral": "true",
        },
        "StringLike": {
            "aws:RequestTag/Name": "leadpoet-parity-pp-*-*",
            "aws:RequestTag/leadpoet:parity-run": "pp-*-*",
            "aws:RequestTag/leadpoet:candidate-sha": (
                "????????????????????????????????????????"
            ),
        },
        "ForAllValues:StringEquals": {
            "aws:TagKeys": parity_tag_keys,
        },
    }
    ec2_resource_tag_condition = {
        "StringEquals": {
            "ec2:ResourceTag/leadpoet:ephemeral": "true",
        },
        "StringLike": {
            "ec2:ResourceTag/Name": "leadpoet-parity-pp-*-*",
            "ec2:ResourceTag/leadpoet:parity-run": "pp-*-*",
            "ec2:ResourceTag/leadpoet:candidate-sha": (
                "????????????????????????????????????????"
            ),
        },
    }
    cloudfront_resource_tag_condition = {
        "StringEquals": {
            "aws:ResourceTag/leadpoet:ephemeral": "true",
        },
        "StringLike": {
            "aws:ResourceTag/Name": "leadpoet-parity-pp-*-*",
            "aws:ResourceTag/leadpoet:parity-run": "pp-*-*",
            "aws:ResourceTag/leadpoet:candidate-sha": (
                "????????????????????????????????????????"
            ),
        },
    }
    ssm_resource_tag_condition = {
        "StringEquals": {
            "ssm:resourceTag/leadpoet:ephemeral": "true",
        },
        "StringLike": {
            "ssm:resourceTag/Name": "leadpoet-parity-pp-*-*",
            "ssm:resourceTag/leadpoet:parity-run": "pp-*-*",
            "ssm:resourceTag/leadpoet:candidate-sha": (
                "????????????????????????????????????????"
            ),
        },
    }
    cloudfront_create_tag_condition = {
        **request_tag_condition,
        "StringEquals": {
            **request_tag_condition["StringEquals"],
            **cloudfront_resource_tag_condition["StringEquals"],
        },
        "StringLike": {
            **request_tag_condition["StringLike"],
            **cloudfront_resource_tag_condition["StringLike"],
        },
    }
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadEc2Inventory",
                "Effect": "Allow",
                "Action": "ec2:Describe*",
                "Resource": "*",
            },
            {
                "Sid": "LaunchFromAccountImage",
                "Effect": "Allow",
                "Action": "ec2:RunInstances",
                "Resource": image_arn,
            },
            {
                "Sid": "LaunchInAccountSubnet",
                "Effect": "Allow",
                "Action": "ec2:RunInstances",
                "Resource": subnet_arn,
            },
            {
                "Sid": "LaunchWithOwnedSecurityGroup",
                "Effect": "Allow",
                "Action": "ec2:RunInstances",
                "Resource": security_group_arn,
                "Condition": {
                    **ec2_resource_tag_condition,
                    "ArnEquals": {"ec2:Vpc": vpc_arn},
                },
            },
            {
                "Sid": "CreateTaggedParityInstance",
                "Effect": "Allow",
                "Action": "ec2:RunInstances",
                "Resource": instance_arn,
                "Condition": {
                    **request_tag_condition,
                    "StringEquals": {
                        **request_tag_condition["StringEquals"],
                        "ec2:InstanceMetadataTags": "enabled",
                        "ec2:InstanceType": PRODUCTION_INSTANCE_TYPE,
                        "ec2:MetadataHttpEndpoint": "enabled",
                        "ec2:MetadataHttpTokens": "required",
                        "ec2:Region": region,
                    },
                    "NumericEquals": {
                        "ec2:MetadataHttpPutResponseHopLimit": "2",
                    },
                    "ArnEquals": {
                        "ec2:InstanceProfile": (
                            f"arn:aws:iam::{account_id}:instance-profile/"
                            f"{RUNNER_PROFILE}"
                        ),
                    },
                },
            },
            {
                "Sid": "CreateTaggedParityVolume",
                "Effect": "Allow",
                "Action": "ec2:RunInstances",
                "Resource": volume_arn,
                "Condition": {
                    **request_tag_condition,
                    "StringEquals": {
                        **request_tag_condition["StringEquals"],
                        "ec2:Region": region,
                        "ec2:VolumeType": "gp3",
                    },
                    "Bool": {"ec2:Encrypted": "true"},
                    "NumericEquals": {"ec2:VolumeSize": "512"},
                },
            },
            {
                "Sid": "CreateTaggedParityNetworkInterface",
                "Effect": "Allow",
                "Action": "ec2:RunInstances",
                "Resource": network_interface_arn,
                "Condition": {
                    **request_tag_condition,
                    "StringEquals": {
                        **request_tag_condition["StringEquals"],
                        "ec2:Region": region,
                    },
                    "ArnEquals": {
                        "ec2:Subnet": subnet_arn,
                        "ec2:Vpc": vpc_arn,
                    },
                    "Bool": {"ec2:AssociatePublicIpAddress": "true"},
                },
            },
            {
                "Sid": "CreateTaggedLaunchResourceTags",
                "Effect": "Allow",
                "Action": "ec2:CreateTags",
                "Resource": [instance_arn, volume_arn, network_interface_arn],
                "Condition": {
                    **request_tag_condition,
                    "StringEquals": {
                        **request_tag_condition["StringEquals"],
                        "ec2:CreateAction": "RunInstances",
                    },
                },
            },
            {
                "Sid": "CreateTaggedSecurityGroup",
                "Effect": "Allow",
                "Action": "ec2:CreateSecurityGroup",
                "Resource": security_group_arn,
                "Condition": request_tag_condition,
            },
            {
                "Sid": "CreateSecurityGroupInAccountVpc",
                "Effect": "Allow",
                "Action": "ec2:CreateSecurityGroup",
                "Resource": vpc_arn,
            },
            {
                "Sid": "CreateSecurityGroupTags",
                "Effect": "Allow",
                "Action": "ec2:CreateTags",
                "Resource": security_group_arn,
                "Condition": {
                    **request_tag_condition,
                    "StringEquals": {
                        **request_tag_condition["StringEquals"],
                        "ec2:CreateAction": "CreateSecurityGroup",
                    },
                },
            },
            {
                "Sid": "TerminateOnlyOwnedParityInstances",
                "Effect": "Allow",
                "Action": "ec2:TerminateInstances",
                "Resource": instance_arn,
                "Condition": ec2_resource_tag_condition,
            },
            {
                "Sid": "MutateOnlyOwnedParitySecurityGroups",
                "Effect": "Allow",
                "Action": [
                    "ec2:DeleteSecurityGroup",
                    "ec2:AuthorizeSecurityGroupIngress",
                ],
                "Resource": security_group_arn,
                "Condition": {
                    **ec2_resource_tag_condition,
                    "ArnEquals": {"ec2:Vpc": vpc_arn},
                },
            },
            {
                "Sid": "CreateTaggedCloudFrontDistribution",
                "Effect": "Allow",
                "Action": "cloudfront:CreateDistribution",
                "Resource": "*",
                "Condition": request_tag_condition,
            },
            {
                "Sid": "ApplyOnlyParityTagsToAccountDistribution",
                "Effect": "Allow",
                "Action": "cloudfront:TagResource",
                "Resource": distribution_arn,
                "Condition": cloudfront_create_tag_condition,
            },
            {
                "Sid": "ReadCloudFrontInventory",
                "Effect": "Allow",
                "Action": [
                    "cloudfront:ListDistributions",
                    "cloudfront:ListCachePolicies",
                    "cloudfront:ListOriginRequestPolicies",
                ],
                "Resource": "*",
            },
            {
                "Sid": "ReadAccountDistributionTags",
                "Effect": "Allow",
                "Action": "cloudfront:ListTagsForResource",
                "Resource": distribution_arn,
            },
            {
                "Sid": "ReadAndDeleteOnlyOwnedDistributions",
                "Effect": "Allow",
                "Action": [
                    "cloudfront:GetDistribution",
                    "cloudfront:GetDistributionConfig",
                    "cloudfront:UpdateDistribution",
                    "cloudfront:DeleteDistribution",
                ],
                "Resource": distribution_arn,
                "Condition": cloudfront_resource_tag_condition,
            },
            {
                "Sid": "SendOnlyAwsShellDocument",
                "Effect": "Allow",
                "Action": "ssm:SendCommand",
                "Resource": (
                    f"arn:aws:ssm:{region}::document/AWS-RunShellScript"
                ),
            },
            {
                "Sid": "SendOnlyToOwnedParityInstances",
                "Effect": "Allow",
                "Action": "ssm:SendCommand",
                "Resource": instance_arn,
                "Condition": ssm_resource_tag_condition,
            },
            {
                "Sid": "ReadSsmCommandState",
                "Effect": "Allow",
                "Action": [
                    "ssm:GetCommandInvocation",
                    "ssm:DescribeInstanceInformation",
                ],
                "Resource": "*",
            },
            {
                "Sid": "PassOnlyParityRunnerToEc2",
                "Effect": "Allow",
                "Action": ["iam:PassRole"],
                "Resource": runner_arn,
                "Condition": {
                    "StringEquals": {
                        "iam:PassedToService": "ec2.amazonaws.com",
                    },
                    "ArnLike": {
                        "iam:AssociatedResourceArn": instance_arn,
                    },
                },
            },
            {
                "Effect": "Allow",
                "Action": ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"],
                "Resource": [
                    f"arn:aws:secretsmanager:{region}:{account_id}:secret:{production_secret_id}-??????",
                    readonly_secret,
                    run_secret,
                ],
            },
            {
                "Effect": "Allow",
                "Action": "secretsmanager:CreateSecret",
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "aws:RequestTag/leadpoet:ephemeral": "true",
                    },
                    "StringLike": {
                        "secretsmanager:Name": (
                            "leadpoet/staging/production-parity/runs/"
                            "pp-*/gateway"
                        ),
                        "aws:RequestTag/leadpoet:parity-run": "????????*",
                        "aws:RequestTag/leadpoet:candidate-sha": "????????????????????????????????????????",
                    },
                },
            },
            {
                "Effect": "Allow",
                "Action": "secretsmanager:TagResource",
                "Resource": run_secret,
                "Condition": {
                    "StringEquals": {
                        "aws:RequestTag/leadpoet:ephemeral": "true",
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
                "Action": "secretsmanager:DeleteSecret",
                "Resource": legacy_run_secret,
                "Condition": {
                    "StringEquals": {
                        "secretsmanager:ResourceTag/leadpoet:ephemeral": "true"
                    },
                    "StringLike": {
                        "secretsmanager:ResourceTag/leadpoet:parity-run": "pp-*",
                        "secretsmanager:ResourceTag/leadpoet:candidate-sha": (
                            "????????????????????????????????????????"
                        ),
                    },
                },
            },
            {
                "Effect": "Deny",
                "Action": [
                    "secretsmanager:CreateSecret",
                    "secretsmanager:PutSecretValue",
                    "secretsmanager:UpdateSecret",
                    "secretsmanager:DeleteSecret",
                    "secretsmanager:TagResource",
                    "secretsmanager:UntagResource",
                ],
                "Resource": [readonly_secret, miner_intake_secret],
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
                    "s3:GetBucketObjectLockConfiguration",
                    "s3:PutBucketObjectLockConfiguration",
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


def _controller_policy_slices(
    *,
    account_id: str,
    region: str,
    production_secret_id: str,
    readonly_secret_id: str,
    miner_intake_secret_id: str,
    runner_arn: str,
) -> dict[str, dict[str, Any]]:
    complete = _controller_policy(
        account_id=account_id,
        region=region,
        production_secret_id=production_secret_id,
        readonly_secret_id=readonly_secret_id,
        miner_intake_secret_id=miner_intake_secret_id,
        runner_arn=runner_arn,
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {
        name: [] for name in CONTROLLER_POLICY_NAMES
    }
    lifecycle_ec2 = {
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:DeleteSecurityGroup",
        "ec2:TerminateInstances",
    }
    for statement in complete["Statement"]:
        raw_actions = statement.get("Action", [])
        actions = {
            str(value)
            for value in (
                raw_actions if isinstance(raw_actions, list) else [raw_actions]
            )
        }
        if actions and all(value.startswith("cloudfront:") for value in actions):
            group = "cloudfront"
        elif actions & lifecycle_ec2 or any(
            value.startswith(("ssm:", "iam:")) for value in actions
        ):
            group = "lifecycle"
        elif any(value.startswith("ec2:") for value in actions):
            group = "ec2_launch"
        else:
            group = "data"
        grouped[group].append(statement)
    slices = {
        CONTROLLER_POLICY_NAMES[group]: {
            "Version": "2012-10-17",
            "Statement": statements,
        }
        for group, statements in grouped.items()
    }
    if any(not document["Statement"] for document in slices.values()):
        raise SetupError("controller IAM managed-policy partition is incomplete")
    if sum(len(value["Statement"]) for value in slices.values()) != len(
        complete["Statement"]
    ):
        raise SetupError("controller IAM managed-policy partition differs")
    for name, document in slices.items():
        if len(_json(document)) > 6144:
            raise SetupError(f"IAM managed policy {name} exceeds the AWS quota")
    return slices


def _simulation_context(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, value in sorted(values.items()):
        items = value if isinstance(value, list) else [value]
        if name == "aws:TagKeys":
            value_type = "stringList"
        elif name in {
            "ec2:MetadataHttpPutResponseHopLimit",
            "ec2:VolumeSize",
        }:
            value_type = "numeric"
        elif name in {
            "ec2:AssociatePublicIpAddress",
            "ec2:Encrypted",
        }:
            value_type = "boolean"
        elif name == "aws:TokenIssueTime":
            value_type = "date"
        else:
            value_type = "string"
        result.append({
            "ContextKeyName": name,
            "ContextKeyValues": [str(item) for item in items],
            "ContextKeyType": value_type,
        })
    return result


def _aws_aggregate_resource(value: Any) -> bool:
    return (
        value == "*"
        or (
            isinstance(value, str)
            and 1 <= len(value) <= 2048
            and value.startswith("arn:")
            and not any(character.isspace() for character in value)
            and re.search(r"\$\{[A-Za-z][A-Za-z0-9]*\}", value) is not None
        )
    )


def _normalize_simulation_results(
    results: Any,
    *,
    action: str,
    requested_resources: Sequence[str],
) -> tuple[dict[str, str], set[str]]:
    if (
        not isinstance(results, list)
        or not results
        or not requested_resources
        or len(requested_resources) != len(set(requested_resources))
        or any(
            not isinstance(resource, str) or not resource
            for resource in requested_resources
        )
    ):
        raise SetupError("controller IAM simulation response differs")

    decisions: dict[str, str] = {}
    missing_context: set[str] = set()
    representation: str | None = None
    valid_decisions = {"allowed", "explicitDeny", "implicitDeny"}

    def add_missing(values: Any) -> None:
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise SetupError("controller IAM simulation response differs")
        missing_context.update(values)

    def add_decision(resource: Any, decision: Any) -> str:
        if (
            not isinstance(resource, str)
            or not resource
            or decision not in valid_decisions
            or resource in decisions
        ):
            raise SetupError("controller IAM simulation response differs")
        decisions[resource] = decision
        return decision

    for result in results:
        if not isinstance(result, Mapping):
            raise SetupError("controller IAM simulation response differs")
        returned_action = result.get("EvalActionName")
        if (
            not isinstance(returned_action, str)
            or returned_action.lower() != action.lower()
        ):
            raise SetupError("controller IAM simulation response differs")
        has_top_resource = "EvalResourceName" in result
        if has_top_resource and (
            not isinstance(result["EvalResourceName"], str)
            or not result["EvalResourceName"]
        ):
            raise SetupError("controller IAM simulation response differs")
        specific = result.get("ResourceSpecificResults", [])
        if not isinstance(specific, list):
            raise SetupError("controller IAM simulation response differs")
        current = "nested" if specific else "flat"
        if representation is not None and representation != current:
            raise SetupError("controller IAM simulation response differs")
        representation = current
        add_missing(result.get("MissingContextValues", []))

        if current == "flat":
            resource = result.get("EvalResourceName")
            decision = result.get("EvalDecision")
            aggregate_only = len(results) == 1 and (
                not has_top_resource or _aws_aggregate_resource(resource)
            )
            if aggregate_only:
                if decision not in valid_decisions:
                    raise SetupError("controller IAM simulation response differs")
                if len(requested_resources) == 1:
                    add_decision(requested_resources[0], decision)
                elif decision == "allowed":
                    for requested_resource in requested_resources:
                        add_decision(requested_resource, decision)
                else:
                    raise SetupError("controller IAM simulation response differs")
                continue
            add_decision(resource, decision)
            continue
        if len(results) != 1:
            raise SetupError("controller IAM simulation response differs")
        aggregate = str(result.get("EvalDecision") or "")
        if aggregate not in valid_decisions:
            raise SetupError("controller IAM simulation response differs")
        nested_decisions: set[str] = set()
        for item in specific:
            if not isinstance(item, Mapping):
                raise SetupError("controller IAM simulation response differs")
            nested_decisions.add(add_decision(
                item.get("EvalResourceName"),
                item.get("EvalResourceDecision"),
            ))
            add_missing(item.get("MissingContextValues", []))
        expected_aggregate = (
            "explicitDeny"
            if "explicitDeny" in nested_decisions
            else (
                "implicitDeny"
                if "implicitDeny" in nested_decisions
                else "allowed"
            )
        )
        if aggregate != expected_aggregate:
            raise SetupError("controller IAM simulation response differs")

    if not decisions:
        raise SetupError("controller IAM simulation response differs")
    if representation == "flat" and len(results) > 1 and set(
        decisions.values()
    ) != {"allowed"}:
        # Historical N-row responses repeated one aggregate decision rather
        # than reporting per-resource decisions.  Aggregate allowed proves all
        # requested resources allowed; an aggregate denial cannot prove every
        # negative resource was denied.
        raise SetupError("controller IAM simulation response differs")
    return decisions, missing_context


def _simulate_controller_policy(
    iam: Any,
    *,
    account_id: str,
    region: str,
    controller_arn: str,
    runner_arn: str,
    session_cutoff: str,
    policy_documents: Sequence[Mapping[str, Any]],
) -> None:
    ec2_prefix = f"arn:aws:ec2:{region}:{account_id}"
    instance = ec2_prefix + ":instance/i-0123456789abcdef0"
    volume = ec2_prefix + ":volume/vol-0123456789abcdef0"
    interface = ec2_prefix + ":network-interface/eni-0123456789abcdef0"
    security_group = ec2_prefix + ":security-group/sg-0123456789abcdef0"
    subnet = ec2_prefix + f":subnet/{PRODUCTION_SUBNET_ID}"
    vpc = ec2_prefix + f":vpc/{PRODUCTION_VPC_ID}"
    image = f"arn:aws:ec2:{region}::image/{PRODUCTION_AMI_ID}"
    distribution = f"arn:aws:cloudfront::{account_id}:distribution/E123456789ABCD"
    document = f"arn:aws:ssm:{region}::document/AWS-RunShellScript"
    tags = {
        "Name": "leadpoet-parity-pp-123456-1",
        "leadpoet:candidate-sha": "a" * 40,
        "leadpoet:ephemeral": "true",
        "leadpoet:parity-run": "pp-123456-1",
    }
    session = {
        "aws:TokenIssueTime": (
            datetime.fromisoformat(session_cutoff.replace("Z", "+00:00"))
            + timedelta(seconds=1)
        ).isoformat().replace("+00:00", "Z"),
    }
    request = {
        **session,
        **{f"aws:RequestTag/{key}": value for key, value in tags.items()},
        "aws:TagKeys": sorted(tags),
    }
    ec2_owned = {
        **session,
        **{f"ec2:ResourceTag/{key}": value for key, value in tags.items()},
        "ec2:Vpc": vpc,
    }
    ssm_owned = {
        **session,
        **{f"ssm:resourceTag/{key}": value for key, value in tags.items()},
    }
    cloudfront_owned = {
        **session,
        **{f"aws:ResourceTag/{key}": value for key, value in tags.items()},
    }
    launch = {
        **request,
        **ec2_owned,
        "ec2:AssociatePublicIpAddress": "true",
        "ec2:Encrypted": "true",
        "ec2:InstanceMetadataTags": "enabled",
        "ec2:InstanceProfile": (
            f"arn:aws:iam::{account_id}:instance-profile/{RUNNER_PROFILE}"
        ),
        "ec2:InstanceType": PRODUCTION_INSTANCE_TYPE,
        "ec2:MetadataHttpEndpoint": "enabled",
        "ec2:MetadataHttpPutResponseHopLimit": "2",
        "ec2:MetadataHttpTokens": "required",
        "ec2:Region": region,
        "ec2:Subnet": subnet,
        "ec2:VolumeSize": "512",
        "ec2:VolumeType": "gp3",
    }
    policy_statements = [
        statement
        for document in policy_documents
        for statement in document.get("Statement", [])
    ]
    if not policy_statements:
        raise SetupError("controller IAM simulation policy set is empty")
    all_condition_keys = {
        str(key)
        for statement in policy_statements
        for conditions in statement.get("Condition", {}).values()
        for key in conditions
    }

    def applicable_condition_keys(action: str) -> set[str]:
        keys: set[str] = set()
        for statement in policy_statements:
            raw_actions = statement.get("Action", [])
            actions = raw_actions if isinstance(raw_actions, list) else [raw_actions]
            if not any(
                fnmatchcase(action.lower(), str(pattern).lower())
                for pattern in actions
            ):
                continue
            for conditions in statement.get("Condition", {}).values():
                keys.update(str(key) for key in conditions)
        return keys

    cases: list[tuple[str, str, list[str], Mapping[str, Any], str]] = [
        (
            "launch-positive",
            "ec2:RunInstances",
            [image, subnet, security_group, instance, volume, interface],
            launch,
            "allowed",
        ),
        (
            "launch-wrong-instance-type",
            "ec2:RunInstances",
            [instance],
            {**launch, "ec2:InstanceType": "m5.large"},
            "implicitDeny",
        ),
        (
            "launch-alternate-image",
            "ec2:RunInstances",
            [f"arn:aws:ec2:{region}::image/ami-0123456789abcdef0"],
            launch,
            "implicitDeny",
        ),
        (
            "launch-alternate-subnet",
            "ec2:RunInstances",
            [ec2_prefix + ":subnet/subnet-0123456789abcdef0"],
            launch,
            "implicitDeny",
        ),
        (
            "launch-wrong-instance-profile",
            "ec2:RunInstances",
            [instance],
            {
                **launch,
                "ec2:InstanceProfile": (
                    f"arn:aws:iam::{account_id}:instance-profile/other"
                ),
            },
            "implicitDeny",
        ),
        (
            "launch-wrong-volume-size",
            "ec2:RunInstances",
            [volume],
            {**launch, "ec2:VolumeSize": "200"},
            "implicitDeny",
        ),
        (
            "launch-unencrypted-volume",
            "ec2:RunInstances",
            [volume],
            {**launch, "ec2:Encrypted": "false"},
            "implicitDeny",
        ),
        (
            "launch-imdsv1",
            "ec2:RunInstances",
            [instance],
            {**launch, "ec2:MetadataHttpTokens": "optional"},
            "implicitDeny",
        ),
        (
            "launch-network-wrong-subnet",
            "ec2:RunInstances",
            [interface],
            {
                **launch,
                "ec2:Subnet": (
                    ec2_prefix + ":subnet/subnet-0123456789abcdef0"
                ),
            },
            "implicitDeny",
        ),
        (
            "launch-unowned-security-group",
            "ec2:RunInstances",
            [security_group],
            {
                **launch,
                "ec2:ResourceTag/leadpoet:ephemeral": "false",
            },
            "implicitDeny",
        ),
        (
            "launch-missing-request-tag",
            "ec2:RunInstances",
            [instance],
            {
                key: value
                for key, value in launch.items()
                if key != "aws:RequestTag/leadpoet:candidate-sha"
            },
            "implicitDeny",
        ),
        (
            "launch-tags-positive",
            "ec2:CreateTags",
            [instance, volume, interface],
            {**request, "ec2:CreateAction": "RunInstances"},
            "allowed",
        ),
        (
            "launch-tags-wrong-create-action",
            "ec2:CreateTags",
            [instance],
            {**request, "ec2:CreateAction": "CreateSecurityGroup"},
            "implicitDeny",
        ),
        (
            "security-group-tags-positive",
            "ec2:CreateTags",
            [security_group],
            {**request, "ec2:CreateAction": "CreateSecurityGroup"},
            "allowed",
        ),
        (
            "security-group-create-positive",
            "ec2:CreateSecurityGroup",
            [security_group, vpc],
            request,
            "allowed",
        ),
        (
            "security-group-create-unowned",
            "ec2:CreateSecurityGroup",
            [security_group],
            {
                **request,
                "aws:RequestTag/leadpoet:ephemeral": "false",
            },
            "implicitDeny",
        ),
        (
            "terminate-owned",
            "ec2:TerminateInstances",
            [instance],
            ec2_owned,
            "allowed",
        ),
        (
            "terminate-unowned",
            "ec2:TerminateInstances",
            [instance],
            {**ec2_owned, "ec2:ResourceTag/leadpoet:ephemeral": "false"},
            "implicitDeny",
        ),
        (
            "security-group-delete-owned",
            "ec2:DeleteSecurityGroup",
            [security_group],
            ec2_owned,
            "allowed",
        ),
        (
            "security-group-delete-unowned",
            "ec2:DeleteSecurityGroup",
            [security_group],
            {**ec2_owned, "ec2:ResourceTag/leadpoet:ephemeral": "false"},
            "implicitDeny",
        ),
        (
            "security-group-ingress-owned",
            "ec2:AuthorizeSecurityGroupIngress",
            [security_group],
            ec2_owned,
            "allowed",
        ),
        (
            "security-group-ingress-unowned",
            "ec2:AuthorizeSecurityGroupIngress",
            [security_group],
            {**ec2_owned, "ec2:ResourceTag/leadpoet:ephemeral": "false"},
            "implicitDeny",
        ),
        (
            "ssm-positive",
            "ssm:SendCommand",
            [document, instance],
            ssm_owned,
            "allowed",
        ),
        (
            "ssm-other-document",
            "ssm:SendCommand",
            [f"arn:aws:ssm:{region}::document/OtherDocument"],
            ssm_owned,
            "implicitDeny",
        ),
        (
            "ssm-unowned-instance",
            "ssm:SendCommand",
            [instance],
            {
                **ssm_owned,
                "ssm:resourceTag/leadpoet:ephemeral": "false",
            },
            "implicitDeny",
        ),
        (
            "cloudfront-create-positive",
            "cloudfront:CreateDistribution",
            ["*"],
            request,
            "allowed",
        ),
        (
            "cloudfront-create-unowned",
            "cloudfront:CreateDistribution",
            ["*"],
            {**request, "aws:RequestTag/leadpoet:ephemeral": "false"},
            "implicitDeny",
        ),
        (
            "cloudfront-tag-positive",
            "cloudfront:TagResource",
            [distribution],
            {**request, **cloudfront_owned},
            "allowed",
        ),
        (
            "cloudfront-tag-takeover",
            "cloudfront:TagResource",
            [distribution],
            {
                **request,
                **cloudfront_owned,
                "aws:ResourceTag/leadpoet:ephemeral": "false",
            },
            "implicitDeny",
        ),
        (
            "cloudfront-update-owned",
            "cloudfront:UpdateDistribution",
            [distribution],
            cloudfront_owned,
            "allowed",
        ),
        (
            "cloudfront-update-unowned",
            "cloudfront:UpdateDistribution",
            [distribution],
            {
                **cloudfront_owned,
                "aws:ResourceTag/leadpoet:ephemeral": "false",
            },
            "implicitDeny",
        ),
        (
            "cloudfront-delete-owned",
            "cloudfront:DeleteDistribution",
            [distribution],
            cloudfront_owned,
            "allowed",
        ),
        (
            "cloudfront-delete-unowned",
            "cloudfront:DeleteDistribution",
            [distribution],
            {
                **cloudfront_owned,
                "aws:ResourceTag/leadpoet:ephemeral": "false",
            },
            "implicitDeny",
        ),
        (
            "pass-role-positive",
            "iam:PassRole",
            [runner_arn],
            {
                **session,
                "iam:AssociatedResourceArn": instance,
                "iam:PassedToService": "ec2.amazonaws.com",
            },
            "allowed",
        ),
        (
            "pass-role-wrong-service",
            "iam:PassRole",
            [runner_arn],
            {
                **session,
                "iam:AssociatedResourceArn": instance,
                "iam:PassedToService": "lambda.amazonaws.com",
            },
            "implicitDeny",
        ),
        (
            "pass-role-wrong-role",
            "iam:PassRole",
            [f"arn:aws:iam::{account_id}:role/other"],
            {
                **session,
                "iam:AssociatedResourceArn": instance,
                "iam:PassedToService": "ec2.amazonaws.com",
            },
            "implicitDeny",
        ),
    ]
    for name, action, resources, context, expected in cases:
        response = iam.simulate_principal_policy(
            PolicySourceArn=controller_arn,
            ActionNames=[action],
            ResourceArns=resources,
            # S3 bucket and object ARNs omit their owner account. Keep the
            # simulator bound to the same account as the controller instead of
            # relying on an implicit caller-derived default.
            ResourceOwner=f"arn:aws:iam::{account_id}:root",
            ContextEntries=_simulation_context(context),
        )
        if response.get("IsTruncated"):
            raise SetupError(f"controller IAM simulation {name} was truncated")
        try:
            decisions, missing_context = _normalize_simulation_results(
                response.get("EvaluationResults"),
                action=action,
                requested_resources=resources,
            )
        except SetupError as exc:
            raise SetupError(
                f"controller IAM simulation {name} differs"
            ) from exc
        expected_missing = (
            {"aws:RequestTag/leadpoet:candidate-sha"}
            if name == "launch-missing-request-tag"
            else set()
        )
        applicable_missing = missing_context & applicable_condition_keys(action)
        if (
            set(decisions) != set(resources)
            or set(decisions.values()) != {expected}
            or not missing_context.issubset(all_condition_keys)
            or applicable_missing != expected_missing
        ):
            raise SetupError(f"controller IAM simulation {name} differs")


def _runner_policy(
    *,
    account_id: str,
    region: str,
    production_secret_id: str,
    readonly_secret_id: str,
    miner_intake_secret_id: str,
) -> dict[str, Any]:
    run_secret = (
        f"arn:aws:secretsmanager:{region}:{account_id}:secret:"
        "leadpoet/staging/production-parity/runs/pp-*/gateway-??????"
    )
    readonly_secret = (
        f"arn:aws:secretsmanager:{region}:{account_id}:secret:"
        f"{readonly_secret_id}-??????"
    )
    miner_intake_secret = (
        f"arn:aws:secretsmanager:{region}:{account_id}:secret:"
        f"{miner_intake_secret_id}-??????"
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
                    "arn:aws:s3:::leadpoet-attested-v2-artifacts-*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": [
                    run_objects,
                    "arn:aws:s3:::leadpoet-attested-v2-artifacts-*/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": ["s3:GetObjectVersion"],
                "Resource": (
                    "arn:aws:s3:::leadpoet-attested-v2-artifacts-*/*"
                ),
            },
            {
                "Effect": "Allow",
                "Action": ["s3:GetObjectRetention"],
                "Resource": [
                    run_objects,
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
                    "s3:GetBucketObjectLockConfiguration",
                    "s3:GetBucketLocation",
                ],
                "Resource": parity_bucket,
            },
            {
                "Effect": "Allow",
                "Action": ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"],
                "Resource": [
                    f"arn:aws:secretsmanager:{region}:{account_id}:secret:{production_secret_id}-??????",
                    readonly_secret,
                    miner_intake_secret,
                    run_secret,
                ],
            },
            {
                "Effect": "Allow",
                "Action": "secretsmanager:CreateSecret",
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "aws:RequestTag/leadpoet:ephemeral": "true",
                    },
                    "StringLike": {
                        "secretsmanager:Name": (
                            "leadpoet/staging/production-parity/runs/"
                            "pp-*/gateway"
                        ),
                        "aws:RequestTag/leadpoet:parity-run": "????????*",
                        "aws:RequestTag/leadpoet:candidate-sha": "????????????????????????????????????????",
                    },
                },
            },
            {
                "Effect": "Allow",
                "Action": "secretsmanager:TagResource",
                "Resource": run_secret,
                "Condition": {
                    "StringEquals": {
                        "aws:RequestTag/leadpoet:ephemeral": "true",
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
                "Effect": "Deny",
                "Action": [
                    "secretsmanager:CreateSecret",
                    "secretsmanager:PutSecretValue",
                    "secretsmanager:UpdateSecret",
                    "secretsmanager:DeleteSecret",
                    "secretsmanager:TagResource",
                    "secretsmanager:UntagResource",
                ],
                "Resource": [readonly_secret, miner_intake_secret],
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


def _static_bootstrap_policy(
    *,
    account_id: str,
    region: str,
    readonly_secret_id: str,
    miner_intake_secret_id: str,
    expires_at: str,
) -> dict[str, Any]:
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SetupError("static bootstrap policy expiry is invalid") from exc
    if expiry.tzinfo is None:
        raise SetupError("static bootstrap policy expiry is invalid")
    static_secret_bases = [
        (
            f"arn:aws:secretsmanager:{region}:{account_id}:secret:"
            f"{secret_id}"
        )
        for secret_id in (readonly_secret_id, miner_intake_secret_id)
    ]
    static_secrets = [
        resource
        for base in static_secret_bases
        for resource in (base, base + "-??????")
    ]
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "sts:GetCallerIdentity",
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:GetSecretValue",
                    "secretsmanager:DescribeSecret",
                    "secretsmanager:GetResourcePolicy",
                    "secretsmanager:ListSecretVersionIds",
                ],
                "Resource": static_secrets,
            },
            {
                "Effect": "Allow",
                "Action": "secretsmanager:CreateSecret",
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "secretsmanager:Name": [
                            readonly_secret_id,
                            miner_intake_secret_id,
                        ],
                        "aws:RequestTag/leadpoet:purpose": (
                            "production-parity-static"
                        ),
                        "aws:RequestTag/leadpoet:parity-static-bootstrap": (
                            "true"
                        ),
                    },
                    "StringLike": {
                        "aws:RequestTag/leadpoet:candidate-sha": (
                            "????????????????????????????????????????"
                        ),
                    },
                    "ForAllValues:StringEquals": {
                        "aws:TagKeys": [
                            "leadpoet:purpose",
                            "leadpoet:parity-static-bootstrap",
                            "leadpoet:candidate-sha",
                        ]
                    },
                },
            },
            {
                "Effect": "Allow",
                "Action": "secretsmanager:TagResource",
                "Resource": static_secrets,
                "Condition": {
                    "StringEquals": {
                        "aws:RequestTag/leadpoet:purpose": (
                            "production-parity-static"
                        ),
                        "aws:RequestTag/leadpoet:parity-static-bootstrap": (
                            "true"
                        ),
                    },
                    "StringLike": {
                        "aws:RequestTag/leadpoet:candidate-sha": (
                            "????????????????????????????????????????"
                        ),
                    },
                    "ForAllValues:StringEquals": {
                        "aws:TagKeys": [
                            "leadpoet:purpose",
                            "leadpoet:parity-static-bootstrap",
                            "leadpoet:candidate-sha",
                        ]
                    },
                },
            },
            {
                "Effect": "Deny",
                "Action": [
                    "secretsmanager:PutSecretValue",
                    "secretsmanager:UpdateSecret",
                    "secretsmanager:DeleteSecret",
                    "secretsmanager:UntagResource",
                ],
                "Resource": static_secrets,
            },
            {
                "Effect": "Deny",
                "Action": [
                    "secretsmanager:GetSecretValue",
                    "secretsmanager:DescribeSecret",
                    "secretsmanager:GetResourcePolicy",
                    "secretsmanager:ListSecretVersionIds",
                    "secretsmanager:PutSecretValue",
                    "secretsmanager:UpdateSecret",
                    "secretsmanager:DeleteSecret",
                    "secretsmanager:TagResource",
                    "secretsmanager:UntagResource",
                ],
                "NotResource": static_secrets,
            },
            {
                # IAM role sessions can outlive the trust-policy cutoff that
                # prevents new sessions.  This absolute deny makes any already
                # issued bootstrap session lose all authority at the same
                # cutoff even if role cleanup is interrupted.
                "Effect": "Deny",
                "Action": "*",
                "Resource": "*",
                "Condition": {
                    "DateGreaterThanEquals": {
                        "aws:CurrentTime": expires_at,
                    }
                },
            },
        ],
    }


def _static_bootstrap_trust(
    *, account_id: str, validator_role_name: str, expires_at: str
) -> dict[str, Any]:
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SetupError("static bootstrap trust expiry is invalid") from exc
    if expiry.tzinfo is None:
        raise SetupError("static bootstrap trust expiry is invalid")
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "AWS": (
                    f"arn:aws:iam::{account_id}:role/{validator_role_name}"
                )
            },
            "Action": "sts:AssumeRole",
            "Condition": {"DateLessThan": {"aws:CurrentTime": expires_at}},
        }],
    }


def _delete_static_bootstrap_role(iam: Any, *, account_id: str) -> None:
    try:
        role = iam.get_role(RoleName=STATIC_BOOTSTRAP_ROLE)["Role"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "NoSuchEntity":
            return
        raise
    tags = {
        str(item.get("Key") or ""): str(item.get("Value") or "")
        for item in iam.list_role_tags(RoleName=STATIC_BOOTSTRAP_ROLE).get("Tags", [])
    }
    if (
        str(role.get("Arn") or "")
        != f"arn:aws:iam::{account_id}:role/{STATIC_BOOTSTRAP_ROLE}"
        or str(role.get("Path") or "/") != "/"
        or tags != {"leadpoet:purpose": "production-parity"}
        or role.get("PermissionsBoundary")
    ):
        raise SetupError("static bootstrap role collision is not parity-owned")
    policies = iam.list_role_policies(RoleName=STATIC_BOOTSTRAP_ROLE).get(
        "PolicyNames", []
    )
    if set(policies) - {STATIC_BOOTSTRAP_POLICY}:
        raise SetupError("static bootstrap role has an unexpected inline policy")
    attached = iam.list_attached_role_policies(RoleName=STATIC_BOOTSTRAP_ROLE)
    if attached.get("AttachedPolicies"):
        raise SetupError("static bootstrap role has an unexpected attached policy")
    for policy_name in policies:
        iam.delete_role_policy(
            RoleName=STATIC_BOOTSTRAP_ROLE,
            PolicyName=str(policy_name),
        )
    iam.delete_role(RoleName=STATIC_BOOTSTRAP_ROLE)
    try:
        iam.get_role(RoleName=STATIC_BOOTSTRAP_ROLE)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "NoSuchEntity":
            return
        raise
    raise SetupError("static bootstrap role deletion did not converge")


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


def _gh_variable_value(repository: str, name: str) -> str:
    result = subprocess.run(
        [
            "gh",
            "variable",
            "get",
            name,
            "--repo",
            repository,
            "--json",
            "value",
            "--jq",
            ".value",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise SetupError(f"GitHub variable {name} could not be read back")
    return result.stdout.strip()


def _gh_variable_names(repository: str) -> set[str]:
    result = subprocess.run(
        [
            "gh",
            "variable",
            "list",
            "--repo",
            repository,
            "--json",
            "name",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise SetupError("GitHub variables could not be listed")
    try:
        values = json.loads(result.stdout)
    except ValueError as exc:
        raise SetupError("GitHub variables list is invalid") from exc
    if not isinstance(values, list):
        raise SetupError("GitHub variables list is invalid")
    return {
        str(item.get("name") or "")
        for item in values
        if isinstance(item, Mapping) and item.get("name")
    }


def _delete_gh_variable(repository: str, name: str) -> None:
    if name not in _gh_variable_names(repository):
        return
    result = subprocess.run(
        ["gh", "variable", "delete", name, "--repo", repository],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0 or name in _gh_variable_names(repository):
        raise SetupError(f"GitHub variable {name} could not be removed")


def _validate_identity_inputs(args: argparse.Namespace) -> None:
    if (
        args.repository != DEFAULT_REPOSITORY
        or args.region != EXPECTED_REGION
        or args.production_gateway_ip != PRODUCTION_GATEWAY_IP
        or args.production_gateway_secret_id != PRODUCTION_GATEWAY_SECRET_ID
        or args.readonly_dsn_secret_id != READONLY_DSN_SECRET_ID
        or args.miner_intake_secret_id != DEFAULT_MINER_INTAKE_SECRET_ID
        or int(args.volume_gib) != DEFAULT_VOLUME_GIB
    ):
        raise SetupError("setup inputs are invalid")


def _validate_gateway_and_images(args: argparse.Namespace) -> tuple[str, str]:
    _validate_identity_inputs(args)
    if args.production_gateway_url.rstrip("/") != PRODUCTION_GATEWAY_URL:
        raise SetupError("production gateway URL differs")
    with urlopen(
        args.production_gateway_url.rstrip("/") + "/build-info", timeout=20
    ) as response:
        value = json.load(response)
    commit = str(value.get("git_commit") or value.get("commit_sha") or "")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SetupError("production gateway build identity is unavailable")
    postgres = _docker_hub_image(
        "library/postgres", "15", "postgres"
    )
    postgrest = _docker_hub_image(
        "postgrest/postgrest", "v12.2.3", "postgrest/postgrest"
    )
    if (
        PINNED_IMAGE_RE.fullmatch(postgres) is None
        or PINNED_IMAGE_RE.fullmatch(postgrest) is None
    ):
        raise SetupError("container images must be immutable digest references")
    return postgres, postgrest


def _gateway_iam_session() -> Any:
    forbidden = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    }
    if any(os.environ.get(name) for name in forbidden):
        raise SetupError("ambient AWS credential selectors are forbidden")
    try:
        before = GATEWAY_IAM_CACHE.lstat()
    except OSError as exc:
        raise SetupError("gateway IAM cache is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or GATEWAY_IAM_CACHE.is_symlink()
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or before.st_mode & 0o077
        or not 1 <= before.st_size <= 1024 * 1024
    ):
        raise SetupError("gateway IAM cache metadata is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(GATEWAY_IAM_CACHE, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or opened.st_mode & 0o077
            or opened.st_size != before.st_size
        ):
            raise SetupError("gateway IAM cache changed during open")
        raw = bytearray()
        while len(raw) <= 1024 * 1024:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) != opened.st_size:
            raise SetupError("gateway IAM cache read differs")
    finally:
        os.close(descriptor)

    text = raw.decode("utf-8")
    raw[:] = b"\0" * len(raw)
    values: dict[str, str] = {}
    if text.lstrip().startswith("{"):
        # Walk one top-level JSON object and retain only the two exact fields;
        # never construct or return the full secret-bearing environment map.
        decoder = json.JSONDecoder()
        index = len(text) - len(text.lstrip()) + 1
        seen: set[str] = set()
        if text.lstrip()[:1] != "{":
            raise SetupError("gateway IAM cache is invalid")
        while True:
            while index < len(text) and text[index].isspace():
                index += 1
            if index < len(text) and text[index] == "}":
                index += 1
                break
            try:
                name, index = decoder.raw_decode(text, index)
            except ValueError as exc:
                raise SetupError("gateway IAM cache is invalid") from exc
            if not isinstance(name, str) or name in seen:
                raise SetupError("gateway IAM cache fields are ambiguous")
            seen.add(name)
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text) or text[index] != ":":
                raise SetupError("gateway IAM cache is invalid")
            index += 1
            while index < len(text) and text[index].isspace():
                index += 1
            try:
                value, index = decoder.raw_decode(text, index)
            except ValueError as exc:
                raise SetupError("gateway IAM cache is invalid") from exc
            if name in {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}:
                if not isinstance(value, str):
                    raise SetupError("gateway IAM cache field is invalid")
                values[name] = value
            value = None
            while index < len(text) and text[index].isspace():
                index += 1
            if index < len(text) and text[index] == ",":
                index += 1
                continue
            if index < len(text) and text[index] == "}":
                index += 1
                break
            raise SetupError("gateway IAM cache is invalid")
        if text[index:].strip():
            raise SetupError("gateway IAM cache is invalid")
    else:
        counts: dict[str, int] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise SetupError("gateway IAM cache is invalid")
            name, value = line.split("=", 1)
            name = name.strip()
            if name in {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}:
                counts[name] = counts.get(name, 0) + 1
                values[name] = value.strip().strip("'\"")
        if any(counts.get(name) != 1 for name in (
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"
        )):
            raise SetupError("gateway IAM cache fields are ambiguous")
    text = ""
    access_key = values.pop("AWS_ACCESS_KEY_ID", "")
    secret_key = values.pop("AWS_SECRET_ACCESS_KEY", "")
    if (
        re.fullmatch(r"AKIA[A-Z0-9]{16}", access_key) is None
        or not 32 <= len(secret_key) <= 128
        or any(character.isspace() for character in secret_key)
    ):
        raise SetupError("gateway IAM credentials are invalid")
    session = boto3.session.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=EXPECTED_REGION,
    )
    access_key = ""
    secret_key = ""
    return session


def _iam_clients(region: str) -> tuple[Any, Any, str]:
    if region != EXPECTED_REGION:
        raise SetupError("production-parity IAM must use us-east-1")
    session = _gateway_iam_session()
    sts = session.client("sts")
    iam = session.client("iam")
    account_id = str(sts.get_caller_identity().get("Account") or "")
    if account_id != EXPECTED_ACCOUNT_ID:
        raise SetupError("AWS identity is not the production account")
    return sts, iam, account_id


def setup_iam_only(args: argparse.Namespace) -> dict[str, Any]:
    _validate_identity_inputs(args)
    _sts, iam, account_id = _iam_clients(args.region)
    expected_validator_arn = (
        f"arn:aws:iam::{account_id}:role/{VALIDATOR_ROLE}"
    )
    validator = iam.get_role(RoleName=VALIDATOR_ROLE).get("Role", {})
    if str(validator.get("Arn") or "") != expected_validator_arn:
        raise SetupError("validator role identity differs")

    oidc_arn = _ensure_oidc_provider(iam, account_id)
    runner_trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }
    runner_policy_name = "LeadpoetProductionParityRunner"
    runner_managed_policy = (
        "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    )
    runner_policy = _runner_policy(
        account_id=account_id,
        region=args.region,
        production_secret_id=args.production_gateway_secret_id,
        readonly_secret_id=args.readonly_dsn_secret_id,
        miner_intake_secret_id=args.miner_intake_secret_id,
    )
    inert_trust = _inert_trust()
    controller_trust = _controller_trust(oidc_arn=oidc_arn)
    legacy_controller_policy = "LeadpoetProductionParityController"
    managed_policy_arns = {
        _managed_policy_arn(account_id=account_id, name=name)
        for name in CONTROLLER_POLICY_NAMES.values()
    }
    runner_arn = f"arn:aws:iam::{account_id}:role/{RUNNER_ROLE}"
    controller_arn = f"arn:aws:iam::{account_id}:role/{CONTROLLER_ROLE}"
    runner_ready = False
    controller_ready = False
    runner_revocation: dict[str, Any] = {}
    controller_revocation: dict[str, Any] = {}
    bootstrap_cleanup_required = False
    try:
        runner_arn = _ensure_role(
            iam,
            account_id=account_id,
            name=RUNNER_ROLE,
            trust=inert_trust,
            expected_inline_policies={
                runner_policy_name,
                RUNNER_REVOKE_POLICY,
            },
            expected_attached_policies={runner_managed_policy},
            revoke_policy_name=RUNNER_REVOKE_POLICY,
            revocation_receipt=runner_revocation,
        )
        runner_ready = True
        runner_session_cutoff = str(runner_revocation["cutoff"])
        runner_revoke_policy = _policy_document(
            runner_revocation["document"]
        )
        _put_policy(
            iam,
            role=RUNNER_ROLE,
            name=runner_policy_name,
            document=runner_policy,
        )
        iam.attach_role_policy(
            RoleName=RUNNER_ROLE,
            PolicyArn=runner_managed_policy,
        )
        _assert_role_configuration(
            iam,
            account_id=account_id,
            name=RUNNER_ROLE,
            trust=inert_trust,
            inline_policies={
                runner_policy_name: runner_policy,
                RUNNER_REVOKE_POLICY: runner_revoke_policy,
            },
            attached_policies={runner_managed_policy},
            max_session_duration=43200,
        )
        _ensure_instance_profile(iam, account_id=account_id)

        controller_slices = _controller_policy_slices(
            account_id=account_id,
            region=args.region,
            production_secret_id=args.production_gateway_secret_id,
            readonly_secret_id=args.readonly_dsn_secret_id,
            miner_intake_secret_id=args.miner_intake_secret_id,
            runner_arn=runner_arn,
        )
        controller_arn = _ensure_role(
            iam,
            account_id=account_id,
            name=CONTROLLER_ROLE,
            trust=inert_trust,
            expected_inline_policies={
                legacy_controller_policy,
                CONTROLLER_REVOKE_POLICY,
            },
            expected_attached_policies=managed_policy_arns,
            revoke_policy_name=CONTROLLER_REVOKE_POLICY,
            revocation_receipt=controller_revocation,
        )
        controller_ready = True
        controller_session_cutoff = str(controller_revocation["cutoff"])
        controller_revoke_policy = _policy_document(
            controller_revocation["document"]
        )
        # Existing sessions dynamically observe role policies.  Detach every
        # known permission while trust is inert before replacing any managed
        # policy version, then attach the complete exact set atomically enough
        # for a final simulator/readback gate.
        _neutralize_controller_role(
            iam,
            account_id=account_id,
            managed_policy_arns=managed_policy_arns,
            legacy_inline_policy=legacy_controller_policy,
            revoke_document=controller_revoke_policy,
        )
        for name, document in sorted(controller_slices.items()):
            _ensure_managed_policy(
                iam,
                account_id=account_id,
                name=name,
                document=document,
            )
        for policy_arn in sorted(managed_policy_arns):
            iam.attach_role_policy(
                RoleName=CONTROLLER_ROLE,
                PolicyArn=policy_arn,
            )
        _assert_role_configuration(
            iam,
            account_id=account_id,
            name=CONTROLLER_ROLE,
            trust=inert_trust,
            inline_policies={
                CONTROLLER_REVOKE_POLICY: controller_revoke_policy,
            },
            attached_policies=managed_policy_arns,
            max_session_duration=43200,
        )
        for name, document in sorted(controller_slices.items()):
            _assert_managed_policy(
                iam,
                account_id=account_id,
                name=name,
                document=document,
                expected_roles={CONTROLLER_ROLE},
            )
        _simulate_controller_policy(
            iam,
            account_id=account_id,
            region=args.region,
            controller_arn=controller_arn,
            runner_arn=runner_arn,
            session_cutoff=controller_session_cutoff,
            policy_documents=[
                *controller_slices.values(),
                controller_revoke_policy,
            ],
        )

        # Never reuse an assumable bootstrap role across runs.  Ownership and
        # inventory are checked before deletion, then the replacement starts
        # with an inert trust while its least-privilege policy is installed.
        _delete_static_bootstrap_role(iam, account_id=account_id)
        bootstrap_cleanup_required = True
        _ensure_role(
            iam,
            account_id=account_id,
            name=STATIC_BOOTSTRAP_ROLE,
            trust=inert_trust,
            expected_inline_policies=set(),
            expected_attached_policies=set(),
            max_session_duration=3600,
        )
        bootstrap_expires_at = (
            datetime.now(timezone.utc) + timedelta(minutes=15)
        ).isoformat(timespec="seconds").replace("+00:00", "Z")
        bootstrap_policy = _static_bootstrap_policy(
            account_id=account_id,
            region=args.region,
            readonly_secret_id=args.readonly_dsn_secret_id,
            miner_intake_secret_id=args.miner_intake_secret_id,
            expires_at=bootstrap_expires_at,
        )
        _put_policy(
            iam,
            role=STATIC_BOOTSTRAP_ROLE,
            name=STATIC_BOOTSTRAP_POLICY,
            document=bootstrap_policy,
        )
        _assert_role_configuration(
            iam,
            account_id=account_id,
            name=STATIC_BOOTSTRAP_ROLE,
            trust=inert_trust,
            inline_policies={STATIC_BOOTSTRAP_POLICY: bootstrap_policy},
            attached_policies=set(),
            max_session_duration=3600,
        )
        bootstrap_trust = _static_bootstrap_trust(
            account_id=account_id,
            validator_role_name=VALIDATOR_ROLE,
            expires_at=bootstrap_expires_at,
        )
        iam.update_assume_role_policy(
            RoleName=STATIC_BOOTSTRAP_ROLE,
            PolicyDocument=_json(bootstrap_trust),
        )
        _assert_role_configuration(
            iam,
            account_id=account_id,
            name=STATIC_BOOTSTRAP_ROLE,
            trust=bootstrap_trust,
            inline_policies={STATIC_BOOTSTRAP_POLICY: bootstrap_policy},
            attached_policies=set(),
            max_session_duration=3600,
        )

        # These are the only two activation writes.  The runner remains inert
        # until the controller lattice, simulator, and temporary installer
        # authority are all proven; controller OIDC trust is last.
        _wait_until_after_session_cutoffs(
            [runner_session_cutoff, controller_session_cutoff]
        )
        iam.update_assume_role_policy(
            RoleName=RUNNER_ROLE,
            PolicyDocument=_json(runner_trust),
        )
        _assert_role_configuration(
            iam,
            account_id=account_id,
            name=RUNNER_ROLE,
            trust=runner_trust,
            inline_policies={
                runner_policy_name: runner_policy,
                RUNNER_REVOKE_POLICY: runner_revoke_policy,
            },
            attached_policies={runner_managed_policy},
            max_session_duration=43200,
        )
        iam.update_assume_role_policy(
            RoleName=CONTROLLER_ROLE,
            PolicyDocument=_json(controller_trust),
        )
        _assert_role_configuration(
            iam,
            account_id=account_id,
            name=CONTROLLER_ROLE,
            trust=controller_trust,
            inline_policies={
                CONTROLLER_REVOKE_POLICY: controller_revoke_policy,
            },
            attached_policies=managed_policy_arns,
            max_session_duration=43200,
        )
    except Exception as exc:
        cleanup_errors: list[str] = []
        controller_cleanup_required = controller_ready or bool(
            controller_revocation
        )
        runner_cleanup_required = runner_ready or bool(runner_revocation)
        # Stop both new-session paths first and independently.  Only then mint
        # the future cutoff used to revoke every session that could have been
        # issued before the inert trusts propagated.
        if controller_cleanup_required:
            try:
                iam.update_assume_role_policy(
                    RoleName=CONTROLLER_ROLE,
                    PolicyDocument=_json(inert_trust),
                )
            except Exception as cleanup_exc:
                cleanup_errors.append(
                    f"controller-trust: {type(cleanup_exc).__name__}"
                )
        if runner_cleanup_required:
            try:
                iam.update_assume_role_policy(
                    RoleName=RUNNER_ROLE,
                    PolicyDocument=_json(inert_trust),
                )
            except Exception as cleanup_exc:
                cleanup_errors.append(
                    f"runner-trust: {type(cleanup_exc).__name__}"
                )
        failure_cutoff = _new_session_cutoff()
        failure_revoke_policy = _revoke_older_sessions_policy(
            cutoff=failure_cutoff
        )
        if controller_cleanup_required:
            try:
                _neutralize_controller_role(
                    iam,
                    account_id=account_id,
                    managed_policy_arns=managed_policy_arns,
                    legacy_inline_policy=legacy_controller_policy,
                    revoke_document=failure_revoke_policy,
                )
            except Exception as cleanup_exc:
                cleanup_errors.append(f"controller: {type(cleanup_exc).__name__}")
        if runner_cleanup_required:
            try:
                _put_policy(
                    iam,
                    role=RUNNER_ROLE,
                    name=RUNNER_REVOKE_POLICY,
                    document=failure_revoke_policy,
                )
                _assert_role_configuration(
                    iam,
                    account_id=account_id,
                    name=RUNNER_ROLE,
                    trust=inert_trust,
                    inline_policies={
                        runner_policy_name: runner_policy,
                        RUNNER_REVOKE_POLICY: failure_revoke_policy,
                    },
                    attached_policies={runner_managed_policy},
                    max_session_duration=43200,
                )
            except Exception as cleanup_exc:
                cleanup_errors.append(f"runner: {type(cleanup_exc).__name__}")
        if bootstrap_cleanup_required:
            try:
                _delete_static_bootstrap_role(iam, account_id=account_id)
            except Exception as cleanup_exc:
                cleanup_errors.append(f"bootstrap: {type(cleanup_exc).__name__}")
        if controller_cleanup_required or runner_cleanup_required:
            try:
                _wait_until_after_session_cutoffs([failure_cutoff])
            except Exception as cleanup_exc:
                cleanup_errors.append(f"session-cutoff: {type(cleanup_exc).__name__}")
        if cleanup_errors:
            raise SetupError(
                "IAM setup failed and fail-closed cleanup differed: "
                + ", ".join(cleanup_errors)
            ) from exc
        raise
    return {
        "status": "iam_ready",
        "repository": args.repository,
        "account_id": account_id,
        "controller_role_arn": controller_arn,
        "runner_role_arn": runner_arn,
        "runner_instance_profile": RUNNER_PROFILE,
        "static_bootstrap_role_arn": (
            f"arn:aws:iam::{account_id}:role/{STATIC_BOOTSTRAP_ROLE}"
        ),
        "static_bootstrap_trusted_role_arn": expected_validator_arn,
        "static_bootstrap_installer_requested_assume_seconds": 900,
        "static_bootstrap_trust_expires_at": bootstrap_expires_at,
        "readonly_secret_id": args.readonly_dsn_secret_id,
        "miner_intake_secret_id": args.miner_intake_secret_id,
        "github_variables_mutated": False,
        "secret_values_printed": False,
    }


def cleanup_bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    _validate_identity_inputs(args)
    _sts, iam, account_id = _iam_clients(args.region)
    _delete_static_bootstrap_role(iam, account_id=account_id)
    return {
        "status": "static_bootstrap_authority_removed",
        "account_id": account_id,
        "role": STATIC_BOOTSTRAP_ROLE,
    }


def _read_bootstrap_receipt(descriptor: int) -> dict[str, Any]:
    _require_anonymous_fd(descriptor, description="bootstrap receipt")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, 65536)
        if not chunk:
            break
        total += len(chunk)
        if total > 65536:
            raise SetupError("bootstrap receipt is too large")
        chunks.append(chunk)
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SetupError("bootstrap receipt is invalid") from exc
    if not isinstance(value, dict):
        raise SetupError("bootstrap receipt is invalid")
    return value


def _validate_bootstrap_receipt(
    args: argparse.Namespace, receipt: Mapping[str, Any]
) -> None:
    expected_instance_prefix = (
        f"arn:aws:sts::{EXPECTED_ACCOUNT_ID}:assumed-role/"
        f"{VALIDATOR_ROLE}/"
    )
    expected_bootstrap_prefix = (
        f"arn:aws:sts::{EXPECTED_ACCOUNT_ID}:assumed-role/"
        f"{STATIC_BOOTSTRAP_ROLE}/"
    )
    if (
        receipt.get("schema_version")
        != "leadpoet.production_parity_static_bootstrap.v1"
        or receipt.get("status") != "installed"
        or receipt.get("commit") != args.commit
        or receipt.get("migration")
        != "scripts/156-production-parity-readonly-role.sql"
        or receipt.get("migration_sha256") != args.migration_sha256
        or receipt.get("account_id") != EXPECTED_ACCOUNT_ID
        or not str(receipt.get("instance_role") or "").startswith(
            expected_instance_prefix
        )
        or not str(receipt.get("bootstrap_role") or "").startswith(
            expected_bootstrap_prefix
        )
        or receipt.get("readonly_secret_id") != args.readonly_dsn_secret_id
        or receipt.get("miner_intake_secret_id")
        != args.miner_intake_secret_id
        or receipt.get("reader_role") != "leadpoet_parity_reader"
        or receipt.get("reader_default_read_only_verified") is not True
        or receipt.get("secret_values_printed") is not False
    ):
        raise SetupError("bootstrap receipt does not match configuration inputs")


def configure_repository(args: argparse.Namespace) -> dict[str, Any]:
    _validate_identity_inputs(args)
    enable = args.enabled == "true"
    controller_arn = (
        f"arn:aws:iam::{EXPECTED_ACCOUNT_ID}:role/{CONTROLLER_ROLE}"
    )
    _gh_variable(args.repository, "LEADPOET_PARITY_ENABLED", "false")
    if _gh_variable_value(args.repository, "LEADPOET_PARITY_ENABLED") != "false":
        raise SetupError("GitHub parity disable readback differs")
    try:
        postgres, postgrest = _validate_gateway_and_images(args)
        if enable:
            if (
                not re.fullmatch(r"[0-9a-f]{40}", str(args.commit or ""))
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(args.migration_sha256 or "")
                )
            ):
                raise SetupError(
                    "enabled configuration requires exact bootstrap identity"
                )
            receipt = _read_bootstrap_receipt(args.receipt_fd)
            _validate_bootstrap_receipt(args, receipt)
        variables = {
            "LEADPOET_PARITY_AWS_ROLE_ARN": controller_arn,
            "LEADPOET_PARITY_AWS_REGION": args.region,
            "LEADPOET_PARITY_PRODUCTION_GATEWAY_IP": args.production_gateway_ip,
            "LEADPOET_PARITY_PRODUCTION_GATEWAY_URL": (
                args.production_gateway_url.rstrip("/")
            ),
            "LEADPOET_PARITY_PRODUCTION_GATEWAY_SECRET_ID": (
                args.production_gateway_secret_id
            ),
            "LEADPOET_PARITY_READONLY_DSN_SECRET_ID": args.readonly_dsn_secret_id,
            "LEADPOET_PARITY_MINER_INTAKE_SECRET_ID": args.miner_intake_secret_id,
            "LEADPOET_PARITY_RUNNER_INSTANCE_PROFILE": RUNNER_PROFILE,
            "LEADPOET_PARITY_POSTGRES_IMAGE": postgres,
            "LEADPOET_PARITY_POSTGREST_IMAGE": postgrest,
            "LEADPOET_PARITY_VOLUME_GIB": str(args.volume_gib),
        }
        for name, value in variables.items():
            _gh_variable(args.repository, name, value)
            if _gh_variable_value(args.repository, name) != value:
                raise SetupError(f"GitHub variable {name} readback differs")
        _delete_gh_variable(args.repository, "LEADPOET_PARITY_INSTANCE_TYPE")
        expected_parity_names = set(variables) | {"LEADPOET_PARITY_ENABLED"}
        actual_parity_names = {
            name for name in _gh_variable_names(args.repository)
            if name.startswith("LEADPOET_PARITY_")
        }
        if actual_parity_names != expected_parity_names:
            raise SetupError("GitHub parity variable inventory differs")
        if enable:
            _gh_variable(args.repository, "LEADPOET_PARITY_ENABLED", "true")
            if (
                _gh_variable_value(args.repository, "LEADPOET_PARITY_ENABLED")
                != "true"
            ):
                raise SetupError("GitHub parity enable readback differs")
    except Exception:
        _gh_variable(args.repository, "LEADPOET_PARITY_ENABLED", "false")
        if (
            _gh_variable_value(args.repository, "LEADPOET_PARITY_ENABLED")
            != "false"
        ):
            raise SetupError("GitHub parity failure-disable readback differs")
        raise
    return {
        "status": "configured",
        "repository": args.repository,
        "account_id": EXPECTED_ACCOUNT_ID,
        "controller_role_arn": controller_arn,
        "enabled": enable,
        "volume_gib": args.volume_gib,
        "all_variables_read_back": True,
        "secret_values_printed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_identity_inputs(command: Any) -> None:
        command.add_argument("--repository", default=DEFAULT_REPOSITORY)
        command.add_argument("--region", default=EXPECTED_REGION)
        command.add_argument("--production-gateway-ip", default="52.91.135.79")
        command.add_argument(
            "--production-gateway-secret-id", default="leadpoet/prod/gateway/env"
        )
        command.add_argument(
            "--readonly-dsn-secret-id",
            default="leadpoet/staging/production-parity/readonly-dsn",
        )
        command.add_argument(
            "--miner-intake-secret-id",
            default=DEFAULT_MINER_INTAKE_SECRET_ID,
        )
        command.add_argument("--volume-gib", type=int, default=DEFAULT_VOLUME_GIB)

    iam_only = subparsers.add_parser("iam-only")
    add_identity_inputs(iam_only)

    cleanup = subparsers.add_parser("cleanup-bootstrap")
    add_identity_inputs(cleanup)

    configure = subparsers.add_parser("configure")
    add_identity_inputs(configure)
    configure.add_argument("--production-gateway-url", required=True)
    configure.add_argument("--enabled", choices=("false", "true"), required=True)
    configure.add_argument("--commit")
    configure.add_argument("--migration-sha256")
    configure.add_argument("--receipt-fd", type=int, default=-1)

    args = parser.parse_args(argv)
    try:
        if args.command == "iam-only":
            result = setup_iam_only(args)
        elif args.command == "cleanup-bootstrap":
            result = cleanup_bootstrap(args)
        else:
            result = configure_repository(args)
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
