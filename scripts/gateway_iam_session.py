"""Resolve short-lived production operator authority without gateway secrets."""

from __future__ import annotations

import os
import re
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import boto3
from botocore.config import Config
from botocore.configloader import load_config


ACCOUNT = "493765492819"
REGION = "us-east-1"
DEFAULT_PROFILE = "leadpoet-lab"
ADMIN_PROFILE = "leadpoet-admin"
ADMIN_ROLE = "Admin"
ADMIN_SESSION = "pranav-admin"
ADMIN_STS_ROLE = re.compile(
    rf"arn:aws:sts::{ACCOUNT}:assumed-role/AWSReservedSSO_Admin_[0-9a-f]{{16}}/{ADMIN_SESSION}"
)
RETIRED_ROOT_PROFILE = "leadpoet-recovery-console-20261026"
AWS_CLIENT_CONFIG = Config(ignore_configured_endpoint_urls=True)
FORBIDDEN = frozenset({
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE",
    "AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN", "AWS_ROLE_SESSION_NAME",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT", "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
    "BOTO_CONFIG",
})
_TEMPORARY_METHODS = frozenset({"sso", "iam-role", "assume-role", "assume-role-with-web-identity", "container-role"})


class GatewayIamSessionError(RuntimeError):
    """A redacted failure that never contains credentials or SDK responses."""


def validate_gateway_iam_identity(identity: Mapping[str, object]) -> None:
    arn = str(identity.get("Arn") or "")
    role = re.fullmatch(rf"arn:aws:sts::{ACCOUNT}:assumed-role/[^/]+/[^/]+", arn)
    if str(identity.get("Account") or "") != ACCOUNT or not role:
        raise GatewayIamSessionError("gateway IAM caller identity differs")


def validate_admin_iam_identity(identity: Mapping[str, object]) -> None:
    validate_gateway_iam_identity(identity)
    if ADMIN_STS_ROLE.fullmatch(str(identity.get("Arn") or "")) is None:
        raise GatewayIamSessionError("gateway IAM caller identity differs")


def _cached_admin_credentials() -> dict[str, str]:
    """Read only the CLI cache entry bound to the configured Admin SSO role."""
    aws_dir = Path.home() / ".aws"
    config = load_config(str(aws_dir / "config"))
    profile = config.get("profiles", {}).get(ADMIN_PROFILE, {})
    session_name = profile.get("sso_session")
    sso_session = config.get("sso_sessions", {}).get(session_name, {})
    if (profile.get("sso_account_id") != ACCOUNT
            or profile.get("sso_role_name") != ADMIN_ROLE
            or session_name != ADMIN_PROFILE
            or not sso_session.get("sso_start_url", "").startswith("https://")
            or not sso_session.get("sso_region")):
        raise GatewayIamSessionError("gateway IAM Admin profile authority is invalid")
    # This is botocore SSOCredentialFetcher._create_cache_key's exact format.
    args = {"accountId": ACCOUNT, "roleName": ADMIN_ROLE, "sessionName": session_name}
    key = hashlib.sha1(json.dumps(args, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cached = json.loads((aws_dir / "cli" / "cache" / f"{key}.json").read_text())
    if cached.get("ProviderType") != "sso":
        raise GatewayIamSessionError("gateway IAM Admin cache is unavailable")
    role = cached["Credentials"]
    expiration = datetime.fromisoformat(role["Expiration"].replace("Z", "+00:00"))
    access_key = role["AccessKeyId"]
    secret_key = role["SecretAccessKey"]
    token = role["SessionToken"]
    if (role.get("AccountId") != ACCOUNT
            or expiration.tzinfo is None
            or expiration <= datetime.now(timezone.utc) + timedelta(seconds=60)
            or not isinstance(access_key, str) or not access_key.startswith("ASIA")
            or not isinstance(secret_key, str) or not secret_key
            or not isinstance(token, str) or not token):
        raise GatewayIamSessionError("gateway IAM Admin cache is unavailable")
    return {"aws_access_key_id": access_key, "aws_secret_access_key": secret_key,
            "aws_session_token": token}


def gateway_iam_session(*, profile: str | None = DEFAULT_PROFILE, region: str = REGION) -> Any:
    """Use a named temporary profile, or the IAM role chain with profile=None.

    Ambient authority overrides and the retired root profile are rejected.
    """
    if (profile == RETIRED_ROOT_PROFILE or region != REGION
            or any(os.environ.get(name) for name in FORBIDDEN)
            or (profile is not None and (not profile or re.fullmatch(r"[A-Za-z0-9_+=,.@-]+", profile) is None))):
        raise GatewayIamSessionError("gateway IAM session authority is invalid")
    try:
        if profile == ADMIN_PROFILE:
            # Explicit credentials bypass SSO and all refreshable providers.
            session = boto3.session.Session(region_name=region, **_cached_admin_credentials())
        else:
            session = boto3.session.Session(profile_name=profile, region_name=region)
        # SSO/assume-role refreshes create their own clients before the final
        # STS check. Apply endpoint safety to those provider clients too.
        session._session.set_default_client_config(AWS_CLIENT_CONFIG)
        credentials = session.get_credentials()
        accepted_methods = ({"explicit"} if profile == ADMIN_PROFILE else
                            {"iam-role"} if profile is None else _TEMPORARY_METHODS)
        if credentials is None or credentials.method not in accepted_methods:
            raise GatewayIamSessionError("gateway IAM requires temporary role credentials")
        frozen = credentials.get_frozen_credentials()
        if not frozen.token or not frozen.access_key.startswith("ASIA"):
            raise GatewayIamSessionError("gateway IAM requires temporary role credentials")
        identity = session.client("sts", region_name=region, config=AWS_CLIENT_CONFIG).get_caller_identity()
        (validate_admin_iam_identity if profile == ADMIN_PROFILE else validate_gateway_iam_identity)(identity)
        return session
    except GatewayIamSessionError:
        raise
    except Exception:
        raise GatewayIamSessionError("gateway IAM authority is unavailable") from None
