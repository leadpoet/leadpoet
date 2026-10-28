"""Resolve short-lived production operator authority without gateway secrets."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import re
import subprocess
from typing import Any, Mapping

import boto3
from botocore.config import Config


ACCOUNT = "493765492819"
REGION = "us-east-1"
DEFAULT_PROFILE = "leadpoet-lab"
RECOVERY_PROFILE = "leadpoet-recovery-console-20261026"
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


def validate_gateway_iam_identity(identity: Mapping[str, object], *, allow_recovery: bool = False) -> None:
    arn = str(identity.get("Arn") or "")
    role = re.fullmatch(rf"arn:aws:sts::{ACCOUNT}:assumed-role/[^/]+/[^/]+", arn)
    recovery = allow_recovery and arn == f"arn:aws:iam::{ACCOUNT}:root"
    if str(identity.get("Account") or "") != ACCOUNT or not (role or recovery):
        raise GatewayIamSessionError("gateway IAM caller identity differs")


def _recovery_session(region: str):
    # AWS CLI supports login_session before the deployed boto3 version does.
    # Capture its official export in memory; never use shell exports or a file.
    result = subprocess.run(
        ["aws", "configure", "export-credentials", "--profile", RECOVERY_PROFILE, "--format", "process"],
        capture_output=True, text=True, check=False, timeout=30,
        env={**os.environ, "AWS_REGION": region, "AWS_DEFAULT_REGION": region,
             "AWS_IGNORE_CONFIGURED_ENDPOINT_URLS": "true"},
    )
    if result.returncode != 0:
        raise GatewayIamSessionError("temporary recovery authority is unavailable")
    document = json.loads(result.stdout)
    if not isinstance(document, dict):
        raise GatewayIamSessionError("temporary recovery authority is invalid")
    expiration = datetime.fromisoformat(str(document.get("Expiration") or "").replace("Z", "+00:00"))
    access = document.get("AccessKeyId")
    secret = document.get("SecretAccessKey")
    token = document.get("SessionToken")
    if (document.get("Version") != 1 or not isinstance(access, str)
            or re.fullmatch(r"ASIA[A-Z0-9]{16}", access) is None
            or not isinstance(secret, str) or not secret or any(c.isspace() for c in secret)
            or not isinstance(token, str) or not token or any(c.isspace() for c in token)
            or expiration.tzinfo is None or expiration <= datetime.now(timezone.utc)):
        raise GatewayIamSessionError("temporary recovery authority is invalid")
    return boto3.session.Session(aws_access_key_id=access, aws_secret_access_key=secret,
                                 aws_session_token=token, region_name=region)


def gateway_iam_session(*, profile: str | None = DEFAULT_PROFILE, region: str = REGION) -> Any:
    """Use a named temporary profile, or the IAM role chain with profile=None.

    Ambient authority overrides are rejected. The explicit recovery profile uses
    AWS CLI's temporary export because deployed boto3 cannot read login_session.
    """
    if (region != REGION or any(os.environ.get(name) for name in FORBIDDEN)
            or (profile is not None and (not profile or re.fullmatch(r"[A-Za-z0-9_+=,.@-]+", profile) is None))):
        raise GatewayIamSessionError("gateway IAM session authority is invalid")
    try:
        recovery = profile == RECOVERY_PROFILE
        if recovery:
            session = _recovery_session(region)
        else:
            session = boto3.session.Session(profile_name=profile, region_name=region)
            # SSO/assume-role refreshes create their own clients before the final
            # STS check. Apply endpoint safety to those provider clients too.
            session._session.set_default_client_config(AWS_CLIENT_CONFIG)
            credentials = session.get_credentials()
            if credentials is None or credentials.method not in ({"iam-role"} if profile is None else _TEMPORARY_METHODS):
                raise GatewayIamSessionError("gateway IAM requires temporary role credentials")
            frozen = credentials.get_frozen_credentials()
            if not frozen.token or not frozen.access_key.startswith("ASIA"):
                raise GatewayIamSessionError("gateway IAM requires temporary role credentials")
        identity = session.client("sts", region_name=region, config=AWS_CLIENT_CONFIG).get_caller_identity()
        validate_gateway_iam_identity(identity, allow_recovery=recovery)
        return session
    except GatewayIamSessionError:
        raise
    except Exception:
        raise GatewayIamSessionError("gateway IAM authority is unavailable") from None
