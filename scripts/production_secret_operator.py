"""Named temporary operator authority for the two production env secrets."""

from botocore.config import Config

from scripts.gateway_iam_session import (
    ADMIN_PROFILE,
    AWS_CLIENT_CONFIG,
    DEFAULT_PROFILE,
    REGION,
    gateway_iam_session,
    validate_admin_iam_identity,
    validate_gateway_iam_identity,
)


class SecretOperatorError(RuntimeError):
    """A redacted authority failure."""


def secret_operator_client(*, profile=DEFAULT_PROFILE, timeout_seconds=30):
    if not isinstance(profile, str) or not profile:
        raise SecretOperatorError("named temporary operator profile is required")
    try:
        session = gateway_iam_session(profile=profile)
        credentials = session.get_credentials()
        method = getattr(credentials, "method", "")
        if method in {"iam-role", "container-role"}:
            raise SecretOperatorError("instance authority cannot configure secrets")
        cached_admin = profile == ADMIN_PROFILE and method == "explicit"
        if not cached_admin and method not in {"sso", "assume-role", "assume-role-with-web-identity"}:
            raise SecretOperatorError("temporary operator authority is required")
        identity = session.client("sts", region_name=REGION, config=AWS_CLIENT_CONFIG).get_caller_identity()
        (validate_admin_iam_identity if cached_admin else validate_gateway_iam_identity)(identity)
        config = AWS_CLIENT_CONFIG.merge(Config(
            connect_timeout=max(5, min(float(timeout_seconds), 45)),
            read_timeout=max(5, min(float(timeout_seconds), 45)),
        ))
        client = session.client("secretsmanager", region_name=REGION, config=config)
        return client, str(identity["Account"])
    except SecretOperatorError:
        raise
    except Exception:
        raise SecretOperatorError("secret operator authority is unavailable") from None
