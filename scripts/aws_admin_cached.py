"""Run an ordinary AWS CLI API call with the fixed, validated Admin cache entry.

Run as ``python3 -m scripts.aws_admin_cached SERVICE OPERATION [OPTIONS]``.
This launcher never invokes SSO or another credential provider.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

from scripts.gateway_iam_session import (
    ADMIN_PROFILE, GatewayIamSessionError, REGION, gateway_iam_session,
)


_BLOCKED_OPTIONS = (
    "--profile", "--endpoint-url", "--debug", "--no-verify-ssl",
    "--ca-bundle", "--no-sign-request", "--region", "--cli-auto-prompt",
)
_STS_READ_ONLY = frozenset({"get-caller-identity", "get-access-key-info", "decode-authorization-message"})
_IAM_ISSUANCE = frozenset({
    "create-access-key", "create-service-specific-credential", "create-login-profile",
    "reset-service-specific-credential",
})


def _validate_argv(argv: list[str]) -> None:
    if (len(argv) < 2 or any(re.fullmatch(r"[a-z0-9][a-z0-9-]*", arg) is None for arg in argv[:2])):
        raise GatewayIamSessionError("cached Admin CLI arguments are invalid")
    service, operation = argv[:2]
    if (service in {"sso", "sso-oidc", "configure"}
            or service == "sts" and operation not in _STS_READ_ONLY
            or service == "iam" and operation in _IAM_ISSUANCE):
        raise GatewayIamSessionError("cached Admin CLI operation is invalid")
    for arg in argv[2:]:
        if arg.startswith("--"):
            option = arg.split("=", 1)[0]
            # AWS CLI accepts unambiguous global option abbreviations.
            if any(blocked.startswith(option) or option == blocked for blocked in _BLOCKED_OPTIONS):
                raise GatewayIamSessionError("cached Admin CLI option is invalid")


def run_cached_admin_aws(argv: list[str]) -> int:
    _validate_argv(argv)
    session = gateway_iam_session(profile=ADMIN_PROFILE)
    frozen = session.get_credentials().get_frozen_credentials()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("AWS_ENDPOINT_URL") and key != "AWS_CA_BUNDLE"}
    env.update({
        "AWS_ACCESS_KEY_ID": frozen.access_key,
        "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
        "AWS_SESSION_TOKEN": frozen.token,
        "AWS_DEFAULT_REGION": REGION,
        "AWS_REGION": REGION,
        "AWS_CONFIG_FILE": os.devnull,
        "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
        "AWS_IGNORE_CONFIGURED_ENDPOINT_URLS": "true",
        "AWS_PAGER": "",
        "AWS_CLI_AUTO_PROMPT": "off",
    })
    env.pop("AWS_PROFILE", None)
    env.pop("AWS_DEFAULT_PROFILE", None)
    return subprocess.run(["aws", *argv], env=env, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    try:
        return run_cached_admin_aws(list(sys.argv[1:] if argv is None else argv))
    except Exception:
        print("cached Admin CLI authority is unavailable", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
