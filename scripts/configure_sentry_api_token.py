#!/usr/bin/env python3
"""Store one hidden-input Sentry API token using local temporary operator authority.

Secret documents and the token stay in memory. No secret or AWS credential is
sent through SSH, argv, shell exports or a temporary file. Existing version,
format and AWSCURRENT checks remain in place.
"""

from __future__ import annotations

import argparse
import getpass
import hmac
import json
import re
import shlex
import sys
import uuid
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOKEN_ENV_NAME = "LEADPOET_SENTRY_API_TOKEN"
NAME = TOKEN_ENV_NAME
TOKEN_RE = re.compile(r"^sntryu_[A-Za-z0-9_-]{32,}$")
MAX_TIMEOUT_SECONDS = 45.0
TARGETS = {
    "gateway": "leadpoet/prod/gateway/env",
    "validator": "leadpoet/prod/validator/env",
}


class ConfigurationError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


def dotenv_key(line):
    candidate = line.strip()
    if not candidate or candidate.startswith("#"):
        return ""
    if candidate.startswith("export "):
        candidate = candidate[7:].strip()
    try:
        parts = shlex.split(candidate, posix=True)
    except ValueError:
        parts = [candidate]
    assignment = parts[0] if len(parts) == 1 else candidate
    if "=" not in assignment:
        return ""
    key = assignment.split("=", 1)[0].strip()
    return key if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) else ""


def update_document(raw, token):
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        parsed[NAME] = token
        return json.dumps(parsed, separators=(",", ":")), "json"

    lines = []
    inserted = False
    for source_line in raw.replace("\x00", "\n").splitlines():
        if dotenv_key(source_line) == NAME:
            if not inserted:
                lines.append("export %s=%s" % (NAME, shlex.quote(token)))
                inserted = True
            continue
        lines.append(source_line)
    if not inserted:
        lines.append("export %s=%s" % (NAME, shlex.quote(token)))
    return "\n".join(lines) + "\n", "dotenv"


def parsed_value(raw):
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return str(parsed.get(NAME) or "")
    for source_line in raw.replace("\x00", "\n").splitlines():
        if dotenv_key(source_line) != NAME:
            continue
        candidate = source_line.strip()
        if candidate.startswith("export "):
            candidate = candidate[7:].strip()
        parts = shlex.split(candidate, posix=True)
        if len(parts) == 1 and "=" in parts[0]:
            return parts[0].split("=", 1)[1]
        return candidate.split("=", 1)[1]
    return ""


def _write_failure_code(error):
    try:
        code = error.response["Error"]["Code"]
    except Exception:
        code = ""
    if code in {"AccessDenied", "AccessDeniedException", "UnauthorizedException"}:
        return "secret_writer_access_denied"
    if code == "ResourceNotFoundException":
        return "secret_not_found"
    if code in {
        "DecryptionFailure", "EncryptionFailure", "InvalidParameterException",
        "InvalidRequestException", "LimitExceededException",
        "MalformedPolicyDocumentException", "PreconditionNotMetException",
    }:
        return "secret_write_rejected"
    return "secret_write_failed"


def _update_secret(client, secret_id, token, *, apply=True):
    if secret_id not in TARGETS.values():
        raise ConfigurationError("secret_scope_invalid")
    if not TOKEN_RE.fullmatch(token):
        raise ConfigurationError("token_format_invalid")

    def get(version_id=""):
        request = {"SecretId": secret_id}
        if version_id:
            request["VersionId"] = version_id
        try:
            document = client.get_secret_value(**request)
        except Exception:
            raise ConfigurationError("secret_read_unavailable") from None
        try:
            return str(document["VersionId"]), str(document["SecretString"])
        except Exception:
            raise ConfigurationError("secret_document_invalid") from None

    version_id, current = get()
    updated, document_format = update_document(current, token)
    latest_version_id, _ = get()
    if latest_version_id != version_id:
        raise ConfigurationError("secret_changed_concurrently")
    if not apply:
        return document_format
    candidate_version_id = str(uuid.uuid4())
    try:
        client.put_secret_value(
            SecretId=secret_id, ClientRequestToken=candidate_version_id,
            SecretString=updated,
        )
    except Exception as exc:
        raise ConfigurationError(_write_failure_code(exc)) from None
    readback_version_id, readback = get(candidate_version_id)
    if (readback_version_id != candidate_version_id
            or not hmac.compare_digest(parsed_value(readback), token)):
        raise ConfigurationError("secret_write_readback_mismatch")
    try:
        stages = client.describe_secret(SecretId=secret_id)["VersionIdsToStages"]
    except Exception:
        raise ConfigurationError("secret_read_unavailable") from None
    if not isinstance(stages, dict):
        raise ConfigurationError("secret_document_invalid")
    if "AWSCURRENT" not in stages.get(candidate_version_id, []):
        raise ConfigurationError("secret_write_superseded")
    return document_format


def _bounded_timeout(value: float) -> float:
    return max(5.0, min(float(value), MAX_TIMEOUT_SECONDS))


def _read_token() -> str:
    if sys.stdin.isatty():
        token = getpass.getpass("Sentry API token (input hidden): ")
    else:
        token = sys.stdin.read()
    token = token.strip()
    if not TOKEN_RE.fullmatch(token):
        raise ConfigurationError("token_format_invalid")
    return token


def _update_target(name, token, *, profile="leadpoet-lab", timeout=30, apply=True):
    try:
        from scripts.production_secret_operator import secret_operator_client
        client, _account = secret_operator_client(
            profile=profile, timeout_seconds=_bounded_timeout(timeout)
        )
        return _update_secret(client, TARGETS[name], token, apply=apply)
    except ConfigurationError:
        raise
    except Exception:
        raise ConfigurationError("secret_update_unavailable", name) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aws-profile", default="leadpoet-lab",
                        help="named temporary operator profile")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--check", action="store_true", help="validate both documents without a write")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        token = _read_token()
        results = {}
        # Both read-only checks must pass before the first possible write.
        for name in ("gateway", "validator"):
            results[name] = _update_target(
                name, token, profile=args.aws_profile,
                timeout=args.timeout_seconds, apply=False,
            )
        if not args.check:
            applied = 0
            for name in ("gateway", "validator"):
                try:
                    results[name] = _update_target(
                        name, token, profile=args.aws_profile,
                        timeout=args.timeout_seconds, apply=True,
                    )
                    applied += 1
                except ConfigurationError as exc:
                    raise ConfigurationError(
                        exc.code, f"{name}:applied_targets={applied}; inspect version stages before retry"
                    ) from None
        print(
            ("sentry_api_token_checked " if args.check else "sentry_api_token_configured ")
            + " ".join("%s_format=%s" % (name, results[name]) for name in ("gateway", "validator"))
        )
        return 0
    except ConfigurationError as exc:
        detail = " detail=%s" % exc.detail if exc.detail else ""
        print("sentry_api_token_configuration_failed code=%s%s" % (exc.code, detail), file=sys.stderr)
        return 1
    except Exception as exc:
        print("sentry_api_token_configuration_failed code=unexpected type=%s"
              % type(exc).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
