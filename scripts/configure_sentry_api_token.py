#!/usr/bin/env python3
"""Store one hidden-input Sentry API token in both production env secrets.

This is an operator tool. It sends the token to each host over SSH stdin and
never places it in argv, output, shell history, or a local file. Secrets
Manager retains the prior version for rollback. The gateway dotenv and
validator JSON document formats are preserved.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


TOKEN_ENV_NAME = "LEADPOET_SENTRY_API_TOKEN"
SSH_KEY_ENV = "LEADPOET_SENTRY_SSH_KEY"
DEFAULT_SSH_KEY = Path.home() / "Downloads" / "leadpoet-2026-07-28.pem"
TOKEN_RE = re.compile(r"^sntryu_[A-Za-z0-9_-]{32,}$")
MAX_TIMEOUT_SECONDS = 45.0
TARGETS = {
    "gateway": ("ec2-user@52.91.135.79", "leadpoet/prod/gateway/env"),
    "validator": ("ec2-user@100.59.201.156", "leadpoet/prod/validator/env"),
}


class ConfigurationError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


_REMOTE_UPDATER = r'''
import hmac
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile

NAME = "LEADPOET_SENTRY_API_TOKEN"
TOKEN_RE = re.compile(r"^sntryu_[A-Za-z0-9_-]{32,}$")


def aws_get(secret_id):
    result = subprocess.run(
        [
            "aws", "secretsmanager", "get-secret-value",
            "--secret-id", secret_id,
            "--output", "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(31)
    try:
        document = json.loads(result.stdout)
        return str(document["VersionId"]), str(document["SecretString"])
    except Exception:
        raise SystemExit(32)


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


secret_id = sys.argv[1]
token = sys.stdin.read().strip()
if not TOKEN_RE.fullmatch(token):
    raise SystemExit(33)

version_id, current = aws_get(secret_id)
updated, document_format = update_document(current, token)

# Re-read immediately before the write. Secrets Manager has no compare-and-swap
# update API, so this closes the ordinary concurrent-operator window while the
# previous version remains available as AWSPREVIOUS for rollback.
latest_version_id, _ = aws_get(secret_id)
if latest_version_id != version_id:
    raise SystemExit(34)

path = ""
try:
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="leadpoet-sentry-token-",
        delete=False,
    )
    path = handle.name
    os.chmod(path, 0o600)
    with handle:
        handle.write(updated)
    result = subprocess.run(
        [
            "aws", "secretsmanager", "update-secret",
            "--secret-id", secret_id,
            "--secret-string", "file://" + path,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(35)
finally:
    if path:
        try:
            os.remove(path)
        except OSError:
            pass

_, readback = aws_get(secret_id)
if not hmac.compare_digest(parsed_value(readback), token):
    raise SystemExit(36)
print(json.dumps({"updated": True, "format": document_format}, separators=(",", ":")))
'''


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


def _update_target(
    name: str,
    token: str,
    *,
    ssh_key: Path,
    timeout: float,
) -> str:
    host, secret_id = TARGETS[name]
    remote_command = "python3 -c %s %s" % (
        shlex.quote(_REMOTE_UPDATER),
        shlex.quote(secret_id),
    )
    command = [
        "ssh",
        "-i",
        str(ssh_key),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        host,
        remote_command,
    ]
    try:
        result = subprocess.run(
            command,
            input=token,
            capture_output=True,
            text=True,
            check=False,
            timeout=_bounded_timeout(timeout),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigurationError(
            "secret_update_unavailable",
            "%s:%s" % (name, type(exc).__name__),
        ) from None
    if result.returncode != 0:
        raise ConfigurationError(
            "secret_update_failed",
            "%s:ssh_status=%d" % (name, result.returncode),
        )
    try:
        response = json.loads(result.stdout)
    except (TypeError, ValueError):
        raise ConfigurationError("secret_update_readback_invalid", name) from None
    if not isinstance(response, dict) or response.get("updated") is not True:
        raise ConfigurationError("secret_update_readback_invalid", name)
    document_format = str(response.get("format") or "unknown")
    if document_format not in {"dotenv", "json"}:
        raise ConfigurationError("secret_update_readback_invalid", name)
    return document_format


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssh-key",
        type=Path,
        default=Path(os.getenv(SSH_KEY_ENV) or DEFAULT_SSH_KEY),
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        token = _read_token()
        results = {}
        for name in ("gateway", "validator"):
            results[name] = _update_target(
                name,
                token,
                ssh_key=args.ssh_key,
                timeout=args.timeout_seconds,
            )
        print(
            "sentry_api_token_configured "
            + " ".join(
                "%s_format=%s" % (name, results[name])
                for name in ("gateway", "validator")
            )
        )
        return 0
    except ConfigurationError as exc:
        detail = " detail=%s" % exc.detail if exc.detail else ""
        print(
            "sentry_api_token_configuration_failed code=%s%s" % (exc.code, detail),
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # never render token-bearing exception state
        print(
            "sentry_api_token_configuration_failed code=unexpected type=%s"
            % type(exc).__name__,
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
