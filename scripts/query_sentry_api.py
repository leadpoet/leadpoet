#!/usr/bin/env python3
"""Read bounded, redacted Leadpoet events from the Sentry API.

The API token is loaded from a production Secrets Manager document over
read-only SSH and remains inside this process. It is never accepted as a CLI
argument, printed, persisted, or forwarded to a Leadpoet runtime. The helper
uses only the Python standard library and has no relationship to Sentry SDK
initialization or any V2 authority path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from leadpoet_observability.sentry_scrubbing import REDACTED, scrub_text


API_BASE_URL = "https://sentry.io/api/0"
API_TOKEN_ENV = "LEADPOET_SENTRY_API_TOKEN"
DSN_ENV = "LEADPOET_SENTRY_DSN"
ORGANIZATION_ENV = "LEADPOET_SENTRY_ORGANIZATION"
PROJECT_ENV = "LEADPOET_SENTRY_PROJECT"
SSH_KEY_ENV = "LEADPOET_SENTRY_SSH_KEY"
DEFAULT_SSH_KEY = Path.home() / "Downloads" / "leadpoet-2026-07-28.pem"
MAX_RESPONSE_BYTES = 1_048_576
MAX_ITEMS = 100
MAX_TIMEOUT_SECONDS = 30.0
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SENTRY_DSN_ORG_RE = re.compile(r"^o(?P<organization>[0-9]+)(?:\.|$)")


@dataclass(frozen=True)
class SecretSource:
    host: str
    secret_id: str


@dataclass(frozen=True)
class Credentials:
    token: str
    organization: str
    project: str


SECRET_SOURCES: Mapping[str, SecretSource] = {
    "gateway": SecretSource(
        host="ec2-user@52.91.135.79",
        secret_id="leadpoet/prod/gateway/env",
    ),
    "validator": SecretSource(
        host="ec2-user@100.59.201.156",
        secret_id="leadpoet/prod/validator/env",
    ),
}


class SentryQueryError(RuntimeError):
    """Sanitized failure safe to print to an operator."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(scrub_text(detail, 200)) if detail else ""
        super().__init__(self.code)


_REMOTE_SECRET_READER = r'''
import json
import re
import shlex
import subprocess
import sys

WANTED = (
    "LEADPOET_SENTRY_API_TOKEN",
    "LEADPOET_SENTRY_DSN",
    "LEADPOET_SENTRY_ORGANIZATION",
    "LEADPOET_SENTRY_PROJECT",
)


def parse_secret(raw):
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return {str(key): "" if value is None else str(value) for key, value in parsed.items()}

    values = {}
    for source_line in raw.replace("\x00", "\n").splitlines():
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            parts = [line]
        candidate = parts[0] if len(parts) == 1 else line
        if "=" not in candidate:
            continue
        key, value = candidate.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            values[key] = value
    return values


result = subprocess.run(
    [
        "aws",
        "secretsmanager",
        "get-secret-value",
        "--secret-id",
        sys.argv[1],
        "--query",
        "SecretString",
        "--output",
        "text",
    ],
    check=False,
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    raise SystemExit(21)
values = parse_secret(result.stdout)
print(json.dumps({key: values.get(key, "") for key in WANTED}, separators=(",", ":")))
'''


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _bounded_timeout(value: float) -> float:
    return max(1.0, min(float(value), MAX_TIMEOUT_SECONDS))


def _parse_dsn_identity(
    dsn: str,
    organization_override: str = "",
    project_override: str = "",
) -> tuple[str, str]:
    organization = organization_override.strip()
    project = project_override.strip()
    try:
        parsed = urllib.parse.urlparse(dsn.strip())
    except ValueError:
        parsed = urllib.parse.urlparse("")

    if not organization:
        match = _SENTRY_DSN_ORG_RE.match((parsed.hostname or "").split(":", 1)[0])
        if match:
            organization = match.group("organization")
    if not project:
        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts:
            project = path_parts[-1]

    if not _IDENTIFIER_RE.fullmatch(organization):
        raise SentryQueryError("configuration_invalid", "organization identity unavailable")
    if not _IDENTIFIER_RE.fullmatch(project):
        raise SentryQueryError("configuration_invalid", "project identity unavailable")
    return organization, project


def _credentials_from_values(values: Mapping[str, Any]) -> Credentials:
    token = str(values.get(API_TOKEN_ENV) or "").strip()
    dsn = str(values.get(DSN_ENV) or "").strip()
    if len(token) < 16 or any(character.isspace() for character in token):
        raise SentryQueryError("api_token_unavailable")
    organization, project = _parse_dsn_identity(
        dsn,
        str(values.get(ORGANIZATION_ENV) or ""),
        str(values.get(PROJECT_ENV) or ""),
    )
    return Credentials(token=token, organization=organization, project=project)


def _load_remote_credentials(
    source_name: str,
    *,
    ssh_key: Path,
    timeout: float,
) -> Credentials:
    source = SECRET_SOURCES[source_name]
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
        source.host,
        "python3",
        "-",
        source.secret_id,
    ]
    try:
        result = subprocess.run(
            command,
            input=_REMOTE_SECRET_READER,
            capture_output=True,
            text=True,
            check=False,
            timeout=_bounded_timeout(timeout),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SentryQueryError("secret_source_unavailable", type(exc).__name__) from None
    if result.returncode != 0:
        raise SentryQueryError(
            "secret_source_unavailable",
            "ssh_status=%d" % result.returncode,
        )
    try:
        values = json.loads(result.stdout)
    except (TypeError, ValueError):
        raise SentryQueryError("secret_source_invalid") from None
    if not isinstance(values, dict):
        raise SentryQueryError("secret_source_invalid")
    return _credentials_from_values(values)


def _load_credentials(
    source_name: str,
    *,
    ssh_key: Path,
    timeout: float,
) -> Credentials:
    if source_name == "environment":
        return _credentials_from_values(os.environ)
    return _load_remote_credentials(source_name, ssh_key=ssh_key, timeout=timeout)


def _redact(value: Any, credentials: Credentials, max_length: int = 500) -> Any:
    if not isinstance(value, str):
        return value
    without_exact_token = value.replace(credentials.token, REDACTED)
    return scrub_text(without_exact_token, max_length)


def _request_json(
    path: str,
    *,
    query: Mapping[str, Any],
    credentials: Credentials,
    timeout: float,
) -> Any:
    encoded = urllib.parse.urlencode(
        [(key, str(value)) for key, value in query.items() if value is not None],
        doseq=True,
    )
    url = API_BASE_URL.rstrip("/") + "/" + path.lstrip("/")
    if encoded:
        url += "?" + encoded
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + credentials.token,
            "User-Agent": "leadpoet-codex-sentry-reader/1",
        },
    )
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=_bounded_timeout(timeout)) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(4096).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        detail = "status=%d" % int(exc.code)
        if body:
            detail += " response=" + str(_redact(body, credentials, 160))
        raise SentryQueryError("api_http_error", detail) from None
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise SentryQueryError("api_unavailable", type(exc).__name__) from None
    if len(payload) > MAX_RESPONSE_BYTES:
        raise SentryQueryError("api_response_too_large")
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise SentryQueryError("api_response_invalid") from None


def _string_field(
    item: Mapping[str, Any],
    names: Iterable[str],
    credentials: Credentials,
    *,
    max_length: int = 500,
) -> str:
    for name in names:
        value = item.get(name)
        if value is not None:
            return str(_redact(str(value), credentials, max_length))
    return ""


def _safe_tags(value: Any, credentials: Credentials) -> Dict[str, str]:
    pairs: List[tuple[str, Any]] = []
    if isinstance(value, dict):
        pairs = [(str(key), item) for key, item in value.items()]
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "key" in item:
                pairs.append((str(item.get("key")), item.get("value")))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((str(item[0]), item[1]))

    allowed = {"environment", "level", "release", "transaction"}
    result: Dict[str, str] = {}
    for key, item in pairs[:100]:
        if key not in allowed and not key.startswith("leadpoet."):
            continue
        safe_key = str(scrub_text(key, 128))
        result[safe_key] = str(_redact(str(item), credentials, 300))
    return dict(sorted(result.items()))


def _normalize_issue(item: Mapping[str, Any], credentials: Credentials) -> Dict[str, Any]:
    return {
        "id": _string_field(item, ("id",), credentials, max_length=64),
        "short_id": _string_field(item, ("shortId", "short_id"), credentials, max_length=64),
        "title": _string_field(item, ("title",), credentials),
        "culprit": _string_field(item, ("culprit",), credentials),
        "level": _string_field(item, ("level",), credentials, max_length=32),
        "status": _string_field(item, ("status",), credentials, max_length=32),
        "count": _string_field(item, ("count",), credentials, max_length=32),
        "first_seen": _string_field(item, ("firstSeen", "first_seen"), credentials, max_length=64),
        "last_seen": _string_field(item, ("lastSeen", "last_seen"), credentials, max_length=64),
        "permalink": _string_field(item, ("permalink",), credentials, max_length=500),
    }


def _normalize_event(item: Mapping[str, Any], credentials: Credentials) -> Dict[str, Any]:
    return {
        "id": _string_field(item, ("eventID", "event_id", "id"), credentials, max_length=64),
        "issue_id": _string_field(item, ("groupID", "group_id"), credentials, max_length=64),
        "title": _string_field(item, ("title",), credentials),
        "message": _string_field(item, ("message",), credentials),
        "culprit": _string_field(item, ("culprit",), credentials),
        "level": _string_field(item, ("level",), credentials, max_length=32),
        "platform": _string_field(item, ("platform",), credentials, max_length=64),
        "date_created": _string_field(
            item,
            ("dateCreated", "date_created", "timestamp"),
            credentials,
            max_length=64,
        ),
        "tags": _safe_tags(item.get("tags"), credentials),
    }


def _bounded_list(payload: Any) -> List[Mapping[str, Any]]:
    if not isinstance(payload, list):
        raise SentryQueryError("api_response_invalid", "expected list")
    return [item for item in payload[:MAX_ITEMS] if isinstance(item, dict)]


def query_issues(
    credentials: Credentials,
    *,
    query: str,
    stats_period: str,
    limit: int,
    timeout: float,
) -> List[Dict[str, Any]]:
    payload = _request_json(
        "projects/%s/%s/issues/" % (credentials.organization, credentials.project),
        query={
            "query": query,
            "statsPeriod": stats_period,
            "sort": "date",
            "limit": limit,
        },
        credentials=credentials,
        timeout=timeout,
    )
    return [_normalize_issue(item, credentials) for item in _bounded_list(payload)[:limit]]


def query_events(
    credentials: Credentials,
    *,
    stats_period: str,
    limit: int,
    timeout: float,
) -> List[Dict[str, Any]]:
    payload = _request_json(
        "projects/%s/%s/events/" % (credentials.organization, credentials.project),
        query={"statsPeriod": stats_period, "full": "false", "limit": limit},
        credentials=credentials,
        timeout=timeout,
    )
    return [_normalize_event(item, credentials) for item in _bounded_list(payload)[:limit]]


def _bounded_limit(value: int) -> int:
    if value < 1 or value > MAX_ITEMS:
        raise SentryQueryError("argument_invalid", "limit must be between 1 and 100")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("auth-check", "issues", "events"),
        help="read credentials without disclosure, or query recent issues/events",
    )
    parser.add_argument(
        "--secret-source",
        choices=("gateway", "validator", "environment"),
        default="gateway",
        help="where to read the API token and DSN (default: gateway Secrets Manager)",
    )
    parser.add_argument(
        "--ssh-key",
        type=Path,
        default=Path(os.getenv(SSH_KEY_ENV) or DEFAULT_SSH_KEY),
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--query", default="is:unresolved")
    parser.add_argument("--stats-period", default="24h")
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        limit = _bounded_limit(args.limit)
        timeout = _bounded_timeout(args.timeout_seconds)
        credentials = _load_credentials(
            args.secret_source,
            ssh_key=args.ssh_key,
            timeout=timeout,
        )
        if args.command == "auth-check":
            query_issues(
                credentials,
                query="is:unresolved",
                stats_period=args.stats_period,
                limit=1,
                timeout=timeout,
            )
            result: Dict[str, Any] = {
                "ok": True,
                "kind": "auth-check",
                "secret_source": args.secret_source,
                "organization": credentials.organization,
                "project": credentials.project,
            }
        elif args.command == "issues":
            items = query_issues(
                credentials,
                query=args.query,
                stats_period=args.stats_period,
                limit=limit,
                timeout=timeout,
            )
            result = {
                "kind": "issues",
                "secret_source": args.secret_source,
                "count": len(items),
                "items": items,
            }
        else:
            items = query_events(
                credentials,
                stats_period=args.stats_period,
                limit=limit,
                timeout=timeout,
            )
            result = {
                "kind": "events",
                "secret_source": args.secret_source,
                "count": len(items),
                "items": items,
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except SentryQueryError as exc:
        suffix = " detail=%s" % exc.detail if exc.detail else ""
        print("sentry_query_failed code=%s%s" % (exc.code, suffix), file=sys.stderr)
        return 1
    except Exception as exc:  # defensive: token-bearing state is never rendered
        print(
            "sentry_query_failed code=unexpected type=%s" % type(exc).__name__,
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
