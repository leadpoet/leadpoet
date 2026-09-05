from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from leadpoet_observability.sentry_scrubbing import REDACTED, scrub_text
from scripts import configure_sentry_api_token as configure
from scripts import query_sentry_api as query


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_TOKEN = "sntryu_" + "a" * 48
FAKE_DSN = "https://public@example.ingest.sentry.io/4511844334239744"
REALISTIC_DSN = (
    "https://public@o4511244334333952.ingest.us.sentry.io/4511844334239744"
)


class _Result:
    def __init__(self, *, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, _limit):
        return self.payload


def _credentials():
    return query.Credentials(
        token=FAKE_TOKEN,
        organization="4511244334333952",
        project="4511844334239744",
    )


def _heredoc(source: str, marker: str) -> str:
    start = source.index(marker)
    body_start = source.index("<<'PY'\n", start) + len("<<'PY'\n")
    return source[body_start : source.index("\nPY", body_start)]


def test_credentials_derive_numeric_project_identity_from_dsn():
    credentials = query._credentials_from_values(
        {
            query.API_TOKEN_ENV: FAKE_TOKEN,
            query.DSN_ENV: REALISTIC_DSN,
        }
    )

    assert credentials.organization == "4511244334333952"
    assert credentials.project == "4511844334239744"
    assert credentials.token == FAKE_TOKEN


def test_remote_secret_read_never_places_token_in_ssh_command_or_script(monkeypatch):
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["input"] = kwargs["input"]
        return _Result(
            stdout=json.dumps(
                {
                    query.API_TOKEN_ENV: FAKE_TOKEN,
                    query.DSN_ENV: REALISTIC_DSN,
                }
            )
        )

    monkeypatch.setattr(query.subprocess, "run", run)

    credentials = query._load_remote_credentials(
        "gateway",
        ssh_key=Path("/tmp/test.pem"),
        timeout=2,
    )

    assert credentials.token == FAKE_TOKEN
    assert FAKE_TOKEN not in " ".join(observed["command"])
    assert FAKE_TOKEN not in observed["input"]
    assert query.API_TOKEN_ENV in observed["input"]


def test_remote_failure_does_not_echo_captured_secret(monkeypatch):
    monkeypatch.setattr(
        query.subprocess,
        "run",
        lambda *args, **kwargs: _Result(
            returncode=21,
            stdout=FAKE_TOKEN,
            stderr="failure " + FAKE_TOKEN,
        ),
    )

    with pytest.raises(query.SentryQueryError) as raised:
        query._load_remote_credentials(
            "validator",
            ssh_key=Path("/tmp/test.pem"),
            timeout=2,
        )

    assert raised.value.code == "secret_source_unavailable"
    assert FAKE_TOKEN not in str(raised.value)
    assert FAKE_TOKEN not in raised.value.detail


def test_configurator_sends_token_only_over_ssh_stdin(monkeypatch):
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["input"] = kwargs["input"]
        return _Result(stdout='{"updated":true,"format":"dotenv"}')

    monkeypatch.setattr(configure.subprocess, "run", run)

    document_format = configure._update_target(
        "gateway",
        FAKE_TOKEN,
        ssh_key=Path("/tmp/test.pem"),
        timeout=2,
    )

    assert document_format == "dotenv"
    assert observed["input"] == FAKE_TOKEN
    assert FAKE_TOKEN not in " ".join(observed["command"])


def test_configurator_failure_never_echoes_remote_output(monkeypatch):
    monkeypatch.setattr(
        configure.subprocess,
        "run",
        lambda *args, **kwargs: _Result(
            returncode=35,
            stdout=FAKE_TOKEN,
            stderr=FAKE_TOKEN,
        ),
    )

    with pytest.raises(configure.ConfigurationError) as raised:
        configure._update_target(
            "validator",
            FAKE_TOKEN,
            ssh_key=Path("/tmp/test.pem"),
            timeout=2,
        )

    assert FAKE_TOKEN not in str(raised.value)
    assert FAKE_TOKEN not in raised.value.detail


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (31, "secret_read_unavailable"),
        (32, "secret_document_invalid"),
        (34, "secret_changed_concurrently"),
        (35, "secret_writer_access_denied"),
        (36, "secret_write_readback_mismatch"),
        (37, "secret_not_found"),
        (38, "secret_write_rejected"),
        (39, "secret_write_failed"),
        (40, "secret_write_superseded"),
        (255, "secret_update_unavailable"),
    ],
)
def test_configurator_maps_remote_failure_without_exposing_stderr(
    monkeypatch,
    status,
    expected_code,
):
    monkeypatch.setattr(
        configure.subprocess,
        "run",
        lambda *args, **kwargs: _Result(
            returncode=status,
            stdout=FAKE_TOKEN,
            stderr="failure " + FAKE_TOKEN,
        ),
    )

    with pytest.raises(configure.ConfigurationError) as raised:
        configure._update_target(
            "gateway",
            FAKE_TOKEN,
            ssh_key=Path("/tmp/test.pem"),
            timeout=2,
        )

    assert raised.value.code == expected_code
    assert FAKE_TOKEN not in str(raised.value)
    assert FAKE_TOKEN not in raised.value.detail


@pytest.mark.parametrize(
    ("initial", "expected_format"),
    [
        (
            "export LEADPOET_SENTRY_ENABLED=1\n"
            "export LEADPOET_SENTRY_API_TOKEN=old-value\n"
            "export SAFE_VALUE=present\n",
            "dotenv",
        ),
        (
            json.dumps(
                {
                    "LEADPOET_SENTRY_ENABLED": "1",
                    "LEADPOET_SENTRY_API_TOKEN": "old-value",
                    "SAFE_VALUE": "present",
                }
            ),
            "json",
        ),
    ],
)
def test_remote_configurator_preserves_document_format_and_verifies_readback(
    tmp_path,
    initial,
    expected_format,
):
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "current": "v1",
                "versions": {"v1": initial},
                "stages": {"v1": ["AWSCURRENT"]},
                "write_argv": [],
            }
        ),
        encoding="utf-8",
    )
    aws = tmp_path / "aws"
    aws.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

state_path = Path(os.environ["FAKE_SECRET_STATE"])
state = json.loads(state_path.read_text(encoding="utf-8"))
args = sys.argv[1:]
if args[:2] == ["secretsmanager", "get-secret-value"]:
    version_id = (
        args[args.index("--version-id") + 1]
        if "--version-id" in args
        else state["current"]
    )
    print(json.dumps({
        "VersionId": version_id,
        "SecretString": state["versions"][version_id],
    }))
    raise SystemExit(0)
if args[:2] == ["secretsmanager", "put-secret-value"]:
    value = args[args.index("--secret-string") + 1]
    version_id = args[args.index("--client-request-token") + 1]
    if not value.startswith("file://"):
        raise SystemExit(91)
    previous = state["current"]
    state["versions"][version_id] = Path(value[7:]).read_text(encoding="utf-8")
    state["current"] = version_id
    state["stages"] = {
        previous: ["AWSPREVIOUS"],
        version_id: ["AWSCURRENT"],
    }
    state["write_argv"] = args
    state_path.write_text(json.dumps(state), encoding="utf-8")
    print(json.dumps({"VersionId": version_id}))
    raise SystemExit(0)
if args[:2] == ["secretsmanager", "describe-secret"]:
    print(json.dumps(state["stages"]))
    raise SystemExit(0)
raise SystemExit(92)
""",
        encoding="utf-8",
    )
    aws.chmod(0o700)
    env = dict(os.environ)
    env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
    env["FAKE_SECRET_STATE"] = str(state)

    result = subprocess.run(
        ["python3", "-c", configure._REMOTE_UPDATER, "test/secret"],
        input=FAKE_TOKEN,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "updated": True,
        "format": expected_format,
    }
    persisted = json.loads(state.read_text(encoding="utf-8"))
    updated = persisted["versions"][persisted["current"]]
    assert "SAFE_VALUE" in updated
    assert "old-value" not in updated
    assert updated.count(configure.TOKEN_ENV_NAME) == 1
    assert updated.count(FAKE_TOKEN) == 1
    assert persisted["stages"]["v1"] == ["AWSPREVIOUS"]
    assert persisted["stages"][persisted["current"]] == ["AWSCURRENT"]
    assert persisted["write_argv"][:2] == ["secretsmanager", "put-secret-value"]
    assert "--client-request-token" in persisted["write_argv"]
    assert FAKE_TOKEN not in persisted["write_argv"]


def test_remote_configurator_uses_versioning_not_secret_metadata_update():
    assert '"put-secret-value"' in configure._REMOTE_UPDATER
    assert '"--client-request-token"' in configure._REMOTE_UPDATER
    assert '"update-secret"' not in configure._REMOTE_UPDATER


def test_remote_configurator_sanitizes_put_secret_value_access_denied(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "VersionId": "v1",
                "SecretString": "export SAFE_VALUE=present\n",
            }
        ),
        encoding="utf-8",
    )
    aws = tmp_path / "aws"
    aws.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if args[:2] == ["secretsmanager", "get-secret-value"]:
    print(Path(os.environ["FAKE_SECRET_STATE"]).read_text(encoding="utf-8"))
    raise SystemExit(0)
if args[:2] == ["secretsmanager", "put-secret-value"]:
    print(
        "An error occurred (AccessDeniedException) when calling the "
        "PutSecretValue operation: seeded-sensitive-detail",
        file=sys.stderr,
    )
    raise SystemExit(254)
raise SystemExit(92)
""",
        encoding="utf-8",
    )
    aws.chmod(0o700)
    env = dict(os.environ)
    env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
    env["FAKE_SECRET_STATE"] = str(state)

    result = subprocess.run(
        ["python3", "-c", configure._REMOTE_UPDATER, "test/secret"],
        input=FAKE_TOKEN,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 35
    assert result.stdout == ""
    assert result.stderr == ""
    assert FAKE_TOKEN not in result.stdout + result.stderr
    assert "seeded-sensitive-detail" not in result.stdout + result.stderr


def test_issue_query_is_bounded_allowlisted_and_redacted(monkeypatch):
    observed = {}

    class Opener:
        def open(self, request, timeout):
            observed["url"] = request.full_url
            observed["authorization"] = request.headers["Authorization"]
            observed["timeout"] = timeout
            return _Response(
                [
                    {
                        "id": "123",
                        "shortId": "LEADPOET-1",
                        "title": "failure token=" + FAKE_TOKEN,
                        "culprit": "gateway.restart",
                        "level": "error",
                        "status": "unresolved",
                        "count": "4",
                        "firstSeen": "2026-08-03T00:00:00Z",
                        "lastSeen": "2026-08-03T01:00:00Z",
                        "permalink": "https://example.sentry.io/issues/123/",
                        "assignedTo": {"email": "private@example.com"},
                        "raw_payload": "must-not-survive",
                    }
                ]
            )

    monkeypatch.setattr(query.urllib.request, "build_opener", lambda *_: Opener())

    items = query.query_issues(
        _credentials(),
        query="is:unresolved",
        stats_period="24h",
        limit=10,
        timeout=5,
    )

    serialized = json.dumps(items)
    assert "projects/4511244334333952/4511844334239744/issues/" in observed["url"]
    assert observed["authorization"] == "Bearer " + FAKE_TOKEN
    assert observed["timeout"] == 5
    assert FAKE_TOKEN not in serialized
    assert REDACTED in serialized
    assert "private@example.com" not in serialized
    assert "must-not-survive" not in serialized
    assert set(items[0]) == {
        "id",
        "short_id",
        "title",
        "culprit",
        "level",
        "status",
        "count",
        "first_seen",
        "last_seen",
        "permalink",
    }


def test_event_query_keeps_only_operational_tags(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            return _Response(
                [
                    {
                        "eventID": "a" * 32,
                        "groupID": "123",
                        "title": "restart failed",
                        "message": "credential " + FAKE_TOKEN,
                        "culprit": "gw_restart",
                        "level": "error",
                        "platform": "python",
                        "dateCreated": "2026-08-03T01:00:00Z",
                        "tags": [
                            ["leadpoet.failure_code", "restart.failed"],
                            ["release", "b" * 40],
                            ["customer_email", "private@example.com"],
                        ],
                        "entries": [{"data": {"raw": "must-not-survive"}}],
                    }
                ]
            )

    monkeypatch.setattr(query.urllib.request, "build_opener", lambda *_: Opener())

    items = query.query_events(
        _credentials(),
        stats_period="24h",
        limit=10,
        timeout=5,
    )

    serialized = json.dumps(items)
    assert FAKE_TOKEN not in serialized
    assert "private@example.com" not in serialized
    assert "must-not-survive" not in serialized
    assert items[0]["tags"] == {
        "leadpoet.failure_code": "restart.failed",
        "release": "b" * 40,
    }


def test_cli_error_is_sanitized(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise query.SentryQueryError("api_http_error", "response=" + FAKE_TOKEN)

    monkeypatch.setattr(query, "_load_credentials", lambda *args, **kwargs: _credentials())
    monkeypatch.setattr(query, "query_issues", fail)

    assert query.main(["issues", "--secret-source", "environment"]) == 1
    captured = capsys.readouterr()
    assert FAKE_TOKEN not in captured.err
    assert "code=api_http_error" in captured.err


def test_sentry_user_token_shape_is_redacted_by_shared_scrubber():
    assert FAKE_TOKEN not in str(scrub_text("token=" + FAKE_TOKEN))


def test_repository_guides_match_and_sentry_runbook_keeps_safe_access_workflow():
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert agents == claude
    runbook = (REPO_ROOT / "docs/sentry_error_monitoring.md").read_text(
        encoding="utf-8"
    )
    assert "## Read-only Codex API access" in runbook
    assert "query_sentry_api.py auth-check --secret-source gateway" in runbook
    assert "query_sentry_api.py issues --secret-source gateway" in runbook
    assert "query_sentry_api.py events --secret-source gateway" in runbook
    assert "never prints or persists the token" in runbook
    assert "does not support raw event bodies or a raw-token output mode" in runbook


@pytest.mark.parametrize(
    ("relative", "marker", "secret_document"),
    [
        (
            "gw_restart.sh",
            'python3 - "$SECRET_TMP" "$GATEWAY_ENV_FILE"',
            "export LEADPOET_SENTRY_ENABLED=1\n"
            "export LEADPOET_SENTRY_API_TOKEN='" + FAKE_TOKEN + "'\n"
            "export SAFE_VALUE=present\n",
        ),
        (
            "validator_restart.sh",
            'python3 - "$SECRET_TMP" "$VALIDATOR_ENV_FILE" "$VALIDATOR_ENV_EXPORT"',
            json.dumps(
                {
                    "LEADPOET_SENTRY_ENABLED": "1",
                    "LEADPOET_SENTRY_API_TOKEN": FAKE_TOKEN,
                    "SAFE_VALUE": "present",
                }
            ),
        ),
    ],
)
def test_restart_hydration_never_caches_or_exports_api_token(
    tmp_path,
    relative,
    marker,
    secret_document,
):
    source = (REPO_ROOT / relative).read_text(encoding="utf-8")
    block = _heredoc(source, marker)
    secret = tmp_path / "secret"
    cache = tmp_path / "cache.env"
    export_file = tmp_path / "exports.sh"
    secret.write_text(secret_document, encoding="utf-8")
    arguments = ["python3", "-c", block, str(secret), str(cache)]
    if relative == "validator_restart.sh":
        arguments.append(str(export_file))

    subprocess.run(arguments, check=True, capture_output=True, text=True)

    assert "SAFE_VALUE" in cache.read_text(encoding="utf-8")
    assert FAKE_TOKEN not in cache.read_text(encoding="utf-8")
    assert query.API_TOKEN_ENV not in cache.read_text(encoding="utf-8")
    if export_file.exists():
        assert FAKE_TOKEN not in export_file.read_text(encoding="utf-8")
        assert query.API_TOKEN_ENV not in export_file.read_text(encoding="utf-8")
