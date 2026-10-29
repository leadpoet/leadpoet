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


class _SentryMemoryClient:
    def __init__(self, initial):
        self.current = "v1"
        self.versions = {"v1": initial}
        self.stages = {"v1": ["AWSCURRENT"]}
        self.puts = []
        self.reads = 0

    def get_secret_value(self, **kwargs):
        self.reads += 1
        version = kwargs.get("VersionId", self.current)
        return {"VersionId": version, "SecretString": self.versions[version]}

    def put_secret_value(self, **kwargs):
        previous = self.current
        self.current = kwargs["ClientRequestToken"]
        self.versions[self.current] = kwargs["SecretString"]
        self.stages = {previous: ["AWSPREVIOUS"], self.current: ["AWSCURRENT"]}
        self.puts.append(kwargs)
        return {"VersionId": self.current}

    def describe_secret(self, **kwargs):
        return {"VersionIdsToStages": self.stages}


def test_configurator_keeps_token_in_local_sdk_memory_without_subprocess(monkeypatch, capsys):
    from scripts import production_secret_operator
    client = _SentryMemoryClient("export SAFE_VALUE=present\n")
    seen = []
    monkeypatch.setattr(production_secret_operator, "secret_operator_client",
                        lambda **kwargs: seen.append(kwargs) or (client, "493765492819"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("secret subprocess"))
    assert configure._update_target("gateway", FAKE_TOKEN, timeout=2) == "dotenv"
    assert seen == [{"profile": "leadpoet-lab", "timeout_seconds": 5.0}]
    assert client.puts[0]["SecretString"].count(FAKE_TOKEN) == 1
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(("initial", "expected_format"), [
    ("export LEADPOET_SENTRY_ENABLED=1\nexport LEADPOET_SENTRY_API_TOKEN=old-value\nexport SAFE_VALUE=present\n", "dotenv"),
    (json.dumps({"LEADPOET_SENTRY_ENABLED": "1", "LEADPOET_SENTRY_API_TOKEN": "old-value", "SAFE_VALUE": "present"}), "json"),
])
def test_local_configurator_preserves_document_format_and_verifies_readback(initial, expected_format):
    client = _SentryMemoryClient(initial)
    assert configure._update_secret(client, configure.TARGETS["gateway"], FAKE_TOKEN) == expected_format
    updated = client.versions[client.current]
    assert "SAFE_VALUE" in updated and "old-value" not in updated
    assert updated.count(configure.TOKEN_ENV_NAME) == 1
    assert updated.count(FAKE_TOKEN) == 1
    assert client.stages["v1"] == ["AWSPREVIOUS"]
    assert client.stages[client.current] == ["AWSCURRENT"]
    assert set(client.puts[0]) == {"SecretId", "ClientRequestToken", "SecretString"}


@pytest.mark.parametrize(("provider_code", "expected_code"), [
    ("AccessDeniedException", "secret_writer_access_denied"),
    ("ResourceNotFoundException", "secret_not_found"),
    ("InvalidRequestException", "secret_write_rejected"),
    ("UnknownProviderFailure", "secret_write_failed"),
])
def test_configurator_preserves_redacted_sdk_failure_codes(provider_code, expected_code):
    from botocore.exceptions import ClientError
    client = _SentryMemoryClient("export SAFE_VALUE=present\n")
    def fail(**kwargs):
        raise ClientError({"Error": {"Code": provider_code, "Message": FAKE_TOKEN}}, "PutSecretValue")
    client.put_secret_value = fail
    with pytest.raises(configure.ConfigurationError) as raised:
        configure._update_secret(client, configure.TARGETS["gateway"], FAKE_TOKEN)
    assert raised.value.code == expected_code
    assert FAKE_TOKEN not in str(raised.value) + raised.value.detail


@pytest.mark.parametrize("failure", ["read", "document", "race", "readback", "stages"])
def test_configurator_preserves_read_race_readback_and_stage_guards(failure):
    client = _SentryMemoryClient("export SAFE_VALUE=present\n")
    get = client.get_secret_value
    def altered(**kwargs):
        if failure == "read":
            raise RuntimeError(FAKE_TOKEN)
        if failure == "document":
            return {}
        value = get(**kwargs)
        if failure == "race" and client.reads == 2:
            value["VersionId"] = "concurrent"
        if failure == "readback" and kwargs.get("VersionId"):
            value["SecretString"] = "different"
        return value
    client.get_secret_value = altered
    if failure == "stages":
        client.describe_secret = lambda **kwargs: {"VersionIdsToStages": {}}
    expected = {"read": "secret_read_unavailable", "document": "secret_document_invalid",
                "race": "secret_changed_concurrently", "readback": "secret_write_readback_mismatch",
                "stages": "secret_write_superseded"}[failure]
    with pytest.raises(configure.ConfigurationError) as raised:
        configure._update_secret(client, configure.TARGETS["gateway"], FAKE_TOKEN)
    assert raised.value.code == expected
    assert FAKE_TOKEN not in str(raised.value) + raised.value.detail
    if failure in {"read", "document", "race"}:
        assert client.puts == []


def test_sentry_check_is_read_only_and_uses_explicit_operator(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(configure, "_read_token", lambda: FAKE_TOKEN)
    monkeypatch.setattr(configure, "_update_target", lambda name, token, **kwargs: calls.append((name, kwargs)) or "dotenv")
    assert configure.main(["--check", "--aws-profile", "leadpoet-lab"]) == 0
    assert [kwargs["apply"] for _, kwargs in calls] == [False, False]
    assert [kwargs["profile"] for _, kwargs in calls] == ["leadpoet-lab", "leadpoet-lab"]
    assert FAKE_TOKEN not in capsys.readouterr().out


@pytest.mark.parametrize("failed_stage", ["preflight", "apply"])
def test_sentry_two_target_preflight_and_partial_apply_are_reported(monkeypatch, capsys, failed_stage):
    calls = []
    monkeypatch.setattr(configure, "_read_token", lambda: FAKE_TOKEN)
    def target(name, token, **kwargs):
        calls.append((name, kwargs["apply"]))
        if name == "validator" and kwargs["apply"] == (failed_stage == "apply"):
            raise configure.ConfigurationError("secret_read_unavailable")
        return "dotenv"
    monkeypatch.setattr(configure, "_update_target", target)
    assert configure.main([]) == 1
    if failed_stage == "preflight":
        assert calls == [("gateway", False), ("validator", False)]
    else:
        assert calls == [("gateway", False), ("validator", False), ("gateway", True), ("validator", True)]
    output = capsys.readouterr()
    assert FAKE_TOKEN not in output.out + output.err
    if failed_stage == "apply":
        assert "applied_targets=1" in output.err


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


@pytest.mark.parametrize("command", ["events", "auth-check"])
def test_cli_rejects_unsupported_query_before_credentials(monkeypatch, capsys, command):
    def unexpected_credentials(*args, **kwargs):
        pytest.fail("invalid arguments must not read production credentials")

    monkeypatch.setattr(query, "_load_credentials", unexpected_credentials)
    assert query.main([command, "--query", "timestamp:>2026-10-08T07:40:00Z"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--query is supported only for issues" in captured.err


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [([], "is:unresolved"), (["--query", "lastSeen:>2026-10-08T07:40:00Z"],
                            "lastSeen:>2026-10-08T07:40:00Z"),
     (["--query", ""], "")],
)
def test_cli_preserves_issue_query_and_default(monkeypatch, capsys, arguments, expected):
    observed = {}

    def request(path, *, query, **kwargs):
        observed.update(query)
        return []

    monkeypatch.setattr(query, "_load_credentials", lambda *args, **kwargs: _credentials())
    monkeypatch.setattr(query, "_request_json", request)
    assert query.main(["issues", *arguments]) == 0
    assert observed["query"] == expected
    assert json.loads(capsys.readouterr().out)["kind"] == "issues"


def test_cli_event_listing_still_uses_its_time_window(monkeypatch, capsys):
    observed = {}

    def request(path, *, query, **kwargs):
        observed.update(query)
        return []

    monkeypatch.setattr(query, "_load_credentials", lambda *args, **kwargs: _credentials())
    monkeypatch.setattr(query, "_request_json", request)
    assert query.main(["events", "--stats-period", "1h", "--limit", "10"]) == 0
    assert observed == {"statsPeriod": "1h", "full": "false", "limit": 10}
    assert json.loads(capsys.readouterr().out)["kind"] == "events"


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
    subprocess.run(arguments, check=True, capture_output=True, text=True)

    assert "SAFE_VALUE" in cache.read_text(encoding="utf-8")
    assert FAKE_TOKEN not in cache.read_text(encoding="utf-8")
    assert query.API_TOKEN_ENV not in cache.read_text(encoding="utf-8")
    if export_file.exists():
        assert FAKE_TOKEN not in export_file.read_text(encoding="utf-8")
        assert query.API_TOKEN_ENV not in export_file.read_text(encoding="utf-8")
