from types import SimpleNamespace

import pytest

from scripts import aws_admin_cached as subject
from scripts.gateway_iam_session import GatewayIamSessionError


class CachedSession:
    def get_credentials(self):
        return SimpleNamespace(get_frozen_credentials=lambda: SimpleNamespace(
            access_key="ASIA" + "A" * 16,
            secret_key="s" * 40,
            token="temporary-token",
        ))


def test_cli_passes_fixed_credentials_to_child_without_profile_or_config(monkeypatch):
    calls = []
    monkeypatch.setattr(subject, "gateway_iam_session", lambda **kwargs: calls.append(kwargs) or CachedSession())
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("AWS_ENDPOINT_URL_STS", "http://127.0.0.1:9")

    def child(argv, *, env, check):
        assert argv == ["aws", "rds", "describe-db-instances"]
        assert check is False
        assert env["AWS_ACCESS_KEY_ID"].startswith("ASIA")
        assert env["AWS_SESSION_TOKEN"] == "temporary-token"
        assert env["AWS_DEFAULT_REGION"] == "us-east-1"
        assert env["AWS_IGNORE_CONFIGURED_ENDPOINT_URLS"] == "true"
        assert env["AWS_CONFIG_FILE"] == "/dev/null"
        assert env["AWS_SHARED_CREDENTIALS_FILE"] == "/dev/null"
        assert "AWS_PROFILE" not in env
        assert "AWS_ENDPOINT_URL" not in env
        assert "AWS_ENDPOINT_URL_STS" not in env
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(subject.subprocess, "run", child)
    assert subject.run_cached_admin_aws(["rds", "describe-db-instances"]) == 7
    assert calls == [{"profile": "leadpoet-admin"}]


@pytest.mark.parametrize("argv", [
    ["--profile", "leadpoet-admin", "sts", "get-caller-identity"],
    ["--profile=leadpoet-admin", "sts", "get-caller-identity"],
    ["--endpoint-url", "https://example.invalid", "rds", "describe-db-instances"],
    ["--no-sign-request", "rds", "describe-db-instances"],
    ["rds", "describe-db-instances", "--prof=leadpoet-admin"],
    ["rds", "describe-db-instances", "--deb"],
    ["rds", "describe-db-instances", "--no-verify-ssl"],
    ["rds", "describe-db-instances", "--ca-bundle", "/tmp/unsafe.pem"],
    ["rds", "describe-db-instances", "--region", "us-west-2"],
    ["rds", "describe-db-instances", "--cli-auto-prompt"],
    ["sso", "get-role-credentials"], ["sso", "login"],
    ["sso-oidc", "create-token"], ["configure", "export-credentials"],
    ["sts", "assume-role"], ["sts", "assume-role-with-web-identity"],
    ["sts", "assume-role-with-saml"], ["sts", "get-session-token"],
    ["sts", "get-federation-token"], ["sts", "assume-root"],
    ["iam", "create-access-key"], ["iam", "reset-service-specific-credential"],
])
def test_cli_refuses_credential_issuance_or_unsafe_overrides(monkeypatch, argv):
    monkeypatch.setattr(subject, "gateway_iam_session", lambda **_k: pytest.fail("must not resolve authority"))
    monkeypatch.setattr(subject.subprocess, "run", lambda *_a, **_k: pytest.fail("must not run CLI"))
    with pytest.raises(subject.GatewayIamSessionError):
        subject.run_cached_admin_aws(argv)


def test_cli_bad_cache_never_starts_child_and_redacts_failure(monkeypatch, capsys):
    def unavailable(**_kwargs):
        raise GatewayIamSessionError("synthetic-private-cache-detail")
    monkeypatch.setattr(subject, "gateway_iam_session", unavailable)
    monkeypatch.setattr(subject.subprocess, "run", lambda *_a, **_k: pytest.fail("must not run CLI"))
    assert subject.main(["rds", "describe-db-instances"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "synthetic-private" not in captured.err
