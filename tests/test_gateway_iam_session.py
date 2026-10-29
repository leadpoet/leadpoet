from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.session import Session as BotocoreSession
from botocore.credentials import SSOProvider
from botocore.httpsession import URLLib3Session
from botocore.tokens import SSOTokenProvider

from scripts import gateway_iam_session as subject


@pytest.fixture(autouse=True)
def isolated_authority_environment(monkeypatch, tmp_path):
    for name in subject.FORBIDDEN:
        monkeypatch.delenv(name, raising=False)
    # Isolate both Path.home() and botocore's os.path.expanduser() lookup.
    monkeypatch.setenv("HOME", str(tmp_path))
    # These SDK defaults are fixed at import time, before HOME changes.
    token_cache = str(tmp_path / ".aws" / "sso" / "cache")
    monkeypatch.setattr(SSOProvider, "_SSO_TOKEN_CACHE_DIR", token_cache)
    monkeypatch.setattr(SSOTokenProvider, "_SSO_TOKEN_CACHE_DIR", token_cache)
    monkeypatch.setattr(socket.socket, "connect", lambda *_a, **_k: pytest.fail("unexpected socket connection"))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_k: pytest.fail("unexpected DNS lookup"))
    monkeypatch.setattr(URLLib3Session, "send", lambda *_a, **_k: pytest.fail("unexpected network call"))


class FakeSession:
    def __init__(self, method="sso", token="temporary-token", account="493765492819", arn=None):
        self.credentials = SimpleNamespace(method=method, get_frozen_credentials=lambda: SimpleNamespace(
            access_key="ASIA" + "A" * 16, secret_key="s" * 40, token=token))
        self.identity = {"Account": account, "Arn": arn or f"arn:aws:sts::{account}:assumed-role/LabOperator/test"}
        self.clients = []
        self._session = SimpleNamespace(set_default_client_config=lambda config: None)

    def get_credentials(self):
        return self.credentials

    def client(self, name, **kwargs):
        assert name == "sts"
        assert kwargs["config"].ignore_configured_endpoint_urls is True
        assert kwargs["region_name"] == "us-east-1"
        self.clients.append(name)
        return SimpleNamespace(get_caller_identity=lambda: self.identity)


def test_sdk_uses_only_isolated_home_and_token_cache(tmp_path):
    assert os.path.expanduser("~") == str(tmp_path)
    sdk = BotocoreSession()
    assert os.path.expanduser(sdk.get_config_variable("config_file")) == str(tmp_path / ".aws" / "config")
    assert os.path.expanduser(sdk.get_config_variable("credentials_file")) == str(tmp_path / ".aws" / "credentials")
    expected = str(tmp_path / ".aws" / "sso" / "cache")
    assert SSOProvider._SSO_TOKEN_CACHE_DIR == expected
    assert SSOTokenProvider._SSO_TOKEN_CACHE_DIR == expected


@pytest.mark.parametrize("profile,method", [("leadpoet-lab", "sso"), (None, "iam-role"), ("operator-role", "assume-role")])
def test_temporary_roles_work_without_gateway_cache(monkeypatch, profile, method):
    session = FakeSession(method=method)
    captured = {}
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **kwargs: captured.update(kwargs) or session)
    monkeypatch.setattr(Path, "read_text", lambda *_a, **_k: pytest.fail("must not read gateway.env"))
    assert subject.gateway_iam_session(profile=profile) is session
    assert captured == {"profile_name": profile, "region_name": "us-east-1"}
    assert session.clients == ["sts"]


def test_default_explicitly_selects_operator_profile(monkeypatch):
    session = FakeSession()
    captured = {}
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **kwargs: captured.update(kwargs) or session)
    assert subject.gateway_iam_session() is session
    assert captured["profile_name"] == "leadpoet-lab"


@pytest.mark.parametrize("method,token", [("env", ""), ("shared-credentials-file", "temporary"), ("config-file", "temporary"), ("sso", "")])
def test_rejects_static_or_unapproved_providers(monkeypatch, method, token):
    session = FakeSession(method=method, token=token)
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **_kwargs: session)
    with pytest.raises(subject.GatewayIamSessionError, match="temporary"):
        subject.gateway_iam_session()
    assert session.clients == []


@pytest.mark.parametrize("account,arn", [
    ("187445349696", "arn:aws:sts::187445349696:assumed-role/LabOperator/test"),
    ("493765492819", "arn:aws:iam::493765492819:user/pranav-main"),
    ("493765492819", "arn:aws:iam::493765492819:root"),
    ("493765492819", "arn:aws:sts::187445349696:assumed-role/LabOperator/test"),
])
def test_rejects_wrong_account_or_nonrole_identity(monkeypatch, account, arn):
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **_kwargs: FakeSession(account=account, arn=arn))
    with pytest.raises(subject.GatewayIamSessionError, match="identity"):
        subject.gateway_iam_session()


@pytest.mark.parametrize("name", sorted(subject.FORBIDDEN))
def test_rejects_ambient_authority_before_resolving(monkeypatch, name):
    monkeypatch.setenv(name, "unsafe-test-authority")
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **_k: pytest.fail("must refuse before resolving"))
    with pytest.raises(subject.GatewayIamSessionError, match="authority"):
        subject.gateway_iam_session()


def test_retired_root_profile_is_rejected_before_session_or_export(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("must not export root credentials"))
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **_k: pytest.fail("must not resolve retired profile"))
    with pytest.raises(subject.GatewayIamSessionError, match="authority"):
        subject.gateway_iam_session(profile="leadpoet-recovery-console-20261026")


@pytest.mark.parametrize("profile", ["leadpoet-lab"])
def test_root_is_rejected_without_export_for_each_operator_profile(monkeypatch, profile):
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("must not export root credentials"))
    session = FakeSession(arn="arn:aws:iam::493765492819:root")
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **_k: session)
    with pytest.raises(subject.GatewayIamSessionError, match="identity"):
        subject.gateway_iam_session(profile=profile)
    assert session.clients == ["sts"]


def admin_cache(tmp_path, monkeypatch, *, expires=None, account=subject.ACCOUNT,
                role="Admin", session_name="leadpoet-admin", data=None):
    from botocore.credentials import SSOCredentialFetcher

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    aws_dir = tmp_path / ".aws"
    aws_dir.mkdir()
    (aws_dir / "config").write_text(
        f"[profile leadpoet-admin]\nsso_account_id = {account}\n"
        f"sso_role_name = {role}\nsso_session = {session_name}\n"
        f"[sso-session {session_name}]\nsso_start_url = https://example.awsapps.com/start\n"
        "sso_region = us-east-1\n"
    )
    cache_dir = aws_dir / "cli" / "cache"
    cache_dir.mkdir(parents=True)
    key = hashlib.sha1(json.dumps({"accountId": account, "roleName": role,
                                   "sessionName": session_name},
                                  sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache_file = cache_dir / f"{key}.json"
    if data is not None:
        cache_file.write_text(json.dumps(data))
    elif expires is not None:
        # Generate the persisted response with the installed SDK fetcher.
        client = SimpleNamespace(get_role_credentials=lambda **_k: {"roleCredentials": {
            "accessKeyId": "ASIA" + "A" * 16, "secretAccessKey": "s" * 40,
            "sessionToken": "temporary-token", "expiration": int(expires.timestamp() * 1000)}})
        fetcher = SSOCredentialFetcher(
            start_url="https://example.awsapps.com/start", sso_region="us-east-1",
            role_name=role, account_id=account, sso_session_name=session_name,
            client_creator=lambda *_a, **_k: client,
            token_loader=lambda *_a: {"accessToken": "synthetic-token",
                                     "expiresAt": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()},
            cache={})
        assert fetcher._cache_key == key
        cache_file.write_text(json.dumps(fetcher._get_credentials()))
    return cache_file


def test_admin_uses_exact_valid_cache_without_sso_provider(monkeypatch, tmp_path):
    admin_cache(tmp_path, monkeypatch, expires=datetime.now(timezone.utc) + timedelta(hours=1))
    real_session = subject.boto3.session.Session
    seen = []

    def explicit_session(**kwargs):
        seen.append(kwargs)
        assert "profile_name" not in kwargs
        session = real_session(**kwargs)
        assert session.get_credentials().method == "explicit"
        monkeypatch.setattr(session, "client", lambda service, **opts: (
            FakeSession(arn=f"arn:aws:sts::{subject.ACCOUNT}:assumed-role/AWSReservedSSO_Admin_9f135d1e5dbc915b/pranav-admin")
            .client(service, **opts)))
        return session

    monkeypatch.setattr(subject.boto3.session, "Session", explicit_session)
    session = subject.gateway_iam_session(profile="leadpoet-admin")
    assert session.get_credentials().method == "explicit"
    assert len(seen) == 1


@pytest.mark.parametrize("case", ["missing", "expired", "near_expiry", "malformed", "static_key", "wrong_cached_account"])
def test_admin_bad_cache_fails_before_credential_provider(monkeypatch, tmp_path, case):
    expires = datetime.now(timezone.utc) + (timedelta(hours=-1) if case == "expired" else
                                            timedelta(seconds=30) if case == "near_expiry" else timedelta(hours=1))
    path = admin_cache(tmp_path, monkeypatch, expires=None if case == "missing" else expires)
    if case == "malformed":
        path.write_text("{invalid")
    if case == "static_key":
        payload = json.loads(path.read_text())
        payload["Credentials"]["AccessKeyId"] = "AKIA" + "A" * 16
        path.write_text(json.dumps(payload))
    if case == "wrong_cached_account":
        payload = json.loads(path.read_text())
        payload["Credentials"]["AccountId"] = "111111111111"
        path.write_text(json.dumps(payload))
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **_k: pytest.fail("must not resolve a provider"))
    with pytest.raises(subject.GatewayIamSessionError) as failure:
        subject.gateway_iam_session(profile="leadpoet-admin")
    assert "credential" not in str(failure.value).lower() or "cache" in str(failure.value).lower()


@pytest.mark.parametrize("field,value", [("account", "111111111111"), ("role", "Other"),
                                          ("session_name", "other")])
def test_admin_wrong_profile_authority_never_resolves(monkeypatch, tmp_path, field, value):
    values = {field: value}
    admin_cache(tmp_path, monkeypatch, expires=datetime.now(timezone.utc) + timedelta(hours=1), **values)
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **_k: pytest.fail("must not resolve a provider"))
    with pytest.raises(subject.GatewayIamSessionError, match="authority"):
        subject.gateway_iam_session(profile="leadpoet-admin")


@pytest.mark.parametrize("arn", [
    f"arn:aws:sts::{subject.ACCOUNT}:assumed-role/AWSReservedSSO_Admin_9f135d1e5dbc915b/other-user",
    f"arn:aws:sts::{subject.ACCOUNT}:assumed-role/AWSReservedSSO_Admin_bad/pranav-admin",
    f"arn:aws:sts::{subject.ACCOUNT}:assumed-role/Other/pranav-admin",
    f"arn:aws:iam::{subject.ACCOUNT}:root",
])
def test_admin_wrong_sts_identity_is_rejected(monkeypatch, tmp_path, arn):
    admin_cache(tmp_path, monkeypatch, expires=datetime.now(timezone.utc) + timedelta(hours=1))
    real_session = subject.boto3.session.Session

    def explicit_session(**kwargs):
        assert "profile_name" not in kwargs
        assert {"aws_access_key_id", "aws_secret_access_key", "aws_session_token"} <= kwargs.keys()
        session = real_session(**kwargs)
        monkeypatch.setattr(session, "client", lambda service, **opts: FakeSession(arn=arn).client(service, **opts))
        return session

    monkeypatch.setattr(subject.boto3.session, "Session", explicit_session)
    with pytest.raises(subject.GatewayIamSessionError, match="identity"):
        subject.gateway_iam_session(profile="leadpoet-admin")


def test_authority_errors_do_not_expose_provider_output(monkeypatch):
    def fail(**_kwargs):
        raise RuntimeError("synthetic-sensitive-provider-detail")
    monkeypatch.setattr(subject.boto3.session, "Session", fail)
    with pytest.raises(subject.GatewayIamSessionError) as failure:
        subject.gateway_iam_session()
    assert str(failure.value) == "gateway IAM authority is unavailable"


def test_instance_role_mode_rejects_default_sso_profile(monkeypatch):
    session = FakeSession(method="sso")
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **_kwargs: session)
    with pytest.raises(subject.GatewayIamSessionError, match="temporary"):
        subject.gateway_iam_session(profile=None)
    assert session.clients == []


@pytest.mark.parametrize("service", ["sts", "iam"])
def test_clients_ignore_ambient_endpoint_urls_without_network(monkeypatch, service):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("AWS_ENDPOINT_URL_" + service.upper(), "http://127.0.0.1:9")
    session = subject.boto3.session.Session(aws_access_key_id="ASIA" + "A" * 16,
                                           aws_secret_access_key="s" * 40,
                                           aws_session_token="temporary-token", region_name="us-east-1")
    client = session.client(service, region_name="us-east-1", config=subject.AWS_CLIENT_CONFIG)
    assert client.meta.endpoint_url.startswith("https://")
    assert "amazonaws.com" in client.meta.endpoint_url


def test_sso_refresh_client_ignores_endpoint_overrides_before_resolution(monkeypatch):
    from botocore.credentials import DeferredRefreshableCredentials, SSOCredentialFetcher

    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("AWS_ENDPOINT_URL_SSO", "http://127.0.0.1:9")
    session = subject.boto3.session.Session(region_name="us-east-1",
        aws_access_key_id="ASIA" + "A" * 16, aws_secret_access_key="s" * 40,
        aws_session_token="temporary-token")
    selected_endpoints = []

    def provider_client(service, **kwargs):
        client = session._session.create_client(service, **kwargs)
        selected_endpoints.append(client.meta.endpoint_url)
        return SimpleNamespace(get_role_credentials=lambda **_k: {"roleCredentials": {
            "accessKeyId": "ASIA" + "A" * 16, "secretAccessKey": "s" * 40,
            "sessionToken": "temporary-token", "expiration": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000)}})

    fetcher = SSOCredentialFetcher(start_url="https://test.awsapps.com/start", sso_region="us-east-1",
        role_name="LabOperator", account_id="493765492819", client_creator=provider_client,
        token_loader=lambda *_a: {"accessToken": "temporary-sso-token",
                                 "expiresAt": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()})
    credentials = DeferredRefreshableCredentials(method="sso", refresh_using=fetcher.fetch_credentials)
    monkeypatch.setattr(session, "get_credentials", lambda: credentials)
    monkeypatch.setattr(session, "client", FakeSession().client)
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **_k: session)
    assert subject.gateway_iam_session() is session
    assert selected_endpoints == ["https://portal.sso.us-east-1.amazonaws.com"]


def test_sso_oidc_token_refresh_inherits_endpoint_safety(monkeypatch):
    from botocore.tokens import SSOTokenProvider

    monkeypatch.setenv("AWS_ENDPOINT_URL_SSO_OIDC", "http://127.0.0.1:9")
    session = subject.boto3.session.Session(region_name="us-east-1",
        aws_access_key_id="ASIA" + "A" * 16, aws_secret_access_key="s" * 40,
        aws_session_token="temporary-token")
    provider = SSOTokenProvider(session._session, cache={})
    provider.__dict__["_sso_config"] = {"sso_region": "us-east-1"}
    endpoints = []

    def frozen():
        # This is the real provider's lazy token-refresh client construction.
        endpoints.append(provider._client.meta.endpoint_url)
        return SimpleNamespace(access_key="ASIA" + "A" * 16, token="temporary-token")

    monkeypatch.setattr(session, "get_credentials", lambda: SimpleNamespace(method="sso", get_frozen_credentials=frozen))
    monkeypatch.setattr(session, "client", FakeSession().client)
    monkeypatch.setattr(subject.boto3.session, "Session", lambda **_k: session)
    assert subject.gateway_iam_session() is session
    assert endpoints == ["https://oidc.us-east-1.amazonaws.com"]
