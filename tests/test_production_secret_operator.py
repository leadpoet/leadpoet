from types import SimpleNamespace

import boto3
import pytest

from scripts import production_secret_operator as subject


class Session:
    def __init__(self, *, method="sso", account="493765492819", arn=None):
        self.method = method
        self.identity = {"Account": account, "Arn": arn or f"arn:aws:sts::{account}:assumed-role/LabOperator/operator"}
        self.calls = []
        self.secrets = object()

    def get_credentials(self):
        return SimpleNamespace(method=self.method)

    def client(self, service, **kwargs):
        assert kwargs["region_name"] == "us-east-1"
        assert kwargs["config"].ignore_configured_endpoint_urls is True
        self.calls.append(service)
        return SimpleNamespace(get_caller_identity=lambda: self.identity) if service == "sts" else self.secrets


@pytest.mark.parametrize("method", ["sso", "assume-role", "assume-role-with-web-identity"])
def test_named_temporary_operator_creates_only_hardened_local_client(monkeypatch, method):
    session = Session(method=method)
    seen = []
    monkeypatch.setattr(subject, "gateway_iam_session", lambda **kwargs: seen.append(kwargs) or session)
    client, account = subject.secret_operator_client(profile="leadpoet-lab")
    assert client is session.secrets and account == "493765492819"
    assert seen == [{"profile": "leadpoet-lab"}]
    assert session.calls == ["sts", "secretsmanager"]


@pytest.mark.parametrize("method", ["iam-role", "container-role", "env", "shared-credentials-file", "explicit", ""])
def test_nonoperator_authority_fails_before_any_secret_client(monkeypatch, method):
    session = Session(method=method)
    monkeypatch.setattr(subject, "gateway_iam_session", lambda **kwargs: session)
    with pytest.raises(subject.SecretOperatorError):
        subject.secret_operator_client()
    assert session.calls == []


@pytest.mark.parametrize("profile", [None, ""])
def test_implicit_role_chain_is_never_used_for_secret_operations(monkeypatch, profile):
    monkeypatch.setattr(subject, "gateway_iam_session", lambda **kwargs: pytest.fail("ambient role chain"))
    with pytest.raises(subject.SecretOperatorError, match="named"):
        subject.secret_operator_client(profile=profile)


@pytest.mark.parametrize(("account", "arn"), [
    ("187445349696", "arn:aws:sts::187445349696:assumed-role/LabOperator/operator"),
    ("493765492819", "arn:aws:iam::493765492819:user/pranav-main"),
    ("493765492819", "arn:aws:iam::493765492819:root"),
])
def test_wrong_account_user_or_implicit_root_never_gets_secret_client(monkeypatch, account, arn):
    session = Session(account=account, arn=arn)
    monkeypatch.setattr(subject, "gateway_iam_session", lambda **kwargs: session)
    with pytest.raises(subject.SecretOperatorError):
        subject.secret_operator_client()
    assert session.calls == ["sts"]


def test_recovery_root_requires_explicit_existing_profile(monkeypatch):
    session = Session(method="explicit", arn="arn:aws:iam::493765492819:root")
    monkeypatch.setattr(subject, "gateway_iam_session", lambda **kwargs: session)
    assert subject.secret_operator_client(profile=subject.RECOVERY_PROFILE)[0] is session.secrets


def test_authority_failure_is_redacted(monkeypatch):
    def failure(**kwargs):
        raise RuntimeError("synthetic-sensitive-provider-detail")
    monkeypatch.setattr(subject, "gateway_iam_session", failure)
    with pytest.raises(subject.SecretOperatorError) as error:
        subject.secret_operator_client()
    assert str(error.value) == "secret operator authority is unavailable"
    assert "synthetic-sensitive" not in str(error.value)


def test_secret_endpoint_overrides_are_ignored_before_network(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("AWS_ENDPOINT_URL_SECRETS_MANAGER", "http://127.0.0.1:9")
    session = boto3.session.Session(
        region_name="us-east-1", aws_access_key_id="ASIA" + "A" * 16,
        aws_secret_access_key="s" * 40, aws_session_token="temporary-token",
    )
    session.get_credentials().method = "sso"
    real_client = session.client
    identity = Session().identity
    monkeypatch.setattr(session, "client", lambda service, **kwargs:
                        SimpleNamespace(get_caller_identity=lambda: identity)
                        if service == "sts" else real_client(service, **kwargs))
    monkeypatch.setattr(subject, "gateway_iam_session", lambda **kwargs: session)
    client, _ = subject.secret_operator_client(timeout_seconds=100)
    assert client.meta.endpoint_url == "https://secretsmanager.us-east-1.amazonaws.com"
    assert client.meta.config.connect_timeout == 45
    assert client.meta.config.read_timeout == 45
