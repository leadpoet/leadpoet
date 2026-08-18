from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from botocore.exceptions import ClientError
import pytest

from scripts import bootstrap_production_parity_staging as bootstrap
from scripts import install_production_parity_static_secrets as installer
from scripts import setup_production_parity_staging as setup


ACCOUNT = "493765492819"
COMMIT = "a" * 40
MIGRATION_HASH = "b" * 64
PASSWORD = "c" * 64
DSN = (
    "postgresql://leadpoet_parity_reader.qplwoislplkcegvdmbim:"
    + PASSWORD
    + "@aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require"
)


def _not_found(operation: str = "GetSecretValue") -> ClientError:
    return ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "absent"}},
        operation,
    )


def test_iam_policies_use_true_run_namespace_and_bounded_static_trust():
    expiry = "2026-08-18T12:15:00Z"
    trust = setup._static_bootstrap_trust(
        account_id=ACCOUNT,
        validator_role_name=setup.VALIDATOR_ROLE,
        expires_at=expiry,
    )
    encoded_trust = json.dumps(trust, sort_keys=True)
    assert "sts:DurationSeconds" not in encoded_trust
    assert trust["Statement"][0]["Condition"] == {
        "DateLessThan": {"aws:CurrentTime": expiry}
    }
    static = setup._static_bootstrap_policy(
        account_id=ACCOUNT,
        region="us-east-1",
        readonly_secret_id=setup.READONLY_DSN_SECRET_ID,
        miner_intake_secret_id=setup.DEFAULT_MINER_INTAKE_SECRET_ID,
        expires_at=expiry,
    )
    encoded_static = json.dumps(static, sort_keys=True)
    assert "secretsmanager:GetResourcePolicy" in encoded_static
    assert "secretsmanager:ListSecretVersionIds" in encoded_static
    create = next(
        item for item in static["Statement"]
        if "secretsmanager:CreateSecret" in (
            item["Action"] if isinstance(item["Action"], list) else [item["Action"]]
        )
    )
    assert create["Resource"] == "*"
    assert create["Condition"]["StringEquals"]["secretsmanager:Name"] == [
        setup.READONLY_DSN_SECRET_ID,
        setup.DEFAULT_MINER_INTAKE_SECRET_ID,
    ]
    tag_static = next(
        item for item in static["Statement"]
        if item["Effect"] == "Allow"
        and item["Action"] == "secretsmanager:TagResource"
    )
    assert all(
        resource.endswith("-??????") for resource in tag_static["Resource"]
    )
    expiry_deny = next(
        item for item in static["Statement"]
        if item["Effect"] == "Deny" and item["Action"] == "*"
    )
    assert expiry_deny == {
        "Effect": "Deny",
        "Action": "*",
        "Resource": "*",
        "Condition": {
            "DateGreaterThanEquals": {"aws:CurrentTime": expiry}
        },
    }

    controller = setup._controller_policy(
        account_id=ACCOUNT,
        region="us-east-1",
        production_secret_id=setup.PRODUCTION_GATEWAY_SECRET_ID,
        readonly_secret_id=setup.READONLY_DSN_SECRET_ID,
        miner_intake_secret_id=setup.DEFAULT_MINER_INTAKE_SECRET_ID,
        runner_arn=f"arn:aws:iam::{ACCOUNT}:role/{setup.RUNNER_ROLE}",
    )
    encoded = json.dumps(controller, sort_keys=True)
    assert "production-parity/runs/pp-*/gateway-??????" in encoded
    assert f"{setup.PRODUCTION_GATEWAY_SECRET_ID}-??????" in encoded
    create_run = next(
        item for item in controller["Statement"]
        if item["Effect"] == "Allow"
        and item["Action"] == "secretsmanager:CreateSecret"
    )
    assert create_run["Resource"] == "*"
    assert create_run["Condition"]["StringLike"]["secretsmanager:Name"] == (
        "leadpoet/staging/production-parity/runs/pp-*/gateway"
    )
    deny = [item for item in controller["Statement"] if item["Effect"] == "Deny"]
    assert setup.READONLY_DSN_SECRET_ID in json.dumps(deny)
    assert setup.DEFAULT_MINER_INTAKE_SECRET_ID in json.dumps(deny)


def test_iam_only_has_no_github_or_secretsmanager_dependency(monkeypatch):
    calls: list[tuple[str, object]] = []

    class IAM:
        def get_role(self, *, RoleName):
            assert RoleName == setup.VALIDATOR_ROLE
            return {
                "Role": {
                    "Arn": f"arn:aws:iam::{ACCOUNT}:role/{setup.VALIDATOR_ROLE}"
                }
            }

        def attach_role_policy(self, **kwargs):
            calls.append(("attach", kwargs))

        def update_assume_role_policy(self, **kwargs):
            calls.append(("trust", kwargs))

    iam = IAM()
    monkeypatch.setattr(setup, "_iam_clients", lambda region: (object(), iam, ACCOUNT))
    monkeypatch.setattr(
        setup, "_ensure_oidc_provider", lambda *_: f"arn:aws:iam::{ACCOUNT}:oidc-provider/token.actions.githubusercontent.com"
    )
    monkeypatch.setattr(
        setup,
        "_ensure_role",
        lambda _iam, *, name, **kwargs: f"arn:aws:iam::{ACCOUNT}:role/{name}",
    )
    monkeypatch.setattr(
        setup, "_put_policy", lambda _iam, **kwargs: calls.append(("policy", kwargs))
    )
    monkeypatch.setattr(setup, "_assert_role_configuration", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup, "_ensure_instance_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup, "_delete_static_bootstrap_role", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        setup, "_gh_variable", lambda *args, **kwargs: pytest.fail("GitHub call")
    )
    args = argparse.Namespace(
        repository=setup.DEFAULT_REPOSITORY,
        region=setup.EXPECTED_REGION,
        production_gateway_ip=setup.PRODUCTION_GATEWAY_IP,
        production_gateway_secret_id=setup.PRODUCTION_GATEWAY_SECRET_ID,
        readonly_dsn_secret_id=setup.READONLY_DSN_SECRET_ID,
        miner_intake_secret_id=setup.DEFAULT_MINER_INTAKE_SECRET_ID,
        volume_gib=512,
    )
    result = setup.setup_iam_only(args)
    assert result["status"] == "iam_ready"
    assert result["github_variables_mutated"] is False
    assert result["static_bootstrap_installer_requested_assume_seconds"] == 900
    assert "static_bootstrap_max_assume_seconds" not in result
    expiry = datetime.fromisoformat(
        result["static_bootstrap_trust_expires_at"].replace("Z", "+00:00")
    )
    assert datetime.now(timezone.utc) < expiry <= datetime.now(timezone.utc) + timedelta(minutes=16)
    assert all("secretsmanager" not in name.lower() for name, _ in calls)


def test_oidc_provider_creation_and_retry_preserve_tagless_shared_shape():
    class IAM:
        created = False
        create_calls: list[dict[str, object]] = []

        def list_open_id_connect_providers(self):
            return {
                "OpenIDConnectProviderList": (
                    [{"Arn": f"arn:aws:iam::{ACCOUNT}:oidc-provider/"
                        "token.actions.githubusercontent.com"}]
                    if self.created else []
                )
            }

        def create_open_id_connect_provider(self, **kwargs):
            self.create_calls.append(kwargs)
            self.created = True

        def get_open_id_connect_provider(self, **kwargs):
            return {
                "Url": "token.actions.githubusercontent.com",
                "ClientIDList": ["sts.amazonaws.com"],
                "ThumbprintList": ["a" * 40],
                "Tags": [],
            }

    iam = IAM()
    expected = (
        f"arn:aws:iam::{ACCOUNT}:oidc-provider/"
        "token.actions.githubusercontent.com"
    )
    assert setup._ensure_oidc_provider(iam, ACCOUNT) == expected
    assert setup._ensure_oidc_provider(iam, ACCOUNT) == expected
    assert iam.create_calls == [{
        "Url": setup.OIDC_URL,
        "ClientIDList": ["sts.amazonaws.com"],
    }]


@pytest.mark.parametrize(
    ("role_name", "expected_attached"),
    (
        (setup.CONTROLLER_ROLE, set()),
        (
            setup.RUNNER_ROLE,
            {"arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"},
        ),
    ),
    ids=("partial-controller", "partial-runner"),
)
def test_partial_owned_role_converges_and_reads_back_inert_trust_before_return(
    role_name, expected_attached
):
    active = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": "*"},
            "Action": "sts:AssumeRole",
        }],
    }

    class IAM:
        trust = active
        duration = 3600
        events: list[str] = []

        def get_role(self, **kwargs):
            self.events.append("get-role")
            return {"Role": {
                "Arn": f"arn:aws:iam::{ACCOUNT}:role/{role_name}",
                "Path": "/",
                "AssumeRolePolicyDocument": self.trust,
                "MaxSessionDuration": self.duration,
            }}

        def list_role_tags(self, **kwargs):
            return {"Tags": [{
                "Key": "leadpoet:purpose",
                "Value": "production-parity",
            }]}

        def list_role_policies(self, **kwargs):
            # Simulate response loss immediately after create_role, before
            # put_role_policy/attach_role_policy completed.
            return {"PolicyNames": []}

        def list_attached_role_policies(self, **kwargs):
            return {"AttachedPolicies": []}

        def update_assume_role_policy(self, **kwargs):
            self.events.append("set-inert")
            self.trust = json.loads(kwargs["PolicyDocument"])

        def update_role(self, **kwargs):
            self.events.append("set-duration")
            self.duration = kwargs["MaxSessionDuration"]

    iam = IAM()
    inert = setup._inert_trust()
    arn = setup._ensure_role(
        iam,
        account_id=ACCOUNT,
        name=role_name,
        trust=inert,
        expected_inline_policies={"ExpectedPolicy"},
        expected_attached_policies=expected_attached,
        max_session_duration=43200,
    )
    assert arn == f"arn:aws:iam::{ACCOUNT}:role/{role_name}"
    assert iam.events == [
        "get-role", "set-inert", "set-duration", "get-role"
    ]
    assert iam.trust == inert
    assert iam.duration == 43200


def test_owned_role_with_unexpected_policy_fails_before_trust_mutation():
    class IAM:
        def get_role(self, **kwargs):
            return {"Role": {
                "Arn": (
                    f"arn:aws:iam::{ACCOUNT}:role/{setup.CONTROLLER_ROLE}"
                ),
                "Path": "/",
            }}

        def list_role_tags(self, **kwargs):
            return {"Tags": [{
                "Key": "leadpoet:purpose",
                "Value": "production-parity",
            }]}

        def list_role_policies(self, **kwargs):
            return {"PolicyNames": ["UnexpectedPolicy"]}

        def list_attached_role_policies(self, **kwargs):
            return {"AttachedPolicies": []}

        def update_assume_role_policy(self, **kwargs):
            pytest.fail("unexpected-policy role trust was mutated")

    with pytest.raises(setup.SetupError, match="policy inventory differs"):
        setup._ensure_role(
            IAM(),
            account_id=ACCOUNT,
            name=setup.CONTROLLER_ROLE,
            trust=setup._inert_trust(),
            expected_inline_policies={"ExpectedPolicy"},
            expected_attached_policies=set(),
        )


def test_gateway_iam_cache_rejects_session_key_without_token(
    monkeypatch, tmp_path: Path
):
    cache = tmp_path / "gateway.env"
    cache.write_text(
        "AWS_ACCESS_KEY_ID=" + "AS" + "IA" + "A" * 16 + "\n"
        "AWS_SECRET_ACCESS_KEY=" + "x" * 40 + "\n",
        encoding="utf-8",
    )
    cache.chmod(0o600)
    monkeypatch.setattr(setup, "GATEWAY_IAM_CACHE", cache)
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(setup.SetupError, match="credentials are invalid"):
        setup._gateway_iam_session()


class _StaticSecrets:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.tags: dict[str, list[dict[str, str]]] = {}
        self.creates = 0
        self.policy_reads = 0
        self.descriptions: dict[str, str] = {}

    def get_secret_value(self, *, SecretId):
        if SecretId not in self.values:
            raise _not_found()
        return {"SecretString": self.values[SecretId]}

    def describe_secret(self, *, SecretId):
        return {
            "Name": SecretId,
            "ARN": f"arn:aws:secretsmanager:us-east-1:{ACCOUNT}:secret:{SecretId}-ABC123",
            "Description": self.descriptions[SecretId],
            "Tags": self.tags[SecretId],
            "RotationEnabled": False,
        }

    def get_resource_policy(self, *, SecretId):
        assert SecretId in self.values
        self.policy_reads += 1
        return {}

    def list_secret_version_ids(self, **kwargs):
        assert kwargs["SecretId"] in self.values
        assert kwargs["IncludeDeprecated"] is True
        return {
            "Versions": [{
                "VersionId": "version-one",
                "VersionStages": ["AWSCURRENT"],
            }]
        }

    def create_secret(self, **kwargs):
        assert kwargs["Name"] not in self.values
        self.creates += 1
        self.values[kwargs["Name"]] = kwargs["SecretString"]
        self.tags[kwargs["Name"]] = kwargs["Tags"]
        self.descriptions[kwargs["Name"]] = kwargs["Description"]


def test_static_installer_first_create_and_retry_are_immutable(monkeypatch):
    client = _StaticSecrets()
    monkeypatch.setattr(
        installer,
        "_instance_bootstrap_client",
        lambda **kwargs: (
            client,
            f"arn:aws:sts::{ACCOUNT}:assumed-role/{installer.VALIDATOR_ROLE}/i-1",
            f"arn:aws:sts::{ACCOUNT}:assumed-role/{installer.BOOTSTRAP_ROLE}/session",
        ),
    )
    monkeypatch.setattr(
        installer, "_builtwith_from_validator_container", lambda name: "builtwith-canary"
    )
    args = argparse.Namespace(
        commit=COMMIT,
        migration_sha256=MIGRATION_HASH,
        validator_container="leadpoet-validator-main",
        readonly_dsn_secret_id=installer.DEFAULT_READONLY_SECRET_ID,
        miner_intake_secret_id=installer.DEFAULT_MINER_INTAKE_SECRET_ID,
    )
    request = {
        "mode": "ensure",
        "migration_sha256": MIGRATION_HASH,
        "readonly_dsn": DSN,
    }
    first, first_secret = installer.install(args, request)
    retry, retry_secret = installer.install(args, request)
    assert first["status"] == retry["status"] == "installed"
    assert first_secret == retry_secret == {
        "readonly_dsn_available": True,
        "readonly_dsn": DSN,
    }
    assert client.creates == 2
    assert client.policy_reads >= 6


def test_static_installer_rejects_deprecated_or_alternate_versions(monkeypatch):
    client = _StaticSecrets()
    client.values[installer.DEFAULT_READONLY_SECRET_ID] = json.dumps(
        {"readonly_dsn": DSN}, sort_keys=True, separators=(",", ":")
    )
    client.tags[installer.DEFAULT_READONLY_SECRET_ID] = [
        {"Key": "leadpoet:purpose", "Value": "production-parity-static"},
        {"Key": "leadpoet:parity-static-bootstrap", "Value": "true"},
        {"Key": "leadpoet:candidate-sha", "Value": COMMIT},
    ]
    client.descriptions[installer.DEFAULT_READONLY_SECRET_ID] = (
        installer.STATIC_DESCRIPTIONS[installer.DEFAULT_READONLY_SECRET_ID]
    )
    monkeypatch.setattr(
        client,
        "list_secret_version_ids",
        lambda **kwargs: {
            "Versions": [
                {"VersionId": "current", "VersionStages": ["AWSCURRENT"]},
                {"VersionId": "old", "VersionStages": []},
            ]
        },
    )
    with pytest.raises(
        installer.StaticSecretInstallError,
        match="ownership differs",
    ):
        installer._static_secret_value(
            client, installer.DEFAULT_READONLY_SECRET_ID
        )


def test_installer_requires_three_distinct_pipe_descriptors(monkeypatch, capsys):
    receipt = {"status": "installed"}
    secret = {"readonly_dsn": DSN}
    monkeypatch.setattr(installer, "install", lambda args, request: (receipt, secret))
    request_r, request_w = os.pipe()
    receipt_r, receipt_w = os.pipe()
    secret_r, secret_w = os.pipe()
    try:
        os.write(request_w, b'{"mode":"probe"}')
        os.close(request_w)
        request_w = -1
        status = installer.main([
            "--commit", COMMIT,
            "--migration-sha256", MIGRATION_HASH,
            "--request-fd", str(request_r),
            "--receipt-fd", str(receipt_w),
            "--secret-response-fd", str(secret_w),
        ])
        os.close(receipt_w); receipt_w = -1
        os.close(secret_w); secret_w = -1
        assert status == 0
        assert json.loads(os.read(receipt_r, 65536)) == receipt
        assert json.loads(os.read(secret_r, 65536)) == secret
        captured = capsys.readouterr()
        assert PASSWORD not in captured.out
        assert PASSWORD not in captured.err
    finally:
        for descriptor in (request_r, request_w, receipt_r, receipt_w, secret_r, secret_w):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def test_installer_rejects_regular_file_descriptor(tmp_path: Path):
    path = tmp_path / "request"
    path.write_text("{}", encoding="utf-8")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(installer.StaticSecretInstallError, match="pipe or socket"):
            installer._read_request(descriptor)
    finally:
        os.close(descriptor)


def test_orchestrator_stages_secret_before_parameterized_bind_and_enables_last(monkeypatch):
    events: list[object] = []
    migration = b"-- exact idempotent migration\n"
    migration_hash = hashlib.sha256(migration).hexdigest()

    def fake_run(*args, **kwargs):
        events.append(("run", args))
        if args[:3] == ("git", "rev-parse", "HEAD"):
            return COMMIT.encode()
        return b""

    monkeypatch.setattr(bootstrap, "_run", fake_run)
    monkeypatch.setattr(
        bootstrap,
        "_committed_blob",
        lambda commit, path: migration if path == bootstrap.MIGRATION_PATH else b"source",
    )
    monkeypatch.setattr(bootstrap, "_disable_repository", lambda: events.append("disable"))

    def gateway(source, *, path, argv):
        events.append(("gateway", argv[0]))
        if argv[0] == "iam-only":
            return {
                "status": "iam_ready",
                "account_id": ACCOUNT,
                "github_variables_mutated": False,
                "static_bootstrap_trust_expires_at": (
                    datetime.now(timezone.utc) + timedelta(minutes=10)
                ).isoformat(),
            }
        return {"status": "static_bootstrap_authority_removed"}

    monkeypatch.setattr(bootstrap, "_gateway_command", gateway)
    monkeypatch.setattr(bootstrap, "_access_token", lambda: "management-token")

    def query(token, query, *, parameters=None):
        assert token == "management-token"
        if query == bootstrap.ROLE_STATE_QUERY:
            return [{
                "database_ready": True,
                "migration_role_ready": True,
                "reader_exists": True,
                "reader_can_login": False,
                "binder_ready": True,
                "contract_ready": True,
                "reader_contract": {
                    "schema_version": "leadpoet.production-parity-reader-contract.v1",
                    "database_name": "postgres",
                    "reader_role": "leadpoet_parity_reader",
                    "login_enabled": False,
                    "superuser": False,
                    "bypass_rls": True,
                    "createdb": False,
                    "createrole": False,
                    "inherit": False,
                    "replication": False,
                    "connection_limit": 2,
                    "default_read_only": True,
                    "membership_count": 0,
                    "schema_create_capable": False,
                    "table_write_capable": False,
                    "sequence_write_capable": False,
                },
            }]
        if query == bootstrap.BIND_QUERY:
            events.append(("bind", query, parameters))
            assert PASSWORD not in query
            assert parameters == [PASSWORD]
            return [{"result": {
                "status": "bound",
                "reader_role": "leadpoet_parity_reader",
                "login_enabled": True,
            }}]
        events.append("migration")
        return []

    monkeypatch.setattr(bootstrap, "_management_query", query)
    monkeypatch.setattr(bootstrap.secrets, "token_hex", lambda size: PASSWORD)
    remote_calls = 0

    def validator(source, *, commit, migration_sha256, request):
        nonlocal remote_calls
        remote_calls += 1
        events.append(("validator", request["mode"]))
        receipt = {
            "schema_version": "leadpoet.production_parity_static_bootstrap.v1",
            "status": "installed" if request["mode"] == "ensure" else "probed",
        }
        if request["mode"] == "probe":
            return receipt, {"readonly_dsn_available": False, "readonly_dsn": ""}
        return receipt, {"readonly_dsn_available": True, "readonly_dsn": request["readonly_dsn"]}

    monkeypatch.setattr(bootstrap, "_validator_command", validator)
    monkeypatch.setattr(
        bootstrap.setup, "_verify_readonly_dsn", lambda dsn: events.append("verify")
    )

    def configure(args):
        receipt = json.loads(os.read(args.receipt_fd, 65536))
        assert receipt["reader_default_read_only_verified"] is True
        events.append("configure")
        return {"enabled": True}

    monkeypatch.setattr(bootstrap.setup, "configure_repository", configure)
    result = bootstrap.bootstrap(commit=COMMIT, migration_sha256=migration_hash)
    assert result["status"] == "commissioned"
    assert remote_calls == 2
    assert events.index(("validator", "ensure")) < next(
        index for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "bind"
    )
    assert events.index(("gateway", "cleanup-bootstrap")) < events.index("configure")
    assert events[0][0] == "run"
    assert events.count("disable") == 1


def test_malformed_iam_receipt_still_cleans_bounded_role_and_stays_disabled(
    monkeypatch,
):
    migration = b"-- exact idempotent migration\n"
    migration_hash = hashlib.sha256(migration).hexdigest()
    events: list[str] = []

    def fake_run(*args, **kwargs):
        if args[:3] == ("git", "rev-parse", "HEAD"):
            return COMMIT.encode()
        return b""

    monkeypatch.setattr(bootstrap, "_run", fake_run)
    monkeypatch.setattr(
        bootstrap,
        "_committed_blob",
        lambda commit, path: (
            migration if path == bootstrap.MIGRATION_PATH else b"source"
        ),
    )
    monkeypatch.setattr(
        bootstrap, "_disable_repository", lambda: events.append("disable")
    )

    def gateway(source, *, path, argv):
        events.append(argv[0])
        if argv[0] == "iam-only":
            return {"status": "malformed-after-role-create"}
        return {"status": "static_bootstrap_authority_removed"}

    monkeypatch.setattr(bootstrap, "_gateway_command", gateway)
    with pytest.raises(bootstrap.BootstrapError, match="IAM-only receipt differs"):
        bootstrap.bootstrap(commit=COMMIT, migration_sha256=migration_hash)
    assert events == ["disable", "iam-only", "cleanup-bootstrap", "disable"]


def test_lost_iam_receipt_still_cleans_ambiguous_remote_role_and_disables(
    monkeypatch,
):
    migration = b"-- exact idempotent migration\n"
    migration_hash = hashlib.sha256(migration).hexdigest()
    events: list[str] = []

    def fake_run(*args, **kwargs):
        if args[:3] == ("git", "rev-parse", "HEAD"):
            return COMMIT.encode()
        return b""

    monkeypatch.setattr(bootstrap, "_run", fake_run)
    monkeypatch.setattr(
        bootstrap,
        "_committed_blob",
        lambda commit, path: (
            migration if path == bootstrap.MIGRATION_PATH else b"source"
        ),
    )
    monkeypatch.setattr(
        bootstrap, "_disable_repository", lambda: events.append("disable")
    )

    def gateway(source, *, path, argv):
        events.append(argv[0])
        if argv[0] == "iam-only":
            raise bootstrap.BootstrapError("IAM receipt transport was lost")
        return {"status": "static_bootstrap_authority_removed"}

    monkeypatch.setattr(bootstrap, "_gateway_command", gateway)
    with pytest.raises(
        bootstrap.BootstrapError,
        match="receipt transport was lost",
    ):
        bootstrap.bootstrap(commit=COMMIT, migration_sha256=migration_hash)
    assert events == ["disable", "iam-only", "cleanup-bootstrap", "disable"]


def test_orchestrator_rejects_stale_checkout_before_any_commission_write(
    monkeypatch,
):
    calls: list[tuple[str, ...]] = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        if args[:3] == ("git", "rev-parse", "HEAD"):
            return ("f" * 40).encode()
        return b""

    monkeypatch.setattr(bootstrap, "_run", fake_run)
    monkeypatch.setattr(
        bootstrap,
        "_disable_repository",
        lambda: pytest.fail("commissioning write preceded exact-main proof"),
    )
    with pytest.raises(
        bootstrap.BootstrapError,
        match="current checkout HEAD",
    ):
        bootstrap.bootstrap(commit=COMMIT, migration_sha256=MIGRATION_HASH)
    assert calls[0][:3] == ("git", "fetch", "--no-tags")


def test_orchestrator_rejects_login_without_recoverable_static_dsn(
    monkeypatch,
):
    migration = b"-- exact idempotent migration\n"
    migration_hash = hashlib.sha256(migration).hexdigest()
    events: list[str] = []

    def fake_run(*args, **kwargs):
        if args[:3] == ("git", "rev-parse", "HEAD"):
            return COMMIT.encode()
        return b""

    monkeypatch.setattr(bootstrap, "_run", fake_run)
    monkeypatch.setattr(
        bootstrap,
        "_committed_blob",
        lambda commit, path: (
            migration if path == bootstrap.MIGRATION_PATH else b"source"
        ),
    )
    monkeypatch.setattr(
        bootstrap, "_disable_repository", lambda: events.append("disable")
    )

    def gateway(source, *, path, argv):
        events.append(argv[0])
        if argv[0] == "iam-only":
            return {
                "status": "iam_ready",
                "account_id": ACCOUNT,
                "github_variables_mutated": False,
                "static_bootstrap_trust_expires_at": (
                    datetime.now(timezone.utc) + timedelta(minutes=10)
                ).isoformat(),
            }
        return {"status": "static_bootstrap_authority_removed"}

    monkeypatch.setattr(bootstrap, "_gateway_command", gateway)
    monkeypatch.setattr(bootstrap, "_access_token", lambda: "management-token")
    monkeypatch.setattr(
        bootstrap, "_role_state", lambda token: {"reader_can_login": True}
    )
    monkeypatch.setattr(
        bootstrap,
        "_validator_command",
        lambda *args, **kwargs: (
            {"status": "probed"},
            {"readonly_dsn_available": False, "readonly_dsn": ""},
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "_management_query",
        lambda *args, **kwargs: pytest.fail("unsafe reader was rebound"),
    )
    with pytest.raises(
        bootstrap.BootstrapError,
        match="LOGIN has no recoverable static DSN",
    ):
        bootstrap.bootstrap(commit=COMMIT, migration_sha256=migration_hash)
    assert events[-2:] == ["cleanup-bootstrap", "disable"]


def test_migration_is_clone_safe_and_rechecks_complete_contract():
    source = (Path(__file__).parents[1] / bootstrap.MIGRATION_PATH).read_text(
        encoding="utf-8"
    )
    assert "current_database() <> 'postgres'" not in source
    assert "ALTER ROLE leadpoet_parity_reader NOLOGIN" in source
    assert "REFERENCES,TRIGGER" in source
    assert "has_sequence_privilege" in source
    assert "default_transaction_read_only=on" in source
    assert "pg_auth_members" in source
    assert "EXCEPTION WHEN OTHERS" in source
    assert PASSWORD not in source


def test_full_workflow_uses_self_hosted_bounded_windows_and_exact_volume():
    source = (
        Path(__file__).parents[1]
        / ".github/workflows/physical-v2-staging.yml"
    ).read_text(encoding="utf-8")
    assert "leadpoet-gateway-v2-builder" in source
    assert "runs-on: ubuntu-latest" not in source
    assert 'FULL_TIMEOUT_SECONDS: "72000"' in source
    assert 'SSM_TIMEOUT_SECONDS: "77400"' in source
    assert '${PARITY_VOLUME_GIB:-' not in source
    assert 'test "$PARITY_VOLUME_GIB" = "512"' in source
    assert source.count("unset-current-credentials: true") >= 7
    assert source.count("allowed-account-ids: \"493765492819\"") >= 7
    assert "--max-wait-seconds 16200" in source
    assert "--max-wait-seconds 12600" in source
    assert "Refresh parity AWS role for cleanup" in source
    assert "id: provenance" in source
    assert "echo 'verified=true' >> \"$GITHUB_OUTPUT\"" in source
    assert source.count(
        "always() && steps.provenance.outputs.verified == 'true'"
    ) == 2
    assert "id: evidence_path" in source
    assert (
        "always() && steps.evidence_path.outputs.verified == 'true'" in source
    )


def test_controller_dependencies_use_a_scrubbed_per_run_virtualenv():
    root = Path(__file__).parents[1]
    action = (
        root / ".github/actions/setup-production-parity-controller/action.yml"
    ).read_text(encoding="utf-8")
    full = (
        root / ".github/workflows/physical-v2-staging.yml"
    ).read_text(encoding="utf-8")

    assert 'controller_root="$PARITY_TEMP/controller"' in action
    assert (
        'controller_root="$RUNNER_TEMP/production-parity-controller-'
        '$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"' in action
    )
    assert 'python3 -m venv "$venv_root"' in action
    assert '"$venv_python" -m pip install' in action
    assert "--no-cache-dir" in action
    assert "python3 -m pip install" not in action
    assert "cache: pip" not in action
    assert 'include-system-site-packages = false' in action
    assert 'printf \'%s\\n\' "$venv_root/bin" >> "$GITHUB_PATH"' in action
    assert 'printf \'VIRTUAL_ENV=%s\\n\' "$venv_root"' in action
    assert 'test "$(command -v python3)" = "$VIRTUAL_ENV/bin/python3"' in action
    assert '"$venv_python" "$script" --help' in action
    assert 'PARITY_TEMP: ${{ runner.temp }}/production-parity-' in full
    assert 'rm -rf -- "$PARITY_TEMP"' in full


def test_fast_and_cleanup_pin_account_and_reject_stale_cleanup_code():
    root = Path(__file__).parents[1]
    fast = (root / ".github/workflows/production-parity-fast.yml").read_text(
        encoding="utf-8"
    )
    cleanup = (
        root / ".github/workflows/production-parity-cleanup.yml"
    ).read_text(encoding="utf-8")
    for source in (fast, cleanup):
        assert 'allowed-account-ids: "493765492819"' in source
        assert "unset-current-credentials: true" in source
    cleanup_gate = cleanup.index("name: Require exact current main before credentials")
    cleanup_action = cleanup.index(
        "uses: ./.github/actions/setup-production-parity-controller"
    )
    cleanup_credentials = cleanup.index(
        "uses: aws-actions/configure-aws-credentials@v4"
    )
    assert "git rev-parse origin/main" in cleanup
    assert cleanup_gate < cleanup_action < cleanup_credentials


def test_configure_requires_exact_parity_variable_inventory():
    source = (
        Path(__file__).parents[1]
        / "scripts/setup_production_parity_staging.py"
    ).read_text(encoding="utf-8")
    assert "actual_parity_names != expected_parity_names" in source
    assert "GitHub parity variable inventory differs" in source
