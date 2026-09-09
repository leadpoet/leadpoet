from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts" / "configure_lab_arena_production.py"
SPEC = importlib.util.spec_from_file_location("configure_lab_arena_guard", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

SIGNING_KEY = (
    "arn:aws:kms:us-east-1:493765492819:key/"
    "864c6f6d-fbff-44c4-9001-e8adc0e0a75a"
)
BEFORE = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["kms:Sign", "kms:GetPublicKey", "kms:DescribeKey"],
            "Resource": SIGNING_KEY,
        },
        {
            "Effect": "Allow",
            "Action": [
                "kms:GenerateDataKey",
                "kms:Encrypt",
                "kms:Decrypt",
                "kms:DescribeKey",
            ],
            "Resource": MODULE.LAB_ARENA_CREDENTIAL_KMS_KEY_ARN,
        },
    ],
}


class FakeSts:
    def __init__(self, *, account=MODULE.IAM_ACCOUNT, arn=MODULE.IAM_CALLER_ARN):
        self.account = account
        self.arn = arn

    def get_caller_identity(self):
        return {"Account": self.account, "Arn": self.arn}


class FakeIam:
    def __init__(self, document=BEFORE):
        self.document = json.loads(json.dumps(document))
        self.puts = []
        self.reads = 0
        self.drift_on_second_read = False
        self.corrupt_after_put = False

    def get_role(self, *, RoleName):
        assert RoleName == MODULE.VALIDATOR_IAM_ROLE
        return {"Role": {"Arn": MODULE.VALIDATOR_IAM_ROLE_ARN}}

    def list_role_policies(self, **kwargs):
        assert kwargs == {"RoleName": MODULE.VALIDATOR_IAM_ROLE}
        return {"PolicyNames": [MODULE.VALIDATOR_IAM_POLICY]}

    def get_role_policy(self, **kwargs):
        assert kwargs == {
            "RoleName": MODULE.VALIDATOR_IAM_ROLE,
            "PolicyName": MODULE.VALIDATOR_IAM_POLICY,
        }
        self.reads += 1
        if self.drift_on_second_read and self.reads == 2:
            self.document["Statement"][0]["Action"].append("kms:Verify")
        if self.corrupt_after_put and self.puts:
            self.document["Statement"][-1]["Action"] = ["kms:Decrypt"]
        return {"PolicyDocument": json.loads(json.dumps(self.document))}

    def put_role_policy(self, **kwargs):
        self.puts.append(kwargs)
        self.document = json.loads(kwargs["PolicyDocument"])


class FakeSession:
    def __init__(self, iam, sts=None):
        self.iam = iam
        self.sts = sts or FakeSts()

    def client(self, name):
        return {"sts": self.sts, "iam": self.iam}[name]


def _install(monkeypatch, iam, sts=None):
    monkeypatch.setattr(MODULE, "_gateway_iam_session", lambda: FakeSession(iam, sts))


def test_read_only_plan_adds_only_exact_deny_and_preserves_existing_statements(
    monkeypatch,
):
    iam = FakeIam()
    _install(monkeypatch, iam)

    result = MODULE.configure_validator_credential_kms_guard(apply=False)

    desired, changed = MODULE._policy_with_credential_deny(BEFORE)
    assert changed is True
    assert desired["Statement"][:-1] == BEFORE["Statement"]
    assert desired["Statement"][-1] == MODULE._credential_deny_statement()
    assert desired["Statement"][-1]["Condition"] == {
        "StringEquals": {
            "kms:EncryptionContext:purpose": (
                "leadpoet_lab_arena_miner_runtime_credential"
            )
        }
    }
    assert result["changed"] is True and result["applied"] is False
    assert iam.puts == []


def test_apply_rechecks_before_write_and_requires_exact_post_write(monkeypatch):
    iam = FakeIam()
    _install(monkeypatch, iam)

    result = MODULE.configure_validator_credential_kms_guard(apply=True)

    assert result["applied"] is True
    assert result["readback_document_hash"] == result["desired_document_hash"]
    assert len(iam.puts) == 1
    assert iam.puts[0]["RoleName"] == MODULE.VALIDATOR_IAM_ROLE
    assert iam.puts[0]["PolicyName"] == MODULE.VALIDATOR_IAM_POLICY
    assert iam.document["Statement"][:-1] == BEFORE["Statement"]


def test_apply_rejects_prewrite_drift_without_overwrite(monkeypatch):
    iam = FakeIam()
    iam.drift_on_second_read = True
    _install(monkeypatch, iam)

    with pytest.raises(MODULE.ConfigurationError, match="changed before write"):
        MODULE.configure_validator_credential_kms_guard(apply=True)

    assert iam.puts == []


def test_apply_rejects_nonexact_post_write_readback(monkeypatch):
    iam = FakeIam()
    iam.corrupt_after_put = True
    _install(monkeypatch, iam)

    with pytest.raises(MODULE.ConfigurationError, match="post-write differs"):
        MODULE.configure_validator_credential_kms_guard(apply=True)

    assert len(iam.puts) == 1


def test_exact_existing_deny_is_idempotent(monkeypatch):
    guarded, _ = MODULE._policy_with_credential_deny(BEFORE)
    iam = FakeIam(guarded)
    _install(monkeypatch, iam)

    result = MODULE.configure_validator_credential_kms_guard(apply=True)

    assert result["changed"] is False and result["applied"] is False
    assert iam.puts == []


@pytest.mark.parametrize(
    ("account", "arn"),
    (("111111111111", MODULE.IAM_CALLER_ARN), (MODULE.IAM_ACCOUNT, "arn:aws:iam::493765492819:user/other")),
)
def test_exact_operator_identity_is_required(monkeypatch, account, arn):
    iam = FakeIam()
    _install(monkeypatch, iam, FakeSts(account=account, arn=arn))

    with pytest.raises(MODULE.ConfigurationError, match="caller identity differs"):
        MODULE.configure_validator_credential_kms_guard(apply=False)

    assert iam.reads == 0 and iam.puts == []


def test_aws_failure_is_reported_without_provider_detail(monkeypatch):
    class FailedSts:
        def get_caller_identity(self):
            raise RuntimeError("synthetic-sensitive-provider-detail")

    iam = FakeIam()
    _install(monkeypatch, iam, FailedSts())

    with pytest.raises(MODULE.ConfigurationError) as failure:
        MODULE.configure_validator_credential_kms_guard(apply=False)

    assert str(failure.value) == "validator IAM guard operation failed"
    assert "synthetic-sensitive" not in str(failure.value)


def test_wrong_existing_guard_is_rejected(monkeypatch):
    guarded, _ = MODULE._policy_with_credential_deny(BEFORE)
    guarded["Statement"][-1]["Condition"]["StringEquals"][
        "kms:EncryptionContext:purpose"
    ] = "unrelated-purpose"
    iam = FakeIam(guarded)
    _install(monkeypatch, iam)

    with pytest.raises(MODULE.ConfigurationError, match="deny differs"):
        MODULE.configure_validator_credential_kms_guard(apply=False)

    assert iam.puts == []


@pytest.mark.parametrize("wrong_inventory", ("role", "policy"))
def test_exact_validator_role_and_inline_policy_inventory_are_required(
    monkeypatch, wrong_inventory
):
    iam = FakeIam()
    if wrong_inventory == "role":
        iam.get_role = lambda **_kwargs: {
            "Role": {"Arn": "arn:aws:iam::493765492819:role/other"}
        }
    else:
        iam.list_role_policies = lambda **_kwargs: {"PolicyNames": ["other"]}
    _install(monkeypatch, iam)

    with pytest.raises(MODULE.ConfigurationError, match="identity differs"):
        MODULE.configure_validator_credential_kms_guard(apply=False)

    assert iam.puts == []


def test_cli_defaults_to_plan_and_apply_uses_existing_arena_authorization(
    monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(
        MODULE,
        "configure_validator_credential_kms_guard",
        lambda *, apply: calls.append(apply) or {"ok": True, "applied": apply},
    )
    assert MODULE.main(
        ["--validator-credential-kms-guard", "--allowed-account", MODULE.IAM_ACCOUNT]
    ) == 0
    assert calls == [False]
    assert json.loads(capsys.readouterr().out)["applied"] is False

    monkeypatch.setenv(MODULE.AUTH_ENV, "1")
    assert MODULE.main(
        [
            "--validator-credential-kms-guard",
            "--apply",
            "--allowed-account",
            MODULE.IAM_ACCOUNT,
        ]
    ) == 0
    assert calls == [False, True]
    capsys.readouterr()

    monkeypatch.delenv(MODULE.AUTH_ENV, raising=False)
    assert MODULE.main(
        [
            "--validator-credential-kms-guard",
            "--apply",
            "--allowed-account",
            MODULE.IAM_ACCOUNT,
        ]
    ) == 2
    assert calls == [False, True]
