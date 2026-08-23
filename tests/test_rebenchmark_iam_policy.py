from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_production_parity_contract as contract_builder
from scripts import operate_rebenchmark_iam_policy as operator


ACCOUNT = operator.EXPECTED_ACCOUNT_ID
CALLER = operator.EXPECTED_CALLER_ARN
COMMIT = "a" * 40
SOURCE_HASH = "sha256:" + "b" * 64
ROLE = "leadpoet-test-role"
POLICY_NAME = "LeadpoetTestPolicy"
MANAGED_ARN = f"arn:aws:iam::{ACCOUNT}:policy/leadpoet/{POLICY_NAME}"
RESOURCE = f"arn:aws:s3:::leadpoet-test-{ACCOUNT}/*"


def _policy(*actions: str) -> dict[str, object]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "LeadpoetTest",
                "Effect": "Allow",
                "Action": list(actions),
                "Resource": RESOURCE,
            }
        ],
    }


BEFORE = _policy("s3:GetObject")
AFTER = _policy("s3:GetObject", "s3:PutObject")
THIRD = _policy("s3:DeleteObject")


def _simulation() -> list[dict[str, object]]:
    return [
        {
            "name": "leadpoet-positive",
            "action": "s3:PutObject",
            "resources": [RESOURCE],
            "context": {},
            "expected": "allowed",
        }
    ]


def _request(
    state: dict[str, object],
    *,
    target: dict[str, str],
    desired: dict[str, object] = AFTER,
    prune: dict[str, str] | None = None,
) -> dict[str, object]:
    canonical_desired = operator._canonical_policy(desired)
    delta = operator._policy_delta(state["document"], canonical_desired)
    return {
        "schema_version": operator.REQUEST_SCHEMA,
        "change_id": "test-change",
        "target": target,
        "expected_prior_document_hash": state["document_hash"],
        "expected_inventory_hash": state["inventory_hash"],
        "desired_document": desired,
        "expected_delta": delta,
        "simulations": _simulation(),
        "prune_managed_version": prune,
    }


class FakeSts:
    def __init__(self, arn: str = CALLER, account: str = ACCOUNT):
        self.arn = arn
        self.account = account

    def get_caller_identity(self):
        return {"Account": self.account, "Arn": self.arn, "UserId": "redacted"}


class FakeInlineIam:
    def __init__(self, *, existing: bool = True):
        self.document = operator._canonical_policy(BEFORE)
        self.existing = existing
        self.events: list[tuple[str, str]] = []
        self.simulation_calls = 0
        self.mutate_after_first_simulation = False
        self.fail_post_simulation = False
        self.third_state_on_post_simulation = False

    def get_role(self, *, RoleName):
        assert RoleName == ROLE
        return {
            "Role": {
                "Arn": f"arn:aws:iam::{ACCOUNT}:role/{ROLE}",
                "Path": "/",
                "MaxSessionDuration": 3600,
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "ec2.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                },
            }
        }

    def list_role_policies(self, **kwargs):
        assert kwargs["RoleName"] == ROLE
        return {"PolicyNames": [POLICY_NAME] if self.existing else []}

    def list_attached_role_policies(self, **kwargs):
        return {"AttachedPolicies": []}

    def list_instance_profiles_for_role(self, **kwargs):
        return {"InstanceProfiles": []}

    def list_role_tags(self, **kwargs):
        return {"Tags": [{"Key": "leadpoet:purpose", "Value": "test"}]}

    def get_role_policy(self, **kwargs):
        assert kwargs == {"RoleName": ROLE, "PolicyName": POLICY_NAME}
        assert self.existing
        return {"PolicyDocument": self.document}

    def put_role_policy(self, **kwargs):
        assert kwargs["RoleName"] == ROLE
        assert kwargs["PolicyName"] == POLICY_NAME
        self.document = json.loads(kwargs["PolicyDocument"])
        self.existing = True
        self.events.append(("put", operator._policy_hash(self.document)))

    def delete_role_policy(self, **kwargs):
        assert kwargs == {"RoleName": ROLE, "PolicyName": POLICY_NAME}
        assert self.existing
        self.existing = False
        self.events.append(("delete", operator._policy_hash(self.document)))

    def simulate_custom_policy(self, **kwargs):
        self.simulation_calls += 1
        if self.mutate_after_first_simulation and self.simulation_calls == 1:
            self.document = operator._canonical_policy(THIRD)
        decision = "allowed"
        if self.fail_post_simulation and self.simulation_calls == 2:
            decision = "implicitDeny"
            if self.third_state_on_post_simulation:
                self.document = operator._canonical_policy(THIRD)
        action = kwargs["ActionNames"][0]
        return {
            "EvaluationResults": [
                {
                    "EvalActionName": action,
                    "EvalResourceName": RESOURCE,
                    "EvalDecision": decision,
                    "MissingContextValues": [],
                }
            ]
        }


class FakeManagedIam:
    def __init__(self, *, version_count: int = 1):
        self.versions: dict[str, dict[str, object]] = {
            f"v{index}": operator._canonical_policy(BEFORE)
            for index in range(1, version_count + 1)
        }
        self.default = f"v{version_count}"
        self.events: list[tuple[str, str]] = []
        self.simulation_calls = 0
        self.fail_post_simulation = False

    def get_policy(self, *, PolicyArn):
        assert PolicyArn == MANAGED_ARN
        return {
            "Policy": {
                "Arn": MANAGED_ARN,
                "PolicyName": POLICY_NAME,
                "Path": "/leadpoet/",
                "Description": "Leadpoet test policy",
                "DefaultVersionId": self.default,
            }
        }

    def list_policy_versions(self, *, PolicyArn):
        assert PolicyArn == MANAGED_ARN
        return {
            "Versions": [
                {"VersionId": version, "IsDefaultVersion": version == self.default}
                for version in sorted(self.versions)
            ]
        }

    def get_policy_version(self, *, PolicyArn, VersionId):
        assert PolicyArn == MANAGED_ARN
        return {"PolicyVersion": {"Document": self.versions[VersionId]}}

    def list_policy_tags(self, **kwargs):
        return {"Tags": [{"Key": "leadpoet:purpose", "Value": "test"}]}

    def list_entities_for_policy(self, **kwargs):
        return {
            "PolicyRoles": [{"RoleName": ROLE}],
            "PolicyUsers": [],
            "PolicyGroups": [],
        }

    def simulate_custom_policy(self, **kwargs):
        self.simulation_calls += 1
        decision = (
            "implicitDeny"
            if self.fail_post_simulation and self.simulation_calls == 2
            else "allowed"
        )
        return {
            "EvaluationResults": [
                {
                    "EvalActionName": kwargs["ActionNames"][0],
                    "EvalResourceName": RESOURCE,
                    "EvalDecision": decision,
                    "MissingContextValues": [],
                }
            ]
        }

    def create_policy_version(self, **kwargs):
        assert kwargs["PolicyArn"] == MANAGED_ARN
        assert kwargs["SetAsDefault"] is False
        next_id = max(int(value[1:]) for value in self.versions) + 1
        version = f"v{next_id}"
        self.versions[version] = json.loads(kwargs["PolicyDocument"])
        self.events.append(("create", version))
        return {"PolicyVersion": {"VersionId": version}}

    def set_default_policy_version(self, *, PolicyArn, VersionId):
        assert PolicyArn == MANAGED_ARN
        assert VersionId in self.versions
        self.default = VersionId
        self.events.append(("default", VersionId))

    def delete_policy_version(self, *, PolicyArn, VersionId):
        assert PolicyArn == MANAGED_ARN
        assert VersionId != self.default
        del self.versions[VersionId]
        self.events.append(("delete", VersionId))


def _apply_inline(iam: FakeInlineIam, request: dict[str, object]):
    return operator._apply_inline(
        iam,
        operator._validate_request(request),
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
    )


def _apply_managed(iam: FakeManagedIam, request: dict[str, object]):
    return operator._apply_managed(
        iam,
        operator._validate_request(request),
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
    )


def test_policy_validation_rejects_trust_material_and_unrelated_targets():
    invalid = _policy("s3:GetObject")
    invalid["Statement"][0]["Principal"] = {"AWS": "*"}
    with pytest.raises(operator.OperationError, match="identity-policy"):
        operator._canonical_policy(invalid)

    with pytest.raises(operator.OperationError, match="outside Leadpoet scope"):
        operator._target(
            {"kind": "inline_role", "role_name": "other-role", "policy_name": POLICY_NAME}
        )
    assert operator._target(
        {
            "kind": "inline_role",
            "role_name": "leadpoet-gateway-s3-cloudwatch-role",
            "policy_name": "leadpoet-gateway-env-secretsmanager",
        }
    )
    with pytest.raises(operator.OperationError, match="outside Leadpoet scope"):
        operator._target(
            {
                "kind": "managed",
                "policy_arn": "arn:aws:iam::111111111111:policy/LeadpoetOther",
            }
        )


def test_structural_delta_contains_only_hashes_and_statement_paths():
    before = operator._canonical_policy(BEFORE)
    after = operator._canonical_policy(AFTER)
    delta = operator._structural_delta(before, after)
    assert delta
    assert all(item["path"].startswith("/Statement/") for item in delta)
    serialized = json.dumps(delta)
    assert "s3:GetObject" not in serialized
    assert "s3:PutObject" not in serialized
    assert RESOURCE not in serialized
    assert operator._validate_expected_delta(delta) == delta


def test_gateway_identity_rejects_same_account_principal_drift():
    setup = {
        "_iam_clients": lambda region: (
            FakeSts(f"arn:aws:iam::{ACCOUNT}:user/other"),
            object(),
            ACCOUNT,
        )
    }
    with pytest.raises(operator.OperationError, match="principal differs"):
        operator._gateway_clients(setup)


def test_inline_update_has_exact_readback_and_redacted_receipt():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(operator._inline_state(iam, target), target=target)
    receipt = _apply_inline(iam, request)

    assert receipt["status"] == "updated"
    assert receipt["concurrency_model"] == operator.CONCURRENCY_MODEL
    assert receipt["aws_native_compare_and_swap"] is False
    assert receipt["secret_values_printed"] is False
    assert receipt["policy_material_printed"] is False
    assert receipt["readback_document_hash"] == operator._policy_hash(AFTER)
    assert iam.simulation_calls == 2
    assert len(iam.events) == 1
    serialized = json.dumps(receipt)
    assert "s3:PutObject" not in serialized
    assert RESOURCE not in serialized


def test_inline_stale_prior_hash_fails_before_write():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(operator._inline_state(iam, target), target=target)
    request["expected_prior_document_hash"] = "sha256:" + "0" * 64
    with pytest.raises(operator.OperationError, match="prior document hash differs"):
        _apply_inline(iam, request)
    assert iam.events == []


def test_inline_concurrent_prewrite_change_is_not_overwritten():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(operator._inline_state(iam, target), target=target)
    iam.mutate_after_first_simulation = True
    with pytest.raises(operator.OperationError, match="changed before write"):
        _apply_inline(iam, request)
    assert iam.events == []
    assert operator._policy_hash(iam.document) == operator._policy_hash(THIRD)


def test_inline_postwrite_simulation_failure_rolls_back_only_intended_state():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(operator._inline_state(iam, target), target=target)
    iam.fail_post_simulation = True
    with pytest.raises(operator.OperationError, match="simulation rolled back"):
        _apply_inline(iam, request)
    assert operator._policy_hash(iam.document) == operator._policy_hash(BEFORE)
    assert len(iam.events) == 2


def test_inline_create_rolls_back_to_absent_without_touching_surrounding_state():
    iam = FakeInlineIam(existing=False)
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(operator._inline_state(iam, target), target=target)
    assert all(item["path"].startswith("/Statement/") for item in request["expected_delta"])
    iam.fail_post_simulation = True

    with pytest.raises(operator.OperationError, match="simulation rolled back"):
        _apply_inline(iam, request)

    assert iam.existing is False
    assert [event[0] for event in iam.events] == ["put", "delete"]


def test_inline_postwrite_third_state_is_never_rolled_back():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(operator._inline_state(iam, target), target=target)
    iam.fail_post_simulation = True
    iam.third_state_on_post_simulation = True
    with pytest.raises(operator.OperationError, match="third state"):
        _apply_inline(iam, request)
    assert operator._policy_hash(iam.document) == operator._policy_hash(THIRD)
    assert len(iam.events) == 1


def test_managed_update_keeps_prior_default_as_rollback_version():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _request(operator._managed_state(iam, target), target=target)
    receipt = _apply_managed(iam, request)

    assert receipt["status"] == "updated"
    assert receipt["managed_version_id"] == "v2"
    assert receipt["pruned_managed_version_id"] is None
    assert iam.default == "v2"
    assert set(iam.versions) == {"v1", "v2"}
    assert operator._policy_hash(iam.versions["v1"]) == operator._policy_hash(BEFORE)
    assert operator._policy_hash(iam.versions["v2"]) == operator._policy_hash(AFTER)


def test_managed_capacity_fails_before_irreversible_prune_or_write():
    iam = FakeManagedIam(version_count=5)
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    state = operator._managed_state(iam, target)
    request = _request(state, target=target)

    with pytest.raises(operator.OperationError, match="preserve an exact rollback"):
        _apply_managed(iam, request)
    assert iam.default == "v5"
    assert len(iam.versions) == 5
    assert iam.events == []


def test_managed_postwrite_simulation_failure_restores_prior_default():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _request(operator._managed_state(iam, target), target=target)
    iam.fail_post_simulation = True
    with pytest.raises(operator.OperationError, match="simulation rolled back"):
        _apply_managed(iam, request)
    assert iam.default == "v1"
    assert set(iam.versions) == {"v1"}


def test_local_command_environment_drops_every_aws_selector(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout=b"ok", stderr=b"")

    for name in operator._AWS_SELECTORS:
        monkeypatch.setenv(name, "wrong-local-authority")
    monkeypatch.setattr(operator.subprocess, "run", fake_run)
    assert operator._run("test-command") == b"ok"
    assert not (set(captured["env"]) & operator._AWS_SELECTORS)
    assert 'os.environ.pop(name, None)' in operator.REMOTE_LOADER


def test_remote_receipt_validation_rejects_any_unexpected_output_field():
    authority = operator._remote_entry(
        "probe",
        None,
        {"_iam_clients": lambda region: (FakeSts(), object(), ACCOUNT)},
        source_hash=SOURCE_HASH,
        commit=COMMIT,
    )
    authority["unexpected"] = "must-not-reach-stdout"

    with pytest.raises(operator.OperationError, match="authority receipt differs"):
        operator._validate_remote_receipt(
            "probe",
            authority,
            request=None,
            commit=COMMIT,
            source_hash=SOURCE_HASH,
        )


def test_policy_receipt_is_bound_to_exact_normalized_request():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = operator._validate_request(
        _request(operator._inline_state(iam, target), target=target)
    )
    receipt = operator._apply_inline(
        iam,
        request,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
    )

    assert operator._validate_remote_receipt(
        "apply",
        receipt,
        request=request,
        commit=COMMIT,
        source_hash=SOURCE_HASH,
    ) == receipt
    receipt["change_id"] = "different-change"
    with pytest.raises(operator.OperationError, match="policy receipt differs"):
        operator._validate_remote_receipt(
            "apply",
            receipt,
            request=request,
            commit=COMMIT,
            source_hash=SOURCE_HASH,
        )


def test_typed_active_ledger_is_required_for_apply(tmp_path: Path):
    ledger = tmp_path / "ledger.json"
    value = {
        "schema_version": operator.LEDGER_SCHEMA,
        "status": "running",
        "iam_authority_route": {
            "schema_version": operator.AUTHORITY_SCHEMA,
            "status": "authority_ready",
            "origin_main_sha": COMMIT,
            "bridge_source_hash": SOURCE_HASH,
            "account_id": ACCOUNT,
            "caller_arn": CALLER,
            "route": "gateway_bridge",
            "local_chain": "ignored_non_authority",
            "secret_values_printed": False,
            "policy_material_printed": False,
        },
        "iam_never_pause_invariant": {
            "schema_version": operator.NEVER_PAUSE_SCHEMA,
            "status": "enforced",
            "local_chain_failure_disposition": "ignored_non_authority",
            "blocks_recovery": False,
            "operator_iam_request_allowed": False,
        },
    }
    ledger.write_text(json.dumps(value), encoding="utf-8")
    ledger.chmod(0o600)
    operator._validate_active_ledger(ledger, commit=COMMIT, source_hash=SOURCE_HASH)

    value["iam_authority_route"]["caller_arn"] = (
        f"arn:aws:iam::{ACCOUNT}:user/other"
    )
    ledger.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(operator.OperationError, match="ledger gate"):
        operator._validate_active_ledger(
            ledger, commit=COMMIT, source_hash=SOURCE_HASH
        )


def test_iam_operator_is_bound_into_production_parity_source_commitments():
    assert operator.OPERATOR_PATH in contract_builder.ALWAYS_COMMITTED_PATHS
