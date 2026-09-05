from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from scripts import build_production_parity_contract as contract_builder
from scripts import operate_rebenchmark_iam_policy as operator
from scripts import setup_production_parity_staging as parity_setup


ACCOUNT = operator.EXPECTED_ACCOUNT_ID
CALLER = operator.EXPECTED_CALLER_ARN
COMMIT = "a" * 40
SOURCE_HASH = "sha256:" + "b" * 64
ROLE = "leadpoet-gateway-s3-cloudwatch-role"
CONTROLLER_ROLE = parity_setup.CONTROLLER_ROLE
POLICY_NAME = "leadpoet-gateway-env-secretsmanager"
MANAGED_POLICY_NAME = "LeadpoetParityControllerData"
MANAGED_ARN = (
    f"arn:aws:iam::{ACCOUNT}:policy/leadpoet/production-parity/"
    f"{MANAGED_POLICY_NAME}"
)
RESOURCE = f"arn:aws:s3:::leadpoet-test-{ACCOUNT}/*"


@pytest.fixture(autouse=True)
def _instant_readback(monkeypatch):
    monkeypatch.setattr(operator, "READBACK_SLEEP_SECONDS", 0)


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
    iam,
    state: dict[str, object],
    *,
    target: dict[str, str],
    desired: dict[str, object] = AFTER,
    prune: dict[str, str] | None = None,
) -> dict[str, object]:
    canonical_desired = operator._canonical_policy(desired)
    normalized_target = operator._target(target)
    scope = operator._computed_task_scope(
        state["document"],
        canonical_desired,
        scope_id="test-change",
        target=normalized_target,
    )
    plan_request = operator._validate_plan_request(
        {
            "schema_version": operator.PLAN_REQUEST_SCHEMA,
            "change_id": "test-change",
            "target": target,
            "desired_document": desired,
            "task_scope": scope,
            "simulations": _simulation(),
            "prune_managed_version": None,
        }
    )
    plan = operator._plan_receipt(
        iam,
        plan_request,
        state=state,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
    )
    iam.simulation_calls = 0
    if hasattr(iam, "principal_simulation_calls"):
        iam.principal_simulation_calls = 0
    return {
        "schema_version": operator.REQUEST_SCHEMA,
        "change_id": "test-change",
        "target": target,
        "desired_document": desired,
        "task_scope": scope,
        "simulations": _simulation(),
        "plan": plan,
        "prune_managed_version": prune,
    }


def _test_intent(plan: dict[str, object]) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": operator.INTENT_SCHEMA,
        "operation_id": "1" * 32,
        "control_run_id": "2" * 32,
        "invocation_id": "3" * 32,
        "status": "reserved",
        "reservation_generation": 4,
        "ledger_generation": 5,
        "change_id": plan["change_id"],
        "plan_hash": plan["plan_hash"],
        "target": plan["target"],
        "origin_main_sha": plan["origin_main_sha"],
        "bridge_source_hash": plan["bridge_source_hash"],
        "prior_document_hash": plan["prior_document_hash"],
        "inventory_hash": plan["inventory_hash"],
        "desired_document_hash": plan["desired_document_hash"],
        "reserved_at": plan["planned_at"],
        "stop_requested_at": None,
        "last_reconciliation_status": None,
        "last_reconciled_at": None,
        "completed_at": None,
    }
    value["intent_hash"] = operator._sha256_json(operator._intent_material(value))
    return value


def _bound_request(request: dict[str, object]) -> dict[str, object]:
    normalized = operator._validate_request(request)
    normalized["intent"] = _test_intent(normalized["plan"])
    return normalized


def _with_expired_plan(request: dict[str, object]) -> dict[str, object]:
    value = json.loads(json.dumps(request))
    now = datetime.now(timezone.utc).replace(microsecond=0)
    value["plan"]["planned_at"] = (now - timedelta(seconds=30)).isoformat().replace(
        "+00:00", "Z"
    )
    value["plan"]["expires_at"] = (now - timedelta(seconds=15)).isoformat().replace(
        "+00:00", "Z"
    )
    material = dict(value["plan"])
    material.pop("plan_hash")
    value["plan"]["plan_hash"] = operator._sha256_json(material)
    return value


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
        self.principal_simulation_calls = 0
        self.mutate_after_first_simulation = False
        self.fail_post_simulation = False
        self.third_state_on_post_simulation = False
        self.third_state_on_passing_post_simulation = False
        self.post_simulation_transport_error = False
        self.put_response_lost = False

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
        if self.put_response_lost:
            self.put_response_lost = False
            raise RuntimeError("redacted response loss")

    def delete_role_policy(self, **kwargs):
        assert kwargs == {"RoleName": ROLE, "PolicyName": POLICY_NAME}
        assert self.existing
        self.existing = False
        self.events.append(("delete", operator._policy_hash(self.document)))

    def simulate_custom_policy(self, **kwargs):
        self.simulation_calls += 1
        if self.mutate_after_first_simulation and self.simulation_calls == 1:
            self.document = operator._canonical_policy(THIRD)
        resource = kwargs["ResourceArns"][0]
        decision = "implicitDeny" if "iam-decoy" in resource else "allowed"
        if self.fail_post_simulation and self.simulation_calls == 3:
            decision = "implicitDeny"
            if self.third_state_on_post_simulation:
                self.document = operator._canonical_policy(THIRD)
        if self.post_simulation_transport_error and self.simulation_calls == 3:
            raise RuntimeError("redacted transport failure")
        if (
            self.third_state_on_passing_post_simulation
            and self.simulation_calls == 3
        ):
            self.document = operator._canonical_policy(THIRD)
        action = kwargs["ActionNames"][0]
        return {
            "EvaluationResults": [
                {
                    "EvalActionName": action,
                    "EvalResourceName": resource,
                    "EvalDecision": decision,
                    "MissingContextValues": [],
                }
            ]
        }

    def simulate_principal_policy(self, **kwargs):
        self.principal_simulation_calls += 1
        resource = kwargs["ResourceArns"][0]
        return {
            "EvaluationResults": [
                {
                    "EvalActionName": kwargs["ActionNames"][0],
                    "EvalResourceName": resource,
                    "EvalDecision": (
                        "implicitDeny" if "iam-decoy" in resource else "allowed"
                    ),
                    "MissingContextValues": [],
                }
            ]
        }


class FakeManagedIam:
    def __init__(
        self,
        *,
        version_count: int = 1,
        policy_name: str = MANAGED_POLICY_NAME,
        attached_roles: tuple[str, ...] = (CONTROLLER_ROLE,),
    ):
        self.policy_name = policy_name
        self.policy_arn = (
            f"arn:aws:iam::{ACCOUNT}:policy/leadpoet/production-parity/"
            f"{policy_name}"
        )
        self.attached_roles = attached_roles
        self.versions: dict[str, dict[str, object]] = {
            f"v{index}": operator._canonical_policy(BEFORE)
            for index in range(1, version_count + 1)
        }
        self.default = f"v{version_count}"
        self.events: list[tuple[str, str]] = []
        self.simulation_calls = 0
        self.principal_simulation_calls = 0
        self.fail_post_simulation = False
        self.post_simulation_transport_error = False
        self.third_state_on_passing_post_simulation = False
        self.create_response_lost = False
        self.set_default_response_lost = False
        self.set_default_fails_before_once = False
        self.rollback_default_fails_before_once = False
        self.delete_fails_before_once = False

    def get_policy(self, *, PolicyArn):
        assert PolicyArn == self.policy_arn
        return {
            "Policy": {
                "Arn": self.policy_arn,
                "PolicyName": self.policy_name,
                "Path": "/leadpoet/production-parity/",
                "Description": "Leadpoet test policy",
                "DefaultVersionId": self.default,
            }
        }

    def list_policy_versions(self, *, PolicyArn):
        assert PolicyArn == self.policy_arn
        return {
            "Versions": [
                {"VersionId": version, "IsDefaultVersion": version == self.default}
                for version in sorted(self.versions)
            ]
        }

    def get_policy_version(self, *, PolicyArn, VersionId):
        assert PolicyArn == self.policy_arn
        return {"PolicyVersion": {"Document": self.versions[VersionId]}}

    def list_policy_tags(self, **kwargs):
        return {"Tags": [{"Key": "leadpoet:purpose", "Value": "test"}]}

    def list_entities_for_policy(self, **kwargs):
        return {
            "PolicyRoles": [
                {"RoleName": role_name} for role_name in self.attached_roles
            ],
            "PolicyUsers": [],
            "PolicyGroups": [],
        }

    def simulate_custom_policy(self, **kwargs):
        self.simulation_calls += 1
        resource = kwargs["ResourceArns"][0]
        decision = "implicitDeny" if "iam-decoy" in resource else "allowed"
        if self.fail_post_simulation and self.simulation_calls == 3:
            decision = "implicitDeny"
        if self.post_simulation_transport_error and self.simulation_calls == 3:
            raise RuntimeError("redacted transport failure")
        if (
            self.third_state_on_passing_post_simulation
            and self.simulation_calls == 3
        ):
            next_id = max(int(value[1:]) for value in self.versions) + 1
            third_version = f"v{next_id}"
            self.versions[third_version] = operator._canonical_policy(THIRD)
            self.default = third_version
        return {
            "EvaluationResults": [
                {
                    "EvalActionName": kwargs["ActionNames"][0],
                    "EvalResourceName": resource,
                    "EvalDecision": decision,
                    "MissingContextValues": [],
                }
            ]
        }

    def simulate_principal_policy(self, **kwargs):
        self.principal_simulation_calls += 1
        resource = kwargs["ResourceArns"][0]
        return {
            "EvaluationResults": [
                {
                    "EvalActionName": kwargs["ActionNames"][0],
                    "EvalResourceName": resource,
                    "EvalDecision": (
                        "implicitDeny" if "iam-decoy" in resource else "allowed"
                    ),
                    "MissingContextValues": [],
                }
            ]
        }

    def create_policy_version(self, **kwargs):
        assert kwargs["PolicyArn"] == self.policy_arn
        assert kwargs["SetAsDefault"] is False
        next_id = max(int(value[1:]) for value in self.versions) + 1
        version = f"v{next_id}"
        self.versions[version] = json.loads(kwargs["PolicyDocument"])
        self.events.append(("create", version))
        if self.create_response_lost:
            self.create_response_lost = False
            raise RuntimeError("redacted response loss")
        return {"PolicyVersion": {"VersionId": version}}

    def set_default_policy_version(self, *, PolicyArn, VersionId):
        assert PolicyArn == self.policy_arn
        assert VersionId in self.versions
        if self.set_default_fails_before_once:
            self.set_default_fails_before_once = False
            raise RuntimeError("redacted response loss before default")
        if VersionId == "v1" and self.rollback_default_fails_before_once:
            self.rollback_default_fails_before_once = False
            raise RuntimeError("redacted response loss before rollback")
        self.default = VersionId
        self.events.append(("default", VersionId))
        if self.set_default_response_lost:
            self.set_default_response_lost = False
            raise RuntimeError("redacted response loss")

    def delete_policy_version(self, *, PolicyArn, VersionId):
        assert PolicyArn == self.policy_arn
        assert VersionId != self.default
        if self.delete_fails_before_once:
            self.delete_fails_before_once = False
            raise RuntimeError("redacted response loss before delete")
        del self.versions[VersionId]
        self.events.append(("delete", VersionId))


def _apply_inline(iam: FakeInlineIam, request: dict[str, object]):
    return operator._apply_inline(
        iam,
        _bound_request(request),
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
    )


def _apply_managed(iam: FakeManagedIam, request: dict[str, object]):
    desired = operator._canonical_policy(request["desired_document"])
    return operator._apply_managed(
        iam,
        _bound_request(request),
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
        prewrite_condition_documents_loader=lambda _principal_arn: (desired,),
    )


def _remote_setup(iam) -> dict[str, object]:
    setup = dict(vars(parity_setup))
    setup["_iam_clients"] = lambda region: (FakeSts(), iam, ACCOUNT)
    return setup


def test_policy_validation_rejects_trust_material_and_unrelated_targets():
    invalid = _policy("s3:GetObject")
    invalid["Statement"][0]["Principal"] = {"AWS": "*"}
    with pytest.raises(operator.OperationError, match="identity-policy"):
        operator._canonical_policy(invalid)

    with pytest.raises(operator.OperationError, match="outside Leadpoet scope"):
        operator._target(
            {
                "kind": "inline_role",
                "role_name": "leadpoet-unlisted-role",
                "policy_name": POLICY_NAME,
            }
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


def test_policy_validation_rejects_negative_selectors_and_unscoped_grants():
    for selector in ("NotAction", "NotResource"):
        invalid = _policy("s3:GetObject")
        invalid["Statement"][0][selector] = invalid["Statement"][0].pop(
            "Action" if selector == "NotAction" else "Resource"
        )
        with pytest.raises(operator.OperationError, match="identity-policy"):
            operator._canonical_policy(invalid)

    unscoped = _policy("s3:GetObject")
    unscoped["Statement"][0]["Resource"] = "*"
    canonical = operator._canonical_policy(unscoped)
    with pytest.raises(operator.OperationError, match="global resource"):
        operator._computed_task_scope(
            operator._canonical_policy(BEFORE),
            canonical,
            scope_id="unscoped",
            target={
                "kind": "inline_role",
                "role_name": ROLE,
                "policy_name": POLICY_NAME,
            },
        )


def test_task_scope_allows_unchanged_legacy_wildcards_but_not_new_ones():
    legacy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "LegacyInventory",
                "Effect": "Allow",
                "Action": "ec2:Describe*",
                "Resource": "*",
            },
            BEFORE["Statement"][0],
        ],
    }
    desired = json.loads(json.dumps(legacy))
    desired["Statement"][1]["Action"].append("s3:PutObject")
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}

    scope = operator._computed_task_scope(
        operator._canonical_policy(legacy),
        operator._canonical_policy(desired),
        scope_id="retain-legacy",
        target=target,
    )

    assert operator._validate_task_scope(
        scope,
        before=operator._canonical_policy(legacy),
        after=operator._canonical_policy(desired),
        change_id="retain-legacy",
        target=target,
    ) == scope

    desired["Statement"][1]["Action"].append("s3:Get*")
    with pytest.raises(operator.OperationError, match="forbidden action"):
        operator._computed_task_scope(
            operator._canonical_policy(legacy),
            operator._canonical_policy(desired),
            scope_id="new-wildcard",
            target=target,
        )


def test_allowlisted_parity_policy_shapes_survive_noop_scope_validation():
    slices = parity_setup._controller_policy_slices(
        account_id=ACCOUNT,
        region=operator.EXPECTED_REGION,
        production_secret_id="leadpoet/prod/gateway/env",
        readonly_secret_id=parity_setup.READONLY_DSN_SECRET_ID,
        miner_intake_secret_id=parity_setup.DEFAULT_MINER_INTAKE_SECRET_ID,
        runner_arn=(
            f"arn:aws:iam::{ACCOUNT}:role/leadpoet-production-parity-runner"
        ),
    )
    for name, document in slices.items():
        canonical = operator._canonical_policy(document)
        target = {
            "kind": "managed",
            "policy_arn": (
                f"arn:aws:iam::{ACCOUNT}:policy/leadpoet/production-parity/{name}"
            ),
        }
        scope = operator._computed_task_scope(
            canonical,
            canonical,
            scope_id=f"noop-{name.lower()}",
            target=target,
        )
        assert scope["statement_changes"] == []
        assert operator._validate_task_scope(
            scope,
            before=canonical,
            after=canonical,
            change_id=f"noop-{name.lower()}",
            target=target,
        ) == scope


def test_sidless_revoke_condition_has_exact_hash_only_task_scope():
    before = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Deny",
                "Action": "*",
                "Resource": "*",
                "Condition": {
                    "DateLessThan": {"aws:TokenIssueTime": "2026-08-23T00:00:00Z"}
                },
            }
        ],
    }
    after = json.loads(json.dumps(before))
    after["Statement"][0]["Condition"]["DateLessThan"]["aws:TokenIssueTime"] = (
        "2026-08-23T00:05:00Z"
    )
    target = {
        "kind": "inline_role",
        "role_name": "leadpoet-production-parity-controller",
        "policy_name": "LeadpoetProductionParityRevokeOlderSessions",
    }

    scope = operator._computed_task_scope(
        operator._canonical_policy(before),
        operator._canonical_policy(after),
        scope_id="revoke-cutoff",
        target=target,
    )

    assert scope["statement_changes"][0]["sid"].startswith("unsided:sha256:")
    assert scope["statement_changes"][0]["added_actions"] == []
    assert scope["statement_changes"][0]["removed_actions"] == []
    assert scope["statement_changes"][0]["added_resources"] == []
    assert scope["statement_changes"][0]["removed_resources"] == []
    assert len(scope["statement_changes"][0]["condition_hashes"]) == 2
    assert "TokenIssueTime" not in json.dumps(
        operator._structural_delta(
            operator._canonical_policy(before), operator._canonical_policy(after)
        )
    )


def test_simulation_context_preserves_explicit_iam_data_type():
    cases = operator._validate_simulations(
        [
            {
                "name": "revoked-session",
                "action": "s3:GetObject",
                "resources": [RESOURCE],
                "context": {
                    "aws:TokenIssueTime": {
                        "type": "date",
                        "values": ["2026-08-23T00:00:00Z"],
                    }
                },
                "expected": "explicitDeny",
            }
        ]
    )

    assert operator._context_entries(cases[0]["context"]) == [
        {
            "ContextKeyName": "aws:TokenIssueTime",
            "ContextKeyValues": ["2026-08-23T00:00:00Z"],
            "ContextKeyType": "date",
        }
    ]
    revoke_document = operator._canonical_policy(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Deny",
                    "Action": "*",
                    "Resource": "*",
                    "Condition": {
                        "DateLessThan": {
                            "aws:TokenIssueTime": "2026-08-23T00:05:00Z"
                        }
                    },
                }
            ],
        }
    )
    decoys = operator._fixed_decoy_cases(revoke_document, cases)
    assert decoys[0]["expected"] == "explicitDeny"
    assert decoys[0]["resources"] != cases[0]["resources"]


def test_simulations_bind_owner_for_resource_arns_without_account_component():
    resource = f"arn:aws:s3:::leadpoet-test-{ACCOUNT}/artifact.json"
    case = {
        "name": "s3-owner-binding",
        "action": "s3:GetObject",
        "resources": [resource],
        "context": {},
        "expected": "allowed",
    }
    calls: list[dict[str, object]] = []

    class IAM:
        def simulate_custom_policy(self, **kwargs):
            calls.append(kwargs)
            return {
                "EvaluationResults": [{
                    "EvalActionName": "s3:GetObject",
                    "EvalResourceName": resource,
                    "EvalDecision": "allowed",
                    "MissingContextValues": [],
                }]
            }

        def simulate_principal_policy(self, **kwargs):
            calls.append(kwargs)
            return {
                "EvaluationResults": [{
                    "EvalActionName": "s3:GetObject",
                    "EvalResourceName": resource,
                    "EvalDecision": "allowed",
                    "MissingContextValues": [],
                }]
            }

    iam = IAM()
    operator._simulate_custom(iam, AFTER, [case])
    operator._simulate_principals(
        iam,
        [f"arn:aws:iam::{ACCOUNT}:role/{CONTROLLER_ROLE}"],
        [case],
    )

    assert [call["ResourceOwner"] for call in calls] == [
        f"arn:aws:iam::{ACCOUNT}:root",
        f"arn:aws:iam::{ACCOUNT}:root",
    ]


def test_exact_global_ecr_grant_uses_fixed_action_decoy():
    before = operator._canonical_policy(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "EcrRead",
                    "Effect": "Allow",
                    "Action": "ecr:BatchGetImage",
                    "Resource": "*",
                }
            ],
        }
    )
    after = json.loads(json.dumps(before))
    after["Statement"][0]["Action"].append("ecr:GetAuthorizationToken")
    after = operator._canonical_policy(after)
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}

    scope = operator._computed_task_scope(
        before,
        after,
        scope_id="ecr-auth",
        target=target,
    )
    assert scope["statement_changes"][0]["added_actions"] == [
        "ecr:GetAuthorizationToken"
    ]
    cases = operator._validate_simulations(
        [
            {
                "name": "ecr-auth",
                "action": "ecr:GetAuthorizationToken",
                "resources": ["*"],
                "context": {},
                "expected": "allowed",
            }
        ]
    )
    decoy = operator._fixed_decoy_cases(after, cases)[0]
    assert decoy["action"] == "ecr:DeleteRepository"
    assert decoy["expected"] == "implicitDeny"
    assert decoy["resources"][0].startswith(
        f"arn:aws:ecr:{operator.EXPECTED_REGION}:{ACCOUNT}:leadpoet-iam-decoy/"
    )

    with pytest.raises(operator.OperationError, match="unpermitted global resource"):
        operator._scoped_added_resource(
            "*", effect="Allow", actions=["ecr:DeleteRepository"]
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

    conditioned_before = _policy("s3:GetObject")
    conditioned_before["Statement"][0]["Condition"] = {
        "StringEquals": {"leadpoet:private-condition-key": "before"}
    }
    conditioned_after = json.loads(json.dumps(conditioned_before))
    conditioned_after["Statement"][0]["Condition"]["StringEquals"][
        "leadpoet:private-condition-key"
    ] = "after"
    condition_delta = operator._structural_delta(
        operator._canonical_policy(conditioned_before),
        operator._canonical_policy(conditioned_after),
    )
    assert condition_delta[0]["path"] == "/Statement/0"
    assert "private-condition-key" not in json.dumps(condition_delta)


def test_plan_receipt_is_hash_only_and_binds_exact_task_scope():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    plan = request["plan"]

    assert "desired_document" not in plan
    assert "task_scope" not in plan
    assert "simulations" not in plan
    serialized = json.dumps(plan)
    assert "s3:GetObject" not in serialized
    assert "s3:PutObject" not in serialized
    assert RESOURCE not in serialized
    assert plan["expected_delta_hash"] == operator._sha256_json(
        plan["expected_delta"]
    )
    assert plan["simulation_hash"] == operator._sha256_json(
        request["simulations"]
    )

    altered = dict(request)
    altered_scope = json.loads(json.dumps(request["task_scope"]))
    altered_scope["statement_changes"][0]["added_actions"] = ["s3:DeleteObject"]
    altered["task_scope"] = altered_scope
    with pytest.raises(operator.OperationError, match="plan binding differs"):
        operator._validate_request(altered)

    swapped = json.loads(json.dumps(request))
    swapped["simulations"][0]["action"] = "s3:GetObject"
    assert len(swapped["simulations"]) == plan["simulation_case_count"]
    assert operator._sha256_json(swapped["simulations"]) != plan["simulation_hash"]
    with pytest.raises(operator.OperationError, match="plan binding differs"):
        operator._validate_request(swapped)


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
    request = _request(iam, operator._inline_state(iam, target), target=target)
    receipt = _apply_inline(iam, request)

    assert receipt["status"] == "updated"
    assert receipt["concurrency_model"] == operator.CONCURRENCY_MODEL
    assert receipt["aws_native_compare_and_swap"] is False
    assert receipt["secret_values_printed"] is False
    assert receipt["policy_material_printed"] is False
    assert receipt["readback_document_hash"] == operator._policy_hash(AFTER)
    assert receipt["fixed_decoy_case_count"] == 1
    assert receipt["principal_simulation_count"] == 2
    assert iam.simulation_calls == 4
    assert iam.principal_simulation_calls == 2
    assert len(iam.events) == 1
    serialized = json.dumps(receipt)
    assert "s3:PutObject" not in serialized
    assert RESOURCE not in serialized


def test_inline_stale_prior_hash_fails_before_write():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    iam.document = operator._canonical_policy(THIRD)
    with pytest.raises(operator.OperationError, match="plan no longer matches"):
        _apply_inline(iam, request)
    assert iam.events == []


def test_inline_concurrent_prewrite_change_is_not_overwritten():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    iam.mutate_after_first_simulation = True
    with pytest.raises(operator.OperationError, match="changed before write"):
        _apply_inline(iam, request)
    assert iam.events == []
    assert operator._policy_hash(iam.document) == operator._policy_hash(THIRD)


def test_inline_postwrite_simulation_failure_rolls_back_only_intended_state():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    iam.fail_post_simulation = True
    with pytest.raises(operator.OperationError, match="simulation rolled back"):
        _apply_inline(iam, request)
    assert operator._policy_hash(iam.document) == operator._policy_hash(BEFORE)
    assert len(iam.events) == 2


def test_inline_create_rolls_back_to_absent_without_touching_surrounding_state():
    iam = FakeInlineIam(existing=False)
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    assert all(
        item["path"].startswith("/Statement/")
        for item in request["plan"]["expected_delta"]
    )
    iam.fail_post_simulation = True

    with pytest.raises(operator.OperationError, match="simulation rolled back"):
        _apply_inline(iam, request)

    assert iam.existing is False
    assert [event[0] for event in iam.events] == ["put", "delete"]


def test_inline_postwrite_third_state_is_never_rolled_back():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    iam.fail_post_simulation = True
    iam.third_state_on_post_simulation = True
    with pytest.raises(operator.OperationError, match="third state"):
        _apply_inline(iam, request)
    assert operator._policy_hash(iam.document) == operator._policy_hash(THIRD)
    assert len(iam.events) == 1


def test_inline_response_loss_is_reconciled_without_duplicate_write():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    iam.put_response_lost = True

    receipt = _apply_inline(iam, request)

    assert receipt["status"] == "reconciled"
    assert operator._policy_hash(iam.document) == operator._policy_hash(AFTER)
    assert [event[0] for event in iam.events] == ["put"]


def test_expired_plan_cannot_start_write_but_applied_state_reconciles():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    expired = _with_expired_plan(request)

    with pytest.raises(operator.OperationError, match="apply window"):
        _apply_inline(iam, expired)
    assert iam.events == []

    iam.document = operator._canonical_policy(AFTER)
    receipt = _apply_inline(iam, expired)

    assert receipt["status"] == "reconciled"
    assert iam.events == []


def test_read_only_reconciliation_classifies_before_applied_and_third_state():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _bound_request(
        _request(iam, operator._inline_state(iam, target), target=target)
    )

    before = operator._reconcile_policy(
        iam,
        request,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
    )
    assert before["status"] == "before"
    assert iam.events == []

    iam.document = operator._canonical_policy(AFTER)
    applied = operator._reconcile_policy(
        iam,
        request,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
    )
    assert applied["status"] == "reconciled"
    assert iam.events == []

    iam.document = operator._canonical_policy(THIRD)
    ambiguous = operator._reconcile_policy(
        iam,
        request,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
    )
    assert ambiguous["status"] == "ambiguous"
    assert iam.events == []


def test_inline_transport_failure_rolls_back_and_redacts_provider_error():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    iam.post_simulation_transport_error = True

    with pytest.raises(operator.OperationError, match="simulation rolled back") as exc:
        _apply_inline(iam, request)

    assert "transport failure" not in str(exc.value)
    assert operator._policy_hash(iam.document) == operator._policy_hash(BEFORE)


def test_inline_final_reread_detects_passing_simulation_race():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    iam.third_state_on_passing_post_simulation = True

    with pytest.raises(operator.OperationError, match="final verification"):
        _apply_inline(iam, request)

    assert operator._policy_hash(iam.document) == operator._policy_hash(THIRD)
    assert [event[0] for event in iam.events] == ["put"]


def test_managed_update_keeps_prior_default_as_rollback_version():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _request(iam, operator._managed_state(iam, target), target=target)
    receipt = _apply_managed(iam, request)

    assert receipt["status"] == "updated"
    assert receipt["managed_version_id"] == "v2"
    assert receipt["pruned_managed_version_id"] is None
    assert iam.default == "v2"
    assert set(iam.versions) == {"v1", "v2"}
    assert operator._policy_hash(iam.versions["v1"]) == operator._policy_hash(BEFORE)
    assert operator._policy_hash(iam.versions["v2"]) == operator._policy_hash(AFTER)


def test_managed_reconciliation_is_idempotent_and_staged_state_is_ambiguous():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _request(iam, operator._managed_state(iam, target), target=target)
    bound = _bound_request(request)
    iam.create_policy_version(
        PolicyArn=MANAGED_ARN,
        PolicyDocument=operator._json(operator._canonical_policy(AFTER)),
        SetAsDefault=False,
    )
    staged_event_count = len(iam.events)

    staged = operator._reconcile_policy(
        iam,
        bound,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
    )
    assert staged["status"] == "ambiguous"
    assert len(iam.events) == staged_event_count

    iam.set_default_policy_version(PolicyArn=MANAGED_ARN, VersionId="v2")
    applied_event_count = len(iam.events)
    applied = operator._reconcile_policy(
        iam,
        bound,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
    )
    assert applied["status"] == "reconciled"
    assert len(iam.events) == applied_event_count


def test_historical_managed_reconciliation_is_strictly_before_only():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _bound_request(
        _request(iam, operator._managed_state(iam, target), target=target)
    )

    before = operator._reconcile_policy(
        iam,
        request,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
        before_only=True,
    )
    assert before["status"] == "before"
    assert iam.events == []
    assert iam.simulation_calls == 0
    assert iam.principal_simulation_calls == 0

    iam.create_policy_version(
        PolicyArn=MANAGED_ARN,
        PolicyDocument=operator._json(operator._canonical_policy(AFTER)),
        SetAsDefault=False,
    )
    iam.set_default_policy_version(PolicyArn=MANAGED_ARN, VersionId="v2")
    event_count = len(iam.events)
    applied = operator._reconcile_policy(
        iam,
        request,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
        before_only=True,
    )
    assert applied["status"] == "ambiguous"
    assert len(iam.events) == event_count
    assert iam.simulation_calls == 0
    assert iam.principal_simulation_calls == 0


def test_managed_capacity_fails_before_irreversible_prune_or_write():
    iam = FakeManagedIam(version_count=5)
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    state = operator._managed_state(iam, target)
    with pytest.raises(operator.OperationError, match="preserve an exact rollback"):
        _request(iam, state, target=target)
    assert iam.default == "v5"
    assert len(iam.versions) == 5
    assert iam.events == []


def test_managed_plan_rejects_user_or_group_attachments_before_write():
    iam = FakeManagedIam()
    iam.list_entities_for_policy = lambda **kwargs: {
        "PolicyRoles": [{"RoleName": ROLE}],
        "PolicyUsers": [{"UserName": "unrelated-user"}],
        "PolicyGroups": [],
    }
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    state = operator._managed_state(iam, target)

    with pytest.raises(operator.OperationError, match="user or group"):
        _request(iam, state, target=target)

    assert iam.events == []


def test_managed_target_principals_are_derived_from_setup_authority():
    expected_arns = {
        f"arn:aws:iam::{ACCOUNT}:policy/leadpoet/production-parity/{name}"
        for name in parity_setup.CONTROLLER_POLICY_NAMES.values()
    }
    assert set(operator.MANAGED_TARGET_PRINCIPAL_ALLOWLIST) == expected_arns
    assert set(operator.MANAGED_TARGET_PRINCIPAL_ALLOWLIST.values()) == {
        frozenset({parity_setup.CONTROLLER_ROLE})
    }
    operator._validate_setup_managed_authority(vars(parity_setup))

    with pytest.raises(operator.OperationError, match="authority differs"):
        invalid_setup = dict(vars(parity_setup))
        invalid_setup["CONTROLLER_ROLE"] = "leadpoet-production-parity-runner"
        operator._validate_setup_managed_authority(invalid_setup)


@pytest.mark.parametrize(
    "policy_name", sorted(parity_setup.CONTROLLER_POLICY_NAMES.values())
)
def test_each_managed_target_requires_exact_controller_principal(policy_name):
    iam = FakeManagedIam(policy_name=policy_name)
    target = {"kind": "managed", "policy_arn": iam.policy_arn}

    request = _request(iam, operator._managed_state(iam, target), target=target)

    assert request["plan"]["principal_arns"] == [
        f"arn:aws:iam::{ACCOUNT}:role/{CONTROLLER_ROLE}"
    ]


@pytest.mark.parametrize(
    "policy_name", sorted(parity_setup.CONTROLLER_POLICY_NAMES.values())
)
@pytest.mark.parametrize(
    "unexpected_role",
    (
        "leadpoet-gateway-s3-cloudwatch-role",
        "leadpoet-validator-s3-cloudwatch-role",
        "leadpoet-production-parity-runner",
        "leadpoet-production-parity-static-bootstrap",
    ),
)
def test_managed_target_rejects_each_unexpected_allowlisted_principal(
    policy_name,
    unexpected_role,
):
    iam = FakeManagedIam(
        policy_name=policy_name,
        attached_roles=(unexpected_role,),
    )
    target = {"kind": "managed", "policy_arn": iam.policy_arn}
    state = operator._managed_state(iam, target)

    with pytest.raises(operator.OperationError, match="principals differ"):
        _request(iam, state, target=target)

    assert iam.simulation_calls == 0
    assert iam.events == []


@pytest.mark.parametrize(
    "policy_name", sorted(parity_setup.CONTROLLER_POLICY_NAMES.values())
)
@pytest.mark.parametrize(
    "attached_roles",
    (
        (),
        (CONTROLLER_ROLE, "leadpoet-gateway-s3-cloudwatch-role"),
    ),
    ids=("missing", "extra"),
)
def test_managed_target_rejects_missing_or_extra_principal(
    policy_name,
    attached_roles,
):
    iam = FakeManagedIam(
        policy_name=policy_name,
        attached_roles=attached_roles,
    )
    target = {"kind": "managed", "policy_arn": iam.policy_arn}
    state = operator._managed_state(iam, target)

    with pytest.raises(operator.OperationError, match="principals differ"):
        _request(iam, state, target=target)

    assert iam.simulation_calls == 0
    assert iam.events == []


def _controller_documents() -> dict[str, dict[str, object]]:
    return parity_setup._controller_policy_slices(
        account_id=ACCOUNT,
        region=operator.EXPECTED_REGION,
        production_secret_id=parity_setup.PRODUCTION_GATEWAY_SECRET_ID,
        readonly_secret_id=parity_setup.READONLY_DSN_SECRET_ID,
        miner_intake_secret_id=parity_setup.DEFAULT_MINER_INTAKE_SECRET_ID,
        runner_arn=f"arn:aws:iam::{ACCOUNT}:role/{parity_setup.RUNNER_ROLE}",
    )


@pytest.mark.parametrize(
    "policy_name", sorted(parity_setup.CONTROLLER_POLICY_NAMES.values())
)
def test_each_managed_document_requires_exact_setup_authority(policy_name):
    document = _controller_documents()[policy_name]
    target = {
        "kind": "managed",
        "policy_arn": (
            f"arn:aws:iam::{ACCOUNT}:policy/leadpoet/production-parity/"
            f"{policy_name}"
        ),
    }

    operator._validate_target_policy_authority(
        target,
        before=operator._canonical_policy(document),
        after=operator._canonical_policy(document),
    )

    added_action = json.loads(json.dumps(document))
    added_action["Statement"].append(
        {
            "Sid": "UnapprovedTermination",
            "Effect": "Allow",
            "Action": "ec2:TerminateInstances",
            "Resource": f"arn:aws:ec2:us-east-1:{ACCOUNT}:instance/*",
        }
    )
    with pytest.raises(operator.OperationError, match="document authority"):
        operator._validate_target_policy_authority(
            target,
            before=operator._canonical_policy(document),
            after=operator._canonical_policy(added_action),
        )


def test_managed_document_rejects_condition_removal_and_resource_broadening():
    policy_name = "LeadpoetParityControllerLifecycle"
    document = _controller_documents()[policy_name]
    target = {
        "kind": "managed",
        "policy_arn": (
            f"arn:aws:iam::{ACCOUNT}:policy/leadpoet/production-parity/"
            f"{policy_name}"
        ),
    }
    without_condition = json.loads(json.dumps(document))
    next(
        statement
        for statement in without_condition["Statement"]
        if statement.get("Condition")
    ).pop("Condition")
    broadened_resource = json.loads(json.dumps(document))
    next(
        statement
        for statement in broadened_resource["Statement"]
        if statement.get("Resource") != "*"
    )["Resource"] = "*"

    for altered in (without_condition, broadened_resource):
        with pytest.raises(operator.OperationError, match="document authority"):
            operator._validate_target_policy_authority(
                target,
                before=operator._canonical_policy(document),
                after=operator._canonical_policy(altered),
            )


def _revoke_document(cutoff: datetime) -> dict[str, object]:
    return parity_setup._revoke_older_sessions_policy(
        cutoff=cutoff.isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def test_revocation_authority_allows_only_a_newer_exact_bounded_cutoff():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    before = _revoke_document(now - timedelta(minutes=2))
    after = _revoke_document(
        now
        + timedelta(
            seconds=operator.REVOCATION_NEW_CUTOFF_MIN_FUTURE_SECONDS + 5
        )
    )
    target = {
        "kind": "inline_role",
        "role_name": parity_setup.CONTROLLER_ROLE,
        "policy_name": parity_setup.CONTROLLER_REVOKE_POLICY,
    }

    operator._validate_target_policy_authority(
        target,
        before=operator._canonical_policy(before),
        after=operator._canonical_policy(after),
    )

    weakened = _revoke_document(now - timedelta(minutes=3))
    with pytest.raises(operator.OperationError, match="weaken"):
        operator._validate_target_policy_authority(
            target,
            before=operator._canonical_policy(before),
            after=operator._canonical_policy(weakened),
        )
    for invalid_cutoff in (
        now - timedelta(minutes=1),
        now
        + timedelta(
            seconds=operator.REVOCATION_NEW_CUTOFF_MIN_FUTURE_SECONDS - 2
        ),
        now
        + timedelta(
            seconds=operator.REVOCATION_NEW_CUTOFF_FUTURE_LIMIT_SECONDS + 2
        ),
    ):
        with pytest.raises(operator.OperationError, match="safe window"):
            operator._validate_target_policy_authority(
                target,
                before=operator._canonical_policy(before),
                after=operator._canonical_policy(
                    _revoke_document(invalid_cutoff)
                ),
            )
    with pytest.raises(operator.OperationError, match="safe window"):
        operator._validate_target_policy_authority(
            target,
            before=operator._canonical_policy(before),
            after=operator._canonical_policy(
                _revoke_document(now + timedelta(seconds=1))
            ),
            phase="apply",
        )
    removed_condition = json.loads(json.dumps(after))
    removed_condition["Statement"][0].pop("Condition")
    with pytest.raises(operator.OperationError, match="structure"):
        operator._validate_target_policy_authority(
            target,
            before=operator._canonical_policy(before),
            after=operator._canonical_policy(removed_condition),
        )


def test_revocation_plan_rechecks_cutoff_after_simulation_latency(monkeypatch):
    initial_now = datetime.now(timezone.utc).replace(microsecond=0)
    before = operator._canonical_policy(
        _revoke_document(initial_now - timedelta(minutes=2))
    )
    desired = operator._canonical_policy(
        _revoke_document(
            initial_now
            + timedelta(
                seconds=operator.REVOCATION_NEW_CUTOFF_MIN_FUTURE_SECONDS + 5
            )
        )
    )
    target = {
        "kind": "inline_role",
        "role_name": parity_setup.CONTROLLER_ROLE,
        "policy_name": parity_setup.CONTROLLER_REVOKE_POLICY,
    }
    operator._validate_target_policy_authority(
        target,
        before=before,
        after=desired,
    )
    scope = operator._computed_task_scope(
        before,
        desired,
        scope_id="revocation-after-simulation",
        target=target,
    )
    request = operator._validate_plan_request(
        {
            "schema_version": operator.PLAN_REQUEST_SCHEMA,
            "change_id": "revocation-after-simulation",
            "target": target,
            "desired_document": desired,
            "task_scope": scope,
            "simulations": [
                {
                    "name": "old-session-denied",
                    "action": "sts:AssumeRole",
                    "resources": [
                        f"arn:aws:iam::{ACCOUNT}:role/{CONTROLLER_ROLE}"
                    ],
                    "context": {},
                    "expected": "explicitDeny",
                }
            ],
            "prune_managed_version": None,
        }
    )

    class ExplicitDenyIam:
        @staticmethod
        def simulate_custom_policy(**kwargs):
            resource = kwargs["ResourceArns"][0]
            return {
                "EvaluationResults": [
                    {
                        "EvalActionName": kwargs["ActionNames"][0],
                        "EvalResourceName": resource,
                        "EvalDecision": "explicitDeny",
                        "MissingContextValues": [],
                    }
                ]
            }

    class AdvancedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            advanced = initial_now + timedelta(
                seconds=operator.REVOCATION_NEW_CUTOFF_FUTURE_LIMIT_SECONDS + 10
            )
            return advanced if tz is not None else advanced.replace(tzinfo=None)

    monkeypatch.setattr(operator, "datetime", AdvancedDatetime)
    state = {
        "document": before,
        "document_hash": operator._sha256_json(before),
        "inventory": {},
        "inventory_hash": "sha256:" + "8" * 64,
        "target_present": True,
    }
    with pytest.raises(operator.OperationError, match="safe window"):
        operator._plan_receipt(
            ExplicitDenyIam(),
            request,
            state=state,
            source_hash=SOURCE_HASH,
            commit=COMMIT,
            account_id=ACCOUNT,
            caller_arn=CALLER,
        )


@pytest.mark.parametrize(
    ("aggregate_resource", "decision"),
    (
        ("arn:${Partition}:s3:::${BucketName}/${KeyName}", "allowed"),
        ("*", "implicitDeny"),
        (None, "explicitDeny"),
    ),
    ids=("template-allowed", "star-implicit", "omitted-explicit"),
)
def test_single_resource_simulation_binds_safe_aggregate_to_request(
    aggregate_resource,
    decision,
):
    resource = f"arn:aws:s3:::leadpoet-artifacts-{ACCOUNT}/exact/current.json"
    case = {
        "name": "signed-pointer-read",
        "action": "s3:GetObject",
        "resources": [resource],
        "context": {},
        "expected": decision,
    }

    operator._check_simulation_response(
        {
            "EvaluationResults": [
                {
                    "EvalActionName": case["action"].swapcase(),
                    "EvalDecision": decision,
                    "MissingContextValues": [],
                    **(
                        {"EvalResourceName": aggregate_resource}
                        if aggregate_resource is not None
                        else {}
                    ),
                }
            ]
        },
        case,
        label="custom-policy",
    )


def test_multi_resource_simulation_uses_exact_resource_specific_rows():
    resources = [
        f"arn:aws:s3:::leadpoet-artifacts-{ACCOUNT}/exact/current.json",
        f"arn:aws:s3:::leadpoet-artifacts-{ACCOUNT}/exact/archive.json",
    ]
    case = {
        "name": "signed-object-reads",
        "action": "s3:GetObject",
        "resources": resources,
        "context": {},
        "expected": "allowed",
    }

    operator._check_simulation_response(
        {
            "EvaluationResults": [
                {
                    "EvalActionName": case["action"],
                    "EvalResourceName": (
                        "arn:${Partition}:s3:::${BucketName}/${KeyName}"
                    ),
                    "EvalDecision": "allowed",
                    "MissingContextValues": [],
                    "ResourceSpecificResults": [
                        {
                            "EvalResourceName": resource,
                            "EvalResourceDecision": "allowed",
                            "MissingContextValues": [],
                        }
                        for resource in resources
                    ],
                }
            ]
        },
        case,
        label="custom-policy",
    )


def test_multi_resource_simulation_preserves_legacy_flat_rows():
    resources = [
        f"arn:aws:s3:::leadpoet-artifacts-{ACCOUNT}/exact/current.json",
        f"arn:aws:s3:::leadpoet-artifacts-{ACCOUNT}/exact/archive.json",
    ]
    results = [
        {
            "EvalActionName": "s3:GetObject",
            "EvalResourceName": resource,
            "EvalDecision": "allowed",
            "MissingContextValues": [],
        }
        for resource in reversed(resources)
    ]
    case = {
        "name": "legacy-flat-rows",
        "action": "s3:GetObject",
        "resources": resources,
        "context": {},
        "expected": "allowed",
    }

    operator._check_simulation_response(
        {"EvaluationResults": results}, case, label="custom-policy"
    )
    observed = operator._normalize_simulation_results(
        results,
        action=case["action"],
        requested_resources=resources,
    )
    assert observed == parity_setup._normalize_simulation_results(
        results,
        action=case["action"],
        requested_resources=resources,
    )


def _semantic_context_policy(*, applicable: bool = False) -> dict[str, object]:
    conditioned_action = "s3:Get*" if applicable else "kms:Verify"
    return operator._canonical_policy(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "UnconditionalRead",
                    "Effect": "Allow",
                    "Action": "s3:GetObject",
                    "Resource": RESOURCE,
                },
                {
                    "Sid": "ConditionedStatement",
                    "Effect": "Allow",
                    "Action": conditioned_action,
                    "Resource": "*" if not applicable else RESOURCE,
                    "Condition": {
                        "ForAnyValue:StringEquals": {
                            "kms:ResourceAliases": "alias/leadpoet-test"
                        }
                    },
                },
            ],
        }
    )


def _semantic_context_response(*, top: list[str], nested: list[str]):
    return {
        "EvaluationResults": [
            {
                "EvalActionName": "s3:GetObject",
                "EvalResourceName": (
                    "arn:${Partition}:s3:::${BucketName}/${KeyName}"
                ),
                "EvalDecision": "allowed",
                "MissingContextValues": top,
                "ResourceSpecificResults": [
                    {
                        "EvalResourceName": RESOURCE,
                        "EvalResourceDecision": "allowed",
                        "MissingContextValues": nested,
                    }
                ],
            }
        ]
    }


@pytest.mark.parametrize(
    ("top", "nested"),
    (
        (["kms:ResourceAliases"], []),
        ([], ["kms:ResourceAliases"]),
        (["KMS:RESOURCEALIASES"], ["kms:ResourceAliases"]),
    ),
    ids=("top", "nested", "both-casefolded"),
)
def test_simulation_tolerates_only_known_action_inapplicable_context(
    top,
    nested,
):
    case = {
        "name": "semantic-context",
        "action": "s3:GetObject",
        "resources": [RESOURCE],
        "context": {},
        "expected": "allowed",
    }

    operator._check_simulation_response(
        _semantic_context_response(top=top, nested=nested),
        case,
        label="custom-policy",
        condition_documents=(_semantic_context_policy(),),
    )


@pytest.mark.parametrize(
    ("missing", "documents", "expected_code"),
    (
        (
            "private:UnknownContext",
            (_semantic_context_policy(),),
            "IAM_SIM_MISSING_CONTEXT_UNKNOWN",
        ),
        (
            "kms:ResourceAliases",
            (_semantic_context_policy(applicable=True),),
            "IAM_SIM_MISSING_CONTEXT_APPLICABLE",
        ),
        (
            "kms:ResourceAliases",
            (
                _semantic_context_policy(),
                _semantic_context_policy(applicable=True),
            ),
            "IAM_SIM_MISSING_CONTEXT_APPLICABLE",
        ),
    ),
    ids=("unknown", "action-applicable", "unrelated-and-applicable"),
)
@pytest.mark.parametrize(
    "placement",
    ("top", "nested", "both"),
)
def test_simulation_context_semantics_fail_closed_without_leaking(
    missing,
    documents,
    expected_code,
    placement,
):
    case = {
        "name": "semantic-context-negative",
        "action": "s3:GetObject",
        "resources": [RESOURCE],
        "context": {},
        "expected": "allowed",
    }
    top = [missing] if placement in {"top", "both"} else []
    nested = [missing] if placement in {"nested", "both"} else []
    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._check_simulation_response(
            _semantic_context_response(top=top, nested=nested),
            case,
            label="custom-policy",
            condition_documents=documents,
        )

    assert failure.value.remote_diagnostic_code == expected_code
    assert missing not in str(failure.value)
    assert RESOURCE not in str(failure.value)


def test_context_authority_includes_policy_variables_and_wildcard_denies():
    document = operator._canonical_policy(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Deny",
                    "Action": "S3:*",
                    "Resource": "arn:aws:s3:::bucket/${aws:PrincipalTag/team}/*",
                    "Condition": {
                        "StringEquals": {"aws:RequestedRegion": "us-east-1"}
                    },
                }
            ],
        }
    )

    known, applicable = operator._context_key_authority(
        (document,), action="s3:GetObject"
    )

    assert known == {"aws:principaltag/team", "aws:requestedregion"}
    assert applicable == known


def test_principal_context_inventory_is_hash_bound_around_simulation():
    case = {
        "name": "principal-context-snapshot",
        "action": "s3:GetObject",
        "resources": [RESOURCE],
        "context": {},
        "expected": "allowed",
    }

    class IAM:
        def simulate_principal_policy(self, **kwargs):
            assert kwargs["PolicySourceArn"].endswith(CONTROLLER_ROLE)
            return _semantic_context_response(
                top=["kms:ResourceAliases"], nested=[]
            )

    calls = 0

    def stable_loader(_principal_arn):
        nonlocal calls
        calls += 1
        return (_semantic_context_policy(),)

    operator._simulate_principals(
        IAM(),
        [f"arn:aws:iam::{ACCOUNT}:role/{CONTROLLER_ROLE}"],
        [case],
        condition_documents_loader=stable_loader,
    )
    assert calls == 2

    def changing_loader(_principal_arn):
        nonlocal calls
        calls += 1
        return (
            _semantic_context_policy(applicable=(calls % 2 == 0)),
        )

    calls = 0
    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._simulate_principals(
            IAM(),
            [f"arn:aws:iam::{ACCOUNT}:role/{CONTROLLER_ROLE}"],
            [case],
            condition_documents_loader=changing_loader,
        )
    assert failure.value.remote_diagnostic_code == "IAM_SIM_PRINCIPAL_INVENTORY"


def test_managed_principal_context_inventory_is_exact_and_complete():
    setup_documents = operator._validate_setup_managed_authority(
        dict(vars(parity_setup))
    )
    revoke = operator._canonical_policy(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Deny",
                    "Action": "*",
                    "Resource": "*",
                    "Condition": {
                        "DateLessThan": {
                            "aws:TokenIssueTime": "2026-08-23T00:05:00Z"
                        }
                    },
                }
            ],
        }
    )

    class IAM:
        boundary = None
        extra_attachment = False

        def __init__(self):
            self.revoke = revoke
            self.defaults: dict[str, str] = {}
            self.live_versions: dict[str, dict[str, dict[str, object]]] = {}
            for arn in sorted(operator.MANAGED_TARGET_ALLOWLIST):
                name = arn.rsplit("/", 1)[-1]
                self.defaults[arn] = "v1"
                self.live_versions[arn] = {
                    "v1": operator._canonical_policy(setup_documents[name])
                }

        def set_default_document(self, arn, document):
            version = f"v{len(self.live_versions[arn]) + 1}"
            self.live_versions[arn][version] = operator._canonical_policy(document)
            self.defaults[arn] = version

        def get_role(self, **kwargs):
            assert kwargs == {"RoleName": CONTROLLER_ROLE}
            role = {
                "Arn": f"arn:aws:iam::{ACCOUNT}:role/{CONTROLLER_ROLE}",
            }
            if self.boundary is not None:
                role["PermissionsBoundary"] = self.boundary
            return {"Role": role}

        def list_role_policies(self, **kwargs):
            assert kwargs["RoleName"] == CONTROLLER_ROLE
            return {
                "PolicyNames": [operator.MANAGED_CONTROLLER_REVOKE_POLICY]
            }

        def list_attached_role_policies(self, **kwargs):
            assert kwargs["RoleName"] == CONTROLLER_ROLE
            arns = sorted(operator.MANAGED_TARGET_ALLOWLIST)
            if self.extra_attachment:
                arns.append(
                    f"arn:aws:iam::{ACCOUNT}:policy/leadpoet/production-parity/Extra"
                )
            return {
                "AttachedPolicies": [
                    {"PolicyArn": arn, "PolicyName": arn.rsplit("/", 1)[-1]}
                    for arn in arns
                ]
            }

        def get_role_policy(self, **kwargs):
            assert kwargs == {
                "RoleName": CONTROLLER_ROLE,
                "PolicyName": operator.MANAGED_CONTROLLER_REVOKE_POLICY,
            }
            return {"PolicyDocument": self.revoke}

        def get_policy(self, **kwargs):
            arn = kwargs["PolicyArn"]
            assert arn in self.live_versions
            return {
                "Policy": {
                    "Arn": arn,
                    "PolicyName": arn.rsplit("/", 1)[-1],
                    "DefaultVersionId": self.defaults[arn],
                }
            }

        def get_policy_version(self, **kwargs):
            arn = kwargs["PolicyArn"]
            version = kwargs["VersionId"]
            return {
                "PolicyVersion": {
                    "VersionId": version,
                    "Document": self.live_versions[arn][version],
                }
            }

    principal = f"arn:aws:iam::{ACCOUNT}:role/{CONTROLLER_ROLE}"
    iam = IAM()
    target_hash = operator._policy_hash(setup_documents[MANAGED_POLICY_NAME])

    def load():
        return operator._managed_principal_condition_documents(
            iam,
            setup_documents,
            principal,
            target_policy_arn=MANAGED_ARN,
            target_allowed_document_hashes=frozenset({target_hash}),
        )

    observed = load()
    assert len(observed) == len(setup_documents) + 1
    assert operator._policy_hash(observed[-1]) == operator._policy_hash(revoke)

    iam.extra_attachment = True
    with pytest.raises(operator.RemoteDiagnosticError) as extra:
        load()
    assert extra.value.remote_diagnostic_code == "IAM_SIM_PRINCIPAL_INVENTORY"

    iam.extra_attachment = False
    iam.boundary = {
        "PermissionsBoundaryType": "Policy",
        "PermissionsBoundaryArn": (
            f"arn:aws:iam::{ACCOUNT}:policy/leadpoet/Boundary"
        ),
    }
    with pytest.raises(operator.RemoteDiagnosticError) as boundary:
        load()
    assert boundary.value.remote_diagnostic_code == "IAM_SIM_PRINCIPAL_INVENTORY"

    iam.boundary = None
    non_target = next(
        arn for arn in sorted(operator.MANAGED_TARGET_ALLOWLIST)
        if arn != MANAGED_ARN
    )
    iam.set_default_document(non_target, _semantic_context_policy())
    with pytest.raises(operator.RemoteDiagnosticError) as drifted:
        load()
    assert drifted.value.remote_diagnostic_code == "IAM_SIM_PRINCIPAL_INVENTORY"

    non_target_name = non_target.rsplit("/", 1)[-1]
    iam.set_default_document(non_target, setup_documents[non_target_name])
    prior = operator._canonical_policy(BEFORE)
    prior_hash = operator._policy_hash(prior)
    iam.set_default_document(MANAGED_ARN, prior)
    transitioned = operator._managed_principal_condition_documents(
        iam,
        setup_documents,
        principal,
        target_policy_arn=MANAGED_ARN,
        target_allowed_document_hashes=frozenset({target_hash, prior_hash}),
    )
    assert operator._policy_hash(
        transitioned[sorted(operator.MANAGED_TARGET_ALLOWLIST).index(MANAGED_ARN)]
    ) == prior_hash
    with pytest.raises(operator.RemoteDiagnosticError) as wrong_phase:
        load()
    assert wrong_phase.value.remote_diagnostic_code == "IAM_SIM_PRINCIPAL_INVENTORY"

    iam.set_default_document(MANAGED_ARN, setup_documents[MANAGED_POLICY_NAME])
    drift_case = {
        "name": "live-default-drift",
        "action": "s3:GetObject",
        "resources": [RESOURCE],
        "context": {},
        "expected": "allowed",
    }

    def simulate_principal_policy(**_kwargs):
        iam.set_default_document(MANAGED_ARN, prior)
        # Test live inventory drift without depending on a retired KMS condition.
        return _semantic_context_response(top=[], nested=[])

    iam.simulate_principal_policy = simulate_principal_policy

    def transition_loader(_principal_arn):
        return operator._managed_principal_condition_documents(
            iam,
            setup_documents,
            principal,
            target_policy_arn=MANAGED_ARN,
            target_allowed_document_hashes=frozenset({target_hash, prior_hash}),
        )

    with pytest.raises(operator.RemoteDiagnosticError) as default_drift:
        operator._simulate_principals(
            iam,
            [principal],
            [drift_case],
            condition_documents_loader=transition_loader,
        )
    assert (
        default_drift.value.remote_diagnostic_code
        == "IAM_SIM_PRINCIPAL_INVENTORY"
    )

    iam.set_default_document(MANAGED_ARN, setup_documents[MANAGED_POLICY_NAME])
    iam.revoke = operator._canonical_policy(BEFORE)
    with pytest.raises(operator.RemoteDiagnosticError) as invalid_revoke:
        load()
    assert (
        invalid_revoke.value.remote_diagnostic_code
        == "IAM_SIM_PRINCIPAL_INVENTORY"
    )


def test_multi_resource_legacy_flat_denial_is_not_misattributed_per_resource():
    resources = [RESOURCE, f"{RESOURCE}archive"]
    results = [
        {
            "EvalActionName": "s3:GetObject",
            "EvalResourceName": resource,
            "EvalDecision": "implicitDeny",
        }
        for resource in resources
    ]
    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._normalize_simulation_results(
            results,
            action="s3:GetObject",
            requested_resources=resources,
        )
    assert failure.value.remote_diagnostic_code == "IAM_SIM_AGGREGATE"


@pytest.mark.parametrize("specific", (pytest.param("omitted"), pytest.param([])))
def test_single_resource_flat_result_accepts_omitted_or_empty_specific_rows(specific):
    result = {
        "EvalActionName": "s3:GetObject",
        "EvalResourceName": RESOURCE,
        "EvalDecision": "allowed",
    }
    if specific != "omitted":
        result["ResourceSpecificResults"] = specific
    case = {
        "name": "flat-specific-optional",
        "action": "s3:GetObject",
        "resources": [RESOURCE],
        "context": {},
        "expected": "allowed",
    }

    operator._check_simulation_response(
        {"EvaluationResults": [result]}, case, label="custom-policy"
    )


def test_single_resource_simulation_rejects_wrong_concrete_aggregate_resource():
    case = {
        "name": "wrong-concrete-resource",
        "action": "s3:GetObject",
        "resources": [RESOURCE],
        "context": {},
        "expected": "allowed",
    }
    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._check_simulation_response(
            {
                "EvaluationResults": [
                    {
                        "EvalActionName": case["action"],
                        "EvalResourceName": f"{RESOURCE}-other",
                        "EvalDecision": "allowed",
                    }
                ]
            },
            case,
            label="custom-policy",
        )
    assert failure.value.remote_diagnostic_code == "IAM_SIM_EXPECTATION"


@pytest.mark.parametrize(
    "results",
    (
        [
            {
                "EvalActionName": "s3:GetObject",
                "EvalResourceName": RESOURCE,
                "EvalDecision": "allowed",
                "MissingContextValues": None,
            }
        ],
        [
            {
                "EvalActionName": "s3:GetObject",
                "EvalResourceName": RESOURCE,
                "EvalDecision": "allowed",
                "ResourceSpecificResults": None,
            }
        ],
        [
            {
                "EvalActionName": "s3:GetObject",
                "EvalResourceName": "arn:${Partition}:s3:::${BucketName}/${KeyName}",
                "EvalDecision": "allowed",
                "ResourceSpecificResults": [
                    {
                        "EvalResourceName": RESOURCE,
                        "EvalResourceDecision": "allowed",
                        "MissingContextValues": None,
                    }
                ],
            }
        ],
    ),
    ids=("null-missing-context", "null-specific-rows", "null-nested-context"),
)
def test_simulation_rejects_explicit_null_contract_fields(results):
    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._normalize_simulation_results(
            results,
            action="s3:GetObject",
            requested_resources=[RESOURCE],
        )
    assert failure.value.remote_diagnostic_code == "IAM_SIM_RESULT_SHAPE"


def test_simulation_rejects_mixed_flat_and_nested_representations():
    results = [
        {
            "EvalActionName": "s3:GetObject",
            "EvalResourceName": RESOURCE,
            "EvalDecision": "allowed",
        },
        {
            "EvalActionName": "s3:GetObject",
            "EvalResourceName": "arn:${Partition}:s3:::${BucketName}/${KeyName}",
            "EvalDecision": "allowed",
            "ResourceSpecificResults": [
                {
                    "EvalResourceName": f"{RESOURCE}archive",
                    "EvalResourceDecision": "allowed",
                }
            ],
        },
    ]
    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._normalize_simulation_results(
            results,
            action="s3:GetObject",
            requested_resources=[RESOURCE, f"{RESOURCE}archive"],
        )
    assert failure.value.remote_diagnostic_code == "IAM_SIM_RESOURCE_ROWS"


def test_simulation_rejects_nested_aggregate_decision_mismatch():
    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._normalize_simulation_results(
            [
                {
                    "EvalActionName": "s3:GetObject",
                    "EvalResourceName": (
                        "arn:${Partition}:s3:::${BucketName}/${KeyName}"
                    ),
                    "EvalDecision": "implicitDeny",
                    "ResourceSpecificResults": [
                        {
                            "EvalResourceName": RESOURCE,
                            "EvalResourceDecision": "allowed",
                        }
                    ],
                }
            ],
            action="s3:GetObject",
            requested_resources=[RESOURCE],
        )
    assert failure.value.remote_diagnostic_code == "IAM_SIM_AGGREGATE"


def test_simulation_accepts_most_restrictive_nested_aggregate():
    allowed = RESOURCE
    denied = f"{RESOURCE}archive"
    decisions, missing = operator._normalize_simulation_results(
        [
            {
                "EvalActionName": "s3:GetObject",
                "EvalResourceName": (
                    "arn:${Partition}:s3:::${BucketName}/${KeyName}"
                ),
                "EvalDecision": "implicitDeny",
                "ResourceSpecificResults": [
                    {
                        "EvalResourceName": allowed,
                        "EvalResourceDecision": "allowed",
                    },
                    {
                        "EvalResourceName": denied,
                        "EvalResourceDecision": "implicitDeny",
                    },
                ],
            }
        ],
        action="s3:GetObject",
        requested_resources=[allowed, denied],
    )
    assert decisions == {allowed: "allowed", denied: "implicitDeny"}
    assert missing == set()


@pytest.mark.parametrize(
    ("resource", "decision"),
    (("", "allowed"), (RESOURCE, "unknown"), (None, "allowed")),
)
def test_simulation_rejects_malformed_flat_resource_or_decision(resource, decision):
    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._normalize_simulation_results(
            [
                {
                    "EvalActionName": "s3:GetObject",
                    "EvalResourceName": resource,
                    "EvalDecision": decision,
                }
            ],
            action="s3:GetObject",
            requested_resources=[RESOURCE],
        )
    assert failure.value.remote_diagnostic_code == "IAM_SIM_RESULT_SHAPE"


@pytest.mark.parametrize(
    ("specific_rows", "message"),
    (
        (
            [
                {
                    "EvalResourceName": RESOURCE,
                    "EvalResourceDecision": "allowed",
                    "MissingContextValues": [],
                },
                {
                    "EvalResourceName": RESOURCE,
                    "EvalResourceDecision": "allowed",
                    "MissingContextValues": [],
                },
            ],
            "ambiguous",
        ),
        (
            [
                {
                    "EvalResourceName": f"{RESOURCE}-unexpected",
                    "EvalResourceDecision": "allowed",
                    "MissingContextValues": [],
                }
            ],
            "differs",
        ),
        (
            [
                {
                    "EvalResourceName": RESOURCE,
                    "EvalResourceDecision": "allowed",
                    "MissingContextValues": ["aws:RequestedRegion"],
                }
            ],
            "differs",
        ),
    ),
    ids=("duplicate", "unexpected", "missing-context"),
)
def test_resource_specific_simulation_rejects_unbound_rows(
    specific_rows,
    message,
):
    case = {
        "name": "resource-specific-negative",
        "action": "s3:GetObject",
        "resources": [RESOURCE],
        "context": {},
        "expected": "allowed",
    }

    with pytest.raises(operator.OperationError, match=message):
        operator._check_simulation_response(
            {
                "EvaluationResults": [
                    {
                        "EvalActionName": case["action"],
                        "EvalResourceName": (
                            "arn:${Partition}:s3:::${BucketName}/${KeyName}"
                        ),
                        "EvalDecision": "allowed",
                        "MissingContextValues": [],
                        "ResourceSpecificResults": specific_rows,
                    }
                ]
            },
            case,
            label="principal-policy",
        )


def test_multi_resource_simulation_accepts_aggregate_only_allowed_result():
    case = {
        "name": "aggregate-only-multi-resource",
        "action": "s3:GetObject",
        "resources": [RESOURCE, f"{RESOURCE}archive"],
        "context": {},
        "expected": "allowed",
    }

    operator._check_simulation_response(
        {
            "EvaluationResults": [
                {
                    "EvalActionName": case["action"],
                    "EvalResourceName": (
                        "arn:${Partition}:s3:::${BucketName}/${KeyName}"
                    ),
                    "EvalDecision": "allowed",
                    "MissingContextValues": [],
                }
            ]
        },
        case,
        label="custom-policy",
    )


@pytest.mark.parametrize("decision", ("implicitDeny", "explicitDeny"))
def test_multi_resource_simulation_rejects_aggregate_only_denial(decision):
    resources = [RESOURCE, f"{RESOURCE}archive"]
    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._normalize_simulation_results(
            [
                {
                    "EvalActionName": "s3:GetObject",
                    "EvalResourceName": "*",
                    "EvalDecision": decision,
                }
            ],
            action="s3:GetObject",
            requested_resources=resources,
        )
    assert failure.value.remote_diagnostic_code == "IAM_SIM_AGGREGATE"


def test_simulation_api_failure_has_only_fixed_diagnostic_category():
    class BrokenIam:
        def simulate_custom_policy(self, **kwargs):
            del kwargs
            raise RuntimeError(f"private resource: {RESOURCE}")

    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._simulate_custom(BrokenIam(), operator._canonical_policy(AFTER), _simulation())
    assert failure.value.remote_diagnostic_code == "IAM_SIM_API_CALL"
    assert RESOURCE not in str(failure.value)


def test_plan_window_has_apply_liveness_and_revocation_safety_margin():
    assert operator.PLAN_VALIDITY_SECONDS == 60
    assert (
        operator.REVOCATION_NEW_CUTOFF_MIN_FUTURE_SECONDS
        >= operator.PLAN_VALIDITY_SECONDS
        + operator.REVOCATION_APPLY_MIN_FUTURE_SECONDS
        + 3
    )
    assert (
        operator.REVOCATION_NEW_CUTOFF_FUTURE_LIMIT_SECONDS
        > operator.REVOCATION_NEW_CUTOFF_MIN_FUTURE_SECONDS
    )

    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    plan = request["plan"]
    planned_at = operator._parse_timestamp(
        plan["planned_at"], label="IAM plan creation time"
    )
    operator._validate_plan_freshness(
        plan,
        now=planned_at + timedelta(seconds=operator.PLAN_VALIDITY_SECONDS - 1),
    )
    with pytest.raises(operator.OperationError, match="apply window"):
        operator._validate_plan_freshness(
            plan,
            now=planned_at + timedelta(seconds=operator.PLAN_VALIDITY_SECONDS),
        )


def test_gateway_target_rejects_self_authorized_action_before_simulation():
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    desired = json.loads(json.dumps(AFTER))
    desired["Statement"].append(
        {
            "Sid": "TerminateProduction",
            "Effect": "Allow",
            "Action": "ec2:TerminateInstances",
            "Resource": f"arn:aws:ec2:us-east-1:{ACCOUNT}:instance/*",
        }
    )
    canonical_before = operator._canonical_policy(BEFORE)
    canonical_desired = operator._canonical_policy(desired)
    request = {
        "schema_version": operator.PLAN_REQUEST_SCHEMA,
        "change_id": "unapproved-terminate",
        "target": target,
        "desired_document": desired,
        "task_scope": operator._computed_task_scope(
            canonical_before,
            canonical_desired,
            scope_id="unapproved-terminate",
            target=target,
        ),
        "simulations": [
            {
                "name": "unrelated-s3-read",
                "action": "s3:GetObject",
                "resources": [RESOURCE],
                "context": {},
                "expected": "allowed",
            }
        ],
        "prune_managed_version": None,
    }

    with pytest.raises(operator.OperationError, match="no repository-authorized"):
        operator._remote_entry(
            "plan",
            request,
            _remote_setup(iam),
            source_hash=SOURCE_HASH,
            commit=COMMIT,
        )

    assert iam.simulation_calls == 0
    assert iam.events == []


def test_task_scope_unsided_statement_identity_survives_canonical_reordering():
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    unchanged = {
        "Effect": "Allow",
        "Action": "s3:GetObject",
        "Resource": RESOURCE,
    }
    before = operator._canonical_policy(
        {
            "Version": "2012-10-17",
            "Statement": [unchanged],
        }
    )
    after = operator._canonical_policy(
        {
            "Version": "2012-10-17",
            "Statement": [
                unchanged,
                {
                    "Sid": "AddExactArtifactRead",
                    "Effect": "Allow",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::leadpoet-artifacts-{ACCOUNT}/exact/*",
                },
            ],
        }
    )

    scope = operator._computed_task_scope(
        before,
        after,
        scope_id="add-exact-artifact-read",
        target=target,
    )

    assert len(scope["statement_changes"]) == 1
    assert scope["statement_changes"][0]["sid"] == "sid:AddExactArtifactRead"
    assert scope["statement_changes"][0]["operation"] == "add"


def test_task_scope_rejects_duplicate_unsided_statements():
    duplicate = operator._canonical_policy(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "s3:GetObject",
                    "Resource": RESOURCE,
                },
                {
                    "Effect": "Allow",
                    "Action": "s3:GetObject",
                    "Resource": RESOURCE,
                },
            ],
        }
    )

    with pytest.raises(operator.OperationError, match="unique statements"):
        operator._computed_task_scope(
            None,
            duplicate,
            scope_id="duplicate-unsided-statements",
            target={"kind": "managed", "policy_arn": MANAGED_ARN},
        )


def test_managed_postwrite_simulation_failure_restores_prior_default():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _request(iam, operator._managed_state(iam, target), target=target)
    iam.fail_post_simulation = True
    with pytest.raises(operator.OperationError, match="simulation rolled back"):
        _apply_managed(iam, request)
    assert iam.default == "v1"
    assert set(iam.versions) == {"v1"}


def test_managed_inventory_drift_blocks_version_creation_before_write():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _bound_request(
        _request(iam, operator._managed_state(iam, target), target=target)
    )

    def drifted_inventory(_principal_arn):
        raise operator.RemoteDiagnosticError("IAM_SIM_PRINCIPAL_INVENTORY")

    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._apply_managed(
            iam,
            request,
            source_hash=SOURCE_HASH,
            commit=COMMIT,
            account_id=ACCOUNT,
            caller_arn=CALLER,
            prewrite_condition_documents_loader=drifted_inventory,
        )

    assert failure.value.remote_diagnostic_code == "IAM_SIM_PRINCIPAL_INVENTORY"
    assert iam.default == "v1"
    assert set(iam.versions) == {"v1"}
    assert iam.events == []


def test_managed_inventory_drift_after_create_cleans_before_default_write():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _bound_request(
        _request(iam, operator._managed_state(iam, target), target=target)
    )
    inventory_calls = 0

    def drifted_inventory(_principal_arn):
        nonlocal inventory_calls
        inventory_calls += 1
        if inventory_calls == 1:
            return (operator._canonical_policy(BEFORE),)
        raise operator.RemoteDiagnosticError("IAM_SIM_PRINCIPAL_INVENTORY")

    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._apply_managed(
            iam,
            request,
            source_hash=SOURCE_HASH,
            commit=COMMIT,
            account_id=ACCOUNT,
            caller_arn=CALLER,
            prewrite_condition_documents_loader=drifted_inventory,
        )

    assert failure.value.remote_diagnostic_code == "IAM_SIM_PRINCIPAL_INVENTORY"
    assert inventory_calls == 2
    assert iam.default == "v1"
    assert set(iam.versions) == {"v1"}
    assert [event[0] for event in iam.events] == ["create", "delete"]


def test_managed_resumed_staged_inventory_drift_cleans_before_default_write():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _bound_request(
        _request(iam, operator._managed_state(iam, target), target=target)
    )
    iam.create_policy_version(
        PolicyArn=MANAGED_ARN,
        PolicyDocument=operator._json(operator._canonical_policy(AFTER)),
        SetAsDefault=False,
    )
    iam.events.clear()

    def drifted_inventory(_principal_arn):
        raise operator.RemoteDiagnosticError("IAM_SIM_PRINCIPAL_INVENTORY")

    with pytest.raises(operator.RemoteDiagnosticError) as failure:
        operator._apply_managed(
            iam,
            request,
            source_hash=SOURCE_HASH,
            commit=COMMIT,
            account_id=ACCOUNT,
            caller_arn=CALLER,
            prewrite_condition_documents_loader=drifted_inventory,
        )

    assert failure.value.remote_diagnostic_code == "IAM_SIM_PRINCIPAL_INVENTORY"
    assert iam.default == "v1"
    assert set(iam.versions) == {"v1"}
    assert [event[0] for event in iam.events] == ["delete"]


def test_managed_principal_inventory_drift_rolls_back_updated_default():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _request(iam, operator._managed_state(iam, target), target=target)
    bound = _bound_request(request)
    loader_calls = 0

    def changing_loader(_principal_arn):
        nonlocal loader_calls
        loader_calls += 1
        return (
            _semantic_context_policy(applicable=loader_calls > 1),
        )

    with pytest.raises(operator.OperationError, match="simulation rolled back"):
        operator._apply_managed(
            iam,
            bound,
            source_hash=SOURCE_HASH,
            commit=COMMIT,
            account_id=ACCOUNT,
            caller_arn=CALLER,
            prewrite_condition_documents_loader=(
                lambda _principal_arn: (_semantic_context_policy(),)
            ),
            condition_documents_loader=changing_loader,
        )

    assert loader_calls == 2
    assert iam.principal_simulation_calls == 2
    assert iam.default == "v1"
    assert set(iam.versions) == {"v1"}


def test_remote_entry_propagates_exact_managed_inventory_loader(monkeypatch):
    desired = operator._canonical_policy(AFTER)
    desired_hash = operator._policy_hash(desired)
    prior_hash = operator._policy_hash(operator._canonical_policy(BEFORE))
    setup_documents = {"sentinel": desired}
    validated = {
        "target": {"kind": "managed", "policy_arn": MANAGED_ARN},
        "desired_document": desired,
        "plan": {
            "prior_document_hash": prior_hash,
            "desired_document_hash": desired_hash,
        },
        "intent": {"status": "reserved"},
    }
    iam = object()
    principal = f"arn:aws:iam::{ACCOUNT}:role/{CONTROLLER_ROLE}"
    loaded_documents = (_semantic_context_policy(),)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        operator,
        "_validate_setup_managed_authority",
        lambda _setup: setup_documents,
    )
    monkeypatch.setattr(
        operator,
        "_gateway_clients",
        lambda _setup: (FakeSts(), iam, ACCOUNT, CALLER),
    )
    monkeypatch.setattr(
        operator,
        "_validate_request",
        lambda *_args, **_kwargs: validated,
    )
    monkeypatch.setattr(
        operator,
        "_managed_state",
        lambda *_args, **_kwargs: {"document": desired},
    )
    monkeypatch.setattr(
        operator,
        "_validate_target_policy_authority",
        lambda *_args, **_kwargs: None,
    )

    def load_inventory(
        loaded_iam,
        loaded_setup_documents,
        loaded_principal,
        *,
        target_policy_arn,
        target_allowed_document_hashes,
    ):
        assert loaded_iam is iam
        assert loaded_setup_documents is setup_documents
        assert loaded_principal == principal
        assert target_policy_arn == MANAGED_ARN
        if target_allowed_document_hashes == frozenset({desired_hash}):
            observed["postwrite_loaded"] = True
        elif target_allowed_document_hashes == frozenset(
            {prior_hash, desired_hash}
        ):
            observed["prewrite_loaded"] = True
        else:
            raise AssertionError("unexpected managed target transition set")
        return loaded_documents

    monkeypatch.setattr(
        operator,
        "_managed_principal_condition_documents",
        load_inventory,
    )

    def apply_managed(
        loaded_iam,
        loaded_request,
        *,
        prewrite_condition_documents_loader,
        condition_documents_loader,
        **_kwargs,
    ):
        assert loaded_iam is iam
        assert loaded_request is validated
        assert prewrite_condition_documents_loader(principal) == loaded_documents
        assert condition_documents_loader(principal) == loaded_documents
        return {"status": "propagated"}

    monkeypatch.setattr(operator, "_apply_managed", apply_managed)

    assert operator._remote_entry(
        "apply",
        {},
        {},
        source_hash=SOURCE_HASH,
        commit=COMMIT,
    ) == {"status": "propagated"}
    assert observed == {
        "prewrite_loaded": True,
        "postwrite_loaded": True,
    }


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    (
        (RuntimeError("private provider failure"), "IAM_AUTHORITY_INVENTORY_API"),
        (
            operator.OperationError("private inventory detail"),
            "IAM_AUTHORITY_INVENTORY_SHAPE",
        ),
    ),
)
def test_remote_entry_classifies_authority_inventory_without_private_detail(
    monkeypatch,
    failure,
    expected_code,
):
    desired = operator._canonical_policy(AFTER)
    validated = {
        "target": {"kind": "managed", "policy_arn": MANAGED_ARN},
        "desired_document": desired,
        "plan": {"desired_document_hash": operator._policy_hash(desired)},
        "intent": {"status": "reserved"},
    }
    iam = object()
    monkeypatch.setattr(
        operator,
        "_validate_setup_managed_authority",
        lambda _setup: {},
    )
    monkeypatch.setattr(
        operator,
        "_gateway_clients",
        lambda _setup: (FakeSts(), iam, ACCOUNT, CALLER),
    )
    monkeypatch.setattr(
        operator,
        "_validate_request",
        lambda *_args, **_kwargs: validated,
    )

    def fail_inventory(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(operator, "_managed_state", fail_inventory)

    with pytest.raises(operator.RemoteDiagnosticError) as raised:
        operator._remote_entry(
            "apply",
            {},
            {},
            source_hash=SOURCE_HASH,
            commit=COMMIT,
        )

    assert raised.value.remote_diagnostic_code == expected_code
    assert "private" not in str(raised.value)


def test_authority_inventory_retries_transient_api_failure():
    calls = 0

    def transient_loader():
        nonlocal calls
        calls += 1
        if calls < operator.INVENTORY_READ_ATTEMPTS:
            raise RuntimeError("private transient provider failure")
        return {"document_hash": "sha256:" + "1" * 64}

    assert operator._redacted_inventory_read(
        transient_loader,
        stage="authority",
    ) == {"document_hash": "sha256:" + "1" * 64}
    assert calls == operator.INVENTORY_READ_ATTEMPTS


@pytest.mark.parametrize(
    ("failure", "expected_code", "expected_reads"),
    (
        (
            RuntimeError("private provider failure"),
            "IAM_RECONCILE_INVENTORY_API",
            1 + operator.INVENTORY_READ_ATTEMPTS,
        ),
        (
            operator.OperationError("private inventory detail"),
            "IAM_RECONCILE_INVENTORY_SHAPE",
            2,
        ),
    ),
)
def test_remote_entry_classifies_reconciliation_inventory_without_writes(
    monkeypatch,
    failure,
    expected_code,
    expected_reads,
):
    desired = operator._canonical_policy(AFTER)
    state = {"document": desired}
    validated = {
        "target": {"kind": "managed", "policy_arn": MANAGED_ARN},
        "desired_document": desired,
        "plan": {"desired_document_hash": operator._policy_hash(desired)},
        "intent": {"status": "reserved"},
    }
    iam = object()
    reads = 0
    monkeypatch.setattr(
        operator,
        "_validate_setup_managed_authority",
        lambda _setup: {},
    )
    monkeypatch.setattr(
        operator,
        "_gateway_clients",
        lambda _setup: (FakeSts(), iam, ACCOUNT, CALLER),
    )
    monkeypatch.setattr(
        operator,
        "_validate_request",
        lambda *_args, **_kwargs: validated,
    )
    monkeypatch.setattr(
        operator,
        "_validate_target_policy_authority",
        lambda *_args, **_kwargs: None,
    )

    def fail_stability_read(*_args, **_kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            return state
        raise failure

    monkeypatch.setattr(operator, "_managed_state", fail_stability_read)

    with pytest.raises(operator.RemoteDiagnosticError) as raised:
        operator._remote_entry(
            "reconcile",
            {},
            {},
            source_hash=SOURCE_HASH,
            commit=COMMIT,
        )

    assert reads == expected_reads
    assert raised.value.remote_diagnostic_code == expected_code
    assert "private" not in str(raised.value)


def test_managed_create_and_default_response_loss_are_reconciled():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _request(iam, operator._managed_state(iam, target), target=target)
    iam.create_response_lost = True
    iam.set_default_fails_before_once = True
    iam.set_default_response_lost = True

    receipt = _apply_managed(iam, request)

    assert receipt["status"] == "reconciled"
    assert receipt["managed_version_id"] == "v2"
    assert iam.default == "v2"
    assert set(iam.versions) == {"v1", "v2"}


def test_managed_guarded_cleanup_retries_delete_transport_failure():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _request(iam, operator._managed_state(iam, target), target=target)
    iam.fail_post_simulation = True
    iam.rollback_default_fails_before_once = True
    iam.delete_fails_before_once = True

    with pytest.raises(operator.OperationError, match="simulation rolled back"):
        _apply_managed(iam, request)

    assert iam.default == "v1"
    assert set(iam.versions) == {"v1"}


def test_managed_final_reread_preserves_unexpected_third_state():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _request(iam, operator._managed_state(iam, target), target=target)
    iam.third_state_on_passing_post_simulation = True

    with pytest.raises(operator.OperationError, match="final verification"):
        _apply_managed(iam, request)

    assert iam.default == "v3"
    assert operator._policy_hash(iam.versions["v3"]) == operator._policy_hash(THIRD)
    assert not any(event == ("default", "v1") for event in iam.events)


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


def test_local_command_surfaces_only_exact_allowlisted_remote_diagnostic(monkeypatch):
    def fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            returncode=1,
            stdout=b"",
            stderr=b"REMOTE_REBENCHMARK_IAM_ERROR:IAM_SIM_RESULT_SHAPE\n",
        )

    monkeypatch.setattr(operator.subprocess, "run", fake_run)
    with pytest.raises(operator.OperationError, match="IAM_SIM_RESULT_SHAPE"):
        operator._run(
            "test-command",
            redacted_error_codes=operator.REMOTE_DIAGNOSTIC_CODES,
        )


@pytest.mark.parametrize(
    "stderr",
    (
        b"REMOTE_REBENCHMARK_IAM_ERROR:UNKNOWN_CODE\n",
        b"REMOTE_REBENCHMARK_IAM_ERROR:IAM_SIM_RESULT_SHAPE\nssh noise\n",
        b"REMOTE_REBENCHMARK_IAM_ERROR:IAM_SIM_RESULT_SHAPE:private-value\n",
        b"arbitrary private resource material\n",
    ),
)
def test_local_command_never_surfaces_unknown_or_noisy_remote_stderr(
    monkeypatch, stderr
):
    def fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=1, stdout=b"", stderr=stderr)

    monkeypatch.setattr(operator.subprocess, "run", fake_run)
    with pytest.raises(operator.OperationError, match="^command failed: test-command$") as failure:
        operator._run(
            "test-command",
            redacted_error_codes=operator.REMOTE_DIAGNOSTIC_CODES,
        )
    assert stderr.decode("utf-8", errors="replace") not in str(failure.value)


def test_gateway_bridge_uses_isolated_production_python(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["input_value"] = kwargs["input_value"]
        return json.dumps(
            {
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
            }
        ).encode()

    monkeypatch.setattr(operator, "_validate_ssh_key", lambda: None)
    monkeypatch.setattr(operator, "_run", fake_run)

    receipt = operator._remote_call(
        "probe",
        None,
        commit=COMMIT,
        sources=[],
        source_hash=SOURCE_HASH,
    )

    assert receipt["status"] == "authority_ready"
    command = captured["args"][-1]
    assert isinstance(command, str)
    assert command.startswith(operator.REMOTE_PYTHON + " -I -c ")
    assert captured["input_value"]


def test_gateway_bridge_round_trips_validated_policy_request_as_wire_shape(
    monkeypatch,
):
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    external = _request(iam, operator._managed_state(iam, target), target=target)
    request = _bound_request(external)
    state = operator._managed_state(iam, target)
    receipt = operator._reconciliation_receipt(
        status="before",
        state=state,
        request=request,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
    )
    captured: dict[str, object] = {}

    def fake_run(*_args, **kwargs):
        payload = json.loads(kwargs["input_value"])
        captured["request"] = payload["request"]
        return operator._json(receipt).encode()

    monkeypatch.setattr(operator, "_validate_ssh_key", lambda: None)
    monkeypatch.setattr(operator, "_run", fake_run)

    observed = operator._remote_call(
        "reconcile",
        request,
        commit=COMMIT,
        sources=[],
        source_hash=SOURCE_HASH,
    )

    wire = captured["request"]
    assert isinstance(wire, dict)
    assert set(wire) == {
        "schema_version",
        "change_id",
        "target",
        "desired_document",
        "task_scope",
        "simulations",
        "plan",
        "prune_managed_version",
        "intent",
    }
    assert operator._validate_request(
        wire,
        commit=COMMIT,
        source_hash=SOURCE_HASH,
        require_intent=True,
    ) == request
    assert observed["status"] == "before"


def test_policy_wire_round_trip_allows_historical_plan_only_when_explicit():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    external = _request(iam, operator._managed_state(iam, target), target=target)
    plan = dict(external["plan"])
    plan["origin_main_sha"] = "c" * 40
    plan["bridge_source_hash"] = "sha256:" + "d" * 64
    plan_material = dict(plan)
    plan_material.pop("plan_hash")
    plan["plan_hash"] = operator._sha256_json(plan_material)
    external["plan"] = plan
    request = operator._validate_request(external)
    request["intent"] = _test_intent(plan)

    with pytest.raises(operator.OperationError, match="plan receipt differs"):
        operator._wire_policy_change_request(
            request,
            commit=COMMIT,
            source_hash=SOURCE_HASH,
        )

    wire = operator._wire_policy_change_request(
        request,
        commit=COMMIT,
        source_hash=SOURCE_HASH,
        allow_historical_plan=True,
    )
    assert operator._validate_request(wire, require_intent=True) == request


@pytest.mark.parametrize(
    ("plan_source_hash", "expected_before_only"),
    (
        ("sha256:" + "d" * 64, True),
        (SOURCE_HASH, False),
    ),
)
def test_remote_entry_routes_reconcile_by_exact_bridge_authority(
    monkeypatch,
    plan_source_hash: str,
    expected_before_only: bool,
):
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    external = _request(iam, operator._managed_state(iam, target), target=target)
    plan = dict(external["plan"])
    plan["origin_main_sha"] = "c" * 40
    plan["bridge_source_hash"] = plan_source_hash
    plan_material = dict(plan)
    plan_material.pop("plan_hash")
    plan["plan_hash"] = operator._sha256_json(plan_material)
    external["plan"] = plan
    normalized = operator._validate_request(external)
    normalized["intent"] = _test_intent(plan)
    wire = operator._wire_policy_change_request(
        normalized,
        commit=COMMIT,
        source_hash=SOURCE_HASH,
        allow_historical_plan=True,
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        operator, "_validate_setup_managed_authority", lambda _setup: {}
    )
    monkeypatch.setattr(
        operator,
        "_gateway_clients",
        lambda _setup: (FakeSts(), iam, ACCOUNT, CALLER),
    )
    monkeypatch.setattr(
        operator,
        "_validate_target_policy_authority",
        lambda *_args, **_kwargs: None,
    )

    def reconcile(_iam, request, *, before_only, **_kwargs):
        observed["request"] = request
        observed["before_only"] = before_only
        return {"status": "historical-before-only"}

    monkeypatch.setattr(operator, "_reconcile_policy", reconcile)

    assert operator._remote_entry(
        "reconcile",
        wire,
        {},
        source_hash=SOURCE_HASH,
        commit=COMMIT,
    ) == {"status": "historical-before-only"}
    assert observed["request"] == normalized
    assert observed["before_only"] is expected_before_only

    with pytest.raises(operator.OperationError, match="plan receipt differs"):
        operator._remote_entry(
            "apply",
            wire,
            {},
            source_hash=SOURCE_HASH,
            commit=COMMIT,
        )


def test_exact_source_gate_rejects_origin_alias_and_untracked_bytes(monkeypatch):
    monkeypatch.setattr(
        operator,
        "_git",
        lambda *args, **kwargs: b"git@github.com:leadpoet/leadpoet.git\n",
    )
    with pytest.raises(operator.OperationError, match="repository identity"):
        operator._exact_sources()

    def untracked_run(*args, **kwargs):
        command = args
        if command in {
            ("config", "--local", "--get", "remote.origin.url"),
            ("remote", "get-url", "origin"),
        }:
            return (operator.EXPECTED_ORIGIN_URL + "\n").encode()
        if command[0] == "fetch" or command[0] == "diff":
            return b""
        if command == ("rev-parse", "origin/main") or command == (
            "rev-parse",
            "HEAD",
        ):
            return (COMMIT + "\n").encode()
        if command[0] == "status":
            return b"?? unreviewed-policy.json\n"
        raise AssertionError(command)

    monkeypatch.setattr(operator, "_git", untracked_run)
    with pytest.raises(operator.OperationError, match="not pristine"):
        operator._exact_sources()


def test_request_reader_rejects_inherited_tcp_socket():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname())
    server, _ = listener.accept()
    try:
        with pytest.raises(operator.OperationError, match="local AF_UNIX"):
            operator._read_fd(server.fileno(), limit=1024)
    finally:
        server.close()
        client.close()
        listener.close()


def test_remote_receipt_validation_rejects_any_unexpected_output_field():
    authority = operator._remote_entry(
        "probe",
        None,
        _remote_setup(object()),
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
    request = _bound_request(
        _request(iam, operator._inline_state(iam, target), target=target)
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
    route = {
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
    }
    value = {
        "schema_version": operator.LEDGER_SCHEMA,
        "status": "running",
        "repo": str(tmp_path),
        "generation": 1,
        "control_run_id": "2" * 32,
        "invocation_id": "3" * 32,
        "started_at": "2026-08-23T00:00:00Z",
        "updated_at": "2026-08-23T00:00:01Z",
        "stages": [],
        "iam_authority_route": route,
        "iam_authority_routes": [route],
        "iam_policy_plan_history": [],
        "iam_policy_plans": [],
        "iam_policy_changes": [],
        "iam_policy_intents": [],
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


def test_active_ledger_shared_guard_blocks_a_stale_exclusive_writer(
    tmp_path: Path,
):
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{}", encoding="utf-8")
    ledger.chmod(0o600)
    lock_path = ledger.parent / f".{ledger.name}.lock"

    with operator._active_ledger_lock(ledger):
        descriptor = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)


def test_iam_execution_lock_is_exclusive(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    lock_path = ledger.parent / f".{ledger.name}.iam-operation.lock"

    with operator._iam_operation_lock(ledger):
        descriptor = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)


def test_duplicate_managed_apply_executes_aws_once_and_replays_redacted_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    ledger = tmp_path / "ledger.json"
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    raw_request = _request(iam, operator._managed_state(iam, target), target=target)
    request = _bound_request(raw_request)
    barrier = Barrier(2)
    calls: list[str] = []
    calls_lock = Lock()

    def remote_call(operation, call_request, **_kwargs):
        with calls_lock:
            calls.append(operation)
        return operator._apply_managed(
            iam,
            call_request,
            source_hash=SOURCE_HASH,
            commit=COMMIT,
            account_id=ACCOUNT,
            caller_arn=CALLER,
            prewrite_condition_documents_loader=(
                lambda _principal_arn: (
                    operator._canonical_policy(call_request["desired_document"]),
                )
            ),
        )

    monkeypatch.setattr(operator, "_remote_call", remote_call)

    def apply_once(_index: int) -> dict[str, object]:
        barrier.wait(timeout=2)
        with operator._iam_operation_lock(ledger):
            return operator._execute_intent_operation(
                "apply",
                request,
                ledger=ledger,
                commit=COMMIT,
                sources=[],
                source_hash=SOURCE_HASH,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = list(executor.map(apply_once, range(2)))

    assert receipts[0] == receipts[1]
    assert calls == ["apply"]
    assert iam.events == [("create", "v2"), ("default", "v2")]
    assert sorted(iam.versions) == ["v1", "v2"]
    record = operator._read_operation_record(ledger, request)
    assert record is not None
    assert record["state"] == "outcome"
    assert record["receipt"] == receipts[0]


def test_unknown_apply_outcome_requires_reconciliation_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    ledger = tmp_path / "ledger.json"
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    raw_request = _request(iam, operator._managed_state(iam, target), target=target)
    request = _bound_request(raw_request)
    calls: list[str] = []

    def lost_apply(operation, _request, **_kwargs):
        calls.append(operation)
        raise operator.OperationError("redacted lost apply outcome")

    monkeypatch.setattr(operator, "_remote_call", lost_apply)
    with operator._iam_operation_lock(ledger):
        with pytest.raises(operator.OperationError, match="lost apply outcome"):
            operator._execute_intent_operation(
                "apply",
                request,
                ledger=ledger,
                commit=COMMIT,
                sources=[],
                source_hash=SOURCE_HASH,
            )
    with operator._iam_operation_lock(ledger):
        with pytest.raises(operator.OperationError, match="reconciliation is required"):
            operator._execute_intent_operation(
                "apply",
                request,
                ledger=ledger,
                commit=COMMIT,
                sources=[],
                source_hash=SOURCE_HASH,
            )
    assert calls == ["apply"]
    pending = operator._read_operation_record(ledger, request)
    assert pending is not None and pending["state"] == "pending"

    def reconcile(operation, call_request, **_kwargs):
        calls.append(operation)
        state = operator._managed_state(iam, target)
        return operator._reconciliation_receipt(
            status="before",
            state=state,
            request=call_request,
            source_hash=SOURCE_HASH,
            commit=COMMIT,
            account_id=ACCOUNT,
            caller_arn=CALLER,
        )

    monkeypatch.setattr(operator, "_remote_call", reconcile)
    with operator._iam_operation_lock(ledger):
        first = operator._execute_intent_operation(
            "reconcile",
            request,
            ledger=ledger,
            commit=COMMIT,
            sources=[],
            source_hash=SOURCE_HASH,
        )
    with operator._iam_operation_lock(ledger):
        second = operator._execute_intent_operation(
            "reconcile",
            request,
            ledger=ledger,
            commit=COMMIT,
            sources=[],
            source_hash=SOURCE_HASH,
        )

    assert first == second
    assert first["status"] == "before"
    assert calls == ["apply", "reconcile"]


def test_historical_reconcile_never_persists_a_policy_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    ledger = tmp_path / "ledger.json"
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    external = _request(iam, operator._managed_state(iam, target), target=target)
    plan = dict(external["plan"])
    plan["origin_main_sha"] = "c" * 40
    plan["bridge_source_hash"] = "sha256:" + "d" * 64
    plan_material = dict(plan)
    plan_material.pop("plan_hash")
    plan["plan_hash"] = operator._sha256_json(plan_material)
    external["plan"] = plan
    request = _bound_request(external)
    desired = operator._canonical_policy(request["desired_document"])
    policy_receipt = operator._apply_managed(
        iam,
        request,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
        prewrite_condition_documents_loader=lambda _principal_arn: (desired,),
    )
    assert policy_receipt["schema_version"] == operator.RECEIPT_SCHEMA

    monkeypatch.setattr(operator, "_validate_ssh_key", lambda: None)
    monkeypatch.setattr(
        operator,
        "_run",
        lambda *_args, **_kwargs: operator._json(policy_receipt).encode(),
    )

    with pytest.raises(
        operator.OperationError, match="historical IAM reconciliation receipt schema"
    ):
        operator._execute_intent_operation(
            "reconcile",
            request,
            ledger=ledger,
            commit=COMMIT,
            sources=[],
            source_hash=SOURCE_HASH,
        )

    record = operator._read_operation_record(ledger, request)
    assert record is not None
    assert record["state"] == "pending"
    assert record["attempt_operation"] == "reconcile"
    assert record["receipt"] is None


def test_commit_only_reconcile_accepts_current_receipt_for_identical_authority():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    external = _request(iam, operator._managed_state(iam, target), target=target)
    plan = dict(external["plan"])
    plan["origin_main_sha"] = "c" * 40
    plan_material = dict(plan)
    plan_material.pop("plan_hash")
    plan["plan_hash"] = operator._sha256_json(plan_material)
    external["plan"] = plan
    request = _bound_request(external)
    desired = operator._canonical_policy(request["desired_document"])
    receipt = operator._apply_managed(
        iam,
        request,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
        prewrite_condition_documents_loader=lambda _principal_arn: (desired,),
    )

    validated = operator._validate_remote_receipt(
        "reconcile",
        receipt,
        request=request,
        commit=COMMIT,
        source_hash=SOURCE_HASH,
    )

    assert validated == receipt


def test_stale_generation_is_bypassed_only_by_fresh_read_only_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    ledger = tmp_path / "ledger.json"
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    old_request = _bound_request(
        _request(iam, operator._managed_state(iam, target), target=target)
    )
    operator._write_operation_record(
        ledger,
        old_request,
        attempt_operation="apply",
        receipt=None,
    )
    current_request = json.loads(json.dumps(old_request))
    current_request["intent"]["ledger_generation"] += 1
    current_commit = "c" * 40
    current_source = "sha256:" + "d" * 64
    calls: list[str] = []

    def reconcile(operation, call_request, **_kwargs):
        calls.append(operation)
        return operator._reconciliation_receipt(
            status="before",
            state=operator._managed_state(iam, target),
            request=call_request,
            source_hash=current_source,
            commit=current_commit,
            account_id=ACCOUNT,
            caller_arn=CALLER,
        )

    monkeypatch.setattr(operator, "_remote_call", reconcile)
    receipt = operator._execute_intent_operation(
        "reconcile",
        current_request,
        ledger=ledger,
        commit=current_commit,
        sources=[],
        source_hash=current_source,
    )

    assert receipt["status"] == "before"
    assert calls == ["reconcile"]
    record = operator._read_operation_record(ledger, current_request)
    assert record is not None
    assert record["attempt_generation"] == current_request["intent"][
        "ledger_generation"
    ]
    assert record["attempt_operation"] == "reconcile"
    assert record["receipt"]["schema_version"] == operator.RECONCILIATION_SCHEMA


@pytest.mark.parametrize(
    ("current_commit", "current_source"),
    (
        ("c" * 40, SOURCE_HASH),
        ("c" * 40, "sha256:" + "d" * 64),
    ),
)
def test_historical_reconcile_preserves_a_stale_valid_policy_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_commit: str,
    current_source: str,
) -> None:
    tmp_path.chmod(0o700)
    ledger = tmp_path / "ledger.json"
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    old_request = _bound_request(
        _request(iam, operator._managed_state(iam, target), target=target)
    )
    desired = operator._canonical_policy(old_request["desired_document"])
    receipt = operator._apply_managed(
        iam,
        old_request,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
        prewrite_condition_documents_loader=lambda _principal_arn: (desired,),
    )
    operator._write_operation_record(
        ledger,
        old_request,
        attempt_operation="apply",
        receipt=receipt,
    )
    record_path = operator._operation_record_path(ledger, old_request["intent"])
    before = record_path.read_bytes()
    current_request = json.loads(json.dumps(old_request))
    current_request["intent"]["ledger_generation"] += 1
    calls: list[str] = []
    monkeypatch.setattr(
        operator,
        "_remote_call",
        lambda operation, *_args, **_kwargs: calls.append(operation),
    )

    replay = operator._execute_intent_operation(
        "reconcile",
        current_request,
        ledger=ledger,
        commit=current_commit,
        sources=[],
        source_hash=current_source,
    )

    assert replay == receipt
    assert calls == []
    assert record_path.read_bytes() == before


def test_corrupt_stale_policy_outcome_fails_closed_without_remote_or_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    ledger = tmp_path / "ledger.json"
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    old_request = _bound_request(
        _request(iam, operator._managed_state(iam, target), target=target)
    )
    desired = operator._canonical_policy(old_request["desired_document"])
    receipt = operator._apply_managed(
        iam,
        old_request,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
        prewrite_condition_documents_loader=lambda _principal_arn: (desired,),
    )
    operator._write_operation_record(
        ledger,
        old_request,
        attempt_operation="apply",
        receipt=receipt,
    )
    record_path = operator._operation_record_path(ledger, old_request["intent"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["receipt"]["readback_document_hash"] = "sha256:" + "9" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    record_path.chmod(0o600)
    before = record_path.read_bytes()
    current_request = json.loads(json.dumps(old_request))
    current_request["intent"]["ledger_generation"] += 1
    calls: list[str] = []
    monkeypatch.setattr(
        operator,
        "_remote_call",
        lambda operation, *_args, **_kwargs: calls.append(operation),
    )

    with pytest.raises(operator.OperationError, match="policy receipt differs"):
        operator._execute_intent_operation(
            "reconcile",
            current_request,
            ledger=ledger,
            commit="c" * 40,
            sources=[],
            source_hash="sha256:" + "d" * 64,
        )

    assert calls == []
    assert record_path.read_bytes() == before


@pytest.mark.parametrize("record_kind", ("pending", "reconciliation"))
def test_stale_operation_record_can_never_trigger_another_remote_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_kind: str,
) -> None:
    tmp_path.chmod(0o700)
    ledger = tmp_path / "ledger.json"
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    old_request = _bound_request(
        _request(iam, operator._managed_state(iam, target), target=target)
    )
    receipt = None
    if record_kind == "reconciliation":
        receipt = operator._reconciliation_receipt(
            status="before",
            state=operator._managed_state(iam, target),
            request=old_request,
            source_hash=SOURCE_HASH,
            commit=COMMIT,
            account_id=ACCOUNT,
            caller_arn=CALLER,
        )
    operator._write_operation_record(
        ledger,
        old_request,
        attempt_operation=("apply" if receipt is None else "reconcile"),
        receipt=receipt,
    )
    current_request = json.loads(json.dumps(old_request))
    current_request["intent"]["ledger_generation"] += 1
    calls: list[str] = []
    monkeypatch.setattr(
        operator,
        "_remote_call",
        lambda operation, *_args, **_kwargs: calls.append(operation),
    )

    with pytest.raises(operator.OperationError, match="already attempted"):
        operator._execute_intent_operation(
            "apply",
            current_request,
            ledger=ledger,
            commit=COMMIT,
            sources=[],
            source_hash=SOURCE_HASH,
        )

    assert calls == []


def test_stale_policy_outcome_replays_without_another_remote_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    ledger = tmp_path / "ledger.json"
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    old_request = _bound_request(
        _request(iam, operator._managed_state(iam, target), target=target)
    )
    desired = operator._canonical_policy(old_request["desired_document"])
    receipt = operator._apply_managed(
        iam,
        old_request,
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
        prewrite_condition_documents_loader=lambda _principal_arn: (desired,),
    )
    operator._write_operation_record(
        ledger,
        old_request,
        attempt_operation="apply",
        receipt=receipt,
    )
    current_request = json.loads(json.dumps(old_request))
    current_request["intent"]["ledger_generation"] += 1
    calls: list[str] = []
    monkeypatch.setattr(
        operator,
        "_remote_call",
        lambda operation, *_args, **_kwargs: calls.append(operation),
    )

    replay = operator._execute_intent_operation(
        "apply",
        current_request,
        ledger=ledger,
        commit=COMMIT,
        sources=[],
        source_hash=SOURCE_HASH,
    )

    assert replay == receipt
    assert calls == []


def test_active_ledger_apply_requires_exact_run_generation_and_intent(
    tmp_path: Path,
) -> None:
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    plan = request["plan"]
    intent = _test_intent(plan)
    route = {
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
    }
    value = {
        "schema_version": operator.LEDGER_SCHEMA,
        "status": "running",
        "repo": str(tmp_path),
        "generation": intent["ledger_generation"],
        "control_run_id": intent["control_run_id"],
        "invocation_id": intent["invocation_id"],
        "started_at": "2026-08-23T00:00:00Z",
        "updated_at": "2026-08-23T00:00:01Z",
        "stages": [],
        "iam_authority_route": route,
        "iam_authority_routes": [route],
        "iam_policy_plan_history": [],
        "iam_policy_plans": [plan],
        "iam_policy_changes": [],
        "iam_policy_intents": [intent],
        "iam_never_pause_invariant": {
            "schema_version": operator.NEVER_PAUSE_SCHEMA,
            "status": "enforced",
            "local_chain_failure_disposition": "ignored_non_authority",
            "blocks_recovery": False,
            "operator_iam_request_allowed": False,
        },
    }
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(value), encoding="utf-8")
    ledger.chmod(0o600)

    loaded = operator._validate_active_ledger(
        ledger,
        commit=COMMIT,
        source_hash=SOURCE_HASH,
        required_plan_hash=plan["plan_hash"],
        require_intent=True,
    )
    assert loaded["iam_policy_intents"] == [intent]

    value["generation"] += 1
    ledger.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(operator.OperationError, match="intent ledger gate differs"):
        operator._validate_active_ledger(
            ledger,
            commit=COMMIT,
            source_hash=SOURCE_HASH,
            required_plan_hash=plan["plan_hash"],
            require_intent=True,
        )


def test_active_ledger_reconcile_accepts_only_retained_historical_plan_route(
    tmp_path: Path,
) -> None:
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    plan = request["plan"]
    intent = _test_intent(plan)
    current_commit = "c" * 40
    current_source = "sha256:" + "d" * 64

    def route(commit: str, source_hash: str) -> dict[str, object]:
        return {
            "schema_version": operator.AUTHORITY_SCHEMA,
            "status": "authority_ready",
            "origin_main_sha": commit,
            "bridge_source_hash": source_hash,
            "account_id": ACCOUNT,
            "caller_arn": CALLER,
            "route": "gateway_bridge",
            "local_chain": "ignored_non_authority",
            "secret_values_printed": False,
            "policy_material_printed": False,
        }

    intent["ledger_generation"] = 6
    value = {
        "schema_version": operator.LEDGER_SCHEMA,
        "status": "running",
        "repo": str(tmp_path),
        "generation": 6,
        "control_run_id": intent["control_run_id"],
        "invocation_id": intent["invocation_id"],
        "started_at": "2026-08-23T00:00:00Z",
        "updated_at": "2026-08-23T00:00:01Z",
        "stages": [],
        "iam_authority_route": route(current_commit, current_source),
        "iam_authority_routes": [
            route(COMMIT, SOURCE_HASH),
            route(current_commit, current_source),
        ],
        "iam_policy_plan_history": [],
        "iam_policy_plans": [plan],
        "iam_policy_changes": [],
        "iam_policy_intents": [intent],
        "iam_never_pause_invariant": {
            "schema_version": operator.NEVER_PAUSE_SCHEMA,
            "status": "enforced",
            "local_chain_failure_disposition": "ignored_non_authority",
            "blocks_recovery": False,
            "operator_iam_request_allowed": False,
        },
    }
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(value), encoding="utf-8")
    ledger.chmod(0o600)

    loaded = operator._validate_active_ledger(
        ledger,
        commit=current_commit,
        source_hash=current_source,
        required_plan_hash=plan["plan_hash"],
        require_intent=True,
        allow_historical_plan=True,
    )
    assert loaded["iam_policy_intents"] == [intent]

    with pytest.raises(operator.OperationError, match="plan receipt differs"):
        operator._validate_active_ledger(
            ledger,
            commit=current_commit,
            source_hash=current_source,
            required_plan_hash=plan["plan_hash"],
            require_intent=True,
        )

    value["iam_authority_routes"] = [route(current_commit, current_source)]
    ledger.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(operator.OperationError, match="historical authority"):
        operator._validate_active_ledger(
            ledger,
            commit=current_commit,
            source_hash=current_source,
            required_plan_hash=plan["plan_hash"],
            require_intent=True,
            allow_historical_plan=True,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"status": "completed"}, "lifecycle"),
        ({"status": "stop_requested"}, "lifecycle"),
        (
            {
                "last_reconciliation_status": "ambiguous",
                "last_reconciled_at": None,
            },
            "lifecycle",
        ),
    ),
)
def test_intent_lifecycle_is_independently_validated(
    overrides: dict[str, object], message: str
) -> None:
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    intent = _test_intent(request["plan"])
    intent.update(overrides)

    with pytest.raises(operator.OperationError, match=message):
        operator._validate_intent(intent, request["plan"])


def test_completed_plan_cannot_be_replayed_through_active_ledger(tmp_path: Path):
    iam = FakeInlineIam()
    target = {"kind": "inline_role", "role_name": ROLE, "policy_name": POLICY_NAME}
    request = _request(iam, operator._inline_state(iam, target), target=target)
    receipt = _apply_inline(iam, request)
    route = {
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
    }
    value = {
        "schema_version": operator.LEDGER_SCHEMA,
        "status": "running",
        "repo": str(tmp_path),
        "generation": 4,
        "control_run_id": "2" * 32,
        "invocation_id": "3" * 32,
        "started_at": "2026-08-23T00:00:00Z",
        "updated_at": "2026-08-23T00:00:01Z",
        "stages": [],
        "iam_authority_route": route,
        "iam_authority_routes": [route],
        "iam_policy_plan_history": [],
        "iam_policy_plans": [request["plan"]],
        "iam_policy_changes": [receipt],
        "iam_policy_intents": [],
        "iam_never_pause_invariant": {
            "schema_version": operator.NEVER_PAUSE_SCHEMA,
            "status": "enforced",
            "local_chain_failure_disposition": "ignored_non_authority",
            "blocks_recovery": False,
            "operator_iam_request_allowed": False,
        },
    }
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(value), encoding="utf-8")
    ledger.chmod(0o600)

    with pytest.raises(operator.OperationError, match="already completed"):
        operator._validate_active_ledger(
            ledger,
            commit=COMMIT,
            source_hash=SOURCE_HASH,
            required_plan_hash=request["plan"]["plan_hash"],
        )


def test_iam_operator_is_bound_into_production_parity_source_commitments():
    assert operator.OPERATOR_PATH in contract_builder.ALWAYS_COMMITTED_PATHS
