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
    return operator._apply_managed(
        iam,
        _bound_request(request),
        source_hash=SOURCE_HASH,
        commit=COMMIT,
        account_id=ACCOUNT,
        caller_arn=CALLER,
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

    assert scope["statement_changes"][0]["sid"] == "index:0"
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


def test_managed_postwrite_simulation_failure_restores_prior_default():
    iam = FakeManagedIam()
    target = {"kind": "managed", "policy_arn": MANAGED_ARN}
    request = _request(iam, operator._managed_state(iam, target), target=target)
    iam.fail_post_simulation = True
    with pytest.raises(operator.OperationError, match="simulation rolled back"):
        _apply_managed(iam, request)
    assert iam.default == "v1"
    assert set(iam.versions) == {"v1"}


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
