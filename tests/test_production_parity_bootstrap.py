from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

from botocore.exceptions import ClientError
import pytest
import yaml

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


def _controller_policy() -> dict:
    return setup._controller_policy(
        account_id=ACCOUNT,
        region=setup.EXPECTED_REGION,
        production_secret_id=setup.PRODUCTION_GATEWAY_SECRET_ID,
        readonly_secret_id=setup.READONLY_DSN_SECRET_ID,
        miner_intake_secret_id=setup.DEFAULT_MINER_INTAKE_SECRET_ID,
        runner_arn=f"arn:aws:iam::{ACCOUNT}:role/{setup.RUNNER_ROLE}",
    )


def _matches(value: str, expected: object, *, pattern: bool) -> bool:
    values = expected if isinstance(expected, list) else [expected]
    return any(
        fnmatchcase(value, str(item)) if pattern else value == str(item)
        for item in values
    )


def _condition_matches(
    condition: dict[str, object], context: dict[str, object]
) -> bool:
    for operator, entries in condition.items():
        assert isinstance(entries, dict)
        for key, expected in entries.items():
            if key not in context:
                return False
            actual = context[key]
            actual_values = actual if isinstance(actual, list) else [actual]
            if operator == "ForAllValues:StringEquals":
                expected_values = (
                    expected if isinstance(expected, list) else [expected]
                )
                if not all(str(value) in expected_values for value in actual_values):
                    return False
            elif operator == "ForAnyValue:StringEquals":
                expected_values = (
                    expected if isinstance(expected, list) else [expected]
                )
                if not any(str(value) in expected_values for value in actual_values):
                    return False
            elif operator in {"StringEquals", "ArnEquals", "Bool"}:
                if not all(
                    _matches(str(value), expected, pattern=False)
                    for value in actual_values
                ):
                    return False
            elif operator in {"StringLike", "ArnLike"}:
                if not all(_matches(str(value), expected, pattern=True) for value in actual_values):
                    return False
            elif operator == "NumericEquals":
                if not all(float(value) == float(expected) for value in actual_values):
                    return False
            elif operator == "DateLessThan":
                if not all(str(value) < str(expected) for value in actual_values):
                    return False
            elif operator == "DateGreaterThanEquals":
                if not all(str(value) >= str(expected) for value in actual_values):
                    return False
            else:
                raise AssertionError(f"unsupported condition operator {operator}")
    return True


def _policy_allows(
    policy: dict,
    action: str,
    resource: str,
    context: dict[str, object],
) -> bool:
    allowed = False
    for statement in policy["Statement"]:
        actions = statement["Action"]
        actions = actions if isinstance(actions, list) else [actions]
        if not any(fnmatchcase(action.lower(), str(item).lower()) for item in actions):
            continue
        if "Resource" in statement:
            resources = statement["Resource"]
            resources = resources if isinstance(resources, list) else [resources]
            if not any(fnmatchcase(resource, str(item)) for item in resources):
                continue
        else:
            resources = statement["NotResource"]
            resources = resources if isinstance(resources, list) else [resources]
            if any(fnmatchcase(resource, str(item)) for item in resources):
                continue
        if not _condition_matches(statement.get("Condition", {}), context):
            continue
        if statement["Effect"] == "Deny":
            return False
        allowed = True
    return allowed


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
    static_bases = {
        (
            f"arn:aws:secretsmanager:us-east-1:{ACCOUNT}:secret:"
            f"{secret_id}"
        )
        for secret_id in (
            setup.READONLY_DSN_SECRET_ID,
            setup.DEFAULT_MINER_INTAKE_SECRET_ID,
        )
    }
    expected_static_resources = static_bases | {
        base + "-??????" for base in static_bases
    }
    read_static = next(
        item for item in static["Statement"]
        if item["Effect"] == "Allow"
        and "secretsmanager:GetSecretValue" in (
            item["Action"] if isinstance(item["Action"], list) else [item["Action"]]
        )
    )
    assert set(read_static["Resource"]) == expected_static_resources
    tag_static = next(
        item for item in static["Statement"]
        if item["Effect"] == "Allow"
        and item["Action"] == "secretsmanager:TagResource"
    )
    assert set(tag_static["Resource"]) == expected_static_resources
    context = {"aws:CurrentTime": "2026-08-18T12:00:00Z"}
    for base in static_bases:
        assert _policy_allows(
            static, "secretsmanager:GetSecretValue", base, context
        )
        assert _policy_allows(
            static,
            "secretsmanager:GetSecretValue",
            base + "-ABC123",
            context,
        )
        assert not _policy_allows(
            static,
            "secretsmanager:GetSecretValue",
            base + "-adjacent-ABC123",
            context,
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

    runner = setup._runner_policy(
        account_id=ACCOUNT,
        region="us-east-1",
        production_secret_id=setup.PRODUCTION_GATEWAY_SECRET_ID,
        readonly_secret_id=setup.READONLY_DSN_SECRET_ID,
        miner_intake_secret_id=setup.DEFAULT_MINER_INTAKE_SECRET_ID,
    )
    attested_version = (
        "arn:aws:s3:::leadpoet-attested-v2-artifacts-493765492819/"
        "encrypted-artifacts/credential.json"
    )
    unrelated_object = (
        "arn:aws:s3:::unrelated-artifacts-493765492819/"
        "current.json"
    )
    assert _policy_allows(
        runner, "s3:GetObjectVersion", attested_version, {}
    )
    assert not _policy_allows(
        runner, "s3:GetObjectVersion", unrelated_object, {}
    )
    assert _policy_allows(runner, "s3:GetObject", attested_version, {})
    assert not _policy_allows(runner, "s3:GetObject", unrelated_object, {})
    run_object = (
        f"arn:aws:s3:::leadpoet-parity-{ACCOUNT}-fixture/"
        "production-parity/runs/pp-1-1/full-evidence.json"
    )
    assert _policy_allows(
        runner, "s3:GetObjectRetention", attested_version, {}
    )
    assert _policy_allows(runner, "s3:GetObjectRetention", run_object, {})
    assert not _policy_allows(
        runner, "s3:GetObjectRetention", unrelated_object, {}
    )
    run_checkpoint = (
        f"arn:aws:s3:::leadpoet-parity-{ACCOUNT}-fixture/"
        "production-parity/runs/pp-1-1/baseline-checkpoints/"
        f"{COMMIT}/checkpoint.json"
    )
    assert _policy_allows(runner, "s3:PutObject", run_checkpoint, {})


def test_controller_managed_policy_partition_and_adversarial_boundaries():
    policy = _controller_policy()
    slices = setup._controller_policy_slices(
        account_id=ACCOUNT,
        region=setup.EXPECTED_REGION,
        production_secret_id=setup.PRODUCTION_GATEWAY_SECRET_ID,
        readonly_secret_id=setup.READONLY_DSN_SECRET_ID,
        miner_intake_secret_id=setup.DEFAULT_MINER_INTAKE_SECRET_ID,
        runner_arn=f"arn:aws:iam::{ACCOUNT}:role/{setup.RUNNER_ROLE}",
    )
    assert set(slices) == set(setup.CONTROLLER_POLICY_NAMES.values())
    assert len(slices) <= 10
    assert all(len(setup._json(document)) <= 6144 for document in slices.values())
    partitioned = [
        setup._json(statement)
        for document in slices.values()
        for statement in document["Statement"]
    ]
    assert sorted(partitioned) == sorted(
        setup._json(statement) for statement in policy["Statement"]
    )

    restricted_writes = {
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:CreateSecurityGroup",
        "ec2:CreateTags",
        "ec2:DeleteSecurityGroup",
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ssm:SendCommand",
        "cloudfront:TagResource",
        "cloudfront:UpdateDistribution",
        "cloudfront:DeleteDistribution",
    }
    for statement in policy["Statement"]:
        actions = statement["Action"]
        actions = actions if isinstance(actions, list) else [actions]
        assert not (
            restricted_writes.intersection(actions)
            and statement["Resource"] == "*"
        )
    create_distribution = next(
        statement
        for statement in policy["Statement"]
        if statement["Action"] == "cloudfront:CreateDistribution"
    )
    assert create_distribution["Resource"] == "*"
    assert create_distribution["Condition"]["ForAllValues:StringEquals"][
        "aws:TagKeys"
    ] == [
        "Name",
        "leadpoet:candidate-sha",
        "leadpoet:ephemeral",
        "leadpoet:parity-run",
    ]

    prefix = f"arn:aws:ec2:{setup.EXPECTED_REGION}:{ACCOUNT}"
    instance = prefix + ":instance/i-0123456789abcdef0"
    volume = prefix + ":volume/vol-0123456789abcdef0"
    interface = prefix + ":network-interface/eni-0123456789abcdef0"
    group = prefix + ":security-group/sg-0123456789abcdef0"
    subnet = prefix + f":subnet/{setup.PRODUCTION_SUBNET_ID}"
    vpc = prefix + f":vpc/{setup.PRODUCTION_VPC_ID}"
    image = (
        f"arn:aws:ec2:{setup.EXPECTED_REGION}::image/"
        f"{setup.PRODUCTION_AMI_ID}"
    )
    distribution = (
        f"arn:aws:cloudfront::{ACCOUNT}:distribution/E123456789ABCD"
    )
    tags = {
        "Name": "leadpoet-parity-pp-123456-1",
        "leadpoet:candidate-sha": "a" * 40,
        "leadpoet:ephemeral": "true",
        "leadpoet:parity-run": "pp-123456-1",
    }
    request = {
        **{f"aws:RequestTag/{key}": value for key, value in tags.items()},
        "aws:TagKeys": sorted(tags),
    }
    owned_ec2 = {
        **{f"ec2:ResourceTag/{key}": value for key, value in tags.items()},
        "ec2:Vpc": vpc,
    }
    launch = {
        **request,
        **owned_ec2,
        "ec2:AssociatePublicIpAddress": "true",
        "ec2:Encrypted": "true",
        "ec2:InstanceMetadataTags": "enabled",
        "ec2:InstanceProfile": (
            f"arn:aws:iam::{ACCOUNT}:instance-profile/{setup.RUNNER_PROFILE}"
        ),
        "ec2:InstanceType": setup.PRODUCTION_INSTANCE_TYPE,
        "ec2:MetadataHttpEndpoint": "enabled",
        "ec2:MetadataHttpPutResponseHopLimit": "2",
        "ec2:MetadataHttpTokens": "required",
        "ec2:Region": setup.EXPECTED_REGION,
        "ec2:Subnet": subnet,
        "ec2:VolumeSize": "512",
        "ec2:VolumeType": "gp3",
    }
    assert _policy_allows(policy, "ec2:RunInstances", image, launch)
    assert _policy_allows(policy, "ec2:RunInstances", subnet, launch)
    assert _policy_allows(policy, "ec2:RunInstances", instance, launch)
    assert _policy_allows(policy, "ec2:RunInstances", volume, launch)
    assert _policy_allows(policy, "ec2:RunInstances", interface, launch)
    assert _policy_allows(policy, "ec2:RunInstances", group, launch)
    assert not _policy_allows(
        policy,
        "ec2:RunInstances",
        f"arn:aws:ec2:{setup.EXPECTED_REGION}::image/ami-0123456789abcdef0",
        launch,
    )
    assert not _policy_allows(
        policy,
        "ec2:RunInstances",
        prefix + ":subnet/subnet-0123456789abcdef0",
        launch,
    )
    assert not _policy_allows(
        policy,
        "ec2:RunInstances",
        instance,
        {**launch, "ec2:InstanceProfile": "arn:aws:iam::493765492819:instance-profile/other"},
    )
    assert not _policy_allows(
        policy,
        "ec2:RunInstances",
        volume,
        {**launch, "ec2:VolumeSize": "200"},
    )
    assert not _policy_allows(
        policy,
        "ec2:RunInstances",
        volume,
        {**launch, "ec2:Encrypted": "false"},
    )
    assert not _policy_allows(
        policy,
        "ec2:RunInstances",
        instance,
        {**launch, "ec2:MetadataHttpTokens": "optional"},
    )
    assert not _policy_allows(
        policy,
        "ec2:RunInstances",
        interface,
        {
            **launch,
            "ec2:Subnet": prefix + ":subnet/subnet-0123456789abcdef0",
        },
    )
    assert not _policy_allows(
        policy,
        "ec2:RunInstances",
        group,
        {**launch, "ec2:ResourceTag/leadpoet:ephemeral": "false"},
    )
    missing_sha = dict(launch)
    missing_sha.pop("aws:RequestTag/leadpoet:candidate-sha")
    assert not _policy_allows(
        policy, "ec2:RunInstances", instance, missing_sha
    )

    assert _policy_allows(
        policy,
        "ec2:CreateTags",
        instance,
        {**request, "ec2:CreateAction": "RunInstances"},
    )
    assert not _policy_allows(
        policy,
        "ec2:CreateTags",
        instance,
        {**request, "ec2:CreateAction": "CreateSecurityGroup"},
    )
    assert _policy_allows(
        policy,
        "ec2:CreateTags",
        group,
        {**request, "ec2:CreateAction": "CreateSecurityGroup"},
    )
    assert _policy_allows(
        policy, "ec2:CreateSecurityGroup", group, request
    )
    assert _policy_allows(
        policy, "ec2:CreateSecurityGroup", vpc, request
    )
    assert not _policy_allows(
        policy,
        "ec2:CreateSecurityGroup",
        prefix + ":vpc/vpc-0123456789abcdef0",
        request,
    )
    for action, resource in (
        ("ec2:TerminateInstances", instance),
        ("ec2:DeleteSecurityGroup", group),
        ("ec2:AuthorizeSecurityGroupIngress", group),
    ):
        assert _policy_allows(policy, action, resource, owned_ec2)
        assert not _policy_allows(
            policy,
            action,
            resource,
            {**owned_ec2, "ec2:ResourceTag/leadpoet:ephemeral": "false"},
        )

    document = (
        f"arn:aws:ssm:{setup.EXPECTED_REGION}::document/AWS-RunShellScript"
    )
    ssm_owned = {
        **{f"ssm:resourceTag/{key}": value for key, value in tags.items()},
    }
    assert _policy_allows(policy, "ssm:SendCommand", document, ssm_owned)
    assert _policy_allows(policy, "ssm:SendCommand", instance, ssm_owned)
    assert not _policy_allows(
        policy,
        "ssm:SendCommand",
        f"arn:aws:ssm:{setup.EXPECTED_REGION}::document/Other",
        ssm_owned,
    )
    assert not _policy_allows(
        policy,
        "ssm:SendCommand",
        instance,
        {**ssm_owned, "ssm:resourceTag/leadpoet:ephemeral": "false"},
    )

    owned_distribution = {
        **{f"aws:ResourceTag/{key}": value for key, value in tags.items()},
    }
    assert _policy_allows(
        policy,
        "cloudfront:TagResource",
        distribution,
        {**request, **owned_distribution},
    )
    assert not _policy_allows(
        policy,
        "cloudfront:TagResource",
        distribution,
        {
            **request,
            **owned_distribution,
            "aws:ResourceTag/leadpoet:ephemeral": "false",
        },
    )
    for action in (
        "cloudfront:UpdateDistribution",
        "cloudfront:DeleteDistribution",
    ):
        assert _policy_allows(policy, action, distribution, owned_distribution)
        assert not _policy_allows(
            policy,
            action,
            distribution,
            {
                **owned_distribution,
                "aws:ResourceTag/leadpoet:ephemeral": "false",
            },
        )

    runner = f"arn:aws:iam::{ACCOUNT}:role/{setup.RUNNER_ROLE}"
    pass_context = {
        "iam:AssociatedResourceArn": instance,
        "iam:PassedToService": "ec2.amazonaws.com",
    }
    assert _policy_allows(policy, "iam:PassRole", runner, pass_context)
    assert not _policy_allows(
        policy,
        "iam:PassRole",
        runner,
        {**pass_context, "iam:PassedToService": "lambda.amazonaws.com"},
    )
    assert not _policy_allows(
        policy,
        "iam:PassRole",
        f"arn:aws:iam::{ACCOUNT}:role/other",
        pass_context,
    )


def test_setup_simulator_executes_positive_and_adversarial_policy_matrix():
    policy = _controller_policy()
    revoke_policy = setup._revoke_older_sessions_policy(
        cutoff="2026-08-18T12:00:00Z"
    )
    policy_documents = [policy, revoke_policy]

    class IAM:
        calls = 0

        def simulate_principal_policy(self, **kwargs):
            self.calls += 1
            assert kwargs["ResourceOwner"] == f"arn:aws:iam::{ACCOUNT}:root"
            context: dict[str, object] = {}
            for item in kwargs["ContextEntries"]:
                values = item["ContextKeyValues"]
                context[item["ContextKeyName"]] = (
                    values if item["ContextKeyType"].endswith("List") else values[0]
                )
            action = kwargs["ActionNames"][0]
            return {
                "IsTruncated": False,
                "EvaluationResults": [
                    {
                        "EvalActionName": action,
                        "EvalResourceName": resource,
                        "EvalDecision": (
                            "allowed"
                            if _policy_allows(policy, action, resource, context)
                            else "implicitDeny"
                        ),
                        "MissingContextValues": (
                            ["aws:RequestTag/leadpoet:candidate-sha"]
                            if action == "ec2:RunInstances"
                            and resource.endswith(":instance/i-0123456789abcdef0")
                            and "aws:RequestTag/leadpoet:candidate-sha"
                            not in context
                            else []
                        ),
                    }
                    for resource in kwargs["ResourceArns"]
                ],
            }

    iam = IAM()
    setup._simulate_controller_policy(
        iam,
        account_id=ACCOUNT,
        region=setup.EXPECTED_REGION,
        controller_arn=f"arn:aws:iam::{ACCOUNT}:role/{setup.CONTROLLER_ROLE}",
        runner_arn=f"arn:aws:iam::{ACCOUNT}:role/{setup.RUNNER_ROLE}",
        session_cutoff="2026-08-18T12:00:00Z",
        policy_documents=policy_documents,
    )
    assert iam.calls >= 30

    class AWSShaped(IAM):
        first_specific_count = 0

        def simulate_principal_policy(self, **kwargs):
            response = super().simulate_principal_policy(**kwargs)
            flat = response["EvaluationResults"]
            decisions = {item["EvalDecision"] for item in flat}
            assert len(decisions) == 1
            if self.calls == 1:
                self.first_specific_count = len(flat)
            unrelated = (
                {"ssm:resourceTag/Name"}
                if self.calls == 2 else set()
            )
            missing = {
                value
                for item in flat
                for value in item["MissingContextValues"]
            } | unrelated
            return {
                "IsTruncated": False,
                "EvaluationResults": [{
                    "EvalActionName": kwargs["ActionNames"][0],
                    "EvalResourceName": "*",
                    "EvalDecision": decisions.pop(),
                    "MissingContextValues": sorted(missing),
                    "ResourceSpecificResults": [{
                        "EvalResourceName": item["EvalResourceName"],
                        "EvalResourceDecision": item["EvalDecision"],
                        "MissingContextValues": sorted(
                            set(item["MissingContextValues"]) | unrelated
                        ),
                    } for item in flat],
                }],
            }

    aws_shaped = AWSShaped()
    setup._simulate_controller_policy(
        aws_shaped,
        account_id=ACCOUNT,
        region=setup.EXPECTED_REGION,
        controller_arn=f"arn:aws:iam::{ACCOUNT}:role/{setup.CONTROLLER_ROLE}",
        runner_arn=f"arn:aws:iam::{ACCOUNT}:role/{setup.RUNNER_ROLE}",
        session_cutoff="2026-08-18T12:00:00Z",
        policy_documents=policy_documents,
    )
    assert aws_shaped.calls >= 30
    assert aws_shaped.first_specific_count == 6

    class Broken(IAM):
        def simulate_principal_policy(self, **kwargs):
            response = super().simulate_principal_policy(**kwargs)
            if self.calls == 1:
                response["EvaluationResults"][0]["EvalDecision"] = "implicitDeny"
            return response

    with pytest.raises(setup.SetupError, match="simulation launch-positive differs"):
        setup._simulate_controller_policy(
            Broken(),
            account_id=ACCOUNT,
            region=setup.EXPECTED_REGION,
            controller_arn=(
                f"arn:aws:iam::{ACCOUNT}:role/{setup.CONTROLLER_ROLE}"
            ),
            runner_arn=f"arn:aws:iam::{ACCOUNT}:role/{setup.RUNNER_ROLE}",
            session_cutoff="2026-08-18T12:00:00Z",
            policy_documents=policy_documents,
        )


def test_simulator_normalizes_aws_resource_specific_results():
    image = f"arn:aws:ec2:{setup.EXPECTED_REGION}::image/{setup.PRODUCTION_AMI_ID}"
    instance = (
        f"arn:aws:ec2:{setup.EXPECTED_REGION}:{ACCOUNT}:"
        "instance/i-0123456789abcdef0"
    )
    decisions, missing = setup._normalize_simulation_results(
        [{
            "EvalActionName": "ec2:RunInstances",
            "EvalResourceName": "*",
            "EvalDecision": "allowed",
            "MissingContextValues": [],
            "ResourceSpecificResults": [
                {
                    "EvalResourceName": image,
                    "EvalResourceDecision": "allowed",
                    "MissingContextValues": [],
                },
                {
                    "EvalResourceName": instance,
                    "EvalResourceDecision": "allowed",
                    "MissingContextValues": [],
                },
            ],
        }],
        action="ec2:RunInstances",
        requested_resources=[image, instance],
    )
    assert decisions == {image: "allowed", instance: "allowed"}
    assert missing == set()


@pytest.mark.parametrize(
    "results",
    (
        [],
        [{
            "EvalActionName": "ec2:RunInstances",
            "EvalDecision": "allowed",
            "MissingContextValues": [],
            "ResourceSpecificResults": "invalid",
        }],
        [{
            "EvalActionName": "ec2:RunInstances",
            "EvalDecision": "allowed",
            "MissingContextValues": [],
            "ResourceSpecificResults": [{
                "EvalResourceName": "arn:aws:ec2:us-east-1:1:instance/i-1",
                "MissingContextValues": [],
            }],
        }],
    ),
)
def test_simulator_rejects_malformed_results(results):
    with pytest.raises(setup.SetupError, match="simulation response differs"):
        setup._normalize_simulation_results(
            results,
            action="ec2:RunInstances",
            requested_resources=["arn:aws:ec2:us-east-1:1:instance/i-1"],
        )


def test_simulator_rejects_duplicate_resource_results():
    resource = "arn:aws:ec2:us-east-1:1:instance/i-1"
    with pytest.raises(setup.SetupError, match="simulation response differs"):
        setup._normalize_simulation_results(
            [{
                "EvalActionName": "ec2:RunInstances",
                "EvalDecision": "allowed",
                "MissingContextValues": [],
                "ResourceSpecificResults": [
                    {
                        "EvalResourceName": resource,
                        "EvalResourceDecision": "allowed",
                        "MissingContextValues": [],
                    },
                    {
                        "EvalResourceName": resource,
                        "EvalResourceDecision": "allowed",
                        "MissingContextValues": [],
                    },
                ],
            }],
            action="ec2:RunInstances",
            requested_resources=[resource],
        )


def test_simulator_rejects_legacy_multi_resource_aggregate_denial():
    resources = [
        "arn:aws:s3:::leadpoet-one/object",
        "arn:aws:s3:::leadpoet-two/object",
    ]
    with pytest.raises(setup.SetupError, match="simulation response differs"):
        setup._normalize_simulation_results(
            [
                {
                    "EvalActionName": "s3:GetObject",
                    "EvalResourceName": resource,
                    "EvalDecision": "implicitDeny",
                }
                for resource in resources
            ],
            action="s3:GetObject",
            requested_resources=resources,
        )


def test_simulator_accepts_most_restrictive_resource_specific_aggregate():
    allowed = "arn:aws:s3:::leadpoet-one/object"
    denied = "arn:aws:s3:::leadpoet-two/object"
    decisions, missing = setup._normalize_simulation_results(
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


def test_simulator_accepts_safe_aggregate_only_decisions():
    resources = [
        "arn:aws:s3:::leadpoet-one/object",
        "arn:aws:s3:::leadpoet-two/object",
    ]
    decisions, missing = setup._normalize_simulation_results(
        [
            {
                "EvalActionName": "S3:gEToBJECT",
                "EvalResourceName": (
                    "arn:${Partition}:s3:::${BucketName}/${KeyName}"
                ),
                "EvalDecision": "allowed",
            }
        ],
        action="s3:GetObject",
        requested_resources=resources,
    )
    assert decisions == {resource: "allowed" for resource in resources}
    assert missing == set()

    decisions, missing = setup._normalize_simulation_results(
        [{"EvalActionName": "s3:GetObject", "EvalDecision": "implicitDeny"}],
        action="s3:GetObject",
        requested_resources=[resources[0]],
    )
    assert decisions == {resources[0]: "implicitDeny"}
    assert missing == set()


@pytest.mark.parametrize("mixed", ("representation", "decision"))
def test_simulator_rejects_mixed_results(mixed):
    resource = "arn:aws:ec2:us-east-1:1:instance/i-1"
    aggregate = {
        "EvalActionName": "ec2:RunInstances",
        "EvalResourceName": "*",
        "EvalDecision": "allowed",
        "MissingContextValues": [],
        "ResourceSpecificResults": [{
            "EvalResourceName": resource,
            "EvalResourceDecision": (
                "implicitDeny" if mixed == "decision" else "allowed"
            ),
            "MissingContextValues": [],
        }],
    }
    results = [aggregate]
    if mixed == "representation":
        results.append({
            "EvalActionName": "ec2:RunInstances",
            "EvalResourceName": "arn:aws:ec2:us-east-1:1:volume/vol-1",
            "EvalDecision": "allowed",
            "MissingContextValues": [],
        })
    with pytest.raises(setup.SetupError, match="simulation response differs"):
        setup._normalize_simulation_results(
            results,
            action="ec2:RunInstances",
            requested_resources=[resource],
        )


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
        setup,
        "_ensure_oidc_provider",
        lambda *_: (
            f"arn:aws:iam::{ACCOUNT}:oidc-provider/"
            "token.actions.githubusercontent.com"
        ),
    )

    def ensure_role(_iam, *, name, **kwargs):
        receipt = kwargs.get("revocation_receipt")
        revoke_name = kwargs.get("revoke_policy_name")
        if receipt is not None and revoke_name:
            cutoff = "2026-08-18T12:00:30Z"
            document = setup._revoke_older_sessions_policy(cutoff=cutoff)
            receipt.update({"cutoff": cutoff, "document": document})
            calls.append(("policy", {
                "role": name,
                "name": revoke_name,
                "document": document,
            }))
        return f"arn:aws:iam::{ACCOUNT}:role/{name}"

    monkeypatch.setattr(
        setup,
        "_ensure_role",
        ensure_role,
    )
    monkeypatch.setattr(
        setup, "_put_policy", lambda _iam, **kwargs: calls.append(("policy", kwargs))
    )
    monkeypatch.setattr(setup, "_assert_role_configuration", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        setup,
        "_neutralize_controller_role",
        lambda *args, **kwargs: calls.append(("neutralize", kwargs)),
    )
    monkeypatch.setattr(
        setup,
        "_ensure_managed_policy",
        lambda _iam, *, account_id, name, **kwargs: setup._managed_policy_arn(
            account_id=account_id, name=name
        ),
    )
    monkeypatch.setattr(setup, "_assert_managed_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        setup,
        "_simulate_controller_policy",
        lambda *args, **kwargs: calls.append(("simulate", kwargs)),
    )
    monkeypatch.setattr(
        setup,
        "_wait_until_after_session_cutoffs",
        lambda cutoffs: calls.append(("wait-cutoff", cutoffs)),
    )
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
    simulator_index = next(
        index for index, item in enumerate(calls) if item[0] == "simulate"
    )
    wait_index = next(
        index for index, item in enumerate(calls) if item[0] == "wait-cutoff"
    )
    runner_active = next(
        index
        for index, (name, payload) in enumerate(calls)
        if name == "trust" and "ec2.amazonaws.com" in payload["PolicyDocument"]
    )
    controller_active = next(
        index
        for index, (name, payload) in enumerate(calls)
        if name == "trust"
        and "AssumeRoleWithWebIdentity" in payload["PolicyDocument"]
    )
    assert simulator_index < wait_index < runner_active < controller_active
    runner_policy_writes = [
        payload["name"]
        for name, payload in calls
        if name == "policy" and payload["role"] == setup.RUNNER_ROLE
    ]
    assert runner_policy_writes[:2] == [
        setup.RUNNER_REVOKE_POLICY,
        "LeadpoetProductionParityRunner",
    ]


def test_simulator_permission_denial_leaves_both_roles_inert_and_controller_detached(
    monkeypatch,
):
    events: list[tuple[str, object]] = []

    class IAM:
        def get_role(self, *, RoleName):
            assert RoleName == setup.VALIDATOR_ROLE
            return {"Role": {"Arn": (
                f"arn:aws:iam::{ACCOUNT}:role/{setup.VALIDATOR_ROLE}"
            )}}

        def attach_role_policy(self, **kwargs):
            events.append(("attach", kwargs))

        def update_assume_role_policy(self, **kwargs):
            events.append(("trust", kwargs))

    iam = IAM()
    monkeypatch.setattr(setup, "_iam_clients", lambda region: (object(), iam, ACCOUNT))
    monkeypatch.setattr(
        setup,
        "_ensure_oidc_provider",
        lambda *_: (
            f"arn:aws:iam::{ACCOUNT}:oidc-provider/"
            "token.actions.githubusercontent.com"
        ),
    )

    def ensure_role(_iam, *, name, **kwargs):
        receipt = kwargs.get("revocation_receipt")
        if receipt is not None and kwargs.get("revoke_policy_name"):
            cutoff = "2026-08-18T12:00:30Z"
            receipt.update({
                "cutoff": cutoff,
                "document": setup._revoke_older_sessions_policy(cutoff=cutoff),
            })
        return f"arn:aws:iam::{ACCOUNT}:role/{name}"

    monkeypatch.setattr(
        setup,
        "_ensure_role",
        ensure_role,
    )
    monkeypatch.setattr(setup, "_put_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup, "_assert_role_configuration", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup, "_ensure_instance_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        setup,
        "_neutralize_controller_role",
        lambda *args, **kwargs: events.append(("neutralize", kwargs)),
    )
    monkeypatch.setattr(
        setup,
        "_ensure_managed_policy",
        lambda _iam, *, account_id, name, **kwargs: setup._managed_policy_arn(
            account_id=account_id, name=name
        ),
    )
    monkeypatch.setattr(setup, "_assert_managed_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        setup,
        "_simulate_controller_policy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "denied"}},
                "SimulatePrincipalPolicy",
            )
        ),
    )
    monkeypatch.setattr(
        setup,
        "_wait_until_after_session_cutoffs",
        lambda cutoffs: events.append(("wait-cutoff", cutoffs)),
    )
    monkeypatch.setattr(setup, "_delete_static_bootstrap_role", lambda *args, **kwargs: None)
    args = argparse.Namespace(
        repository=setup.DEFAULT_REPOSITORY,
        region=setup.EXPECTED_REGION,
        production_gateway_ip=setup.PRODUCTION_GATEWAY_IP,
        production_gateway_secret_id=setup.PRODUCTION_GATEWAY_SECRET_ID,
        readonly_dsn_secret_id=setup.READONLY_DSN_SECRET_ID,
        miner_intake_secret_id=setup.DEFAULT_MINER_INTAKE_SECRET_ID,
        volume_gib=512,
    )
    with pytest.raises(ClientError) as denied:
        setup.setup_iam_only(args)
    assert denied.value.response["Error"]["Code"] == "AccessDenied"
    assert [name for name, _ in events].count("neutralize") == 2
    trust_documents = [
        payload["PolicyDocument"]
        for name, payload in events
        if name == "trust"
    ]
    assert trust_documents == [setup._json(setup._inert_trust())] * 2
    assert [name for name, _ in events].count("wait-cutoff") == 1


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


def test_inert_trust_has_valid_explicit_deny_and_no_allow():
    trust = setup._inert_trust()
    assert trust == {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Deny",
            "Principal": {
                "AWS": f"arn:aws:iam::{setup.EXPECTED_ACCOUNT_ID}:root",
            },
            "Action": "sts:AssumeRole",
        }],
    }
    assert all(
        statement.get("Effect") != "Allow"
        for statement in trust["Statement"]
    )
    assert all(
        statement.get("Principal") != "*"
        for statement in trust["Statement"]
    )


def test_controller_trust_is_exactly_bound_to_unique_main_workflows():
    oidc = (
        f"arn:aws:iam::{ACCOUNT}:oidc-provider/"
        "token.actions.githubusercontent.com"
    )
    trust = setup._controller_trust(oidc_arn=oidc)
    statement = trust["Statement"]
    assert len(statement) == 1
    assert set(statement[0]["Condition"]) == {"StringEquals"}
    claims = statement[0]["Condition"]["StringEquals"]
    prefix = "token.actions.githubusercontent.com:"
    assert claims == {
        prefix + "aud": "sts.amazonaws.com",
        prefix + "sub": "repo:leadpoet/leadpoet:ref:refs/heads/main",
        prefix + "repository": "leadpoet/leadpoet",
        prefix + "repository_id": "1075412927",
        prefix + "ref": "refs/heads/main",
        prefix + "workflow": [
            "Production Parity Full",
            "Production Parity Fast",
            "Production Parity Cleanup",
        ],
    }
    root = Path(__file__).resolve().parents[1]
    paths = {
        "Production Parity Full": root / ".github/workflows/physical-v2-staging.yml",
        "Production Parity Fast": root / ".github/workflows/production-parity-fast.yml",
        "Production Parity Cleanup": root / ".github/workflows/production-parity-cleanup.yml",
    }
    all_names: dict[str, list[Path]] = {}
    for path in (root / ".github/workflows").glob("*.yml"):
        first = path.read_text(encoding="utf-8").splitlines()[0]
        if first.startswith("name: "):
            all_names.setdefault(first.removeprefix("name: "), []).append(path)
    assert set(paths) == set(setup.CONTROLLER_WORKFLOWS)
    assert all(all_names.get(name) == [path] for name, path in paths.items())
    assert "branches: [main]" in paths["Production Parity Fast"].read_text()
    assert "head_branch == 'main'" in paths["Production Parity Full"].read_text()
    cleanup = paths["Production Parity Cleanup"].read_text()
    assert "ref: main" in cleanup
    assert "refs/heads/main:refs/remotes/origin/main" in cleanup


def test_session_cutoff_covers_inert_convergence_race_and_waits_to_activate(
    monkeypatch,
):
    class Clock(datetime):
        current = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            assert tz == timezone.utc
            return cls.current

    def sleep(seconds):
        Clock.current += timedelta(seconds=seconds)

    monkeypatch.setattr(setup, "datetime", Clock)
    monkeypatch.setattr(setup.time, "sleep", sleep)
    setup_started = Clock.current
    minted_before_inert_readback = setup_started + timedelta(seconds=3)
    Clock.current += timedelta(seconds=5)
    inert_readback = Clock.current
    cutoff = setup._new_session_cutoff()
    cutoff_time = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    assert (
        setup_started
        < minted_before_inert_readback
        < inert_readback
        < cutoff_time
    )
    revoke = setup._revoke_older_sessions_policy(cutoff=cutoff)
    assert revoke["Statement"][0]["Condition"] == {
        "DateLessThan": {"aws:TokenIssueTime": cutoff}
    }
    setup._wait_until_after_session_cutoffs([cutoff])
    assert Clock.current > cutoff_time
    Clock.current = datetime(2026, 8, 18, 13, 0, tzinfo=timezone.utc)
    future = setup._new_session_cutoff()
    monkeypatch.setattr(setup.time, "sleep", lambda seconds: None)
    with pytest.raises(setup.SetupError, match="has not elapsed"):
        setup._wait_until_after_session_cutoffs([future])


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
        inline: dict[str, dict] = {}
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
            self.events.append("inventory")
            return {"PolicyNames": sorted(self.inline)}

        def list_attached_role_policies(self, **kwargs):
            return {"AttachedPolicies": []}

        def update_assume_role_policy(self, **kwargs):
            self.events.append("set-inert")
            self.trust = json.loads(kwargs["PolicyDocument"])

        def update_role(self, **kwargs):
            self.events.append("set-duration")
            self.duration = kwargs["MaxSessionDuration"]

        def put_role_policy(self, **kwargs):
            self.events.append("put-revoke")
            self.inline[kwargs["PolicyName"]] = json.loads(
                kwargs["PolicyDocument"]
            )

        def get_role_policy(self, **kwargs):
            self.events.append("read-revoke")
            return {"PolicyDocument": self.inline[kwargs["PolicyName"]]}

    iam = IAM()
    inert = setup._inert_trust()
    revoke_name = (
        setup.CONTROLLER_REVOKE_POLICY
        if role_name == setup.CONTROLLER_ROLE
        else setup.RUNNER_REVOKE_POLICY
    )
    receipt: dict[str, object] = {}
    arn = setup._ensure_role(
        iam,
        account_id=ACCOUNT,
        name=role_name,
        trust=inert,
        expected_inline_policies={"ExpectedPolicy", revoke_name},
        expected_attached_policies=expected_attached,
        max_session_duration=43200,
        revoke_policy_name=revoke_name,
        revocation_receipt=receipt,
    )
    assert arn == f"arn:aws:iam::{ACCOUNT}:role/{role_name}"
    assert iam.events == [
        "get-role",
        "inventory",
        "set-inert",
        "set-duration",
        "get-role",
        "put-revoke",
        "read-revoke",
        "inventory",
    ]
    assert iam.trust == inert
    assert iam.duration == 43200
    assert iam.inline == {revoke_name: receipt["document"]}


def test_owned_role_with_unexpected_policy_is_rejected_without_mutation():
    class IAM:
        trust = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "sts:AssumeRole",
            }],
        }
        duration = 3600
        inline = {"UnexpectedPolicy": {"Version": "2012-10-17", "Statement": []}}
        events: list[str] = []

        def get_role(self, **kwargs):
            return {"Role": {
                "Arn": (
                    f"arn:aws:iam::{ACCOUNT}:role/{setup.CONTROLLER_ROLE}"
                ),
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
            self.events.append("inventory")
            return {"PolicyNames": sorted(self.inline)}

        def list_attached_role_policies(self, **kwargs):
            return {"AttachedPolicies": []}

        def update_assume_role_policy(self, **kwargs):
            self.events.append("inert")
            self.trust = json.loads(kwargs["PolicyDocument"])

        def update_role(self, **kwargs):
            self.duration = kwargs["MaxSessionDuration"]

        def put_role_policy(self, **kwargs):
            self.events.append("revoke")
            self.inline[kwargs["PolicyName"]] = json.loads(
                kwargs["PolicyDocument"]
            )

        def get_role_policy(self, **kwargs):
            return {"PolicyDocument": self.inline[kwargs["PolicyName"]]}

    iam = IAM()
    receipt: dict[str, object] = {}
    with pytest.raises(setup.SetupError, match="policy inventory differs"):
        setup._ensure_role(
            iam,
            account_id=ACCOUNT,
            name=setup.CONTROLLER_ROLE,
            trust=setup._inert_trust(),
            expected_inline_policies={
                "ExpectedPolicy",
                setup.CONTROLLER_REVOKE_POLICY,
            },
            expected_attached_policies=set(),
            revoke_policy_name=setup.CONTROLLER_REVOKE_POLICY,
            revocation_receipt=receipt,
        )
    assert iam.trust != setup._inert_trust()
    assert iam.events == ["inventory"]
    assert receipt == {}
    assert setup.CONTROLLER_REVOKE_POLICY not in iam.inline


def test_controller_neutralization_detaches_allows_and_retains_only_cutoff_deny():
    legacy = "LeadpoetProductionParityController"
    managed = {
        setup._managed_policy_arn(account_id=ACCOUNT, name=name)
        for name in setup.CONTROLLER_POLICY_NAMES.values()
    }
    revoke = setup._revoke_older_sessions_policy(
        cutoff="2026-08-18T12:00:30Z"
    )

    class IAM:
        def __init__(self):
            self.trust = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": "sts:AssumeRole",
                }],
            }
            self.duration = 3600
            self.inline = {
                legacy: {"Version": "2012-10-17", "Statement": []}
            }
            self.attached = set(managed)
            self.events: list[str] = []

        def update_assume_role_policy(self, **kwargs):
            self.events.append("inert")
            self.trust = json.loads(kwargs["PolicyDocument"])

        def update_role(self, **kwargs):
            self.events.append("duration")
            self.duration = kwargs["MaxSessionDuration"]

        def list_role_policies(self, **kwargs):
            return {"PolicyNames": sorted(self.inline)}

        def list_attached_role_policies(self, **kwargs):
            return {"AttachedPolicies": [
                {"PolicyArn": value} for value in sorted(self.attached)
            ]}

        def delete_role_policy(self, **kwargs):
            self.events.append("delete-legacy")
            del self.inline[kwargs["PolicyName"]]

        def detach_role_policy(self, **kwargs):
            self.events.append("detach")
            self.attached.remove(kwargs["PolicyArn"])

        def put_role_policy(self, **kwargs):
            self.events.append("put-revoke")
            self.inline[kwargs["PolicyName"]] = json.loads(
                kwargs["PolicyDocument"]
            )

        def get_role(self, **kwargs):
            return {"Role": {
                "Arn": f"arn:aws:iam::{ACCOUNT}:role/{setup.CONTROLLER_ROLE}",
                "Path": "/",
                "AssumeRolePolicyDocument": self.trust,
                "MaxSessionDuration": self.duration,
            }}

        def list_role_tags(self, **kwargs):
            return {"Tags": [{
                "Key": "leadpoet:purpose",
                "Value": "production-parity",
            }]}

        def get_role_policy(self, **kwargs):
            return {"PolicyDocument": self.inline[kwargs["PolicyName"]]}

    iam = IAM()
    setup._neutralize_controller_role(
        iam,
        account_id=ACCOUNT,
        managed_policy_arns=managed,
        legacy_inline_policy=legacy,
        revoke_document=revoke,
    )
    assert iam.trust == setup._inert_trust()
    assert iam.attached == set()
    assert iam.inline == {setup.CONTROLLER_REVOKE_POLICY: revoke}
    assert iam.events[:4] == [
        "inert",
        "duration",
        "put-revoke",
        "delete-legacy",
    ]
    assert iam.events.count("detach") == len(managed)

    collision = IAM()
    collision.inline["UnexpectedAllow"] = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}],
    }
    with pytest.raises(setup.SetupError, match="inventory changed"):
        setup._neutralize_controller_role(
            collision,
            account_id=ACCOUNT,
            managed_policy_arns=managed,
            legacy_inline_policy=legacy,
            revoke_document=revoke,
        )
    assert collision.inline[setup.CONTROLLER_REVOKE_POLICY] == revoke
    assert collision.events[:3] == ["inert", "duration", "put-revoke"]


class _ManagedPolicyIam:
    def __init__(self, name: str, *, roles: set[str] | None = None):
        self.name = name
        self.arn = setup._managed_policy_arn(account_id=ACCOUNT, name=name)
        self.default = "v1"
        self.documents = {
            f"v{index}": {"Version": "2012-10-17", "Statement": []}
            for index in range(1, 6)
        }
        self.roles = set(roles or set())
        self.events: list[tuple[str, str]] = []

    def get_policy(self, **kwargs):
        assert kwargs == {"PolicyArn": self.arn}
        return {"Policy": {
            "Arn": self.arn,
            "PolicyName": self.name,
            "Path": setup.CONTROLLER_POLICY_PATH,
            "Description": setup.CONTROLLER_POLICY_DESCRIPTION,
            "DefaultVersionId": self.default,
        }}

    def list_policy_tags(self, **kwargs):
        assert kwargs == {"PolicyArn": self.arn}
        return {"Tags": [{
            "Key": "leadpoet:purpose",
            "Value": "production-parity",
        }]}

    def list_entities_for_policy(self, **kwargs):
        assert kwargs == {"PolicyArn": self.arn}
        return {
            "PolicyRoles": [{"RoleName": value} for value in sorted(self.roles)],
            "PolicyUsers": [],
            "PolicyGroups": [],
            "IsTruncated": False,
        }

    def get_policy_version(self, **kwargs):
        return {"PolicyVersion": {"Document": self.documents[kwargs["VersionId"]]}}

    def list_policy_versions(self, **kwargs):
        assert kwargs == {"PolicyArn": self.arn}
        return {"Versions": [
            {
                "VersionId": version,
                "IsDefaultVersion": version == self.default,
            }
            for version in self.documents
        ]}

    def delete_policy_version(self, **kwargs):
        version = kwargs["VersionId"]
        assert version != self.default
        self.events.append(("delete", version))
        del self.documents[version]

    def create_policy_version(self, **kwargs):
        assert kwargs["SetAsDefault"] is True
        version = "v6"
        self.documents[version] = json.loads(kwargs["PolicyDocument"])
        self.default = version
        self.events.append(("create", version))
        return {"PolicyVersion": {"VersionId": version}}


def test_managed_policy_replacement_cleans_five_versions_and_is_exact():
    name = setup.CONTROLLER_POLICY_NAMES["lifecycle"]
    iam = _ManagedPolicyIam(name)
    document = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": "ec2:DescribeInstances",
            "Resource": "*",
        }],
    }
    arn = setup._ensure_managed_policy(
        iam,
        account_id=ACCOUNT,
        name=name,
        document=document,
    )
    assert arn == iam.arn
    assert iam.default == "v6"
    assert iam.documents == {"v6": document}
    assert iam.events == [
        ("delete", "v2"),
        ("delete", "v3"),
        ("delete", "v4"),
        ("delete", "v5"),
        ("create", "v6"),
        ("delete", "v1"),
    ]


def test_managed_policy_attached_to_other_entity_fails_before_version_write():
    name = setup.CONTROLLER_POLICY_NAMES["cloudfront"]
    iam = _ManagedPolicyIam(name, roles={"other-role"})
    with pytest.raises(setup.SetupError, match="not owned by parity"):
        setup._ensure_managed_policy(
            iam,
            account_id=ACCOUNT,
            name=name,
            document=iam.documents["v1"],
        )
    assert iam.events == []
    assert len(iam.documents) == 5


def test_managed_policy_first_create_and_exact_retry_are_idempotent():
    name = setup.CONTROLLER_POLICY_NAMES["data"]
    document = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": "s3:ListAllMyBuckets",
            "Resource": "*",
        }],
    }

    class IAM(_ManagedPolicyIam):
        exists = False

        def __init__(self):
            super().__init__(name)
            self.documents = {}

        def get_policy(self, **kwargs):
            if not self.exists:
                raise ClientError(
                    {"Error": {"Code": "NoSuchEntity", "Message": "absent"}},
                    "GetPolicy",
                )
            return super().get_policy(**kwargs)

        def create_policy(self, **kwargs):
            assert kwargs == {
                "PolicyName": name,
                "Path": setup.CONTROLLER_POLICY_PATH,
                "PolicyDocument": setup._json(document),
                "Description": setup.CONTROLLER_POLICY_DESCRIPTION,
                "Tags": [{
                    "Key": "leadpoet:purpose",
                    "Value": "production-parity",
                }],
            }
            self.exists = True
            self.default = "v1"
            self.documents = {"v1": document}
            self.events.append(("create-policy", "v1"))

    iam = IAM()
    expected = setup._managed_policy_arn(account_id=ACCOUNT, name=name)
    assert setup._ensure_managed_policy(
        iam, account_id=ACCOUNT, name=name, document=document
    ) == expected
    assert setup._ensure_managed_policy(
        iam, account_id=ACCOUNT, name=name, document=document
    ) == expected
    assert iam.documents == {"v1": document}
    assert iam.events == [("create-policy", "v1")]


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
    assert source.count("AND rolsuper") == 2
    assert "unexpectedly superuser" in source
    assert "NOSUPERUSER" not in source
    assert "EXCEPTION WHEN OTHERS" in source
    assert PASSWORD not in source


def test_full_workflow_uses_self_hosted_bounded_windows_and_exact_volume():
    workflow_path = (
        Path(__file__).parents[1]
        / ".github/workflows/physical-v2-staging.yml"
    )
    source = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(source)
    job = workflow["jobs"]["validate"]
    assert "leadpoet-gateway-v2-builder" in source
    assert "runs-on: ubuntu-latest" not in source
    assert job["timeout-minutes"] == 1430
    assert "PARITY_TEMP" not in job["env"]
    assert not any(
        "${{ runner." in str(value) for value in job["env"].values()
    )
    assert 'FULL_TIMEOUT_SECONDS: "72000"' in source
    assert 'SSM_TIMEOUT_SECONDS: "77400"' in source
    assert '${PARITY_VOLUME_GIB:-' not in source
    assert 'test "$PARITY_VOLUME_GIB" = "512"' in source
    assert source.count("unset-current-credentials: true") >= 7
    assert "allowed-account-ids:" not in source
    assert "--max-wait-seconds 16200" in source
    assert "--max-wait-seconds 12600" in source
    assert "Refresh parity AWS role for cleanup" in source
    assert "id: provenance" in source
    assert "echo 'verified=true' >> \"$GITHUB_OUTPUT\"" in source
    assert source.count(
        "always() && steps.provenance.outputs.verified == 'true'"
    ) == 3
    assert "steps.cleanup_account.outcome == 'success'" in source
    assert "id: evidence_path" in source
    assert (
        "always() && steps.evidence_path.outputs.verified == 'true'" in source
    )


def test_full_workflow_retains_failed_redacted_evidence_after_poll_failure():
    workflow_path = (
        Path(__file__).parents[1]
        / ".github/workflows/physical-v2-staging.yml"
    )
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["validate"]["steps"]
    by_name = {step.get("name"): step for step in steps}
    names = [step.get("name") for step in steps]
    verify = by_name["Verify redacted full evidence"]

    assert verify["if"] == (
        "always() && steps.stack.outcome == 'success' && "
        "steps.execute.outcome == 'success'"
    )
    assert names.index("Poll candidate window 5") < names.index(
        "Verify redacted full evidence"
    )
    assert names.index("Verify redacted full evidence") < names.index(
        "Validate redacted evidence upload path"
    ) < names.index("Upload redacted evidence")
    verify_script = verify["run"]
    assert verify_script.index("aws s3 cp") < verify_script.index(
        "full production-parity evidence is incomplete"
    )
    upload = by_name["Upload redacted evidence"]
    assert "always()" in upload["if"]
    assert upload["with"]["path"].endswith("/full-evidence.json")


def test_controller_dependencies_use_a_scrubbed_per_run_virtualenv():
    root = Path(__file__).parents[1]
    action = (
        root / ".github/actions/setup-production-parity-controller/action.yml"
    ).read_text(encoding="utf-8")
    full = (
        root / ".github/workflows/physical-v2-staging.yml"
    ).read_text(encoding="utf-8")
    fast = (
        root / ".github/workflows/production-parity-fast.yml"
    ).read_text(encoding="utf-8")
    cleanup = (
        root / ".github/workflows/production-parity-cleanup.yml"
    ).read_text(encoding="utf-8")

    assert "actions/setup-python" not in action
    assert "python-executable:" in action
    assert "required: true" in action
    assert 'controller_root="$PARITY_TEMP/controller"' in action
    assert (
        'controller_root="$RUNNER_TEMP/production-parity-controller-'
        '$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"' in action
    )
    assert '"$host_python" -I -m venv "$venv_root"' in action
    assert 'test "$observed_host_python" = "3.11|$resolved_host_python"' in action
    assert 'test "$host_python" = "/usr/bin/python3.11"' in action
    assert 'test "$resolved_host_python" = "/usr/bin/python3.11"' in action
    assert 'stat -Lc \'%u:%g:%a\' "$host_python"' in action
    assert '= "0:0:755"' in action
    assert "[0-7][0145][0145]" in action
    assert 'sys.version_info.major}.{sys.version_info.minor}' in action
    assert '"$venv_python" -m pip install' in action
    assert "--no-cache-dir" in action
    assert "python3 -m pip install" not in action
    assert "cache: pip" not in action
    assert 'include-system-site-packages = false' in action
    assert 'printf \'%s\\n\' "$venv_root/bin" >> "$GITHUB_PATH"' in action
    assert 'printf \'VIRTUAL_ENV=%s\\n\' "$venv_root"' in action
    assert 'test "$(command -v python3)" = "$VIRTUAL_ENV/bin/python3"' in action
    assert '"$venv_python" "$script" --help' in action
    assert "uses: actions/setup-python@v5" not in full
    assert "python-executable: /usr/bin/python3.11" in full
    for github_hosted in (fast, cleanup):
        assert "uses: actions/setup-python@v5" in github_hosted
        assert "python-version: \"3.11\"" in github_hosted
        assert "python-executable: ${{ env.pythonLocation }}/bin/python3" in github_hosted
    assert 'printf \'PARITY_TEMP=%s\\n\' "$PARITY_TEMP" >> "$GITHUB_ENV"' in full
    assert 'rm -rf -- "$parity_temp"' in full


def test_full_controller_python_identity_rejects_owner_or_write_poison():
    def accepted(
        *,
        file_identity: tuple[int, int, int],
        parent_identities: list[tuple[int, int, int]],
    ) -> bool:
        if file_identity != (0, 0, 0o755):
            return False
        return all(
            uid == 0 and gid == 0 and mode & 0o022 == 0
            for uid, gid, mode in parent_identities
        )

    trusted_parents = [(0, 0, 0o555), (0, 0, 0o755), (0, 0, 0o555)]
    assert accepted(file_identity=(0, 0, 0o755), parent_identities=trusted_parents)
    assert not accepted(
        file_identity=(1000, 0, 0o755), parent_identities=trusted_parents
    )
    assert not accepted(
        file_identity=(0, 0, 0o775), parent_identities=trusted_parents
    )
    assert not accepted(
        file_identity=(0, 0, 0o755),
        parent_identities=[*trusted_parents[:-1], (0, 0, 0o777)],
    )


def test_full_workflow_derives_temp_before_use_and_cleans_safe_exact_path():
    path = (
        Path(__file__).parents[1]
        / ".github/workflows/physical-v2-staging.yml"
    )
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["validate"]["steps"]
    by_name = {step.get("name"): step for step in steps}

    prepare = by_name["Prepare isolated self-hosted controller workspace"]["run"]
    export = 'printf \'PARITY_TEMP=%s\\n\' "$PARITY_TEMP" >> "$GITHUB_ENV"'
    assert 'expected="$RUNNER_TEMP/production-parity-' in prepare
    assert 'PARITY_TEMP="$expected"' in prepare
    assert export in prepare
    absent = 'test ! -e "$expected"'
    create = 'mkdir -m 700 "$expected"'
    marker = 'printf \'%s\\n\' "$owner_token" > "$owner_marker"'
    mask = 'printf \'::add-mask::%s\\n\' "$owner_token"'
    owner_export = (
        'printf \'PARITY_OWNER_TOKEN=%s\\n\' "$owner_token" >> "$GITHUB_ENV"'
    )
    assert prepare.index(absent) < prepare.index(create)
    assert prepare.index(create) < prepare.index(marker) < prepare.index(export)
    assert prepare.index(mask) < prepare.index(owner_export) < prepare.index(create)
    assert ".leadpoet-production-parity-owner" in prepare

    for name in (
        "Validate redacted evidence upload path",
        "Destroy every run-scoped resource",
        "Scrub self-hosted controller workspace",
    ):
        cleanup = by_name[name]["run"]
        assert "${RUNNER_TEMP:-}" in cleanup
        assert "${GITHUB_RUN_ID:-}" in cleanup
        assert "${GITHUB_RUN_ATTEMPT:-}" in cleanup
        assert (
            'expected="$RUNNER_TEMP/production-parity-' in cleanup
            or 'expected="$runner_temp/production-parity-' in cleanup
        )
        assert (
            'parity_temp="${PARITY_TEMP:-$expected}"' in cleanup
            or 'parity_temp="${explicit_temp:-$expected}"' in cleanup
        )
        assert 'owner_token="${PARITY_OWNER_TOKEN:-}"' in cleanup
        assert ".leadpoet-production-parity-owner" in cleanup
        assert '"$parity_temp"' in cleanup
        assert '"$expected"' in cleanup

    upload = by_name["Upload redacted evidence"]["with"]["path"]
    assert upload == (
        "${{ runner.temp }}/production-parity-${{ github.run_id }}-"
        "${{ github.run_attempt }}/full-evidence.json"
    )


def _dirty_disposable_git_checkout(root: Path) -> tuple[Path, Path, Path]:
    checkout = root / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "parity-test@example.invalid"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Parity Test"],
        cwd=checkout,
        check=True,
    )
    tracked = checkout / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "initial"],
        cwd=checkout,
        check=True,
    )
    tracked.write_text("dirty\n", encoding="utf-8")
    untracked = checkout / "untracked.txt"
    untracked.write_text("remove me\n", encoding="utf-8")
    return checkout, tracked, untracked


def test_full_workflow_never_scrubs_a_preexisting_temp_collision(tmp_path):
    path = (
        Path(__file__).parents[1]
        / ".github/workflows/physical-v2-staging.yml"
    )
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["validate"]["steps"]
    by_name = {step.get("name"): step for step in steps}
    prepare = by_name["Prepare isolated self-hosted controller workspace"]["run"]
    evidence = by_name["Validate redacted evidence upload path"]["run"]
    scrub = by_name["Scrub self-hosted controller workspace"]["run"]

    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    expected = runner_temp / "production-parity-12345-2"
    expected.mkdir()
    sentinel = expected / "must-survive"
    sentinel.write_text("preexisting\n", encoding="utf-8")
    (expected / "full-evidence.json").write_text(
        '{"status":"fabricated-collision"}\n', encoding="utf-8"
    )
    (expected / ".leadpoet-production-parity-owner").write_text(
        "different-owner\n", encoding="utf-8"
    )
    github_env = tmp_path / "github-env"
    github_env.touch()
    github_output = tmp_path / "github-output"
    github_output.touch()
    env = {
        **os.environ,
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_RUN_ID": "12345",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_ENV": str(github_env),
        "GITHUB_OUTPUT": str(github_output),
    }
    checkout, tracked, untracked = _dirty_disposable_git_checkout(tmp_path)

    prepared = subprocess.run(
        ["bash", "-c", prepare],
        cwd=checkout,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert prepared.returncode != 0
    exported = dict(
        line.split("=", 1)
        for line in github_env.read_text(encoding="utf-8").splitlines()
    )
    assert re.fullmatch(r"[0-9a-f]{64}", exported["PARITY_OWNER_TOKEN"])
    assert "PARITY_TEMP" not in exported

    validated = subprocess.run(
        ["bash", "-c", evidence],
        cwd=checkout,
        env={**env, **exported},
        check=False,
        capture_output=True,
        text=True,
    )
    assert validated.returncode != 0
    assert github_output.read_text(encoding="utf-8") == ""

    scrubbed = subprocess.run(
        ["bash", "-c", scrub],
        cwd=checkout,
        env={**env, **exported},
        check=False,
        capture_output=True,
        text=True,
    )
    assert scrubbed.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "preexisting\n"
    assert tracked.read_text(encoding="utf-8") == "committed\n"
    assert not untracked.exists()


def test_full_workflow_scrubs_only_its_owned_temp_after_later_failure(tmp_path):
    path = (
        Path(__file__).parents[1]
        / ".github/workflows/physical-v2-staging.yml"
    )
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["validate"]["steps"]
    by_name = {step.get("name"): step for step in steps}
    prepare = by_name["Prepare isolated self-hosted controller workspace"]["run"]
    destroy = by_name["Destroy every run-scoped resource"]["run"]
    scrub = by_name["Scrub self-hosted controller workspace"]["run"]

    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    github_env = tmp_path / "github-env"
    github_env.touch()
    env = {
        **os.environ,
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_RUN_ID": "67890",
        "GITHUB_RUN_ATTEMPT": "3",
        "GITHUB_ENV": str(github_env),
        "AWS_REGION": "us-east-1",
    }
    checkout, tracked, untracked = _dirty_disposable_git_checkout(tmp_path)
    subprocess.run(
        ["bash", "-c", prepare],
        cwd=checkout,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    exported = dict(
        line.split("=", 1)
        for line in github_env.read_text(encoding="utf-8").splitlines()
    )
    owned = Path(exported["PARITY_TEMP"])
    assert owned.is_dir()

    destroyed = subprocess.run(
        ["bash", "-c", destroy],
        cwd=checkout,
        env={**env, **exported},
        check=False,
        capture_output=True,
        text=True,
    )
    assert destroyed.returncode == 0
    assert not owned.exists()

    scrubbed = subprocess.run(
        ["bash", "-c", scrub],
        cwd=checkout,
        env={**env, **exported},
        check=False,
        capture_output=True,
        text=True,
    )
    assert scrubbed.returncode == 0
    assert not owned.exists()
    assert tracked.read_text(encoding="utf-8") == "committed\n"
    assert not untracked.exists()


def test_fast_and_cleanup_pin_account_and_reject_stale_cleanup_code():
    root = Path(__file__).parents[1]
    workflow_jobs = (
        (".github/workflows/production-parity-fast.yml", "validate"),
        (".github/workflows/physical-v2-staging.yml", "validate"),
        (".github/workflows/production-parity-cleanup.yml", "cleanup"),
    )
    sources = {}
    for relative_path, job_name in workflow_jobs:
        source = (root / relative_path).read_text(encoding="utf-8")
        sources[relative_path] = source
        assert "allowed-account-ids:" not in source
        workflow = yaml.safe_load(source)
        steps = workflow["jobs"][job_name]["steps"]
        credential_indexes = [
            index
            for index, step in enumerate(steps)
            if step.get("uses") == "aws-actions/configure-aws-credentials@v4"
        ]
        assert credential_indexes
        for index in credential_indexes:
            credential = steps[index]
            account_gate = steps[index + 1]
            assert account_gate["name"].startswith(
                "Require exact parity AWS account"
            )
            if credential.get("id") == "cleanup_credentials":
                assert credential["if"] in account_gate["if"]
                assert (
                    "steps.cleanup_credentials.outcome == 'success'"
                    in account_gate["if"]
                )
            else:
                assert account_gate.get("if") == credential.get("if")
            assert "aws sts get-caller-identity" in account_gate["run"]
            assert '"493765492819"' in account_gate["run"]
            assert credential["with"]["unset-current-credentials"] is True

    fast = sources[".github/workflows/production-parity-fast.yml"]
    cleanup = sources[".github/workflows/production-parity-cleanup.yml"]
    host_export = fast.index(
        'export LEADPOET_PARITY_PRODUCTION_DB_HOST="$production_host"'
    )
    snapshot_capture = fast.index(
        "python3 scripts/production_parity_snapshot.py capture"
    )
    host_unset = fast.index(
        "unset LEADPOET_PARITY_PRODUCTION_READONLY_DSN "
        "LEADPOET_PARITY_PRODUCTION_DB_HOST"
    )
    assert host_export < snapshot_capture < host_unset
    cleanup_gate = cleanup.index("name: Require exact current main before credentials")
    cleanup_action = cleanup.index(
        "uses: ./.github/actions/setup-production-parity-controller"
    )
    cleanup_credentials = cleanup.index(
        "uses: aws-actions/configure-aws-credentials@v4"
    )
    assert "git rev-parse origin/main" in cleanup
    assert cleanup_gate < cleanup_action < cleanup_credentials


def test_full_cleanup_rejects_stale_credentials_after_refresh_failure():
    path = (
        Path(__file__).parents[1]
        / ".github/workflows/physical-v2-staging.yml"
    )
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["validate"]["steps"]
    by_name = {step.get("name"): step for step in steps}
    refresh = by_name["Refresh parity AWS role for cleanup"]
    account = by_name["Require exact parity AWS account for cleanup"]
    destroy = by_name["Destroy every run-scoped resource"]

    assert refresh["id"] == "cleanup_credentials"
    assert "steps.cleanup_credentials.outcome == 'success'" in account["if"]
    assert "steps.cleanup_account.outcome == 'success'" in destroy["if"]


def test_configure_requires_exact_parity_variable_inventory():
    source = (
        Path(__file__).parents[1]
        / "scripts/setup_production_parity_staging.py"
    ).read_text(encoding="utf-8")
    assert "actual_parity_names != expected_parity_names" in source
    assert "GitHub parity variable inventory differs" in source
