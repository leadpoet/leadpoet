from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from botocore.exceptions import ClientError

from scripts import setup_production_parity_staging as parity_setup
from scripts.cleanup_production_parity_staging import (
    _owned_run,
    cleanup_stale,
)
from scripts.provision_production_parity_staging import (
    CLOUDFRONT_MANAGED_POLICY_MAX_PAGES,
    PARITY_VOLUME_GIB,
    PRODUCTION_ACCOUNT_ID,
    PRODUCTION_AMI_ID,
    PRODUCTION_GATEWAY_INSTANCE_ID,
    PRODUCTION_INSTANCE_TYPE,
    PRODUCTION_REGION,
    PRODUCTION_RUNNER_PROFILE,
    PRODUCTION_SUBNET_ID,
    PRODUCTION_VPC_ID,
    SCHEMA_VERSION,
    ProvisioningError,
    _artifact_bucket_name,
    _distribution_arn,
    _managed_policy_id,
    _resource_arn,
    _single_production_instance,
    create_stack,
    delete_stack,
)
from leadpoet_canonical.attested_v2 import sha256_json
from tests.restart_rehearsal.sanitized_weight_fixture import (
    SanitizedWeightFixture,
)


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
RUN_ID = "pp-123456-1"
INSTANCE_ID = "i-0123456789abcdef0"
SECURITY_GROUP_ID = "sg-0123456789abcdef0"
DISTRIBUTION_ID = "E123456789ABCD"


def _reference_instance() -> dict:
    return {
        "ImageId": PRODUCTION_AMI_ID,
        "InstanceId": PRODUCTION_GATEWAY_INSTANCE_ID,
        "InstanceType": PRODUCTION_INSTANCE_TYPE,
        "SubnetId": PRODUCTION_SUBNET_ID,
        "VpcId": PRODUCTION_VPC_ID,
        "RootDeviceName": "/dev/xvda",
        "EnclaveOptions": {"Enabled": True},
        "MetadataOptions": {"HttpTokens": "required"},
    }


def test_streamed_iam_and_provisioner_share_exact_launch_identity():
    assert (
        PRODUCTION_AMI_ID,
        PRODUCTION_INSTANCE_TYPE,
        PRODUCTION_SUBNET_ID,
        PRODUCTION_VPC_ID,
        PRODUCTION_RUNNER_PROFILE,
    ) == (
        parity_setup.PRODUCTION_AMI_ID,
        parity_setup.PRODUCTION_INSTANCE_TYPE,
        parity_setup.PRODUCTION_SUBNET_ID,
        parity_setup.PRODUCTION_VPC_ID,
        parity_setup.RUNNER_PROFILE,
    )


class _Waiter:
    def wait(self, **_kwargs):
        return None


class _EC2:
    def __init__(self):
        self.reference = _reference_instance()
        self.ingress = None
        self.launch = None
        self.terminated = []
        self.deleted_groups = []

    def describe_instances(self, **kwargs):
        if "Filters" in kwargs:
            return {"Reservations": [{"Instances": [self.reference]}]}
        return {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": INSTANCE_ID,
                            "PublicDnsName": "ec2-parity.example.invalid",
                        }
                    ]
                }
            ]
        }

    def describe_managed_prefix_lists(self, **_kwargs):
        return {
            "PrefixLists": [
                {
                    "PrefixListId": "pl-cloudfront",
                    "State": "create-complete",
                }
            ]
        }

    def create_security_group(self, **_kwargs):
        return {"GroupId": SECURITY_GROUP_ID}

    def authorize_security_group_ingress(self, **kwargs):
        self.ingress = kwargs

    def run_instances(self, **kwargs):
        self.launch = kwargs
        return {"Instances": [{"InstanceId": INSTANCE_ID}]}

    def get_waiter(self, _name):
        return _Waiter()

    def terminate_instances(self, **kwargs):
        self.terminated.extend(kwargs["InstanceIds"])

    def delete_security_group(self, **kwargs):
        self.deleted_groups.append(kwargs["GroupId"])


class _CloudFront:
    def __init__(self):
        self.config = None
        self.tags = None

    def list_cache_policies(self, **kwargs):
        assert kwargs == {"Type": "managed"}
        return {
            "CachePolicyList": {
                "Items": [
                    {
                        "CachePolicy": {
                            "Id": "cache-disabled",
                            "CachePolicyConfig": {
                                "Name": "Managed-CachingDisabled"
                            },
                        }
                    }
                ],
            }
        }

    def list_origin_request_policies(self, **kwargs):
        assert kwargs == {"Type": "managed"}
        return {
            "OriginRequestPolicyList": {
                "Items": [
                    {
                        "OriginRequestPolicy": {
                            "Id": "all-viewer",
                            "OriginRequestPolicyConfig": {
                                "Name": "Managed-AllViewerExceptHostHeader"
                            },
                        }
                    }
                ],
            }
        }

    def create_distribution_with_tags(self, **kwargs):
        value = kwargs["DistributionConfigWithTags"]
        self.config = value["DistributionConfig"]
        self.tags = value["Tags"]
        return {
            "Distribution": {
                "Id": DISTRIBUTION_ID,
                "ARN": _distribution_arn(DISTRIBUTION_ID),
            }
        }


def _managed_policy_item(policy_type: str, *, policy_id: str, name: str) -> dict:
    if policy_type == "cache":
        return {
            "CachePolicy": {
                "Id": policy_id,
                "CachePolicyConfig": {"Name": name},
            }
        }
    return {
        "OriginRequestPolicy": {
            "Id": policy_id,
            "OriginRequestPolicyConfig": {"Name": name},
        }
    }


class _ManagedPolicyPages:
    def __init__(self, policy_type: str, pages: list[dict]):
        self.policy_type = policy_type
        self.pages = list(pages)
        self.calls = []

    def _next(self, policy_type: str, kwargs: dict) -> dict:
        assert policy_type == self.policy_type
        self.calls.append((policy_type, kwargs))
        assert self.pages
        return self.pages.pop(0)

    def list_cache_policies(self, **kwargs):
        return self._next("cache", kwargs)

    def list_origin_request_policies(self, **kwargs):
        return self._next("origin_request", kwargs)


@pytest.mark.parametrize(
    ("policy_type", "list_key", "target_name"),
    (
        ("cache", "CachePolicyList", "Managed-CachingDisabled"),
        (
            "origin_request",
            "OriginRequestPolicyList",
            "Managed-AllViewerExceptHostHeader",
        ),
    ),
)
def test_managed_policy_lookup_uses_manual_marker_pagination(
    policy_type, list_key, target_name
):
    client = _ManagedPolicyPages(
        policy_type,
        [
            {
                list_key: {
                    "Items": [
                        _managed_policy_item(
                            policy_type,
                            policy_id="other-policy",
                            name="Managed-OtherPolicy",
                        )
                    ],
                    "NextMarker": "second-page",
                }
            },
            {
                list_key: {
                    "Items": [
                        _managed_policy_item(
                            policy_type,
                            policy_id="target-policy",
                            name=target_name,
                        )
                    ],
                }
            },
        ],
    )

    assert (
        _managed_policy_id(
            client,
            policy_type=policy_type,
            name=target_name,
        )
        == "target-policy"
    )
    assert client.calls == [
        (policy_type, {"Type": "managed"}),
        (policy_type, {"Type": "managed", "Marker": "second-page"}),
    ]


@pytest.mark.parametrize(
    "listing",
    (
        {"Items": [], "NextMarker": None},
        {"Items": [], "NextMarker": ""},
        {"Items": [], "NextMarker": "   "},
        {"Items": [], "NextMarker": 2},
        {"Items": {}, "NextMarker": "next"},
    ),
)
def test_managed_policy_lookup_rejects_malformed_pages(listing):
    client = _ManagedPolicyPages("cache", [{"CachePolicyList": listing}])

    with pytest.raises(ProvisioningError, match="pagination|page"):
        _managed_policy_id(
            client,
            policy_type="cache",
            name="Managed-CachingDisabled",
        )


def test_managed_policy_lookup_rejects_repeated_marker():
    client = _ManagedPolicyPages(
        "cache",
        [
            {
                "CachePolicyList": {
                    "Items": [],
                    "NextMarker": "repeated",
                }
            },
            {
                "CachePolicyList": {
                    "Items": [],
                    "NextMarker": "repeated",
                }
            },
        ],
    )

    with pytest.raises(ProvisioningError, match="pagination"):
        _managed_policy_id(
            client,
            policy_type="cache",
            name="Managed-CachingDisabled",
        )
    assert len(client.calls) == 2


def test_managed_policy_lookup_rejects_excess_pages():
    class _EndlessManagedPolicies:
        def __init__(self):
            self.calls = []

        def list_cache_policies(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "CachePolicyList": {
                    "Items": [],
                    "NextMarker": f"page-{len(self.calls)}",
                }
            }

    client = _EndlessManagedPolicies()
    with pytest.raises(ProvisioningError, match="limit exceeded"):
        _managed_policy_id(
            client,
            policy_type="cache",
            name="Managed-CachingDisabled",
        )
    assert len(client.calls) == CLOUDFRONT_MANAGED_POLICY_MAX_PAGES


class _SSM:
    def describe_instance_information(self, **_kwargs):
        return {
            "InstanceInformationList": [
                {"InstanceId": INSTANCE_ID, "PingStatus": "Online"}
            ]
        }


class _S3:
    def __init__(self):
        self.created = None
        self.lock = None
        self.tags = None
        self.deleted = []

    def create_bucket(self, **kwargs):
        self.created = kwargs

    def put_public_access_block(self, **_kwargs):
        return None

    def put_bucket_encryption(self, **_kwargs):
        return None

    def put_bucket_versioning(self, **_kwargs):
        return None

    def put_object_lock_configuration(self, **kwargs):
        self.lock = kwargs

    def put_bucket_tagging(self, **kwargs):
        self.tags = kwargs

    def delete_bucket(self, **kwargs):
        self.deleted.append(kwargs["Bucket"])


def _owned_tags() -> list[dict[str, str]]:
    return [
        {"Key": "leadpoet:parity-run", "Value": RUN_ID},
        {"Key": "leadpoet:candidate-sha", "Value": SHA},
        {"Key": "leadpoet:ephemeral", "Value": "true"},
        {"Key": "Name", "Value": f"leadpoet-parity-{RUN_ID}"},
    ]


def _delete_state(**overrides) -> dict:
    value = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "account_id": PRODUCTION_ACCOUNT_ID,
        "region": PRODUCTION_REGION,
        "run_id": RUN_ID,
        "candidate_sha": SHA,
        "instance_id": INSTANCE_ID,
        "instance_arn": _resource_arn(
            kind="instance", resource_id=INSTANCE_ID
        ),
        "security_group_id": SECURITY_GROUP_ID,
        "security_group_arn": _resource_arn(
            kind="security-group", resource_id=SECURITY_GROUP_ID
        ),
        "cloudfront_distribution_id": DISTRIBUTION_ID,
        "cloudfront_distribution_arn": _distribution_arn(DISTRIBUTION_ID),
        "artifact_bucket": _artifact_bucket_name(
            account_id=PRODUCTION_ACCOUNT_ID,
            run_id=RUN_ID,
            candidate_sha=SHA,
        ),
        "volume_gib": PARITY_VOLUME_GIB,
    }
    value.update(overrides)
    return value


def _aws_error(code: str, operation: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": "redacted test failure"}},
        operation,
    )


class _DeleteCloudFront:
    def __init__(
        self,
        *,
        absent: bool = False,
        tags: list[dict[str, str]] | None = None,
        distribution_id: str = DISTRIBUTION_ID,
        distribution_arn: str | None = None,
        fail_delete: bool = False,
        enabled: bool = False,
    ):
        self.absent = absent
        self.tags = list(tags if tags is not None else _owned_tags())
        self.distribution_id = distribution_id
        self.distribution_arn = distribution_arn or _distribution_arn(
            DISTRIBUTION_ID
        )
        self.fail_delete = fail_delete
        self.enabled = enabled
        self.calls: list[str] = []

    def get_distribution(self, **_kwargs):
        self.calls.append("get_distribution")
        if self.absent:
            raise _aws_error("NoSuchDistribution", "GetDistribution")
        return {
            "ETag": "etag-1",
            "Distribution": {
                "Id": self.distribution_id,
                "ARN": self.distribution_arn,
                "Status": "Deployed",
                "DistributionConfig": {"Enabled": self.enabled},
            },
        }

    def list_tags_for_resource(self, **_kwargs):
        self.calls.append("list_tags_for_resource")
        return {"Tags": {"Items": self.tags}}

    def delete_distribution(self, **_kwargs):
        self.calls.append("delete_distribution")
        if self.fail_delete:
            raise RuntimeError("injected CloudFront deletion failure")

    def update_distribution(self, **kwargs):
        self.calls.append("update_distribution")
        self.enabled = bool(kwargs["DistributionConfig"]["Enabled"])


class _DeleteEC2:
    def __init__(
        self,
        *,
        instance_absent: bool = False,
        group_absent: bool = False,
        instance_tags: list[dict[str, str]] | None = None,
        instance_id: str = INSTANCE_ID,
        group_arn: str | None = None,
    ):
        self.instance_absent = instance_absent
        self.group_absent = group_absent
        self.instance_tags = list(
            instance_tags if instance_tags is not None else _owned_tags()
        )
        self.instance_id = instance_id
        self.group_arn = group_arn or _resource_arn(
            kind="security-group", resource_id=SECURITY_GROUP_ID
        )
        self.calls: list[str] = []

    def describe_instances(self, **_kwargs):
        self.calls.append("describe_instances")
        if self.instance_absent:
            return {"Reservations": []}
        return {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": self.instance_id,
                            "State": {"Name": "running"},
                            "Tags": self.instance_tags,
                        }
                    ]
                }
            ]
        }

    def terminate_instances(self, **_kwargs):
        self.calls.append("terminate_instances")

    def get_waiter(self, _name):
        return _Waiter()

    def describe_security_groups(self, **_kwargs):
        self.calls.append("describe_security_groups")
        if self.group_absent:
            raise _aws_error("InvalidGroup.NotFound", "DescribeSecurityGroups")
        return {
            "SecurityGroups": [
                {
                    "GroupId": SECURITY_GROUP_ID,
                    "OwnerId": PRODUCTION_ACCOUNT_ID,
                    "SecurityGroupArn": self.group_arn,
                    "Tags": _owned_tags(),
                }
            ]
        }

    def delete_security_group(self, **_kwargs):
        self.calls.append("delete_security_group")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("InstanceId", "i-0123456789abcdef0"),
        ("ImageId", "ami-0123456789abcdef0"),
        ("InstanceType", "m5.large"),
        ("SubnetId", "subnet-0123456789abcdef0"),
        ("VpcId", "vpc-0123456789abcdef0"),
        ("EnclaveOptions", {"Enabled": False}),
        ("MetadataOptions", {"HttpTokens": "optional"}),
    ),
)
def test_reference_gateway_must_match_exact_nitro_imdsv2_instance(field, value):
    ec2 = _EC2()
    ec2.reference[field] = value
    with pytest.raises(ProvisioningError, match="Nitro/IMDSv2"):
        _single_production_instance(ec2, "52.91.135.79")


def test_transient_stack_is_one_encrypted_nitro_host_with_tls_viewer_boundary(monkeypatch):
    ec2 = _EC2()
    cloudfront = _CloudFront()
    s3 = _S3()
    monkeypatch.setattr(
        "scripts.provision_production_parity_staging._wait_distribution",
        lambda _client, _distribution_id, *, enabled, timeout_seconds=1200: {
            "ARN": _distribution_arn(DISTRIBUTION_ID),
            "DomainName": "parity.cloudfront.net",
            "DistributionConfig": {"Enabled": enabled},
            "Status": "Deployed",
        },
    )
    state = create_stack(
        ec2=ec2,
        ssm=_SSM(),
        cloudfront=cloudfront,
        s3=s3,
        region="us-east-1",
        account_id="493765492819",
        run_id=RUN_ID,
        candidate_sha=SHA,
        production_gateway_ip="52.91.135.79",
        instance_profile_name="leadpoet-production-parity-runner",
        instance_type=None,
        volume_gib=PARITY_VOLUME_GIB,
    )
    assert state["instance_id"] == INSTANCE_ID
    assert state["account_id"] == PRODUCTION_ACCOUNT_ID
    assert state["region"] == PRODUCTION_REGION
    assert state["instance_arn"] == _resource_arn(
        kind="instance", resource_id=INSTANCE_ID
    )
    assert state["security_group_arn"] == _resource_arn(
        kind="security-group", resource_id=SECURITY_GROUP_ID
    )
    assert state["cloudfront_distribution_arn"] == _distribution_arn(
        DISTRIBUTION_ID
    )
    assert state["supabase_origin"] == "https://parity.cloudfront.net"
    assert state["artifact_bucket"].startswith(
        "leadpoet-parity-493765492819-"
    )
    assert s3.created["ObjectLockEnabledForBucket"] is True
    assert (
        s3.lock["ObjectLockConfiguration"]["Rule"]["DefaultRetention"]
        == {"Mode": "COMPLIANCE", "Days": 1}
    )
    assert ec2.launch["MinCount"] == ec2.launch["MaxCount"] == 1
    assert ec2.launch["ImageId"] == PRODUCTION_AMI_ID
    assert ec2.launch["InstanceType"] == PRODUCTION_INSTANCE_TYPE
    assert ec2.launch["IamInstanceProfile"] == {
        "Name": "leadpoet-production-parity-runner"
    }
    assert ec2.launch["NetworkInterfaces"] == [{
        "DeviceIndex": 0,
        "SubnetId": PRODUCTION_SUBNET_ID,
        "Groups": [SECURITY_GROUP_ID],
        "AssociatePublicIpAddress": True,
        "DeleteOnTermination": True,
    }]
    assert ec2.launch["EnclaveOptions"] == {"Enabled": True}
    assert ec2.launch["MetadataOptions"] == {
        "HttpEndpoint": "enabled",
        "HttpTokens": "required",
        "HttpPutResponseHopLimit": 2,
        "InstanceMetadataTags": "enabled",
    }
    assert ec2.launch["TagSpecifications"] == [
        {"ResourceType": resource_type, "Tags": _owned_tags()}
        for resource_type in ("instance", "volume", "network-interface")
    ]
    assert ec2.launch["UserData"].startswith("#cloud-boothook")
    assert "systemctl mask --now" in ec2.launch["UserData"]
    assert "leadpoet-production-parity/early-boot-isolated" in ec2.launch["UserData"]
    root = ec2.launch["BlockDeviceMappings"][0]["Ebs"]
    assert root["VolumeSize"] == PARITY_VOLUME_GIB
    assert root["Encrypted"] is True
    assert root["DeleteOnTermination"] is True
    assert root["VolumeType"] == "gp3"
    assert root["VolumeSize"] == 512
    permission = ec2.ingress["IpPermissions"][0]
    assert permission["FromPort"] == permission["ToPort"] == 3000
    assert permission["PrefixListIds"] == [{"PrefixListId": "pl-cloudfront"}]
    assert "IpRanges" not in permission
    assert cloudfront.config["DefaultCacheBehavior"]["ViewerProtocolPolicy"] == "https-only"
    assert cloudfront.config["Origins"]["Items"][0]["CustomOriginConfig"]["HTTPPort"] == 3000
    assert cloudfront.tags == {"Items": _owned_tags()}


@pytest.mark.parametrize(
    ("profile", "instance_type"),
    (
        ("other-profile", None),
        ("leadpoet-production-parity-runner", "m5.large"),
    ),
)
def test_transient_stack_rejects_unpinned_profile_or_instance_type(
    profile, instance_type
):
    s3 = _S3()
    with pytest.raises(ProvisioningError, match="inputs|instance type"):
        create_stack(
            ec2=_EC2(),
            ssm=_SSM(),
            cloudfront=_CloudFront(),
            s3=s3,
            region=PRODUCTION_REGION,
            account_id=PRODUCTION_ACCOUNT_ID,
            run_id=RUN_ID,
            candidate_sha=SHA,
            production_gateway_ip="52.91.135.79",
            instance_profile_name=profile,
            instance_type=instance_type,
            volume_gib=PARITY_VOLUME_GIB,
        )
    assert s3.created is None


def test_partial_provisioning_failure_immediately_rolls_back_costly_resources():
    class _FailingCloudFront(_CloudFront):
        def create_distribution_with_tags(self, **_kwargs):
            raise RuntimeError("injected distribution failure")

    ec2 = _EC2()
    s3 = _S3()
    with pytest.raises(RuntimeError, match="injected distribution failure"):
        create_stack(
            ec2=ec2,
            ssm=_SSM(),
            cloudfront=_FailingCloudFront(),
            s3=s3,
            region="us-east-1",
            account_id="493765492819",
            run_id=RUN_ID,
            candidate_sha=SHA,
            production_gateway_ip="52.91.135.79",
            instance_profile_name="leadpoet-production-parity-runner",
            instance_type=None,
            volume_gib=PARITY_VOLUME_GIB,
        )
    assert ec2.terminated == [INSTANCE_ID]
    assert ec2.deleted_groups == [SECURITY_GROUP_ID]
    assert len(s3.deleted) == 1
    assert s3.deleted[0].startswith("leadpoet-parity-493765492819-")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("account_id", "000000000000"),
        ("region", "us-west-2"),
        ("status", "creating"),
        ("run_id", "run-123456"),
        ("candidate_sha", "b" * 39),
        ("instance_id", "i-wrong"),
        ("security_group_id", "sg-wrong"),
        ("cloudfront_distribution_id", "wrong"),
        ("instance_arn", "arn:aws:ec2:us-east-1:000000000000:instance/i-wrong"),
        ("volume_gib", 511),
        ("artifact_bucket", "leadpoet-parity-wrong"),
    ),
)
def test_delete_rejects_unbound_state_before_any_resource_read(field, value):
    state = _delete_state(**{field: value})
    ec2 = _DeleteEC2()
    cloudfront = _DeleteCloudFront()
    with pytest.raises(ProvisioningError):
        delete_stack(ec2=ec2, cloudfront=cloudfront, state=state)
    assert ec2.calls == []
    assert cloudfront.calls == []


@pytest.mark.parametrize(
    "identity",
    (
        {"account_id": "000000000000"},
        {"region": "us-west-2"},
    ),
)
def test_delete_rejects_wrong_live_aws_identity_before_resource_read(identity):
    ec2 = _DeleteEC2()
    cloudfront = _DeleteCloudFront()
    with pytest.raises(ProvisioningError, match="AWS deletion identity"):
        delete_stack(
            ec2=ec2,
            cloudfront=cloudfront,
            state=_delete_state(),
            **identity,
        )
    assert ec2.calls == []
    assert cloudfront.calls == []


def test_delete_reads_exact_live_ownership_before_each_mutation():
    ec2 = _DeleteEC2()
    cloudfront = _DeleteCloudFront()
    result = delete_stack(
        ec2=ec2, cloudfront=cloudfront, state=_delete_state()
    )
    assert result["distribution_deleted"] is True
    assert result["instance_terminated"] is True
    assert result["security_group_deleted"] is True
    assert cloudfront.calls == [
        "get_distribution",
        "list_tags_for_resource",
        "get_distribution",
        "list_tags_for_resource",
        "delete_distribution",
    ]
    assert ec2.calls.index("describe_instances") < ec2.calls.index(
        "terminate_instances"
    )
    assert ec2.calls.index("describe_security_groups") < ec2.calls.index(
        "delete_security_group"
    )


def test_delete_revalidates_cloudfront_after_disabling_before_delete():
    ec2 = _DeleteEC2()
    cloudfront = _DeleteCloudFront(enabled=True)
    delete_stack(ec2=ec2, cloudfront=cloudfront, state=_delete_state())
    update_index = cloudfront.calls.index("update_distribution")
    delete_index = cloudfront.calls.index("delete_distribution")
    tag_reads = [
        index
        for index, call in enumerate(cloudfront.calls)
        if call == "list_tags_for_resource"
    ]
    assert tag_reads[0] < update_index
    assert any(update_index < index < delete_index for index in tag_reads)


def test_delete_is_idempotent_when_every_resource_is_already_absent():
    ec2 = _DeleteEC2(instance_absent=True, group_absent=True)
    cloudfront = _DeleteCloudFront(absent=True)
    result = delete_stack(
        ec2=ec2, cloudfront=cloudfront, state=_delete_state()
    )
    assert result["distribution_deleted"] is True
    assert result["instance_terminated"] is True
    assert result["security_group_deleted"] is True
    assert "terminate_instances" not in ec2.calls
    assert "delete_security_group" not in ec2.calls
    assert "delete_distribution" not in cloudfront.calls


def test_delete_aggregates_failure_after_other_resource_attempts():
    ec2 = _DeleteEC2()
    cloudfront = _DeleteCloudFront(fail_delete=True)
    with pytest.raises(
        ProvisioningError, match="cloudfront:RuntimeError"
    ):
        delete_stack(ec2=ec2, cloudfront=cloudfront, state=_delete_state())
    assert "terminate_instances" in ec2.calls
    assert "delete_security_group" in ec2.calls


def test_delete_refuses_wrong_live_tags_but_cleans_independent_resources():
    wrong_tags = _owned_tags()
    wrong_tags[1] = {"Key": "leadpoet:candidate-sha", "Value": "b" * 40}
    ec2 = _DeleteEC2(instance_tags=wrong_tags)
    cloudfront = _DeleteCloudFront()
    with pytest.raises(
        ProvisioningError, match="instance:ProvisioningError"
    ):
        delete_stack(ec2=ec2, cloudfront=cloudfront, state=_delete_state())
    assert "terminate_instances" not in ec2.calls
    assert "delete_security_group" in ec2.calls
    assert "delete_distribution" in cloudfront.calls


def test_delete_refuses_wrong_cloudfront_tags_without_mutating_distribution():
    wrong_tags = _owned_tags()
    wrong_tags[0] = {"Key": "leadpoet:parity-run", "Value": "pp-999999-1"}
    ec2 = _DeleteEC2()
    cloudfront = _DeleteCloudFront(tags=wrong_tags)
    with pytest.raises(
        ProvisioningError, match="cloudfront:ProvisioningError"
    ):
        delete_stack(ec2=ec2, cloudfront=cloudfront, state=_delete_state())
    assert "delete_distribution" not in cloudfront.calls
    assert "terminate_instances" in ec2.calls
    assert "delete_security_group" in ec2.calls


@pytest.mark.parametrize(
    "ec2",
    (
        _DeleteEC2(instance_id="i-11111111111111111"),
        _DeleteEC2(
            group_arn=(
                "arn:aws:ec2:us-east-1:493765492819:"
                "security-group/sg-11111111111111111"
            )
        ),
    ),
)
def test_delete_refuses_wrong_live_resource_identity(ec2):
    cloudfront = _DeleteCloudFront()
    with pytest.raises(ProvisioningError, match="ProvisioningError"):
        delete_stack(ec2=ec2, cloudfront=cloudfront, state=_delete_state())


def test_delete_refuses_wrong_live_cloudfront_arn_without_mutating_it():
    ec2 = _DeleteEC2()
    cloudfront = _DeleteCloudFront(
        distribution_arn=(
            "arn:aws:cloudfront::000000000000:distribution/"
            f"{DISTRIBUTION_ID}"
        )
    )
    with pytest.raises(
        ProvisioningError, match="cloudfront:ProvisioningError"
    ):
        delete_stack(ec2=ec2, cloudfront=cloudfront, state=_delete_state())
    assert "delete_distribution" not in cloudfront.calls
    assert "terminate_instances" in ec2.calls
    assert "delete_security_group" in ec2.calls


def test_cleanup_requires_all_exact_ownership_tags():
    tags = [
        {"Key": "leadpoet:parity-run", "Value": RUN_ID},
        {"Key": "leadpoet:candidate-sha", "Value": SHA},
        {"Key": "leadpoet:ephemeral", "Value": "true"},
    ]
    assert _owned_run(tags) == RUN_ID
    assert _owned_run(tags[:-1]) is None
    assert _owned_run([{**tags[0]}, tags[2]]) is None


class _EmptyPaginator:
    def paginate(self, **_kwargs):
        return []


class _CleanupClient:
    def describe_instances(self, **_kwargs):
        return {"Reservations": []}

    def describe_security_groups(self, **_kwargs):
        return {"SecurityGroups": []}

    def list_distributions(self, **_kwargs):
        return {"DistributionList": {"Items": [], "IsTruncated": False}}

    def get_paginator(self, _name):
        return _EmptyPaginator()

    def list_buckets(self):
        return {"Buckets": []}


def test_cleanup_dry_run_is_non_destructive_and_empty_without_owned_resources():
    client = _CleanupClient()
    result = cleanup_stale(
        ec2=client,
        cloudfront=client,
        secretsmanager=client,
        s3=client,
        now=datetime.now(timezone.utc),
        max_age_hours=30,
        apply=False,
    )
    assert result["mode"] == "dry-run"
    assert result["runs"] == []
    assert result["instances"] == []


def test_full_workflow_uses_exact_candidate_and_tears_down_without_testnet():
    source = (ROOT / ".github/workflows/physical-v2-staging.yml").read_text()
    assert "Production Parity Full" in source
    assert "CANDIDATE_SHA" in source
    assert "scripts/provision_production_parity_staging.py" in source
    assert "scripts/run_production_parity_full_host.py" in source
    assert "if: always()" in source
    assert "testnet" not in source.lower()
    assert "funded" not in source.lower()
    assert "environment:" not in source
    assert "LEADPOET_PARITY_MINER_INTAKE_SECRET_ID" in source
    assert "leadpoet.production_parity_full.v3" in source
    assert 'test "$AWS_REGION" = "us-east-1"' in source
    assert "export AWS_REGION={q(required['AWS_REGION'])}" in source
    assert "export AWS_DEFAULT_REGION={q(required['AWS_REGION'])}" in source
    assert 'get("external_write_boundaries", {}).get("arweave")' in source
    assert '!= "blocked-production-parity"' in source
    assert "leadpoet.production_parity_arena_rebenchmark_evidence.v1" in source
    assert (
        "https://github.com/leadpoet/pydantic-harness/"
        "archive/refs/heads/main.tar.gz"
    ) in source
    assert 'arena_counts.get("accepted_execute_runs")' in source
    assert 'arena_counts.get("accepted_score_runs")' in source
    assert "configured_icps != 20" in source
    assert "per_icp_evidence_is_complete" in source
    assert 'item.get("execute_accepted") is True' in source
    assert 'item.get("score_accepted") is True' in source
    assert '"valid_company_with_https_evidence_count"' in source
    assert '"successful_openrouter_execute_call_count"' in source
    assert '"successful_openrouter_score_settlement_count"' in source
    assert 'arena_recovery.get("service_restarted") is not True' in source


def test_full_workflow_fetches_exact_bundle_head_then_binds_canonical_main():
    source = (
        ROOT / ".github/workflows/physical-v2-staging.yml"
    ).read_text(encoding="utf-8")
    initialize = source.index('candidate_git -C "$candidate_repo" init')
    bundle_fetch = source.index(
        'candidate_git -C "$candidate_repo" fetch --no-tags'
    )
    exact_fetch = source.index(
        'test "$(candidate_git -C "$candidate_repo" rev-parse FETCH_HEAD)" ='
    )
    checkout = source.index(
        'candidate_git -C "$candidate_repo" checkout --detach'
    )
    exact_checkout = source.index(
        'test "$(candidate_git -C "$candidate_repo" rev-parse HEAD)" ='
    )
    canonical_origin = source.index(
        'candidate_git -C "$candidate_repo" remote add origin'
    )
    exact_origin = source.index(
        'test "$(candidate_git -C "$candidate_repo" remote get-url origin)" ='
    )
    fetch = source.index(
        'candidate_git -C "$candidate_repo" fetch --no-tags origin'
    )
    exact_main = source.index(
        'test "$(candidate_git -C "$candidate_repo" rev-parse origin/main)" ='
    )
    runner = source.index(
        '"$host_python" scripts/run_production_parity_full_host.py'
    )

    assert (
        initialize
        < bundle_fetch
        < exact_fetch
        < checkout
        < exact_checkout
        < canonical_origin
        < exact_origin
        < fetch
        < exact_main
        < runner
    )
    assert '"$candidate_git_bin" -c init.templateDir=' in source
    assert "/usr/bin/env -i" in source
    assert "sudo -n /usr/bin/dnf -q -y install git-core" in source
    assert "scripts/resolve_production_parity_controller_requirements.py" in source
    assert 'PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1' in source
    assert 'host_python="$host_venv/bin/python3"' in source
    assert 'test ! -e "$host_venv"' in source
    assert (
        'sudo -n /usr/bin/dnf -q -y install \\\n'
        '            python3.11-pip >/dev/null 2>&1'
        in source
    )
    assert (
        '/usr/bin/python3.11 -I -m venv "$host_venv"'
        in source
    )
    container_runtime_package = source.index(
        "host_bootstrap_step=container-runtime-package"
    )
    container_runtime_identity = source.index(
        "host_bootstrap_step=container-runtime-identity"
    )
    container_runtime_service = source.index(
        "host_bootstrap_step=container-runtime-service"
    )
    venv_absence = source.index("host_bootstrap_step=venv-absence")
    python_runtime_package = source.index("host_bootstrap_step=runtime-package")
    venv_create = source.index("host_bootstrap_step=venv-create")
    assert (
        container_runtime_package
        < container_runtime_identity
        < container_runtime_service
        < venv_absence
        < python_runtime_package
        < venv_create
    )
    assert "if [ ! -x /usr/bin/docker ]; then" in source
    assert "sudo -n /usr/bin/dnf -q -y install docker" in source
    assert "/usr/bin/rpm -qf /usr/bin/docker" in source
    assert "sudo -n /usr/bin/systemctl start docker.service" in source
    assert "sudo -n /usr/bin/docker info" in source
    assert "python3.11-pip-wheel" not in source
    assert "GIT_CONFIG_NOSYSTEM=1" in source
    assert "git clone" not in source


def test_parity_workflows_reject_non_main_code_before_aws_credentials():
    for relative_path in (
        ".github/workflows/production-parity-fast.yml",
        ".github/workflows/physical-v2-staging.yml",
    ):
        workflow = yaml.safe_load(
            (ROOT / relative_path).read_text(encoding="utf-8")
        )
        steps = workflow["jobs"]["validate"]["steps"]
        main_gate = next(
            index
            for index, step in enumerate(steps)
            if step.get("name")
            == "Require exact current main candidate before credentials"
        )
        local_action = next(
            index
            for index, step in enumerate(steps)
            if step.get("uses")
            == "./.github/actions/setup-production-parity-controller"
        )
        aws_credentials = next(
            index
            for index, step in enumerate(steps)
            if step.get("uses") == "aws-actions/configure-aws-credentials@v4"
        )

        assert "git rev-parse origin/main" in steps[main_gate]["run"]
        assert main_gate < local_action < aws_credentials


def test_full_manual_attestation_lookup_has_no_runner_cli_dependency():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/physical-v2-staging.yml").read_text(
            encoding="utf-8"
        )
    )
    gate = next(
        step
        for step in workflow["jobs"]["validate"]["steps"]
        if step.get("name")
        == "Require exact current main candidate before credentials"
    )["run"]

    assert "gh run list" not in gate
    assert "https://api.github.com/repos/" in gate
    assert 'os.environ.get("GH_TOKEN", "")' in gate
    assert "request.urlopen(api_request, timeout=30)" in gate
    assert 'run.get("head_sha") == candidate_sha' in gate
    assert 'run.get("head_branch") == "main"' in gate
    assert 'run.get("conclusion") == "success"' in gate


def test_full_bootstrap_diagnostic_exposes_only_a_bounded_substage():
    source = (
        ROOT / ".github/workflows/physical-v2-staging.yml"
    ).read_text(encoding="utf-8")
    allowed = {
        "container-runtime-package",
        "container-runtime-identity",
        "container-runtime-service",
        "venv-absence",
        "runtime-package",
        "venv-create",
        "venv-identity",
        "python-version",
        "requirements-resolve",
        "pip-bootstrap",
        "pip-install",
        "pip-check",
    }

    assert "LEADPOET_BOOTSTRAP_STEP=%s" in source
    assert "Report bounded bootstrap substage" in source
    assert "exact[0] if len(exact) == 1 else \"unavailable\"" in source
    assert all(marker in source for marker in allowed)
    assert "print(combined)" not in source


def test_full_host_binds_real_handoff_to_nonforwarding_primary_audit_path():
    source = (ROOT / "scripts/run_production_parity_full_host.py").read_text()
    required = (
        "capture_snapshot(",
        "restore_snapshot(",
        "gw_restart.sh",
        "_validate_real_handoff(",
        "--production-allocation",
        "primary/audit workflow did not consume the clone allocation",
        '"chain_boundary": "strict-non-forwarding"',
        "_run_miner_intake_path(",
        "/research-lab/source-adapters",
        '"chain_registration_boundary": "strict-ephemeral-hotkey"',
        "_run_arena_rebenchmark_path(",
        '"baseline_source_url": ARENA_BASELINE_SOURCE_URL',
        '"sandbox": "gvisor-runsc"',
        '"transport": "live-httpx"',
        '"publication_visible": True',
        '"production_database_mutated": False',
        '"production_chain_mutated": False',
    )
    assert all(item in source for item in required)
    forbidden = ("chain.submit_extrinsic(", "subtensor.set_weights(", "testnet")
    assert all(item not in source for item in forbidden)
    assert source.index("arena_rebenchmark = _run_arena_rebenchmark_path(") < source.index(
        'failure_stage = "weight-readiness"'
    )


def test_full_miner_intake_keeps_public_source_credentials_forbidden():
    source = (ROOT / "scripts/run_production_parity_full_host.py").read_text()
    assert '"RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false"' in source
    assert '"RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED": "false"' in source
    assert '"global_miner_submissions_enabled": False' in source
    assert '"source_add_paused": False' in source
    assert 'retired_response.status_code != 410' in source
    assert 'forbidden_response.status_code != 422' in source
    assert '"credential_transport": "operator-managed-production-contract"' in source
    assert "builtwith_credential in source_persistence" in source
    assert '"Authorization": f"API {credential}"' in source
    assert "KEY=" not in source


def test_fast_contract_binds_every_exact_miner_intake_source():
    source = (
        ROOT / "scripts/build_production_parity_contract.py"
    ).read_text()
    for path in (
        "gateway/research_lab/api.py",
        "gateway/research_lab/models.py",
        "gateway/research_lab/key_vault.py",
        "neurons/miner.py",
        "research_lab/source_add_miner.py",
    ):
        assert f'"{path}"' in source


def test_rehearsal_override_is_hash_bound_and_read_only():
    driver = (ROOT / "scripts/run_local_restart_rehearsal.py").read_text()
    workflow = (
        ROOT / "tests/restart_rehearsal/production_workflow_runner.py"
    ).read_text()
    assert "dst=/rehearsal-production-allocation.json,readonly" in driver
    assert "production allocation override hash differs" in workflow
    assert "production_allocation_hash" in workflow


def test_production_allocation_drives_candidate_weight_fixture():
    allocation = {
        "epoch": 24570,
        "lab_cap_percent": 20.0,
        "unallocated_percent": 15.0,
        "source_add_allocations": [],
        "reimbursement_allocations": [],
        "champion_allocations": [
            {
                "uid": 7,
                "miner_hotkey": "production-miner-hotkey",
                "paid_alpha_percent": 5.0,
            }
        ],
        "queued_champion_allocations": [],
    }
    allocation["allocation_hash"] = sha256_json(allocation)
    fixture = SanitizedWeightFixture(
        candidate_sha=SHA,
        epoch_id=30_000,
        production_allocation_doc=allocation,
    )
    snapshot = fixture.calculation_snapshot([], "sha256:" + "b" * 64)
    assert snapshot["research_lab_allocation_doc"] == allocation
    assert snapshot["metagraph_hotkeys"][7] == "production-miner-hotkey"
    bad = dict(allocation)
    bad["lab_cap_percent"] = 19.0
    fixture = SanitizedWeightFixture(
        candidate_sha=SHA,
        epoch_id=30_000,
        production_allocation_doc=bad,
    )
    with pytest.raises(ValueError, match="hash differs"):
        fixture.calculation_snapshot([], "sha256:" + "b" * 64)
