from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import temporary_testnet_signing_host as temporary_host


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
RUN_ID = "pp-123456-1"
INSTANCE_ID = "i-0123456789abcdef0"
GROUP_ID = "sg-0123456789abcdef0"
NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)


class _Waiter:
    def wait(self, **_kwargs):
        return None


class _SSM:
    def describe_instance_information(self, **_kwargs):
        return {
            "InstanceInformationList": [{
                "InstanceId": INSTANCE_ID,
                "PingStatus": "Online",
            }]
        }


class _EC2:
    def __init__(self):
        self.launch = None
        self.group_create = None
        self.instance_active = True
        self.group_active = True
        self.terminated: list[str] = []
        self.deleted_groups: list[str] = []

    def create_security_group(self, **kwargs):
        self.group_create = kwargs
        return {"GroupId": GROUP_ID}

    def run_instances(self, **kwargs):
        self.launch = kwargs
        return {"Instances": [{"InstanceId": INSTANCE_ID}]}

    def get_waiter(self, _name):
        return _Waiter()

    def _tags(self):
        assert self.launch is not None
        return self.launch["TagSpecifications"][0]["Tags"]

    def describe_instances(self, **kwargs):
        if not self.instance_active:
            return {"Reservations": []}
        if "Filters" in kwargs or kwargs.get("InstanceIds") == [INSTANCE_ID]:
            return {
                "Reservations": [{
                    "Instances": [{
                        "InstanceId": INSTANCE_ID,
                        "ImageId": temporary_host.AMI_ID,
                        "InstanceType": temporary_host.INSTANCE_TYPE,
                        "SubnetId": temporary_host.SUBNET_ID,
                        "VpcId": temporary_host.VPC_ID,
                        "State": {"Name": "running"},
                        "EnclaveOptions": {"Enabled": True},
                        "MetadataOptions": {"HttpTokens": "required"},
                        "Tags": self._tags(),
                    }]
                }]
            }
        raise AssertionError(kwargs)

    def describe_security_groups(self, **_kwargs):
        if not self.group_active:
            return {"SecurityGroups": []}
        assert self.group_create is not None
        return {
            "SecurityGroups": [{
                "GroupId": GROUP_ID,
                "VpcId": temporary_host.VPC_ID,
                "IpPermissions": [],
                "Tags": self.group_create["TagSpecifications"][0]["Tags"],
            }]
        }

    def terminate_instances(self, **kwargs):
        self.terminated.extend(kwargs["InstanceIds"])
        self.instance_active = False

    def delete_security_group(self, **kwargs):
        self.deleted_groups.append(kwargs["GroupId"])
        self.group_active = False


def _create(ec2: _EC2, *, ttl_hours: int = 6):
    return temporary_host.create_host(
        ec2=ec2,
        ssm=_SSM(),
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        run_id=RUN_ID,
        candidate_sha=SHA,
        ttl_hours=ttl_hours,
        now=NOW,
    )


def test_create_is_one_pinned_nitro_host_with_no_ingress_or_ssh_key():
    ec2 = _EC2()
    state = _create(ec2)

    assert state == {
        "schema_version": temporary_host.SCHEMA_VERSION,
        "status": "ready",
        "account_id": temporary_host.ACCOUNT_ID,
        "region": temporary_host.REGION,
        "run_id": RUN_ID,
        "candidate_sha": SHA,
        "function": "testnet401",
        "created_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(hours=6)).isoformat(),
        "instance_id": INSTANCE_ID,
        "security_group_id": GROUP_ID,
        "instance_type": "r7i.4xlarge",
        "volume_gib": 512,
        "nitro_enclaves_enabled": True,
        "ssm_online": True,
        "inbound_rules": 0,
        "required_combined_allocator": {
            "cpu_count": 10,
            "memory_mib": 66_560,
        },
    }
    launch = ec2.launch
    assert launch["MinCount"] == launch["MaxCount"] == 1
    assert launch["ImageId"] == temporary_host.AMI_ID
    assert launch["InstanceType"] == "r7i.4xlarge"
    assert launch["IamInstanceProfile"] == {
        "Name": "leadpoet-production-parity-runner"
    }
    assert launch["EnclaveOptions"] == {"Enabled": True}
    assert launch["NetworkInterfaces"] == [{
        "DeviceIndex": 0,
        "SubnetId": temporary_host.SUBNET_ID,
        "Groups": [GROUP_ID],
        "AssociatePublicIpAddress": True,
        "DeleteOnTermination": True,
    }]
    assert "KeyName" not in launch
    assert "PrivateIpAddress" not in launch["NetworkInterfaces"][0]
    assert ec2.group_create["VpcId"] == temporary_host.VPC_ID
    assert "IpPermissions" not in ec2.group_create
    assert "systemctl mask --now" in launch["UserData"]
    volume = launch["BlockDeviceMappings"][0]["Ebs"]
    assert volume == {
        "DeleteOnTermination": True,
        "Encrypted": True,
        "VolumeSize": 512,
        "VolumeType": "gp3",
    }


def test_existing_four_tag_policy_carries_exact_owner_and_expiry_in_name():
    tags = temporary_host._tags(
        run_id=RUN_ID,
        candidate_sha=SHA,
        expires_at=NOW + timedelta(hours=6),
    )
    values = {item["Key"]: item["Value"] for item in tags}
    assert set(values) == {
        "Name",
        "leadpoet:parity-run",
        "leadpoet:candidate-sha",
        "leadpoet:ephemeral",
    }
    assert values["Name"] == (
        "leadpoet-parity-pp-123456-1-testnet401-exp-1788890400"
    )
    assert temporary_host._owned_identity(tags) == (
        RUN_ID,
        SHA,
        NOW + timedelta(hours=6),
    )
    assert temporary_host._owned_identity(
        tags + [{"Key": "unexpected", "Value": "tag"}]
    ) is None


@pytest.mark.parametrize("ttl", (0, 13))
def test_create_rejects_unbounded_ttl_before_aws_writes(ttl):
    ec2 = _EC2()
    with pytest.raises(temporary_host.TemporaryHostError, match="inputs"):
        _create(ec2, ttl_hours=ttl)
    assert ec2.group_create is None
    assert ec2.launch is None


def test_expired_cleanup_terminates_only_after_protected_expiry():
    ec2 = _EC2()
    _create(ec2, ttl_hours=2)

    before = temporary_host.cleanup_hosts(
        ec2=ec2,
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        now=NOW + timedelta(hours=1),
        apply=True,
    )
    assert before["instances"] == []
    assert before["security_groups"] == []
    assert ec2.terminated == []

    after = temporary_host.cleanup_hosts(
        ec2=ec2,
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        now=NOW + timedelta(hours=2),
        apply=True,
    )
    assert after["instances"] == [INSTANCE_ID]
    assert after["security_groups"] == [GROUP_ID]
    assert after["residue"] == {}
    assert ec2.terminated == [INSTANCE_ID]
    assert ec2.deleted_groups == [GROUP_ID]


def test_exact_cleanup_does_not_wait_for_expiry():
    ec2 = _EC2()
    _create(ec2)
    result = temporary_host.cleanup_hosts(
        ec2=ec2,
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        now=NOW,
        apply=True,
        run_id=RUN_ID,
        candidate_sha=SHA,
    )
    assert result["selection"] == "exact-run"
    assert result["instances"] == [INSTANCE_ID]
    assert result["security_groups"] == [GROUP_ID]
    assert result["residue"] == {}


def test_temporary_workflow_reuses_oidc_route_and_scheduled_expiry_cleanup():
    workflow = (
        ROOT / ".github/workflows/temporary-testnet-signing-host.yml"
    ).read_text(
        encoding="utf-8"
    )
    cleanup = (
        ROOT / ".github/workflows/production-parity-cleanup.yml"
    ).read_text(encoding="utf-8")
    assert "name: Production Parity Full" in workflow
    assert "on:\n  workflow_dispatch:" in workflow
    assert "TESTNET_FUNCTION: ${{ inputs.function }}" in workflow
    assert 'test "$TESTNET_FUNCTION" = "testnet401"' in workflow
    assert "scripts/temporary_testnet_signing_host.py" in workflow
    assert "leadpoet-production-parity-runner" in workflow
    assert "cleanup-run" in workflow
    assert "KeyName" not in workflow
    assert "cleanup-expired --apply" in cleanup


def test_combined_one_host_nitro_capacity_leaves_bounded_parent_headroom():
    parent_vcpus = 16
    parent_memory_mib = 128 * 1024
    gateway_vcpus = 2 + 6
    gateway_memory_mib = (8 + 56) * 1024
    validator_vcpus = 2
    validator_memory_mib = 1024
    assert parent_vcpus - gateway_vcpus - validator_vcpus == 6
    assert parent_memory_mib - gateway_memory_mib - validator_memory_mib == 63 * 1024
