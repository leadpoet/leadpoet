from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.cleanup_production_parity_staging import (
    _owned_run,
    cleanup_stale,
)
from scripts.provision_production_parity_staging import (
    ProvisioningError,
    _single_production_instance,
    create_stack,
)
from leadpoet_canonical.attested_v2 import sha256_json
from tests.restart_rehearsal.sanitized_weight_fixture import (
    SanitizedWeightFixture,
)


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
RUN_ID = "run-123456"


def _reference_instance() -> dict:
    return {
        "ImageId": "ami-production",
        "InstanceId": "i-production",
        "InstanceType": "m6i.xlarge",
        "SubnetId": "subnet-production",
        "VpcId": "vpc-production",
        "RootDeviceName": "/dev/xvda",
        "EnclaveOptions": {"Enabled": True},
        "MetadataOptions": {"HttpTokens": "required"},
    }


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
                            "InstanceId": "i-parity",
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
        return {"GroupId": "sg-parity"}

    def authorize_security_group_ingress(self, **kwargs):
        self.ingress = kwargs

    def run_instances(self, **kwargs):
        self.launch = kwargs
        return {"Instances": [{"InstanceId": "i-parity"}]}

    def get_waiter(self, _name):
        return _Waiter()

    def terminate_instances(self, **kwargs):
        self.terminated.extend(kwargs["InstanceIds"])

    def delete_security_group(self, **kwargs):
        self.deleted_groups.append(kwargs["GroupId"])


class _Paginator:
    def __init__(self, policy_type):
        self.policy_type = policy_type

    def paginate(self, **_kwargs):
        if self.policy_type == "cache":
            yield {
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
                    ]
                }
            }
        else:
            yield {
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
                    ]
                }
            }


class _CloudFront:
    def __init__(self):
        self.config = None
        self.tags = None

    def get_paginator(self, name):
        return _Paginator("cache" if name == "list_cache_policies" else "origin")

    def create_distribution(self, **kwargs):
        self.config = kwargs["DistributionConfig"]
        return {
            "Distribution": {
                "Id": "EDIST",
                "ARN": "arn:aws:cloudfront::123:distribution/EDIST",
            }
        }

    def tag_resource(self, **kwargs):
        self.tags = kwargs


class _SSM:
    def describe_instance_information(self, **_kwargs):
        return {
            "InstanceInformationList": [
                {"InstanceId": "i-parity", "PingStatus": "Online"}
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


def test_reference_gateway_must_be_one_nitro_imdsv2_instance():
    ec2 = _EC2()
    assert _single_production_instance(ec2, "52.91.135.79")["ImageId"] == "ami-production"
    ec2.reference["EnclaveOptions"] = {"Enabled": False}
    with pytest.raises(ProvisioningError, match="Nitro/IMDSv2"):
        _single_production_instance(ec2, "52.91.135.79")


def test_transient_stack_is_one_encrypted_nitro_host_with_tls_viewer_boundary(monkeypatch):
    ec2 = _EC2()
    cloudfront = _CloudFront()
    s3 = _S3()
    monkeypatch.setattr(
        "scripts.provision_production_parity_staging._wait_distribution",
        lambda _client, _distribution_id, *, enabled, timeout_seconds=1200: {
            "ARN": "arn:aws:cloudfront::123:distribution/EDIST",
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
        volume_gib=512,
    )
    assert state["instance_id"] == "i-parity"
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
    assert ec2.launch["EnclaveOptions"] == {"Enabled": True}
    assert ec2.launch["MetadataOptions"]["HttpTokens"] == "required"
    assert ec2.launch["UserData"].startswith("#cloud-boothook")
    assert "systemctl mask --now" in ec2.launch["UserData"]
    assert "leadpoet-production-parity/early-boot-isolated" in ec2.launch["UserData"]
    root = ec2.launch["BlockDeviceMappings"][0]["Ebs"]
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


def test_partial_provisioning_failure_immediately_rolls_back_costly_resources():
    class _FailingCloudFront(_CloudFront):
        def create_distribution(self, **_kwargs):
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
            volume_gib=512,
        )
    assert ec2.terminated == ["i-parity"]
    assert ec2.deleted_groups == ["sg-parity"]
    assert len(s3.deleted) == 1
    assert s3.deleted[0].startswith("leadpoet-parity-493765492819-")


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


def test_parity_workflows_reject_non_main_code_before_aws_credentials():
    for relative_path in (
        ".github/workflows/production-parity-fast.yml",
        ".github/workflows/physical-v2-staging.yml",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        main_gate = source.index(
            "name: Require exact current main candidate before credentials"
        )
        local_action = source.index(
            "uses: ./.github/actions/setup-production-parity-controller"
        )
        aws_credentials = source.index(
            "uses: aws-actions/configure-aws-credentials@v4"
        )

        assert 'git rev-parse origin/main' in source
        assert main_gate < local_action < aws_credentials


def test_full_host_binds_real_handoff_to_nonforwarding_primary_audit_path():
    source = (ROOT / "scripts/run_production_parity_full_host.py").read_text()
    required = (
        "capture_snapshot(",
        "restore_snapshot(",
        "gw_restart.sh",
        "_wait_rebenchmark(",
        "_validate_real_handoff(",
        "--production-allocation",
        "primary/audit workflow did not consume the clone allocation",
        '"chain_boundary": "strict-non-forwarding"',
        "_run_miner_intake_path(",
        "/research-lab/openrouter-keys/credential-recipient",
        "/research-lab/openrouter-keys",
        "/research-lab/source-adapters",
        '"chain_registration_boundary": "strict-ephemeral-hotkey"',
        '"production_database_mutated": False',
        '"production_chain_mutated": False',
    )
    assert all(item in source for item in required)
    forbidden = ("chain.submit_extrinsic(", "subtensor.set_weights(", "testnet")
    assert all(item not in source for item in forbidden)


def test_full_miner_intake_keeps_public_source_credentials_forbidden():
    source = (ROOT / "scripts/run_production_parity_full_host.py").read_text()
    assert '"RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED": "false"' in source
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
        "leadpoet_canonical/credential_recipient_v2.py",
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
