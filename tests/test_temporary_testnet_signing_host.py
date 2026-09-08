from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import temporary_testnet_signing_host as temporary_host


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
RUN_ID = "pp-123456-1"
INSTANCE_ID = "i-0123456789abcdef0"
GROUP_ID = "sg-0123456789abcdef0"
VOLUME_ID = "vol-0123456789abcdef0"
NETWORK_INTERFACE_ID = "eni-0123456789abcdef0"
NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)


def test_direct_workflow_script_can_import_repository_siblings(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", (
            "import runpy, sys; runpy.run_path(sys.argv[1]); "
            "from scripts.provision_production_parity_staging "
            "import _create_artifact_bucket"
        ), str(ROOT / "scripts/temporary_testnet_signing_host.py")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_fixed_public_release_reader_executes_locally_and_writes_canonical_pair(tmp_path):
    from gateway.tee.release_channel_v2 import (
        build_release_channel_v2,
        build_release_lineage_v2,
    )
    from scripts.stage_temporary_testnet_weights_host import build_config
    from tests.test_release_channel_v2 import _gateway_manifest, _validator_manifest

    gateway = _gateway_manifest(SHA)
    validator = _validator_manifest(SHA)
    channel = build_release_channel_v2(
        gateway_release_manifest=gateway,
        validator_release_manifest=validator,
    )
    lineage = build_release_lineage_v2([channel], current_commit=SHA)
    gateway_path = tmp_path / "gateway.json"
    validator_path = tmp_path / "validator.json"
    lineage_path = tmp_path / "lineage.json"
    gateway_path.write_text(json.dumps(gateway))
    validator_path.write_text(json.dumps(validator))
    lineage_path.write_text(json.dumps(lineage))
    config = build_config(
        repository=ROOT,
        candidate=SHA,
        run_id=RUN_ID,
        instance_id=INSTANCE_ID,
        expiry=1788890400,
    )
    config["gateway"]["release_manifest"] = str(gateway_path)
    config["gateway"]["release_lineage"] = str(lineage_path)
    config["validator"]["release_manifest"] = str(validator_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    config_path.chmod(0o600)
    channel_output = tmp_path / "channel-output.json"
    lineage_output = tmp_path / "lineage-output.json"
    program = temporary_host.public_release_export_program(
        candidate_sha=SHA,
        repository=str(ROOT),
        config_path=str(config_path),
        channel_output=str(channel_output),
        lineage_output=str(lineage_output),
    )

    subprocess.run([sys.executable, "-I", "-c", program], check=True, timeout=30)

    assert channel_output.read_text().endswith("\n")
    assert lineage_output.read_text().endswith("\n")
    assert json.loads(channel_output.read_text()) == channel
    assert json.loads(lineage_output.read_text()) == lineage
    assert "secret" not in channel_output.read_text().lower()


class _Waiter:
    def wait(self, **_kwargs):
        return None


class _SSM:
    def __init__(self):
        self.sent = None

    def describe_instance_information(self, **_kwargs):
        return {
            "InstanceInformationList": [{
                "InstanceId": INSTANCE_ID,
                "PingStatus": "Online",
            }]
        }

    def send_command(self, **kwargs):
        self.sent = kwargs
        return {"Command": {"CommandId": "12345678-1234-1234-1234-123456789abc"}}

    def get_command_invocation(self, **_kwargs):
        return {
            "Status": "Success",
            "ResponseCode": 0,
            "StandardOutputContent": "expiry_timer_ready\n",
            "StandardErrorContent": "",
        }


class _EC2:
    def __init__(self):
        self.launch = None
        self.group_create = None
        self.instance_active = True
        self.group_active = True
        self.volume_active = True
        self.network_interface_active = True
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
                        "BlockDeviceMappings": [{
                            "DeviceName": "/dev/xvda",
                            "Ebs": {
                                "VolumeId": VOLUME_ID,
                                "DeleteOnTermination": True,
                            },
                        }],
                        "NetworkInterfaces": [{
                            "NetworkInterfaceId": NETWORK_INTERFACE_ID,
                            "Attachment": {"DeleteOnTermination": True},
                        }],
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

    def describe_volumes(self, **_kwargs):
        if not self.volume_active:
            return {"Volumes": []}
        return {"Volumes": [{
            "VolumeId": VOLUME_ID,
            "Encrypted": True,
            "Size": 512,
            "VolumeType": "gp3",
        }]}

    def describe_network_interfaces(self, **_kwargs):
        if not self.network_interface_active:
            return {"NetworkInterfaces": []}
        return {"NetworkInterfaces": [{
            "NetworkInterfaceId": NETWORK_INTERFACE_ID,
        }]}

    def terminate_instances(self, **kwargs):
        self.terminated.extend(kwargs["InstanceIds"])
        self.instance_active = False
        self.volume_active = False
        self.network_interface_active = False

    def delete_security_group(self, **kwargs):
        self.deleted_groups.append(kwargs["GroupId"])
        self.group_active = False


def _create(ec2: _EC2, *, ttl_hours: int = 6, ssm: _SSM | None = None):
    return temporary_host.create_host(
        ec2=ec2,
        ssm=ssm or _SSM(),
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        run_id=RUN_ID,
        candidate_sha=SHA,
        ttl_hours=ttl_hours,
        now=NOW,
    )


def test_create_is_one_pinned_nitro_host_with_no_ingress_or_ssh_key():
    ec2 = _EC2()
    ssm = _SSM()
    state = _create(ec2, ssm=ssm)

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
        "volume_id": VOLUME_ID,
        "network_interface_id": NETWORK_INTERFACE_ID,
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
    assert "OnCalendar=@1788890400" in launch["UserData"]
    assert "leadpoet-testnet401-expiry.timer" in launch["UserData"]
    assert ssm.sent["InstanceIds"] == [INSTANCE_ID]
    assert ssm.sent["DocumentName"] == "AWS-RunShellScript"
    timer_probe = ssm.sent["Parameters"]["commands"]
    assert len(timer_probe) == 1
    assert "leadpoet-testnet401-expiry.timer" in timer_probe[0]
    assert "1788890400" in timer_probe[0]
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


class _StageSSM(_SSM):
    def __init__(self, stdout: str | list[str]):
        super().__init__()
        self.stdout = [stdout] if isinstance(stdout, str) else list(stdout)
        self.commands = []

    def send_command(self, **kwargs):
        self.commands.append(kwargs)
        return super().send_command(**kwargs)

    def get_command_invocation(self, **_kwargs):
        return {
            "Status": "Success",
            "ResponseCode": 0,
            "StandardOutputContent": self.stdout.pop(0),
            "StandardErrorContent": "",
        }


def test_source_bootstrap_is_fixed_to_exact_private_parity_prefix():
    ec2 = _EC2()
    _create(ec2)
    bucket = temporary_host._artifact_bucket_name(
        run_id=RUN_ID,
        candidate_sha=SHA,
    )
    prefix = f"production-parity/runs/{RUN_ID}/testnet401"
    ssm = _StageSSM("temporary_testnet401_source_ready\n")

    result = temporary_host.run_source_bootstrap(
        ec2=ec2,
        ssm=ssm,
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        run_id=RUN_ID,
        candidate_sha=SHA,
        instance_id=INSTANCE_ID,
        assets_bucket=bucket,
        assets_prefix=prefix,
        now=NOW,
    )

    assert result["status"] == "ready"
    command = ssm.sent["Parameters"]["commands"][0]
    assert ssm.sent["Parameters"]["executionTimeout"] == [
        str(ssm.sent["TimeoutSeconds"])
    ]
    assert "scripts.stage_temporary_testnet_weights_host" in command
    assert f"--candidate-sha {SHA}" in command
    assert f"--run-id {RUN_ID}" in command
    assert f"--instance-id {INSTANCE_ID}" in command
    assert f"--assets-bucket {bucket}" in command
    assert f"--assets-prefix {prefix}" in command
    assert "/run/leadpoet-testnet401/config.json" in command
    assert "candidate-bundle-binding.json" in command
    assert "refs/heads/main:refs/remotes/origin/main" not in command
    assert f"update-ref refs/remotes/origin/main {SHA}" in command
    assert f'rev-parse origin/main)" = {SHA}' in command
    assert "--requirement /run/leadpoet-testnet401/requirements.txt" in command
    terminal = subprocess.run(["bash", "-c", command.splitlines()[-1]],
                              capture_output=True, text=True, timeout=5, check=True)
    assert terminal.stdout == "temporary_testnet401_source_ready\n"
    assert not terminal.stderr
    host_dependencies = command.index(
        "aws-nitro-enclaves-cli aws-nitro-enclaves-cli-devel docker rsync jq tar gzip"
    )
    docker_ready = command.index("/usr/bin/docker info")
    native_stage = command.index("-m scripts.stage_temporary_testnet_weights_host")
    assert host_dependencies < docker_ready < native_stage
    assert "/usr/bin/systemctl enable --now docker.service" in command
    assert "test -x /usr/bin/curl" in command
    assert "install curl" not in command
    assert "nitro-enclaves-allocator.service" not in command

    with pytest.raises(temporary_host.TemporaryHostError, match="assets differ"):
        temporary_host.source_bootstrap_command(
            run_id=RUN_ID,
            candidate_sha=SHA,
            instance_id=INSTANCE_ID,
            assets_bucket=bucket,
            assets_prefix="attacker-controlled/prefix",
        )


def test_source_bootstrap_loads_only_fixed_prior_release_objects():
    bucket = temporary_host._artifact_bucket_name(run_id=RUN_ID, candidate_sha=SHA)
    prefix = f"production-parity/runs/{RUN_ID}/testnet401"
    prior = "b" * 40

    command = temporary_host.source_bootstrap_command(
        run_id=RUN_ID,
        candidate_sha=SHA,
        instance_id=INSTANCE_ID,
        assets_bucket=bucket,
        assets_prefix=prefix,
        prior_release_commit=prior,
    )

    for name in temporary_host.PUBLIC_RELEASE_ASSET_NAMES:
        assert f"--key {prefix}/{name}" in command
        assert f"/run/leadpoet-testnet401/{name}" in command
    assert f"--prior-release-commit {prior}" in command
    assert "prior-release-run" not in command


def test_asset_bucket_reuses_locked_parity_bucket_and_exact_prefix(
    monkeypatch, tmp_path
):
    bundle = tmp_path / "candidate.bundle"
    binding = tmp_path / "candidate-bundle-binding.json"
    bundle.write_bytes(b"bounded candidate bundle")
    digest = __import__("hashlib").sha256(bundle.read_bytes()).hexdigest()
    binding.write_text(
        json.dumps({
            "candidate-sha": SHA,
            "bundle-sha256": digest,
            "bundle-size-bytes": str(bundle.stat().st_size),
        }),
        encoding="utf-8",
    )
    uploads = []

    class S3:
        def upload_file(self, *args, **kwargs):
            uploads.append((args, kwargs))

    expected_bucket = temporary_host._artifact_bucket_name(
        run_id=RUN_ID, candidate_sha=SHA
    )
    calls = []

    def create_bucket(s3, **kwargs):
        calls.append((s3, kwargs))
        return expected_bucket

    from scripts import provision_production_parity_staging as parity_provision

    monkeypatch.setattr(parity_provision, "_create_artifact_bucket", create_bucket)
    s3 = S3()
    result = temporary_host.create_asset_bucket(
        s3=s3,
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        run_id=RUN_ID,
        candidate_sha=SHA,
        bundle_path=bundle,
        binding_path=binding,
    )

    assert calls == [(s3, {
        "region": temporary_host.REGION,
        "account_id": temporary_host.ACCOUNT_ID,
        "run_id": RUN_ID,
        "candidate_sha": SHA,
    })]
    assert result["bucket"] == expected_bucket
    assert result["prefix"] == f"production-parity/runs/{RUN_ID}/testnet401"
    assert result["compliance_retention_days"] == 1
    assert [item[0][2] for item in uploads] == result["source_objects"]
    assert all(item[1] == {"ExtraArgs": {"ServerSideEncryption": "AES256"}} for item in uploads)


def test_controller_wait_uses_locked_bucket_when_object_retention_is_hidden():
    calls = []

    class S3:
        def get_bucket_tagging(self, **_kwargs):
            return {"TagSet": [
                {"Key": "leadpoet:parity-run", "Value": RUN_ID},
                {"Key": "leadpoet:candidate-sha", "Value": SHA},
                {"Key": "leadpoet:ephemeral", "Value": "true"},
                {"Key": "Name", "Value": f"leadpoet-parity-{RUN_ID}"},
            ]}

        def get_bucket_versioning(self, **_kwargs):
            return {"Status": "Enabled"}

        def get_object_lock_configuration(self, **_kwargs):
            return {"ObjectLockConfiguration": {
                "ObjectLockEnabled": "Enabled",
                "Rule": {"DefaultRetention": {"Mode": "COMPLIANCE", "Days": 1}},
            }}

        def get_public_access_block(self, **_kwargs):
            return {"PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            }}

        def get_bucket_encryption(self, **_kwargs):
            return {"ServerSideEncryptionConfiguration": {"Rules": [{
                "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}
            }]}}

        def head_object(self, **kwargs):
            calls.append(kwargs)
            return {
                "ContentLength": 123,
                "ServerSideEncryption": "AES256",
            }

    result = temporary_host.wait_for_private_assets(
        s3=S3(),
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        run_id=RUN_ID,
        candidate_sha=SHA,
        timeout_seconds=1800,
    )
    prefix = f"production-parity/runs/{RUN_ID}/testnet401"
    assert result["status"] == "ready"
    assert [item["Key"] for item in calls] == [
        f"{prefix}/{name}" for name in temporary_host.PRIVATE_ASSET_NAMES
    ]


def test_asset_cleanup_removes_only_staging_heads_and_preserves_proof_prefix():
    deleted = []

    class S3:
        def get_bucket_tagging(self, **_kwargs):
            return {"TagSet": [
                {"Key": "leadpoet:parity-run", "Value": RUN_ID},
                {"Key": "leadpoet:candidate-sha", "Value": SHA},
                {"Key": "leadpoet:ephemeral", "Value": "true"},
                {"Key": "Name", "Value": f"leadpoet-parity-{RUN_ID}"},
            ]}

        def delete_object(self, **kwargs):
            deleted.append(kwargs)

    result = temporary_host.remove_staging_asset_heads(
        s3=S3(),
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        run_id=RUN_ID,
        candidate_sha=SHA,
    )
    expected = [
        f"production-parity/runs/{RUN_ID}/testnet401/{name}"
        for name in (
            *temporary_host.SOURCE_ASSET_NAMES,
            *temporary_host.PRIVATE_ASSET_NAMES,
            *temporary_host.PUBLIC_RELEASE_ASSET_NAMES,
        )
    ]
    assert [item["Key"] for item in deleted] == expected
    assert result["staging_object_heads_removed"] == expected
    assert result["proof_prefix_preserved"].endswith("/evidence/")


@pytest.mark.parametrize(
    ("stage", "confirmed"),
    (("preflight", False), ("launch", True), ("status", False), ("cleanup", True)),
)
def test_native_ssm_stage_exposes_no_arbitrary_command(stage, confirmed):
    ec2 = _EC2()
    _create(ec2)
    receipt = {
        "schema_version": temporary_host.NATIVE_RECEIPT_SCHEMA_VERSION,
        "stage": stage,
        "status": "passed",
        "run_id": RUN_ID,
        "candidate_sha": SHA,
        "instance_id": INSTANCE_ID,
        "recorded_at_unix": int(NOW.timestamp()),
        "evidence": {},
    }
    outputs = [json.dumps(receipt, sort_keys=True) + "\n"]
    if stage == "status":
        outputs.append(json.dumps({
            "status": "ready",
            "logs": [
                {
                    "process_name": name, "process_live": True,
                    "log_size_bytes": 10, "tail_bytes_read": 10,
                    "progress_markers": [], "failure_markers": [],
                    "exception_types": [], "reason_codes": [],
                    "http_statuses": [], "allocation_build_callbacks": [],
                    "source_locations": [],
                }
                for name in ("gateway_application", "validator_application")
            ],
        }))
    ssm = _StageSSM(outputs)

    result = temporary_host.run_native_stage(
        ec2=ec2,
        ssm=ssm,
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        run_id=RUN_ID,
        candidate_sha=SHA,
        instance_id=INSTANCE_ID,
        stage=stage,
        now=NOW,
    )

    command = ssm.commands[0]["Parameters"]["commands"][0]
    assert result["receipt"] == receipt
    assert f"scripts.bootstrap_temporary_testnet_weights_host {stage}" in command
    assert "--config /run/leadpoet-testnet401/config.json" in command
    assert ("--confirm-instance-id" in command) is confirmed
    if stage == "status":
        assert "[ ! -f /run/leadpoet-testnet401/processes.json ]" in command
        assert "/run/leadpoet-testnet401/evidence/launch.json" in command
        assert "v.get" in command and "failed" in command
        diagnostic = ssm.commands[1]["Parameters"]["commands"][0]
        assert "gateway_application" in diagnostic
        assert "validator_application" in diagnostic
        assert "logs" in diagnostic
        assert result["runtime_log_diagnostics"]["status"] == "ready"
    assert ssm.sent["DocumentName"] == "AWS-RunShellScript"


def test_failed_ssm_command_reports_only_redacted_identity():
    class FailedSSM:
        def get_command_invocation(self, **_kwargs):
            return {
                "Status": "Failed",
                "ResponseCode": 51,
                "StandardOutputContent": (
                    '{"status":"failed","error_type":"RuntimeError",'
                    '"operation":"stage","code":"E51",'
                    '"location":"scripts/stage_host.py:314",'
                    '"secret":"secret-canary"}\n'
                ),
                "StandardErrorContent": (
                    'secret-canary File "/private/secret/runtime.py", line 42\n'
                    "build_role_enclaves.sh: line 105: unbound variable secret-canary"
                ),
            }

    with pytest.raises(temporary_host.TemporaryHostError) as failure:
        temporary_host._wait_ssm_command(
            FailedSSM(), instance_id=INSTANCE_ID,
            command_id="12345678-1234-1234-1234-123456789abc",
            timeout_seconds=1,
        )
    message = str(failure.value)
    assert "secret-canary" not in message
    evidence = json.loads(message.split("failed ", 1)[1])
    assert evidence["response_code"] == 51
    assert evidence["ssm_status"] == "Failed"
    assert evidence["native_failure"] == {
        "code": "E51", "error_type": "RuntimeError",
        "location": "scripts/stage_host.py:314", "operation": "stage",
    }
    assert evidence["source_locations"] == [{"file": "runtime.py", "line": 42}]
    assert evidence["shell_locations"] == [
        {"file": "build_role_enclaves.sh", "line": 105}
    ]


def test_native_ssm_stage_rechecks_owner_before_command():
    ec2 = _EC2()
    _create(ec2)
    ec2.launch["TagSpecifications"][0]["Tags"] = [
        {"Key": "leadpoet:ephemeral", "Value": "false"}
    ]
    ssm = _StageSSM("{}\n")

    with pytest.raises(temporary_host.TemporaryHostError, match="authority differs"):
        temporary_host.run_native_stage(
            ec2=ec2,
            ssm=ssm,
            account_id=temporary_host.ACCOUNT_ID,
            region=temporary_host.REGION,
            run_id=RUN_ID,
            candidate_sha=SHA,
            instance_id=INSTANCE_ID,
            stage="status",
            now=NOW,
        )
    assert ssm.sent is None


def test_staging_diagnostics_execute_without_runtime_and_redact_logs(tmp_path):
    program = temporary_host.staging_diagnostic_program(
        run_id=RUN_ID, candidate_sha=SHA, instance_id=INSTANCE_ID,
    )
    task = tmp_path / "task"
    logs = task / "staging-logs"
    logs.mkdir(parents=True)
    (logs / "native_host_dependencies.log").write_text(
        "secret-value-never-print\nModuleNotFoundError: hidden module\n"
        'File "/private/path/native.py", line 42\n'
        '/private/build_local_release_v2.sh: line 83: Killed private-arguments\n'
        'Building one local gateway identity for private-identity\n'
    )
    (task / "source-stage.json").write_text(
        '{"stage":"native_host_dependencies","status":"running","secret":"hidden"}\n'
        '{"status":"failed","error_type":"RuntimeError","operation":"stage",'
        '"code":"E51","location":"scripts/stage_host.py:314","secret":"hidden"}\n'
    )
    program = program.replace(repr(temporary_host.RUNTIME_ROOT), repr(str(task)))
    result = subprocess.run([sys.executable, "-I", "-c", program],
                            capture_output=True, text=True, timeout=10, check=True)
    assert not result.stderr
    assert "secret" not in result.stdout
    assert "hidden" not in result.stdout
    value = json.loads(result.stdout)
    assert value["status"] == "staging_incomplete"
    assert value["stage_states"] == [{"stage": "native_host_dependencies", "status": "running"}]
    assert value["staging_failure"] == {
        "code": "E51", "error_type": "RuntimeError",
        "location": "scripts/stage_host.py:314", "operation": "stage",
    }
    assert value["log_diagnostics"][0]["categories"] == ["ModuleNotFoundError", "Killed"]
    assert value["log_diagnostics"][0]["trace_locations"] == [["native.py", "42"]]
    assert value["log_diagnostics"][0]["shell_locations"] == [["build_local_release_v2.sh", "83"]]
    assert value["log_diagnostics"][0]["build_milestones"] == ["Building one local gateway identity"]
    assert "private-arguments" not in result.stdout
    assert "private-identity" not in result.stdout


def _runtime_log_probe(tmp_path, *, symlink_validator=False):
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "__init__.py").write_text("")
    (scripts / "bootstrap_temporary_testnet_weights_host.py").write_text(
        "import json\n"
        "def load_config(path): return json.loads(path.read_text())\n"
        "def _load_process_state(config): return config['process_state']\n"
        "def _same_process(record): return record.get('live') is True\n"
    )
    runtime = tmp_path / "runtime"
    logs = runtime / "logs"
    logs.mkdir(parents=True)
    canary = "provider-secret-canary=https://private.example/token-value"
    (logs / "gateway_application.log").write_text(
        canary + "\nApplication startup complete\n"
        'POST /weights/inputs/v2 HTTP/1.1" 503 private-request-body\n'
        "research_lab_allocation_build_failed epoch=22058 "
        "persist_snapshot=True error_type=HTTPException error=500: TypeError: "
        "champion_v2_cutover_readiness got an unexpected keyword argument "
        "_fresh_testnet401_empty_origin token=credential-canary "
        "https://private.example/raw-body\n"
        "research_lab_allocation_build_failed epoch=22058 "
        "persist_snapshot=True error_type=ResearchLabV2AuthorityError "
        "error=fresh testnet401 cutover authority is unavailable or ambiguous "
        "raw-payload=https://private.example/secret\n"
        + (
            "research_lab_allocation_build_failed epoch=22058 "
            "persist_snapshot=True error_type=HTTPException "
            "error=500: durable allocation retry is cooling down\n"
        ) * 3
    )
    validator = logs / "validator_application.log"
    target = tmp_path / "validator-real.log"
    target.write_text(
        canary + "\nSUBMITTING WEIGHTS FOR EPOCH 22058\n"
        "Block: 7961407 (block 10/360, 350 remaining)\n"
        "Authoritative V2 gateway bundle persisted: sha256:secret-value\n"
        "Authoritative V2 Research Lab allocation failed closed: HTTPError: "
        "HTTP Error 503: champion V2 cutover blocked: private counts\n"
        "chain-realized settlement activation is unavailable or ambiguous\n"
        "chain-realized settlement activation is invalid\n"
        "fresh testnet401 allocation origin is invalid\n"
        '{"event": "automatic_weight_tick_failed", '
        '"failure_type": "RuntimeError"} secret-private-detail\n'
        'File "/private/path/validator.py", line 5557\n'
    )
    if symlink_validator:
        validator.symlink_to(target)
    else:
        validator.write_bytes(target.read_bytes())
    config = {
        "run_id": RUN_ID, "candidate_sha": SHA,
        "expected_instance_id": INSTANCE_ID, "runtime_root": str(runtime),
        "process_state": {"processes": [
            {"name": "gateway_application", "live": True},
            {"name": "validator_application", "live": True},
        ]},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    program = temporary_host.runtime_log_diagnostic_program(
        run_id=RUN_ID, candidate_sha=SHA, instance_id=INSTANCE_ID,
        repository=str(repository), config_path=str(config_path),
    )
    return subprocess.run(
        [sys.executable, "-I", "-c", program], capture_output=True,
        text=True, timeout=10,
    )


def test_runtime_log_diagnostics_execute_and_return_only_allowlisted_fields(tmp_path):
    result = _runtime_log_probe(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "provider-secret-canary" not in result.stdout
    assert "private.example" not in result.stdout
    assert "secret-private-detail" not in result.stdout
    assert "secret-value" not in result.stdout
    assert "private-request-body" not in result.stdout
    assert "raw-payload" not in result.stdout
    value = json.loads(result.stdout)
    temporary_host._validate_runtime_log_diagnostics(value)
    gateway, validator = value["logs"]
    assert gateway["progress_markers"] == [
        "application_ready", "weight_inputs_request",
    ]
    assert gateway["http_statuses"] == [
        {"endpoint": "weight_inputs", "status": 503}
    ]
    assert gateway["failure_markers"] == ["allocation_build_failed"]
    assert gateway["exception_types"] == [
        "HTTPException", "ResearchLabV2AuthorityError",
    ]
    assert gateway["reason_codes"] == [
        "fresh_testnet401_cutover_authority_unavailable_or_ambiguous"
    ]
    assert gateway["allocation_build_callbacks"] == [
        {
            "error_type": "HTTPException",
            "tokens": [
                "typeerror", "champion_v2_cutover_readiness", "[redacted]",
                "unexpected", "keyword", "argument",
                "_fresh_testnet401_empty_origin",
            ],
        },
        {
            "error_type": "HTTPException",
            "tokens": [
                "durable", "allocation", "retry", "[redacted]", "cooling",
                "down",
            ],
        },
    ]
    assert validator["latest_epoch_id"] == 22058
    assert validator["latest_block"] == 7961407
    assert validator["progress_markers"] == [
        "epoch_submission_started", "gateway_bundle_persisted",
    ]
    assert validator["failure_markers"] == [
        "automatic_weight_tick_failed", "allocation_failed_closed",
    ]
    assert validator["reason_codes"] == [
        "champion_v2_cutover_blocked",
        "chain_realized_settlement_activation_unavailable_or_ambiguous",
        "chain_realized_settlement_activation_invalid",
        "fresh_testnet401_allocation_origin_invalid",
    ]
    assert validator["http_statuses"] == [
        {"endpoint": "allocation_handoff", "status": 503}
    ]
    assert validator["exception_types"] == ["HTTPError", "RuntimeError"]
    assert validator["source_locations"] == [
        {"file": "validator.py", "line": 5557}
    ]


def test_runtime_log_diagnostics_reject_symlink_and_arbitrary_remote_fields(tmp_path):
    result = _runtime_log_probe(tmp_path, symlink_validator=True)
    assert result.returncode != 0
    assert "provider-secret-canary" not in result.stdout + result.stderr
    with pytest.raises(temporary_host.TemporaryHostError, match="differ"):
        temporary_host._validate_runtime_log_diagnostics({
            "status": "ready", "logs": [
                {
                    "process_name": name, "process_live": True,
                    "log_size_bytes": 1, "tail_bytes_read": 1,
                    "progress_markers": [], "failure_markers": [],
                    "exception_types": [], "reason_codes": [],
                    "http_statuses": [], "allocation_build_callbacks": [],
                    "source_locations": [],
                    "raw_line": "provider-secret-canary",
                }
                for name in ("gateway_application", "validator_application")
            ],
        })


def test_failed_launch_diagnostics_include_only_bounded_runtime_identity(tmp_path):
    program = temporary_host.staging_diagnostic_program(
        run_id=RUN_ID, candidate_sha=SHA, instance_id=INSTANCE_ID,
    )
    task = tmp_path / "task"
    (task / "logs").mkdir(parents=True)
    (task / "evidence").mkdir()
    (task / "processes.json").write_text('{"secret":"secret-canary"}')
    (task / "logs" / "gateway_runtime_readiness.log").write_text(
        "secret-canary\nV2RuntimeReadinessError: coordinator provider broker "
        "is not ready\n"
        'File "/private/gateway/tee/verify_v2_runtime_ready.py", line 47\n'
    )
    (task / "evidence" / "launch.json").write_text(json.dumps({
        "schema_version": temporary_host.NATIVE_RECEIPT_SCHEMA_VERSION,
        "stage": "launch", "status": "failed", "run_id": RUN_ID,
        "candidate_sha": SHA, "instance_id": INSTANCE_ID,
        "evidence": {"failure_type": "TemporaryTestnetBootstrapError",
                     "secret": "secret-canary"},
    }))
    program = program.replace(repr(temporary_host.RUNTIME_ROOT), repr(str(task)))
    result = subprocess.run([sys.executable, "-I", "-c", program],
                            capture_output=True, text=True, timeout=10, check=True)
    assert "secret-canary" not in result.stdout
    value = json.loads(result.stdout)
    assert value["paths_present"]["processes.json"] is True
    assert value["launch_failure"] == {
        "failure_type": "TemporaryTestnetBootstrapError", "status": "failed",
    }
    runtime = next(row for row in value["log_diagnostics"]
                   if row["file"] == "gateway_runtime_readiness.log")
    assert runtime["categories"] == [
        "V2RuntimeReadinessError", "coordinator provider broker is not ready",
    ]
    assert runtime["trace_locations"] == [["verify_v2_runtime_ready.py", "47"]]


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


def test_exact_inventory_uses_exact_filters_and_consumes_every_page():
    ec2 = _EC2()
    _create(ec2)
    instance_calls = []
    group_calls = []
    original_instances = ec2.describe_instances
    original_groups = ec2.describe_security_groups

    def paged_instances(**kwargs):
        if "Filters" not in kwargs:
            return original_instances(**kwargs)
        instance_calls.append(kwargs)
        if "NextToken" not in kwargs:
            return {"Reservations": [], "NextToken": "instance-page-2"}
        assert kwargs["NextToken"] == "instance-page-2"
        return original_instances(Filters=kwargs["Filters"])

    def paged_groups(**kwargs):
        if "Filters" not in kwargs:
            return original_groups(**kwargs)
        group_calls.append(kwargs)
        if "NextToken" not in kwargs:
            return {"SecurityGroups": [], "NextToken": "group-page-2"}
        assert kwargs["NextToken"] == "group-page-2"
        return original_groups(Filters=kwargs["Filters"])

    ec2.describe_instances = paged_instances
    ec2.describe_security_groups = paged_groups
    result = temporary_host.cleanup_hosts(
        ec2=ec2,
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        now=NOW,
        apply=False,
        run_id=RUN_ID,
        candidate_sha=SHA,
    )

    expected_filters = [
        {"Name": "tag:leadpoet:ephemeral", "Values": ["true"]},
        {"Name": "tag:leadpoet:parity-run", "Values": [RUN_ID]},
        {"Name": "tag:leadpoet:candidate-sha", "Values": [SHA]},
    ]
    assert [call["Filters"] for call in instance_calls] == [
        expected_filters,
        expected_filters,
    ]
    assert [call["Filters"] for call in group_calls] == [
        expected_filters,
        expected_filters,
    ]
    assert result["instances"] == [INSTANCE_ID]
    assert result["security_groups"] == [GROUP_ID]


def test_inventory_fails_closed_on_repeated_page_token():
    class RepeatedTokenEC2:
        def describe_instances(self, **_kwargs):
            return {"Reservations": [], "NextToken": "same-token"}

    with pytest.raises(
        temporary_host.TemporaryHostError,
        match="pagination",
    ):
        temporary_host._inventory(
            RepeatedTokenEC2(),
            run_id=RUN_ID,
            candidate_sha=SHA,
        )


def test_cleanup_reports_ownership_loss_without_terminating():
    ec2 = _EC2()
    _create(ec2)
    original = ec2.describe_instances

    def ownership_changes(**kwargs):
        response = original(**kwargs)
        if "InstanceIds" in kwargs:
            response["Reservations"][0]["Instances"][0]["Tags"] = [
                {"Key": "leadpoet:ephemeral", "Value": "false"}
            ]
        return response

    ec2.describe_instances = ownership_changes
    result = temporary_host.cleanup_hosts(
        ec2=ec2,
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        now=NOW,
        apply=True,
        run_id=RUN_ID,
        candidate_sha=SHA,
    )

    assert ec2.terminated == []
    assert result["residue"]["instances"] == [INSTANCE_ID]
    assert result["errors"] == [
        f"instance:{INSTANCE_ID}:TemporaryHostError"
    ]


def test_cleanup_reports_security_group_ownership_loss_without_deleting():
    ec2 = _EC2()
    _create(ec2)
    ec2.instance_active = False
    ec2.volume_active = False
    ec2.network_interface_active = False
    original = ec2.describe_security_groups

    def ownership_changes(**kwargs):
        response = original(**kwargs)
        if "GroupIds" in kwargs:
            response["SecurityGroups"][0]["Tags"] = [
                {"Key": "leadpoet:ephemeral", "Value": "false"}
            ]
        return response

    ec2.describe_security_groups = ownership_changes
    result = temporary_host.cleanup_hosts(
        ec2=ec2,
        account_id=temporary_host.ACCOUNT_ID,
        region=temporary_host.REGION,
        now=NOW,
        apply=True,
        run_id=RUN_ID,
        candidate_sha=SHA,
    )

    assert ec2.deleted_groups == []
    assert result["residue"]["security_groups"] == [GROUP_ID]
    assert result["errors"] == [
        f"security-group:{GROUP_ID}:TemporaryHostError"
    ]


def test_failed_provision_writes_cleanup_incomplete_receipt_with_public_ids(
    monkeypatch, tmp_path, capsys
):
    class FailingEC2(_EC2):
        def terminate_instances(self, **_kwargs):
            raise RuntimeError("injected terminate failure")

        def delete_security_group(self, **_kwargs):
            raise RuntimeError("injected group failure")

        def get_waiter(self, name):
            if name == "volume_deleted":
                class FailedWaiter:
                    def wait(self, **_kwargs):
                        raise RuntimeError("injected volume residue")

                return FailedWaiter()
            return _Waiter()

    class FailingSSM(_SSM):
        def get_command_invocation(self, **_kwargs):
            return {
                "Status": "Failed",
                "ResponseCode": 1,
                "StandardOutputContent": "",
                "StandardErrorContent": "timer unavailable",
            }

    ec2 = FailingEC2()

    class STS:
        def get_caller_identity(self):
            return {"Account": temporary_host.ACCOUNT_ID}

    class Session:
        def client(self, name):
            return {"sts": STS(), "ec2": ec2, "ssm": FailingSSM()}[name]

    monkeypatch.setattr(temporary_host.boto3.session, "Session", lambda **_kwargs: Session())
    monkeypatch.setattr(temporary_host.time, "sleep", lambda _seconds: None)
    state = tmp_path / "failed-host.json"
    status = temporary_host.main([
        "--region",
        temporary_host.REGION,
        "create",
        "--run-id",
        RUN_ID,
        "--candidate-sha",
        SHA,
        "--ttl-hours",
        "6",
        "--state",
        str(state),
    ])

    assert status == 1
    receipt = json.loads(state.read_text(encoding="utf-8"))
    assert receipt["status"] == "provision_failed_cleanup_incomplete"
    assert receipt["instance_id"] == INSTANCE_ID
    assert receipt["security_group_id"] == GROUP_ID
    assert receipt["volume_id"] == VOLUME_ID
    assert receipt["network_interface_id"] == NETWORK_INTERFACE_ID
    assert receipt["rollback"]["cleanup_complete"] is False
    assert receipt["rollback"]["residue"] == {
        "instances": [INSTANCE_ID],
        "network_interfaces": [NETWORK_INTERFACE_ID],
        "security_groups": [GROUP_ID],
        "volumes": [VOLUME_ID],
    }
    assert "see receipt" in capsys.readouterr().err


def test_temporary_workflow_reuses_oidc_route_and_scheduled_expiry_cleanup():
    assert not (
        ROOT / ".github/workflows/temporary-testnet-signing-host.yml"
    ).exists()
    workflow = (ROOT / ".github/workflows/physical-v2-staging.yml").read_text(
        encoding="utf-8"
    )
    cleanup = (
        ROOT / ".github/workflows/production-parity-cleanup.yml"
    ).read_text(encoding="utf-8")
    assert "name: Production Parity Full" in workflow
    assert "inputs.operation == 'production-parity'" in workflow
    assert "- testnet401-create" in workflow
    assert "- testnet401-launch" in workflow
    assert "- testnet401-status" in workflow
    assert "- testnet401-cleanup" in workflow
    assert "inputs.operation != 'production-parity'" in workflow
    assert "format('testnet401-{0}'," in workflow
    assert "cancel-in-progress: >-" in workflow
    assert "scripts/temporary_testnet_signing_host.py" in workflow
    assert "create-assets" in workflow
    assert "--timeout-seconds 1800" in workflow
    assert 'TESTNET401_HOST_TTL_HOURS: "12"' in workflow
    assert "timeout-minutes: 660" in workflow
    assert "role-duration-seconds: 21600" in workflow
    assert "ssm-source-bootstrap" in workflow
    assert "ssm-native-stage" in workflow
    assert "for stage in preflight launch" in workflow
    assert "--stage status" in workflow
    assert "--stage cleanup" in workflow
    assert "cleanup-assets" in workflow
    assert "cleanup_production_parity_staging.py" in workflow
    assert "leadpoet-production-parity-runner" in workflow
    assert "cleanup-run" in workflow
    assert "KeyName" not in workflow
    assert "cleanup-expired --apply" in cleanup


def test_physical_workflow_wires_public_release_export_and_prior_pair():
    workflow = (
        ROOT / ".github/workflows/physical-v2-staging.yml"
    ).read_text(encoding="utf-8")

    assert "- testnet401-export-public-release" in workflow
    assert "testnet401_prior_release_run_id:" in workflow
    assert "testnet401_prior_release_commit:" in workflow
    assert "ssm-export-public-release" in workflow
    assert "temporary-testnet401-public-release.json" in workflow
    assert '--prior-release-run-id "$TESTNET401_PRIOR_RELEASE_RUN_ID"' in workflow
    assert '--prior-release-commit "$TESTNET401_PRIOR_RELEASE_COMMIT"' in workflow
    assert workflow.count("inputs.operation == 'testnet401-export-public-release'") == 1


def test_combined_one_host_nitro_capacity_leaves_bounded_parent_headroom():
    parent_vcpus = 16
    parent_memory_mib = 128 * 1024
    gateway_vcpus = 2 + 6
    gateway_memory_mib = (8 + 56) * 1024
    validator_vcpus = 2
    validator_memory_mib = 1024
    assert parent_vcpus - gateway_vcpus - validator_vcpus == 6
    assert parent_memory_mib - gateway_memory_mib - validator_memory_mib == 63 * 1024
