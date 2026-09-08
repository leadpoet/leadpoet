from pathlib import Path
import subprocess
import time

import pytest

from scripts import bootstrap_temporary_testnet_weights_host as bootstrap


EXPIRY_EPOCH = int(time.time()) + 3600


def _config():
    return {
        "run_id": "testnet401-native-20260908",
        "candidate_sha": "a" * 40,
        "expected_instance_id": "i-0123456789abcdef0",
        "expires_at_epoch": EXPIRY_EPOCH,
    }


class STS:
    def get_caller_identity(self):
        return {"Account": bootstrap.EXPECTED_AWS_ACCOUNT}


class EC2:
    def describe_instances(self, *, InstanceIds):
        assert InstanceIds == ["i-0123456789abcdef0"]
        return {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": InstanceIds[0],
                            "InstanceType": bootstrap.EXPECTED_INSTANCE_TYPE,
                            "ImageId": bootstrap.EXPECTED_AMI,
                            "SubnetId": bootstrap.EXPECTED_SUBNET,
                            "VpcId": bootstrap.EXPECTED_VPC,
                            "State": {"Name": "running"},
                            "EnclaveOptions": {"Enabled": True},
                            "IamInstanceProfile": {
                                "Arn": (
                                    "arn:aws:iam::493765492819:instance-profile/"
                                    + bootstrap.EXPECTED_INSTANCE_PROFILE
                                )
                            },
                            "PrivateIpAddress": "10.0.0.8",
                            "SecurityGroups": [{"GroupId": "sg-12345678"}],
                            "Tags": [
                                {
                                    "Key": "Name",
                                    "Value": (
                                        "leadpoet-parity-testnet401-native-20260908-"
                                        "testnet401-exp-"
                                        f"{EXPIRY_EPOCH:010d}"
                                    ),
                                },
                                {
                                    "Key": "leadpoet:parity-run",
                                    "Value": "testnet401-native-20260908",
                                },
                                {
                                    "Key": "leadpoet:candidate-sha",
                                    "Value": "a" * 40,
                                },
                                {"Key": "leadpoet:ephemeral", "Value": "true"},
                            ],
                            "BlockDeviceMappings": [
                                {
                                    "DeviceName": "/dev/xvda",
                                    "Ebs": {
                                        "VolumeId": "vol-12345678",
                                        "DeleteOnTermination": True,
                                    },
                                }
                            ],
                        }
                    ]
                }
            ]
        }

    def describe_volumes(self, *, VolumeIds):
        assert VolumeIds == ["vol-12345678"]
        return {"Volumes": [{"Encrypted": True, "Size": 512}]}

    def describe_security_groups(self, *, GroupIds):
        assert GroupIds == ["sg-12345678"]
        return {"SecurityGroups": [{"GroupId": GroupIds[0], "IpPermissions": []}]}


def test_host_authority_requires_exact_task_owned_isolated_instance():
    result = bootstrap.verify_host_authority(
        _config(),
        instance_id="i-0123456789abcdef0",
        sts_client=STS(),
        ec2_client=EC2(),
    )
    assert result["instance_type"] == "r7i.4xlarge"
    assert result["security_group_ingress_rule_count"] == 0
    assert result["volume_encrypted"] is True


def test_host_authority_rejects_inbound_security_group():
    class UnsafeEC2(EC2):
        def describe_security_groups(self, *, GroupIds):
            return {
                "SecurityGroups": [
                    {"GroupId": GroupIds[0], "IpPermissions": [{"IpProtocol": "tcp"}]}
                ]
            }

    with pytest.raises(
        bootstrap.TemporaryTestnetBootstrapError, match="inbound"
    ):
        bootstrap.verify_host_authority(
            _config(),
            instance_id="i-0123456789abcdef0",
            sts_client=STS(),
            ec2_client=UnsafeEC2(),
        )


def test_combined_allocator_preserves_gateway_and_validator_capacity():
    assert bootstrap.ALLOCATOR_CPUS == (
        bootstrap.GATEWAY_CPUS + bootstrap.VALIDATOR_CPUS
    )
    assert bootstrap.ALLOCATOR_MEMORY_MIB == (
        bootstrap.GATEWAY_MEMORY_MIB + bootstrap.VALIDATOR_MEMORY_MIB
    )
    assert bootstrap.EXPECTED_PARENT_CPUS - bootstrap.ALLOCATOR_CPUS == 6


def test_native_launch_order_keeps_validator_after_full_gateway_launch():
    stages = bootstrap.launch_sequence_names()
    assert stages.index("combined_allocator") < stages.index("gateway_enclaves")
    assert stages.index("gateway_enclaves") < stages.index("validator_enclave_cid18")
    assert stages.index("validator_runtime_bootstrap_cid18") < stages.index(
        "validator_hotkey_recipient_cid18"
    )
    assert stages.index("validator_hotkey_recipient_cid18") < stages.index(
        "validator_epoch_boundary_capture_cid18"
    )
    assert stages.index("validator_epoch_boundary_capture_cid18") < stages.index(
        "gateway_epoch_cutover_attestation"
    )
    assert stages.index("gateway_epoch_cutover_attestation") < stages.index(
        "gateway_http_readiness"
    )
    assert stages.index("gateway_http_readiness") < stages.index(
        "validator_application_cid18"
    )


def test_fresh_epoch_authority_is_created_after_measured_boot_before_apps():
    source = Path(bootstrap.__file__).read_text(encoding="utf-8")
    stages = bootstrap.launch_sequence_names()
    assert stages.index("validator_hotkey_recipient_cid18") < stages.index(
        "validator_epoch_candidate_ingest"
    )
    assert stages.index("durable_epoch_authority_readback") < stages.index(
        "gateway_application"
    )
    assert '"--fresh-testnet401-network"' in source
    assert "validate_stateful_cutover_authority" in source


def test_validator_runtime_is_pinned_to_cid18_and_loopback_gateway():
    assert bootstrap.SAFE_VALIDATOR_ENV["ENCLAVE_CID"] == "18"
    assert bootstrap.SAFE_VALIDATOR_ENV["GATEWAY_URL"] == "http://127.0.0.1:8000"
    assert (
        bootstrap.SAFE_VALIDATOR_ENV["VALIDATOR_V2_GATEWAY_URL"]
        == "http://127.0.0.1:8000"
    )
    assert bootstrap.SAFE_VALIDATOR_ENV["VALIDATOR_NETUID"] == "401"
    assert bootstrap.SAFE_VALIDATOR_ENV["VALIDATOR_SUBTENSOR_NETWORK"] == "test"
    assert bootstrap.SAFE_VALIDATOR_ENV["RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED"] == "true"
    assert bootstrap.SAFE_VALIDATOR_ENV["BURN_TARGET_UID"] == "0"
    assert (
        bootstrap.SAFE_VALIDATOR_ENV["EXPECTED_BURN_TARGET_HOTKEY"]
        == bootstrap.EXPECTED_BURN_TARGET_HOTKEY
    )
    assert bootstrap.SAFE_VALIDATOR_ENV["LAB_ARENA_REWARDS_ENABLED"] == "true"


def test_gateway_runtime_is_pinned_to_testnet401_validator():
    assert bootstrap.SAFE_GATEWAY_ENV["ALLOWED_NETUIDS"] == "401"
    assert bootstrap.SAFE_GATEWAY_ENV["BITTENSOR_NETWORK"] == "test"
    assert bootstrap.SAFE_GATEWAY_ENV["EXPECTED_CHAIN"] == bootstrap.CHAIN_ENDPOINT
    assert (
        bootstrap.SAFE_GATEWAY_ENV["PRIMARY_VALIDATOR_HOTKEYS"]
        == bootstrap.EXPECTED_VALIDATOR_HOTKEY
    )


def test_testnet_gateway_flags_do_not_modify_global_intake_control():
    assert bootstrap.SAFE_GATEWAY_ENV["DISABLE_BACKGROUND_TASKS"] == "true"
    assert bootstrap.SAFE_GATEWAY_ENV["ENABLE_FULFILLMENT"] == "false"
    assert bootstrap.SAFE_GATEWAY_ENV["LAB_ARENA_MODE"] == "off"
    assert (
        bootstrap.SAFE_GATEWAY_ENV["RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED"]
        == "false"
    )
    assert "RESEARCH_LAB_SOURCE_ADD_PAUSED" not in bootstrap.SAFE_GATEWAY_ENV
    assert "RESEARCH_LAB_AUTORESEARCH_ENABLED" not in bootstrap.SAFE_GATEWAY_ENV


def test_gateway_bootstrap_uses_only_expected_ciphertext_envelopes():
    config = {
        "python_bin": "/usr/bin/python3",
        "gateway": {
            "release_manifest": "/runtime/gateway-release.json",
            "release_lineage": "/runtime/release-lineage.json",
            "protected_workflow_manifest": "/runtime/protected.json",
            "artifact_policy": "/runtime/artifact-policy.json",
            "config_dir": "/runtime/ciphertext",
        },
    }
    command = bootstrap._gateway_bootstrap_command(config)
    envelope_args = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--credential-envelope"
    ]
    assert [Path(value).name for value in envelope_args] == list(
        bootstrap.GATEWAY_ENVELOPE_NAMES
    )
    assert all("plaintext" not in value.lower() for value in command)


def test_candidate_checkout_allows_only_exact_native_build_outputs(tmp_path):
    repository = tmp_path / "candidate"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repository, check=True
    )
    tracked = repository / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "candidate"], cwd=repository, check=True)
    candidate = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    for relative in bootstrap.ALLOWED_NATIVE_BUILD_OUTPUTS:
        output = repository / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("measured build output\n", encoding="utf-8")

    bootstrap._validate_candidate_checkout(repository, candidate)

    (repository / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(
        bootstrap.TemporaryTestnetBootstrapError, match="unexpected untracked"
    ):
        bootstrap._validate_candidate_checkout(repository, candidate)
    (repository / "unexpected.txt").unlink()
    tracked.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(
        bootstrap.TemporaryTestnetBootstrapError, match="tracked source"
    ):
        bootstrap._validate_candidate_checkout(repository, candidate)


def test_cleanup_never_uses_global_enclave_termination():
    source = Path(bootstrap.__file__).read_text(encoding="utf-8")
    cleanup = source[source.index("def _stop_owned_enclaves") :]
    assert "terminate-enclave --all" not in cleanup
    assert '{16, 17, VALIDATOR_CID}' in cleanup
    assert 'name != "leadpoet-testnet401-validator"' in cleanup


def test_chain_proof_requires_joined_native_finalization_and_new_chain_state():
    chain = {
        "finalized_block": 8_000_100,
        "validator_last_update": 8_000_000,
        "revealed_weights": [[0, 65535], [11, 21845]],
    }
    authority = {
        "status": "available",
        "epoch_id": 22_054,
        "authority_stage": "finalized",
        "authority_hash": "sha256:" + "1" * 64,
        "bundle_hash": "sha256:" + "2" * 64,
        "weights_hash": "3" * 64,
        "uids": [0, 11],
        "weights_u16": [65535, 21845],
        "weight_submission_event_hash": "sha256:" + "4" * 64,
        "publication_receipt_hash": "sha256:" + "5" * 64,
        "weight_finalization_event_hash": "sha256:" + "6" * 64,
        "extrinsic_hash": "0x" + "7" * 64,
        "finalized_block": 8_000_000,
        "finalized_block_hash": "0x" + "8" * 64,
    }

    proof = bootstrap._automatic_chain_proof(
        chain=chain, weight_authorities=[authority]
    )

    assert proof["status"] == "candidate_match"
    assert proof["independent_verification_pending"] is True
    assert proof["proof_basis"].startswith("native_finalized_authority_joined")
    assert proof["revealed_weights"] == [(0, 65535), (11, 21845)]

    authority["authority_stage"] = "published"
    assert (
        bootstrap._automatic_chain_proof(
            chain=chain, weight_authorities=[authority]
        )["status"]
        == "pending"
    )
    authority["authority_stage"] = "finalized"
    authority["weights_u16"] = [65535, 1000]
    chain["revealed_weights"] = [[0, 65535], [11, 1000]]
    assert (
        bootstrap._automatic_chain_proof(
            chain=chain, weight_authorities=[authority]
        )["status"]
        == "pending"
    )
