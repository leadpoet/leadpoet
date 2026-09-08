from pathlib import Path
import json
import subprocess
import sys
import time

import pytest

from gateway.research_lab.config import ResearchLabGatewayConfig
from leadpoet_canonical.lab_arena_rewards import (
    rewards_enabled_from_environment,
    signing_key_hash_from_environment,
)
from scripts import bootstrap_temporary_testnet_weights_host as bootstrap
from validator_tee.host.release_v2 import (
    build_local_validator_release_identity,
    build_validator_release,
)


EXPIRY_EPOCH = int(time.time()) + 3600


def _config():
    return {
        "run_id": "testnet401-native-20260908",
        "candidate_sha": "a" * 40,
        "expected_instance_id": "i-0123456789abcdef0",
        "expires_at_epoch": EXPIRY_EPOCH,
    }


def _profile_install_config(tmp_path):
    from leadpoet_canonical.attested_v2 import sha256_json
    from validator_tee.enclave.hotkey_authority_v2 import load_chain_signing_profile

    profile_path = (
        Path(__file__).resolve().parents[1]
        / "validator_tee/enclave/chain_signing_profile_test_v2.json"
    )
    profile = load_chain_signing_profile(profile_path)
    hotkey_path = tmp_path / "validator-hotkey-config.json"
    hotkey_path.write_text(json.dumps({
        "schema_version": "leadpoet.validator_hotkey_config.v2",
        "validator_hotkey": bootstrap.EXPECTED_VALIDATOR_HOTKEY,
        "hotkey_public_key": "4" * 64,
        "chain_signing_profile_hash": sha256_json(profile),
        "drand_library_path": "/app/validator_tee/enclave/libbittensor_drand_v2.so",
        "drand_library_sha256": "5" * 64,
    }), encoding="utf-8")
    return {
        "validator": {
            "chain_profile": str(profile_path),
            "hotkey_config": str(hotkey_path),
        }
    }, profile


def test_canonical_validator_profile_installer_is_exact_repeatable_and_default_readable(
    tmp_path, monkeypatch,
):
    from validator_tee.enclave import hotkey_authority_v2
    from validator_tee.host import publication_journal_v2

    config, profile = _profile_install_config(tmp_path)
    target = tmp_path / "app/validator_tee/enclave/chain_signing_profile_v2.json"
    first = bootstrap._install_canonical_validator_chain_profile(
        config, destination=target, privileged=False,
    )
    second = bootstrap._install_canonical_validator_chain_profile(
        config, destination=target, privileged=False,
    )
    assert first == second == profile
    assert target.stat().st_mode & 0o777 == 0o644

    original_defaults = hotkey_authority_v2.load_chain_signing_profile.__defaults__
    monkeypatch.setattr(
        hotkey_authority_v2.load_chain_signing_profile,
        "__defaults__",
        (target,),
    )
    assert publication_journal_v2.load_chain_signing_profile() == profile
    assert original_defaults != hotkey_authority_v2.load_chain_signing_profile.__defaults__


def test_canonical_validator_profile_installer_refuses_existing_difference(tmp_path):
    config, _profile = _profile_install_config(tmp_path)
    target = tmp_path / "app/validator_tee/enclave/chain_signing_profile_v2.json"
    target.parent.mkdir(parents=True)
    target.write_text("different-public-profile\n", encoding="ascii")
    before = target.read_bytes()
    with pytest.raises(
        bootstrap.TemporaryTestnetBootstrapError, match="installation failed"
    ):
        bootstrap._install_canonical_validator_chain_profile(
            config, destination=target, privileged=False,
        )
    assert target.read_bytes() == before


def test_canonical_validator_profile_installer_rejects_hotkey_profile_mismatch(tmp_path):
    config, _profile = _profile_install_config(tmp_path)
    hotkey_path = Path(config["validator"]["hotkey_config"])
    hotkey = json.loads(hotkey_path.read_text(encoding="utf-8"))
    hotkey["chain_signing_profile_hash"] = "sha256:" + "0" * 64
    hotkey_path.write_text(json.dumps(hotkey), encoding="utf-8")
    with pytest.raises(
        bootstrap.TemporaryTestnetBootstrapError, match="identity differs"
    ):
        bootstrap._install_canonical_validator_chain_profile(
            config,
            destination=tmp_path / "app/profile.json",
            privileged=False,
        )


@pytest.mark.parametrize("content,expected", [
    ('GIT_SSH_COMMAND=ssh -i /task/key -o IdentitiesOnly=yes\n'
     'SSH_CLIENT=192.0.2.1 12000 22\nQUOTED="two words"\n',
     {"GIT_SSH_COMMAND": "ssh -i /task/key -o IdentitiesOnly=yes",
      "SSH_CLIENT": "192.0.2.1 12000 22", "QUOTED": "two words"}),
    ('{"KEY":"two words","EMPTY":null}', {"KEY": "two words", "EMPTY": ""}),
])
def test_temporary_environment_uses_production_secret_formats(tmp_path, content, expected):
    path = tmp_path / "runtime.env"
    path.write_text(content)
    assert bootstrap._environment_file(path) == expected


@pytest.mark.parametrize("content", ["INVALID-KEY=hidden", "KEY=one\nKEY=two", "not an assignment"])
def test_temporary_environment_rejects_invalid_inputs_without_values(tmp_path, content):
    path = tmp_path / "runtime.env"
    path.write_text(content)
    with pytest.raises(bootstrap.TemporaryTestnetBootstrapError, match="malformed") as error:
        bootstrap._environment_file(path)
    assert content not in str(error.value)


def test_child_launchers_use_installed_python_not_captured_production_path(tmp_path):
    path = tmp_path / "runtime.env"
    path.write_text("PATH=/old-host/bin:/usr/bin\nAWS_ACCESS_KEY_ID=secret-canary\n")
    result = bootstrap._runtime_environment(
        path, overrides={}, repo_root=tmp_path, candidate_sha="a" * 40)
    assert result["PATH"].split(":", 1)[0] == str(Path(sys.executable).parent)
    assert "AWS_ACCESS_KEY_ID" not in result


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


@pytest.mark.parametrize("resume_existing", [False, True])
def test_fresh_epoch_authority_reuses_only_explicit_durable_resume(
    monkeypatch, resume_existing
):
    mapping_hash = bootstrap.EXPECTED_CUTOVER_MAPPING_HASH
    config = {
        "python_bin": sys.executable,
        "repo_root": "/runtime/repository",
        "runtime_root": "/run/leadpoet-testnet401",
        "candidate_sha": "a" * 40,
        "gateway": {
            "release_manifest": "/runtime/gateway-release.json",
            "release_lineage": "/runtime/gateway-lineage.json",
            "source_env_file": "/runtime/gateway.env",
        },
        "validator": {
            "cutover_manifest": "/runtime/cutover.json",
            "release_manifest": "/runtime/validator-release.json",
            "wallet_name": "validator",
            "wallet_hotkey": "default",
            "wallet_path": "/runtime/wallets",
        },
    }
    if resume_existing:
        config["resume_existing_epoch_authority"] = True
    calls = []

    class Runner:
        def run_json(self, stage, command, **_kwargs):
            calls.append((stage, command))
            if stage == "gateway_epoch_cutover_resume_preflight":
                return {
                    "status": "fresh_network_eligible",
                    "coordinator_receipt_exists": True,
                }
            if stage == "validator_epoch_candidate_preview":
                return {
                    "status": "validated_no_writes",
                    "candidate_payload_hash": "sha256:" + "1" * 64,
                }
            if stage == "validator_epoch_candidate_ingest":
                return {
                    "status": "durably_staged",
                    "candidate_authorization_hash": "sha256:" + "2" * 64,
                }
            if stage == "gateway_epoch_cutover_attestation":
                return {"status": "fresh_network_durable"}
            return {}

    monkeypatch.setattr(
        bootstrap,
        "_load_json",
        lambda *_args, **_kwargs: {"mapping_hash": mapping_hash},
    )
    monkeypatch.setattr(
        bootstrap,
        "_durable_epoch_authority_check",
        lambda _config: {"mapping_hash": mapping_hash, "netuid": 401},
    )
    result = bootstrap._bootstrap_fresh_epoch_authority(
        config,
        Runner(),
        gateway_env={},
        validator_env={},
    )

    stages = [stage for stage, _command in calls]
    if resume_existing:
        assert stages == [
            "gateway_epoch_cutover_resume_preflight",
            "gateway_epoch_cutover_attestation",
        ]
        assert result["candidate_payload_hash"] is None
    else:
        assert stages == [
            "validator_epoch_boundary_capture",
            "validator_epoch_candidate_preview",
            "validator_epoch_candidate_ingest",
            "gateway_epoch_cutover_attestation",
        ]
    attestation_command = calls[-1][1]
    assert attestation_command[
        attestation_command.index("--approved-release-lineage") + 1
    ] == config["gateway"]["release_lineage"]


def test_fresh_epoch_authority_resume_rejects_invalid_preflight(monkeypatch):
    config = {
        "python_bin": sys.executable,
        "repo_root": "/runtime/repository",
        "runtime_root": "/run/leadpoet-testnet401",
        "candidate_sha": "a" * 40,
        "resume_existing_epoch_authority": True,
        "gateway": {
            "release_manifest": "/runtime/gateway-release.json",
            "release_lineage": "/runtime/gateway-lineage.json",
        },
        "validator": {
            "cutover_manifest": "/runtime/cutover.json",
            "release_manifest": "/runtime/validator-release.json",
        },
    }

    class Runner:
        def run_json(self, *_args, **_kwargs):
            return {
                "status": "fresh_network_eligible",
                "coordinator_receipt_exists": "unknown",
            }

    monkeypatch.setattr(
        bootstrap,
        "_load_json",
        lambda *_args, **_kwargs: {
            "mapping_hash": bootstrap.EXPECTED_CUTOVER_MAPPING_HASH
        },
    )
    with pytest.raises(
        bootstrap.TemporaryTestnetBootstrapError,
        match="stored fresh testnet401 epoch authority is incomplete",
    ):
        bootstrap._bootstrap_fresh_epoch_authority(
            config,
            Runner(),
            gateway_env={},
            validator_env={},
        )


def test_validator_runtime_is_pinned_to_cid18_and_loopback_gateway():
    source = Path(bootstrap.__file__).read_text(encoding="utf-8")
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
    assert '"--chain-signing-profile"' in source
    assert '"--port",\n                "5004"' in source
    assert (
        bootstrap.SAFE_VALIDATOR_ENV["EXPECTED_BURN_TARGET_HOTKEY"]
        == bootstrap.EXPECTED_BURN_TARGET_HOTKEY
    )
    assert bootstrap.SAFE_VALIDATOR_ENV["LAB_ARENA_REWARDS_ENABLED"] == "true"


def test_gateway_runtime_is_pinned_to_testnet401_validator():
    assert bootstrap.SAFE_GATEWAY_ENV["ALLOWED_NETUIDS"] == "401"
    assert bootstrap.SAFE_GATEWAY_ENV["BITTENSOR_NETWORK"] == "test"
    assert bootstrap.SAFE_GATEWAY_ENV["BT_SUBTENSOR_NETWORK"] == "test"
    assert (
        bootstrap.SAFE_GATEWAY_ENV["BT_SUBTENSOR_CHAIN_ENDPOINT"]
        == bootstrap.CHAIN_ENDPOINT
    )
    assert bootstrap.SAFE_GATEWAY_ENV["EXPECTED_CHAIN"] == bootstrap.CHAIN_ENDPOINT
    assert (
        bootstrap.SAFE_GATEWAY_ENV["PRIMARY_VALIDATOR_HOTKEYS"]
        == bootstrap.EXPECTED_VALIDATOR_HOTKEY
    )
    source = Path(bootstrap.__file__).read_text(encoding="utf-8")
    assert "v(execution_config=b())" in source


def test_live_455_selects_the_exact_approved_testnet_profile():
    from leadpoet_canonical.attested_v2 import sha256_json
    from leadpoet_canonical.hotkey_authority_v2 import select_chain_signing_profile
    from validator_tee.enclave.hotkey_authority_v2 import load_chain_signing_profile

    profile = load_chain_signing_profile(
        Path(bootstrap.__file__).resolve().parents[1]
        / "validator_tee/enclave/chain_signing_profile_test_v2.json"
    )
    selected = select_chain_signing_profile(
        profile,
        runtime_version={"specVersion": 455, "transactionVersion": 1},
        genesis_hash=profile["genesis_hash"],
    )

    assert selected["spec_version"] == bootstrap.EXPECTED_PROFILE_SPEC_VERSION
    assert sha256_json(selected) == bootstrap.EXPECTED_PROFILE_HASH


def test_gateway_dynamic_pcr_builder_uses_only_task_owned_build_paths(monkeypatch):
    config = {
        "runtime_root": "/run/leadpoet-testnet401",
        "resume_existing_epoch_authority": True,
        "gateway": {"artifact_policy": "/unused/artifact-policy.json"},
        "validator": {"cutover_manifest": "/unused/cutover.json"},
    }
    policy = {
        "bucket_host": (
            "leadpoet-parity-493765492819-task.s3.us-east-1.amazonaws.com"
        )
    }
    cutover = {"mapping_hash": bootstrap.EXPECTED_CUTOVER_MAPPING_HASH}

    def load(path, _description):
        return policy if "artifact-policy" in str(path) else cutover

    monkeypatch.setattr(bootstrap, "_load_json", load)
    environment = bootstrap._gateway_runtime_overrides(config)

    assert environment["NITRO_CLI_ARTIFACTS"] == (
        "/run/leadpoet-testnet401/nitro-cli-artifacts"
    )
    assert environment["NITRO_CLI_BLOBS"] == "/usr/share/nitro_enclaves/blobs"
    assert environment["VALIDATOR_DRAND_CARGO_CACHE_DIR"] == (
        "/run/leadpoet-testnet401/drand-cargo-cache"
    )
    assert environment["VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT"] == (
        "/run/leadpoet-testnet401/offline-artifacts/validator-runtime"
    )
    assert environment["PCR0_BUILD_DIR"] == (
        "/run/leadpoet-testnet401/pcr0-builder"
    )
    assert environment["PCR0_STARTUP_HISTORICAL_WARM_ENABLED"] == "false"
    assert environment[
        "LEADPOET_TEMPORARY_TESTNET401_LOCAL_RELEASE_CHANNELS"
    ] == "true"


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


def test_testnet_gateway_overrides_enable_only_required_weight_api_gates(monkeypatch):
    from gateway.research_lab import config as gateway_config

    monkeypatch.setattr(gateway_config.os, "environ", dict(bootstrap.SAFE_GATEWAY_ENV))
    config = ResearchLabGatewayConfig.from_env()

    assert config.api_enabled is True
    assert config.reports_enabled is True
    assert config.shadow_bundles_enabled is True
    assert config.weight_mutation_enabled is True
    assert rewards_enabled_from_environment(bootstrap.SAFE_GATEWAY_ENV) is True
    assert (
        signing_key_hash_from_environment(bootstrap.SAFE_GATEWAY_ENV)
        == bootstrap.EXPECTED_ARENA_SIGNING_KEY_HASH
    )


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


def test_static_inputs_reads_hash_from_real_local_validator_release(
    monkeypatch, tmp_path
):
    digest = "sha256:" + "1" * 64
    release = build_validator_release(
        commit_sha="a" * 40,
        pcr0="2" * 96,
        app_manifest_hash=digest,
        dependency_lock_hash=digest,
        normalized_image_hash=digest,
        eif_hash=digest,
        dockerfile_hash=digest,
        base_dockerfile_hash=digest,
    )
    validator_manifest = build_local_validator_release_identity(release)
    profile = {
        "network": bootstrap.NETWORK,
        "chain_endpoint": bootstrap.CHAIN_ENDPOINT,
        "genesis_hash": "3" * 64,
    }
    from leadpoet_canonical.attested_v2 import sha256_json
    from gateway.tee import release_lineage_v2, release_manifest_v2
    from Leadpoet.utils import subnet_epoch
    from validator_tee.enclave import hotkey_authority_v2
    from validator_tee.host import hotkey_bootstrap_v2

    class Cutover:
        netuid = bootstrap.NETUID
        cutover_block = bootstrap.EXPECTED_CUTOVER_BLOCK
        first_settlement_epoch_id = bootstrap.EXPECTED_FIRST_SETTLEMENT_EPOCH
        mapping_hash = bootstrap.EXPECTED_CUTOVER_MAPPING_HASH
        network_genesis_hash = "0x" + profile["genesis_hash"]

        @classmethod
        def from_mapping(cls, _value):
            return cls()

    documents = {
        "gateway release manifest": {},
        "validator release manifest": validator_manifest,
        "gateway release lineage": {},
        "validator hotkey config": {},
        "validator hotkey envelope": {},
        "testnet cutover manifest": {},
    }
    monkeypatch.setattr(bootstrap, "_validate_candidate_checkout", lambda *_: None)
    monkeypatch.setattr(bootstrap, "_private_regular_file", lambda *_: None)
    monkeypatch.setattr(bootstrap, "_regular_file", lambda *_: None)
    monkeypatch.setattr(bootstrap, "_load_json", lambda _path, field: documents[field])
    monkeypatch.setattr(
        release_manifest_v2, "validate_release_manifest",
        lambda _value: {"commit_sha": "a" * 40, "release_hash": digest},
    )
    monkeypatch.setattr(
        release_lineage_v2, "validate_compact_release_lineage_v2", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        hotkey_authority_v2, "load_chain_signing_profile", lambda _path: profile
    )
    monkeypatch.setattr(
        hotkey_authority_v2, "validate_hotkey_authority_configuration",
        lambda _value: {
            "validator_hotkey": bootstrap.EXPECTED_VALIDATOR_HOTKEY,
            "hotkey_public_key": "4" * 64,
            "chain_signing_profile_hash": sha256_json(profile),
        },
    )
    monkeypatch.setattr(
        hotkey_bootstrap_v2, "validate_hotkey_envelope",
        lambda _value: {
            "validator_hotkey": bootstrap.EXPECTED_VALIDATOR_HOTKEY,
            "hotkey_public_key": "4" * 64,
        },
    )
    monkeypatch.setattr(subnet_epoch, "SubnetEpochCutover", Cutover)
    config = {
        "repo_root": str(tmp_path), "python_bin": sys.executable,
        "candidate_sha": "a" * 40,
        "gateway": {name: "/unused" for name in (
            "source_env_file", "release_manifest", "release_lineage", "eif_root",
            "artifact_policy", "protected_workflow_manifest",
        )},
        "validator": {name: "/unused" for name in (
            "source_env_file", "release_manifest", "eif_path", "hotkey_config",
            "hotkey_envelope", "chain_profile", "cutover_manifest",
        )},
    }

    result = bootstrap.validate_static_inputs(config)

    assert "release_hash" not in validator_manifest
    assert result["validator_release_hash"] == release["release_hash"]


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
