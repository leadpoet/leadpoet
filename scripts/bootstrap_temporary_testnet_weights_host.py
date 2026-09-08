"""Bootstrap one task-owned native testnet401 weight-signing host.

This helper is intentionally temporary and narrow.  It reuses the measured
gateway and validator release artifacts and the normal authoritative V2
runtime modules.  It does not build releases, alter database authority, create
weights, or call a chain extrinsic directly.

The destructive launch path is available only on the exact EC2 instance in a
configuration document.  The instance must have the task tags, no inbound
security-group rules, Nitro Enclaves enabled, and no existing Leadpoet runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
from urllib import request


SCHEMA_VERSION = "leadpoet.temporary_testnet401_native_bootstrap.v1"
RECEIPT_SCHEMA_VERSION = (
    "leadpoet.temporary_testnet401_native_bootstrap_receipt.v1"
)
PROCESS_STATE_SCHEMA_VERSION = (
    "leadpoet.temporary_testnet401_native_process_state.v1"
)
NETWORK = "test"
NETUID = 401
CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai:443"
VALIDATOR_CID = 18
ALLOCATOR_CPUS = 10
ALLOCATOR_MEMORY_MIB = 66_560
GATEWAY_CPUS = 8
GATEWAY_MEMORY_MIB = 65_536
VALIDATOR_CPUS = 2
VALIDATOR_MEMORY_MIB = 1_024
EXPECTED_PARENT_CPUS = 16
MINIMUM_PARENT_MEMORY_MIB = 125_000
EXPECTED_INSTANCE_TYPE = "r7i.4xlarge"
EXPECTED_AWS_ACCOUNT = "493765492819"
EXPECTED_AWS_REGION = "us-east-1"
EXPECTED_AMI = "ami-0cae6d6fe6048ca2c"
EXPECTED_SUBNET = "subnet-025170c1eff61494d"
EXPECTED_VPC = "vpc-0c975a643bc1e0e79"
EXPECTED_INSTANCE_PROFILE = "leadpoet-production-parity-runner"
EXPECTED_PROFILE_HASH = (
    "sha256:a2db2db86ffb10bbf41dd6923e1310726031bc4183841e07e6d2da50e6e58677"
)
EXPECTED_CUTOVER_BLOCK = 7_955_391
EXPECTED_FIRST_SETTLEMENT_EPOCH = 22_042
EXPECTED_CUTOVER_MAPPING_HASH = (
    "sha256:4b3941c091d3a29daf9ea863bb6cb587bad7dfd9426cf66a56bcf7de04ce4328"
)
EXPECTED_VALIDATOR_HOTKEY = (
    "5CJyMxw6YJJvLhPf58gSpMB7mvSKSCMx9RXhXJum6cNfqMEz"
)
EXPECTED_BURN_TARGET_UID = 0
EXPECTED_BURN_TARGET_HOTKEY = (
    "5E6zy3Dt8BrwsSKbocSF2uxjpMAbr4EksVzxgPJiXh8jq4vf"
)
EXPECTED_REWARD_ROUND_ID = "arena-2026-09-08-promote4"
EXPECTED_REWARD_EFFECTIVE_EPOCH = 22_051
EXPECTED_REWARD_BASIS_HASH = (
    "sha256:179de18778a81f8ec0205660c30395a82e489d11b7eb91ff07c06561ef275ad1"
)
EXPECTED_REWARD_KING_HOTKEY = (
    "5FEtvBzsh5Zc8nDyq4Jb2nZ7o6ZD2homYsKjbZtFj5tybqth"
)
EXPECTED_ARENA_SIGNING_KEY_HASH = (
    "sha256:fb0a422d437700f468beda94b4d3e05bb22dbaa6141f0e6c5f1dac9e7257d99a"
)
BEFORE_TESTNET401_LAST_UPDATE = 7_431_466
EXPECTED_RUNTIME_ROOT = Path("/run/leadpoet-testnet401")
KNOWN_FINNEY_ADDRESSES = frozenset({"52.91.135.79", "100.59.201.156"})
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{7,79}$")
INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
SAFE_GATEWAY_ENV = {
    "ALLOWED_NETUIDS": str(NETUID),
    "AWS_DEFAULT_REGION": EXPECTED_AWS_REGION,
    "AWS_REGION": EXPECTED_AWS_REGION,
    "BITTENSOR_NETWORK": NETWORK,
    "BITTENSOR_NETUID": str(NETUID),
    "DISABLE_BACKGROUND_TASKS": "true",
    "ENABLE_FULFILLMENT": "false",
    "EXPECTED_CHAIN": CHAIN_ENDPOINT,
    "LAB_ARENA_MODE": "off",
    "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true",
    "PRIMARY_VALIDATOR_HOTKEYS": EXPECTED_VALIDATOR_HOTKEY,
    "RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED": "false",
    "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": "true",
    "SUBTENSOR_NETWORK": NETWORK,
}
SAFE_VALIDATOR_ENV = {
    "AWS_DEFAULT_REGION": EXPECTED_AWS_REGION,
    "AWS_REGION": EXPECTED_AWS_REGION,
    "BITTENSOR_NETWORK": NETWORK,
    "BITTENSOR_NETUID": str(NETUID),
    "BURN_TARGET_UID": str(EXPECTED_BURN_TARGET_UID),
    "ENABLE_FULFILLMENT": "false",
    "ENCLAVE_CID": str(VALIDATOR_CID),
    "EXPECTED_BURN_TARGET_HOTKEY": EXPECTED_BURN_TARGET_HOTKEY,
    "EXPECTED_CHAIN": CHAIN_ENDPOINT,
    "GATEWAY_URL": "http://127.0.0.1:8000",
    "LAB_ARENA_MODE": "off",
    "LAB_ARENA_REWARDS_ENABLED": "true",
    "LAB_ARENA_SIGNING_PUBLIC_KEY_HASH": EXPECTED_ARENA_SIGNING_KEY_HASH,
    "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true",
    "LEADPOET_WRAPPER_ACTIVE": "1",
    "RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED": "true",
    "RESEARCH_LAB_VALIDATOR_FETCH_ENABLED": "true",
    "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": "true",
    "SUBTENSOR_NETWORK": NETWORK,
    "VALIDATOR_NETUID": str(NETUID),
    "VALIDATOR_SUBTENSOR_NETWORK": NETWORK,
    "VALIDATOR_V2_GATEWAY_URL": "http://127.0.0.1:8000",
    "VALIDATOR_WEIGHT_PROTOCOL": "authoritative_v2",
}
STATIC_AWS_CREDENTIAL_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_PROFILE",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SECURITY_TOKEN",
        "AWS_SESSION_TOKEN",
    }
)
OWNED_PROCESS_MODULES = (
    "gateway.utils.tee_egress_forwarder",
    "gateway.utils.tee_inter_enclave_relay",
    "gateway.main",
    "validator_tee.host.chain_relay_v2",
    "scripts/run_temporary_testnet401_weight_only_validator.py",
)
GATEWAY_ENVELOPE_NAMES = (
    "artifact_master_key.json",
    "openrouter.json",
    "exa.json",
    "scrapingdog.json",
    "deepline.json",
    "supabase_service_role.json",
    "truelist.json",
)
ALLOWED_NATIVE_BUILD_OUTPUTS = frozenset(
    {
        ".validator-base.dockerfile.sha256",
        "validator_tee/enclave_build_output.txt",
        "validator_tee/validator-enclave.eif",
        "validator_tee/validator-v2-release.json",
    }
)


class TemporaryTestnetBootstrapError(RuntimeError):
    """The temporary native testnet host cannot be used safely."""


def _object(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TemporaryTestnetBootstrapError(f"{field} must be an object")
    return dict(value)


def _absolute_path(value: Any, field: str) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute() or path == Path("/") or ".." in path.parts:
        raise TemporaryTestnetBootstrapError(f"{field} must be a safe absolute path")
    return path


def _private_regular_file(path: Path, field: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TemporaryTestnetBootstrapError(f"{field} is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_mode & 0o077
    ):
        raise TemporaryTestnetBootstrapError(
            f"{field} must be a private regular file"
        )


def _regular_file(path: Path, field: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TemporaryTestnetBootstrapError(f"{field} is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise TemporaryTestnetBootstrapError(f"{field} must be a regular file")


def load_config(path: Path) -> Dict[str, Any]:
    _private_regular_file(path, "bootstrap config")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TemporaryTestnetBootstrapError("bootstrap config is invalid") from exc
    config = _object(value, "bootstrap config")
    expected_fields = {
        "schema_version",
        "run_id",
        "candidate_sha",
        "expected_instance_id",
        "expires_at_epoch",
        "runtime_root",
        "repo_root",
        "python_bin",
        "early_isolation_marker",
        "gateway",
        "validator",
        "aws",
    }
    if set(config) != expected_fields or config.get("schema_version") != SCHEMA_VERSION:
        raise TemporaryTestnetBootstrapError("bootstrap config fields are invalid")
    run_id = str(config.get("run_id") or "")
    candidate_sha = str(config.get("candidate_sha") or "")
    instance_id = str(config.get("expected_instance_id") or "")
    try:
        expires_at_epoch = int(config.get("expires_at_epoch"))
    except (TypeError, ValueError) as exc:
        raise TemporaryTestnetBootstrapError("expiry epoch is invalid") from exc
    if not RUN_ID_RE.fullmatch(run_id):
        raise TemporaryTestnetBootstrapError("run id is invalid")
    if not SHA_RE.fullmatch(candidate_sha):
        raise TemporaryTestnetBootstrapError("candidate SHA is invalid")
    if not INSTANCE_ID_RE.fullmatch(instance_id):
        raise TemporaryTestnetBootstrapError("expected instance id is invalid")
    if len(str(expires_at_epoch)) != 10:
        raise TemporaryTestnetBootstrapError("expiry epoch is invalid")
    runtime_root = _absolute_path(config["runtime_root"], "runtime root")
    if runtime_root != EXPECTED_RUNTIME_ROOT:
        raise TemporaryTestnetBootstrapError("runtime root is not task-owned")
    repo_root = _absolute_path(config["repo_root"], "repository root")
    python_bin = _absolute_path(config["python_bin"], "Python binary")
    marker = _absolute_path(config["early_isolation_marker"], "isolation marker")
    if not marker.is_relative_to(runtime_root):
        raise TemporaryTestnetBootstrapError("isolation marker is outside runtime root")

    aws = _object(config["aws"], "aws")
    if aws != {
        "account_id": EXPECTED_AWS_ACCOUNT,
        "region": EXPECTED_AWS_REGION,
        "ami": EXPECTED_AMI,
        "subnet_id": EXPECTED_SUBNET,
        "vpc_id": EXPECTED_VPC,
        "instance_type": EXPECTED_INSTANCE_TYPE,
        "instance_profile": EXPECTED_INSTANCE_PROFILE,
    }:
        raise TemporaryTestnetBootstrapError("AWS host authority differs")

    gateway = _object(config["gateway"], "gateway")
    gateway_fields = {
        "source_env_file",
        "kms_key_id",
        "release_manifest",
        "release_lineage",
        "eif_root",
        "config_dir",
        "artifact_policy",
        "protected_workflow_manifest",
    }
    if set(gateway) != gateway_fields or not str(gateway.get("kms_key_id") or ""):
        raise TemporaryTestnetBootstrapError("gateway config fields are invalid")
    validator = _object(config["validator"], "validator")
    validator_fields = {
        "source_env_file",
        "release_manifest",
        "eif_path",
        "hotkey_config",
        "hotkey_envelope",
        "chain_profile",
        "cutover_manifest",
        "wallet_path",
        "wallet_name",
        "wallet_hotkey",
        "enclave_cid",
        "expected_hotkey",
        "expected_selected_profile_hash",
    }
    if set(validator) != validator_fields:
        raise TemporaryTestnetBootstrapError("validator config fields are invalid")
    if (
        int(validator.get("enclave_cid") or 0) != VALIDATOR_CID
        or validator.get("expected_hotkey") != EXPECTED_VALIDATOR_HOTKEY
        or validator.get("expected_selected_profile_hash") != EXPECTED_PROFILE_HASH
        or not str(validator.get("wallet_name") or "")
        or not str(validator.get("wallet_hotkey") or "")
    ):
        raise TemporaryTestnetBootstrapError("validator identity differs")
    for name in (
        "source_env_file",
        "release_manifest",
        "eif_root",
        "config_dir",
        "artifact_policy",
        "protected_workflow_manifest",
    ):
        gateway[name] = str(_absolute_path(gateway[name], f"gateway {name}"))
    for name in (
        "source_env_file",
        "release_manifest",
        "eif_path",
        "hotkey_config",
        "hotkey_envelope",
        "chain_profile",
        "cutover_manifest",
        "wallet_path",
    ):
        validator[name] = str(_absolute_path(validator[name], f"validator {name}"))
    return {
        **config,
        "runtime_root": str(runtime_root),
        "repo_root": str(repo_root),
        "python_bin": str(python_bin),
        "early_isolation_marker": str(marker),
        "expires_at_epoch": expires_at_epoch,
        "gateway": gateway,
        "validator": validator,
        "aws": aws,
    }


def _load_json(path: Path, field: str) -> Dict[str, Any]:
    _regular_file(path, field)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TemporaryTestnetBootstrapError(f"{field} is invalid") from exc
    return _object(value, field)


def _validate_candidate_checkout(repo_root: Path, candidate_sha: str) -> None:
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != candidate_sha:
        raise TemporaryTestnetBootstrapError("checkout differs from candidate SHA")
    tracked_dirty = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--quiet", "--no-ext-diff", "--"],
        check=False,
    ).returncode
    staged_dirty = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "--cached",
            "--quiet",
            "--no-ext-diff",
            "--",
        ],
        check=False,
    ).returncode
    if tracked_dirty != 0 or staged_dirty != 0:
        raise TemporaryTestnetBootstrapError("candidate tracked source is not pristine")
    untracked = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    unexpected_untracked = sorted(set(untracked) - ALLOWED_NATIVE_BUILD_OUTPUTS)
    if unexpected_untracked:
        raise TemporaryTestnetBootstrapError(
            "candidate checkout has unexpected untracked files"
        )


def validate_static_inputs(config: Mapping[str, Any]) -> Dict[str, Any]:
    repo_root = Path(config["repo_root"])
    python_bin = Path(config["python_bin"])
    if not repo_root.is_dir() or repo_root.is_symlink():
        raise TemporaryTestnetBootstrapError("repository root is unavailable")
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise TemporaryTestnetBootstrapError("Python binary is unavailable")
    _validate_candidate_checkout(repo_root, config["candidate_sha"])

    gateway = config["gateway"]
    validator = config["validator"]
    for path, field in (
        (Path(gateway["source_env_file"]), "gateway source env"),
        (Path(validator["source_env_file"]), "validator source env"),
        (Path(validator["hotkey_config"]), "validator hotkey config"),
        (Path(validator["hotkey_envelope"]), "validator hotkey envelope"),
    ):
        _private_regular_file(path, field)
    for name in (
        "release_manifest",
        "release_lineage",
        "artifact_policy",
        "protected_workflow_manifest",
    ):
        _regular_file(Path(gateway[name]), f"gateway {name}")
    for name in ("release_manifest", "eif_path", "chain_profile", "cutover_manifest"):
        _regular_file(Path(validator[name]), f"validator {name}")
    eif_root = Path(gateway["eif_root"])
    for role in ("gateway_coordinator", "gateway_scoring"):
        _regular_file(eif_root / f"tee-enclave-{role}.eif", f"gateway {role} EIF")

    sys.path.insert(0, str(repo_root))
    try:
        from gateway.tee.release_manifest_v2 import validate_release_manifest
        from gateway.tee.release_lineage_v2 import validate_compact_release_lineage_v2
        from Leadpoet.utils.subnet_epoch import SubnetEpochCutover
        from leadpoet_canonical.attested_v2 import sha256_json
        from validator_tee.enclave.hotkey_authority_v2 import (
            load_chain_signing_profile,
            validate_hotkey_authority_configuration,
        )
        from validator_tee.host.hotkey_bootstrap_v2 import validate_hotkey_envelope
        from validator_tee.host.release_v2 import validate_validator_release_manifest
    finally:
        sys.path.pop(0)

    gateway_release = validate_release_manifest(
        _load_json(Path(gateway["release_manifest"]), "gateway release manifest")
    )
    validator_release = validate_validator_release_manifest(
        _load_json(Path(validator["release_manifest"]), "validator release manifest")
    )
    candidate_sha = config["candidate_sha"]
    if (
        gateway_release.get("commit_sha") != candidate_sha
        or validator_release.get("release", {}).get("commit_sha") != candidate_sha
    ):
        raise TemporaryTestnetBootstrapError("release commit differs from candidate")
    lineage = _load_json(Path(gateway["release_lineage"]), "gateway release lineage")
    validate_compact_release_lineage_v2(
        lineage,
        expected_current_commit=candidate_sha,
        expected_current_gateway_release_hash=str(gateway_release["release_hash"]),
    )
    profile = load_chain_signing_profile(Path(validator["chain_profile"]))
    if profile.get("network") != NETWORK or profile.get("chain_endpoint") != CHAIN_ENDPOINT:
        raise TemporaryTestnetBootstrapError("validator chain profile is not testnet")
    hotkey_config = validate_hotkey_authority_configuration(
        _load_json(Path(validator["hotkey_config"]), "validator hotkey config")
    )
    envelope = validate_hotkey_envelope(
        _load_json(Path(validator["hotkey_envelope"]), "validator hotkey envelope")
    )
    if (
        hotkey_config["validator_hotkey"] != EXPECTED_VALIDATOR_HOTKEY
        or envelope["validator_hotkey"] != EXPECTED_VALIDATOR_HOTKEY
        or envelope["hotkey_public_key"] != hotkey_config["hotkey_public_key"]
        or hotkey_config["chain_signing_profile_hash"] != sha256_json(profile)
    ):
        raise TemporaryTestnetBootstrapError("hotkey assets differ from test profile")
    cutover = _load_json(Path(validator["cutover_manifest"]), "testnet cutover manifest")
    canonical_cutover = SubnetEpochCutover.from_mapping(cutover)
    if (
        canonical_cutover.netuid != NETUID
        or canonical_cutover.cutover_block != EXPECTED_CUTOVER_BLOCK
        or canonical_cutover.first_settlement_epoch_id
        != EXPECTED_FIRST_SETTLEMENT_EPOCH
        or canonical_cutover.mapping_hash != EXPECTED_CUTOVER_MAPPING_HASH
    ):
        raise TemporaryTestnetBootstrapError("cutover manifest is not testnet401")
    genesis = canonical_cutover.network_genesis_hash.lower().removeprefix("0x")
    if genesis != str(profile["genesis_hash"]).lower().removeprefix("0x"):
        raise TemporaryTestnetBootstrapError("cutover genesis differs from test profile")
    return {
        "candidate_sha": candidate_sha,
        "gateway_release_hash": gateway_release["release_hash"],
        "validator_release_hash": validator_release["release_hash"],
        "validator_hotkey": hotkey_config["validator_hotkey"],
        "chain_profile_hash": sha256_json(profile),
        "cutover_mapping_hash": canonical_cutover.mapping_hash,
    }


def _imds_instance_id() -> str:
    token_request = request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    try:
        with request.urlopen(token_request, timeout=2) as response:
            token = response.read().decode("ascii")
        identity_request = request.Request(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with request.urlopen(identity_request, timeout=2) as response:
            instance_id = response.read().decode("ascii").strip()
    except Exception as exc:
        raise TemporaryTestnetBootstrapError("IMDSv2 instance identity is unavailable") from exc
    if not INSTANCE_ID_RE.fullmatch(instance_id):
        raise TemporaryTestnetBootstrapError("IMDSv2 returned an invalid instance id")
    return instance_id


def verify_host_authority(
    config: Mapping[str, Any],
    *,
    instance_id: Optional[str] = None,
    sts_client: Any = None,
    ec2_client: Any = None,
) -> Dict[str, Any]:
    if instance_id is None:
        instance_id = _imds_instance_id()
    if instance_id != config["expected_instance_id"]:
        raise TemporaryTestnetBootstrapError("host instance differs from task authority")
    if sts_client is None or ec2_client is None:
        import boto3

        sts_client = sts_client or boto3.client("sts", region_name=EXPECTED_AWS_REGION)
        ec2_client = ec2_client or boto3.client("ec2", region_name=EXPECTED_AWS_REGION)
    account = str(sts_client.get_caller_identity().get("Account") or "")
    if account != EXPECTED_AWS_ACCOUNT:
        raise TemporaryTestnetBootstrapError("AWS account differs from task authority")
    response = ec2_client.describe_instances(InstanceIds=[instance_id])
    instances = [
        item
        for reservation in response.get("Reservations") or ()
        for item in reservation.get("Instances") or ()
    ]
    if len(instances) != 1:
        raise TemporaryTestnetBootstrapError("EC2 instance authority is ambiguous")
    instance = instances[0]
    tags = {str(item.get("Key")): str(item.get("Value")) for item in instance.get("Tags") or ()}
    expires_at_epoch = int(config["expires_at_epoch"])
    now = int(time.time())
    if expires_at_epoch <= now or expires_at_epoch > now + 24 * 60 * 60:
        raise TemporaryTestnetBootstrapError("EC2 task expiry is not active and bounded")
    expected_tags = {
        "Name": (
            f"leadpoet-parity-{config['run_id']}-testnet401-"
            f"exp-{expires_at_epoch:010d}"
        ),
        "leadpoet:parity-run": config["run_id"],
        "leadpoet:candidate-sha": config["candidate_sha"],
        "leadpoet:ephemeral": "true",
    }
    if tags != expected_tags:
        raise TemporaryTestnetBootstrapError("EC2 task ownership tags differ")
    profile_arn = str((instance.get("IamInstanceProfile") or {}).get("Arn") or "")
    addresses = {
        str(instance.get("PublicIpAddress") or ""),
        str(instance.get("PrivateIpAddress") or ""),
    }
    if (
        instance.get("InstanceType") != EXPECTED_INSTANCE_TYPE
        or instance.get("ImageId") != EXPECTED_AMI
        or instance.get("SubnetId") != EXPECTED_SUBNET
        or instance.get("VpcId") != EXPECTED_VPC
        or instance.get("State", {}).get("Name") != "running"
        or instance.get("EnclaveOptions", {}).get("Enabled") is not True
        or not profile_arn.endswith("/" + EXPECTED_INSTANCE_PROFILE)
        or addresses & KNOWN_FINNEY_ADDRESSES
    ):
        raise TemporaryTestnetBootstrapError("EC2 host shape or isolation differs")
    mappings = instance.get("BlockDeviceMappings") or ()
    if len(mappings) != 1:
        raise TemporaryTestnetBootstrapError("EC2 root volume authority differs")
    ebs = mappings[0].get("Ebs") or {}
    volumes = ec2_client.describe_volumes(VolumeIds=[str(ebs.get("VolumeId") or "")])
    volume_rows = volumes.get("Volumes") or ()
    if (
        len(volume_rows) != 1
        or volume_rows[0].get("Encrypted") is not True
        or int(volume_rows[0].get("Size") or 0) != 512
        or ebs.get("DeleteOnTermination") is not True
    ):
        raise TemporaryTestnetBootstrapError("EC2 volume shape differs")
    group_ids = [str(item.get("GroupId") or "") for item in instance.get("SecurityGroups") or ()]
    groups = ec2_client.describe_security_groups(GroupIds=group_ids).get("SecurityGroups") or ()
    if not group_ids or len(groups) != len(group_ids) or any(group.get("IpPermissions") for group in groups):
        raise TemporaryTestnetBootstrapError("EC2 security group permits inbound traffic")
    return {
        "account_id": account,
        "instance_id": instance_id,
        "instance_type": instance["InstanceType"],
        "nitro_enclaves": True,
        "security_group_ingress_rule_count": 0,
        "volume_encrypted": True,
        "volume_size_gib": 512,
    }


def verify_host_is_empty(config: Mapping[str, Any]) -> Dict[str, Any]:
    marker = Path(config["early_isolation_marker"])
    _regular_file(marker, "early isolation marker")
    if marker.read_text(encoding="utf-8").strip() != "isolated":
        raise TemporaryTestnetBootstrapError("early isolation marker differs")
    described = subprocess.run(
        ["nitro-cli", "describe-enclaves"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        enclaves = json.loads(described.stdout)
    except json.JSONDecodeError as exc:
        raise TemporaryTestnetBootstrapError("Nitro enclave state is invalid") from exc
    if any(item.get("State") == "RUNNING" for item in enclaves):
        raise TemporaryTestnetBootstrapError("host already has a running enclave")
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="ignore"
            )
        except OSError:
            continue
        if any(module in command for module in OWNED_PROCESS_MODULES):
            raise TemporaryTestnetBootstrapError("host already has a Leadpoet runtime")
    return {"existing_leadpoet_process_count": 0, "running_enclave_count": 0}


def _environment_file(path: Path) -> Dict[str, str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TemporaryTestnetBootstrapError("runtime environment is unavailable") from exc
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        decoded = None
    if decoded is not None:
        if not isinstance(decoded, Mapping):
            raise TemporaryTestnetBootstrapError("runtime environment JSON is invalid")
        return {str(name): str(value) for name, value in decoded.items()}
    result: Dict[str, str] = {}
    for raw_line in raw.replace("\0", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        try:
            parts = shlex.split(line, posix=True)
        except ValueError as exc:
            raise TemporaryTestnetBootstrapError("runtime environment is malformed") from exc
        if len(parts) != 1 or "=" not in parts[0]:
            raise TemporaryTestnetBootstrapError("runtime environment is malformed")
        name, value = parts[0].split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise TemporaryTestnetBootstrapError("runtime environment name is invalid")
        result[name] = value
    return result


def _runtime_environment(
    source: Path,
    *,
    overrides: Mapping[str, str],
    repo_root: Path,
    candidate_sha: str,
) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(_environment_file(source))
    env.update({str(name): str(value) for name, value in overrides.items()})
    for name in STATIC_AWS_CREDENTIAL_NAMES:
        env.pop(name, None)
    env.update(
        {
            "GITHUB_SHA": candidate_sha,
            "GIT_COMMIT": candidate_sha,
            "GIT_COMMIT_HASH": candidate_sha,
            "LEADPOET_REPO_ROOT": str(repo_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(repo_root),
        }
    )
    return env


def _cutover_environment(config: Mapping[str, Any]) -> Dict[str, str]:
    cutover = _load_json(
        Path(config["validator"]["cutover_manifest"]),
        "testnet cutover manifest",
    )
    return {
        "LEADPOET_SUBNET_EPOCH_CUTOVER_JSON": json.dumps(
            cutover, sort_keys=True, separators=(",", ":")
        )
    }


def _gateway_runtime_overrides(config: Mapping[str, Any]) -> Dict[str, str]:
    policy = _load_json(
        Path(config["gateway"]["artifact_policy"]),
        "encrypted artifact policy",
    )
    bucket = str(policy.get("bucket_host") or "").split(".s3", 1)[0]
    if not re.fullmatch(r"leadpoet-parity-493765492819-[a-z0-9-]+", bucket):
        raise TemporaryTestnetBootstrapError(
            "encrypted artifact policy is not task-owned"
        )
    return {
        **SAFE_GATEWAY_ENV,
        **_cutover_environment(config),
        "RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET": bucket,
    }


def _validator_runtime_overrides(config: Mapping[str, Any]) -> Dict[str, str]:
    return {**SAFE_VALIDATOR_ENV, **_cutover_environment(config)}


def _profile_live_check(config: Mapping[str, Any]) -> Dict[str, Any]:
    validator = config["validator"]
    completed = subprocess.run(
        [
            config["python_bin"],
            "-m",
            "validator_tee.host.verify_chain_signing_profile_v2",
            "--network",
            NETWORK,
            "--netuid",
            str(NETUID),
            "--profile",
            validator["chain_profile"],
        ],
        cwd=config["repo_root"],
        env=_runtime_environment(
            Path(validator["source_env_file"]),
            overrides=_validator_runtime_overrides(config),
            repo_root=Path(config["repo_root"]),
            candidate_sha=config["candidate_sha"],
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise TemporaryTestnetBootstrapError("live chain profile evidence is invalid") from exc
    if (
        result.get("status") != "ready"
        or result.get("netuid") != NETUID
        or result.get("selected_profile_hash") != EXPECTED_PROFILE_HASH
        or result.get("spec_version") != 454
        or result.get("transaction_version") != 1
        or result.get("tempo") != 360
        or result.get("subnet_reveal_period_epochs") != 1
    ):
        raise TemporaryTestnetBootstrapError("live testnet401 signing profile differs")
    return result


def _durable_epoch_authority_check(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Prove the exact testnet401 cutover row before any host mutation."""

    gateway = config["gateway"]
    env = _runtime_environment(
        Path(gateway["source_env_file"]),
        overrides=_gateway_runtime_overrides(config),
        repo_root=Path(config["repo_root"]),
        candidate_sha=config["candidate_sha"],
    )
    completed = subprocess.run(
        [
            config["python_bin"],
            "-c",
            (
                "import json; from Leadpoet.utils.subnet_epoch import "
                "load_subnet_epoch_cutover as l; from gateway.utils.epoch import "
                "validate_stateful_cutover_authority as v; c=v(l(), network='test', "
                "netuid=401); print(json.dumps({'mapping_hash':c.mapping_hash, "
                "'netuid':c.netuid}, sort_keys=True))"
            ),
        ],
        cwd=config["repo_root"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise TemporaryTestnetBootstrapError(
            "durable testnet401 epoch authority evidence is invalid"
        ) from exc
    if result.get("netuid") != NETUID or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(result.get("mapping_hash") or "")
    ):
        raise TemporaryTestnetBootstrapError(
            "durable testnet401 epoch authority differs"
        )
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _receipt_path(config: Mapping[str, Any], stage: str) -> Path:
    return Path(config["runtime_root"]) / "evidence" / f"{stage}.json"


def _write_receipt(
    config: Mapping[str, Any], stage: str, status: str, evidence: Mapping[str, Any]
) -> Dict[str, Any]:
    value = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "stage": stage,
        "status": status,
        "run_id": config["run_id"],
        "candidate_sha": config["candidate_sha"],
        "instance_id": config["expected_instance_id"],
        "recorded_at_unix": int(time.time()),
        "evidence": dict(evidence),
    }
    _atomic_json(_receipt_path(config, stage), value)
    return value


def run_preflight(config: Mapping[str, Any]) -> Dict[str, Any]:
    static = validate_static_inputs(config)
    host = verify_host_authority(config)
    empty = verify_host_is_empty(config)
    live = _profile_live_check(config)
    return _write_receipt(
        config,
        "preflight",
        "passed",
        {
            "static": static,
            "host": host,
            "empty_host": empty,
            "live_chain": {
                "finalized_block": live["finalized_block"],
                "finalized_block_hash": live["finalized_block_hash"],
                "selected_profile_hash": live["selected_profile_hash"],
                "spec_version": live["spec_version"],
                "transaction_version": live["transaction_version"],
                "tempo": live["tempo"],
                "subnet_reveal_period_epochs": live[
                    "subnet_reveal_period_epochs"
                ],
            },
            "epoch_authority": {
                "status": "fresh_ceremony_required_after_measured_boot",
                "existing_row_required": False,
            },
            "allocator": {
                "cpu_count": ALLOCATOR_CPUS,
                "memory_mib": ALLOCATOR_MEMORY_MIB,
                "host_reserved_cpu_count": EXPECTED_PARENT_CPUS - ALLOCATOR_CPUS,
                "host_reserved_memory_mib_floor": (
                    MINIMUM_PARENT_MEMORY_MIB - ALLOCATOR_MEMORY_MIB
                ),
            },
        },
    )


class NativeRunner:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.root = Path(config["runtime_root"])
        self.logs = self.root / "logs"
        self.state_path = self.root / "processes.json"
        self.processes: list[Dict[str, Any]] = []

    def _log_path(self, stage: str) -> Path:
        self.logs.mkdir(parents=True, mode=0o700, exist_ok=True)
        return self.logs / f"{stage}.log"

    def run(
        self,
        stage: str,
        argv: Sequence[str],
        *,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[Path] = None,
    ) -> subprocess.CompletedProcess[bytes]:
        with self._log_path(stage).open("ab") as handle:
            completed = subprocess.run(
                list(argv),
                cwd=str(cwd) if cwd else None,
                env=dict(env) if env is not None else None,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise TemporaryTestnetBootstrapError(f"{stage} failed")
        return completed

    def run_json(
        self,
        stage: str,
        argv: Sequence[str],
        *,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[Path] = None,
    ) -> Dict[str, Any]:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else None,
            capture_output=True,
            check=False,
        )
        with self._log_path(stage).open("ab") as handle:
            handle.write(completed.stdout)
            handle.write(completed.stderr)
        if completed.returncode != 0:
            raise TemporaryTestnetBootstrapError(f"{stage} failed")
        try:
            value = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TemporaryTestnetBootstrapError(
                f"{stage} did not return one JSON object"
            ) from exc
        return _object(value, f"{stage} result")

    def start(
        self,
        name: str,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path,
    ) -> Dict[str, Any]:
        handle = self._log_path(name).open("ab")
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=str(cwd),
                env=dict(env),
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        finally:
            handle.close()
        time.sleep(2)
        if process.poll() is not None:
            raise TemporaryTestnetBootstrapError(f"{name} exited during startup")
        record = _process_identity(process.pid, name=name)
        self.processes.append(record)
        self.persist()
        return record

    def persist(self) -> None:
        _atomic_json(
            self.state_path,
            {
                "schema_version": PROCESS_STATE_SCHEMA_VERSION,
                "run_id": self.config["run_id"],
                "candidate_sha": self.config["candidate_sha"],
                "instance_id": self.config["expected_instance_id"],
                "processes": self.processes,
            },
        )


def _process_identity(pid: int, *, name: str) -> Dict[str, Any]:
    proc = Path("/proc") / str(int(pid))
    try:
        stat_fields = (proc / "stat").read_text(encoding="utf-8").split()
        command = (proc / "cmdline").read_bytes()
    except OSError as exc:
        raise TemporaryTestnetBootstrapError(f"{name} process identity is unavailable") from exc
    if len(stat_fields) < 22 or not command:
        raise TemporaryTestnetBootstrapError(f"{name} process identity is invalid")
    return {
        "name": name,
        "pid": int(pid),
        "start_ticks": stat_fields[21],
        "cmdline_hash": "sha256:" + hashlib.sha256(command).hexdigest(),
    }


def _configure_combined_allocator(runner: NativeRunner) -> None:
    total_cpus = int(os.sysconf("SC_NPROCESSORS_CONF"))
    memory_mib = 0
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemTotal:"):
            memory_mib = int(line.split()[1]) // 1024
            break
    if total_cpus < EXPECTED_PARENT_CPUS or memory_mib < MINIMUM_PARENT_MEMORY_MIB:
        raise TemporaryTestnetBootstrapError("parent capacity is below r7i.4xlarge floor")
    allocator = runner.root / "allocator.yaml"
    allocator.write_text(
        f"---\nmemory_mib: {ALLOCATOR_MEMORY_MIB}\ncpu_count: {ALLOCATOR_CPUS}\n",
        encoding="ascii",
    )
    allocator.chmod(0o600)
    runner.run(
        "allocator_install",
        [
            "sudo",
            "install",
            "-D",
            "-m",
            "0644",
            str(allocator),
            "/etc/nitro_enclaves/allocator.yaml",
        ],
    )
    runner.run(
        "allocator_restart",
        ["sudo", "systemctl", "restart", "nitro-enclaves-allocator.service"],
    )
    runner.run(
        "allocator_active",
        ["sudo", "systemctl", "is-active", "--quiet", "nitro-enclaves-allocator.service"],
    )
    runner.run(
        "allocator_readback",
        ["sudo", "cmp", "-s", str(allocator), "/etc/nitro_enclaves/allocator.yaml"],
    )


def _gateway_envelopes(config: Mapping[str, Any]) -> list[str]:
    directory = Path(config["gateway"]["config_dir"])
    return [str(directory / name) for name in GATEWAY_ENVELOPE_NAMES]


def _gateway_bootstrap_command(config: Mapping[str, Any]) -> list[str]:
    gateway = config["gateway"]
    command = [
        config["python_bin"],
        "-m",
        "gateway.utils.tee_v2_bootstrap",
        "--release-manifest",
        gateway["release_manifest"],
        "--gateway-release-lineage",
        gateway["release_lineage"],
    ]
    for envelope in _gateway_envelopes(config):
        command.extend(("--credential-envelope", envelope))
    command.extend(
        (
            "--protected-workflow-manifest",
            gateway["protected_workflow_manifest"],
            "--encrypted-artifact-policy",
            gateway["artifact_policy"],
            "--config-dir",
            gateway["config_dir"],
        )
    )
    return command


def _gateway_provision_command(config: Mapping[str, Any]) -> list[str]:
    command = [config["python_bin"], "-m", "gateway.utils.tee_kms_provision_v2"]
    for envelope in _gateway_envelopes(config):
        command.extend(("--envelope", envelope))
    return command


def launch_sequence_names() -> tuple[str, ...]:
    """Return the security-significant native launch order for regression tests."""

    return (
        "preflight_receipt",
        "gateway_envelope_install",
        "combined_allocator",
        "gateway_enclaves",
        "gateway_egress_relay",
        "gateway_inter_enclave_relay",
        "gateway_runtime_bootstrap",
        "gateway_kms_recipient_provision",
        "gateway_runtime_readiness",
        "validator_enclave_cid18",
        "validator_chain_relay_cid18",
        "validator_runtime_bootstrap_cid18",
        "validator_hotkey_recipient_cid18",
        "validator_epoch_boundary_capture_cid18",
        "validator_epoch_candidate_ingest",
        "gateway_epoch_cutover_attestation",
        "durable_epoch_authority_readback",
        "gateway_application",
        "gateway_http_readiness",
        "validator_application_cid18",
    )


def _bootstrap_fresh_epoch_authority(
    config: Mapping[str, Any],
    runner: NativeRunner,
    *,
    gateway_env: Mapping[str, str],
    validator_env: Mapping[str, str],
) -> Dict[str, Any]:
    validator = config["validator"]
    gateway = config["gateway"]
    repo_root = Path(config["repo_root"])
    cutover = _load_json(
        Path(validator["cutover_manifest"]), "testnet cutover manifest"
    )
    mapping_hash = str(cutover.get("mapping_hash") or "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", mapping_hash):
        raise TemporaryTestnetBootstrapError("cutover mapping hash is invalid")
    candidate_path = Path(config["runtime_root"]) / "evidence" / "epoch-candidate.json"
    runner.run_json(
        "validator_epoch_boundary_capture",
        [
            config["python_bin"],
            "-m",
            "validator_tee.host.subnet_epoch_boundary_capture_v2",
            "--cutover-manifest",
            validator["cutover_manifest"],
            "--validator-release-manifest",
            validator["release_manifest"],
            "--settlement-epoch-id",
            str(EXPECTED_FIRST_SETTLEMENT_EPOCH),
            "--candidate-output",
            str(candidate_path),
            "--wallet-name",
            validator["wallet_name"],
            "--wallet-hotkey",
            validator["wallet_hotkey"],
            "--wallet-path",
            validator["wallet_path"],
        ],
        env=validator_env,
        cwd=repo_root,
    )
    preview = runner.run_json(
        "validator_epoch_candidate_preview",
        [
            config["python_bin"],
            "-m",
            "gateway.research_lab.stateful_epoch_candidate_ingest_cli_v1",
            "--candidate",
            str(candidate_path),
            "--validator-release-manifest",
            validator["release_manifest"],
        ],
        env=gateway_env,
        cwd=repo_root,
    )
    payload_hash = str(preview.get("candidate_payload_hash") or "")
    if (
        preview.get("status") != "validated_no_writes"
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", payload_hash)
    ):
        raise TemporaryTestnetBootstrapError("epoch candidate preview differs")
    staged = runner.run_json(
        "validator_epoch_candidate_ingest",
        [
            config["python_bin"],
            "-m",
            "gateway.research_lab.stateful_epoch_candidate_ingest_cli_v1",
            "--candidate",
            str(candidate_path),
            "--validator-release-manifest",
            validator["release_manifest"],
            "--apply",
            "--confirm-candidate-payload-hash",
            payload_hash,
        ],
        env=gateway_env,
        cwd=repo_root,
    )
    if staged.get("status") != "durably_staged":
        raise TemporaryTestnetBootstrapError("epoch candidate durable ingest differs")
    attested = runner.run_json(
        "gateway_epoch_cutover_attestation",
        [
            config["python_bin"],
            "-m",
            "gateway.research_lab.stateful_epoch_cutover_cli_v1",
            "--manifest",
            validator["cutover_manifest"],
            "--release-manifest",
            gateway["release_manifest"],
            "--validator-release-manifest",
            validator["release_manifest"],
            "--fresh-testnet401-network",
            "--apply",
            "--confirm-mapping-hash",
            mapping_hash,
            "--confirm-first-settlement-epoch-id",
            str(EXPECTED_FIRST_SETTLEMENT_EPOCH),
            "--confirm-all-writers-stopped",
        ],
        env=gateway_env,
        cwd=repo_root,
    )
    if attested.get("status") not in {
        "fresh_network_durable",
        "fresh_network_already_durable",
    }:
        raise TemporaryTestnetBootstrapError(
            "fresh testnet401 cutover attestation differs"
        )
    durable = _durable_epoch_authority_check(config)
    if durable.get("mapping_hash") != mapping_hash:
        raise TemporaryTestnetBootstrapError("durable epoch authority readback differs")
    return {
        "candidate_payload_hash": payload_hash,
        "candidate_authorization_hash": staged.get("candidate_authorization_hash"),
        "mapping_hash": mapping_hash,
        "cutover_status": attested.get("status"),
        "durable_mapping_hash": durable["mapping_hash"],
    }


def _wait_gateway(config: Mapping[str, Any], runner: NativeRunner) -> Dict[str, Any]:
    deadline = time.monotonic() + 600
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            with request.urlopen("http://127.0.0.1:8000/health/v2-authority", timeout=5) as response:
                health = json.loads(response.read().decode("utf-8"))
            with request.urlopen("http://127.0.0.1:8000/build-info", timeout=5) as response:
                build = json.loads(response.read().decode("utf-8"))
            with request.urlopen(
                "http://127.0.0.1:8000/fulfillment/lab-arena-reward-basis"
                f"?epoch={EXPECTED_REWARD_EFFECTIVE_EPOCH}",
                timeout=5,
            ) as response:
                reward = json.loads(response.read().decode("utf-8"))
            basis = reward.get("reward_basis")
            if (
                health.get("status") == "ready"
                and str(health.get("commit_sha") or "").lower() == config["candidate_sha"]
                and str(build.get("git_commit") or "").lower() == config["candidate_sha"]
                and reward.get("lookup_ok") is True
                and reward.get("round_id") == EXPECTED_REWARD_ROUND_ID
                and reward.get("reward_basis_hash") == EXPECTED_REWARD_BASIS_HASH
                and isinstance(basis, Mapping)
                and basis.get("reward_basis_hash") == EXPECTED_REWARD_BASIS_HASH
                and basis.get("effective_reward_epoch")
                == EXPECTED_REWARD_EFFECTIVE_EPOCH
                and basis.get("king_hotkey") == EXPECTED_REWARD_KING_HOTKEY
                and basis.get("king_outcome") == "crowned"
                and isinstance(basis.get("reward_constants"), Mapping)
                and basis["reward_constants"].get("pool_percent") == 25
            ):
                return {
                    "status": "ready",
                    "commit_sha": config["candidate_sha"],
                    "release_hash": health.get("release_hash"),
                    "reward_basis_hash": EXPECTED_REWARD_BASIS_HASH,
                    "reward_round_id": EXPECTED_REWARD_ROUND_ID,
                    "reward_effective_epoch": EXPECTED_REWARD_EFFECTIVE_EPOCH,
                    "reward_king_hotkey": EXPECTED_REWARD_KING_HOTKEY,
                }
        except Exception as exc:
            last_error = exc
        time.sleep(5)
    raise TemporaryTestnetBootstrapError("gateway did not reach exact-release readiness") from last_error


def _wait_validator_poll_readiness(
    process: Mapping[str, Any], readiness_path: Path
) -> Dict[str, Any]:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if not _same_process(process):
            raise TemporaryTestnetBootstrapError(
                "weight-only validator exited before native poll readiness"
            )
        if readiness_path.exists():
            _private_regular_file(readiness_path, "validator poll readiness")
            value = _load_json(readiness_path, "validator poll readiness")
            if value != {
                "schema_version": "leadpoet.temporary_testnet401_weight_poll_ready.v1",
                "status": "ready",
            }:
                raise TemporaryTestnetBootstrapError(
                    "validator poll readiness differs"
                )
            return value
        time.sleep(2)
    raise TemporaryTestnetBootstrapError(
        "weight-only validator did not complete one native poll"
    )


def run_launch(config: Mapping[str, Any], *, confirm_instance_id: str) -> Dict[str, Any]:
    if confirm_instance_id != config["expected_instance_id"]:
        raise TemporaryTestnetBootstrapError("launch instance confirmation differs")
    preflight = _load_json(_receipt_path(config, "preflight"), "preflight receipt")
    if (
        preflight.get("status") != "passed"
        or preflight.get("candidate_sha") != config["candidate_sha"]
        or preflight.get("instance_id") != confirm_instance_id
    ):
        raise TemporaryTestnetBootstrapError("exact preflight receipt is unavailable")
    validate_static_inputs(config)
    verify_host_authority(config)
    verify_host_is_empty(config)
    live = _profile_live_check(config)
    if live.get("selected_profile_hash") != EXPECTED_PROFILE_HASH:
        raise TemporaryTestnetBootstrapError("live profile changed after preflight")
    root = Path(config["runtime_root"])
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    gateway = config["gateway"]
    validator = config["validator"]
    repo_root = Path(config["repo_root"])
    gateway_env_path = root / "gateway.env"
    gateway_env_descriptor = os.open(
        gateway_env_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(gateway_env_descriptor, "wb") as gateway_env_handle:
        gateway_env_handle.write(Path(gateway["source_env_file"]).read_bytes())
        gateway_env_handle.flush()
        os.fsync(gateway_env_handle.fileno())
    gateway_env = _runtime_environment(
        gateway_env_path,
        overrides={
            **_gateway_runtime_overrides(config),
            "GATEWAY_ENV_FILE": str(gateway_env_path),
            "GATEWAY_TEE_EIF_ROOT": gateway["eif_root"],
            "GATEWAY_V2_CONFIG_DIR": gateway["config_dir"],
            "GATEWAY_V2_RELEASE_MANIFEST": gateway["release_manifest"],
        },
        repo_root=repo_root,
        candidate_sha=config["candidate_sha"],
    )
    validator_env = _runtime_environment(
        Path(validator["source_env_file"]),
        overrides={
            **_validator_runtime_overrides(config),
            "VALIDATOR_CHAIN_SIGNING_PROFILE": validator["chain_profile"],
            "VALIDATOR_V2_DEPLOY_COMMIT": config["candidate_sha"],
        },
        repo_root=repo_root,
        candidate_sha=config["candidate_sha"],
    )
    runner = NativeRunner(config)
    try:
        runner.run(
            "gateway_envelope_install",
            [
                config["python_bin"],
                "-m",
                "gateway.tee.prepare_gateway_envelopes_v2",
                "--install",
                "--env-file",
                str(gateway_env_path),
                "--kms-key-id",
                gateway["kms_key_id"],
                "--deploy-commit",
                config["candidate_sha"],
                "--output-dir",
                gateway["config_dir"],
            ],
            env=gateway_env,
            cwd=repo_root,
        )
        runner.run(
            "gateway_env_scrub",
            [
                config["python_bin"],
                "-c",
                (
                    "from gateway.tee.prepare_gateway_envelopes_v2 import "
                    "scrub_parent_environment_file_v2 as s; import sys; "
                    "s(environment_path=__import__('pathlib').Path(sys.argv[1]), "
                    "transition_report_path=__import__('pathlib').Path(sys.argv[2]))"
                ),
                str(gateway_env_path),
                str(Path(gateway["config_dir"]) / "gateway-v2-env-transition.json"),
            ],
            env=gateway_env,
            cwd=repo_root,
        )
        gateway_env = _runtime_environment(
            gateway_env_path,
            overrides={
                **_gateway_runtime_overrides(config),
                "GATEWAY_ENV_FILE": str(gateway_env_path),
                "GATEWAY_TEE_EIF_ROOT": gateway["eif_root"],
                "GATEWAY_V2_CONFIG_DIR": gateway["config_dir"],
                "GATEWAY_V2_RELEASE_MANIFEST": gateway["release_manifest"],
            },
            repo_root=repo_root,
            candidate_sha=config["candidate_sha"],
        )
        _configure_combined_allocator(runner)
        runner.run(
            "gateway_enclaves",
            ["bash", str(repo_root / "gateway" / "tee" / "start_enclave.sh")],
            env={
                **gateway_env,
                "GATEWAY_ROOT": str(repo_root / "gateway"),
                "GATEWAY_TEE_TOPOLOGY_MODE": "full",
            },
            cwd=repo_root / "gateway" / "tee",
        )
        runner.start(
            "gateway_egress_relay",
            [config["python_bin"], "-u", "-m", "gateway.utils.tee_egress_forwarder"],
            env=gateway_env,
            cwd=repo_root,
        )
        runner.start(
            "gateway_inter_enclave_relay",
            [config["python_bin"], "-u", "-m", "gateway.utils.tee_inter_enclave_relay"],
            env=gateway_env,
            cwd=repo_root,
        )
        runner.run(
            "gateway_runtime_bootstrap",
            _gateway_bootstrap_command(config),
            env=gateway_env,
            cwd=repo_root,
        )
        runner.run(
            "gateway_kms_recipient_provision",
            _gateway_provision_command(config),
            env=gateway_env,
            cwd=repo_root,
        )
        runner.run(
            "gateway_runtime_readiness",
            [config["python_bin"], "-m", "gateway.tee.verify_v2_runtime_ready"],
            env=gateway_env,
            cwd=repo_root,
        )
        runner.run(
            "validator_enclave_cid18",
            [
                "sudo",
                "nitro-cli",
                "run-enclave",
                "--eif-path",
                validator["eif_path"],
                "--cpu-count",
                str(VALIDATOR_CPUS),
                "--memory",
                str(VALIDATOR_MEMORY_MIB),
                "--enclave-name",
                "leadpoet-testnet401-validator",
                "--enclave-cid",
                str(VALIDATOR_CID),
            ],
        )
        runner.start(
            "validator_chain_relay",
            [
                config["python_bin"],
                "-u",
                "-m",
                "validator_tee.host.chain_relay_v2",
                "--chain-signing-profile",
                validator["chain_profile"],
                "--port",
                "5004",
            ],
            env=validator_env,
            cwd=repo_root,
        )
        runner.run(
            "validator_runtime_bootstrap",
            [
                config["python_bin"],
                "-m",
                "validator_tee.host.runtime_v2_bootstrap",
                "--validator-release",
                validator["release_manifest"],
                "--gateway-release",
                gateway["release_manifest"],
                "--gateway-release-lineage",
                gateway["release_lineage"],
                "--hotkey-config",
                validator["hotkey_config"],
            ],
            env=validator_env,
            cwd=repo_root,
        )
        runner.run(
            "validator_hotkey_recipient",
            [
                config["python_bin"],
                "-m",
                "validator_tee.host.hotkey_bootstrap_v2",
                "--hotkey-config",
                validator["hotkey_config"],
                "--hotkey-envelope",
                validator["hotkey_envelope"],
            ],
            env=validator_env,
            cwd=repo_root,
        )
        epoch_authority = _bootstrap_fresh_epoch_authority(
            config,
            runner,
            gateway_env=gateway_env,
            validator_env=validator_env,
        )
        runner.start(
            "gateway_application",
            [config["python_bin"], "-u", "-m", "gateway.main"],
            env=gateway_env,
            cwd=repo_root,
        )
        gateway_ready = _wait_gateway(config, runner)
        validator_poll_readiness_path = (
            Path(config["runtime_root"]) / "evidence" / "validator-poll-ready.json"
        )
        if validator_poll_readiness_path.exists():
            raise TemporaryTestnetBootstrapError(
                "validator poll readiness already exists"
            )
        validator_process = runner.start(
            "validator_application",
            [
                config["python_bin"],
                "-u",
                "scripts/run_temporary_testnet401_weight_only_validator.py",
                "--wallet-name",
                validator["wallet_name"],
                "--wallet-hotkey",
                validator["wallet_hotkey"],
                "--wallet-path",
                validator["wallet_path"],
                "--state-path",
                str(Path(config["runtime_root"]) / "validator-state"),
                "--readiness-file",
                str(validator_poll_readiness_path),
            ],
            env=validator_env,
            cwd=repo_root,
        )
        validator_poll_ready = _wait_validator_poll_readiness(
            validator_process,
            validator_poll_readiness_path,
        )
        return _write_receipt(
            config,
            "launch",
            "launched",
            {
                "gateway": gateway_ready,
                "epoch_authority": epoch_authority,
                "validator_hotkey": EXPECTED_VALIDATOR_HOTKEY,
                "validator_enclave_cid": VALIDATOR_CID,
                "automatic_validator_pid": validator_process["pid"],
                "validator_poll_readiness": validator_poll_ready,
                "chain_write_performed_by_helper": False,
                "automatic_chain_proof_pending": True,
            },
        )
    except Exception as exc:
        _stop_owned_processes(runner.processes)
        _stop_owned_enclaves()
        _write_receipt(
            config,
            "launch",
            "failed",
            {"failure_type": type(exc).__name__},
        )
        raise


def _same_process(record: Mapping[str, Any]) -> bool:
    try:
        observed = _process_identity(int(record["pid"]), name=str(record["name"]))
    except TemporaryTestnetBootstrapError:
        return False
    return (
        observed["start_ticks"] == record.get("start_ticks")
        and observed["cmdline_hash"] == record.get("cmdline_hash")
    )


def _stop_owned_processes(processes: Iterable[Mapping[str, Any]]) -> None:
    owned = [dict(item) for item in processes if _same_process(item)]
    for record in reversed(owned):
        try:
            os.killpg(int(record["pid"]), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and any(_same_process(item) for item in owned):
        time.sleep(0.25)
    for record in reversed(owned):
        if not _same_process(record):
            continue
        try:
            os.killpg(int(record["pid"]), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def _stop_owned_enclaves() -> list[int]:
    completed = subprocess.run(
        ["nitro-cli", "describe-enclaves"], check=True, capture_output=True, text=True
    )
    enclaves = json.loads(completed.stdout)
    stopped = []
    for item in enclaves:
        cid = int(item.get("EnclaveCID") or -1)
        name = str(item.get("EnclaveName") or "")
        if cid not in {16, 17, VALIDATOR_CID}:
            continue
        if cid == VALIDATOR_CID and name != "leadpoet-testnet401-validator":
            raise TemporaryTestnetBootstrapError("CID 18 is not task-owned")
        enclave_id = str(item.get("EnclaveID") or "")
        if not enclave_id:
            raise TemporaryTestnetBootstrapError("task enclave id is unavailable")
        subprocess.run(
            ["sudo", "nitro-cli", "terminate-enclave", "--enclave-id", enclave_id],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        stopped.append(cid)
    return sorted(stopped)


def _load_process_state(config: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(config["runtime_root"]) / "processes.json"
    state = _load_json(path, "process state")
    if (
        state.get("schema_version") != PROCESS_STATE_SCHEMA_VERSION
        or state.get("run_id") != config["run_id"]
        or state.get("candidate_sha") != config["candidate_sha"]
        or state.get("instance_id") != config["expected_instance_id"]
        or not isinstance(state.get("processes"), list)
    ):
        raise TemporaryTestnetBootstrapError("process state authority differs")
    return state


def _read_testnet401_chain_status(config: Mapping[str, Any]) -> Dict[str, Any]:
    validator = config["validator"]
    completed = subprocess.run(
        [
            config["python_bin"],
            "-c",
            (
                "import bittensor as bt,json,sys; from pathlib import Path; "
                "from Leadpoet.utils.subnet_epoch import (SubnetEpochCutover,"
                "read_subnet_epoch_snapshot); c=SubnetEpochCutover.from_mapping("
                "json.loads(Path(sys.argv[1]).read_text())); s=bt.Subtensor(network='test'); "
                "x=read_subnet_epoch_snapshot(s,netuid=401,finalized=True); h=x.block_hash; "
                "q=s.substrate.query; vh=sys.argv[2]; mh=sys.argv[3]; "
                "vu=getattr(q('SubtensorModule','Uids',[401,vh],block_hash=h),'value',None); "
                "mu=getattr(q('SubtensorModule','Uids',[401,mh],block_hash=h),'value',None); "
                "lu=getattr(q('SubtensorModule','LastUpdate',[401],block_hash=h),'value',None); "
                "w=getattr(q('SubtensorModule','Weights',[401,vu],block_hash=h),'value',None); "
                "print(json.dumps({'finalized_block':x.current_block,'finalized_block_hash':h,"
                "'settlement_epoch_id':x.settlement_epoch_id(c),'validator_uid':vu,"
                "'miner_uid':mu,'validator_last_update':lu[vu],'revealed_weights':w},"
                "sort_keys=True))"
            ),
            validator["cutover_manifest"],
            EXPECTED_VALIDATOR_HOTKEY,
            EXPECTED_REWARD_KING_HOTKEY,
        ],
        cwd=config["repo_root"],
        env=_runtime_environment(
            Path(validator["source_env_file"]),
            overrides=_validator_runtime_overrides(config),
            repo_root=Path(config["repo_root"]),
            candidate_sha=config["candidate_sha"],
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    try:
        value = _object(json.loads(completed.stdout), "chain status")
    except json.JSONDecodeError as exc:
        raise TemporaryTestnetBootstrapError("chain status is invalid") from exc
    if (
        value.get("validator_uid") != 9
        or value.get("miner_uid") != 11
        or not isinstance(value.get("revealed_weights"), list)
    ):
        raise TemporaryTestnetBootstrapError("testnet401 identities changed")
    return value


def _gateway_weight_authority_summary(epoch_id: int) -> Dict[str, Any]:
    url = f"http://127.0.0.1:8000/weights/v2/published-compact/{NETUID}/{epoch_id}"
    try:
        with request.urlopen(url, timeout=15) as response:
            authority = _object(
                json.loads(response.read().decode("utf-8")),
                "compact weight authority",
            )
    except Exception:
        return {"epoch_id": epoch_id, "status": "not_available"}
    compact = authority.get("compact_submission")
    publication = authority.get("publication")
    finalization = authority.get("finalization")
    weight_result = compact.get("weight_result") if isinstance(compact, Mapping) else None
    finalization_submission = (
        finalization.get("compact_submission")
        if isinstance(finalization, Mapping)
        else None
    )
    finalization_doc = (
        finalization_submission.get("finalization")
        if isinstance(finalization_submission, Mapping)
        else None
    )
    return {
        "epoch_id": epoch_id,
        "status": "available",
        "authority_stage": authority.get("authority_stage"),
        "authority_hash": authority.get("authority_hash"),
        "lineage_id": authority.get("lineage_id"),
        "bundle_hash": authority.get("bundle_hash"),
        "weights_hash": (
            weight_result.get("weights_hash")
            if isinstance(weight_result, Mapping)
            else None
        ),
        "uids": (
            weight_result.get("sparse_uids")
            if isinstance(weight_result, Mapping)
            else None
        ),
        "weights_u16": (
            weight_result.get("sparse_weights_u16")
            if isinstance(weight_result, Mapping)
            else None
        ),
        "weight_submission_event_hash": (
            publication.get("weight_submission_event_hash")
            if isinstance(publication, Mapping)
            else None
        ),
        "publication_receipt_hash": (
            publication.get("publication_receipt_hash")
            if isinstance(publication, Mapping)
            else None
        ),
        "weight_finalization_event_hash": (
            finalization.get("weight_finalization_event_hash")
            if isinstance(finalization, Mapping)
            else None
        ),
        "extrinsic_hash": (
            finalization_doc.get("extrinsic_hash")
            if isinstance(finalization_doc, Mapping)
            else None
        ),
        "finalized_block": (
            finalization_doc.get("finalized_block")
            if isinstance(finalization_doc, Mapping)
            else None
        ),
        "finalized_block_hash": (
            finalization_doc.get("finalized_block_hash")
            if isinstance(finalization_doc, Mapping)
            else None
        ),
    }


def _automatic_chain_proof(
    *,
    chain: Mapping[str, Any],
    weight_authorities: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    revealed = chain.get("revealed_weights")
    if not isinstance(revealed, list):
        return {"status": "pending", "reason": "chain_readback_unavailable"}
    try:
        chain_pairs = sorted((int(item[0]), int(item[1])) for item in revealed)
    except (TypeError, ValueError, IndexError):
        return {"status": "pending", "reason": "chain_readback_invalid"}
    for authority in weight_authorities:
        uids = authority.get("uids")
        weights = authority.get("weights_u16")
        if (
            authority.get("status") != "available"
            or authority.get("authority_stage") != "finalized"
            or not isinstance(uids, list)
            or not isinstance(weights, list)
            or len(uids) != len(weights)
        ):
            continue
        try:
            native_pairs = sorted(
                (int(uid), int(weight)) for uid, weight in zip(uids, weights)
            )
            finalized_block = int(authority.get("finalized_block"))
            last_update = int(chain.get("validator_last_update"))
            finalized_head = int(chain.get("finalized_block"))
        except (TypeError, ValueError):
            continue
        champion_weights = [
            weight for uid, weight in native_pairs if uid == 11
        ]
        if (
            native_pairs != chain_pairs
            or {uid for uid, _weight in native_pairs} != {0, 11}
            or len(champion_weights) != 1
            or 4 * champion_weights[0] != sum(weight for _uid, weight in native_pairs)
            or last_update <= BEFORE_TESTNET401_LAST_UPDATE
            or finalized_block > finalized_head
            or not authority.get("weight_submission_event_hash")
            or not authority.get("publication_receipt_hash")
            or not authority.get("weight_finalization_event_hash")
            or not authority.get("extrinsic_hash")
            or not authority.get("finalized_block_hash")
        ):
            continue
        return {
            "status": "candidate_match",
            "epoch_id": authority.get("epoch_id"),
            "authority_hash": authority.get("authority_hash"),
            "bundle_hash": authority.get("bundle_hash"),
            "weights_hash": authority.get("weights_hash"),
            "weight_submission_event_hash": authority.get(
                "weight_submission_event_hash"
            ),
            "weight_finalization_event_hash": authority.get(
                "weight_finalization_event_hash"
            ),
            "extrinsic_hash": authority.get("extrinsic_hash"),
            "finalized_block": finalized_block,
            "revealed_weights": chain_pairs,
            "validator_last_update": last_update,
            "proof_basis": "native_finalized_authority_joined_to_finalized_chain_readback",
            "independent_verification_pending": True,
        }
    return {"status": "pending", "reason": "joined_native_finalization_unavailable"}


def run_status(config: Mapping[str, Any]) -> Dict[str, Any]:
    verify_host_authority(config)
    state = _load_process_state(config)
    process_status = {
        str(item.get("name")): _same_process(item) for item in state["processes"]
    }
    try:
        with request.urlopen("http://127.0.0.1:8000/health/v2-authority", timeout=5) as response:
            gateway = json.loads(response.read().decode("utf-8"))
    except Exception:
        gateway = {"status": "unavailable"}
    validator_env = _runtime_environment(
        Path(config["validator"]["source_env_file"]),
        overrides=_validator_runtime_overrides(config),
        repo_root=Path(config["repo_root"]),
        candidate_sha=config["candidate_sha"],
    )
    hotkey = subprocess.run(
        [
            config["python_bin"],
            "-c",
            (
                "import json; from validator_tee.host.vsock_client import "
                "ValidatorEnclaveClient; print(json.dumps("
                "ValidatorEnclaveClient(enclave_cid=18).get_hotkey_state_v2(), "
                "sort_keys=True))"
            ),
        ],
        cwd=config["repo_root"],
        env=validator_env,
        check=True,
        capture_output=True,
        text=True,
    )
    hotkey_state = json.loads(hotkey.stdout)
    try:
        chain = _read_testnet401_chain_status(config)
        current_epoch = int(chain["settlement_epoch_id"])
        weight_authorities = [
            _gateway_weight_authority_summary(epoch_id)
            for epoch_id in (current_epoch - 1, current_epoch)
        ]
    except Exception as exc:
        chain = {"status": "unavailable", "failure_type": type(exc).__name__}
        weight_authorities = []
    chain_proof = _automatic_chain_proof(
        chain=chain,
        weight_authorities=weight_authorities,
    )
    healthy = (
        all(process_status.values())
        and gateway.get("status") == "ready"
        and gateway.get("commit_sha") == config["candidate_sha"]
        and hotkey_state.get("provisioned") is True
        and hotkey_state.get("validator_hotkey") == EXPECTED_VALIDATOR_HOTKEY
    )
    return _write_receipt(
        config,
        "status",
        "ready" if healthy else "not_ready",
        {
            "processes": process_status,
            "gateway_status": gateway.get("status"),
            "gateway_commit_sha": gateway.get("commit_sha"),
            "validator_hotkey": hotkey_state.get("validator_hotkey"),
            "validator_hotkey_provisioned": hotkey_state.get("provisioned"),
            "chain": chain,
            "native_weight_authorities": weight_authorities,
            "automatic_chain_proof": chain_proof,
            "evidence_paths": {
                "launch_receipt": str(_receipt_path(config, "launch")),
                "process_state": str(Path(config["runtime_root"]) / "processes.json"),
                "private_stage_logs": str(Path(config["runtime_root"]) / "logs"),
            },
            "automatic_chain_proof_not_inferred": True,
        },
    )


def run_cleanup(config: Mapping[str, Any], *, confirm_instance_id: str) -> Dict[str, Any]:
    if confirm_instance_id != config["expected_instance_id"]:
        raise TemporaryTestnetBootstrapError("cleanup instance confirmation differs")
    verify_host_authority(config)
    state = _load_process_state(config)
    _stop_owned_processes(state["processes"])
    stopped_cids = _stop_owned_enclaves()
    remaining = {
        str(item.get("name")): _same_process(item) for item in state["processes"]
    }
    if any(remaining.values()):
        raise TemporaryTestnetBootstrapError("task-owned process cleanup is incomplete")
    return _write_receipt(
        config,
        "cleanup",
        "passed",
        {
            "remaining_owned_processes": [],
            "terminated_enclave_cids": stopped_cids,
            "instance_termination_pending": True,
            "runtime_root_removal_pending_instance_termination": True,
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "launch", "status", "cleanup"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirm-instance-id")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "preflight":
        result = run_preflight(config)
    elif args.command == "launch":
        result = run_launch(config, confirm_instance_id=str(args.confirm_instance_id or ""))
    elif args.command == "status":
        result = run_status(config)
    else:
        result = run_cleanup(config, confirm_instance_id=str(args.confirm_instance_id or ""))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
