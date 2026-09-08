"""Prepare one owned testnet host with the current native runtime builders.

No service is restarted and no chain write is made here.  Private inputs stay
on the owned host; only stage names and public resource identities are printed.
This helper is temporary and is removed after the live signing proof.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any

import boto3

from scripts import bootstrap_temporary_testnet_weights_host as native


ROOT = Path("/run/leadpoet-testnet401")
KMS_KEY = "arn:aws:kms:us-east-1:493765492819:key/c5412928-093e-4bf5-aafc-7b27c02f1445"
INPUT_NAMES = (
    "validator.env",
    "testnet-hotkey-config.json",
    "testnet-hotkey-envelope.json",
    "testnet401-epoch-cutover.json",
)
MAX_PRIVATE_INPUT_BYTES = 262_144
DIAGNOSTIC_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
DIAGNOSTIC_LOCATION_RE = re.compile(r"^[A-Za-z0-9_./-]{1,220}:[0-9]{1,6}$")
NITRO_CLI_ARTIFACTS = ROOT / "nitro-cli-artifacts"
NITRO_CLI_BLOBS = Path("/usr/share/nitro_enclaves/blobs")
NITRO_CLI_BLOB_NAMES = (
    "bzImage",
    "bzImage.config",
    "cmdline",
    "init",
    "linuxkit",
    "nsm.ko",
)


def read_locked_private_input(
    s3: Any,
    *,
    bucket: str,
    key: str,
    now: datetime | None = None,
) -> bytes:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("private input clock is invalid")
    head = s3.head_object(Bucket=bucket, Key=key)
    size = int(head.get("ContentLength", 0))
    retain_until = head.get("ObjectLockRetainUntilDate")
    version_id = str(head.get("VersionId") or "")
    if (
        size not in range(1, MAX_PRIVATE_INPUT_BYTES + 1)
        or head.get("ServerSideEncryption") != "AES256"
        or head.get("ObjectLockMode") != "COMPLIANCE"
        or not isinstance(retain_until, datetime)
        or retain_until.tzinfo is None
        or retain_until.astimezone(timezone.utc)
        <= current.astimezone(timezone.utc)
        or not 0 < len(version_id.encode("utf-8")) <= 1024
    ):
        raise ValueError("private input retention metadata differs")
    # The runner can read current parity objects, not historical versions.
    # Require the GET to match the locked HEAD; a concurrent replacement fails
    # closed without expanding the runner's IAM permissions.
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        if (
            int(response.get("ContentLength", 0)) != size
            or str(response.get("VersionId") or "") != version_id
            or response.get("ServerSideEncryption") != "AES256"
        ):
            raise ValueError("private input version differs")
        payload = body.read(MAX_PRIVATE_INPUT_BYTES + 1)
    finally:
        body.close()
    if len(payload) != size:
        raise ValueError("private input is outside size limit")
    return payload


def private_write(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def failure_receipt(exc: BaseException) -> dict[str, str]:
    """Return bounded failure identity without exception text or input values."""
    result = {"status": "failed", "error_type": type(exc).__name__}
    response = getattr(exc, "response", {})
    error = response.get("Error", {}) if isinstance(response, dict) else {}
    code = str(error.get("Code") or "") if isinstance(error, dict) else ""
    operation = str(getattr(exc, "operation_name", "") or "")
    repository = Path(__file__).resolve().parents[1]
    location = ""
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        try:
            relative = Path(frame.filename).resolve().relative_to(repository)
        except (OSError, ValueError):
            continue
        candidate = f"{relative.as_posix()}:{frame.lineno}"
        if DIAGNOSTIC_LOCATION_RE.fullmatch(candidate):
            location = candidate
            if not operation:
                operation = frame.name
            break
    for name, value in (("operation", operation), ("code", code)):
        if DIAGNOSTIC_VALUE_RE.fullmatch(value):
            result[name] = value
    if DIAGNOSTIC_LOCATION_RE.fullmatch(location):
        result["location"] = location
    return result


def prepare_nitro_cli_environment(
    *, artifacts: Path = NITRO_CLI_ARTIFACTS, blobs: Path = NITRO_CLI_BLOBS
) -> dict[str, str]:
    """Bind non-login Nitro builds to task-owned artifacts and RPM blobs."""
    if blobs.is_symlink() or not blobs.is_dir():
        raise ValueError("Nitro CLI blobs directory is unavailable")
    for name in NITRO_CLI_BLOB_NAMES:
        native._regular_file(blobs / name, f"Nitro CLI blob {name}")
    artifacts.mkdir(mode=0o700, exist_ok=False)
    if artifacts.is_symlink() or not artifacts.is_dir():
        raise ValueError("Nitro CLI artifacts directory is unavailable")
    return {
        "NITRO_CLI_ARTIFACTS": str(artifacts),
        "NITRO_CLI_BLOBS": str(blobs),
    }


def build_config(*, repository: Path, candidate: str, run_id: str,
                 instance_id: str, expiry: int) -> dict[str, Any]:
    return {
        "schema_version": native.SCHEMA_VERSION,
        "run_id": run_id,
        "candidate_sha": candidate,
        "expected_instance_id": instance_id,
        "expires_at_epoch": expiry,
        "runtime_root": str(ROOT),
        "repo_root": str(repository),
        "python_bin": sys.executable,
        "early_isolation_marker": str(ROOT / "early-boot-isolated"),
        "aws": {
            "account_id": native.EXPECTED_AWS_ACCOUNT,
            "region": native.EXPECTED_AWS_REGION,
            "ami": native.EXPECTED_AMI,
            "subnet_id": native.EXPECTED_SUBNET,
            "vpc_id": native.EXPECTED_VPC,
            "instance_type": native.EXPECTED_INSTANCE_TYPE,
            "instance_profile": native.EXPECTED_INSTANCE_PROFILE,
        },
        "gateway": {
            "source_env_file": str(ROOT / "inputs/gateway.env"),
            "kms_key_id": KMS_KEY,
            "release_manifest": str(ROOT / "gateway-release.json"),
            "release_lineage": str(ROOT / "gateway-lineage.json"),
            "eif_root": str(ROOT / "tee"),
            "config_dir": str(ROOT / "v2-config"),
            "artifact_policy": str(ROOT / "artifact-policy.json"),
            "protected_workflow_manifest": str(
                repository / "gateway/_attested_runtime/protected_workflows.json"
            ),
        },
        "validator": {
            "source_env_file": str(ROOT / "inputs/validator.env"),
            "release_manifest": str(ROOT / "validator-release.json"),
            "eif_path": str(repository / "validator_tee/validator-enclave.eif"),
            "hotkey_config": str(ROOT / "inputs/testnet-hotkey-config.json"),
            "hotkey_envelope": str(ROOT / "inputs/testnet-hotkey-envelope.json"),
            "chain_profile": str(
                repository / "validator_tee/enclave/chain_signing_profile_test_v2.json"
            ),
            "cutover_manifest": str(ROOT / "inputs/testnet401-epoch-cutover.json"),
            "wallet_path": str(ROOT / "wallets"),
            "wallet_name": "validator",
            "wallet_hotkey": "default",
            "enclave_cid": native.VALIDATOR_CID,
            "expected_hotkey": native.EXPECTED_VALIDATOR_HOTKEY,
            "expected_selected_profile_hash": native.EXPECTED_PROFILE_HASH,
        },
    }


def stage(*, candidate: str, run_id: str, instance_id: str,
          assets_bucket: str, assets_prefix: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", candidate):
        raise ValueError("invalid candidate")
    if not re.fullmatch(r"pp-[0-9]{1,20}-[0-9]{1,6}", run_id):
        raise ValueError("invalid run ID")
    if not re.fullmatch(r"i-[0-9a-f]{8,17}", instance_id):
        raise ValueError("invalid instance ID")
    if not re.fullmatch(r"leadpoet-parity-493765492819-[a-z0-9-]+", assets_bucket):
        raise ValueError("invalid task bucket")
    if assets_prefix != f"production-parity/runs/{run_id}/testnet401":
        raise ValueError("invalid task asset prefix")
    repository = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    if head != candidate:
        raise ValueError("candidate differs from checkout")
    native._private_regular_file(ROOT / "expires-epoch", "expiry")
    expiry = int((ROOT / "expires-epoch").read_text(encoding="ascii").strip())
    config = build_config(repository=repository, candidate=candidate,
                          run_id=run_id, instance_id=instance_id, expiry=expiry)
    native.verify_host_authority(config)
    native.verify_host_is_empty(config)
    inputs = ROOT / "inputs"
    inputs.mkdir(mode=0o700, exist_ok=False)
    logs = ROOT / "staging-logs"
    logs.mkdir(mode=0o700, exist_ok=False)
    s3 = boto3.client("s3", region_name=native.EXPECTED_AWS_REGION)
    for name in INPUT_NAMES:
        payload = read_locked_private_input(
            s3,
            bucket=assets_bucket,
            key=f"{assets_prefix}/{name}",
        )
        private_write(inputs / name, payload)
        del payload
    secrets = boto3.client("secretsmanager", region_name=native.EXPECTED_AWS_REGION)
    raw = secrets.get_secret_value(SecretId="leadpoet/prod/gateway/env")["SecretString"]
    if not isinstance(raw, str) or not 0 < len(raw.encode()) <= 262144:
        raise ValueError("gateway secret is outside size limit")
    private_write(inputs / "gateway.env", raw.encode())
    del raw
    policy = {
        "schema_version": "leadpoet.encrypted_artifact_policy.v2",
        "bucket_host": f"{assets_bucket}.s3.us-east-1.amazonaws.com",
        "key_prefix": "/encrypted-artifacts/",
        "minimum_retention_days": 1,
    }
    private_write(ROOT / "artifact-policy.json", json.dumps(policy).encode())
    nitro_environment = prepare_nitro_cli_environment()
    env = {
        key: value for key, value in os.environ.items()
        if key not in native.STATIC_AWS_CREDENTIAL_NAMES
    }
    env.update({
        "PATH": f"{Path(sys.executable).parent}:{env.get('PATH', '/usr/bin:/bin')}",
        "PYTHONPATH": str(repository),
        "AWS_REGION": native.EXPECTED_AWS_REGION,
        "AWS_DEFAULT_REGION": native.EXPECTED_AWS_REGION,
        "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true",
        "GATEWAY_ROOT": str(repository / "gateway"),
        "GATEWAY_DEPLOY_COMMIT": candidate,
        "GATEWAY_TEE_EIF_ROOT": str(ROOT / "tee"),
        "GATEWAY_V2_RELEASE_MANIFEST": config["gateway"]["release_manifest"],
        "GATEWAY_V2_RELEASE_ARCHIVE_ROOT": str(ROOT / "release-archive"),
        "GATEWAY_LAST_GOOD_MANIFEST": str(ROOT / "no-prior-gateway.json"),
        "GATEWAY_TEE_TOPOLOGY_MODE": "full",
        "GATEWAY_V2_BUILD_WORK_ROOT": str(ROOT / "gateway-build"),
        "VALIDATOR_V2_BUILD_WORK_ROOT": str(ROOT / "validator-build"),
        "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT": str(ROOT / "offline-artifacts"),
        "VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT": str(ROOT / "offline-artifacts/validator-runtime"),
        "VALIDATOR_V2_BUILD_COMMIT": candidate,
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(ROOT / "docker.lock"),
        **nitro_environment,
    })

    def run(name: str, command: list[str], *, timeout: int = 3600) -> None:
        print(json.dumps({"stage": name, "status": "running"}), flush=True)
        descriptor = os.open(logs / f"{name}.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as log:
            result = subprocess.run(command, cwd=repository, env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        if result.returncode:
            raise RuntimeError(f"native stage failed: {name}")
        print(json.dumps({"stage": name, "status": "passed"}), flush=True)

    run("native_host_dependencies", [
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-input", "--no-cache-dir", "--requirement", "requirements.txt",
    ], timeout=1200)
    run("native_dependency_check", [sys.executable, "-m", "pip", "check"], timeout=60)
    run("offline_artifacts", ["bash", "gateway/tee/prepare_offline_artifacts_v2.sh"])
    run("local_runtime_identities", [
        "bash", "gateway/tee/build_local_release_v2.sh", "--repository", str(repository),
        "--revision", candidate, "--gateway-output", config["gateway"]["release_manifest"],
        "--validator-output", config["validator"]["release_manifest"],
    ], timeout=7200)
    run("gateway_role_images", ["bash", "gateway/tee/build_role_enclaves.sh"], timeout=7200)
    run("validator_image", ["bash", "validator_tee/scripts/build_enclave.sh"], timeout=3600)
    run("gateway_artifact_verification", [
        sys.executable, "-m", "gateway.tee.verify_release_artifacts_v2",
        "--release-manifest", config["gateway"]["release_manifest"],
        "--gateway-root", str(repository / "gateway"),
        "--eif-root", config["gateway"]["eif_root"],
    ])
    run("validator_artifact_verification", [
        sys.executable, "-m", "validator_tee.host.verify_release_gate_v2",
        "--verify-manifest", config["validator"]["release_manifest"],
        "--local-release", str(repository / "validator_tee/validator-v2-release.json"),
    ])
    from gateway.tee.release_channel_v2 import build_release_channel_v2, build_release_lineage_v2

    gateway_release = json.loads(Path(config["gateway"]["release_manifest"]).read_text())
    validator_release = json.loads(Path(config["validator"]["release_manifest"]).read_text())
    channel = build_release_channel_v2(gateway_release_manifest=gateway_release,
                                      validator_release_manifest=validator_release)
    lineage = build_release_lineage_v2([channel], current_commit=candidate)
    private_write(ROOT / "gateway-lineage.json", json.dumps(lineage).encode())
    private_write(ROOT / "config.json", json.dumps(config).encode())
    checked = native.load_config(ROOT / "config.json")
    native.validate_static_inputs(checked)
    return {"status": "staged", "candidate_sha": candidate, "instance_id": instance_id}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--assets-bucket", required=True)
    parser.add_argument("--assets-prefix", required=True)
    args = parser.parse_args()
    try:
        result = stage(candidate=args.candidate_sha, run_id=args.run_id,
                       instance_id=args.instance_id, assets_bucket=args.assets_bucket,
                       assets_prefix=args.assets_prefix)
    except Exception as exc:
        print(json.dumps(failure_receipt(exc), sort_keys=True), flush=True)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
