#!/usr/bin/env python3
"""Provision and destroy one candidate-bound physical parity stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_canonical.production_parity_wallet import (  # noqa: E402
    ProductionParityWalletError,
    install_base,
    normalize_spec,
    validate_head,
)
from leadpoet_canonical.production_parity_epoch_authority import (  # noqa: E402
    ProductionParityEpochAuthorityError,
    install_base as epoch_authority_install_base,
    normalize_spec as normalize_epoch_authority_spec,
    validate_head as validate_epoch_authority_head,
)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:@+=,-]+$")
INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")
INSTANCE_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,31}$")
REPO_URL_RE = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$"
)
SCHEMA_VERSION = "leadpoet.production_parity_infra.v1"


class ProvisioningError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProvisioningError("parity infrastructure config is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ProvisioningError("parity infrastructure config schema differs")
    return value


def _run(command: Sequence[str], *, timeout: int) -> str:
    result = subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise ProvisioningError(f"command failed: {detail}")
    return result.stdout


def _aws(region: str, *args: str, timeout: int = 300) -> str:
    return _run(["aws", "--region", region, *args], timeout=timeout)


def _public_cidr() -> str:
    with urlopen("https://checkip.amazonaws.com", timeout=15) as response:
        value = response.read().decode("ascii").strip()
    if not re.fullmatch(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", value):
        raise ProvisioningError("controller public address is invalid")
    return value + "/32"


def _require_config_mapping(config: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = config.get(field)
    if not isinstance(value, Mapping):
        raise ProvisioningError(f"{field} must be an object")
    return value


def _parameter_values(config: Mapping[str, Any]) -> dict[str, str]:
    parameters = _require_config_mapping(config, "cloudformation_parameters")
    derived = {
        "GatewayImageId",
        "ValidatorImageId",
        "AuditorImageId",
        "GatewayInstanceType",
        "ValidatorInstanceType",
        "AuditorInstanceType",
    }
    supplied_derived = sorted(derived & set(parameters))
    if supplied_derived:
        raise ProvisioningError(
            "production-derived CloudFormation parameters must not be configured: "
            + ",".join(supplied_derived)
        )
    required = {
        "VpcId",
        "SubnetId",
        "DatabaseImageId",
        "DashboardImageId",
        "DatabaseInstanceType",
        "DashboardInstanceType",
        "GatewayInstanceProfile",
        "ValidatorInstanceProfile",
        "AuditorInstanceProfile",
        "DatabaseInstanceProfile",
        "DashboardInstanceProfile",
        "DatabaseCertificateArn",
        "HostedZoneId",
        "BaseDomain",
    }
    missing = sorted(required - set(parameters))
    if missing:
        raise ProvisioningError("missing CloudFormation parameters: " + ",".join(missing))
    result = {str(key): str(value) for key, value in parameters.items()}
    if any(not SAFE_VALUE_RE.fullmatch(value) for value in result.values()):
        raise ProvisioningError("CloudFormation parameter contains an unsafe value")
    return result


def _production_reference(
    config: Mapping[str, Any], *, role: str
) -> Mapping[str, Any]:
    references = _require_config_mapping(config, "production_references")
    value = _require_config_mapping(references, role)
    instance_id = str(value.get("instance_id") or "").strip()
    public_ip = str(value.get("public_ip") or "").strip()
    if bool(instance_id) == bool(public_ip):
        raise ProvisioningError(
            f"{role} production reference requires exactly one instance_id or public_ip"
        )
    if instance_id and not INSTANCE_ID_RE.fullmatch(instance_id):
        raise ProvisioningError(f"{role} production instance ID is invalid")
    if public_ip and not re.fullmatch(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", public_ip):
        raise ProvisioningError(f"{role} production public IP is invalid")
    return {"instance_id": instance_id, "public_ip": public_ip}


def _describe_reference_instance(
    region: str, *, role: str, reference: Mapping[str, Any]
) -> dict[str, str]:
    selector: list[str]
    if reference.get("instance_id"):
        selector = ["--instance-ids", str(reference["instance_id"])]
    else:
        selector = [
            "--filters",
            f"Name=ip-address,Values={reference['public_ip']}",
        ]
    raw = _aws(
        region,
        "ec2",
        "describe-instances",
        *selector,
        "--output",
        "json",
    )
    try:
        document = json.loads(raw)
        instances = [
            instance
            for reservation in document.get("Reservations", [])
            if isinstance(reservation, Mapping)
            for instance in reservation.get("Instances", [])
            if isinstance(instance, Mapping)
        ]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProvisioningError(
            f"{role} production instance evidence is invalid"
        ) from exc
    if len(instances) != 1:
        raise ProvisioningError(
            f"{role} production reference did not resolve exactly one instance"
        )
    value = instances[0]
    instance_id = str(value.get("InstanceId") or "")
    image_id = str(value.get("ImageId") or "")
    instance_type = str(value.get("InstanceType") or "")
    architecture = str(value.get("Architecture") or "")
    state = value.get("State")
    enclave_options = value.get("EnclaveOptions")
    metadata_options = value.get("MetadataOptions")
    if (
        not INSTANCE_ID_RE.fullmatch(instance_id)
        or not AMI_RE.fullmatch(image_id)
        or not INSTANCE_TYPE_RE.fullmatch(instance_type)
        or architecture not in {"x86_64", "arm64"}
        or not isinstance(state, Mapping)
        or state.get("Name") != "running"
        or not isinstance(enclave_options, Mapping)
        or enclave_options.get("Enabled") is not True
        or not isinstance(metadata_options, Mapping)
        or metadata_options.get("HttpTokens") != "required"
    ):
        raise ProvisioningError(
            f"{role} production instance is not a running IMDSv2/Nitro reference"
        )
    return {
        "instance_id": instance_id,
        "image_id": image_id,
        "instance_type": instance_type,
        "architecture": architecture,
    }


def _wallet_artifacts(
    config: Mapping[str, Any], *, region: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    settings = _require_config_mapping(config, "wallet_artifacts")
    runtime = _require_config_mapping(config, "runtime_metadata")
    network = str(runtime.get("network") or "").strip().lower()
    try:
        netuid = int(runtime.get("netuid"))
    except (TypeError, ValueError) as exc:
        raise ProvisioningError("staging wallet netuid is invalid") from exc
    raw_auditors = settings.get("audit_validators")
    if not isinstance(raw_auditors, list) or len(raw_auditors) != 2:
        raise ProvisioningError("exactly two audit wallet artifacts are required")
    raw_specs = {
        "primary-validator": _require_config_mapping(
            settings, "primary_validator"
        ),
        "auditor-a": _require_config_mapping(
            {"auditor-a": raw_auditors[0]}, "auditor-a"
        ),
        "auditor-b": _require_config_mapping(
            {"auditor-b": raw_auditors[1]}, "auditor-b"
        ),
    }
    specs: dict[str, dict[str, Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    try:
        for role, value in raw_specs.items():
            spec = normalize_spec(value, role=role, network=network, netuid=netuid)
            head = json.loads(
                _aws(
                    region,
                    "s3api",
                    "head-object",
                    "--bucket",
                    spec["bucket"],
                    "--key",
                    spec["key"],
                    "--version-id",
                    spec["version_id"],
                    "--output",
                    "json",
                )
            )
            specs[role] = spec
            evidence[role] = validate_head(spec, head)
    except (ProductionParityWalletError, TypeError, ValueError) as exc:
        raise ProvisioningError("staging wallet artifact validation failed") from exc
    hotkeys = [spec["expected_hotkey"] for spec in specs.values()]
    if len(set(hotkeys)) != 3:
        raise ProvisioningError("staging wallet hotkeys must be distinct")
    return specs, evidence


def _epoch_authority_artifact(
    config: Mapping[str, Any], *, region: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = _require_config_mapping(config, "runtime_metadata")
    value = _require_config_mapping(config, "epoch_authority_artifact")
    network = str(runtime.get("network") or "").strip().lower()
    try:
        netuid = int(runtime.get("netuid"))
        spec = normalize_epoch_authority_spec(
            value,
            network=network,
            netuid=netuid,
        )
        head = json.loads(
            _aws(
                region,
                "s3api",
                "head-object",
                "--bucket",
                spec["bucket"],
                "--key",
                spec["key"],
                "--version-id",
                spec["version_id"],
                "--output",
                "json",
            )
        )
        evidence = validate_epoch_authority_head(spec, head)
    except (
        ProductionParityEpochAuthorityError,
        TypeError,
        ValueError,
    ) as exc:
        raise ProvisioningError(
            "testnet epoch authority artifact validation failed"
        ) from exc
    return spec, evidence


def _stack_outputs(region: str, stack_name: str) -> dict[str, str]:
    raw = _aws(
        region,
        "cloudformation",
        "describe-stacks",
        "--stack-name",
        stack_name,
        "--query",
        "Stacks[0].Outputs",
        "--output",
        "json",
    )
    values = json.loads(raw)
    if not isinstance(values, list):
        raise ProvisioningError("CloudFormation outputs are invalid")
    return {
        str(item["OutputKey"]): str(item["OutputValue"])
        for item in values
        if isinstance(item, Mapping) and item.get("OutputKey") and item.get("OutputValue")
    }


def _resolve_remote_main(repository_url: str) -> str:
    if not REPO_URL_RE.fullmatch(repository_url):
        raise ProvisioningError("dashboard repository URL is invalid")
    output = _run(
        ["git", "ls-remote", repository_url, "refs/heads/main"], timeout=60
    )
    fields = output.strip().split()
    if len(fields) != 2 or fields[1] != "refs/heads/main" or not SHA_RE.fullmatch(fields[0]):
        raise ProvisioningError("dashboard main commit could not be frozen")
    return fields[0]


def _cleanup_failed_stack(
    *,
    region: str,
    stack_name: str,
    key_name: str,
    key_path: Path,
    public_path: Path,
) -> None:
    try:
        try:
            _aws(region, "cloudformation", "delete-stack", "--stack-name", stack_name)
            _aws(
                region,
                "cloudformation",
                "wait",
                "stack-delete-complete",
                "--stack-name",
                stack_name,
                timeout=1800,
            )
        except Exception:
            pass
    finally:
        try:
            try:
                _aws(region, "ec2", "delete-key-pair", "--key-name", key_name)
            except Exception:
                pass
        finally:
            key_path.unlink(missing_ok=True)
            public_path.unlink(missing_ok=True)


def _verify_encrypted_instance_volumes(
    region: str,
    instance_ids: Sequence[str],
) -> dict[str, list[str]]:
    evidence: dict[str, list[str]] = {}
    for instance_id in instance_ids:
        raw = _aws(
            region,
            "ec2",
            "describe-volumes",
            "--filters",
            f"Name=attachment.instance-id,Values={instance_id}",
            "--query",
            "Volumes[].{VolumeId:VolumeId,Encrypted:Encrypted}",
            "--output",
            "json",
        )
        values = json.loads(raw)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, Mapping)
                or value.get("Encrypted") is not True
                or not str(value.get("VolumeId") or "").startswith("vol-")
                for value in values
            )
        ):
            raise ProvisioningError(
                f"ephemeral instance storage is not fully encrypted: {instance_id}"
            )
        evidence[instance_id] = [str(value["VolumeId"]) for value in values]
    return evidence


def _provision(
    *,
    config: Mapping[str, Any],
    candidate_sha: str,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    if not SHA_RE.fullmatch(candidate_sha) or not RUN_RE.fullmatch(run_id):
        raise ProvisioningError("candidate SHA or run ID is invalid")
    region = str(config.get("region") or "").strip()
    if not re.fullmatch(r"[a-z]{2}-[a-z]+-[0-9]", region):
        raise ProvisioningError("AWS region is invalid")
    stack_name = f"leadpoet-parity-{run_id}"
    key_name = stack_name
    metadata = _require_config_mapping(config, "runtime_metadata")
    if str(metadata.get("gateway_public_url_template") or "") != "https://{gateway_domain}":
        raise ProvisioningError(
            "gateway public URL template must use the run-scoped TLS domain"
        )
    dashboard_template = str(metadata.get("dashboard_report_url_template") or "")
    if dashboard_template != "https://{dashboard_domain}/api/research-lab":
        raise ProvisioningError(
            "dashboard URL template must use the real run-scoped dashboard API"
        )
    dashboard_repository_url = str(
        metadata.get("dashboard_repository_url") or ""
    )
    dashboard_source_sha = _resolve_remote_main(dashboard_repository_url)
    wallet_specs, wallet_evidence = _wallet_artifacts(config, region=region)
    epoch_authority_spec, epoch_authority_evidence = _epoch_authority_artifact(
        config,
        region=region,
    )
    auditor_hotkeys = [
        wallet_specs["auditor-a"]["expected_hotkey"],
        wallet_specs["auditor-b"]["expected_hotkey"],
    ]
    parameters = _parameter_values(config)
    gateway_reference = _describe_reference_instance(
        region,
        role="gateway",
        reference=_production_reference(config, role="gateway"),
    )
    validator_reference = _describe_reference_instance(
        region,
        role="validator",
        reference=_production_reference(config, role="validator"),
    )
    if gateway_reference["instance_id"] == validator_reference["instance_id"]:
        raise ProvisioningError("gateway and validator production references must differ")
    parameters.update(
        {
            "GatewayImageId": gateway_reference["image_id"],
            "GatewayInstanceType": gateway_reference["instance_type"],
            "ValidatorImageId": validator_reference["image_id"],
            "ValidatorInstanceType": validator_reference["instance_type"],
            "AuditorImageId": validator_reference["image_id"],
            "AuditorInstanceType": validator_reference["instance_type"],
        }
    )
    allowed_ssh_cidr = _public_cidr()
    output_dir.mkdir(parents=True, exist_ok=True)
    key_path = output_dir / "controller.pem"
    public_path = Path(str(key_path) + ".pub")
    if key_path.exists() or public_path.exists():
        raise ProvisioningError("ephemeral controller key already exists")
    _run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            stack_name,
            "-f",
            str(key_path),
        ],
        timeout=30,
    )
    key_path.chmod(0o600)
    _aws(
        region,
        "ec2",
        "import-key-pair",
        "--key-name",
        key_name,
        "--public-key-material",
        "fileb://" + str(public_path),
        "--tag-specifications",
        (
            "ResourceType=key-pair,Tags=["
            f"{{Key=leadpoet:parity-run,Value={run_id}}},"
            f"{{Key=leadpoet:candidate-sha,Value={candidate_sha}}}]"
        ),
    )
    parameters.update(
        {
            "RunId": run_id,
            "CandidateSha": candidate_sha,
            "AllowedSshCidr": allowed_ssh_cidr,
            "KeyName": key_name,
        }
    )
    parameter_args = [f"{key}={value}" for key, value in sorted(parameters.items())]
    try:
        _aws(
            region,
            "cloudformation",
            "deploy",
            "--stack-name",
            stack_name,
            "--template-file",
            str(ROOT / "infra/production-parity-staging.yml"),
            "--capabilities",
            "CAPABILITY_NAMED_IAM",
            "--no-fail-on-empty-changeset",
            "--tags",
            f"leadpoet:parity-run={run_id}",
            f"leadpoet:candidate-sha={candidate_sha}",
            "--parameter-overrides",
            *parameter_args,
            timeout=1800,
        )
        outputs = _stack_outputs(region, stack_name)
    except Exception:
        _cleanup_failed_stack(
            region=region,
            stack_name=stack_name,
            key_name=key_name,
            key_path=key_path,
            public_path=public_path,
        )
        raise
    required_outputs = {
        "GatewayPublicIp",
        "ValidatorPublicIp",
        "AuditorAPublicIp",
        "AuditorBPublicIp",
        "DatabasePublicIp",
        "DatabasePrivateIp",
        "DashboardPublicIp",
        "GatewayDomain",
        "DatabaseDomain",
        "DashboardDomain",
        "GatewayInstanceId",
        "ValidatorInstanceId",
        "AuditorAInstanceId",
        "AuditorBInstanceId",
        "DatabaseInstanceId",
        "DashboardInstanceId",
        "GatewayWebSecurityGroupId",
        "DatabaseWebSecurityGroupId",
        "DatabaseServiceSecurityGroupId",
        "DashboardWebSecurityGroupId",
    }
    if required_outputs - set(outputs):
        _cleanup_failed_stack(
            region=region,
            stack_name=stack_name,
            key_name=key_name,
            key_path=key_path,
            public_path=public_path,
        )
        raise ProvisioningError("ephemeral stack outputs are incomplete")
    try:
        encrypted_volumes = _verify_encrypted_instance_volumes(
            region,
            [
                outputs["GatewayInstanceId"],
                outputs["ValidatorInstanceId"],
                outputs["AuditorAInstanceId"],
                outputs["AuditorBInstanceId"],
                outputs["DatabaseInstanceId"],
                outputs["DashboardInstanceId"],
            ],
        )
    except Exception:
        _cleanup_failed_stack(
            region=region,
            stack_name=stack_name,
            key_name=key_name,
            key_path=key_path,
            public_path=public_path,
        )
        raise
    replacements = {
        "{gateway_ip}": outputs["GatewayPublicIp"],
        "{database_ip}": outputs["DatabasePublicIp"],
        "{gateway_domain}": outputs["GatewayDomain"],
        "{database_domain}": outputs["DatabaseDomain"],
        "{dashboard_domain}": outputs["DashboardDomain"],
    }
    gateway_url = str(metadata.get("gateway_public_url_template") or "")
    dashboard_url = str(metadata.get("dashboard_report_url_template") or "")
    for marker, replacement in replacements.items():
        gateway_url = gateway_url.replace(marker, replacement)
        dashboard_url = dashboard_url.replace(marker, replacement)
    if gateway_url != f"https://{outputs['GatewayDomain']}":
        raise ProvisioningError("gateway public URL must use the run-scoped TLS domain")
    ssh_user = str(metadata.get("ssh_user") or "ec2-user")
    secret_prefix = f"leadpoet/staging/production-parity/{run_id}"
    physical_config = {
        "schema_version": "leadpoet.physical_v2_staging_config.v2",
        "environment": "production-parity-ephemeral",
        "ephemeral_stack_id": run_id,
        "network": "test",
        "netuid": int(metadata.get("netuid") or 1),
        "chain_endpoint": str(metadata.get("testnet_chain_endpoint") or ""),
        "network_genesis_hash": str(
            epoch_authority_spec["network_genesis_hash"]
        ),
        "gateway_public_url": gateway_url,
        "dashboard_report_url": dashboard_url,
        "dashboard_source_sha": dashboard_source_sha,
        "timeout_seconds": int(metadata.get("timeout_seconds") or 14400),
        "rebenchmark_timeout_seconds": int(
            metadata.get("rebenchmark_timeout_seconds") or 14400
        ),
        "poll_seconds": int(metadata.get("poll_seconds") or 10),
        "required_consecutive_epochs": 3,
        "gateway": {
            "ssh_host": f"{ssh_user}@{outputs['GatewayPublicIp']}",
            "ssh_key": str(key_path),
            "restart_path": str(metadata.get("gateway_restart_path") or "/home/ec2-user/gw_restart.sh"),
            "secret_id": secret_prefix + "/gateway",
            "repo_root": str(
                metadata.get("gateway_repo_root")
                or "/home/ec2-user/leadpoet_repo"
            ),
            "python_bin": str(
                metadata.get("gateway_python_bin")
                or "/home/ec2-user/venv311/bin/python3"
            ),
        },
        "primary_validator": {
            "ssh_host": f"{ssh_user}@{outputs['ValidatorPublicIp']}",
            "ssh_key": str(key_path),
            "restart_path": str(metadata.get("validator_restart_path") or "/home/ec2-user/leadpoet/leadpoet/validator_restart.sh"),
            "secret_id": secret_prefix + "/validator",
            "repo_root": str(metadata.get("validator_repo_root") or "/home/ec2-user/leadpoet/leadpoet"),
            "container_name": str(metadata.get("validator_container_name") or "leadpoet-validator-main"),
            "expected_hotkey": wallet_specs["primary-validator"][
                "expected_hotkey"
            ],
        },
        "audit_validators": [
            {
                "ssh_host": f"{ssh_user}@{outputs['AuditorAPublicIp']}",
                "ssh_key": str(key_path),
                "repo_root": str(metadata.get("auditor_repo_root") or "/home/ec2-user/leadpoet/leadpoet"),
                "unit_name": str(metadata.get("auditor_a_unit") or "leadpoet-auditor-a.service"),
                "expected_hotkey": str(auditor_hotkeys[0]),
                "secret_id": secret_prefix + "/auditor-a",
            },
            {
                "ssh_host": f"{ssh_user}@{outputs['AuditorBPublicIp']}",
                "ssh_key": str(key_path),
                "repo_root": str(metadata.get("auditor_repo_root") or "/home/ec2-user/leadpoet/leadpoet"),
                "unit_name": str(metadata.get("auditor_b_unit") or "leadpoet-auditor-b.service"),
                "expected_hotkey": str(auditor_hotkeys[1]),
                "secret_id": secret_prefix + "/auditor-b",
            },
        ],
        "database": {
            "ssh_host": f"{ssh_user}@{outputs['DatabasePublicIp']}",
            "private_ip": outputs.get("DatabasePrivateIp", ""),
            "public_domain": outputs["DatabaseDomain"],
            "secret_id": secret_prefix + "/database",
        },
        "dashboard": {
            "ssh_host": f"{ssh_user}@{outputs['DashboardPublicIp']}",
            "ssh_key": str(key_path),
            "public_domain": outputs["DashboardDomain"],
            "secret_id": secret_prefix + "/dashboard",
            "source_sha": dashboard_source_sha,
            "repo_root": str(
                metadata.get("dashboard_repo_root")
                or "/home/ec2-user/subnet_dashboard"
            ),
            "unit_name": str(
                metadata.get("dashboard_unit")
                or "leadpoet-parity-dashboard.service"
            ),
        },
    }
    config_path = output_dir / "physical-staging.json"
    state = {
        "schema_version": "leadpoet.production_parity_stack_state.v1",
        "candidate_sha": candidate_sha,
        "run_id": run_id,
        "region": region,
        "stack_name": stack_name,
        "key_name": key_name,
        "key_path": str(key_path),
        "config_path": str(config_path),
        "outputs": outputs,
        "encrypted_volumes": encrypted_volumes,
        "dashboard_source_sha": dashboard_source_sha,
        "dashboard_repository_url": dashboard_repository_url,
        "production_references": {
            "gateway": gateway_reference,
            "validator": validator_reference,
            "auditors_derive_from": "validator",
        },
        "wallet_artifacts": {
            role: {
                **spec,
                "install_base": str(install_base(run_id, role)),
                "immutability": wallet_evidence[role],
            }
            for role, spec in wallet_specs.items()
        },
        "epoch_authority_artifact": {
            **epoch_authority_spec,
            "install_base": str(epoch_authority_install_base(run_id)),
            "immutability": epoch_authority_evidence,
        },
    }
    state_path = output_dir / "stack-state.json"
    try:
        config_path.write_text(
            json.dumps(physical_config, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        config_path.chmod(0o600)
        state_path.write_text(
            json.dumps(state, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        state_path.chmod(0o600)
    except Exception:
        _cleanup_failed_stack(
            region=region,
            stack_name=stack_name,
            key_name=key_name,
            key_path=key_path,
            public_path=public_path,
        )
        raise
    return state


def provision(
    *,
    config: Mapping[str, Any],
    candidate_sha: str,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Provision atomically; every exception removes the deterministic stack/key."""

    try:
        return _provision(
            config=config,
            candidate_sha=candidate_sha,
            run_id=run_id,
            output_dir=output_dir,
        )
    except Exception:
        region = str(config.get("region") or "").strip()
        key_path = output_dir / "controller.pem"
        public_path = Path(str(key_path) + ".pub")
        if (
            RUN_RE.fullmatch(run_id)
            and re.fullmatch(r"[a-z]{2}-[a-z]+-[0-9]", region)
            and (key_path.exists() or public_path.exists())
        ):
            stack_name = f"leadpoet-parity-{run_id}"
            _cleanup_failed_stack(
                region=region,
                stack_name=stack_name,
                key_name=stack_name,
                key_path=key_path,
                public_path=public_path,
            )
        raise


def destroy(*, state: Mapping[str, Any]) -> dict[str, Any]:
    region = str(state.get("region") or "")
    stack_name = str(state.get("stack_name") or "")
    key_name = str(state.get("key_name") or "")
    run_id = str(state.get("run_id") or "")
    if not RUN_RE.fullmatch(run_id) or stack_name != f"leadpoet-parity-{run_id}" or key_name != stack_name:
        raise ProvisioningError("ephemeral stack state is invalid")
    _aws(region, "cloudformation", "delete-stack", "--stack-name", stack_name)
    _aws(
        region,
        "cloudformation",
        "wait",
        "stack-delete-complete",
        "--stack-name",
        stack_name,
        timeout=1800,
    )
    _aws(region, "ec2", "delete-key-pair", "--key-name", key_name)
    for path_value in (state.get("key_path"), str(state.get("key_path") or "") + ".pub"):
        path = Path(str(path_value or ""))
        if path.is_file():
            path.unlink()
    return {"run_id": run_id, "stack_deleted": True, "key_deleted": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    up = subparsers.add_parser("up")
    up.add_argument("--config", type=Path, required=True)
    up.add_argument("--candidate-sha", required=True)
    up.add_argument("--run-id", required=True)
    up.add_argument("--output-dir", type=Path, required=True)
    down = subparsers.add_parser("down")
    down.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "up":
            result = provision(
                config=_load(args.config),
                candidate_sha=str(args.candidate_sha).lower(),
                run_id=str(args.run_id).lower(),
                output_dir=args.output_dir,
            )
        else:
            result = destroy(state=_load_state(args.state))
    except (OSError, ValueError, ProvisioningError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


def _load_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProvisioningError("ephemeral stack state is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "leadpoet.production_parity_stack_state.v1":
        raise ProvisioningError("ephemeral stack state schema differs")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
