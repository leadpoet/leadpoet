#!/usr/bin/env python3
"""Prepare disposable hosts and the production-shaped staging database."""

from __future__ import annotations

import argparse
import base64
import ipaddress
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_canonical.production_parity import (  # noqa: E402
    ProductionParityError,
    file_sha256,
    validate_snapshot_manifest,
    verify_contract_checkout,
)
from scripts.run_physical_v2_staging import (  # noqa: E402
    PhysicalStagingError,
    load_config,
)


RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REPO_URL_RE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$")
IMAGE_RE = re.compile(r"^[A-Za-z0-9._/:@-]+@sha256:[0-9a-f]{64}$")
DATABASE_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
)
SECURITY_GROUP_RE = re.compile(r"^sg-[0-9a-f]{8,17}$")
INSTANCE_ID_RE = re.compile(r"^i-[0-9a-f]{8,17}$")
VERSION_RE = re.compile(r"^[A-Za-z0-9_.=+-]{1,1024}$")


class BootstrapError(RuntimeError):
    pass


def _load(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BootstrapError(f"{field} is unreadable") from exc
    if not isinstance(value, dict):
        raise BootstrapError(f"{field} must be an object")
    return value


def _run(command: Sequence[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _require(result: subprocess.CompletedProcess[str], *, stage: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-800:]
        raise BootstrapError(f"{stage} failed: {detail}")
    return result.stdout


def _verify_ephemeral_instances(
    *,
    region: str,
    outputs: Mapping[str, Any],
    run_id: str,
    candidate_sha: str,
) -> dict[str, Any]:
    expected_roles = {
        "GatewayInstanceId": "gateway",
        "ValidatorInstanceId": "primary-validator",
        "AuditorAInstanceId": "audit-validator-a",
        "AuditorBInstanceId": "audit-validator-b",
        "DatabaseInstanceId": "database",
        "DashboardInstanceId": "dashboard",
    }
    instance_ids = {
        field: str(outputs.get(field) or "") for field in expected_roles
    }
    if (
        not re.fullmatch(r"[a-z]{2}-[a-z]+-[0-9]", region)
        or not RUN_RE.fullmatch(run_id)
        or not SHA_RE.fullmatch(candidate_sha)
        or any(not INSTANCE_ID_RE.fullmatch(value) for value in instance_ids.values())
        or len(set(instance_ids.values())) != len(instance_ids)
    ):
        raise BootstrapError("ephemeral instance identity is invalid")
    raw = _require(
        _run(
            [
                "aws",
                "--region",
                region,
                "ec2",
                "describe-instances",
                "--instance-ids",
                *instance_ids.values(),
                "--output",
                "json",
            ],
            timeout=60,
        ),
        stage="ephemeral instance identity readback",
    )
    try:
        document = json.loads(raw)
        instances = {
            str(instance["InstanceId"]): instance
            for reservation in document.get("Reservations", [])
            if isinstance(reservation, Mapping)
            for instance in reservation.get("Instances", [])
            if isinstance(instance, Mapping) and instance.get("InstanceId")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise BootstrapError("ephemeral instance identity readback is invalid") from exc
    if set(instances) != set(instance_ids.values()):
        raise BootstrapError("ephemeral instance inventory differs")
    evidence: dict[str, Any] = {}
    for field, role in expected_roles.items():
        instance_id = instance_ids[field]
        instance = instances[instance_id]
        tags = {
            str(item.get("Key") or ""): str(item.get("Value") or "")
            for item in instance.get("Tags", [])
            if isinstance(item, Mapping)
        }
        metadata = instance.get("MetadataOptions")
        state = instance.get("State")
        if (
            not isinstance(state, Mapping)
            or state.get("Name") != "running"
            or not isinstance(metadata, Mapping)
            or metadata.get("HttpTokens") != "required"
            or metadata.get("InstanceMetadataTags") != "enabled"
            or tags.get("leadpoet:parity-run") != run_id
            or tags.get("leadpoet:parity-role") != role
            or tags.get("leadpoet:candidate-sha") != candidate_sha
        ):
            raise BootstrapError(f"ephemeral {role} instance identity differs")
        evidence[role] = {
            "instance_id": instance_id,
            "candidate_sha": candidate_sha,
            "run_id": run_id,
            "imds_v2_required": True,
            "instance_metadata_tags_enabled": True,
        }
    return evidence


def _ssh(host: str, key: Path, command: str, *, timeout: int) -> str:
    return _require(
        _run(
            [
                "ssh",
                "-n",
                "-i",
                str(key),
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=20",
                host,
                command,
            ],
            timeout=timeout,
        ),
        stage="staging SSH bootstrap",
    )


def _scp(host: str, key: Path, source: Path, destination: str, *, timeout: int) -> None:
    _require(
        _run(
            [
                "scp",
                "-q",
                "-i",
                str(key),
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                str(source),
                f"{host}:{destination}",
            ],
            timeout=timeout,
        ),
        stage="staging evidence copy",
    )


def _encoded_python(source: str) -> str:
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return f"python3 -c \"import base64;exec(base64.b64decode('{encoded}'))\""


def _is_database_private_address(
    value: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    return any(value in network for network in DATABASE_PRIVATE_NETWORKS)


def _verify_database_ingress(
    *,
    region: str,
    database_tls_security_group: str,
    database_service_security_group: str,
    gateway_security_group: str,
    dashboard_security_group: str,
) -> dict[str, Any]:
    groups = {
        database_tls_security_group,
        database_service_security_group,
        gateway_security_group,
        dashboard_security_group,
    }
    if (
        not re.fullmatch(r"[a-z]{2}-[a-z]+-[0-9]", region)
        or len(groups) != 4
        or any(not SECURITY_GROUP_RE.fullmatch(group) for group in groups)
    ):
        raise BootstrapError("database ingress boundary identity is invalid")
    aws = ["aws", "--region", region, "ec2"]

    def read_permissions(group_id: str) -> list[dict[str, Any]]:
        raw = _require(
            _run(
                [
                    *aws,
                    "describe-security-groups",
                    "--group-ids",
                    group_id,
                    "--query",
                    (
                        "SecurityGroups[0].IpPermissions[].{"
                        "IpProtocol:IpProtocol,FromPort:FromPort,ToPort:ToPort,"
                        "IpRanges:IpRanges[].CidrIp,Ipv6Ranges:Ipv6Ranges[].CidrIpv6,"
                        "PrefixListIds:PrefixListIds[].PrefixListId,"
                        "GroupIds:UserIdGroupPairs[].GroupId}"
                    ),
                    "--output",
                    "json",
                ],
                timeout=60,
            ),
            stage="database ingress readback",
        )
        try:
            permissions = json.loads(raw)
        except ValueError as exc:
            raise BootstrapError("database ingress readback is invalid") from exc
        if not isinstance(permissions, list):
            raise BootstrapError("database ingress readback is invalid")
        for permission in permissions:
            if not isinstance(permission, dict):
                raise BootstrapError("database ingress readback is invalid")
            if isinstance(permission.get("GroupIds"), list):
                permission["GroupIds"] = sorted(permission["GroupIds"])
        return permissions

    expected_tls = [
        {
            "FromPort": 443,
            "GroupIds": sorted(
                [gateway_security_group, dashboard_security_group]
            ),
            "IpProtocol": "tcp",
            "IpRanges": [],
            "Ipv6Ranges": [],
            "PrefixListIds": [],
            "ToPort": 443,
        }
    ]
    expected_service = [
        {
            "FromPort": 3000,
            "GroupIds": [database_tls_security_group],
            "IpProtocol": "tcp",
            "IpRanges": [],
            "Ipv6Ranges": [],
            "PrefixListIds": [],
            "ToPort": 3000,
        }
    ]
    if (
        read_permissions(database_tls_security_group) != expected_tls
        or read_permissions(database_service_security_group) != expected_service
    ):
        raise BootstrapError("database ingress differs from the private TLS boundary")
    return {
        "database_tls_security_group": database_tls_security_group,
        "database_service_security_group": database_service_security_group,
        "gateway_security_group": gateway_security_group,
        "dashboard_security_group": dashboard_security_group,
        "public_ingress_absent": True,
        "consumer_https_only": True,
    }


def _wait_private_database_tls(
    *, host: str, key: Path, database_domain: str
) -> dict[str, Any]:
    if not re.fullmatch(
        r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", database_domain
    ):
        raise BootstrapError("database TLS domain is invalid")
    program = f"""
import ipaddress
import json
import socket
import time
from urllib.error import HTTPError
from urllib.request import urlopen

domain = {database_domain!r}
private_networks = tuple(
    ipaddress.ip_network(value)
    for value in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', 'fc00::/7')
)
deadline = time.monotonic() + 300
last_error = 'unavailable'
while time.monotonic() < deadline:
    try:
        addresses = sorted({{
            item[4][0] for item in socket.getaddrinfo(domain, 443, type=socket.SOCK_STREAM)
        }})
        parsed = [ipaddress.ip_address(value) for value in addresses]
        if not parsed or any(
            not any(value in network for network in private_networks)
            for value in parsed
        ):
            raise RuntimeError('database DNS is not private')
        try:
            with urlopen('https://' + domain + '/', timeout=10) as response:
                status = int(response.status)
        except HTTPError as exc:
            status = int(exc.code)
        if status in {{200, 401, 403}}:
            print(json.dumps({{'addresses': addresses, 'http_status': status}}, sort_keys=True))
            raise SystemExit(0)
        last_error = 'unexpected HTTP status ' + str(status)
    except Exception as exc:
        last_error = type(exc).__name__ + ': ' + str(exc)
    time.sleep(3)
raise SystemExit('private database TLS endpoint unavailable: ' + last_error)
"""
    try:
        value = json.loads(
            _ssh(host, key, _encoded_python(program), timeout=360)
        )
    except ValueError as exc:
        raise BootstrapError(
            "private database TLS evidence is invalid"
        ) from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("addresses"), list):
        raise BootstrapError("private database TLS evidence is invalid")
    try:
        addresses = [
            ipaddress.ip_address(str(item)) for item in value.get("addresses", [])
        ]
    except ValueError as exc:
        raise BootstrapError("private database TLS evidence is invalid") from exc
    if (
        not addresses
        or any(
            not _is_database_private_address(address)
            for address in addresses
        )
        or int(value.get("http_status") or 0) not in {200, 401, 403}
    ):
        raise BootstrapError("private database TLS evidence is invalid")
    return {
        "database_domain": database_domain,
        "private_addresses": [str(address) for address in addresses],
        "http_status": int(value["http_status"]),
    }


def _prepare_repo_command(
    *,
    repo_url: str,
    repo_root: str,
    base_sha: str,
    host_launcher: str,
    launcher_relative: str,
) -> str:
    program = f"""
import pathlib
import shutil
import subprocess

repo_url = {repo_url!r}
repo_root = pathlib.Path({repo_root!r})
base_sha = {base_sha!r}
launcher = pathlib.Path({host_launcher!r})
if repo_root.exists():
    raise SystemExit("ephemeral application repository already exists")
repo_root.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["git", "clone", "--filter=blob:none", repo_url, str(repo_root)], check=True)
subprocess.run(["git", "-C", str(repo_root), "cat-file", "-e", base_sha + "^{{commit}}"], check=True)
subprocess.run(["git", "-C", str(repo_root), "checkout", "--detach", base_sha], check=True)
subprocess.run(["git", "-C", str(repo_root), "branch", "-f", "main", base_sha], check=True)
subprocess.run(["git", "-C", str(repo_root), "checkout", "main"], check=True)
subprocess.run(["git", "-C", str(repo_root), "diff", "--exit-code", base_sha, "--"], check=True)
source = repo_root / {launcher_relative!r}
if not source.is_file():
    raise SystemExit("base launcher is unavailable")
launcher.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, launcher)
launcher.chmod(0o700)
print(base_sha)
"""
    return _encoded_python(program)


def _install_wallet_artifact(
    *,
    host: str,
    key: Path,
    repo_root: str,
    run_id: str,
    region: str,
    role: str,
    spec: Mapping[str, Any],
    helper_root: str,
) -> dict[str, Any]:
    remote_spec = f"/tmp/leadpoet-parity-wallet-{run_id}-{role}.json"
    payload = base64.b64encode(
        (json.dumps(dict(spec), sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        )
    ).decode("ascii")
    write_program = f"""
import base64
import pathlib
path = pathlib.Path({remote_spec!r})
path.write_bytes(base64.b64decode({payload!r}, validate=True))
path.chmod(0o600)
"""
    output = _ssh(
        host,
        key,
        "\n".join(
            [
                "set -Eeuo pipefail",
                "umask 077",
                f"trap 'rm -f -- {remote_spec}' EXIT",
                _encoded_python(write_program),
                (
                    f"PYTHONPATH='{helper_root}:{repo_root}' python3 "
                    f"'{helper_root}/scripts/install_production_parity_wallet.py' "
                    f"--spec '{remote_spec}' --run-id '{run_id}' --region '{region}'"
                ),
            ]
        ),
        timeout=300,
    )
    try:
        value = json.loads(output.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise BootstrapError("staging wallet installer result is invalid") from exc
    if (
        not isinstance(value, Mapping)
        or value.get("role") != role
        or value.get("network") != "test"
        or value.get("expected_hotkey") != spec.get("expected_hotkey")
        or value.get("artifact_sha256") != spec.get("sha256")
        or value.get("artifact_version_id") != spec.get("version_id")
    ):
        raise BootstrapError("staging wallet installation evidence differs")
    return dict(value)


def _stage_candidate_artifact_helpers(
    *,
    host: str,
    key: Path,
    run_id: str,
) -> str:
    helper_root = f"/home/ec2-user/.local/lib/leadpoet-parity-{run_id}"
    files = (
        "leadpoet_canonical/__init__.py",
        "leadpoet_canonical/production_parity_wallet.py",
        "leadpoet_canonical/production_parity_epoch_authority.py",
        "scripts/install_production_parity_wallet.py",
        "scripts/install_production_parity_epoch_authority.py",
    )
    _ssh(
        host,
        key,
        (
            "set -Eeuo pipefail; umask 077; "
            f"test ! -e '{helper_root}'; "
            f"install -d -m 0700 '{helper_root}/leadpoet_canonical' "
            f"'{helper_root}/scripts'"
        ),
        timeout=60,
    )
    for relative in files:
        source = ROOT / relative
        if not source.is_file():
            raise BootstrapError(f"candidate staging helper is missing: {relative}")
        destination = f"{helper_root}/{relative}"
        _scp(host, key, source, destination, timeout=120)
        expected = file_sha256(source).split(":", 1)[1]
        observed = _ssh(
            host,
            key,
            f"sha256sum '{destination}' | cut -d' ' -f1",
            timeout=30,
        ).strip()
        if observed != expected:
            raise BootstrapError(f"candidate staging helper differs: {relative}")
    return helper_root


def _install_epoch_authority_artifact(
    *,
    host: str,
    key: Path,
    repo_root: str,
    helper_root: str,
    run_id: str,
    region: str,
    spec: Mapping[str, Any],
    activate_manifest: bool,
) -> dict[str, Any]:
    remote_spec = f"/tmp/leadpoet-parity-epoch-authority-{run_id}.json"
    payload = base64.b64encode(
        (json.dumps(dict(spec), sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        )
    ).decode("ascii")
    write_program = f"""
import base64
import pathlib
path = pathlib.Path({remote_spec!r})
path.write_bytes(base64.b64decode({payload!r}, validate=True))
path.chmod(0o600)
"""
    commands = [
        "set -Eeuo pipefail",
        "umask 077",
        f"trap 'rm -f -- {remote_spec}' EXIT",
        _encoded_python(write_program),
        (
            f"PYTHONPATH='{helper_root}:{repo_root}' python3 "
            f"'{helper_root}/scripts/install_production_parity_epoch_authority.py' "
            f"--spec '{remote_spec}' --run-id '{run_id}' --region '{region}'"
        ),
    ]
    output = _ssh(host, key, "\n".join(commands), timeout=600)
    try:
        value = json.loads(output.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise BootstrapError(
            "testnet epoch authority installer result is invalid"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("network") != "test"
        or value.get("netuid") != spec.get("netuid")
        or value.get("mapping_hash") != spec.get("mapping_hash")
        or value.get("network_genesis_hash")
        != spec.get("network_genesis_hash")
        or value.get("artifact_sha256") != spec.get("sha256")
        or value.get("artifact_version_id") != spec.get("version_id")
    ):
        raise BootstrapError("testnet epoch authority installation evidence differs")
    if activate_manifest:
        source = str(value.get("cutover_manifest_path") or "")
        expected = f"/home/ec2-user/.config/leadpoet/parity/{run_id}/epoch-authority/stateful-epoch-cutover.json"
        canonical = "/home/ec2-user/.config/leadpoet/stateful-epoch-cutover.json"
        if source != expected:
            raise BootstrapError("testnet epoch authority path differs")
        _ssh(
            host,
            key,
            (
                "set -Eeuo pipefail; "
                "sudo install -d -m 0700 -o ec2-user -g ec2-user "
                "/home/ec2-user/.config/leadpoet; "
                f"sudo install -m 0600 -o ec2-user -g ec2-user '{source}' '{canonical}'; "
                f"cmp -s '{source}' '{canonical}'"
            ),
            timeout=60,
        )
    return dict(value)


def _bootstrap_dashboard(
    *,
    physical: Any,
    stack_state: Mapping[str, Any],
    secret_state: Mapping[str, Any],
    infra_config: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    dashboard = physical.dashboard
    repository_url = str(stack_state.get("dashboard_repository_url") or "")
    source_sha = str(stack_state.get("dashboard_source_sha") or "")
    if (
        not REPO_URL_RE.fullmatch(repository_url)
        or not SHA_RE.fullmatch(source_sha)
        or source_sha != dashboard.source_sha
    ):
        raise BootstrapError("dashboard frozen source identity differs")
    clone_program = f"""
import pathlib
import subprocess
repo = pathlib.Path({str(dashboard.repo_root)!r})
repository_url = {repository_url!r}
source_sha = {source_sha!r}
if repo.exists():
    raise SystemExit('ephemeral dashboard repository already exists')
repo.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(['git', 'clone', '--filter=blob:none', repository_url, str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'fetch', 'origin', 'main'], check=True)
remote = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'origin/main'], text=True).strip()
if remote != source_sha:
    raise SystemExit('dashboard main advanced after it was frozen')
subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', source_sha], check=True)
subprocess.run(['git', '-C', str(repo), 'branch', '-f', 'main', source_sha], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', 'main'], check=True)
"""
    _ssh(
        dashboard.ssh_host,
        dashboard.ssh_key,
        _encoded_python(clone_program),
        timeout=600,
    )
    runner_source = ROOT / "scripts" / "run_production_parity_dashboard.py"
    remote_runner = f"/home/ec2-user/.local/lib/leadpoet-parity-dashboard-{run_id}.py"
    _ssh(
        dashboard.ssh_host,
        dashboard.ssh_key,
        "install -d -m 0700 /home/ec2-user/.local/lib",
        timeout=30,
    )
    _scp(
        dashboard.ssh_host,
        dashboard.ssh_key,
        runner_source,
        remote_runner,
        timeout=120,
    )
    expected_runner_hash = file_sha256(runner_source).split(":", 1)[1]
    remote_hash = _ssh(
        dashboard.ssh_host,
        dashboard.ssh_key,
        f"chmod 0700 '{remote_runner}'; sha256sum '{remote_runner}' | awk '{{print $1}}'",
        timeout=30,
    ).strip()
    if remote_hash != expected_runner_hash:
        raise BootstrapError("dashboard candidate runner bytes differ")
    secret_ids = secret_state.get("secret_ids")
    images = infra_config.get("container_images")
    if not isinstance(secret_ids, Mapping) or not isinstance(images, Mapping):
        raise BootstrapError("dashboard runtime inputs are incomplete")
    secret_id = str(secret_ids.get("dashboard") or "")
    caddy_image = str(images.get("caddy") or "")
    if not caddy_image or not IMAGE_RE.fullmatch(caddy_image):
        raise BootstrapError("dashboard Caddy image is not digest-pinned")
    build_command = [
        "sudo",
        "python3",
        remote_runner,
        "build",
        "--run-id",
        run_id,
        "--source-sha",
        source_sha,
        "--repo-root",
        dashboard.repo_root,
        "--secret-id",
        secret_id,
        "--dashboard-domain",
        str(urlparse(physical.dashboard_report_url).hostname or ""),
        "--caddy-image",
        caddy_image,
    ]
    if any(not re.fullmatch(r"[A-Za-z0-9_./:@+=,-]+", item) for item in build_command):
        raise BootstrapError("dashboard build command contains an unsafe value")
    build_raw = _ssh(
        dashboard.ssh_host,
        dashboard.ssh_key,
        " ".join(build_command),
        timeout=3000,
    )
    try:
        build_result = json.loads(build_raw)
    except ValueError as exc:
        raise BootstrapError("dashboard build result is invalid") from exc
    if not isinstance(build_result, Mapping) or build_result.get("status") != "built":
        raise BootstrapError("dashboard build did not complete")
    dashboard_domain = str(build_result.get("dashboard_domain") or "")
    unit_program = f"""
import pathlib
unit = pathlib.Path('/etc/systemd/system') / {dashboard.unit_name!r}
value = '''[Unit]
Description=Leadpoet disposable production-parity dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory={dashboard.repo_root}
ExecStart=/usr/bin/python3 {remote_runner} serve --source-sha {source_sha} --repo-root {dashboard.repo_root} --secret-id {secret_id} --dashboard-domain {dashboard_domain}
Restart=on-failure
RestartSec=10
TimeoutStopSec=30
'''
unit.write_text(value, encoding='utf-8')
unit.chmod(0o644)
"""
    _ssh(
        dashboard.ssh_host,
        dashboard.ssh_key,
        "sudo "
        + _encoded_python(unit_program)
        + f" && sudo systemctl daemon-reload && sudo systemctl enable --now '{dashboard.unit_name}'",
        timeout=180,
    )
    _ssh(
        dashboard.ssh_host,
        dashboard.ssh_key,
        (
            f"python3 '{remote_runner}' wait-ready "
            f"--dashboard-domain '{dashboard_domain}' --timeout-seconds 300"
        ),
        timeout=360,
    )
    return {
        "source_sha": source_sha,
        "repository_url": repository_url,
        "runner_hash": "sha256:" + expected_runner_hash,
        "dashboard_domain": dashboard_domain,
        "status": "ready",
    }


def bootstrap(
    *,
    infra_config: Mapping[str, Any],
    physical_config_path: Path,
    stack_state: Mapping[str, Any],
    secret_state: Mapping[str, Any],
    contract_path: Path,
    manifest_path: Path,
    archive_version_id: str,
) -> dict[str, Any]:
    if infra_config.get("schema_version") != "leadpoet.production_parity_infra.v1":
        raise BootstrapError("parity infrastructure config schema differs")
    physical = load_config(physical_config_path)
    contract = verify_contract_checkout(
        ROOT, _load(contract_path, field="candidate contract")
    )
    manifest = validate_snapshot_manifest(_load(manifest_path, field="snapshot manifest"))
    if not VERSION_RE.fullmatch(str(archive_version_id or "")):
        raise BootstrapError("snapshot archive version identity is invalid")
    run_id = str(stack_state.get("run_id") or "")
    candidate_sha = str(stack_state.get("candidate_sha") or "")
    if (
        not RUN_RE.fullmatch(run_id)
        or not SHA_RE.fullmatch(candidate_sha)
        or physical.ephemeral_stack_id != run_id
        or contract["candidate_sha"] != candidate_sha
        or secret_state.get("candidate_sha") != candidate_sha
    ):
        raise BootstrapError("staging bootstrap identities differ")
    metadata = infra_config.get("runtime_metadata")
    if not isinstance(metadata, Mapping):
        raise BootstrapError("runtime metadata is missing")
    repo_url = str(metadata.get("repository_url") or "")
    if not REPO_URL_RE.fullmatch(repo_url):
        raise BootstrapError("staging repository URL is invalid")
    gateway_repo = str(metadata.get("gateway_repo_root") or "/home/ec2-user/leadpoet_repo")
    validator_repo = physical.primary_validator.repo_root
    auditor_repo = str(metadata.get("auditor_repo_root") or "/home/ec2-user/leadpoet/leadpoet")
    database_repo = str(metadata.get("database_repo_root") or "/home/ec2-user/leadpoet-parity/repo")
    wallet_artifacts = stack_state.get("wallet_artifacts")
    wallet_identities = secret_state.get("wallet_identities")
    epoch_authority_artifact = stack_state.get("epoch_authority_artifact")
    epoch_authority_identity = secret_state.get("epoch_authority_identity")
    if not isinstance(wallet_artifacts, Mapping) or not isinstance(
        wallet_identities, Mapping
    ):
        raise BootstrapError("staging wallet authority is missing")
    if not isinstance(epoch_authority_artifact, Mapping) or not isinstance(
        epoch_authority_identity, Mapping
    ):
        raise BootstrapError("testnet epoch authority is missing")
    outputs = stack_state.get("outputs")
    if not isinstance(outputs, Mapping):
        raise BootstrapError("staging stack outputs are missing")
    instance_identities = _verify_ephemeral_instances(
        region=str(infra_config.get("region") or ""),
        outputs=outputs,
        run_id=run_id,
        candidate_sha=candidate_sha,
    )
    prepared: dict[str, str] = {}
    prepared["gateway"] = _ssh(
        physical.gateway.ssh_host,
        physical.gateway.ssh_key,
        _prepare_repo_command(
            repo_url=repo_url,
            repo_root=gateway_repo,
            base_sha=contract["base_sha"],
            host_launcher=physical.gateway.restart_path,
            launcher_relative="gw_restart.sh",
        ),
        timeout=600,
    ).strip()
    prepared["validator"] = _ssh(
        physical.primary_validator.ssh_host,
        physical.primary_validator.ssh_key,
        _prepare_repo_command(
            repo_url=repo_url,
            repo_root=validator_repo,
            base_sha=contract["base_sha"],
            host_launcher=physical.primary_validator.restart_path,
            launcher_relative="validator_restart.sh",
        ),
        timeout=600,
    ).strip()
    helper_roots = {
        "gateway": _stage_candidate_artifact_helpers(
            host=physical.gateway.ssh_host,
            key=physical.gateway.ssh_key,
            run_id=run_id,
        ),
        "primary-validator": _stage_candidate_artifact_helpers(
            host=physical.primary_validator.ssh_host,
            key=physical.primary_validator.ssh_key,
            run_id=run_id,
        ),
    }
    epoch_authority_installs: dict[str, dict[str, Any]] = {
        "gateway": _install_epoch_authority_artifact(
            host=physical.gateway.ssh_host,
            key=physical.gateway.ssh_key,
            repo_root=gateway_repo,
            helper_root=helper_roots["gateway"],
            run_id=run_id,
            region=str(infra_config.get("region") or ""),
            spec=epoch_authority_artifact,
            activate_manifest=True,
        ),
        "primary-validator": _install_epoch_authority_artifact(
            host=physical.primary_validator.ssh_host,
            key=physical.primary_validator.ssh_key,
            repo_root=validator_repo,
            helper_root=helper_roots["primary-validator"],
            run_id=run_id,
            region=str(infra_config.get("region") or ""),
            spec=epoch_authority_artifact,
            activate_manifest=True,
        ),
    }
    wallet_installs: dict[str, dict[str, Any]] = {}
    primary_spec = wallet_artifacts.get("primary-validator")
    if not isinstance(primary_spec, Mapping):
        raise BootstrapError("primary validator wallet artifact is missing")
    wallet_installs["primary-validator"] = _install_wallet_artifact(
        host=physical.primary_validator.ssh_host,
        key=physical.primary_validator.ssh_key,
        repo_root=validator_repo,
        run_id=run_id,
        region=str(infra_config.get("region") or ""),
        role="primary-validator",
        spec=primary_spec,
        helper_root=helper_roots["primary-validator"],
    )
    for index, auditor in enumerate(physical.auditors):
        prepared[f"auditor-{index}"] = _ssh(
            auditor.ssh_host,
            auditor.ssh_key,
            _prepare_repo_command(
                repo_url=repo_url,
                repo_root=auditor_repo,
                base_sha=contract["base_sha"],
                host_launcher=f"/home/ec2-user/.local/bin/leadpoet-parity-auditor-{index}-base",
                launcher_relative="validator_restart.sh",
            ),
            timeout=600,
        ).strip()
        role = f"auditor-{'a' if index == 0 else 'b'}"
        helper_roots[role] = _stage_candidate_artifact_helpers(
            host=auditor.ssh_host,
            key=auditor.ssh_key,
            run_id=run_id,
        )
        epoch_authority_installs[role] = _install_epoch_authority_artifact(
            host=auditor.ssh_host,
            key=auditor.ssh_key,
            repo_root=auditor_repo,
            helper_root=helper_roots[role],
            run_id=run_id,
            region=str(infra_config.get("region") or ""),
            spec=epoch_authority_artifact,
            activate_manifest=True,
        )
        auditor_spec = wallet_artifacts.get(role)
        if not isinstance(auditor_spec, Mapping):
            raise BootstrapError(f"{role} wallet artifact is missing")
        wallet_installs[role] = _install_wallet_artifact(
            host=auditor.ssh_host,
            key=auditor.ssh_key,
            repo_root=auditor_repo,
            run_id=run_id,
            region=str(infra_config.get("region") or ""),
            role=role,
            spec=auditor_spec,
            helper_root=helper_roots[role],
        )
        remote_template = f"/tmp/leadpoet-parity-auditor-{run_id}-{index}.service"
        _scp(
            auditor.ssh_host,
            auditor.ssh_key,
            ROOT / "infra" / "leadpoet-production-parity-auditor.service",
            remote_template,
            timeout=120,
        )
        install_program = f"""
import pathlib
import re
source = pathlib.Path({remote_template!r})
unit_name = {auditor.unit_name!r}
repo_root = {auditor.repo_root!r}
if not re.fullmatch(r'[A-Za-z0-9_.@-]+', unit_name):
    raise SystemExit('auditor unit name is invalid')
value = source.read_text(encoding='utf-8')
if value.count('@@REPO_ROOT@@') != 2:
    raise SystemExit('auditor unit template contract differs')
value = value.replace('@@REPO_ROOT@@', repo_root)
target = pathlib.Path('/etc/systemd/system') / unit_name
target.write_text(value, encoding='utf-8')
target.chmod(0o644)
source.unlink()
"""
        _ssh(
            auditor.ssh_host,
            auditor.ssh_key,
            "sudo " + _encoded_python(install_program) + " && sudo systemctl daemon-reload",
            timeout=180,
        )

    for role, install_evidence in wallet_installs.items():
        identity = wallet_identities.get(role)
        if (
            not isinstance(identity, Mapping)
            or identity.get("expected_hotkey")
            != install_evidence.get("expected_hotkey")
            or identity.get("artifact_sha256")
            != install_evidence.get("artifact_sha256")
            or identity.get("artifact_version_id")
            != install_evidence.get("artifact_version_id")
        ):
            raise BootstrapError(
                f"{role} wallet installation and secret identities differ"
            )
    for role, install_evidence in epoch_authority_installs.items():
        if (
            install_evidence.get("artifact_sha256")
            != epoch_authority_identity.get("artifact_sha256")
            or install_evidence.get("artifact_version_id")
            != epoch_authority_identity.get("artifact_version_id")
            or install_evidence.get("mapping_hash")
            != epoch_authority_identity.get("mapping_hash")
            or install_evidence.get("network_genesis_hash")
            != epoch_authority_identity.get("network_genesis_hash")
            or install_evidence.get("netuid")
            != epoch_authority_identity.get("netuid")
        ):
            raise BootstrapError(f"{role} testnet epoch authority differs")

    secret_ids = secret_state.get("secret_ids")
    if not isinstance(secret_ids, Mapping):
        raise BootstrapError("staging stack or secret outputs are missing")
    database_host = f"{str(metadata.get('ssh_user') or 'ec2-user')}@{outputs.get('DatabasePublicIp')}"
    database_key = physical.gateway.ssh_key
    database_domain = str(outputs.get("DatabaseDomain") or "")
    remote_root = f"/home/ec2-user/leadpoet-parity/{run_id}"
    database_program = f"""
import pathlib
import subprocess
repo_url = {repo_url!r}
repo = pathlib.Path({database_repo!r})
candidate = {candidate_sha!r}
if repo.exists():
    raise SystemExit("ephemeral database repository already exists")
repo.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["git", "clone", "--filter=blob:none", repo_url, str(repo)], check=True)
subprocess.run(["git", "-C", str(repo), "checkout", "--detach", candidate], check=True)
subprocess.run(["git", "-C", str(repo), "diff", "--exit-code", candidate, "--"], check=True)
pathlib.Path({remote_root!r}).mkdir(parents=True, mode=0o700, exist_ok=False)
"""
    _ssh(database_host, database_key, _encoded_python(database_program), timeout=600)
    helper_roots["database"] = _stage_candidate_artifact_helpers(
        host=database_host,
        key=database_key,
        run_id=run_id,
    )
    epoch_authority_installs["database"] = _install_epoch_authority_artifact(
        host=database_host,
        key=database_key,
        repo_root=database_repo,
        helper_root=helper_roots["database"],
        run_id=run_id,
        region=str(infra_config.get("region") or ""),
        spec=epoch_authority_artifact,
        activate_manifest=False,
    )
    database_authority = epoch_authority_installs["database"]
    if any(
        database_authority.get(field) != epoch_authority_identity.get(field)
        for field in (
            "artifact_sha256",
            "artifact_version_id",
            "mapping_hash",
            "network_genesis_hash",
            "netuid",
        )
    ):
        raise BootstrapError("database testnet epoch authority differs")
    remote_contract = remote_root + "/contract.json"
    remote_manifest = remote_root + "/snapshot-manifest.json"
    remote_archive = remote_root + "/snapshot.dump"
    _scp(database_host, database_key, contract_path, remote_contract, timeout=120)
    _scp(database_host, database_key, manifest_path, remote_manifest, timeout=120)
    archive_uri = str(manifest["archive"]["s3_uri"])
    parsed_archive = urlparse(archive_uri)
    if (
        parsed_archive.scheme != "s3"
        or not parsed_archive.netloc
        or not parsed_archive.path.strip("/")
    ):
        raise BootstrapError("snapshot archive URI is invalid")
    _ssh(
        database_host,
        database_key,
        (
            "umask 077; aws --region "
            f"{shlex.quote(str(infra_config.get('region') or ''))} s3api get-object "
            f"--bucket {shlex.quote(parsed_archive.netloc)} "
            f"--key {shlex.quote(parsed_archive.path.lstrip('/'))} "
            f"--version-id {shlex.quote(archive_version_id)} "
            f"{shlex.quote(remote_archive)} >/dev/null"
        ),
        timeout=1800,
    )
    images = infra_config.get("container_images")
    if not isinstance(images, Mapping):
        raise BootstrapError("digest-pinned container image configuration is missing")
    image_values = {
        key: str(images.get(key) or "")
        for key in ("postgres", "postgrest")
    }
    if any(not IMAGE_RE.fullmatch(value) for value in image_values.values()):
        raise BootstrapError("staging database image is not digest-pinned")
    command = [
        "sudo",
        "python3",
        f"{database_repo}/scripts/production_parity_database_host.py",
        "start",
        "--run-id",
        run_id,
        "--candidate-sha",
        candidate_sha,
        "--contract",
        remote_contract,
        "--manifest",
        remote_manifest,
        "--archive",
        remote_archive,
        "--epoch-authority-root",
        str(database_authority.get("install_root") or ""),
        "--database-secret-id",
        str(secret_ids.get("database") or ""),
        "--region",
        str(infra_config.get("region") or ""),
        "--database-domain",
        database_domain,
        "--postgres-image",
        image_values["postgres"],
        "--postgrest-image",
        image_values["postgrest"],
    ]
    if any(not re.fullmatch(r"[A-Za-z0-9_./:@+=,-]+", item) for item in command):
        raise BootstrapError("database bootstrap command contains an unsafe value")
    database_result_raw = _ssh(
        database_host,
        database_key,
        " ".join(command),
        timeout=3600,
    )
    try:
        database_result = json.loads(database_result_raw)
    except ValueError as exc:
        raise BootstrapError("database bootstrap result is invalid") from exc
    if not isinstance(database_result, Mapping) or database_result.get("status") != "ready":
        raise BootstrapError("database bootstrap did not become ready")
    ingress = _verify_database_ingress(
        region=str(infra_config.get("region") or ""),
        database_tls_security_group=str(
            outputs.get("DatabaseWebSecurityGroupId") or ""
        ),
        database_service_security_group=str(
            outputs.get("DatabaseServiceSecurityGroupId") or ""
        ),
        gateway_security_group=str(
            outputs.get("GatewayWebSecurityGroupId") or ""
        ),
        dashboard_security_group=str(
            outputs.get("DashboardWebSecurityGroupId") or ""
        ),
    )
    database_tls = _wait_private_database_tls(
        host=physical.gateway.ssh_host,
        key=physical.gateway.ssh_key,
        database_domain=database_domain,
    )
    dashboard_result = _bootstrap_dashboard(
        physical=physical,
        stack_state=stack_state,
        secret_state=secret_state,
        infra_config=infra_config,
        run_id=run_id,
    )
    return {
        "schema_version": "leadpoet.production_parity_bootstrap.v1",
        "candidate_sha": candidate_sha,
        "base_sha": contract["base_sha"],
        "run_id": run_id,
        "prepared_host_count": len(prepared),
        "prepared_commits": prepared,
        "instance_identities": instance_identities,
        "wallet_installs": wallet_installs,
        "epoch_authority_installs": epoch_authority_installs,
        "database": dict(database_result),
        "database_ingress": ingress,
        "database_tls": database_tls,
        "dashboard": dashboard_result,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infra-config", type=Path, required=True)
    parser.add_argument("--physical-config", type=Path, required=True)
    parser.add_argument("--stack-state", type=Path, required=True)
    parser.add_argument("--secret-state", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--snapshot-archive-version-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = bootstrap(
            infra_config=_load(args.infra_config, field="infrastructure config"),
            physical_config_path=args.physical_config,
            stack_state=_load(args.stack_state, field="stack state"),
            secret_state=_load(args.secret_state, field="secret state"),
            contract_path=args.contract,
            manifest_path=args.snapshot_manifest,
            archive_version_id=args.snapshot_archive_version_id,
        )
    except (
        OSError,
        ValueError,
        BootstrapError,
        PhysicalStagingError,
        ProductionParityError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
