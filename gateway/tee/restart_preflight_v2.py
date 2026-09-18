"""Fail-closed gateway release checks that run before shutdown."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.request import Request, urlopen

from gateway.tee.release_manifest_v2 import validate_release_manifest
from gateway.tee.topology import ROLE_SPECS, validate_manifest


FULL_TOPOLOGY_INSTANCE_TYPE = "r7i.4xlarge"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class GatewayRestartPreflightV2Error(RuntimeError):
    """The selected host or release cannot safely replace production."""


def _json(path: Path, field: str) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GatewayRestartPreflightV2Error(
            "%s is unavailable or invalid" % field
        ) from exc
    if not isinstance(value, Mapping):
        raise GatewayRestartPreflightV2Error("%s must be an object" % field)
    return dict(value)


def _imds_instance_type(timeout_seconds: float = 2.0) -> str:
    try:
        token_request = Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urlopen(token_request, timeout=timeout_seconds) as response:
            token = response.read().decode("ascii").strip()
        if not token:
            raise ValueError("empty IMDS token")
        type_request = Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urlopen(type_request, timeout=timeout_seconds) as response:
            return response.read().decode("ascii").strip()
    except Exception as exc:
        raise GatewayRestartPreflightV2Error(
            "gateway EC2 instance type is unavailable from IMDSv2"
        ) from exc


def _configured_processor_count() -> int:
    try:
        cpus = int(os.sysconf("SC_NPROCESSORS_CONF"))
    except (AttributeError, OSError, TypeError, ValueError):
        cpus = 0
    if cpus > 0:
        return cpus
    cpus = sum(
        1
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        if line.partition(":")[0].strip() == "processor"
    )
    return cpus if cpus > 0 else int(os.cpu_count() or 0)


def _observed_capacity() -> tuple[int, int]:
    try:
        cpus = _configured_processor_count()
        memory_kib = next(
            int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemTotal:")
        )
    except Exception as exc:
        raise GatewayRestartPreflightV2Error(
            "gateway parent capacity is unavailable"
        ) from exc
    return cpus, memory_kib // 1024


def verify_gateway_restart_preflight_v2(
    *,
    deploy_commit: str,
    release_manifest: Mapping[str, Any],
    topology_manifest: Mapping[str, Any],
    topology_mode: str,
    instance_type: str,
    parent_vcpus: int,
    parent_memory_mib: int,
) -> Dict[str, Any]:
    commit = str(deploy_commit or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise GatewayRestartPreflightV2Error("gateway deployment commit is invalid")
    release = validate_release_manifest(release_manifest)
    if release["commit_sha"] != commit:
        raise GatewayRestartPreflightV2Error(
            "local gateway release is for another commit"
        )
    topology = validate_manifest(topology_manifest)
    mode = str(topology_mode or "")
    if mode not in {"full", "component"}:
        raise GatewayRestartPreflightV2Error(
            "GATEWAY_TEE_TOPOLOGY_MODE must be full or component"
        )
    if mode == "full":
        if instance_type != FULL_TOPOLOGY_INSTANCE_TYPE:
            raise GatewayRestartPreflightV2Error(
                "full deployment requires r7i.4xlarge"
            )
        if int(parent_vcpus) < 16 or int(parent_memory_mib) < 125000:
            raise GatewayRestartPreflightV2Error(
                "full deployment has insufficient parent capacity"
            )
    return {
        "schema_version": "leadpoet.gateway_restart_preflight.v3",
        "status": "ready",
        "deploy_commit": commit,
        "release_hash": release["release_hash"],
        "topology_hash": topology["topology_hash"],
        "topology_mode": mode,
        "instance_type": instance_type,
        "parent_vcpus": int(parent_vcpus),
        "parent_memory_mib": int(parent_memory_mib),
        "role_count": len(ROLE_SPECS) if mode == "full" else 1,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-commit", required=True)
    parser.add_argument("--release-manifest", required=True, type=Path)
    parser.add_argument("--topology-manifest", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--topology-mode", choices=("full", "component"), required=True)
    args = parser.parse_args(argv)
    try:
        from scripts.gateway_git_deploy import write_tree_verification_evidence

        candidate_tree_evidence = write_tree_verification_evidence(
            repo_root=Path(
                os.environ.get("LEADPOET_REPO_ROOT", "/home/ec2-user/leadpoet_repo")
            ),
            materialized_root=Path.cwd(),
            target_sha=str(args.deploy_commit).lower(),
            phase="prepared_archive",
            strict_extras=True,
            output_path=args.config_dir / "gateway-candidate-tree-preflight.json",
        )
    except Exception as exc:
        raise GatewayRestartPreflightV2Error(
            "prepared gateway candidate tree does not match its exact Git blobs"
        ) from exc
    parent_vcpus, parent_memory_mib = _observed_capacity()
    result = verify_gateway_restart_preflight_v2(
        deploy_commit=args.deploy_commit,
        release_manifest=_json(args.release_manifest, "gateway release manifest"),
        topology_manifest=_json(args.topology_manifest, "gateway topology manifest"),
        topology_mode=args.topology_mode,
        instance_type=_imds_instance_type(),
        parent_vcpus=parent_vcpus,
        parent_memory_mib=parent_memory_mib,
    )
    result["prepared_candidate_tree"] = candidate_tree_evidence
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
