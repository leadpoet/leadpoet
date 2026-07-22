"""Fail-closed V2 gateway deployment checks that must run before shutdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.request import Request, urlopen

from gateway.research_lab.provider_profiles_v2 import (
    verify_required_worker_proxy_profiles_v2,
)
from gateway.research_lab.worker_autostart import (
    build_research_lab_worker_autostart_plan,
)
from gateway.tee.acceptance_corpus_v2 import (
    load_and_validate_acceptance_corpus_v2,
)
from gateway.tee.artifact_persistence_v2 import validate_artifact_policy
from gateway.tee.provider_broker_v2 import (
    credential_reference_hash,
    expected_provider_credential_slots,
)
from gateway.tee.release_manifest_v2 import validate_release_manifest
from gateway.tee.topology import ROLE_SPECS, validate_manifest
from gateway.utils.tee_kms_provision_v2 import load_provider_envelopes


FULL_TOPOLOGY_INSTANCE_TYPE = "r7i.4xlarge"
REQUIRED_BOOT_ENVELOPE_FILES = (
    "artifact_master_key.json",
    "openrouter.json",
    "exa.json",
    "scrapingdog.json",
    "deepline.json",
    "supabase_service_role.json",
    "truelist.json",
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PROTECTED_PARENT_PLAINTEXT_SLOTS = frozenset(
    {"openrouter", "exa", "scrapingdog", "deepline"}
)


class GatewayRestartPreflightV2Error(RuntimeError):
    """The selected host or V2 release cannot safely replace production."""


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
    """Return physical/configured CPUs, including CPUs reserved by Nitro."""

    import os

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


def load_parent_environment(path: Path) -> Dict[str, str]:
    """Parse the restart's shell-quoted env cache without executing it."""

    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GatewayRestartPreflightV2Error(
            "prepared parent environment is unavailable"
        ) from exc
    environment: Dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        try:
            parts = shlex.split(line, posix=True)
        except ValueError as exc:
            raise GatewayRestartPreflightV2Error(
                "prepared parent environment is malformed"
            ) from exc
        if len(parts) != 1 or "=" not in parts[0]:
            raise GatewayRestartPreflightV2Error(
                "prepared parent environment is malformed"
            )
        name, value = parts[0].split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise GatewayRestartPreflightV2Error(
                "prepared parent environment has an invalid name"
            )
        environment[name] = value
    return environment


def _reject_parent_provider_plaintext(
    *,
    envelopes: Sequence[Mapping[str, Any]],
    parent_environment: Mapping[str, str],
) -> None:
    protected_hashes = {
        str(item["credential_ref_hash"]): str(item["credential_slot"])
        for item in envelopes
        if str(item.get("credential_slot") or "")
        in _PROTECTED_PARENT_PLAINTEXT_SLOTS
    }
    for value in parent_environment.values():
        plaintext = str(value or "")
        if not plaintext or "\x00" in plaintext:
            continue
        slot = protected_hashes.get(credential_reference_hash(plaintext))
        if slot:
            raise GatewayRestartPreflightV2Error(
                "parent environment contains the protected %s credential" % slot
            )


def verify_artifact_bucket_lock_v2(
    policy: Mapping[str, Any], *, s3_client: Any
) -> Dict[str, Any]:
    normalized = validate_artifact_policy(policy)
    bucket = normalized["bucket_host"].split(".s3", 1)[0]
    if not bucket:
        raise GatewayRestartPreflightV2Error("artifact bucket name is invalid")
    try:
        versioning = s3_client.get_bucket_versioning(Bucket=bucket)
        object_lock = s3_client.get_object_lock_configuration(Bucket=bucket)
    except Exception as exc:
        raise GatewayRestartPreflightV2Error(
            "artifact bucket protection is unavailable"
        ) from exc
    rule = (
        dict(object_lock.get("ObjectLockConfiguration") or {})
        .get("Rule", {})
        .get("DefaultRetention", {})
    )
    days = int(rule.get("Days") or 0) + int(rule.get("Years") or 0) * 365
    if (
        versioning.get("Status") != "Enabled"
        or (object_lock.get("ObjectLockConfiguration") or {}).get(
            "ObjectLockEnabled"
        )
        != "Enabled"
        or rule.get("Mode") != "COMPLIANCE"
        or days < normalized["minimum_retention_days"]
    ):
        raise GatewayRestartPreflightV2Error(
            "artifact bucket lacks required COMPLIANCE Object Lock"
        )
    return {
        "bucket": bucket,
        "object_lock_mode": "COMPLIANCE",
        "retention_days": days,
        "versioning": "Enabled",
    }


def verify_gateway_restart_preflight_v2(
    *,
    deploy_commit: str,
    release_manifest: Mapping[str, Any],
    topology_manifest: Mapping[str, Any],
    artifact_policy: Mapping[str, Any],
    credential_envelope_paths: Sequence[Path],
    config_dir: Path,
    topology_mode: str,
    instance_type: str,
    parent_vcpus: int,
    parent_memory_mib: int,
    parent_environment: Mapping[str, str],
    acceptance_corpus_manifest_path: Optional[Path] = None,
    acceptance_corpus_root: Optional[Path] = None,
    artifact_s3_client: Any = None,
) -> Dict[str, Any]:
    commit = str(deploy_commit or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise GatewayRestartPreflightV2Error("gateway deployment commit is invalid")
    release = validate_release_manifest(release_manifest)
    if release["commit_sha"] != commit:
        raise GatewayRestartPreflightV2Error(
            "approved gateway V2 release is for another commit"
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
                "full V2 deployment requires r7i.4xlarge"
            )
        if int(parent_vcpus) < 16 or int(parent_memory_mib) < 125000:
            raise GatewayRestartPreflightV2Error(
                "full V2 deployment has insufficient parent capacity"
            )
        if acceptance_corpus_manifest_path is None or acceptance_corpus_root is None:
            raise GatewayRestartPreflightV2Error(
                "full V2 deployment requires the signed acceptance corpus"
            )
        try:
            acceptance_corpus = load_and_validate_acceptance_corpus_v2(
                Path(acceptance_corpus_manifest_path),
                corpus_root=Path(acceptance_corpus_root),
                expected_signing_pubkey_hash=release[
                    "acceptance_signer_pubkey_hash"
                ],
            )
        except Exception as exc:
            raise GatewayRestartPreflightV2Error(
                "signed V2 acceptance corpus is invalid: %s" % exc
            ) from exc
    else:
        acceptance_corpus = None

    normalized_policy = validate_artifact_policy(artifact_policy)
    artifact_storage = (
        verify_artifact_bucket_lock_v2(
            normalized_policy, s3_client=artifact_s3_client
        )
        if artifact_s3_client is not None
        else None
    )
    paths = tuple(Path(path) for path in credential_envelope_paths)
    if len(paths) != len(REQUIRED_BOOT_ENVELOPE_FILES):
        raise GatewayRestartPreflightV2Error(
            "gateway V2 boot credential envelope set is incomplete"
        )
    if {path.name for path in paths} != set(REQUIRED_BOOT_ENVELOPE_FILES):
        raise GatewayRestartPreflightV2Error(
            "gateway V2 boot credential envelope filenames are invalid"
        )
    envelopes = load_provider_envelopes(paths)
    observed_slots = {str(item["credential_slot"]) for item in envelopes}
    expected_slots = set(expected_provider_credential_slots()) | {
        "artifact_master_key"
    }
    if observed_slots != expected_slots:
        raise GatewayRestartPreflightV2Error(
            "gateway V2 boot credential slots differ from measured routes"
        )
    _reject_parent_provider_plaintext(
        envelopes=envelopes,
        parent_environment=parent_environment,
    )
    profile_result = (
        verify_required_worker_proxy_profiles_v2(config_dir=Path(config_dir))
        if mode == "full"
        else {
            "schema_version": "leadpoet.worker_proxy_profile_set.v2",
            "status": "component_only",
            "profile_count": 0,
        }
    )
    if mode == "full":
        # Worker/proxy decoupling must be explicit in production: require both
        # *_PROCESS_COUNT variables to be set to a positive integer. Without this
        # gate a restart silently falls back to one-worker-per-proxy, preserving
        # the oversized fleet the decoupling is meant to shrink. Failing the
        # preflight forces the operator to size the fleet on purpose.
        for count_env in (
            "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT",
            "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT",
        ):
            raw = str(parent_environment.get(count_env, "")).strip()
            try:
                parsed = int(raw)
            except ValueError:
                parsed = 0
            if parsed < 1:
                raise GatewayRestartPreflightV2Error(
                    "full V2 deployment requires %s to be set to a positive "
                    "worker process count (got %r)" % (count_env, raw)
                )
        worker_plan = build_research_lab_worker_autostart_plan(parent_environment)
        if not worker_plan.hosted.enabled or not worker_plan.scoring.enabled:
            raise GatewayRestartPreflightV2Error(
                "full V2 deployment requires both configured worker fleets"
            )
        expected_worker_counts = {
            "gateway_autoresearch": int(worker_plan.hosted.worker_count),
            "gateway_scoring": int(worker_plan.scoring.worker_count),
        }
        # Worker count is decoupled from proxy count: a fleet may run fewer
        # workers than it has encrypted proxy profiles (each worker index still
        # maps to its own profile). Require a profile for every worker index in
        # range, not strict equality, so reducing the process count does not
        # fail the preflight. Running more workers than profiles is still
        # rejected (there would be no TLS envelope for the extra indices).
        available_profile_counts = dict(profile_result.get("worker_counts") or {})
        for role, required in expected_worker_counts.items():
            available = int(available_profile_counts.get(role, 0))
            if required < 1 or required > available:
                raise GatewayRestartPreflightV2Error(
                    "%s worker count %d has no encrypted proxy profile coverage "
                    "(%d profiles available)" % (role, required, available)
                )
    else:
        expected_worker_counts = {}
    return {
        "schema_version": "leadpoet.gateway_restart_preflight.v2",
        "status": "ready",
        "deploy_commit": commit,
        "release_hash": release["release_hash"],
        "topology_hash": topology["topology_hash"],
        "topology_mode": mode,
        "instance_type": instance_type,
        "parent_vcpus": int(parent_vcpus),
        "parent_memory_mib": int(parent_memory_mib),
        "role_count": len(ROLE_SPECS) if mode == "full" else 1,
        "boot_credential_slot_count": len(observed_slots),
        "parent_plaintext_provider_slot_count": 0,
        "artifact_bucket_host": normalized_policy["bucket_host"],
        "artifact_bucket_protection": artifact_storage or "not_requested",
        "worker_proxy_profile_count": int(profile_result["profile_count"]),
        "worker_counts": expected_worker_counts,
        "acceptance_corpus_manifest_hash": (
            str(acceptance_corpus["manifest_hash"])
            if acceptance_corpus is not None
            else "component_only"
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-commit", required=True)
    parser.add_argument("--release-manifest", required=True, type=Path)
    parser.add_argument("--topology-manifest", required=True, type=Path)
    parser.add_argument("--artifact-policy", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--topology-mode", choices=("full", "component"), required=True)
    parser.add_argument("--credential-envelope", action="append", type=Path, default=[])
    parser.add_argument("--parent-env-file", required=True, type=Path)
    parser.add_argument("--acceptance-corpus-manifest", type=Path)
    parser.add_argument("--acceptance-corpus-root", type=Path)
    args = parser.parse_args(argv)
    parent_vcpus, parent_memory_mib = _observed_capacity()
    import boto3

    result = verify_gateway_restart_preflight_v2(
        deploy_commit=args.deploy_commit,
        release_manifest=_json(args.release_manifest, "gateway V2 release manifest"),
        topology_manifest=_json(args.topology_manifest, "gateway topology manifest"),
        artifact_policy=_json(args.artifact_policy, "encrypted artifact policy"),
        credential_envelope_paths=args.credential_envelope,
        config_dir=args.config_dir,
        topology_mode=args.topology_mode,
        instance_type=_imds_instance_type(),
        parent_vcpus=parent_vcpus,
        parent_memory_mib=parent_memory_mib,
        parent_environment=load_parent_environment(args.parent_env_file),
        acceptance_corpus_manifest_path=args.acceptance_corpus_manifest,
        acceptance_corpus_root=args.acceptance_corpus_root,
        artifact_s3_client=boto3.client("s3"),
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
