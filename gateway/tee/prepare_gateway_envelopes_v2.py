"""Prepare all KMS-sealed gateway V2 boot and worker-profile envelopes.

This is an operator-only, pre-cutover command. Plaintext is read from one
protected environment file, sent to AWS KMS Encrypt, and never written or
printed. The output directory contains ciphertext envelopes and a non-secret
transition report only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import tempfile
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from gateway.research_lab.worker_autostart import (
    DEFERRED_WORKER_FLEETS_ENV,
    HOSTED_PROXY_PREFIXES,
    SCORING_PROXY_PREFIXES,
    DeferredWorkerFleetConfigurationError,
    build_research_lab_worker_autostart_plan,
    canonical_deferred_worker_fleet_roles,
    deferred_worker_fleet_roles,
)
from gateway.tee.artifact_vault_v2 import artifact_master_key_reference_hash
from gateway.tee.provider_broker_v2 import (
    ProviderBrokerV2Error,
    _validated_tls_proxy_url,
    credential_reference_hash,
    credential_value_hash,
)
from gateway.tee.proxy_transport_preflight_v2 import (
    WorkerProxyTransportPreflightV2Error,
    verify_worker_proxy_fleets_v2,
)
from gateway.tee.supabase_schema_preflight_v2 import (
    verify_required_supabase_v2_schema,
)
from gateway.utils.tee_kms_provision_v2 import build_provider_envelope_v2


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_BOOT_SOURCES = {
    "openrouter": (
        "RESEARCH_LAB_V2_OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
    ),
    "exa": ("RESEARCH_LAB_V2_EXA_API_KEY", "EXA_API_KEY"),
    "scrapingdog": (
        "RESEARCH_LAB_V2_SCRAPINGDOG_API_KEY",
        "SCRAPINGDOG_API_KEY",
    ),
    "deepline": ("RESEARCH_LAB_V2_DEEPLINE_API_KEY", "DEEPLINE_API_KEY"),
    "supabase_service_role": ("SUPABASE_SERVICE_ROLE_KEY",),
    "truelist": ("TRUELIST_API_KEY",),
}
_SHARED_PARENT_SLOTS = frozenset(("supabase_service_role", "truelist"))
_WORKER_PROXY_TRANSPORT_POLICY = "https_port_443_authenticated_connect.v2"
_SPECIAL_PROFILES = {
    "benchmark_exa.json": (
        "exa",
        ("RESEARCH_LAB_V2_BENCHMARK_EXA_API_KEY", *_BOOT_SOURCES["exa"]),
    ),
    "benchmark_openrouter.json": (
        "openrouter",
        (
            "RESEARCH_LAB_V2_BENCHMARK_OPENROUTER_API_KEY",
            *_BOOT_SOURCES["openrouter"],
        ),
    ),
    "benchmark_scrapingdog.json": (
        "scrapingdog",
        (
            "RESEARCH_LAB_V2_BENCHMARK_SCRAPINGDOG_API_KEY",
            *_BOOT_SOURCES["scrapingdog"],
        ),
    ),
    "stale_parent_openrouter.json": (
        "openrouter",
        _BOOT_SOURCES["openrouter"],
    ),
    "source_add_judge_openrouter.json": (
        "openrouter",
        (
            "RESEARCH_LAB_V2_SOURCE_ADD_JUDGE_OPENROUTER_API_KEY",
            *_BOOT_SOURCES["openrouter"],
        ),
    ),
}


class GatewayEnvelopePreparationV2Error(RuntimeError):
    """The operator input cannot produce a complete encrypted V2 profile."""


def load_environment_file(path: Path) -> Dict[str, str]:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise GatewayEnvelopePreparationV2Error(
            "gateway source environment is unavailable"
        ) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        if not isinstance(parsed, Mapping):
            raise GatewayEnvelopePreparationV2Error(
                "gateway source environment JSON must be an object"
            )
        return {str(name): str(value) for name, value in parsed.items()}
    result: Dict[str, str] = {}
    for raw_line in raw.replace("\x00", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        try:
            parts = shlex.split(line, posix=True)
        except ValueError as exc:
            raise GatewayEnvelopePreparationV2Error(
                "gateway source environment is malformed"
            ) from exc
        if len(parts) != 1 or "=" not in parts[0]:
            raise GatewayEnvelopePreparationV2Error(
                "gateway source environment is malformed"
            )
        name, value = parts[0].split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise GatewayEnvelopePreparationV2Error(
                "gateway source environment name is invalid"
            )
        result[name] = value
    return result


def scrub_parent_environment_file_v2(
    *,
    environment_path: Path,
    transition_report_path: Path,
) -> Dict[str, Any]:
    """Install sealed-fleet counts while removing parent plaintext aliases."""

    try:
        report = json.loads(Path(transition_report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GatewayEnvelopePreparationV2Error(
            "gateway V2 environment transition report is unavailable"
        ) from exc
    if not isinstance(report, Mapping):
        raise GatewayEnvelopePreparationV2Error(
            "gateway V2 environment transition report is invalid"
        )
    remove_names = {
        str(value)
        for value in report.get("plaintext_environment_names_to_remove") or ()
    }
    remove_refs = {
        str(value)
        for value in report.get("plaintext_credential_ref_hashes_to_remove") or ()
    }
    if not remove_refs:
        raise GatewayEnvelopePreparationV2Error(
            "gateway V2 plaintext credential commitments are unavailable"
        )
    count_fields = {
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "hosted_worker_count",
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "scoring_worker_count",
    }
    raw_count_environment = report.get("required_count_environment")
    if (
        not isinstance(raw_count_environment, Mapping)
        or set(raw_count_environment) != set(count_fields)
    ):
        raise GatewayEnvelopePreparationV2Error(
            "gateway V2 worker count environment is invalid"
        )
    count_environment = {}
    for name, report_field in count_fields.items():
        raw_value = str(raw_count_environment.get(name) or "").strip()
        if not raw_value.isdigit():
            raise GatewayEnvelopePreparationV2Error(
                "gateway V2 worker count environment is invalid"
            )
        count = int(raw_value)
        if not 1 <= count <= 500 or report.get(report_field) != count:
            raise GatewayEnvelopePreparationV2Error(
                "gateway V2 worker count environment differs from sealed profiles"
            )
        count_environment[name] = str(count)
    raw_deferred_roles = report.get("deferred_worker_fleet_roles")
    if not isinstance(raw_deferred_roles, list) or any(
        not isinstance(role, str) for role in raw_deferred_roles
    ):
        raise GatewayEnvelopePreparationV2Error(
            "gateway V2 deferred worker fleet state is invalid"
        )
    try:
        deferred_roles = deferred_worker_fleet_roles(
            {
                DEFERRED_WORKER_FLEETS_ENV: ",".join(raw_deferred_roles),
            }
        )
    except DeferredWorkerFleetConfigurationError as exc:
        raise GatewayEnvelopePreparationV2Error(
            "gateway V2 deferred worker fleet state is invalid"
        ) from exc
    if raw_deferred_roles != sorted(deferred_roles):
        raise GatewayEnvelopePreparationV2Error(
            "gateway V2 deferred worker fleet state is not canonical"
        )
    deferred_environment = canonical_deferred_worker_fleet_roles(
        deferred_roles
    )

    environment_path = Path(environment_path)
    try:
        lines = environment_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GatewayEnvelopePreparationV2Error(
            "prepared gateway parent environment is unavailable"
        ) from exc

    kept = []
    removed_names = set()
    removed_line_count = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            kept.append(raw_line)
            continue
        candidate = (
            line[len("export ") :].strip()
            if line.startswith("export ")
            else line
        )
        try:
            parts = shlex.split(candidate, posix=True)
        except ValueError as exc:
            raise GatewayEnvelopePreparationV2Error(
                "prepared gateway parent environment is malformed"
            ) from exc
        if len(parts) != 1 or "=" not in parts[0]:
            raise GatewayEnvelopePreparationV2Error(
                "prepared gateway parent environment is malformed"
            )
        name, value = parts[0].split("=", 1)
        if name in count_environment or name == DEFERRED_WORKER_FLEETS_ENV:
            continue
        if name in remove_names or credential_reference_hash(value) in remove_refs:
            removed_names.add(name)
            removed_line_count += 1
            continue
        kept.append(raw_line)
    kept.extend(
        "export %s=%s" % (name, shlex.quote(value))
        for name, value in sorted(count_environment.items())
    )
    if deferred_environment:
        kept.append(
            "export %s=%s"
            % (
                DEFERRED_WORKER_FLEETS_ENV,
                shlex.quote(deferred_environment),
            )
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".gateway-env-scrub.", dir=str(environment_path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("\n".join(kept).rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, environment_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return {
        "removed_line_count": removed_line_count,
        "removed_names": sorted(removed_names),
        "installed_count_environment": dict(sorted(count_environment.items())),
        "installed_deferred_worker_fleet_roles": sorted(deferred_roles),
    }


def _secret(env: Mapping[str, str], names: Sequence[str]) -> tuple[str, str]:
    for name in names:
        value = str(env.get(name) or "").strip()
        if value:
            return name, value
    raise GatewayEnvelopePreparationV2Error(
        "required gateway credential is unavailable: %s" % ",".join(names)
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, sort_keys=True, indent=2)
        handle.write("\n")
    path.chmod(0o600)


def _proxy_names(
    env: Mapping[str, str], prefixes: Sequence[str]
) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for index in range(1, 501):
        for prefix in prefixes:
            name = "%s_%d" % (prefix, index)
            value = str(env.get(name) or "").strip()
            if value:
                names.setdefault(value, name)
                break
    for prefix in prefixes:
        value = str(env.get(prefix) or "").strip()
        if value:
            names.setdefault(value, prefix)
    return names


def _proxy_environment_names(
    env: Mapping[str, str], prefixes: Sequence[str]
) -> set[str]:
    names = set()
    for index in range(1, 501):
        for prefix in prefixes:
            name = "%s_%d" % (prefix, index)
            if str(env.get(name) or "").strip():
                names.add(name)
    for prefix in prefixes:
        if str(env.get(prefix) or "").strip():
            names.add(prefix)
    return names


def _worker_proxy_profile_values(
    values: Sequence[str],
    worker_count: int,
) -> tuple[str, ...]:
    """Return one sealed profile value for every required worker index."""

    configured = tuple(str(value) for value in values)
    required = int(worker_count)
    if not configured or required <= len(configured):
        return configured
    return tuple(
        configured[index % len(configured)]
        for index in range(required)
    )


def _validated_worker_proxy_configuration(
    environment: Mapping[str, str],
    *,
    proxy_fleet_probe: Optional[
        Callable[[Mapping[str, Sequence[str]]], None]
    ] = verify_worker_proxy_fleets_v2,
) -> tuple[
    Any,
    Dict[str, list[str]],
    set[str],
    Dict[str, tuple[str, ...]],
]:
    plan = build_research_lab_worker_autostart_plan(environment)
    try:
        deferred_roles = deferred_worker_fleet_roles(environment)
    except DeferredWorkerFleetConfigurationError as exc:
        raise GatewayEnvelopePreparationV2Error(str(exc)) from exc
    if not plan.hosted.enabled or not plan.scoring.enabled:
        raise GatewayEnvelopePreparationV2Error(
            "configured hosted and scoring worker fleets are required"
        )
    if not plan.hosted.proxy_values or not plan.scoring.proxy_values:
        raise GatewayEnvelopePreparationV2Error(
            "worker proxy values are required for initial V2 sealing"
        )
    configured_fleets = {
        "gateway_autoresearch": plan.hosted.proxy_values,
        "gateway_scoring": plan.scoring.proxy_values,
    }
    profile_fleets = {
        "gateway_autoresearch": _worker_proxy_profile_values(
            plan.hosted.proxy_values,
            plan.hosted.worker_count,
        ),
        "gateway_scoring": _worker_proxy_profile_values(
            plan.scoring.proxy_values,
            plan.scoring.worker_count,
        ),
    }
    commitments: Dict[str, list[str]] = {}
    for role, values in configured_fleets.items():
        commitments[role] = []
        for index, value in enumerate(values):
            try:
                _validated_tls_proxy_url(value)
            except (ProviderBrokerV2Error, ValueError) as exc:
                raise GatewayEnvelopePreparationV2Error(
                    "%s worker proxy %d from %s configuration is incompatible "
                    "with V2 provider transport"
                    % (
                        role,
                        index + 1,
                        (
                            plan.hosted.proxy_source
                            if role == "gateway_autoresearch"
                            else plan.scoring.proxy_source
                        ),
                    )
                ) from exc
    for role, values in profile_fleets.items():
        commitments[role] = [
            credential_value_hash(value)
            for value in values
        ]
    if proxy_fleet_probe is not None:
        try:
            proxy_fleet_probe(configured_fleets)
        except WorkerProxyTransportPreflightV2Error as exc:
            raise GatewayEnvelopePreparationV2Error(str(exc)) from exc
    configured_names = _proxy_environment_names(
        environment,
        (*HOSTED_PROXY_PREFIXES, *SCORING_PROXY_PREFIXES),
    )
    return plan, commitments, configured_names, profile_fleets


def prepare_gateway_envelopes_v2(
    *,
    environment: Mapping[str, str],
    kms_key_id: str,
    deploy_commit: str,
    output_dir: Path,
    kms_client: Any = None,
    proxy_fleet_probe: Optional[
        Callable[[Mapping[str, Sequence[str]]], None]
    ] = verify_worker_proxy_fleets_v2,
) -> Dict[str, Any]:
    commit = str(deploy_commit or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise GatewayEnvelopePreparationV2Error("gateway deploy commit is invalid")
    destination = Path(output_dir)
    if destination.exists():
        raise GatewayEnvelopePreparationV2Error(
            "gateway V2 envelope output already exists"
        )
    if kms_client is None:
        import boto3

        kms_client = boto3.client("kms")
    (
        plan,
        proxy_commitments,
        proxy_environment_names,
        proxy_profile_values,
    ) = (
        _validated_worker_proxy_configuration(
            environment,
            proxy_fleet_probe=proxy_fleet_probe,
        )
    )
    deferred_roles = deferred_worker_fleet_roles(environment)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gateway-v2-envelopes.", dir=parent))
    os.chmod(staging, 0o700)
    removal_names = set()
    removal_values = set()
    try:
        artifact_key = secrets.token_bytes(32)
        artifact_context = {
            "leadpoet:commit": commit,
            "leadpoet:purpose": "gateway-artifact-master-key-v2",
            "leadpoet:slot": "artifact_master_key",
        }
        _write_json(
            staging / "artifact_master_key.json",
            build_provider_envelope_v2(
                credential_slot="artifact_master_key",
                plaintext=artifact_key,
                credential_ref_hash=artifact_master_key_reference_hash(artifact_key),
                kms_key_id=kms_key_id,
                encryption_context=artifact_context,
                kms_client=kms_client,
                allow_binary=True,
            ),
        )
        del artifact_key

        boot_values: Dict[str, str] = {}
        for slot, names in _BOOT_SOURCES.items():
            source_name, value = _secret(environment, names)
            if slot not in _SHARED_PARENT_SLOTS:
                removal_names.add(source_name)
                removal_values.add(value)
            boot_values[slot] = value
            _write_json(
                staging / (slot + ".json"),
                build_provider_envelope_v2(
                    credential_slot=slot,
                    plaintext=value.encode("utf-8"),
                    credential_ref_hash=credential_reference_hash(value),
                    kms_key_id=kms_key_id,
                    encryption_context={
                        "leadpoet:commit": commit,
                        "leadpoet:purpose": "gateway-boot-credential-v2",
                        "leadpoet:slot": slot,
                    },
                    kms_client=kms_client,
                ),
            )

        for filename, (slot, names) in _SPECIAL_PROFILES.items():
            source_name, value = _secret(environment, names)
            if source_name.startswith("RESEARCH_LAB_V2_"):
                removal_names.add(source_name)
            if slot not in _SHARED_PARENT_SLOTS:
                removal_values.add(value)
            _write_json(
                staging / filename,
                build_provider_envelope_v2(
                    credential_slot=slot,
                    plaintext=value.encode("utf-8"),
                    credential_ref_hash=credential_value_hash(value),
                    kms_key_id=kms_key_id,
                    encryption_context={
                        "leadpoet:commit": commit,
                        "leadpoet:profile": filename.removesuffix(".json"),
                        "leadpoet:purpose": "gateway-provider-profile-v2",
                        "leadpoet:slot": slot,
                    },
                    kms_client=kms_client,
                ),
            )

        proxy_sources = {
            "gateway_autoresearch": _proxy_names(
                environment, HOSTED_PROXY_PREFIXES
            ),
            "gateway_scoring": _proxy_names(environment, SCORING_PROXY_PREFIXES),
        }
        fleets = {
            "gateway_autoresearch": (
                proxy_profile_values["gateway_autoresearch"],
                "autoresearch_proxy_{:02d}.json",
            ),
            "gateway_scoring": (
                proxy_profile_values["gateway_scoring"],
                "scoring_proxy_{:02d}.json",
            ),
        }
        for role, (values, filename_template) in fleets.items():
            for index, value in enumerate(values):
                source_name = proxy_sources[role].get(value)
                if not source_name:
                    raise GatewayEnvelopePreparationV2Error(
                        "worker proxy source identity is unavailable"
                    )
                removal_names.add(source_name)
                removal_values.add(value)
                _write_json(
                    staging / filename_template.format(index),
                    build_provider_envelope_v2(
                        credential_slot="egress_proxy",
                        plaintext=value.encode("utf-8"),
                        credential_ref_hash=credential_value_hash(value),
                        kms_key_id=kms_key_id,
                        encryption_context={
                            "leadpoet:commit": commit,
                            "leadpoet:purpose": "gateway-worker-egress-v2",
                            "leadpoet:role": role,
                            "leadpoet:worker_index": str(index),
                        },
                        kms_client=kms_client,
                    ),
                )
        removal_names.update(
            str(name)
            for name, value in environment.items()
            if str(value) in removal_values
        )
        removal_names.update(proxy_environment_names)
        report = {
            "schema_version": "leadpoet.gateway_envelope_transition.v2",
            "deploy_commit": commit,
            "hosted_worker_count": plan.hosted.worker_count,
            "scoring_worker_count": plan.scoring.worker_count,
            "worker_proxy_transport_policy": _WORKER_PROXY_TRANSPORT_POLICY,
            "worker_proxy_source": {
                "gateway_autoresearch": plan.hosted.proxy_source,
                "gateway_scoring": plan.scoring.proxy_source,
            },
            "worker_proxy_credential_ref_hashes": proxy_commitments,
            "deferred_worker_fleet_roles": sorted(deferred_roles),
            "required_count_environment": {
                "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": str(
                    plan.hosted.worker_count
                ),
                "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": str(
                    plan.scoring.worker_count
                ),
            },
            "plaintext_environment_names_to_remove": sorted(removal_names),
            "plaintext_credential_ref_hashes_to_remove": sorted(
                credential_reference_hash(value) for value in removal_values
            ),
            "envelope_file_count": len(list(staging.glob("*.json"))) + 1,
        }
        _write_json(staging / "gateway-v2-env-transition.json", report)
        staging.rename(destination)
        return report
    except Exception:
        for path in staging.glob("*"):
            path.unlink(missing_ok=True)
        staging.rmdir()
        raise


def install_gateway_envelopes_v2(
    *,
    environment: Mapping[str, str],
    kms_key_id: str,
    deploy_commit: str,
    install_dir: Path,
    kms_client: Any = None,
    proxy_fleet_probe: Optional[
        Callable[[Mapping[str, Sequence[str]]], None]
    ] = verify_worker_proxy_fleets_v2,
) -> Dict[str, Any]:
    """Reuse exact-commit envelopes or atomically install a complete new set."""

    destination = Path(install_dir)
    (
        plan,
        proxy_commitments,
        proxy_environment_names,
        _proxy_profile_values,
    ) = (
        _validated_worker_proxy_configuration(
            environment,
            proxy_fleet_probe=proxy_fleet_probe,
        )
    )
    deferred_roles = deferred_worker_fleet_roles(environment)
    report_path = destination / "gateway-v2-env-transition.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = None
    if (
        isinstance(report, Mapping)
        and report.get("deploy_commit") == deploy_commit
        and report.get("worker_proxy_transport_policy")
        == _WORKER_PROXY_TRANSPORT_POLICY
        and report.get("worker_proxy_credential_ref_hashes")
        == proxy_commitments
        and report.get("deferred_worker_fleet_roles")
        == sorted(deferred_roles)
        and set(proxy_environment_names).issubset(
            {
                str(name)
                for name in report.get("plaintext_environment_names_to_remove")
                or ()
            }
        )
        and report.get("hosted_worker_count") == plan.hosted.worker_count
        and report.get("scoring_worker_count") == plan.scoring.worker_count
    ):
        hosted_count = int(report.get("hosted_worker_count") or 0)
        scoring_count = int(report.get("scoring_worker_count") or 0)
        expected_names = {
            "artifact_master_key.json",
            "openrouter.json",
            "exa.json",
            "scrapingdog.json",
            "deepline.json",
            "supabase_service_role.json",
            "truelist.json",
            *set(_SPECIAL_PROFILES),
            *{
                "autoresearch_proxy_%02d.json" % index
                for index in range(hosted_count)
            },
            *{
                "scoring_proxy_%02d.json" % index
                for index in range(scoring_count)
            },
            "gateway-v2-env-transition.json",
        }
        if expected_names and all(
            (destination / name).is_file() and not (destination / name).is_symlink()
            for name in expected_names
        ):
            return {**dict(report), "status": "reused"}

    destination.mkdir(parents=True, exist_ok=True)
    os.chmod(destination, 0o700)
    staging = destination.parent / (
        ".gateway-v2-envelope-install.%s" % secrets.token_hex(8)
    )
    backup = destination.parent / (
        ".gateway-v2-envelope-backup.%s" % secrets.token_hex(8)
    )
    generated = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id=kms_key_id,
        deploy_commit=deploy_commit,
        output_dir=staging,
        kms_client=kms_client,
        proxy_fleet_probe=None,
    )
    managed_names = {
        path.name for path in staging.iterdir() if path.is_file()
    }
    old_managed = {
        path.name
        for pattern in (
            "artifact_master_key.json",
            "openrouter.json",
            "exa.json",
            "scrapingdog.json",
            "deepline.json",
            "supabase_service_role.json",
            "truelist.json",
            "benchmark_*.json",
            "stale_parent_*.json",
            "source_add_judge_*.json",
            "autoresearch_proxy_*.json",
            "scoring_proxy_*.json",
            "gateway-v2-env-transition.json",
        )
        for path in destination.glob(pattern)
        if path.is_file() and not path.is_symlink()
    }
    backup.mkdir(mode=0o700)
    try:
        for name in sorted(old_managed):
            os.replace(destination / name, backup / name)
        for name in sorted(managed_names):
            os.replace(staging / name, destination / name)
        staging.rmdir()
        shutil.rmtree(backup)
    except Exception:
        for name in sorted(managed_names):
            installed = destination / name
            if installed.exists():
                installed.unlink()
        for path in backup.iterdir():
            os.replace(path, destination / path.name)
        shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**generated, "status": "installed"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--kms-key-id", required=True)
    parser.add_argument("--deploy-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    function = install_gateway_envelopes_v2 if args.install else prepare_gateway_envelopes_v2
    keyword = "install_dir" if args.install else "output_dir"
    environment = load_environment_file(args.env_file)
    inherited_deferral = str(
        os.environ.get(DEFERRED_WORKER_FLEETS_ENV) or ""
    ).strip()
    file_deferral = str(
        environment.get(DEFERRED_WORKER_FLEETS_ENV) or ""
    ).strip()
    if inherited_deferral and file_deferral:
        try:
            inherited_roles = deferred_worker_fleet_roles(
                {DEFERRED_WORKER_FLEETS_ENV: inherited_deferral}
            )
            file_roles = deferred_worker_fleet_roles(
                {DEFERRED_WORKER_FLEETS_ENV: file_deferral}
            )
        except DeferredWorkerFleetConfigurationError as exc:
            raise GatewayEnvelopePreparationV2Error(str(exc)) from exc
        if inherited_roles != file_roles:
            raise GatewayEnvelopePreparationV2Error(
                "inherited and prepared deferred worker fleet state differ"
            )
    requested_deferral = inherited_deferral or file_deferral
    if requested_deferral:
        try:
            requested_roles = deferred_worker_fleet_roles(
                {DEFERRED_WORKER_FLEETS_ENV: requested_deferral}
            )
        except DeferredWorkerFleetConfigurationError as exc:
            raise GatewayEnvelopePreparationV2Error(str(exc)) from exc
        environment[DEFERRED_WORKER_FLEETS_ENV] = (
            canonical_deferred_worker_fleet_roles(requested_roles)
        )
    schema_result = (
        verify_required_supabase_v2_schema(environment) if args.install else None
    )
    result = function(
        environment=environment,
        kms_key_id=args.kms_key_id,
        deploy_commit=args.deploy_commit,
        kms_client=None,
        **{keyword: args.output_dir},
    )
    if schema_result is not None:
        result = {**result, "supabase_v2_schema": schema_result}
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
