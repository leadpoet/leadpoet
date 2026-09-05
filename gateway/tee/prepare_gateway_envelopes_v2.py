"""Prepare KMS-sealed gateway V2 boot and scoring-proxy envelopes.

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
import sys
import tempfile
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from gateway.research_lab.config import (
    LEGACY_SCORING_PROXY_PREFIXES,
    SCORING_PROXY_PREFIXES,
    V2_SCORING_PROXY_PREFIXES,
    resolve_worker_process_count,
)
from gateway.tee.artifact_vault_v2 import artifact_master_key_reference_hash
from gateway.tee.host_memory_guard_v2 import cleanup_stale_vsock_probes
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
from gateway.utils.tee_kms_provision_v2 import (
    build_provider_envelope_v2,
    validate_provider_envelope,
)


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
_WORKER_PROXY_TRANSPORT_POLICY = "authenticated_http_or_https_connect.v2"
_GATEWAY_RESTART_ENVELOPE_STAGE = "v2_credential_envelope_preparation"
_OBSOLETE_WORKER_ENVIRONMENT = frozenset(
    {
        "GATEWAY_V2_DEFER_WORKER_FLEETS",
        "RESEARCH_LAB_AUTO_START_WORKERS",
        "RESEARCH_LAB_AUTO_START_HOSTED_WORKERS",
        "RESEARCH_LAB_AUTO_START_SCORING_WORKERS",
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT",
    }
)
_OBSOLETE_AUTORESEARCH_PROXY_PREFIXES = (
    "RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY",
    "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY",
)
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


def cleanup_stale_gateway_restart_probes_v2() -> list[dict[str, object]]:
    """Clean only the historical EOF-spinning probe during gateway restart."""

    if os.environ.get("GATEWAY_DEPLOY_STAGE") != _GATEWAY_RESTART_ENVELOPE_STAGE:
        return []
    cleaned = cleanup_stale_vsock_probes()
    print(
        "GATEWAY_RESTART_STALE_PROBE_CLEANUP "
        + json.dumps(
            {
                "cleaned": cleaned,
                "schema_version": "leadpoet.gateway_restart_probe_cleanup.v1",
                "status": "ready",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )
    return cleaned


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
    """Install the scoring capacity and remove parent plaintext aliases."""

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
        if name in count_environment:
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


_SCORING_PROXY_CONFIGURATION = {
    "legacy_prefixes": LEGACY_SCORING_PROXY_PREFIXES,
    "process_count_environment": "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT",
    "required_v2_environment": "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1",
}


def _validate_v2_proxy_migration_capacity(
    environment: Mapping[str, str],
    *,
    proxy_source: str,
    selected_profile_count: int,
) -> None:
    """Reject an implicit worker-capacity reduction during V2 migration."""

    process_count_environment = str(
        _SCORING_PROXY_CONFIGURATION["process_count_environment"]
    )
    if proxy_source != "v2_tls" or str(
        environment.get(process_count_environment) or ""
    ).strip():
        return
    legacy_profile_count = len(
        _proxy_names(
            environment,
            _SCORING_PROXY_CONFIGURATION["legacy_prefixes"],
        )
    )
    if legacy_profile_count <= selected_profile_count:
        return
    raise GatewayEnvelopePreparationV2Error(
        "%s V2 proxy migration would reduce worker coverage from %d legacy "
        "slots to %d selected proxy profile(s); set %s=%d explicitly and "
        "configure %s with an authenticated HTTP CONNECT or HTTPS proxy"
        % (
            "gateway_scoring",
            legacy_profile_count,
            selected_profile_count,
            process_count_environment,
            legacy_profile_count,
            _SCORING_PROXY_CONFIGURATION["required_v2_environment"],
        )
    )


def _preferred_scoring_proxy_configuration(
    environment: Mapping[str, str],
) -> tuple[tuple[str, ...], str]:
    v2_values = tuple(
        _proxy_names(environment, V2_SCORING_PROXY_PREFIXES)
    )
    if v2_values:
        return v2_values, "v2_tls"
    legacy_values = tuple(
        _proxy_names(environment, LEGACY_SCORING_PROXY_PREFIXES)
    )
    if legacy_values:
        return legacy_values, "legacy"
    return (), "none"


def _validated_worker_proxy_configuration(
    environment: Mapping[str, str],
    *,
    proxy_fleet_probe: Optional[
        Callable[
            [Mapping[str, Sequence[str]]],
            Optional[Mapping[str, Sequence[str]]],
        ]
    ] = verify_worker_proxy_fleets_v2,
) -> tuple[
    Dict[str, Any],
    Dict[str, list[str]],
    set[str],
    Dict[str, tuple[str, ...]],
    Dict[str, Dict[str, int]],
    Dict[str, tuple[str, ...]],
]:
    scoring_values, scoring_source = _preferred_scoring_proxy_configuration(
        environment
    )
    if not scoring_values:
        raise GatewayEnvelopePreparationV2Error(
            "scoring proxy values are required for V2 sealing"
        )
    raw_count = str(
        environment.get("RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT") or "0"
    ).strip()
    try:
        requested_count = int(raw_count)
    except ValueError as exc:
        raise GatewayEnvelopePreparationV2Error(
            "scoring worker capacity is invalid"
        ) from exc
    scoring_count = resolve_worker_process_count(
        requested_count, len(scoring_values), minimum=0
    )
    if not 1 <= scoring_count <= 500:
        raise GatewayEnvelopePreparationV2Error(
            "scoring worker capacity is invalid"
        )
    configured_fleets = {"gateway_scoring": scoring_values}
    proxy_sources = {"gateway_scoring": scoring_source}
    _validate_v2_proxy_migration_capacity(
        environment,
        proxy_source=scoring_source,
        selected_profile_count=len(scoring_values),
    )
    commitments: Dict[str, list[str]] = {}
    for role, values in configured_fleets.items():
        commitments[role] = []
        for index, value in enumerate(values):
            try:
                _validated_tls_proxy_url(value)
            except (ProviderBrokerV2Error, ValueError) as exc:
                raise GatewayEnvelopePreparationV2Error(
                    "%s worker proxy %d from %s configuration is incompatible "
                    "with V2 provider transport; configure %s with authenticated "
                    "HTTP CONNECT or HTTPS transport and set the intended worker capacity "
                    "in %s (%s)"
                    % (
                        role,
                        index + 1,
                        proxy_sources[role],
                        _SCORING_PROXY_CONFIGURATION["required_v2_environment"],
                        _SCORING_PROXY_CONFIGURATION["process_count_environment"],
                        str(exc),
                    )
                ) from exc

    verified_fleets = {
        role: tuple(str(value) for value in values)
        for role, values in configured_fleets.items()
    }
    if proxy_fleet_probe is not None:
        try:
            probe_result = proxy_fleet_probe(configured_fleets)
        except WorkerProxyTransportPreflightV2Error as exc:
            raise GatewayEnvelopePreparationV2Error(
                "%s; required proxy environment is %s"
                % (
                    str(exc),
                    _SCORING_PROXY_CONFIGURATION["required_v2_environment"],
                )
            ) from exc
        if probe_result is not None:
            if (
                not isinstance(probe_result, Mapping)
                or set(probe_result) != set(configured_fleets)
            ):
                raise GatewayEnvelopePreparationV2Error(
                    "worker proxy preflight returned an invalid fleet selection"
                )
            selected_fleets = {}
            for role, configured_values in configured_fleets.items():
                raw_values = probe_result.get(role)
                if isinstance(raw_values, (str, bytes)) or not isinstance(
                    raw_values, Sequence
                ):
                    raise GatewayEnvelopePreparationV2Error(
                        "worker proxy preflight returned an invalid fleet selection"
                    )
                selected_values = tuple(str(value) for value in raw_values)
                if not selected_values:
                    raise GatewayEnvelopePreparationV2Error(
                        "%s worker proxy fleet has no verified profiles" % role
                    )
                configured_iterator = iter(configured_values)
                if not all(
                    any(
                        configured_value == selected_value
                        for configured_value in configured_iterator
                    )
                    for selected_value in selected_values
                ):
                    raise GatewayEnvelopePreparationV2Error(
                        "worker proxy preflight returned an invalid fleet selection"
                    )
                selected_fleets[role] = selected_values
            verified_fleets = selected_fleets

    profile_fleets = {
        "gateway_scoring": _worker_proxy_profile_values(
            verified_fleets["gateway_scoring"],
            scoring_count,
        ),
    }
    for role, values in profile_fleets.items():
        commitments[role] = [
            credential_value_hash(value)
            for value in values
        ]
    worker_proxy_profile_counts = {
        role: {
            "configured": len(configured_fleets[role]),
            "verified": len(verified_fleets[role]),
            "quarantined": (
                len(configured_fleets[role]) - len(verified_fleets[role])
            ),
            "sealed_worker_slots": len(profile_fleets[role]),
        }
        for role in configured_fleets
    }
    configured_names = _proxy_environment_names(
        environment,
        SCORING_PROXY_PREFIXES,
    )
    return (
        {
            "worker_count": scoring_count,
            "proxy_source": scoring_source,
        },
        commitments,
        configured_names,
        profile_fleets,
        worker_proxy_profile_counts,
        verified_fleets,
    )


def prepare_gateway_envelopes_v2(
    *,
    environment: Mapping[str, str],
    kms_key_id: str,
    deploy_commit: str,
    output_dir: Path,
    kms_client: Any = None,
    artifact_master_key_envelope: Optional[Mapping[str, Any]] = None,
    proxy_fleet_probe: Optional[
        Callable[
            [Mapping[str, Sequence[str]]],
            Optional[Mapping[str, Sequence[str]]],
        ]
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
        scoring_configuration,
        proxy_commitments,
        proxy_environment_names,
        proxy_profile_values,
        worker_proxy_profile_counts,
        _verified_proxy_fleets,
    ) = (
        _validated_worker_proxy_configuration(
            environment,
            proxy_fleet_probe=proxy_fleet_probe,
        )
    )
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gateway-v2-envelopes.", dir=parent))
    os.chmod(staging, 0o700)
    removal_names = set(_OBSOLETE_WORKER_ENVIRONMENT)
    removal_values = set()
    obsolete_proxy_names = _proxy_environment_names(
        environment,
        _OBSOLETE_AUTORESEARCH_PROXY_PREFIXES,
    )
    removal_names.update(obsolete_proxy_names)
    removal_values.update(
        str(environment[name])
        for name in obsolete_proxy_names
        if str(environment.get(name) or "")
    )
    try:
        if artifact_master_key_envelope is None:
            artifact_key = secrets.token_bytes(32)
            artifact_envelope = build_provider_envelope_v2(
                credential_slot="artifact_master_key",
                plaintext=artifact_key,
                credential_ref_hash=artifact_master_key_reference_hash(artifact_key),
                kms_key_id=kms_key_id,
                encryption_context={
                    "leadpoet:key-lineage": "gateway-production-v2",
                    "leadpoet:purpose": "gateway-artifact-master-key-v2",
                    "leadpoet:slot": "artifact_master_key",
                },
                kms_client=kms_client,
                allow_binary=True,
            )
            del artifact_key
        else:
            try:
                normalized_artifact_envelope = validate_provider_envelope(
                    artifact_master_key_envelope
                )
            except Exception as exc:
                raise GatewayEnvelopePreparationV2Error(
                    "existing artifact master key envelope is invalid"
                ) from exc
            if (
                normalized_artifact_envelope["credential_slot"]
                != "artifact_master_key"
                or normalized_artifact_envelope["encryption_context"].get(
                    "leadpoet:purpose"
                )
                != "gateway-artifact-master-key-v2"
                or normalized_artifact_envelope["encryption_context"].get(
                    "leadpoet:slot"
                )
                != "artifact_master_key"
            ):
                raise GatewayEnvelopePreparationV2Error(
                    "existing artifact master key envelope purpose is invalid"
                )
            artifact_envelope = dict(artifact_master_key_envelope)
        _write_json(staging / "artifact_master_key.json", artifact_envelope)

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
            "gateway_scoring": _proxy_names(environment, SCORING_PROXY_PREFIXES),
        }
        fleets = {
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
            "schema_version": "leadpoet.gateway_envelope_transition.v3",
            "deploy_commit": commit,
            "scoring_worker_count": scoring_configuration["worker_count"],
            "worker_proxy_transport_policy": _WORKER_PROXY_TRANSPORT_POLICY,
            "worker_proxy_source": {
                "gateway_scoring": scoring_configuration["proxy_source"],
            },
            "worker_proxy_credential_ref_hashes": proxy_commitments,
            "worker_proxy_profile_counts": worker_proxy_profile_counts,
            "artifact_master_key_ref_hash": artifact_envelope[
                "credential_ref_hash"
            ],
            "required_count_environment": {
                "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": str(
                    scoring_configuration["worker_count"]
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
        Callable[
            [Mapping[str, Sequence[str]]],
            Optional[Mapping[str, Sequence[str]]],
        ]
    ] = verify_worker_proxy_fleets_v2,
) -> Dict[str, Any]:
    """Reuse exact-commit envelopes or atomically install a complete new set."""

    destination = Path(install_dir)
    (
        scoring_configuration,
        proxy_commitments,
        proxy_environment_names,
        _proxy_profile_values,
        worker_proxy_profile_counts,
        verified_proxy_fleets,
    ) = (
        _validated_worker_proxy_configuration(
            environment,
            proxy_fleet_probe=proxy_fleet_probe,
        )
    )
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
        and report.get("worker_proxy_profile_counts")
        == worker_proxy_profile_counts
        and report.get("schema_version")
        == "leadpoet.gateway_envelope_transition.v3"
        and set(proxy_environment_names).issubset(
            {
                str(name)
                for name in report.get("plaintext_environment_names_to_remove")
                or ()
            }
        )
        and report.get("scoring_worker_count")
        == scoring_configuration["worker_count"]
    ):
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
    artifact_envelope_path = destination / "artifact_master_key.json"
    artifact_master_key_envelope = None
    if artifact_envelope_path.exists() or artifact_envelope_path.is_symlink():
        if (
            artifact_envelope_path.is_symlink()
            or not artifact_envelope_path.is_file()
        ):
            raise GatewayEnvelopePreparationV2Error(
                "existing artifact master key envelope is not a regular file"
            )
        try:
            artifact_master_key_envelope = json.loads(
                artifact_envelope_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise GatewayEnvelopePreparationV2Error(
                "existing artifact master key envelope cannot be read"
            ) from exc
    generated = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id=kms_key_id,
        deploy_commit=deploy_commit,
        output_dir=staging,
        kms_client=kms_client,
        artifact_master_key_envelope=artifact_master_key_envelope,
        proxy_fleet_probe=lambda _fleets: verified_proxy_fleets,
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
    if args.install:
        cleanup_stale_gateway_restart_probes_v2()
    function = install_gateway_envelopes_v2 if args.install else prepare_gateway_envelopes_v2
    keyword = "install_dir" if args.install else "output_dir"
    environment = load_environment_file(args.env_file)
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
