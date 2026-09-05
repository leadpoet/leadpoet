"""Gateway/validator deploy readiness checks.

These helpers keep the production resume decision tied to explicit source and
PCR0 evidence instead of ad hoc operator inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from gateway.build_info import UNKNOWN, get_build_info
from gateway.tee.topology import ROLE_SPECS


DEFAULT_DEPLOY_READINESS_MANIFEST = "/home/ec2-user/gateway/deploy_readiness.json"
DEPLOY_READINESS_MANIFEST_ENV = "RESEARCH_LAB_DEPLOY_READINESS_MANIFEST"
DEFAULT_DOCKER_MIN_FREE_GB = 5.0
DEFAULT_DOCKER_HEALTH_TIMEOUT_SECONDS = 60
DEPLOY_READINESS_V2_SCHEMA_VERSION = "leadpoet.deploy_readiness.v2"
DEPLOY_READINESS_TRANSITION_SCHEMA_VERSION = "leadpoet.deploy_readiness.transition.v1"
GATEWAY_READINESS_EVIDENCE_V2_SCHEMA_VERSION = (
    "leadpoet.gateway_deploy_readiness_evidence.v2"
)
VALIDATOR_READINESS_EVIDENCE_V2_SCHEMA_VERSION = (
    "leadpoet.validator_deploy_readiness_evidence.v2"
)
GATEWAY_READINESS_OBSERVATION_V2_SCHEMA_VERSION = (
    "leadpoet.gateway_deploy_readiness_observation.v2"
)
VALIDATOR_READINESS_OBSERVATION_V2_SCHEMA_VERSION = (
    "leadpoet.validator_deploy_readiness_observation.v2"
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_V2_GATEWAY_ROLES = tuple(sorted(ROLE_SPECS))
_V2_REQUIRED_CHECKS = (
    "gateway_source_commit_matches_expected",
    "gateway_build_commit_matches_expected",
    "gateway_release_channel_matches_expected",
    "gateway_compact_lineage_matches_release_channel",
    "gateway_role_boots_match_release_channel",
    "gateway_role_config_hashes_match_runtime_documents",
    "gateway_runtime_health_matches_role_boots",
    "gateway_coordinator_attestation_matches_role_boot",
    "validator_host_commit_matches_expected",
    "validator_release_channel_matches_expected",
    "validator_boot_matches_release_channel",
    "validator_config_hash_matches_runtime_document",
    "gateway_validator_channel_hashes_match",
    "gateway_validator_lineage_hashes_match",
)

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _gateway_role_specs(
    expected_historical_topology_hash: str | None,
) -> Mapping[str, Mapping[str, Any]]:
    if expected_historical_topology_hash is None:
        return ROLE_SPECS
    from gateway.tee.release_manifest_v2 import historical_three_role_specs

    return historical_three_role_specs(
        expected_topology_hash=expected_historical_topology_hash
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "none", "null", "undefined"}:
        return None
    return text


def normalize_commit(value: Any) -> str | None:
    text = clean_string(value)
    return text.lower() if text else None


def normalize_pcr0(value: Any) -> str | None:
    text = clean_string(value)
    if not text:
        return None
    return text.lower()


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return default


def _parse_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def default_manifest_path() -> Path:
    return Path(os.getenv(DEPLOY_READINESS_MANIFEST_ENV, DEFAULT_DEPLOY_READINESS_MANIFEST)).expanduser()


def _source_commit_candidates() -> list[Path]:
    paths: list[Path] = []
    explicit = clean_string(os.getenv("GATEWAY_SOURCE_COMMIT_FILE"))
    if explicit:
        paths.append(Path(explicit).expanduser())
    module_dir = Path(__file__).resolve().parent
    paths.extend([Path.cwd() / ".source_commit", module_dir / ".source_commit", module_dir.parent / ".source_commit"])
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve() if path.exists() else path.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def read_source_commit() -> tuple[str | None, str | None]:
    for path in _source_commit_candidates():
        try:
            value = normalize_commit(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if value:
            return value, str(path)
    return None, None


def _allowlist_file_candidates() -> list[Path]:
    explicit = clean_string(os.getenv("PCR0_ALLOWLIST_FILE"))
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit).expanduser())
    module_dir = Path(__file__).resolve().parent
    paths.extend(
        [
            Path.cwd() / "pcr0_allowlist.json",
            module_dir.parent / "pcr0_allowlist.json",
            module_dir / "pcr0_allowlist.json",
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve() if path.exists() else path.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def load_local_allowlist_entries(role: str) -> tuple[list[dict[str, Any]], str | None]:
    key = f"{role.strip().lower()}_pcr0"
    for path in _allowlist_file_candidates():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        except json.JSONDecodeError:
            continue
        entries = doc.get(key)
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)], str(path)
    return [], None


def extract_allowlist_entry_commit(entry: Mapping[str, Any]) -> str | None:
    for key in ("commit_hash", "git_commit_sha", "git_commit", "commit"):
        commit = normalize_commit(entry.get(key))
        if commit:
            return commit
    notes = entry.get("notes")
    if isinstance(notes, str):
        marker = "commit "
        idx = notes.lower().find(marker)
        if idx >= 0:
            candidate = notes[idx + len(marker) :].split()[0].strip(".,;:)(")
            return normalize_commit(candidate)
    return None


def _static_allowlist_status(pcr0: str | None, *, role: str) -> dict[str, Any]:
    normalized = normalize_pcr0(pcr0)
    entries, local_path = load_local_allowlist_entries(role)
    matching_entries = [
        entry for entry in entries if normalize_pcr0(entry.get("pcr0")) == normalized
    ] if normalized else []
    allowed_values: list[str] = []
    allowed_source = "unavailable"
    allowed_error: str | None = None
    try:
        from leadpoet_canonical.nitro import get_allowed_pcr0_values

        allowed_values = [normalize_pcr0(value) or "" for value in get_allowed_pcr0_values(role)]
        allowed_source = "leadpoet_canonical.nitro"
    except Exception as exc:  # noqa: BLE001 - status report should explain, not raise.
        allowed_error = str(exc)[:500]

    allowed = bool(normalized and normalized in set(allowed_values))
    entry_commits = [commit for entry in matching_entries if (commit := extract_allowlist_entry_commit(entry))]
    return {
        "role": role,
        "pcr0": normalized,
        "allowed": allowed,
        "allowed_count": len([value for value in allowed_values if value]),
        "allowed_source": allowed_source,
        "allowed_error": allowed_error,
        "local_allowlist_path": local_path,
        "local_match_count": len(matching_entries),
        "matched_entry_commits": entry_commits,
        "matched_entries": matching_entries,
    }


def _dynamic_validator_status(
    pcr0: str | None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_pcr0(pcr0)
    try:
        from gateway.utils.pcr0_builder import get_cache_status, verify_pcr0
    except Exception as exc:  # noqa: BLE001 - gateway imports can be unavailable in tests.
        return {
            "available": False,
            "valid": False,
            "error": str(exc)[:500],
            "cache_status": None,
        }
    verification = verify_pcr0(
        normalized or "",
        expected_commit=normalize_commit(expected_commit) or "",
    )
    return {
        "available": True,
        "valid": bool(verification.get("valid")),
        "verification": verification,
        "cache_status": get_cache_status(),
    }


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    ok: bool,
    *,
    severity: str = "error",
    detail: str | None = None,
    expected: Any = None,
    actual: Any = None,
) -> None:
    checks.append(
        {
            "name": name,
            "ok": bool(ok),
            "severity": severity,
            "detail": detail,
            "expected": expected,
            "actual": actual,
        }
    )


def _commit_matches(expected: str | None, actual: str | None) -> bool:
    if not expected or not actual:
        return False
    return actual.startswith(expected) or expected.startswith(actual)


def _pcr0_matches(expected: str | None, actual: str | None) -> bool:
    return bool(expected and actual and expected == actual)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _exact_commit(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _COMMIT_RE.fullmatch(normalized):
        raise RuntimeError(f"{field} must be a full lowercase commit SHA")
    return normalized


def _exact_hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(normalized):
        raise RuntimeError(f"{field} must be a canonical sha256 hash")
    return normalized


def _exact_pcr0(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _PCR0_RE.fullmatch(normalized) or normalized == "0" * 96:
        raise RuntimeError(f"{field} must be a nonzero lowercase PCR0")
    return normalized


def _v2_check(name: str) -> dict[str, Any]:
    return {"name": name, "ok": True, "severity": "error"}


def _validated_v2_release_authority(
    *,
    expected_commit: str,
    gateway_release_manifest: Mapping[str, Any],
    validator_release_manifest: Mapping[str, Any],
    compact_lineage: Mapping[str, Any],
    expected_historical_topology_hash: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate and cross-bind the current immutable V2 release authority."""

    from gateway.tee.release_channel_v2 import (
        build_historical_release_channel_v2,
        build_historical_release_lineage_v2,
        build_release_channel_v2,
        build_release_lineage_v2,
        validate_historical_release_channel_v2,
        validate_release_channel_v2,
    )
    from gateway.tee.release_lineage_v2 import (
        validate_compact_release_lineage_v2,
        validate_historical_compact_release_lineage_v2,
    )

    commit = _exact_commit(expected_commit, "expected_commit")
    if expected_historical_topology_hash is None:
        channel = validate_release_channel_v2(
            build_release_channel_v2(
                gateway_release_manifest=gateway_release_manifest,
                validator_release_manifest=validator_release_manifest,
            ),
            expected_commit=commit,
        )
        gateway_release = channel["gateway_release_manifest"]
        lineage = validate_compact_release_lineage_v2(
            compact_lineage,
            expected_current_commit=commit,
            expected_current_gateway_release_hash=gateway_release["release_hash"],
        )
        expected_current = build_release_lineage_v2(
            [channel], current_commit=commit
        )["releases"][commit]
    else:
        channel = validate_historical_release_channel_v2(
            build_historical_release_channel_v2(
                gateway_release_manifest=gateway_release_manifest,
                validator_release_manifest=validator_release_manifest,
                expected_topology_hash=expected_historical_topology_hash,
            ),
            expected_topology_hash=expected_historical_topology_hash,
            expected_commit=commit,
        )
        gateway_release = channel["gateway_release_manifest"]
        lineage = validate_historical_compact_release_lineage_v2(
            compact_lineage,
            expected_topology_hash=expected_historical_topology_hash,
            expected_current_commit=commit,
            expected_current_gateway_release_hash=gateway_release["release_hash"],
        )
        expected_current = build_historical_release_lineage_v2(
            [channel],
            current_commit=commit,
            expected_topology_hash=expected_historical_topology_hash,
        )["releases"][commit]
    if lineage["releases"].get(commit) != expected_current:
        raise RuntimeError(
            "compact release lineage current entry differs from release channel"
        )
    return channel, lineage, expected_current


def _verified_boot_summary(
    *,
    boot_identity: Mapping[str, Any],
    expectation: Mapping[str, Any],
    physical_role: str,
    boot_verifier: Any = None,
) -> dict[str, str]:
    from leadpoet_canonical.attested_v2 import verify_boot_identity_nitro

    if not isinstance(boot_identity, Mapping):
        raise RuntimeError(f"{physical_role} boot identity is unavailable")
    boot = dict(boot_identity)
    required = {
        "role": expectation["service_role"],
        "physical_role": expectation["physical_role"],
        "commit_sha": expectation["commit_sha"],
        "pcr0": expectation["pcr0"],
        "build_manifest_hash": expectation["build_manifest_hash"],
        "dependency_lock_hash": expectation["dependency_lock_hash"],
    }
    for field, expected in required.items():
        if boot.get(field) != expected:
            raise RuntimeError(
                f"{physical_role} boot differs from release channel at {field}"
            )
    verifier = boot_verifier or verify_boot_identity_nitro
    verifier(
        boot,
        expected_pcr0=str(expectation["pcr0"]),
        certificate_validity_at_attestation_time=True,
    )
    return {
        "commit_sha": _exact_commit(boot.get("commit_sha"), "boot commit"),
        "pcr0": _exact_pcr0(boot.get("pcr0"), "boot PCR0"),
        "boot_identity_hash": _exact_hash(
            boot.get("boot_identity_hash"), "boot identity hash"
        ),
        "build_manifest_hash": _exact_hash(
            boot.get("build_manifest_hash"), "boot build manifest hash"
        ),
        "dependency_lock_hash": _exact_hash(
            boot.get("dependency_lock_hash"), "boot dependency lock hash"
        ),
        "config_hash": _exact_hash(boot.get("config_hash"), "boot config hash"),
    }


def _runtime_readiness_boot_hashes(
    value: Mapping[str, Any],
    *,
    role_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    fields = {
        "schema_version",
        "status",
        "provider_registry_hash",
        "roles",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeError("gateway runtime readiness fields are invalid")
    if (
        value.get("schema_version")
        != "leadpoet.gateway_v2_runtime_readiness.v2"
        or value.get("status") != "ready"
    ):
        raise RuntimeError("gateway runtime readiness is not successful")
    _exact_hash(value.get("provider_registry_hash"), "provider registry hash")
    rows = value.get("roles")
    expected_roles = tuple(sorted(role_specs))
    if not isinstance(rows, list) or len(rows) != len(expected_roles):
        raise RuntimeError("gateway runtime readiness roles are incomplete")
    hashes: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "physical_role",
            "role",
            "worker_count",
            "configured_worker_count",
            "boot_identity_hash",
        }:
            raise RuntimeError("gateway runtime readiness role is invalid")
        role = str(row.get("physical_role") or "")
        if (
            role not in role_specs
            or row.get("role") != role_specs[role]["service_role"]
        ):
            raise RuntimeError("gateway runtime readiness role identity differs")
        if role in hashes:
            raise RuntimeError("gateway runtime readiness role is duplicated")
        if not isinstance(row.get("worker_count"), int) or not isinstance(
            row.get("configured_worker_count"), int
        ):
            raise RuntimeError("gateway runtime readiness worker counts are invalid")
        hashes[role] = _exact_hash(
            row.get("boot_identity_hash"), f"{role} runtime boot identity hash"
        )
    if set(hashes) != set(expected_roles):
        raise RuntimeError("gateway runtime readiness roles are incomplete")
    return hashes


def build_gateway_v2_readiness_evidence(
    *,
    expected_commit: str,
    source_commit: str,
    build_commit: str,
    gateway_release_manifest: Mapping[str, Any],
    validator_release_manifest: Mapping[str, Any],
    compact_lineage: Mapping[str, Any],
    boot_identities: Mapping[str, Mapping[str, Any]],
    expected_role_config_hashes: Mapping[str, Any],
    runtime_readiness: Mapping[str, Any],
    coordinator_attestation_pcr0: str,
    boot_verifier: Any = None,
    expected_historical_topology_hash: str | None = None,
) -> dict[str, Any]:
    """Build redacted evidence after fresh verification of every gateway role."""

    commit = _exact_commit(expected_commit, "expected_commit")
    if _exact_commit(source_commit, "gateway source commit") != commit:
        raise RuntimeError("gateway source commit differs from expected commit")
    if _exact_commit(build_commit, "gateway build commit") != commit:
        raise RuntimeError("gateway build commit differs from expected commit")
    channel, lineage, _ = _validated_v2_release_authority(
        expected_commit=commit,
        gateway_release_manifest=gateway_release_manifest,
        validator_release_manifest=validator_release_manifest,
        compact_lineage=compact_lineage,
        expected_historical_topology_hash=expected_historical_topology_hash,
    )
    role_specs = _gateway_role_specs(expected_historical_topology_hash)
    gateway_roles = tuple(sorted(role_specs))
    if set(boot_identities) != set(gateway_roles):
        raise RuntimeError("gateway boot identities do not cover every role")
    if set(expected_role_config_hashes) != set(gateway_roles):
        raise RuntimeError("gateway runtime documents do not cover every role")
    runtime_role_boot_hashes = _runtime_readiness_boot_hashes(
        runtime_readiness,
        role_specs=role_specs,
    )
    roles: dict[str, dict[str, str]] = {}
    from gateway.tee.release_manifest_v2 import (
        historical_role_expectation,
        role_expectation,
    )

    for role in gateway_roles:
        if expected_historical_topology_hash is None:
            expectation = role_expectation(
                channel["gateway_release_manifest"], role
            )
        else:
            expectation = historical_role_expectation(
                channel["gateway_release_manifest"],
                role,
                expected_topology_hash=expected_historical_topology_hash,
            )
        summary = _verified_boot_summary(
            boot_identity=boot_identities[role],
            expectation=expectation,
            physical_role=role,
            boot_verifier=boot_verifier,
        )
        expected_config_hash = _exact_hash(
            expected_role_config_hashes[role], f"{role} expected config hash"
        )
        if summary["config_hash"] != expected_config_hash:
            raise RuntimeError(
                f"gateway {role} boot config differs from runtime document"
            )
        if runtime_role_boot_hashes.get(role) != summary["boot_identity_hash"]:
            raise RuntimeError(
                f"gateway runtime health differs from fresh {role} boot"
            )
        summary.update(
            {
                "build_identity_hash": _exact_hash(
                    expectation["build_identity_hash"],
                    f"{role} build identity hash",
                ),
                "release_hash": _exact_hash(
                    expectation["release_hash"], f"{role} release hash"
                ),
            }
        )
        roles[role] = summary
    coordinator_attestation = _exact_pcr0(
        coordinator_attestation_pcr0,
        "coordinator attestation PCR0",
    )
    if coordinator_attestation != roles["gateway_coordinator"]["pcr0"]:
        raise RuntimeError("coordinator /attest PCR0 differs from fresh V2 boot")
    validator_release = channel["validator_release_manifest"]
    return {
        "schema_version": GATEWAY_READINESS_EVIDENCE_V2_SCHEMA_VERSION,
        "commit_sha": commit,
        "source_commit_sha": commit,
        "build_commit_sha": commit,
        "channel_hash": _exact_hash(channel["channel_hash"], "channel hash"),
        "gateway_release_hash": _exact_hash(
            channel["gateway_release_manifest"]["release_hash"],
            "gateway release hash",
        ),
        "validator_release_manifest_hash": _exact_hash(
            validator_release["release_manifest_hash"],
            "validator release manifest hash",
        ),
        "validator_release_hash": _exact_hash(
            validator_release["release"]["release_hash"],
            "validator release hash",
        ),
        "lineage_hash": _exact_hash(lineage["lineage_hash"], "lineage hash"),
        "roles": {role: roles[role] for role in gateway_roles},
        "coordinator_attestation_pcr0": coordinator_attestation,
    }


def build_validator_v2_readiness_evidence(
    *,
    expected_commit: str,
    host_commit: str,
    gateway_release_manifest: Mapping[str, Any],
    validator_release_manifest: Mapping[str, Any],
    compact_lineage: Mapping[str, Any],
    boot_identity: Mapping[str, Any],
    expected_config_hash: str,
    boot_verifier: Any = None,
    expected_historical_topology_hash: str | None = None,
) -> dict[str, Any]:
    """Build redacted evidence after fresh verification of validator_weights."""

    commit = _exact_commit(expected_commit, "expected_commit")
    if _exact_commit(host_commit, "validator host commit") != commit:
        raise RuntimeError("validator host commit differs from expected commit")
    channel, lineage, _ = _validated_v2_release_authority(
        expected_commit=commit,
        gateway_release_manifest=gateway_release_manifest,
        validator_release_manifest=validator_release_manifest,
        compact_lineage=compact_lineage,
        expected_historical_topology_hash=expected_historical_topology_hash,
    )
    validator_release = channel["validator_release_manifest"]
    validator_authority = validator_release["release"]
    expectation = {
        "physical_role": "validator_weights",
        "service_role": "validator_weights",
        "commit_sha": validator_authority["commit_sha"],
        "pcr0": validator_authority["pcr0"],
        "build_manifest_hash": validator_authority["app_manifest_hash"],
        "dependency_lock_hash": validator_authority["dependency_lock_hash"],
    }
    role = _verified_boot_summary(
        boot_identity=boot_identity,
        expectation=expectation,
        physical_role="validator_weights",
        boot_verifier=boot_verifier,
    )
    if role["config_hash"] != _exact_hash(
        expected_config_hash, "validator expected config hash"
    ):
        raise RuntimeError("validator boot config differs from runtime document")
    role["release_hash"] = _exact_hash(
        validator_authority["release_hash"], "validator role release hash"
    )
    return {
        "schema_version": VALIDATOR_READINESS_EVIDENCE_V2_SCHEMA_VERSION,
        "commit_sha": commit,
        "host_commit_sha": commit,
        "channel_hash": _exact_hash(channel["channel_hash"], "channel hash"),
        "gateway_release_hash": _exact_hash(
            channel["gateway_release_manifest"]["release_hash"],
            "gateway release hash",
        ),
        "validator_release_manifest_hash": _exact_hash(
            validator_release["release_manifest_hash"],
            "validator release manifest hash",
        ),
        "validator_release_hash": _exact_hash(
            validator_release["release"]["release_hash"],
            "validator release hash",
        ),
        "lineage_hash": _exact_hash(lineage["lineage_hash"], "lineage hash"),
        "role": role,
    }


def build_gateway_v2_readiness_evidence_from_observation(
    *,
    expected_commit: str,
    observation: Mapping[str, Any],
    expected_historical_topology_hash: str | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "source_commit",
        "build_commit",
        "gateway_release_manifest",
        "validator_release_manifest",
        "compact_lineage",
        "boot_identities",
        "expected_role_config_hashes",
        "runtime_readiness",
        "coordinator_attestation_pcr0",
    }
    if not isinstance(observation, Mapping) or set(observation) != fields:
        raise RuntimeError("gateway deploy readiness observation fields are invalid")
    if (
        observation.get("schema_version")
        != GATEWAY_READINESS_OBSERVATION_V2_SCHEMA_VERSION
    ):
        raise RuntimeError("gateway deploy readiness observation schema is invalid")
    return build_gateway_v2_readiness_evidence(
        expected_commit=expected_commit,
        source_commit=observation["source_commit"],
        build_commit=observation["build_commit"],
        gateway_release_manifest=observation["gateway_release_manifest"],
        validator_release_manifest=observation["validator_release_manifest"],
        compact_lineage=observation["compact_lineage"],
        boot_identities=observation["boot_identities"],
        expected_role_config_hashes=observation["expected_role_config_hashes"],
        runtime_readiness=observation["runtime_readiness"],
        coordinator_attestation_pcr0=observation[
            "coordinator_attestation_pcr0"
        ],
        expected_historical_topology_hash=expected_historical_topology_hash,
    )


def build_validator_v2_readiness_evidence_from_observation(
    *,
    expected_commit: str,
    observation: Mapping[str, Any],
    expected_historical_topology_hash: str | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "host_commit",
        "gateway_release_manifest",
        "validator_release_manifest",
        "compact_lineage",
        "boot_identity",
        "expected_config_hash",
    }
    if not isinstance(observation, Mapping) or set(observation) != fields:
        raise RuntimeError("validator deploy readiness observation fields are invalid")
    if (
        observation.get("schema_version")
        != VALIDATOR_READINESS_OBSERVATION_V2_SCHEMA_VERSION
    ):
        raise RuntimeError("validator deploy readiness observation schema is invalid")
    return build_validator_v2_readiness_evidence(
        expected_commit=expected_commit,
        host_commit=observation["host_commit"],
        gateway_release_manifest=observation["gateway_release_manifest"],
        validator_release_manifest=observation["validator_release_manifest"],
        compact_lineage=observation["compact_lineage"],
        boot_identity=observation["boot_identity"],
        expected_config_hash=observation["expected_config_hash"],
        expected_historical_topology_hash=expected_historical_topology_hash,
    )


def _normalize_gateway_v2_evidence(
    value: Mapping[str, Any],
    *,
    expected_historical_topology_hash: str | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "commit_sha",
        "source_commit_sha",
        "build_commit_sha",
        "channel_hash",
        "gateway_release_hash",
        "validator_release_manifest_hash",
        "validator_release_hash",
        "lineage_hash",
        "roles",
        "coordinator_attestation_pcr0",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeError("gateway V2 readiness evidence fields are invalid")
    if value.get("schema_version") != GATEWAY_READINESS_EVIDENCE_V2_SCHEMA_VERSION:
        raise RuntimeError("gateway V2 readiness evidence schema is invalid")
    commit = _exact_commit(value.get("commit_sha"), "gateway evidence commit")
    for field in ("source_commit_sha", "build_commit_sha"):
        if _exact_commit(value.get(field), field) != commit:
            raise RuntimeError(f"gateway evidence {field} differs")
    roles = value.get("roles")
    role_specs = _gateway_role_specs(expected_historical_topology_hash)
    gateway_roles = tuple(sorted(role_specs))
    if not isinstance(roles, Mapping) or set(roles) != set(gateway_roles):
        raise RuntimeError("gateway V2 readiness roles are incomplete")
    normalized_roles = {}
    for role in gateway_roles:
        summary = roles[role]
        if not isinstance(summary, Mapping) or set(summary) != {
            "commit_sha",
            "pcr0",
            "boot_identity_hash",
            "build_manifest_hash",
            "dependency_lock_hash",
            "config_hash",
            "build_identity_hash",
            "release_hash",
        }:
            raise RuntimeError(f"gateway V2 readiness role is invalid: {role}")
        role_commit = _exact_commit(summary.get("commit_sha"), f"{role} commit")
        if role_commit != commit:
            raise RuntimeError(f"gateway V2 readiness role commit differs: {role}")
        normalized_roles[role] = {
            "commit_sha": role_commit,
            "pcr0": _exact_pcr0(summary.get("pcr0"), f"{role} PCR0"),
            "boot_identity_hash": _exact_hash(
                summary.get("boot_identity_hash"), f"{role} boot identity hash"
            ),
            "build_manifest_hash": _exact_hash(
                summary.get("build_manifest_hash"), f"{role} build manifest hash"
            ),
            "dependency_lock_hash": _exact_hash(
                summary.get("dependency_lock_hash"), f"{role} dependency lock hash"
            ),
            "config_hash": _exact_hash(
                summary.get("config_hash"), f"{role} config hash"
            ),
            "build_identity_hash": _exact_hash(
                summary.get("build_identity_hash"), f"{role} build identity hash"
            ),
            "release_hash": _exact_hash(
                summary.get("release_hash"), f"{role} release hash"
            ),
        }
    coordinator_pcr0 = _exact_pcr0(
        value.get("coordinator_attestation_pcr0"),
        "coordinator attestation PCR0",
    )
    if coordinator_pcr0 != normalized_roles["gateway_coordinator"]["pcr0"]:
        raise RuntimeError("coordinator attestation PCR0 differs")
    normalized = dict(value)
    normalized.update(
        {
            "commit_sha": commit,
            "source_commit_sha": commit,
            "build_commit_sha": commit,
            "channel_hash": _exact_hash(value.get("channel_hash"), "channel hash"),
            "gateway_release_hash": _exact_hash(
                value.get("gateway_release_hash"), "gateway release hash"
            ),
            "validator_release_manifest_hash": _exact_hash(
                value.get("validator_release_manifest_hash"),
                "validator release manifest hash",
            ),
            "validator_release_hash": _exact_hash(
                value.get("validator_release_hash"), "validator release hash"
            ),
            "lineage_hash": _exact_hash(value.get("lineage_hash"), "lineage hash"),
            "roles": normalized_roles,
            "coordinator_attestation_pcr0": coordinator_pcr0,
        }
    )
    return normalized


def _normalize_validator_v2_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "commit_sha",
        "host_commit_sha",
        "channel_hash",
        "gateway_release_hash",
        "validator_release_manifest_hash",
        "validator_release_hash",
        "lineage_hash",
        "role",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeError("validator V2 readiness evidence fields are invalid")
    if value.get("schema_version") != VALIDATOR_READINESS_EVIDENCE_V2_SCHEMA_VERSION:
        raise RuntimeError("validator V2 readiness evidence schema is invalid")
    commit = _exact_commit(value.get("commit_sha"), "validator evidence commit")
    if _exact_commit(value.get("host_commit_sha"), "validator host commit") != commit:
        raise RuntimeError("validator host commit differs")
    role = value.get("role")
    if not isinstance(role, Mapping) or set(role) != {
        "commit_sha",
        "pcr0",
        "boot_identity_hash",
        "build_manifest_hash",
        "dependency_lock_hash",
        "config_hash",
        "release_hash",
    }:
        raise RuntimeError("validator V2 readiness role is invalid")
    if _exact_commit(role.get("commit_sha"), "validator boot commit") != commit:
        raise RuntimeError("validator V2 readiness boot commit differs")
    normalized = dict(value)
    normalized.update(
        {
            "commit_sha": commit,
            "host_commit_sha": commit,
            "channel_hash": _exact_hash(value.get("channel_hash"), "channel hash"),
            "gateway_release_hash": _exact_hash(
                value.get("gateway_release_hash"), "gateway release hash"
            ),
            "validator_release_manifest_hash": _exact_hash(
                value.get("validator_release_manifest_hash"),
                "validator release manifest hash",
            ),
            "validator_release_hash": _exact_hash(
                value.get("validator_release_hash"), "validator release hash"
            ),
            "lineage_hash": _exact_hash(value.get("lineage_hash"), "lineage hash"),
            "role": {
                "commit_sha": commit,
                "pcr0": _exact_pcr0(role.get("pcr0"), "validator boot PCR0"),
                "boot_identity_hash": _exact_hash(
                    role.get("boot_identity_hash"), "validator boot identity hash"
                ),
                "build_manifest_hash": _exact_hash(
                    role.get("build_manifest_hash"), "validator build manifest hash"
                ),
                "dependency_lock_hash": _exact_hash(
                    role.get("dependency_lock_hash"), "validator dependency lock hash"
                ),
                "config_hash": _exact_hash(
                    role.get("config_hash"), "validator config hash"
                ),
                "release_hash": _exact_hash(
                    role.get("release_hash"), "validator role release hash"
                ),
            },
        }
    )
    return normalized


def build_v2_deploy_readiness_manifest(
    *,
    expected_commit: str,
    gateway_evidence: Mapping[str, Any],
    validator_evidence: Mapping[str, Any],
    expected_historical_topology_hash: str | None = None,
) -> dict[str, Any]:
    """Join two freshly verified hosts into one exact-release resume authority."""

    commit = _exact_commit(expected_commit, "expected_commit")
    gateway = _normalize_gateway_v2_evidence(
        gateway_evidence,
        expected_historical_topology_hash=expected_historical_topology_hash,
    )
    validator = _normalize_validator_v2_evidence(validator_evidence)
    if gateway["commit_sha"] != commit or validator["commit_sha"] != commit:
        raise RuntimeError("deploy readiness evidence commit differs")
    for field in (
        "channel_hash",
        "gateway_release_hash",
        "validator_release_manifest_hash",
        "validator_release_hash",
        "lineage_hash",
    ):
        if gateway[field] != validator[field]:
            raise RuntimeError(f"gateway and validator evidence differ at {field}")
    body = {
        "schema_version": DEPLOY_READINESS_V2_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "ok": True,
        "enforce_resume_block": True,
        "expected_commit_sha": commit,
        "gateway": gateway,
        "validator": validator,
        "checks": [_v2_check(name) for name in _V2_REQUIRED_CHECKS],
    }
    return {**body, "manifest_hash": _canonical_hash(body)}


def build_deploy_readiness_transition_marker(
    *, expected_commit: str
) -> dict[str, Any]:
    """Build an explicit N-1-compatible block before component activation."""

    return {
        "schema_version": DEPLOY_READINESS_TRANSITION_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "ok": False,
        "enforce_resume_block": True,
        "expected_commit_sha": _exact_commit(expected_commit, "expected_commit"),
        "checks": [
            {
                "name": "canonical_restart_in_progress",
                "ok": False,
                "severity": "error",
            }
        ],
    }


def validate_v2_deploy_readiness_manifest(
    manifest: Mapping[str, Any],
    *,
    runtime_source_commit: str | None = None,
    runtime_build_commit: str | None = None,
    expected_historical_topology_hash: str | None = None,
) -> dict[str, Any]:
    """Validate current-release readiness without a wall-clock freshness guess."""

    allowed_fields = {
        "schema_version",
        "generated_at_utc",
        "ok",
        "enforce_resume_block",
        "expected_commit_sha",
        "gateway",
        "validator",
        "checks",
        "manifest_hash",
    }
    document = dict(manifest)
    document.pop("manifest_path", None)
    if set(document) != allowed_fields:
        raise RuntimeError("deploy readiness v2 manifest fields are invalid")
    if document.get("schema_version") != DEPLOY_READINESS_V2_SCHEMA_VERSION:
        raise RuntimeError("deploy readiness schema v2 is required")
    if document.get("ok") is not True or document.get("enforce_resume_block") is not True:
        raise RuntimeError("deploy readiness v2 manifest is not enforcing and successful")
    manifest_hash = _exact_hash(document.get("manifest_hash"), "manifest hash")
    body = {key: value for key, value in document.items() if key != "manifest_hash"}
    if _canonical_hash(body) != manifest_hash:
        raise RuntimeError("deploy readiness v2 manifest hash differs")
    commit = _exact_commit(document.get("expected_commit_sha"), "expected commit")
    gateway = _normalize_gateway_v2_evidence(
        document.get("gateway"),
        expected_historical_topology_hash=expected_historical_topology_hash,
    )
    validator = _normalize_validator_v2_evidence(document.get("validator"))
    if gateway["commit_sha"] != commit or validator["commit_sha"] != commit:
        raise RuntimeError("deploy readiness v2 commit differs")
    for field in (
        "channel_hash",
        "gateway_release_hash",
        "validator_release_manifest_hash",
        "validator_release_hash",
        "lineage_hash",
    ):
        if gateway[field] != validator[field]:
            raise RuntimeError(f"deploy readiness host evidence differs at {field}")
    checks = document.get("checks")
    if not isinstance(checks, list):
        raise RuntimeError("deploy readiness v2 checks are invalid")
    names = [
        check.get("name") if isinstance(check, Mapping) else None
        for check in checks
    ]
    if sorted(names) != sorted(_V2_REQUIRED_CHECKS) or len(names) != len(set(names)):
        raise RuntimeError("deploy readiness v2 required checks are incomplete")
    for check in checks:
        if (
            not isinstance(check, Mapping)
            or set(check) != {"name", "ok", "severity"}
            or check.get("ok") is not True
            or check.get("severity") != "error"
        ):
            raise RuntimeError("deploy readiness v2 required check failed")
    if runtime_source_commit is None:
        runtime_source_commit, _ = read_source_commit()
    if runtime_build_commit is None:
        runtime_build_commit = get_build_info().get("git_commit")
    if _exact_commit(runtime_source_commit, "current gateway source commit") != commit:
        raise RuntimeError("deploy readiness is stale for current gateway source")
    if _exact_commit(runtime_build_commit, "current gateway build commit") != commit:
        raise RuntimeError("deploy readiness is stale for current gateway build")
    return {**document, "manifest_hash": manifest_hash}


def _allowlist_commit_matches_runtime(status: Mapping[str, Any], runtime_commit: str | None) -> bool:
    commits = [normalize_commit(value) for value in status.get("matched_entry_commits") or []]
    commits = [value for value in commits if value]
    if not commits or not runtime_commit:
        return False
    return any(_commit_matches(commit, runtime_commit) for commit in commits)


def _truncate(value: str, limit: int = 700) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _run_command(command: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
        )
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": _truncate(exc.stdout or ""),
            "stderr": f"timeout after {timeout_seconds}s",
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout": _truncate(completed.stdout),
        "stderr": _truncate(completed.stderr),
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def _docker_base_command() -> list[str] | None:
    docker_path = shutil.which("docker")
    if docker_path:
        return [docker_path]
    sudo_path = shutil.which("sudo")
    if sudo_path and Path("/usr/bin/docker").exists():
        return [sudo_path, "-n", "/usr/bin/docker"]
    return None


def _run_docker(args: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    base = _docker_base_command()
    if base is None:
        return {
            "ok": False,
            "command": ["docker", *args],
            "returncode": None,
            "stdout": "",
            "stderr": "docker CLI not found",
            "duration_seconds": 0,
        }
    result = _run_command([*base, *args], timeout_seconds=timeout_seconds)
    if result["ok"] or base[0].endswith("sudo"):
        return result
    stderr = str(result.get("stderr") or "").lower()
    if not any(marker in stderr for marker in ("permission denied", "cannot connect", "got permission denied")):
        return result
    sudo_path = shutil.which("sudo")
    if not sudo_path:
        return result
    return _run_command([sudo_path, "-n", base[0], *args], timeout_seconds=timeout_seconds)


def _parse_docker_info(stdout: str) -> dict[str, str | None]:
    parts = str(stdout or "").strip().split("|")
    return {
        "driver": parts[0] if len(parts) > 0 and parts[0] else None,
        "docker_root": parts[1] if len(parts) > 1 and parts[1] else None,
        "server_version": parts[2] if len(parts) > 2 and parts[2] else None,
    }


def _disk_status(path: str | None, *, min_free_gb: float) -> dict[str, Any]:
    probe = Path(path or "/var/lib/docker")
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return {
            "ok": False,
            "path": str(probe),
            "error": str(exc)[:500],
            "min_free_gb": min_free_gb,
        }
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    used_gb = usage.used / (1024**3)
    return {
        "ok": free_gb >= min_free_gb,
        "path": str(probe),
        "free_gb": round(free_gb, 3),
        "used_gb": round(used_gb, 3),
        "total_gb": round(total_gb, 3),
        "used_percent": round((usage.used / usage.total) * 100, 2) if usage.total else None,
        "min_free_gb": min_free_gb,
    }


def _docker_smoke_build(*, timeout_seconds: int) -> dict[str, Any]:
    tag = f"leadpoet-deploy-readiness-smoke:{os.getpid()}-{int(time.time())}"
    with tempfile.TemporaryDirectory(prefix="leadpoet_docker_smoke_") as tmpdir:
        dockerfile = Path(tmpdir) / "Dockerfile"
        dockerfile.write_text(
            "FROM scratch\nLABEL leadpoet.deploy_readiness=1\n",
            encoding="utf-8",
        )
        build = _run_docker(
            ["build", "--quiet", "--no-cache", "-t", tag, tmpdir],
            timeout_seconds=timeout_seconds,
        )
    cleanup = _run_docker(["rmi", "-f", tag], timeout_seconds=15)
    return {
        "ok": bool(build.get("ok")),
        "tag": tag,
        "build": build,
        "cleanup": {
            "ok": bool(cleanup.get("ok")),
            "returncode": cleanup.get("returncode"),
            "stderr": cleanup.get("stderr"),
        },
    }


def docker_build_health(
    *,
    smoke_build: bool = False,
    timeout_seconds: int | None = None,
    min_free_gb: float | None = None,
) -> dict[str, Any]:
    """Report Docker availability and, optionally, exercise a tiny local build."""
    timeout = int(
        timeout_seconds
        or os.getenv("DEPLOY_READINESS_DOCKER_HEALTH_TIMEOUT_SECONDS")
        or DEFAULT_DOCKER_HEALTH_TIMEOUT_SECONDS
    )
    min_free = float(
        min_free_gb
        if min_free_gb is not None
        else _parse_float(os.getenv("DEPLOY_READINESS_DOCKER_MIN_FREE_GB"), default=DEFAULT_DOCKER_MIN_FREE_GB)
    )
    docker_cli = _docker_base_command()
    info = _run_docker(
        ["info", "--format", "{{.Driver}}|{{.DockerRootDir}}|{{.ServerVersion}}"],
        timeout_seconds=min(timeout, 15),
    )
    parsed_info = _parse_docker_info(str(info.get("stdout") or "")) if info.get("ok") else {}
    disk = _disk_status(str(parsed_info.get("docker_root") or "/var/lib/docker"), min_free_gb=min_free)
    smoke = _docker_smoke_build(timeout_seconds=timeout) if smoke_build and info.get("ok") else None
    ok = bool(docker_cli and info.get("ok") and disk.get("ok") and (not smoke_build or (smoke and smoke.get("ok"))))
    return {
        "ok": ok,
        "docker_cli": docker_cli,
        "docker_info": {
            "ok": bool(info.get("ok")),
            "driver": parsed_info.get("driver"),
            "docker_root": parsed_info.get("docker_root"),
            "server_version": parsed_info.get("server_version"),
            "returncode": info.get("returncode"),
            "stderr": info.get("stderr"),
            "duration_seconds": info.get("duration_seconds"),
        },
        "disk": disk,
        "smoke_build_requested": bool(smoke_build),
        "smoke_build": smoke,
    }


def build_deploy_readiness(
    *,
    gateway_commit: str | None = None,
    validator_commit: str | None = None,
    gateway_pcr0: str | None = None,
    validator_pcr0: str | None = None,
    expected_gateway_commit: str | None = None,
    expected_validator_commit: str | None = None,
    expected_gateway_pcr0: str | None = None,
    expected_validator_pcr0: str | None = None,
    require_same_commit: bool = False,
    require_pcr0: bool = False,
    require_pcr0_commit_match: bool = False,
    include_docker_health: bool = False,
    require_docker_build_health: bool = False,
) -> dict[str, Any]:
    build_info = get_build_info()
    source_commit, source_commit_path = read_source_commit()
    resolved_gateway_commit = (
        normalize_commit(gateway_commit)
        or source_commit
        or normalize_commit(build_info.get("git_commit"))
    )
    resolved_validator_commit = normalize_commit(validator_commit)
    resolved_gateway_pcr0 = normalize_pcr0(gateway_pcr0)
    resolved_validator_pcr0 = normalize_pcr0(validator_pcr0)
    expected_gateway_commit_norm = normalize_commit(expected_gateway_commit)
    expected_validator_commit_norm = normalize_commit(expected_validator_commit)
    expected_gateway_pcr0_norm = normalize_pcr0(expected_gateway_pcr0)
    expected_validator_pcr0_norm = normalize_pcr0(expected_validator_pcr0)

    gateway_static = _static_allowlist_status(resolved_gateway_pcr0, role="gateway")
    validator_static = _static_allowlist_status(resolved_validator_pcr0, role="validator")
    validator_dynamic = _dynamic_validator_status(
        resolved_validator_pcr0,
        resolved_validator_commit,
    )
    validator_pcr0_accepted = bool(validator_static.get("allowed") or validator_dynamic.get("valid"))
    docker_health = (
        docker_build_health(smoke_build=require_docker_build_health)
        if include_docker_health or require_docker_build_health
        else None
    )

    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        "gateway_commit_known",
        bool(resolved_gateway_commit),
        detail="gateway commit comes from explicit arg, .source_commit, BUILD_INFO, or git",
        actual=resolved_gateway_commit,
    )
    if resolved_validator_commit or expected_validator_commit_norm or require_same_commit:
        _add_check(
            checks,
            "validator_commit_known",
            bool(resolved_validator_commit),
            detail="validator commit must be supplied by the caller or manifest",
            actual=resolved_validator_commit,
        )
    if expected_gateway_commit_norm:
        _add_check(
            checks,
            "gateway_commit_matches_expected",
            _commit_matches(expected_gateway_commit_norm, resolved_gateway_commit),
            expected=expected_gateway_commit_norm,
            actual=resolved_gateway_commit,
        )
    if expected_validator_commit_norm:
        _add_check(
            checks,
            "validator_commit_matches_expected",
            _commit_matches(expected_validator_commit_norm, resolved_validator_commit),
            expected=expected_validator_commit_norm,
            actual=resolved_validator_commit,
        )
    if require_same_commit:
        _add_check(
            checks,
            "gateway_validator_commits_match",
            _commit_matches(resolved_gateway_commit, resolved_validator_commit),
            expected=resolved_gateway_commit,
            actual=resolved_validator_commit,
        )

    if require_pcr0 or resolved_gateway_pcr0 or expected_gateway_pcr0_norm:
        _add_check(
            checks,
            "gateway_pcr0_present",
            bool(resolved_gateway_pcr0),
            actual=resolved_gateway_pcr0,
        )
    if require_pcr0 or resolved_validator_pcr0 or expected_validator_pcr0_norm:
        _add_check(
            checks,
            "validator_pcr0_present",
            bool(resolved_validator_pcr0),
            actual=resolved_validator_pcr0,
        )
    if resolved_gateway_pcr0:
        _add_check(
            checks,
            "gateway_pcr0_static_allowlisted",
            bool(gateway_static.get("allowed")),
            actual=resolved_gateway_pcr0,
            detail="gateway PCR0s are verified by the static allowlist",
        )
    if resolved_validator_pcr0:
        _add_check(
            checks,
            "validator_pcr0_accepted",
            validator_pcr0_accepted,
            actual=resolved_validator_pcr0,
            detail="validator PCR0 is accepted by dynamic cache or static allowlist",
        )
    if expected_gateway_pcr0_norm:
        _add_check(
            checks,
            "gateway_pcr0_matches_expected",
            _pcr0_matches(expected_gateway_pcr0_norm, resolved_gateway_pcr0),
            expected=expected_gateway_pcr0_norm,
            actual=resolved_gateway_pcr0,
        )
    if expected_validator_pcr0_norm:
        _add_check(
            checks,
            "validator_pcr0_matches_expected",
            _pcr0_matches(expected_validator_pcr0_norm, resolved_validator_pcr0),
            expected=expected_validator_pcr0_norm,
            actual=resolved_validator_pcr0,
        )
    if require_pcr0_commit_match and resolved_gateway_pcr0:
        _add_check(
            checks,
            "gateway_pcr0_commit_matches_gateway_commit",
            _allowlist_commit_matches_runtime(gateway_static, resolved_gateway_commit),
            expected=resolved_gateway_commit,
            actual=gateway_static.get("matched_entry_commits"),
        )
    if require_pcr0_commit_match and resolved_validator_pcr0:
        dynamic_commit_matches = bool(
            resolved_validator_commit and validator_dynamic.get("valid")
        )
        static_commit_matches = _allowlist_commit_matches_runtime(
            validator_static,
            resolved_validator_commit,
        )
        _add_check(
            checks,
            "validator_pcr0_commit_matches_validator_commit",
            dynamic_commit_matches or static_commit_matches,
            expected=resolved_validator_commit,
            actual={
                "dynamic": (validator_dynamic.get("verification") or {}).get(
                    "commit_hash"
                ),
                "static": validator_static.get("matched_entry_commits"),
            },
        )
    if docker_health is not None:
        _add_check(
            checks,
            "docker_build_health",
            bool(docker_health.get("ok")),
            severity="error" if require_docker_build_health else "warning",
            detail=(
                "Docker host/build health; require flag runs a tiny scratch-image smoke build "
                "and blocks resume on failure"
            ),
            actual={
                "docker_root": (docker_health.get("docker_info") or {}).get("docker_root"),
                "disk": docker_health.get("disk"),
                "smoke_build_requested": docker_health.get("smoke_build_requested"),
                "smoke_build_ok": (
                    (docker_health.get("smoke_build") or {}).get("ok")
                    if docker_health.get("smoke_build") is not None
                    else None
                ),
            },
        )

    ok = all(check["ok"] for check in checks if check.get("severity") == "error")
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "ok": ok,
        "build_time_utc": build_info.get("build_time_utc", UNKNOWN),
        "source_commit_path": source_commit_path,
        "gateway": {
            "commit": resolved_gateway_commit,
            "build_info": build_info,
            "pcr0": resolved_gateway_pcr0,
            "pcr0_static_allowlist": gateway_static,
        },
        "validator": {
            "commit": resolved_validator_commit,
            "pcr0": resolved_validator_pcr0,
            "pcr0_static_allowlist": validator_static,
            "pcr0_dynamic_cache": validator_dynamic,
            "pcr0_accepted": validator_pcr0_accepted,
        },
        "expected": {
            "gateway_commit": expected_gateway_commit_norm,
            "validator_commit": expected_validator_commit_norm,
            "gateway_pcr0": expected_gateway_pcr0_norm,
            "validator_pcr0": expected_validator_pcr0_norm,
            "require_same_commit": require_same_commit,
            "require_pcr0": require_pcr0,
            "require_pcr0_commit_match": require_pcr0_commit_match,
            "include_docker_health": include_docker_health,
            "require_docker_build_health": require_docker_build_health,
        },
        "host_health": {
            "docker": docker_health,
        },
        "checks": checks,
    }


def write_deploy_readiness_manifest(
    document: Mapping[str, Any],
    path: str | Path | None = None,
    *,
    enforce_resume_block: bool = True,
) -> Path:
    target = Path(path).expanduser() if path else default_manifest_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(document)
    payload["enforce_resume_block"] = bool(enforce_resume_block)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target


def load_deploy_readiness_manifest(path: str | Path | None = None) -> dict[str, Any] | None:
    target = Path(path).expanduser() if path else default_manifest_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        raise RuntimeError(f"deploy readiness manifest is not an object: {target}")
    payload.setdefault("manifest_path", str(target))
    return payload


def assert_resume_allowed(path: str | Path | None = None) -> dict[str, Any] | None:
    manifest = load_deploy_readiness_manifest(path)
    if manifest is None:
        raise RuntimeError("deploy readiness guard blocked resume; manifest is missing")
    if manifest.get("schema_version") == DEPLOY_READINESS_V2_SCHEMA_VERSION:
        return validate_v2_deploy_readiness_manifest(manifest)
    failed = [
        check.get("name")
        for check in manifest.get("checks", [])
        if isinstance(check, Mapping) and check.get("severity") == "error" and not check.get("ok")
    ]
    raise RuntimeError(
        "deploy readiness guard blocked resume; schema v2 is required"
        + (f"; failing checks: {', '.join(str(item) for item in failed)}" if failed else "")
    )
