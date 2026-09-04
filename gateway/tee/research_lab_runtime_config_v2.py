"""Secret-free runtime settings for current gateway enclave work.

The document contains normal qualification behavior, provider preflight,
SOURCE_ADD, retained reward allocation, and the shared chain/provider
boundary. It intentionally contains no model repository, autoresearch,
code-edit, private holdout, or miner-credential configuration.
"""

from __future__ import annotations

from dataclasses import fields
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional

from Leadpoet.utils.subnet_epoch import (
    SubnetEpochCutover,
    SubnetEpochError,
    load_subnet_epoch_cutover,
)
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.tee.scoring_executor import SCORING_CONFIG_ENV_NAMES
from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from leadpoet_canonical.hotkey_authority_v2 import validate_chain_signing_profile
from leadpoet_canonical.production_parity_boundary_v2 import (
    PRODUCTION_PARITY_ENV_NAMES,
    ProductionParityBoundaryV2Error,
    validate_production_parity_boundary_v2,
)


SCHEMA_VERSION = "leadpoet.research_lab_execution_config.v9"

HOST_ONLY_SECRET_FIELDS = frozenset({"internal_api_key"})

PROVIDER_PREFLIGHT_BEHAVIOR_ENV_NAMES = (
    "RESEARCH_LAB_PROVIDER_PREFLIGHT_ENABLED",
    "RESEARCH_LAB_PROVIDER_PREFLIGHT_FAILURE_STREAK",
    "RESEARCH_LAB_PROVIDER_PREFLIGHT_TIMEOUT_SECONDS",
    "RESEARCH_LAB_PROVIDER_PREFLIGHT_TTL_SECONDS",
)

ADDITIONAL_QUALIFICATION_BEHAVIOR_ENV_NAMES = (
    "INTENT_URL_PREFILTER_ENABLED",
    "RESEARCH_LAB_INTENT_CORROBORATION_RESCUE",
    "RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE",
    "VERIFIER_SEMANTIC_GATE_MODELS",
    "VERIFIER_SEMANTIC_GATES_MODE",
)

SOURCE_ADD_BEHAVIOR_ENV_NAMES = ("RESEARCH_LAB_LLM_INCLUDE_REASONING",)

BEHAVIOR_DEFAULTS = {
    "RESEARCH_LAB_INTENT_CORROBORATION_RESCUE": "false",
    "RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE": "shadow",
    "VERIFIER_SEMANTIC_GATE_MODELS": "",
    "VERIFIER_SEMANTIC_GATES_MODE": "disabled",
}

BEHAVIOR_ENV_NAMES = tuple(
    sorted(
        set(SCORING_CONFIG_ENV_NAMES)
        | set(PROVIDER_PREFLIGHT_BEHAVIOR_ENV_NAMES)
        | set(ADDITIONAL_QUALIFICATION_BEHAVIOR_ENV_NAMES)
        | set(SOURCE_ADD_BEHAVIOR_ENV_NAMES)
        | set(PRODUCTION_PARITY_ENV_NAMES)
    )
)

_FORBIDDEN_VALUE_MARKERS = (
    "-----begin private key-----",
    "aws_secret_access_key=",
    "sb_secret_",
    "sk-or-",
)
_URI_WITH_USERINFO_RE = re.compile(
    r"^[a-z][a-z0-9+.-]*://[^/@\s]+:[^/@\s]+@",
    re.I,
)


class ResearchLabRuntimeConfigV2Error(ValueError):
    """The measured gateway configuration is incomplete or unsafe."""


def _field_names() -> tuple[str, ...]:
    names = tuple(sorted(item.name for item in fields(ResearchLabGatewayConfig)))
    if not HOST_ONLY_SECRET_FIELDS.issubset(names):
        raise ResearchLabRuntimeConfigV2Error(
            "host-only Research Lab field classification is invalid"
        )
    return names


def _validate_string(value: str, field: str) -> str:
    if "\x00" in value:
        raise ResearchLabRuntimeConfigV2Error("%s contains NUL" % field)
    if len(value.encode("utf-8")) > 128 * 1024:
        raise ResearchLabRuntimeConfigV2Error("%s exceeds size limit" % field)
    lowered = value.lower()
    if any(marker in lowered for marker in _FORBIDDEN_VALUE_MARKERS):
        raise ResearchLabRuntimeConfigV2Error("%s contains secret material" % field)
    if _URI_WITH_USERINFO_RE.match(value.strip()):
        raise ResearchLabRuntimeConfigV2Error("%s contains URI credentials" % field)
    return value


def _normalize_scalar(value: Any, default: Any, field: str) -> Any:
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ResearchLabRuntimeConfigV2Error("%s must be boolean" % field)
        return value
    if isinstance(default, int) and not isinstance(default, bool):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ResearchLabRuntimeConfigV2Error("%s must be integer" % field)
        return value
    if isinstance(default, float):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ResearchLabRuntimeConfigV2Error("%s must be numeric" % field)
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ResearchLabRuntimeConfigV2Error("%s must be finite" % field)
        return normalized
    if not isinstance(value, str):
        raise ResearchLabRuntimeConfigV2Error("%s must be text" % field)
    return _validate_string(value, field)


def _normalized_fields(value: Mapping[str, Any]) -> Dict[str, Any]:
    safe_names = tuple(
        name for name in _field_names() if name not in HOST_ONLY_SECRET_FIELDS
    )
    if not isinstance(value, Mapping) or set(value) != set(safe_names):
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab execution fields do not match the current schema"
        )
    defaults = ResearchLabGatewayConfig()
    return {
        name: _normalize_scalar(value[name], getattr(defaults, name), name)
        for name in safe_names
    }


def _normalized_environment(
    value: Mapping[str, Any],
) -> Dict[str, Optional[str]]:
    if not isinstance(value, Mapping) or set(value) != set(BEHAVIOR_ENV_NAMES):
        raise ResearchLabRuntimeConfigV2Error(
            "gateway behavior environment does not match the current schema"
        )
    normalized: Dict[str, Optional[str]] = {}
    total = 0
    for name in BEHAVIOR_ENV_NAMES:
        item = value.get(name)
        if item is None:
            normalized[name] = None
            continue
        if not isinstance(item, str):
            raise ResearchLabRuntimeConfigV2Error("%s must be text or null" % name)
        item = _validate_string(item, name)
        total += len(item.encode("utf-8"))
        normalized[name] = item
    if total > 128 * 1024:
        raise ResearchLabRuntimeConfigV2Error(
            "gateway behavior environment exceeds size limit"
        )
    return normalized


def _default_chain_signing_profile() -> Dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[2]
        / "validator_tee"
        / "enclave"
        / "chain_signing_profile_v2.json"
    )
    try:
        return validate_chain_signing_profile(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError) as exc:
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab chain signing profile is unavailable"
        ) from exc


def _normalized_epoch_authority(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "mode",
        "cutover",
        "chain_signing_profile",
    }:
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab epoch authority fields are invalid"
        )
    if str(value.get("mode") or "").strip().lower() != "stateful_v1":
        raise ResearchLabRuntimeConfigV2Error(
            "stateful Research Lab epoch authority is required"
        )
    try:
        cutover = SubnetEpochCutover.from_mapping(value.get("cutover"))
        profile = validate_chain_signing_profile(value.get("chain_signing_profile"))
    except (SubnetEpochError, TypeError, ValueError) as exc:
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab epoch authority is invalid"
        ) from exc
    cutover_genesis = str(cutover.network_genesis_hash).lower().removeprefix("0x")
    if cutover_genesis != str(profile["genesis_hash"]).lower():
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab epoch and signing genesis differ"
        )
    return {
        "mode": "stateful_v1",
        "cutover": cutover.to_dict(),
        "chain_signing_profile": profile,
    }


def build_research_lab_execution_config(
    *,
    config: Optional[ResearchLabGatewayConfig] = None,
    environment: Optional[Mapping[str, Any]] = None,
    network: Optional[str] = None,
    netuid: Optional[int] = None,
    chain_signing_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resolved = config or ResearchLabGatewayConfig.from_env()
    source_environment = os.environ if environment is None else environment
    resolved_network = str(
        network
        if network is not None
        else (
            source_environment.get("BITTENSOR_NETWORK")
            or source_environment.get("SUBTENSOR_NETWORK")
            or "finney"
        )
    ).strip().lower()
    if not resolved_network or len(resolved_network) > 64:
        raise ResearchLabRuntimeConfigV2Error("Research Lab network is invalid")
    raw_netuid = (
        netuid
        if netuid is not None
        else (
            source_environment.get("BITTENSOR_NETUID")
            or source_environment.get("NETUID")
            or 71
        )
    )
    try:
        resolved_netuid = int(raw_netuid)
    except (TypeError, ValueError) as exc:
        raise ResearchLabRuntimeConfigV2Error("Research Lab netuid is invalid") from exc
    if resolved_netuid < 0:
        raise ResearchLabRuntimeConfigV2Error("Research Lab netuid is invalid")
    try:
        cutover = load_subnet_epoch_cutover(source_environment).to_dict()
    except SubnetEpochError as exc:
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab epoch authority is invalid"
        ) from exc
    values = {
        item.name: getattr(resolved, item.name)
        for item in fields(ResearchLabGatewayConfig)
        if item.name not in HOST_ONLY_SECRET_FIELDS
    }
    return validate_research_lab_execution_config(
        {
            "schema_version": SCHEMA_VERSION,
            "deployment": {
                "network": resolved_network,
                "netuid": resolved_netuid,
            },
            "fields": _normalized_fields(values),
            "host_only_secret_fields": sorted(HOST_ONLY_SECRET_FIELDS),
            "epoch_authority": _normalized_epoch_authority(
                {
                    "mode": "stateful_v1",
                    "cutover": cutover,
                    "chain_signing_profile": (
                        dict(chain_signing_profile)
                        if chain_signing_profile is not None
                        else _default_chain_signing_profile()
                    ),
                }
            ),
            "behavior_environment": _normalized_environment(
                {
                    name: source_environment.get(name, BEHAVIOR_DEFAULTS.get(name))
                    for name in BEHAVIOR_ENV_NAMES
                }
            ),
        }
    )


def validate_research_lab_execution_config(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "deployment",
        "fields",
        "host_only_secret_fields",
        "epoch_authority",
        "behavior_environment",
    }:
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab execution configuration fields are invalid"
        )
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab execution configuration schema is invalid"
        )
    deployment = value.get("deployment")
    if not isinstance(deployment, Mapping) or set(deployment) != {
        "network",
        "netuid",
    }:
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab deployment fields are invalid"
        )
    network = str(deployment.get("network") or "").strip().lower()
    netuid = deployment.get("netuid")
    if (
        not network
        or len(network) > 64
        or not isinstance(netuid, int)
        or isinstance(netuid, bool)
        or netuid < 0
    ):
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab deployment configuration is invalid"
        )
    if value.get("host_only_secret_fields") != sorted(HOST_ONLY_SECRET_FIELDS):
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab host-only field classification differs"
        )
    environment = _normalized_environment(value.get("behavior_environment"))
    try:
        validate_production_parity_boundary_v2(
            environment,
            network=network,
            netuid=netuid,
        )
    except ProductionParityBoundaryV2Error as exc:
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab production-parity boundary is invalid"
        ) from exc
    epoch_authority = _normalized_epoch_authority(value.get("epoch_authority"))
    if epoch_authority["chain_signing_profile"]["network"] != network:
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab chain signing profile targets another network"
        )
    if int(epoch_authority["cutover"]["netuid"]) != int(netuid):
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab epoch authority targets another netuid"
        )
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "deployment": {"network": network, "netuid": netuid},
        "fields": _normalized_fields(value.get("fields")),
        "host_only_secret_fields": sorted(HOST_ONLY_SECRET_FIELDS),
        "epoch_authority": epoch_authority,
        "behavior_environment": environment,
    }
    return json.loads(canonical_json(normalized))


def research_lab_config_from_document(
    document: Mapping[str, Any],
) -> ResearchLabGatewayConfig:
    normalized = validate_research_lab_execution_config(document)
    return ResearchLabGatewayConfig(**dict(normalized["fields"]))


def apply_behavior_environment(document: Mapping[str, Any]) -> None:
    normalized = validate_research_lab_execution_config(document)
    for name, value in normalized["behavior_environment"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def research_lab_execution_config_hash(document: Mapping[str, Any]) -> str:
    return sha256_json(validate_research_lab_execution_config(document))
