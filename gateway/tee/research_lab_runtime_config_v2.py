"""Canonical, secret-free Research Lab behavior configuration for V2 enclaves.

The gateway parent resolves the existing production environment once.  The
result is committed by every enclave boot identity and reconstructed inside the
measured roles.  Credentials and proxy URLs remain on the KMS/mTLS relay path;
they are never copied into this document.
"""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
import os
import re
from typing import Any, Dict, Mapping, Optional

from Leadpoet.utils.subnet_epoch import (
    SubnetEpochCutover,
    SubnetEpochError,
    load_subnet_epoch_cutover,
)
from gateway.research_lab.config import (
    DEFAULT_RESEARCH_LAB_DEV_SNAPSHOT_URI,
    DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG,
    RESEARCH_LAB_GIT_TREE_ENV_NAMES,
    ResearchLabGatewayConfig,
    ResearchLabGitTreeConfig,
    ResearchLabGitTreeConfigError,
)
from gateway.tee.scoring_executor import SCORING_CONFIG_ENV_NAMES
from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from research_lab.eval.private_runtime import (
    INCONTAINER_TRACE_CORPUS_MAX_CALL_BYTES,
    INCONTAINER_TRACE_CORPUS_MAX_TOTAL_BYTES,
    INCONTAINER_TRACE_ENV_PASSTHROUGH,
    INCONTAINER_TRACE_MAX_CALL_BYTES_ENV,
    INCONTAINER_TRACE_MAX_TOTAL_BYTES_ENV,
    PROVIDER_COST_ENV_PASSTHROUGH,
    PROVIDER_COST_EVALUATION_SCOPE_ENV,
    PROVIDER_KEY_ENV_PASSTHROUGH,
    private_model_env_passthrough,
)
from research_lab.eval.snapshot_store import (
    MISS_POLICIES,
    MISS_POLICY_STRICT,
    SNAPSHOT_MISS_POLICY_ENV,
)


SCHEMA_VERSION = "leadpoet.research_lab_execution_config.v3"
_CONFIG_FIELD_NAMES_HASH = (
    "sha256:7b7b8623e08d23e400fe2e75f40f076c7b6e6a2c2e58df8779c41c393d2224d6"
)

# These values are either credentials or credential-bearing relay URLs.  Their
# behavior is represented by provider profiles and KMS reference hashes in the
# outer runtime document, not by plaintext values here.
HOST_ONLY_SECRET_FIELDS = frozenset(
    {
        "hosted_worker_proxy_url",
        "internal_api_key",
        "miner_openrouter_key_ref_env_map_json",
        "scoring_worker_proxy_url",
    }
)

# Existing code reads these switches directly instead of through
# ResearchLabGatewayConfig.  Capturing them preserves the exact current
# successful path while preventing enclave-local defaults from taking over.
AUTORESEARCH_BEHAVIOR_ENV_NAMES = (
    "RESEARCH_LAB_DEV_SNAPSHOT_URI",
    "RESEARCH_LAB_LOOP_BUILD_HEARTBEAT",
    "RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED",
    "RESEARCH_LAB_LOOP_DEV_EVAL_ICP_TIMEOUT_SECONDS",
    "RESEARCH_LAB_LOOP_DEV_EVAL_TIMEOUT_SECONDS",
    "RESEARCH_LAB_LOOP_JUDGE_PARSE_SOFT_SKIP",
    "RESEARCH_LAB_LOOP_MIN_RUNTIME_SKIP_WHEN_SELECTED",
    "RESEARCH_LAB_LOOP_PLANNER_PARSE_RETRY",
    "RESEARCH_LAB_LOOP_PROBE_REQUIRE_WINDOW_GUARD",
    "RESEARCH_LAB_LOOP_PROVIDER_PROBES_LIVE",
    "RESEARCH_LAB_LOOP_RESUME_RESTORE_SELECTED",
    "RESEARCH_LAB_LOOP_STAGE_ERROR_CONTAINMENT",
    *RESEARCH_LAB_GIT_TREE_ENV_NAMES,
    "RESEARCH_LAB_SYMBOL_SLICE_BUDGET_SHARE",
    "RESEARCH_LAB_SYMBOL_SLICE_MODE",
    "RESEARCH_LAB_LOOP_WITHIN_RUN_MEMORY",
    "RESEARCH_LAB_PROBE_SNAPSHOT_OVERLAY_URI",
    "RESEARCH_LAB_REFLECTION_EMISSION_ENABLED",
)

AUTORESEARCH_BEHAVIOR_DEFAULTS = {
    "RESEARCH_LAB_DEV_SNAPSHOT_URI": DEFAULT_RESEARCH_LAB_DEV_SNAPSHOT_URI,
    "RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED": "true",
    **DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.to_environment(),
}

PROVIDER_PREFLIGHT_BEHAVIOR_ENV_NAMES = (
    "RESEARCH_LAB_PROVIDER_PREFLIGHT_ENABLED",
    "RESEARCH_LAB_PROVIDER_PREFLIGHT_FAILURE_STREAK",
    "RESEARCH_LAB_PROVIDER_PREFLIGHT_TIMEOUT_SECONDS",
    "RESEARCH_LAB_PROVIDER_PREFLIGHT_TTL_SECONDS",
)

ADDITIONAL_SCORING_BEHAVIOR_ENV_NAMES = (
    "INTENT_URL_PREFILTER_ENABLED",
)

# These values are forwarded into measured model sandboxes by the existing
# scoring path.  Credential values are deliberately excluded; only the names
# that were present at boot are committed below so replay-only models see the
# same placeholder-variable shape as before.
MODEL_BEHAVIOR_ENV_NAMES = tuple(
    sorted(
        set(INCONTAINER_TRACE_ENV_PASSTHROUGH)
        | set(PROVIDER_COST_ENV_PASSTHROUGH)
        | {
            "EXA_MAX_RPS",
            "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX",
            "RESEARCH_LAB_PROVIDER_EVIDENCE_RECORD",
            "SOURCING_DEEPLINE_FALLBACK",
            "SOURCING_DEEPLINE_TIMEOUT_S",
            SNAPSHOT_MISS_POLICY_ENV,
        }
    )
)

MODEL_CREDENTIAL_ENV_NAMES = tuple(sorted(set(PROVIDER_KEY_ENV_PASSTHROUGH)))
DEFAULT_DEV_EVAL_TOTAL_TIMEOUT_SECONDS = 300

BEHAVIOR_ENV_NAMES = tuple(
    sorted(
        set(SCORING_CONFIG_ENV_NAMES)
        | set(AUTORESEARCH_BEHAVIOR_ENV_NAMES)
        | set(PROVIDER_PREFLIGHT_BEHAVIOR_ENV_NAMES)
        | set(ADDITIONAL_SCORING_BEHAVIOR_ENV_NAMES)
        | set(MODEL_BEHAVIOR_ENV_NAMES)
    )
)

_FORBIDDEN_VALUE_MARKERS = (
    "-----begin private key-----",
    "aws_secret_access_key=",
    "sb_secret_",
    "sk-or-",
)
_URI_WITH_USERINFO_RE = re.compile(r"^[a-z][a-z0-9+.-]*://[^/@\s]+:[^/@\s]+@", re.I)


class ResearchLabRuntimeConfigV2Error(ValueError):
    """The measured Research Lab configuration is incomplete or unsafe."""


def _field_names() -> tuple[str, ...]:
    names = tuple(sorted(item.name for item in fields(ResearchLabGatewayConfig)))
    digest = sha256_json(list(names))
    if digest != _CONFIG_FIELD_NAMES_HASH:
        raise ResearchLabRuntimeConfigV2Error(
            "ResearchLabGatewayConfig fields changed without V2 classification"
        )
    if not HOST_ONLY_SECRET_FIELDS.issubset(names):
        raise ResearchLabRuntimeConfigV2Error(
            "V2 host-only Research Lab field classification is invalid"
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
    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, str) for item in value
        ):
            raise ResearchLabRuntimeConfigV2Error("%s must be a string array" % field)
        return [_validate_string(str(item), field) for item in value]
    if default is None:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            raise ResearchLabRuntimeConfigV2Error("%s must be integer or null" % field)
        return value
    if not isinstance(value, str):
        raise ResearchLabRuntimeConfigV2Error("%s must be text" % field)
    return _validate_string(value, field)


def _normalized_fields(value: Mapping[str, Any]) -> Dict[str, Any]:
    names = _field_names()
    safe_names = tuple(name for name in names if name not in HOST_ONLY_SECRET_FIELDS)
    if not isinstance(value, Mapping) or set(value) != set(safe_names):
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab execution fields do not match the reviewed schema"
        )
    defaults = ResearchLabGatewayConfig()
    return {
        name: _normalize_scalar(value[name], getattr(defaults, name), name)
        for name in safe_names
    }


def _normalized_environment(value: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    if not isinstance(value, Mapping) or set(value) != set(BEHAVIOR_ENV_NAMES):
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab behavior environment does not match the reviewed schema"
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
            "Research Lab behavior environment exceeds size limit"
        )
    return normalized


def _normalized_epoch_authority(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the non-secret epoch authority committed into the enclave."""

    if not isinstance(value, Mapping) or set(value) != {"mode", "cutover"}:
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab epoch authority fields are invalid"
        )
    mode = str(value.get("mode") or "").strip().lower()
    cutover_value = value.get("cutover")
    if mode != "stateful_v1" or not isinstance(cutover_value, Mapping):
        raise ResearchLabRuntimeConfigV2Error(
            "stateful Research Lab epoch authority requires a cutover"
        )
    try:
        cutover = SubnetEpochCutover.from_mapping(cutover_value)
    except (SubnetEpochError, TypeError) as exc:
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab epoch cutover is invalid"
        ) from exc
    return {"mode": "stateful_v1", "cutover": cutover.to_dict()}


def build_research_lab_execution_config(
    *,
    config: Optional[ResearchLabGatewayConfig] = None,
    environment: Optional[Mapping[str, Any]] = None,
    network: Optional[str] = None,
    netuid: Optional[int] = None,
) -> Dict[str, Any]:
    resolved = config or ResearchLabGatewayConfig.from_env()
    values = {
        item.name: getattr(resolved, item.name)
        for item in fields(ResearchLabGatewayConfig)
        if item.name not in HOST_ONLY_SECRET_FIELDS
    }
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
        epoch_cutover = load_subnet_epoch_cutover(
            source_environment
        ).to_dict()
    except SubnetEpochError as exc:
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab epoch authority is invalid"
        ) from exc
    document = {
        "schema_version": SCHEMA_VERSION,
        "deployment": {
            "network": resolved_network,
            "netuid": resolved_netuid,
        },
        "fields": _normalized_fields(values),
        "host_only_secret_fields": sorted(HOST_ONLY_SECRET_FIELDS),
        "credential_environment_names": sorted(
            name for name in MODEL_CREDENTIAL_ENV_NAMES if name in source_environment
        ),
        "epoch_authority": _normalized_epoch_authority(
            {"mode": "stateful_v1", "cutover": epoch_cutover}
        ),
        "behavior_environment": _normalized_environment(
            {
                name: source_environment.get(
                    name, AUTORESEARCH_BEHAVIOR_DEFAULTS.get(name)
                )
                for name in BEHAVIOR_ENV_NAMES
            }
        ),
    }
    return validate_research_lab_execution_config(document)


def validate_research_lab_execution_config(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "deployment",
        "fields",
        "host_only_secret_fields",
        "credential_environment_names",
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
    omitted = value.get("host_only_secret_fields")
    if omitted != sorted(HOST_ONLY_SECRET_FIELDS):
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab host-only field classification differs"
        )
    credential_names = value.get("credential_environment_names")
    if (
        not isinstance(credential_names, list)
        or credential_names != sorted(set(credential_names))
        or any(name not in MODEL_CREDENTIAL_ENV_NAMES for name in credential_names)
    ):
        raise ResearchLabRuntimeConfigV2Error(
            "Research Lab credential environment shape is invalid"
        )
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "deployment": {"network": network, "netuid": netuid},
        "fields": _normalized_fields(value.get("fields")),
        "host_only_secret_fields": sorted(HOST_ONLY_SECRET_FIELDS),
        "credential_environment_names": list(credential_names),
        "epoch_authority": _normalized_epoch_authority(
            value.get("epoch_authority")
        ),
        "behavior_environment": _normalized_environment(
            value.get("behavior_environment")
        ),
    }
    # Canonicalization also rejects unsupported object types and NaN values.
    return json.loads(canonical_json(normalized))


def research_lab_config_from_document(
    document: Mapping[str, Any],
) -> ResearchLabGatewayConfig:
    normalized = validate_research_lab_execution_config(document)
    values = dict(normalized["fields"])
    defaults = ResearchLabGatewayConfig()
    for item in fields(ResearchLabGatewayConfig):
        if isinstance(getattr(defaults, item.name), tuple) and item.name in values:
            values[item.name] = tuple(values[item.name])
    return ResearchLabGatewayConfig(**values)


def apply_behavior_environment(document: Mapping[str, Any]) -> None:
    normalized = validate_research_lab_execution_config(document)
    for name, value in normalized["behavior_environment"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def measured_model_environment(document: Mapping[str, Any]) -> Dict[str, str]:
    normalized = validate_research_lab_execution_config(document)
    environment = normalized["behavior_environment"]
    return {
        name: str(environment[name])
        for name in MODEL_BEHAVIOR_ENV_NAMES
        if environment.get(name) is not None
    }


def measured_dev_replay_environment(document: Mapping[str, Any]) -> Dict[str, str]:
    measured = measured_model_environment(document)
    return {
        name: measured[name]
        for name in private_model_env_passthrough()
        if name in measured
    }


def measured_credential_environment_names(
    document: Mapping[str, Any],
) -> tuple[str, ...]:
    normalized = validate_research_lab_execution_config(document)
    return tuple(normalized["credential_environment_names"])


def measured_dev_eval_total_timeout_seconds(document: Mapping[str, Any]) -> int:
    normalized = validate_research_lab_execution_config(document)
    raw = str(
        normalized["behavior_environment"].get(
            "RESEARCH_LAB_LOOP_DEV_EVAL_TIMEOUT_SECONDS"
        )
        or ""
    ).strip()
    try:
        value = int(raw) if raw else DEFAULT_DEV_EVAL_TOTAL_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_DEV_EVAL_TOTAL_TIMEOUT_SECONDS
    return max(30, value)


def measured_dev_eval_icp_timeout_seconds(
    document: Mapping[str, Any],
    *,
    item_count: int,
) -> int:
    normalized = validate_research_lab_execution_config(document)
    raw = str(
        normalized["behavior_environment"].get(
            "RESEARCH_LAB_LOOP_DEV_EVAL_ICP_TIMEOUT_SECONDS"
        )
        or ""
    ).strip()
    if raw:
        try:
            return max(10, int(raw))
        except ValueError:
            pass
    return max(
        30,
        measured_dev_eval_total_timeout_seconds(normalized)
        // max(1, int(item_count)),
    )


def measured_dev_snapshot_miss_policy(document: Mapping[str, Any]) -> str:
    normalized = validate_research_lab_execution_config(document)
    raw = str(
        normalized["behavior_environment"].get(SNAPSHOT_MISS_POLICY_ENV) or ""
    ).strip().lower()
    return raw if raw in MISS_POLICIES else MISS_POLICY_STRICT


def measured_git_tree_config(
    document: Mapping[str, Any],
) -> ResearchLabGitTreeConfig:
    normalized = validate_research_lab_execution_config(document)
    environment = {
        name: value
        for name, value in normalized["behavior_environment"].items()
        if name in RESEARCH_LAB_GIT_TREE_ENV_NAMES and value is not None
    }
    try:
        return ResearchLabGitTreeConfig.from_env(environment)
    except ResearchLabGitTreeConfigError as exc:
        raise ResearchLabRuntimeConfigV2Error(
            f"measured Git-tree configuration is invalid: {exc}"
        ) from exc


def validate_model_sandbox_environment(
    document: Mapping[str, Any],
    environment: Mapping[str, Any],
    *,
    provider_cost_scope: str,
) -> Dict[str, str]:
    normalized = validate_research_lab_execution_config(document)
    if not isinstance(environment, Mapping):
        raise ResearchLabRuntimeConfigV2Error(
            "model sandbox environment must be an object"
        )
    config = research_lab_config_from_document(normalized)
    measured = measured_model_environment(normalized)
    trace_environment: Dict[str, str] = {}
    for name in INCONTAINER_TRACE_ENV_PASSTHROUGH:
        if name in measured:
            trace_environment[name] = measured[name]
    scoring_environment = dict(trace_environment)
    for name in ("SOURCING_DEEPLINE_FALLBACK", "SOURCING_DEEPLINE_TIMEOUT_S"):
        if name in measured:
            scoring_environment[name] = measured[name]
    if measured.get("EXA_MAX_RPS"):
        scoring_environment["EXA_MAX_RPS"] = measured["EXA_MAX_RPS"]
    scoring_environment.update(
        {
            "RESEARCH_LAB_PROVIDER_COST_CAP_USD_PER_ICP": str(
                config.provider_cost_cap_usd_per_icp
            ),
            "RESEARCH_LAB_SCRAPINGDOG_COST_PER_CREDIT_USD": str(
                config.scrapingdog_cost_per_credit_usd
            ),
            "RESEARCH_LAB_SCRAPINGDOG_UNKNOWN_ENDPOINT_CREDITS": str(
                config.scrapingdog_unknown_endpoint_credits
            ),
            "RESEARCH_LAB_PROVIDER_COST_UNKNOWN_ENDPOINT_POLICY": str(
                config.provider_cost_unknown_endpoint_policy
            ),
            "RESEARCH_LAB_PROVIDER_EVIDENCE_RECORD": "1",
            PROVIDER_COST_EVALUATION_SCOPE_ENV: str(provider_cost_scope),
        }
    )
    trace_prefix = measured.get("RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX", "")
    if trace_prefix:
        scoring_environment.setdefault(
            INCONTAINER_TRACE_MAX_CALL_BYTES_ENV,
            str(INCONTAINER_TRACE_CORPUS_MAX_CALL_BYTES),
        )
        scoring_environment.setdefault(
            INCONTAINER_TRACE_MAX_TOTAL_BYTES_ENV,
            str(INCONTAINER_TRACE_CORPUS_MAX_TOTAL_BYTES),
        )
        trace_environment.setdefault(
            INCONTAINER_TRACE_MAX_CALL_BYTES_ENV,
            str(INCONTAINER_TRACE_CORPUS_MAX_CALL_BYTES),
        )
        trace_environment.setdefault(
            INCONTAINER_TRACE_MAX_TOTAL_BYTES_ENV,
            str(INCONTAINER_TRACE_CORPUS_MAX_TOTAL_BYTES),
        )

    allowed_profiles = [scoring_environment]
    shadow = dict(trace_environment)
    if config.benchmark_exa_max_rps > 0:
        benchmark = dict(scoring_environment)
        benchmark["EXA_MAX_RPS"] = str(config.benchmark_exa_max_rps)
        allowed_profiles.append(benchmark)
        retry = dict(scoring_environment)
        retry["EXA_MAX_RPS"] = str(
            round(
                config.benchmark_exa_max_rps
                * config.private_baseline_concurrency
                / max(1, config.private_baseline_retry_concurrency),
                3,
            )
        )
        allowed_profiles.append(retry)
        shadow["EXA_MAX_RPS"] = str(config.benchmark_exa_max_rps)
    allowed_profiles.append(shadow)
    normalized_environment = {
        str(name): str(value) for name, value in environment.items()
    }
    if not any(normalized_environment == profile for profile in allowed_profiles):
        raise ResearchLabRuntimeConfigV2Error(
            "model sandbox environment differs from every measured profile"
        )
    return dict(sorted(normalized_environment.items()))


def research_lab_execution_config_hash(document: Mapping[str, Any]) -> str:
    return sha256_json(validate_research_lab_execution_config(document))
