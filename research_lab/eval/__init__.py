"""Production Research Lab evaluation boundary.

This package contains contracts for real private-model evaluation. It does not
include simulated model improvements or public miner model submission logic.

Exports are resolved lazily so importing one leaf module does not import the
host evaluator inside a sandbox where model-owned packages are authoritative.
"""

from __future__ import annotations

import importlib
from typing import Any


_LAZY_EXPORTS = {
    "CandidatePatchManifest": ("patches", "CandidatePatchManifest"),
    "DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID": (
        "private_runtime",
        "DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID",
    ),
    "DockerPrivateModelRunner": ("private_runtime", "DockerPrivateModelRunner"),
    "DockerPrivateModelSpec": ("private_runtime", "DockerPrivateModelSpec"),
    "PrivateModelArtifactManifest": ("artifacts", "PrivateModelArtifactManifest"),
    "PrivateModelAdapterSpec": ("private_runtime", "PrivateModelAdapterSpec"),
    "PrivateModelRuntimeError": ("private_runtime", "PrivateModelRuntimeError"),
    "RealEvaluatorRequired": ("evaluator", "RealEvaluatorRequired"),
    "SealedBenchmarkSet": ("benchmark", "SealedBenchmarkSet"),
    "SubprocessPrivateModelRunner": (
        "private_runtime",
        "SubprocessPrivateModelRunner",
    ),
    "build_local_private_artifact_manifest": (
        "private_runtime",
        "build_local_private_artifact_manifest",
    ),
    "build_score_bundle_from_scored_icps": (
        "evaluator",
        "build_score_bundle_from_scored_icps",
    ),
    "compute_private_source_tree_hash": (
        "private_runtime",
        "compute_private_source_tree_hash",
    ),
    "ensure_private_model_outputs": (
        "private_runtime",
        "ensure_private_model_outputs",
    ),
    "evaluate_private_model_pair": ("evaluator", "evaluate_private_model_pair"),
    "load_private_artifact_manifest": (
        "private_runtime",
        "load_private_artifact_manifest",
    ),
    "private_model_env_passthrough": (
        "private_runtime",
        "private_model_env_passthrough",
    ),
    "private_model_artifact_replay_identity_v2": (
        "artifacts",
        "private_model_artifact_replay_identity_v2",
    ),
    "score_private_model_pair_items": (
        "evaluator",
        "score_private_model_pair_items",
    ),
    "sign_digest_with_kms": ("private_runtime", "sign_digest_with_kms"),
    "validate_candidate_patch_manifest": (
        "patches",
        "validate_candidate_patch_manifest",
    ),
    "validate_private_model_artifact_manifest": (
        "artifacts",
        "validate_private_model_artifact_manifest",
    ),
    "validate_sealed_benchmark_set": (
        "benchmark",
        "validate_sealed_benchmark_set",
    ),
    "verify_private_artifact_manifest_signature": (
        "private_runtime",
        "verify_private_artifact_manifest_signature",
    ),
    "verify_private_model_artifact_manifest": (
        "artifacts",
        "verify_private_model_artifact_manifest",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def lazy_import_contract() -> dict[str, str]:
    """Return the stable package export mapping for checkpoint identity."""

    return {
        name: f"{module_name}:{attribute_name}"
        for name, (module_name, attribute_name) in sorted(_LAZY_EXPORTS.items())
    }
