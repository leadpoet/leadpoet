"""Production Research Lab evaluation boundary.

This package contains contracts for real private-model evaluation. It does not
include simulated model improvements or public miner model submission logic.
"""

from .artifacts import (
    PrivateModelArtifactManifest,
    validate_private_model_artifact_manifest,
    verify_private_model_artifact_manifest,
)
from .benchmark import SealedBenchmarkSet, validate_sealed_benchmark_set
from .evaluator import (
    RealEvaluatorRequired,
    build_score_bundle_from_scored_icps,
    evaluate_private_model_pair,
)
from .patches import CandidatePatchManifest, validate_candidate_patch_manifest
from .private_runtime import (
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
    PrivateModelAdapterSpec,
    PrivateModelRuntimeError,
    SubprocessPrivateModelRunner,
    build_local_private_artifact_manifest,
    compute_private_source_tree_hash,
    ensure_private_model_outputs,
    load_private_artifact_manifest,
    sign_digest_with_kms,
)

__all__ = [
    "CandidatePatchManifest",
    "DockerPrivateModelRunner",
    "DockerPrivateModelSpec",
    "PrivateModelArtifactManifest",
    "PrivateModelAdapterSpec",
    "PrivateModelRuntimeError",
    "RealEvaluatorRequired",
    "SealedBenchmarkSet",
    "SubprocessPrivateModelRunner",
    "build_local_private_artifact_manifest",
    "build_score_bundle_from_scored_icps",
    "compute_private_source_tree_hash",
    "ensure_private_model_outputs",
    "evaluate_private_model_pair",
    "load_private_artifact_manifest",
    "sign_digest_with_kms",
    "validate_candidate_patch_manifest",
    "validate_private_model_artifact_manifest",
    "validate_sealed_benchmark_set",
    "verify_private_model_artifact_manifest",
]
