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

__all__ = [
    "CandidatePatchManifest",
    "PrivateModelArtifactManifest",
    "RealEvaluatorRequired",
    "SealedBenchmarkSet",
    "build_score_bundle_from_scored_icps",
    "evaluate_private_model_pair",
    "validate_candidate_patch_manifest",
    "validate_private_model_artifact_manifest",
    "validate_sealed_benchmark_set",
    "verify_private_model_artifact_manifest",
]
