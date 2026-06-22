"""Production Research Lab gateway flags.

All live workflow flags default false. Enabling the API only exposes the
Research Lab namespace; paid loops, probes, writes, receipts, and weights each
require their own explicit gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from typing import Any, Mapping


TRUTHY = {"1", "true", "yes", "on"}
logger = logging.getLogger(__name__)


def _truthy(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in TRUTHY


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class ResearchLabGatewayConfig:
    api_enabled: bool = False
    production_writes_enabled: bool = False
    paid_loops_enabled: bool = False
    probes_enabled: bool = False
    hosted_runs_enabled: bool = False
    receipts_enabled: bool = False
    evaluation_bundles_enabled: bool = False
    reports_enabled: bool = False
    shadow_bundles_enabled: bool = False
    shadow_weights_enabled: bool = False
    shadow_reimbursements_enabled: bool = False
    crowning_enabled: bool = False
    reimbursements_enabled: bool = False
    weight_mutation_enabled: bool = False
    fulfillment_mutation_enabled: bool = False
    miner_openrouter_key_required: bool = True
    loop_start_fee_usd: float = 5.0
    internal_api_key: str = ""
    hosted_worker_enabled: bool = False
    hosted_worker_poll_seconds: int = 15
    hosted_worker_max_runs: int = 0
    hosted_worker_max_candidates: int = 3
    hosted_worker_dry_run: bool = True
    hosted_worker_id: str = ""
    hosted_worker_index: int = 0
    hosted_worker_total_workers: int = 1
    hosted_worker_queue_fetch_limit: int = 20
    private_model_manifest_uri: str = (
        "s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json"
    )
    private_benchmark_path: str = ""
    score_bundle_kms_key_id: str = "alias/leadpoet-research-lab-artifact-signing"
    score_bundle_signature_uri_prefix: str = ""
    auto_research_model: str = ""
    approved_auto_research_models_json: str = ""
    default_auto_research_model_tier: str = "default"
    default_compute_budget_usd: float = 5.0
    min_compute_budget_usd: float = 1.0
    max_compute_budget_usd: float = 100.0
    topup_promising_delta_threshold: float = 0.5
    miner_openrouter_key_env_var: str = ""
    miner_openrouter_key_ref_env_map_json: str = ""
    evaluation_epoch: int = 0

    @classmethod
    def from_env(cls) -> "ResearchLabGatewayConfig":
        total_workers = max(1, _int("RESEARCH_LAB_HOSTED_WORKER_TOTAL_WORKERS", 1))
        worker_index = _int("RESEARCH_LAB_HOSTED_WORKER_INDEX", 0)
        if worker_index < 0:
            worker_index = 0
        if worker_index >= total_workers:
            worker_index = worker_index % total_workers
        return cls(
            api_enabled=_truthy("RESEARCH_LAB_GATEWAY_API_ENABLED"),
            production_writes_enabled=_truthy("RESEARCH_LAB_PRODUCTION_WRITES_ENABLED"),
            paid_loops_enabled=_truthy("RESEARCH_LAB_PAID_LOOPS_ENABLED"),
            probes_enabled=_truthy("RESEARCH_LAB_PROBES_ENABLED"),
            hosted_runs_enabled=_truthy("RESEARCH_LAB_HOSTED_RUNS_ENABLED"),
            receipts_enabled=_truthy("RESEARCH_LAB_RECEIPTS_ENABLED"),
            evaluation_bundles_enabled=_truthy("RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED"),
            reports_enabled=_truthy("RESEARCH_LAB_REPORTS_ENABLED"),
            shadow_bundles_enabled=_truthy("RESEARCH_LAB_SHADOW_BUNDLES_ENABLED"),
            shadow_weights_enabled=_truthy("RESEARCH_LAB_SHADOW_WEIGHTS_ENABLED"),
            shadow_reimbursements_enabled=_truthy("RESEARCH_LAB_SHADOW_REIMBURSEMENTS_ENABLED"),
            crowning_enabled=_truthy("RESEARCH_LAB_CROWNING_ENABLED"),
            reimbursements_enabled=_truthy("RESEARCH_LAB_REIMBURSEMENTS_ENABLED"),
            weight_mutation_enabled=_truthy("RESEARCH_LAB_WEIGHT_MUTATION_ENABLED"),
            fulfillment_mutation_enabled=_truthy("RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED"),
            miner_openrouter_key_required=_truthy("RESEARCH_LAB_MINER_OPENROUTER_KEY_REQUIRED", "true"),
            loop_start_fee_usd=_float("RESEARCH_LAB_LOOP_START_FEE_USD", 5.0),
            internal_api_key=os.getenv("RESEARCH_LAB_INTERNAL_API_KEY", ""),
            hosted_worker_enabled=_truthy("RESEARCH_LAB_HOSTED_WORKER_ENABLED"),
            hosted_worker_poll_seconds=_int("RESEARCH_LAB_HOSTED_WORKER_POLL_SECONDS", 15),
            hosted_worker_max_runs=_int("RESEARCH_LAB_HOSTED_WORKER_MAX_RUNS", 0),
            hosted_worker_max_candidates=max(1, _int("RESEARCH_LAB_HOSTED_WORKER_MAX_CANDIDATES", 3)),
            hosted_worker_dry_run=_truthy("RESEARCH_LAB_HOSTED_WORKER_DRY_RUN", "true"),
            hosted_worker_id=os.getenv("RESEARCH_LAB_HOSTED_WORKER_ID", ""),
            hosted_worker_index=worker_index,
            hosted_worker_total_workers=total_workers,
            hosted_worker_queue_fetch_limit=max(1, _int("RESEARCH_LAB_HOSTED_WORKER_QUEUE_FETCH_LIMIT", 20)),
            private_model_manifest_uri=os.getenv(
                "RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI",
                "s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json",
            ),
            private_benchmark_path=os.getenv("RESEARCH_LAB_PRIVATE_BENCHMARK_PATH", ""),
            score_bundle_kms_key_id=os.getenv(
                "RESEARCH_LAB_SCORE_BUNDLE_KMS_KEY_ID",
                "alias/leadpoet-research-lab-artifact-signing",
            ),
            score_bundle_signature_uri_prefix=os.getenv("RESEARCH_LAB_SCORE_BUNDLE_SIGNATURE_URI_PREFIX", ""),
            auto_research_model=os.getenv("RESEARCH_LAB_AUTO_RESEARCH_MODEL", ""),
            approved_auto_research_models_json=os.getenv("RESEARCH_LAB_APPROVED_AUTO_RESEARCH_MODELS_JSON", ""),
            default_auto_research_model_tier=os.getenv("RESEARCH_LAB_DEFAULT_AUTO_RESEARCH_MODEL_TIER", "default"),
            default_compute_budget_usd=_float("RESEARCH_LAB_DEFAULT_COMPUTE_BUDGET_USD", 5.0),
            min_compute_budget_usd=_float("RESEARCH_LAB_MIN_COMPUTE_BUDGET_USD", 1.0),
            max_compute_budget_usd=_float("RESEARCH_LAB_MAX_COMPUTE_BUDGET_USD", 100.0),
            topup_promising_delta_threshold=_float("RESEARCH_LAB_TOPUP_PROMISING_DELTA_THRESHOLD", 0.5),
            miner_openrouter_key_env_var=os.getenv("RESEARCH_LAB_MINER_OPENROUTER_KEY_ENV_VAR", ""),
            miner_openrouter_key_ref_env_map_json=os.getenv("RESEARCH_LAB_OPENROUTER_KEY_REF_ENV_MAP_JSON", ""),
            evaluation_epoch=_int("RESEARCH_LAB_EVALUATION_EPOCH", 0),
        )

    def approved_auto_research_models(self) -> dict[str, dict[str, Any]]:
        configured = self._decode_model_tiers()
        if configured:
            return configured
        if not self.auto_research_model:
            return {}
        return {
            self.default_auto_research_model_tier: {
                "model": self.auto_research_model,
                "max_candidates": self.hosted_worker_max_candidates,
                "description": "Default hosted auto-research model",
            }
        }

    def resolve_auto_research_model(self, tier: str | None) -> tuple[str, str, dict[str, Any]]:
        tiers = self.approved_auto_research_models()
        if not tiers:
            if self.auto_research_model:
                return self.default_auto_research_model_tier, self.auto_research_model, {
                    "model": self.auto_research_model,
                    "max_candidates": self.hosted_worker_max_candidates,
                }
            raise ValueError("no hosted auto-research model is configured")
        effective_tier = str(tier or self.default_auto_research_model_tier or "default")
        if not tier and effective_tier not in tiers:
            effective_tier = sorted(tiers)[0]
        if effective_tier not in tiers:
            raise ValueError(f"auto-research model tier is not approved: {effective_tier}")
        doc = dict(tiers[effective_tier])
        model = str(doc.get("model") or "")
        if not model:
            raise ValueError(f"approved auto-research model tier has no model: {effective_tier}")
        return effective_tier, model, doc

    def clamp_compute_budget_usd(self, value: float | int | str | None) -> float:
        try:
            budget = float(self.default_compute_budget_usd if value is None else value)
        except (TypeError, ValueError):
            budget = float(self.default_compute_budget_usd)
        lower = max(0.0, float(self.min_compute_budget_usd))
        upper = max(lower, float(self.max_compute_budget_usd))
        return min(max(budget, lower), upper)

    def _decode_model_tiers(self) -> dict[str, dict[str, Any]]:
        if not self.approved_auto_research_models_json:
            return {}
        try:
            decoded = json.loads(self.approved_auto_research_models_json)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid RESEARCH_LAB_APPROVED_AUTO_RESEARCH_MODELS_JSON: %s", exc)
            return {}
        if not isinstance(decoded, Mapping):
            logger.warning("RESEARCH_LAB_APPROVED_AUTO_RESEARCH_MODELS_JSON must decode to an object")
            return {}
        tiers: dict[str, dict[str, Any]] = {}
        for name, value in decoded.items():
            if not isinstance(value, Mapping):
                logger.warning("Skipping invalid auto-research model tier %s: expected object", name)
                continue
            model = str(value.get("model") or "")
            if not model:
                logger.warning("Skipping invalid auto-research model tier %s: missing model", name)
                continue
            tiers[str(name)] = {
                key: item
                for key, item in dict(value).items()
                if key in {"model", "max_candidates", "max_compute_budget_usd", "description"}
            }
        return tiers

    def live_mutation_flags(self) -> dict[str, bool]:
        return {
            "RESEARCH_LAB_PAID_LOOPS_ENABLED": self.paid_loops_enabled,
            "RESEARCH_LAB_HOSTED_RUNS_ENABLED": self.hosted_runs_enabled,
            "RESEARCH_LAB_PROBES_ENABLED": self.probes_enabled,
            "RESEARCH_LAB_CROWNING_ENABLED": self.crowning_enabled,
            "RESEARCH_LAB_REIMBURSEMENTS_ENABLED": self.reimbursements_enabled,
            "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": self.weight_mutation_enabled,
            "RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED": self.fulfillment_mutation_enabled,
        }

    def public_status(self) -> dict[str, object]:
        return {
            "api_enabled": self.api_enabled,
            "production_writes_enabled": self.production_writes_enabled,
            "paid_loops_enabled": self.paid_loops_enabled,
            "probes_enabled": self.probes_enabled,
            "hosted_runs_enabled": self.hosted_runs_enabled,
            "receipts_enabled": self.receipts_enabled,
            "evaluation_bundles_enabled": self.evaluation_bundles_enabled,
            "reports_enabled": self.reports_enabled,
            "shadow_flags": {
                "RESEARCH_LAB_SHADOW_BUNDLES_ENABLED": self.shadow_bundles_enabled,
                "RESEARCH_LAB_SHADOW_WEIGHTS_ENABLED": self.shadow_weights_enabled,
                "RESEARCH_LAB_SHADOW_REIMBURSEMENTS_ENABLED": self.shadow_reimbursements_enabled,
            },
            "live_mutation_flags": self.live_mutation_flags(),
            "loop_start_fee_usd": self.loop_start_fee_usd,
            "miner_openrouter_key_required": self.miner_openrouter_key_required,
            "hosted_worker": {
                "enabled": self.hosted_worker_enabled,
                "dry_run": self.hosted_worker_dry_run,
                "poll_seconds": self.hosted_worker_poll_seconds,
                "max_candidates": self.hosted_worker_max_candidates,
                "worker_id": self.hosted_worker_id,
                "worker_index": self.hosted_worker_index,
                "total_workers": self.hosted_worker_total_workers,
                "queue_fetch_limit": self.hosted_worker_queue_fetch_limit,
                "private_model_manifest_uri_configured": bool(self.private_model_manifest_uri),
                "private_benchmark_path_configured": bool(self.private_benchmark_path),
                "auto_research_model_configured": bool(self.auto_research_model),
                "approved_model_tiers": {
                    tier: {
                        "model_configured": bool(doc.get("model")),
                        "max_candidates": doc.get("max_candidates", self.hosted_worker_max_candidates),
                        "max_compute_budget_usd": doc.get("max_compute_budget_usd", self.max_compute_budget_usd),
                        "description": doc.get("description"),
                    }
                    for tier, doc in self.approved_auto_research_models().items()
                },
                "default_model_tier": self.default_auto_research_model_tier,
                "default_compute_budget_usd": self.default_compute_budget_usd,
                "min_compute_budget_usd": self.min_compute_budget_usd,
                "max_compute_budget_usd": self.max_compute_budget_usd,
            },
        }
