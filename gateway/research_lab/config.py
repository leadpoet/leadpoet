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

# Single code-level default for the Research Lab loop-start fee. Operators can
# still override it at runtime with RESEARCH_LAB_LOOP_START_FEE_USD.
DEFAULT_LOOP_START_FEE_USD = 0.2


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
    auto_promotion_enabled: bool = False
    auto_commit_enabled: bool = False
    miner_openrouter_key_required: bool = True
    loop_start_fee_usd: float = DEFAULT_LOOP_START_FEE_USD
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
    hosted_worker_require_proxy: bool = False
    hosted_worker_proxy_url: str = ""
    scoring_worker_enabled: bool = False
    scoring_worker_poll_seconds: int = 15
    scoring_worker_max_candidates: int = 1
    scoring_worker_id: str = ""
    scoring_worker_index: int = 0
    scoring_worker_total_workers: int = 1
    scoring_worker_require_proxy: bool = False
    scoring_worker_proxy_url: str = ""
    scoring_worker_model_timeout_seconds: int = 900
    scoring_worker_allow_partial_icp_window: bool = False
    private_baseline_rebenchmark_enabled: bool = False
    auto_research_min_seconds: int = 600
    auto_research_max_seconds: int = 1800
    auto_research_min_iterations: int = 2
    auto_research_max_iterations: int = 12
    auto_research_draft_timeout_seconds: int = 90
    auto_research_reflection_timeout_seconds: int = 90
    auto_research_estimated_iteration_cost_usd: float = 0.5
    reimbursement_policy_id: str = "alpha-reimbursement-production-v1"
    reimbursement_min_rebate_rate: float = 0.25
    reimbursement_base_rebate_rate: float = 0.50
    reimbursement_max_rebate_rate: float = 0.80
    reimbursement_high_participation_target: float = 10.0
    reimbursement_epochs: int = 20
    reimbursement_max_usd_per_run: float = 100.0
    reimbursement_max_usd_per_hotkey_day: float = 100.0
    reimbursement_max_usd_per_island_day: float = 1000.0
    reimbursement_global_budget_usd: float = 5000.0
    reimbursement_material_spend_ratio: float = 0.80
    reimbursement_default_island: str = "generalist"
    reimbursement_usd_per_0_1_percent_epoch: float = 0.162
    lab_emission_percent: float = 10.0
    fulfillment_emission_percent: float = 80.5
    fulfillment_leaderboard_emission_percent: float = 9.5
    lab_reward_epochs: int = 20
    lab_reimbursement_allow_overpay_without_champions: bool = True
    lab_reimbursement_max_cost_multiplier_with_champions: float = 1.0
    lab_reimbursement_min_alpha_percent: float = 0.0
    lab_champion_min_alpha_percent: float = 2.0
    lab_champion_extra_alpha_percent_per_point: float = 0.1
    lab_champion_max_alpha_percent: float = 5.0
    lab_champion_placeholder_alpha_percent: float = 0.0001
    lab_champion_queue_trigger_ratio: float = 0.50
    lab_champion_threshold_points: float = 2.0
    lab_champion_eval_days: int = 10
    lab_champion_icps_per_day: int = 5
    improvement_threshold_points: float = 1.0
    improvement_min_delta_lcb: float = 0.0
    private_model_manifest_uri: str = (
        "s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json"
    )
    private_repo_url: str = ""
    private_repo_branch: str = "main"
    private_patch_applier_cmd: str = ""
    private_test_cmd: str = ""
    private_build_cmd: str = ""
    private_artifact_manifest_output: str = ""
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
    openrouter_key_kms_key_id: str = ""
    evaluation_epoch: int = 0
    arweave_audit_enabled: bool = False
    arweave_audit_shadow_enabled: bool = False

    @classmethod
    def from_env(cls) -> "ResearchLabGatewayConfig":
        total_workers = max(1, _int("RESEARCH_LAB_HOSTED_WORKER_TOTAL_WORKERS", 1))
        worker_index = _int("RESEARCH_LAB_HOSTED_WORKER_INDEX", 0)
        if worker_index < 0:
            worker_index = 0
        if worker_index >= total_workers:
            worker_index = worker_index % total_workers
        scoring_total_workers = max(1, _int("RESEARCH_LAB_SCORING_WORKER_TOTAL_WORKERS", 1))
        scoring_worker_index = _int("RESEARCH_LAB_SCORING_WORKER_INDEX", 0)
        if scoring_worker_index < 0:
            scoring_worker_index = 0
        if scoring_worker_index >= scoring_total_workers:
            scoring_worker_index = scoring_worker_index % scoring_total_workers
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
            auto_promotion_enabled=_truthy("RESEARCH_LAB_AUTO_PROMOTION_ENABLED"),
            auto_commit_enabled=_truthy("RESEARCH_LAB_AUTO_COMMIT_ENABLED"),
            miner_openrouter_key_required=_truthy("RESEARCH_LAB_MINER_OPENROUTER_KEY_REQUIRED", "true"),
            loop_start_fee_usd=_float("RESEARCH_LAB_LOOP_START_FEE_USD", DEFAULT_LOOP_START_FEE_USD),
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
            hosted_worker_require_proxy=(
                _truthy("RESEARCH_LAB_REQUIRE_WORKER_PROXY")
                or _truthy("RESEARCH_LAB_HOSTED_WORKER_REQUIRE_PROXY")
            ),
            hosted_worker_proxy_url=os.getenv("RESEARCH_LAB_HOSTED_WORKER_PROXY", ""),
            scoring_worker_enabled=_truthy("RESEARCH_LAB_SCORING_WORKER_ENABLED"),
            scoring_worker_poll_seconds=max(1, _int("RESEARCH_LAB_SCORING_WORKER_POLL_SECONDS", 15)),
            scoring_worker_max_candidates=max(1, _int("RESEARCH_LAB_SCORING_WORKER_MAX_CANDIDATES", 1)),
            scoring_worker_id=os.getenv("RESEARCH_LAB_SCORING_WORKER_ID", ""),
            scoring_worker_index=scoring_worker_index,
            scoring_worker_total_workers=scoring_total_workers,
            scoring_worker_require_proxy=(
                _truthy("RESEARCH_LAB_REQUIRE_QUALIFICATION_PROXY")
                or _truthy("RESEARCH_LAB_SCORING_WORKER_REQUIRE_PROXY")
            ),
            scoring_worker_proxy_url=os.getenv("RESEARCH_LAB_SCORING_WORKER_PROXY", ""),
            scoring_worker_model_timeout_seconds=max(
                30,
                _int("RESEARCH_LAB_SCORING_WORKER_MODEL_TIMEOUT_SECONDS", 900),
            ),
            scoring_worker_allow_partial_icp_window=_truthy(
                "RESEARCH_LAB_SCORING_ALLOW_PARTIAL_ICP_WINDOW",
                "false",
            ),
            private_baseline_rebenchmark_enabled=_truthy(
                "RESEARCH_LAB_PRIVATE_BASELINE_REBENCHMARK_ENABLED",
                "false",
            ),
            auto_research_min_seconds=max(0, _int("RESEARCH_LAB_AUTO_RESEARCH_MIN_SECONDS", 600)),
            auto_research_max_seconds=max(1, _int("RESEARCH_LAB_AUTO_RESEARCH_MAX_SECONDS", 1800)),
            auto_research_min_iterations=max(1, _int("RESEARCH_LAB_AUTO_RESEARCH_MIN_ITERATIONS", 2)),
            auto_research_max_iterations=max(1, _int("RESEARCH_LAB_AUTO_RESEARCH_MAX_ITERATIONS", 12)),
            auto_research_draft_timeout_seconds=max(
                10,
                _int("RESEARCH_LAB_AUTO_RESEARCH_DRAFT_TIMEOUT_SECONDS", 90),
            ),
            auto_research_reflection_timeout_seconds=max(
                10,
                _int("RESEARCH_LAB_AUTO_RESEARCH_REFLECTION_TIMEOUT_SECONDS", 90),
            ),
            auto_research_estimated_iteration_cost_usd=max(
                0.01,
                _float("RESEARCH_LAB_AUTO_RESEARCH_ESTIMATED_ITERATION_COST_USD", 0.5),
            ),
            reimbursement_policy_id=os.getenv(
                "RESEARCH_LAB_REIMBURSEMENT_POLICY_ID",
                "alpha-reimbursement-production-v1",
            ),
            reimbursement_min_rebate_rate=max(0.0, _float("RESEARCH_LAB_REIMBURSEMENT_MIN_REBATE_RATE", 0.25)),
            reimbursement_base_rebate_rate=max(0.0, _float("RESEARCH_LAB_REIMBURSEMENT_BASE_REBATE_RATE", 0.50)),
            reimbursement_max_rebate_rate=max(0.0, _float("RESEARCH_LAB_REIMBURSEMENT_MAX_REBATE_RATE", 0.80)),
            reimbursement_high_participation_target=max(
                0.01,
                _float("RESEARCH_LAB_REIMBURSEMENT_HIGH_PARTICIPATION_TARGET", 10.0),
            ),
            reimbursement_epochs=max(
                1,
                _int(
                    "RESEARCH_LAB_REIMBURSEMENT_EPOCHS",
                    _int("RESEARCH_LAB_REWARD_EPOCHS", 20),
                ),
            ),
            reimbursement_max_usd_per_run=max(0.0, _float("RESEARCH_LAB_REIMBURSEMENT_MAX_USD_PER_RUN", 100.0)),
            reimbursement_max_usd_per_hotkey_day=max(
                0.0,
                _float("RESEARCH_LAB_REIMBURSEMENT_MAX_USD_PER_HOTKEY_DAY", 100.0),
            ),
            reimbursement_max_usd_per_island_day=max(
                0.0,
                _float("RESEARCH_LAB_REIMBURSEMENT_MAX_USD_PER_ISLAND_DAY", 1000.0),
            ),
            reimbursement_global_budget_usd=max(
                0.0,
                _float("RESEARCH_LAB_REIMBURSEMENT_GLOBAL_BUDGET_USD", 5000.0),
            ),
            reimbursement_material_spend_ratio=min(
                1.0,
                max(0.0, _float("RESEARCH_LAB_REIMBURSEMENT_MATERIAL_SPEND_RATIO", 0.80)),
            ),
            reimbursement_default_island=os.getenv("RESEARCH_LAB_REIMBURSEMENT_DEFAULT_ISLAND", "generalist"),
            reimbursement_usd_per_0_1_percent_epoch=max(
                0.000001,
                _float("RESEARCH_LAB_REIMBURSEMENT_USD_PER_0_1_PERCENT_EPOCH", 0.162),
            ),
            lab_emission_percent=max(0.0, _float("RESEARCH_LAB_EMISSION_PERCENT", 10.0)),
            fulfillment_emission_percent=max(0.0, _float("RESEARCH_LAB_FULFILLMENT_EMISSION_PERCENT", 80.5)),
            fulfillment_leaderboard_emission_percent=max(
                0.0,
                _float("RESEARCH_LAB_FULFILLMENT_LEADERBOARD_EMISSION_PERCENT", 9.5),
            ),
            lab_reward_epochs=max(1, _int("RESEARCH_LAB_REWARD_EPOCHS", 20)),
            lab_reimbursement_allow_overpay_without_champions=_truthy(
                "RESEARCH_LAB_REIMBURSEMENT_ALLOW_OVERPAY_WITHOUT_CHAMPIONS",
                "true",
            ),
            lab_reimbursement_max_cost_multiplier_with_champions=max(
                0.0,
                _float("RESEARCH_LAB_REIMBURSEMENT_MAX_COST_MULTIPLIER_WITH_CHAMPIONS", 1.0),
            ),
            lab_reimbursement_min_alpha_percent=max(
                0.0,
                _float("RESEARCH_LAB_REIMBURSEMENT_MIN_ALPHA_PERCENT", 0.0),
            ),
            lab_champion_min_alpha_percent=max(0.0, _float("RESEARCH_LAB_CHAMPION_MIN_ALPHA_PERCENT", 2.0)),
            lab_champion_extra_alpha_percent_per_point=max(
                0.0,
                _float("RESEARCH_LAB_CHAMPION_EXTRA_ALPHA_PERCENT_PER_POINT", 0.1),
            ),
            lab_champion_max_alpha_percent=max(0.0, _float("RESEARCH_LAB_CHAMPION_MAX_ALPHA_PERCENT", 5.0)),
            lab_champion_placeholder_alpha_percent=max(
                0.0,
                _float("RESEARCH_LAB_CHAMPION_PLACEHOLDER_ALPHA_PERCENT", 0.0001),
            ),
            lab_champion_queue_trigger_ratio=min(
                1.0,
                max(0.0, _float("RESEARCH_LAB_CHAMPION_QUEUE_TRIGGER_RATIO", 0.50)),
            ),
            lab_champion_threshold_points=max(0.0, _float("RESEARCH_LAB_CHAMPION_THRESHOLD_POINTS", 2.0)),
            lab_champion_eval_days=max(1, _int("RESEARCH_LAB_CHAMPION_EVAL_DAYS", 10)),
            lab_champion_icps_per_day=max(1, _int("RESEARCH_LAB_CHAMPION_ICPS_PER_DAY", 5)),
            improvement_threshold_points=max(
                0.0,
                _float("RESEARCH_LAB_IMPROVEMENT_THRESHOLD_POINTS", 1.0),
            ),
            improvement_min_delta_lcb=_float("RESEARCH_LAB_IMPROVEMENT_MIN_DELTA_LCB", 0.0),
            private_model_manifest_uri=os.getenv(
                "RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI",
                "s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json",
            ),
            private_repo_url=os.getenv("RESEARCH_LAB_PRIVATE_REPO_URL", ""),
            private_repo_branch=os.getenv("RESEARCH_LAB_PRIVATE_REPO_BRANCH", "main") or "main",
            private_patch_applier_cmd=os.getenv("RESEARCH_LAB_PRIVATE_PATCH_APPLIER_CMD", ""),
            private_test_cmd=os.getenv("RESEARCH_LAB_PRIVATE_TEST_CMD", ""),
            private_build_cmd=os.getenv("RESEARCH_LAB_PRIVATE_BUILD_CMD", ""),
            private_artifact_manifest_output=os.getenv("RESEARCH_LAB_PRIVATE_ARTIFACT_MANIFEST_OUTPUT", ""),
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
            openrouter_key_kms_key_id=os.getenv("RESEARCH_LAB_OPENROUTER_KEY_KMS_KEY_ID", ""),
            evaluation_epoch=_int("RESEARCH_LAB_EVALUATION_EPOCH", 0),
            arweave_audit_enabled=_truthy("RESEARCH_LAB_ARWEAVE_AUDIT_ENABLED"),
            arweave_audit_shadow_enabled=_truthy("RESEARCH_LAB_ARWEAVE_AUDIT_SHADOW_ENABLED"),
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
            "RESEARCH_LAB_AUTO_PROMOTION_ENABLED": self.auto_promotion_enabled,
            "RESEARCH_LAB_AUTO_COMMIT_ENABLED": self.auto_commit_enabled,
        }

    def reimbursement_policy_doc(self, *, enabled: bool | None = None) -> dict[str, object]:
        return {
            "policy_id": self.reimbursement_policy_id,
            "enabled": self.reimbursements_enabled if enabled is None else bool(enabled),
            "min_rebate_rate": self.reimbursement_min_rebate_rate,
            "base_rebate_rate": self.reimbursement_base_rebate_rate,
            "max_rebate_rate": self.reimbursement_max_rebate_rate,
            "high_participation_target": self.reimbursement_high_participation_target,
            "reimbursement_epochs": self.reimbursement_epochs,
            "max_usd_per_run": self.reimbursement_max_usd_per_run,
            "max_usd_per_hotkey_day": self.reimbursement_max_usd_per_hotkey_day,
            "max_usd_per_island_day": self.reimbursement_max_usd_per_island_day,
            "global_budget_usd": self.reimbursement_global_budget_usd,
            "include_loop_start_fee_in_base": False,
            "material_spend_ratio": self.reimbursement_material_spend_ratio,
            "default_island": self.reimbursement_default_island,
            "usd_per_0_1_percent_epoch": self.reimbursement_usd_per_0_1_percent_epoch,
            "distinct_funded_hotkey_weight": 1,
            "paid_loop_weight": 1,
            "unique_brief_weight": 1,
            "research_lab_emission_percent": self.lab_emission_percent,
            "fulfillment_emission_percent": self.fulfillment_emission_percent,
            "fulfillment_leaderboard_emission_percent": self.fulfillment_leaderboard_emission_percent,
            "reward_epochs": self.lab_reward_epochs,
            "reimbursement_allow_overpay_without_champions": (
                self.lab_reimbursement_allow_overpay_without_champions
            ),
            "reimbursement_max_cost_multiplier_with_champions": (
                self.lab_reimbursement_max_cost_multiplier_with_champions
            ),
            "reimbursement_min_alpha_percent": self.lab_reimbursement_min_alpha_percent,
            "champion_min_alpha_percent": self.lab_champion_min_alpha_percent,
            "champion_extra_alpha_percent_per_point": self.lab_champion_extra_alpha_percent_per_point,
            "champion_max_alpha_percent": self.lab_champion_max_alpha_percent,
            "champion_placeholder_alpha_percent": self.lab_champion_placeholder_alpha_percent,
            "champion_queue_trigger_ratio": self.lab_champion_queue_trigger_ratio,
            "champion_threshold_points": self.lab_champion_threshold_points,
            "champion_eval_days": self.lab_champion_eval_days,
            "champion_icps_per_day": self.lab_champion_icps_per_day,
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
            "openrouter_key_registration_enabled": bool(self.openrouter_key_kms_key_id),
            "reimbursement": {
                "enabled": self.reimbursements_enabled,
                "shadow_enabled": self.shadow_reimbursements_enabled,
                "policy_id": self.reimbursement_policy_id,
                "default_island": self.reimbursement_default_island,
                "min_rebate_rate": self.reimbursement_min_rebate_rate,
                "base_rebate_rate": self.reimbursement_base_rebate_rate,
                "max_rebate_rate": self.reimbursement_max_rebate_rate,
                "default_rebate_rate_estimate": self.reimbursement_base_rebate_rate,
                "reimbursement_epochs": self.reimbursement_epochs,
                "material_spend_ratio": self.reimbursement_material_spend_ratio,
                "usd_per_0_1_percent_epoch": self.reimbursement_usd_per_0_1_percent_epoch,
                "loop_start_fee_included": False,
                "lab_emission_percent": self.lab_emission_percent,
                "fulfillment_emission_percent": self.fulfillment_emission_percent,
                "fulfillment_leaderboard_emission_percent": self.fulfillment_leaderboard_emission_percent,
                "reward_epochs": self.lab_reward_epochs,
                "allow_overpay_without_champions": self.lab_reimbursement_allow_overpay_without_champions,
                "max_cost_multiplier_with_champions": self.lab_reimbursement_max_cost_multiplier_with_champions,
                "champion_min_alpha_percent": self.lab_champion_min_alpha_percent,
                "champion_extra_alpha_percent_per_point": self.lab_champion_extra_alpha_percent_per_point,
                "champion_max_alpha_percent": self.lab_champion_max_alpha_percent,
                "champion_placeholder_alpha_percent": self.lab_champion_placeholder_alpha_percent,
                "champion_threshold_points": self.lab_champion_threshold_points,
                "champion_eval_days": self.lab_champion_eval_days,
                "champion_icps_per_day": self.lab_champion_icps_per_day,
                "improvement_threshold_points": self.improvement_threshold_points,
                "improvement_min_delta_lcb": self.improvement_min_delta_lcb,
            },
            "arweave_audit": {
                "enabled": self.arweave_audit_enabled,
                "shadow_enabled": self.arweave_audit_shadow_enabled,
                "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
            },
            "hosted_worker": {
                "enabled": self.hosted_worker_enabled,
                "dry_run": self.hosted_worker_dry_run,
                "poll_seconds": self.hosted_worker_poll_seconds,
                "max_candidates": self.hosted_worker_max_candidates,
                "worker_id": self.hosted_worker_id,
                "worker_index": self.hosted_worker_index,
                "total_workers": self.hosted_worker_total_workers,
                "queue_fetch_limit": self.hosted_worker_queue_fetch_limit,
                "require_proxy": self.hosted_worker_require_proxy,
                "worker_proxy_configured": bool(self.hosted_worker_proxy_url),
                "auto_research_min_seconds": self.auto_research_min_seconds,
                "auto_research_max_seconds": self.auto_research_max_seconds,
                "auto_research_min_iterations": self.auto_research_min_iterations,
                "auto_research_max_iterations": self.auto_research_max_iterations,
                "auto_research_draft_timeout_seconds": self.auto_research_draft_timeout_seconds,
                "auto_research_reflection_timeout_seconds": self.auto_research_reflection_timeout_seconds,
                "auto_research_estimated_iteration_cost_usd": self.auto_research_estimated_iteration_cost_usd,
                "private_model_manifest_uri_configured": bool(self.private_model_manifest_uri),
                "scoring_owner": "gateway_qualification_workers",
                "scoring_worker_enabled": self.scoring_worker_enabled,
                "scoring_worker_poll_seconds": self.scoring_worker_poll_seconds,
                "scoring_worker_max_candidates": self.scoring_worker_max_candidates,
                "scoring_worker_id": self.scoring_worker_id,
                "scoring_worker_index": self.scoring_worker_index,
                "scoring_worker_total_workers": self.scoring_worker_total_workers,
                "scoring_worker_require_proxy": self.scoring_worker_require_proxy,
                "scoring_worker_proxy_configured": bool(self.scoring_worker_proxy_url),
                "scoring_worker_allow_partial_icp_window": self.scoring_worker_allow_partial_icp_window,
                "private_baseline_rebenchmark_enabled": self.private_baseline_rebenchmark_enabled,
                "auto_promotion_enabled": self.auto_promotion_enabled,
                "auto_commit_enabled": self.auto_commit_enabled,
                "private_repo_configured": bool(self.private_repo_url),
                "private_patch_applier_configured": bool(self.private_patch_applier_cmd),
                "private_test_cmd_configured": bool(self.private_test_cmd),
                "private_build_cmd_configured": bool(self.private_build_cmd),
                "private_artifact_manifest_output_configured": bool(self.private_artifact_manifest_output),
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
