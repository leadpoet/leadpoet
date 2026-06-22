"""Production Research Lab gateway flags.

All live workflow flags default false. Enabling the API only exposes the
Research Lab namespace; paid loops, probes, writes, receipts, and weights each
require their own explicit gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


TRUTHY = {"1", "true", "yes", "on"}


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
            miner_openrouter_key_env_var=os.getenv("RESEARCH_LAB_MINER_OPENROUTER_KEY_ENV_VAR", ""),
            miner_openrouter_key_ref_env_map_json=os.getenv("RESEARCH_LAB_OPENROUTER_KEY_REF_ENV_MAP_JSON", ""),
            evaluation_epoch=_int("RESEARCH_LAB_EVALUATION_EPOCH", 0),
        )

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
            },
        }
