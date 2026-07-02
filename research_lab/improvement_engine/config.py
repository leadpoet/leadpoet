"""Configuration for the gated Research Lab Improvement Engine."""

from __future__ import annotations

from dataclasses import dataclass
import os


TRUTHY = {"1", "true", "yes", "on"}


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in TRUTHY


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class ImprovementEngineConfig:
    enabled: bool = False
    mode: str = "passive"
    scan_interval_hours: int = 6
    lookback_hours: int = 72
    max_traces_per_scan: int = 5000
    min_cluster_size: int = 3
    min_severity: str = "medium"
    auto_create_evaluators: bool = False
    auto_create_datasets: bool = False
    auto_generate_patches: bool = False
    auto_open_prs: bool = False
    auto_submit_candidates: bool = False
    # Plan §8.3 contract flag: fix_generator additionally hardcodes
    # auto_apply_allowed=False and asserts this stays false — patches are
    # never auto-applied regardless of env.
    auto_apply_patches: bool = False
    publish_miner_opportunities: bool = False
    max_llm_cost_usd_per_scan: float = 20.0
    model: str = ""
    notify_webhook_url: str = ""
    github_repo: str = ""

    @classmethod
    def from_env(cls) -> "ImprovementEngineConfig":
        return cls(
            enabled=_flag("RESEARCH_LAB_IMPROVEMENT_ENGINE_ENABLED", False),
            mode=os.getenv("RESEARCH_LAB_IMPROVEMENT_ENGINE_MODE", "passive").strip() or "passive",
            scan_interval_hours=max(1, _int("RESEARCH_LAB_IMPROVEMENT_ENGINE_SCAN_INTERVAL_HOURS", 6)),
            lookback_hours=max(1, _int("RESEARCH_LAB_IMPROVEMENT_ENGINE_LOOKBACK_HOURS", 72)),
            max_traces_per_scan=max(1, _int("RESEARCH_LAB_IMPROVEMENT_ENGINE_MAX_TRACES_PER_SCAN", 5000)),
            min_cluster_size=max(1, _int("RESEARCH_LAB_IMPROVEMENT_ENGINE_MIN_CLUSTER_SIZE", 3)),
            min_severity=os.getenv("RESEARCH_LAB_IMPROVEMENT_ENGINE_MIN_SEVERITY", "medium").strip() or "medium",
            auto_create_evaluators=_flag("RESEARCH_LAB_IMPROVEMENT_ENGINE_AUTO_CREATE_EVALUATORS", False),
            auto_create_datasets=_flag("RESEARCH_LAB_IMPROVEMENT_ENGINE_AUTO_CREATE_DATASETS", False),
            auto_generate_patches=_flag("RESEARCH_LAB_IMPROVEMENT_ENGINE_AUTO_GENERATE_PATCHES", False),
            auto_open_prs=_flag("RESEARCH_LAB_IMPROVEMENT_ENGINE_AUTO_OPEN_PRS", False),
            auto_submit_candidates=_flag("RESEARCH_LAB_IMPROVEMENT_ENGINE_AUTO_SUBMIT_CANDIDATES", False),
            auto_apply_patches=_flag("RESEARCH_LAB_IMPROVEMENT_ENGINE_AUTO_APPLY_PATCHES", False),
            publish_miner_opportunities=_flag("RESEARCH_LAB_IMPROVEMENT_ENGINE_PUBLISH_MINER_OPPORTUNITIES", False),
            max_llm_cost_usd_per_scan=max(0.0, _float("RESEARCH_LAB_IMPROVEMENT_ENGINE_MAX_LLM_COST_USD_PER_SCAN", 20.0)),
            model=os.getenv("RESEARCH_LAB_IMPROVEMENT_ENGINE_MODEL", "").strip(),
            notify_webhook_url=os.getenv("RESEARCH_LAB_IMPROVEMENT_ENGINE_NOTIFY_WEBHOOK_URL", "").strip(),
            github_repo=os.getenv("RESEARCH_LAB_IMPROVEMENT_ENGINE_GITHUB_REPO", "").strip(),
        )

    @property
    def mutations_allowed(self) -> bool:
        return self.mode == "active" and any(
            (
                self.auto_create_evaluators,
                self.auto_create_datasets,
                self.auto_generate_patches,
                self.auto_open_prs,
                self.auto_submit_candidates,
                self.publish_miner_opportunities,
            )
        )
