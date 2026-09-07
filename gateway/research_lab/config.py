"""Current Research Lab gateway and retained reward settings.

The Research Lab no longer builds or promotes sourcing models. Agent
competition settings live in :mod:`lab_arena`. This module keeps only the
SOURCE_ADD controls and the allocation settings needed to settle existing
Research Lab rewards beside the Arena reward path.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from leadpoet_verifier.economics import (
    DEFAULT_REIMBURSEMENT_MAX_COST_MULTIPLIER_WITH_CHAMPIONS,
    DEFAULT_RESEARCH_LAB_CHAMPION_QUEUE_TRIGGER_RATIO,
    DEFAULT_RESEARCH_LAB_EMISSION_PERCENT,
)


TRUTHY = {"1", "true", "yes", "on"}

# The V2 scoring enclave still supports the normal qualification pipeline.
# These proxy names are used only to size and seal that current worker fleet.
V2_SCORING_PROXY_PREFIXES = ("RESEARCH_LAB_V2_SCORING_HTTPS_PROXY",)
LEGACY_SCORING_PROXY_PREFIXES = (
    "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY",
    "QUALIFICATION_WEBSHARE_PROXY",
    "RESEARCH_LAB_SCORING_WORKER_PROXY",
)
SCORING_PROXY_PREFIXES = (
    *V2_SCORING_PROXY_PREFIXES,
    *LEGACY_SCORING_PROXY_PREFIXES,
)

MAX_WORKER_PROCESSES = 500


def _is_production_subnet() -> bool:
    network = (
        os.getenv("BITTENSOR_NETWORK")
        or os.getenv("SUBTENSOR_NETWORK")
        or ""
    ).strip().lower()
    netuid = (os.getenv("BITTENSOR_NETUID") or os.getenv("NETUID") or "").strip()
    return network == "finney" and netuid == "71"


def _prod_default(default_for_prod: bool, default_for_non_prod: bool = False) -> str:
    enabled = (
        (_is_production_subnet() and default_for_prod)
        or (not _is_production_subnet() and default_for_non_prod)
    )
    return "true" if enabled else "false"


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


def resolve_worker_process_count(
    explicit_count: int,
    fallback_count: int,
    *,
    minimum: int = 0,
) -> int:
    """Return one bounded scoring-worker count for sealing and startup."""

    chosen = explicit_count if explicit_count > 0 else fallback_count
    return max(minimum, min(int(chosen), MAX_WORKER_PROCESSES))


@dataclass(frozen=True)
class ResearchLabGatewayConfig:
    """Controls still consumed by SOURCE_ADD and reward settlement."""

    api_enabled: bool = False
    production_writes_enabled: bool = False
    reports_enabled: bool = False
    shadow_bundles_enabled: bool = False
    shadow_reimbursements_enabled: bool = False
    reimbursements_enabled: bool = False
    weight_mutation_enabled: bool = False
    fulfillment_mutation_enabled: bool = False
    internal_api_key: str = ""

    evaluation_epoch: int = 0

    reimbursement_policy_id: str = "alpha-reimbursement-production-v1"
    reimbursement_epochs: int = 20
    reimbursement_usd_per_0_1_percent_epoch: float = 0.162
    reimbursement_dynamic_alpha_price_enabled: bool = True
    reimbursement_require_live_alpha_price: bool = False
    reimbursement_miner_alpha_per_epoch: float = 147.6
    lab_emission_percent: float = float(DEFAULT_RESEARCH_LAB_EMISSION_PERCENT)
    fulfillment_emission_percent: float = 60.5
    fulfillment_leaderboard_emission_percent: float = 9.5
    lab_reward_epochs: int = 20
    enable_conservative: bool = True
    enable_champ_cap: bool = True
    lab_reimbursement_max_cost_multiplier_with_champions: float = float(
        DEFAULT_REIMBURSEMENT_MAX_COST_MULTIPLIER_WITH_CHAMPIONS
    )
    lab_reimbursement_min_alpha_percent: float = 0.0
    lab_champion_min_alpha_percent: float = 7.0
    lab_champion_extra_alpha_percent_per_point: float = 0.3
    lab_champion_max_alpha_percent: float = 15.0
    lab_champion_placeholder_alpha_percent: float = 0.0001
    lab_champion_queue_trigger_ratio: float = float(
        DEFAULT_RESEARCH_LAB_CHAMPION_QUEUE_TRIGGER_RATIO
    )
    lab_champion_threshold_points: float = 1.0

    # SOURCE_ADD remains a separate live product-input flow.
    source_add_enabled: bool = True
    source_add_rewards_enabled: bool = True
    source_add_dispatcher_enabled: bool = True
    source_add_functional_probes_enabled: bool = True
    source_add_functional_rewards_enabled: bool = True
    source_add_dispatcher_poll_seconds: float = 2.0
    source_add_work_lease_seconds: int = 300
    source_add_probe_timeout_seconds: int = 45
    source_add_probe_max_attempts: int = 5
    source_add_credential_kms_key_id: str = ""
    source_add_sandbox_image: str = "python:3.11-slim"
    source_add_trial_timeout_seconds: int = 300
    source_add_leg1_alpha_percent: float = 0.2
    source_add_leg2_alpha_percent: float = 0.0
    source_add_acceptance_floor_yield: float = 0.10
    source_add_max_concurrent_per_hotkey: int = 3
    source_add_max_per_day_per_hotkey: int = 5
    source_add_max_per_30d_per_hotkey: int = 10
    source_add_leg1_max_per_utc_day: int = 50

    arweave_audit_enabled: bool = True
    arweave_audit_shadow_enabled: bool = False

    @classmethod
    def from_env(cls) -> "ResearchLabGatewayConfig":
        prod_on = _prod_default(True)
        probe_timeout = min(
            120,
            max(5, _int("RESEARCH_LAB_SOURCE_ADD_PROBE_TIMEOUT_SECONDS", 45)),
        )
        work_lease = min(
            900,
            max(
                probe_timeout * 3 + 120,
                _int("RESEARCH_LAB_SOURCE_ADD_WORK_LEASE_SECONDS", 300),
            ),
        )
        return cls(
            api_enabled=_truthy("RESEARCH_LAB_GATEWAY_API_ENABLED", prod_on),
            production_writes_enabled=_truthy(
                "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED", prod_on
            ),
            reports_enabled=_truthy("RESEARCH_LAB_REPORTS_ENABLED", prod_on),
            shadow_bundles_enabled=_truthy(
                "RESEARCH_LAB_SHADOW_BUNDLES_ENABLED", prod_on
            ),
            shadow_reimbursements_enabled=_truthy(
                "RESEARCH_LAB_SHADOW_REIMBURSEMENTS_ENABLED"
            ),
            reimbursements_enabled=_truthy("RESEARCH_LAB_REIMBURSEMENTS_ENABLED"),
            weight_mutation_enabled=_truthy(
                "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED", prod_on
            ),
            fulfillment_mutation_enabled=_truthy(
                "RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED"
            ),
            internal_api_key=os.getenv("RESEARCH_LAB_INTERNAL_API_KEY", ""),
            evaluation_epoch=_int("RESEARCH_LAB_EVALUATION_EPOCH", 0),
            reimbursement_policy_id=os.getenv(
                "RESEARCH_LAB_REIMBURSEMENT_POLICY_ID",
                "alpha-reimbursement-production-v1",
            ),
            reimbursement_epochs=max(
                1,
                _int(
                    "RESEARCH_LAB_REIMBURSEMENT_EPOCHS",
                    _int("RESEARCH_LAB_REWARD_EPOCHS", 20),
                ),
            ),
            reimbursement_usd_per_0_1_percent_epoch=max(
                0.000001,
                _float("RESEARCH_LAB_REIMBURSEMENT_USD_PER_0_1_PERCENT_EPOCH", 0.162),
            ),
            reimbursement_dynamic_alpha_price_enabled=_truthy(
                "RESEARCH_LAB_REIMBURSEMENT_DYNAMIC_ALPHA_PRICE_ENABLED", "true"
            ),
            reimbursement_require_live_alpha_price=_truthy(
                "RESEARCH_LAB_REIMBURSEMENT_REQUIRE_LIVE_ALPHA_PRICE", "false"
            ),
            reimbursement_miner_alpha_per_epoch=max(
                0.000001,
                _float("RESEARCH_LAB_REIMBURSEMENT_MINER_ALPHA_PER_EPOCH", 147.6),
            ),
            lab_emission_percent=max(
                0.0,
                _float(
                    "RESEARCH_LAB_EMISSION_PERCENT",
                    float(DEFAULT_RESEARCH_LAB_EMISSION_PERCENT),
                ),
            ),
            fulfillment_emission_percent=max(
                0.0, _float("RESEARCH_LAB_FULFILLMENT_EMISSION_PERCENT", 60.5)
            ),
            fulfillment_leaderboard_emission_percent=max(
                0.0,
                _float("RESEARCH_LAB_FULFILLMENT_LEADERBOARD_EMISSION_PERCENT", 9.5),
            ),
            lab_reward_epochs=max(1, _int("RESEARCH_LAB_REWARD_EPOCHS", 20)),
            enable_conservative=_truthy("ENABLE_CONSERVATIVE", "true"),
            enable_champ_cap=_truthy("ENABLE_CHAMP_CAP", "true"),
            lab_reimbursement_max_cost_multiplier_with_champions=max(
                0.0,
                _float(
                    "RESEARCH_LAB_REIMBURSEMENT_MAX_COST_MULTIPLIER_WITH_CHAMPIONS",
                    float(DEFAULT_REIMBURSEMENT_MAX_COST_MULTIPLIER_WITH_CHAMPIONS),
                ),
            ),
            lab_reimbursement_min_alpha_percent=max(
                0.0,
                _float("RESEARCH_LAB_REIMBURSEMENT_MIN_ALPHA_PERCENT", 0.0),
            ),
            lab_champion_min_alpha_percent=max(
                0.0, _float("RESEARCH_LAB_CHAMPION_MIN_ALPHA_PERCENT", 7.0)
            ),
            lab_champion_extra_alpha_percent_per_point=max(
                0.0,
                _float("RESEARCH_LAB_CHAMPION_EXTRA_ALPHA_PERCENT_PER_POINT", 0.3),
            ),
            lab_champion_max_alpha_percent=max(
                0.0, _float("RESEARCH_LAB_CHAMPION_MAX_ALPHA_PERCENT", 15.0)
            ),
            lab_champion_placeholder_alpha_percent=max(
                0.0,
                _float("RESEARCH_LAB_CHAMPION_PLACEHOLDER_ALPHA_PERCENT", 0.0001),
            ),
            lab_champion_queue_trigger_ratio=min(
                1.0,
                max(
                    0.0,
                    _float(
                        "RESEARCH_LAB_CHAMPION_QUEUE_TRIGGER_RATIO",
                        float(DEFAULT_RESEARCH_LAB_CHAMPION_QUEUE_TRIGGER_RATIO),
                    ),
                ),
            ),
            lab_champion_threshold_points=max(
                0.0, _float("RESEARCH_LAB_CHAMPION_THRESHOLD_POINTS", 1.0)
            ),
            source_add_enabled=_truthy("RESEARCH_LAB_SOURCE_ADD_ENABLED", "true"),
            source_add_rewards_enabled=_truthy(
                "RESEARCH_LAB_SOURCE_ADD_REWARDS_ENABLED", "true"
            ),
            source_add_dispatcher_enabled=_truthy(
                "RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED", "true"
            ),
            source_add_functional_probes_enabled=_truthy(
                "RESEARCH_LAB_SOURCE_ADD_FUNCTIONAL_PROBES_ENABLED", "true"
            ),
            source_add_functional_rewards_enabled=_truthy(
                "RESEARCH_LAB_SOURCE_ADD_FUNCTIONAL_REWARDS_ENABLED", "true"
            ),
            source_add_dispatcher_poll_seconds=max(
                0.25, _float("RESEARCH_LAB_SOURCE_ADD_DISPATCHER_POLL_SECONDS", 2.0)
            ),
            source_add_work_lease_seconds=work_lease,
            source_add_probe_timeout_seconds=probe_timeout,
            source_add_probe_max_attempts=min(
                5,
                max(1, _int("RESEARCH_LAB_SOURCE_ADD_PROBE_MAX_ATTEMPTS", 5)),
            ),
            source_add_credential_kms_key_id=os.getenv(
                "RESEARCH_LAB_SOURCE_ADD_CREDENTIAL_KMS_KEY_ID", ""
            ),
            source_add_sandbox_image=os.getenv(
                "RESEARCH_LAB_SOURCE_ADD_SANDBOX_IMAGE", "python:3.11-slim"
            ),
            source_add_trial_timeout_seconds=max(
                30, _int("RESEARCH_LAB_SOURCE_ADD_TRIAL_TIMEOUT_SECONDS", 300)
            ),
            source_add_leg1_alpha_percent=max(
                0.0, _float("RESEARCH_LAB_SOURCE_ADD_LEG1_ALPHA_PERCENT", 0.2)
            ),
            source_add_leg2_alpha_percent=max(
                0.0, _float("RESEARCH_LAB_SOURCE_ADD_LEG2_ALPHA_PERCENT", 0.0)
            ),
            source_add_acceptance_floor_yield=max(
                0.0, _float("RESEARCH_LAB_SOURCE_ADD_ACCEPTANCE_FLOOR_YIELD", 0.10)
            ),
            source_add_max_concurrent_per_hotkey=max(
                1, _int("RESEARCH_LAB_SOURCE_ADD_MAX_CONCURRENT_PER_HOTKEY", 3)
            ),
            source_add_max_per_day_per_hotkey=max(
                1, _int("RESEARCH_LAB_SOURCE_ADD_MAX_PER_DAY_PER_HOTKEY", 5)
            ),
            source_add_max_per_30d_per_hotkey=max(
                1, _int("RESEARCH_LAB_SOURCE_ADD_MAX_PER_30D_PER_HOTKEY", 10)
            ),
            source_add_leg1_max_per_utc_day=max(
                1, _int("RESEARCH_LAB_SOURCE_ADD_LEG1_MAX_PER_UTC_DAY", 50)
            ),
            arweave_audit_enabled=_truthy(
                "RESEARCH_LAB_ARWEAVE_AUDIT_ENABLED", "true"
            ),
            arweave_audit_shadow_enabled=_truthy(
                "RESEARCH_LAB_ARWEAVE_AUDIT_SHADOW_ENABLED"
            ),
        )

    def reimbursement_policy_doc(
        self,
        *,
        enabled: bool | None = None,
    ) -> dict[str, object]:
        """Return the existing allocation policy for unsettled obligations."""

        return {
            "policy_id": self.reimbursement_policy_id,
            "enabled": self.reimbursements_enabled if enabled is None else bool(enabled),
            "reimbursement_epochs": self.reimbursement_epochs,
            "usd_per_0_1_percent_epoch": self.reimbursement_usd_per_0_1_percent_epoch,
            "dynamic_alpha_price_enabled": self.reimbursement_dynamic_alpha_price_enabled,
            "require_live_alpha_price": self.reimbursement_require_live_alpha_price,
            "miner_alpha_per_epoch": self.reimbursement_miner_alpha_per_epoch,
            "research_lab_emission_percent": self.lab_emission_percent,
            "fulfillment_emission_percent": self.fulfillment_emission_percent,
            "fulfillment_leaderboard_emission_percent": self.fulfillment_leaderboard_emission_percent,
            "reward_epochs": self.lab_reward_epochs,
            "enable_conservative": self.enable_conservative,
            "enable_champ_cap": self.enable_champ_cap,
            "reimbursement_allow_overpay_without_champions": False,
            "reimbursement_max_cost_multiplier_with_champions": (
                self.lab_reimbursement_max_cost_multiplier_with_champions
            ),
            "reimbursement_min_alpha_percent": self.lab_reimbursement_min_alpha_percent,
            "champion_min_alpha_percent": self.lab_champion_min_alpha_percent,
            "champion_extra_alpha_percent_per_point": (
                self.lab_champion_extra_alpha_percent_per_point
            ),
            "champion_max_alpha_percent": self.lab_champion_max_alpha_percent,
            "champion_placeholder_alpha_percent": (
                self.lab_champion_placeholder_alpha_percent
            ),
            "champion_queue_trigger_ratio": self.lab_champion_queue_trigger_ratio,
            "champion_threshold_points": self.lab_champion_threshold_points,
        }

    def public_status(self) -> dict[str, object]:
        return {
            "api_enabled": self.api_enabled,
            "production_writes_enabled": self.production_writes_enabled,
            # The retired loop intake stays closed during the first upgrade
            # from an older gateway. There is no setting that can reopen it.
            "miner_submissions_enabled": False,
            "source_add_enabled": self.source_add_enabled,
            "source_add": {
                "enabled": self.source_add_enabled,
                "rewards_enabled": self.source_add_rewards_enabled,
                "dispatcher_enabled": self.source_add_dispatcher_enabled,
                "functional_probes_enabled": self.source_add_functional_probes_enabled,
                "functional_rewards_enabled": self.source_add_functional_rewards_enabled,
                "leg1_alpha_percent": self.source_add_leg1_alpha_percent,
                "leg2_alpha_percent": self.source_add_leg2_alpha_percent,
                "reward_epochs": self.lab_reward_epochs,
                "max_concurrent_per_hotkey": self.source_add_max_concurrent_per_hotkey,
                "max_per_day_per_hotkey": self.source_add_max_per_day_per_hotkey,
                "max_per_30d_per_hotkey": self.source_add_max_per_30d_per_hotkey,
                "leg1_max_per_utc_day": self.source_add_leg1_max_per_utc_day,
            },
        }
