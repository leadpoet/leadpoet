"""Controlled production shadow mode for Leadpoet Research Lab.

This module computes production-like Research Lab bundles, reimbursement
schedules, and weight vectors in read-only shadow mode. It has no gateway,
Supabase, validator, Bittensor, fulfillment, or provider side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from leadpoet_verifier.economics import (
    build_improvement_grant_schedule,
    build_reimbursement_schedule,
    compose_final_weight_vector,
    compute_rebate_rate,
    compute_reimbursement_award,
    verify_improvement_grant_schedule,
    verify_reimbursement_schedule,
)

from .canonical import sha256_json


PRODUCTION_SHADOW_VERSION = "production-shadow-local-v0.1.0"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "production_shadow_fixtures.json"
TRUTHY_VALUES = {"1", "true", "yes", "on"}

SHADOW_FLAG_ENV_NAMES = (
    "RESEARCH_LAB_SHADOW_BUNDLES_ENABLED",
    "RESEARCH_LAB_SHADOW_WEIGHTS_ENABLED",
    "RESEARCH_LAB_SHADOW_REIMBURSEMENTS_ENABLED",
)

LIVE_MUTATION_FLAG_ENV_NAMES = (
    "RESEARCH_LAB_PAID_LOOPS_ENABLED",
    "RESEARCH_LAB_HOSTED_RUNS_ENABLED",
    "RESEARCH_LAB_PROBES_ENABLED",
    "RESEARCH_LAB_AUTOPILOT_ENABLED",
    "RESEARCH_LAB_CROWNING_ENABLED",
    "RESEARCH_LAB_REIMBURSEMENTS_ENABLED",
    "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED",
    "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED",
    "RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED",
)


@dataclass(frozen=True)
class ProductionShadowFlags:
    """Runtime flags for controlled production shadow mode.

    Defaults are intentionally inert. Shadow outputs require explicit shadow
    flags, and every live mutation flag must remain false.
    """

    shadow_bundles_enabled: bool = False
    shadow_weights_enabled: bool = False
    shadow_reimbursements_enabled: bool = False
    live_paid_loops_enabled: bool = False
    live_hosted_runs_enabled: bool = False
    live_probes_enabled: bool = False
    live_autopilot_enabled: bool = False
    live_crowning_enabled: bool = False
    live_reimbursements_enabled: bool = False
    live_weight_mutation_enabled: bool = False
    production_writes_enabled: bool = False
    fulfillment_mutation_enabled: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "ProductionShadowFlags":
        data = data or {}
        return cls(
            shadow_bundles_enabled=_flag_value(data, "shadow_bundles_enabled", "RESEARCH_LAB_SHADOW_BUNDLES_ENABLED"),
            shadow_weights_enabled=_flag_value(data, "shadow_weights_enabled", "RESEARCH_LAB_SHADOW_WEIGHTS_ENABLED"),
            shadow_reimbursements_enabled=_flag_value(
                data,
                "shadow_reimbursements_enabled",
                "RESEARCH_LAB_SHADOW_REIMBURSEMENTS_ENABLED",
            ),
            live_paid_loops_enabled=_flag_value(data, "live_paid_loops_enabled", "RESEARCH_LAB_PAID_LOOPS_ENABLED"),
            live_hosted_runs_enabled=_flag_value(data, "live_hosted_runs_enabled", "RESEARCH_LAB_HOSTED_RUNS_ENABLED"),
            live_probes_enabled=_flag_value(data, "live_probes_enabled", "RESEARCH_LAB_PROBES_ENABLED"),
            live_autopilot_enabled=_flag_value(data, "live_autopilot_enabled", "RESEARCH_LAB_AUTOPILOT_ENABLED"),
            live_crowning_enabled=_flag_value(data, "live_crowning_enabled", "RESEARCH_LAB_CROWNING_ENABLED"),
            live_reimbursements_enabled=_flag_value(
                data,
                "live_reimbursements_enabled",
                "RESEARCH_LAB_REIMBURSEMENTS_ENABLED",
            ),
            live_weight_mutation_enabled=_flag_value(
                data,
                "live_weight_mutation_enabled",
                "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED",
            ),
            production_writes_enabled=_flag_value(
                data,
                "production_writes_enabled",
                "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED",
            ),
            fulfillment_mutation_enabled=_flag_value(
                data,
                "fulfillment_mutation_enabled",
                "RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED",
            ),
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ProductionShadowFlags":
        return cls.from_mapping(env or {})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    def live_mutation_flags(self) -> dict[str, bool]:
        return {
            "RESEARCH_LAB_PAID_LOOPS_ENABLED": self.live_paid_loops_enabled,
            "RESEARCH_LAB_HOSTED_RUNS_ENABLED": self.live_hosted_runs_enabled,
            "RESEARCH_LAB_PROBES_ENABLED": self.live_probes_enabled,
            "RESEARCH_LAB_AUTOPILOT_ENABLED": self.live_autopilot_enabled,
            "RESEARCH_LAB_CROWNING_ENABLED": self.live_crowning_enabled,
            "RESEARCH_LAB_REIMBURSEMENTS_ENABLED": self.live_reimbursements_enabled,
            "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": self.live_weight_mutation_enabled,
            "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED": self.production_writes_enabled,
            "RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED": self.fulfillment_mutation_enabled,
        }

    def shadow_flags(self) -> dict[str, bool]:
        return {
            "RESEARCH_LAB_SHADOW_BUNDLES_ENABLED": self.shadow_bundles_enabled,
            "RESEARCH_LAB_SHADOW_WEIGHTS_ENABLED": self.shadow_weights_enabled,
            "RESEARCH_LAB_SHADOW_REIMBURSEMENTS_ENABLED": self.shadow_reimbursements_enabled,
        }


def validate_shadow_flags(flags: ProductionShadowFlags | Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(flags, ProductionShadowFlags):
        flags = ProductionShadowFlags.from_mapping(flags)

    errors: list[str] = []
    disabled_shadow_flags = [name for name, enabled in flags.shadow_flags().items() if not enabled]
    enabled_live_flags = [name for name, enabled in flags.live_mutation_flags().items() if enabled]

    if disabled_shadow_flags:
        errors.append("shadow_flags_not_fully_enabled")
    if enabled_live_flags:
        errors.append("live_mutation_flags_enabled")

    return {
        "passed": not errors,
        "errors": errors,
        "disabled_shadow_flags": disabled_shadow_flags,
        "enabled_live_mutation_flags": enabled_live_flags,
        "shadow_flags": flags.shadow_flags(),
        "live_mutation_flags": flags.live_mutation_flags(),
    }


def build_controlled_production_shadow(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build a read-only production-shadow bundle and operator report."""
    flags = ProductionShadowFlags.from_mapping(record.get("flags", {}))
    flag_check = validate_shadow_flags(flags)
    if not flag_check["passed"]:
        raise ValueError(f"unsafe production shadow flags: {flag_check}")

    epoch = int(record["epoch"])
    netuid = int(record.get("netuid", 71))
    reimbursement_policy = record["reimbursement_policy"]
    reimbursement_start_epoch = int(record.get("reimbursement_start_epoch", epoch))
    miner_uid_by_hotkey = {str(k): int(v) for k, v in record.get("miner_uid_by_hotkey", {}).items()}
    participation_snapshots = {
        str(island): dict(snapshot) for island, snapshot in record.get("participation_snapshots", {}).items()
    }

    reimbursement_awards: list[dict[str, Any]] = []
    reimbursement_schedules: list[dict[str, Any]] = []
    reimbursement_verifications: list[dict[str, Any]] = []
    for run in record.get("reimbursement_runs", []):
        island = str(run["island"])
        snapshot = participation_snapshots[island]
        cap_usage = record.get("reimbursement_cap_usage_by_run_id", {}).get(str(run["run_id"]), {})
        award = compute_reimbursement_award(run, snapshot, reimbursement_policy, cap_usage)
        schedule = build_reimbursement_schedule(award, start_epoch=reimbursement_start_epoch)
        schedule = {
            **schedule,
            "uid": miner_uid_by_hotkey.get(str(award["miner_hotkey"]), int(run.get("miner_uid", -1))),
            "shadow_only": True,
            "submission_allowed": False,
        }
        reimbursement_awards.append({**award, "shadow_only": True, "submission_allowed": False})
        reimbursement_schedules.append(schedule)
        reimbursement_verifications.append(verify_reimbursement_schedule(award, schedule))

    grant_policy = record.get("grant_policy", {})
    improvement_grant_schedules: list[dict[str, Any]] = []
    grant_verifications: list[dict[str, Any]] = []
    for crown in record.get("crowns", []):
        schedule = build_improvement_grant_schedule(crown, grant_policy)
        improvement_grant_schedules.append({**schedule, "shadow_only": True, "submission_allowed": False})
        grant_verifications.append(verify_improvement_grant_schedule(crown, grant_policy, schedule))

    weight_vector = compose_final_weight_vector(
        epoch=epoch,
        uids=record.get("uids", []),
        fulfillment_scores=record.get("fulfillment_scores", {}),
        leaderboard_scores=record.get("leaderboard_scores", {}),
        improvement_grant_schedules=improvement_grant_schedules,
        reimbursement_schedules=reimbursement_schedules,
        active_researcher_floor_scores=record.get("active_researcher_floor_scores", {}),
        policy=record.get("weight_policy", {}),
    )

    verifier_divergence = compute_verifier_divergence(
        weight_vector,
        official_weight_vector=record.get("official_shadow_weight_vector"),
        official_weight_hash=record.get("official_shadow_weight_hash"),
    )
    live_diff = compute_shadow_live_diff(weight_vector["u16_weights"], record.get("live_u16_weights", {}))
    report = build_shadow_observability_report(
        epoch=epoch,
        flags=flags,
        participation_snapshots=participation_snapshots,
        reimbursement_policy=reimbursement_policy,
        reimbursement_awards=reimbursement_awards,
        reimbursement_schedules=reimbursement_schedules,
        live_diff=live_diff,
        verifier_divergence=verifier_divergence,
    )

    bundle_without_id = {
        "bundle_id": "",
        "schema_version": "1.0",
        "shadow_version": PRODUCTION_SHADOW_VERSION,
        "mode": "controlled_production_shadow",
        "shadow_only": True,
        "read_only": True,
        "submission_allowed": False,
        "on_chain_submission_allowed": False,
        "netuid": netuid,
        "epoch": epoch,
        "generated_at": str(record["generated_at"]),
        "source_refs": list(record.get("source_refs", [])),
        "flag_check": flag_check,
        "state_input_hash": sha256_json(
            {
                "epoch": epoch,
                "netuid": netuid,
                "source_refs": list(record.get("source_refs", [])),
                "reimbursement_runs": list(record.get("reimbursement_runs", [])),
                "participation_snapshots": participation_snapshots,
                "crowns": list(record.get("crowns", [])),
                "weight_inputs": {
                    "uids": list(record.get("uids", [])),
                    "fulfillment_scores": dict(record.get("fulfillment_scores", {})),
                    "leaderboard_scores": dict(record.get("leaderboard_scores", {})),
                    "active_researcher_floor_scores": dict(record.get("active_researcher_floor_scores", {})),
                },
            }
        ),
        "reimbursement_awards": reimbursement_awards,
        "reimbursement_schedules": reimbursement_schedules,
        "improvement_grant_schedules": improvement_grant_schedules,
        "weight_vector": weight_vector,
        "shadow_live_diff": live_diff,
        "verifier_divergence": verifier_divergence,
        "report": report,
        "mutation_guards": {
            "live_mutation_flags": flags.live_mutation_flags(),
            "forbidden_runtime_actions": [
                "submit_bittensor_weights",
                "consume_loop_balance",
                "create_live_reimbursement_award",
                "create_live_crown",
                "write_fulfillment_tables",
                "write_production_supabase",
            ],
        },
    }
    bundle = {**bundle_without_id, "bundle_id": "research_shadow_bundle:" + sha256_json(bundle_without_id)}
    return {
        "bundle": bundle,
        "report": report,
        "read_only_evidence": assert_shadow_output_read_only(bundle),
        "reimbursement_verifications": reimbursement_verifications,
        "grant_verifications": grant_verifications,
    }


def build_shadow_observability_report(
    *,
    epoch: int,
    flags: ProductionShadowFlags,
    participation_snapshots: Mapping[str, Mapping[str, Any]],
    reimbursement_policy: Mapping[str, Any],
    reimbursement_awards: Sequence[Mapping[str, Any]],
    reimbursement_schedules: Sequence[Mapping[str, Any]],
    live_diff: Mapping[str, Any],
    verifier_divergence: Mapping[str, Any],
) -> dict[str, Any]:
    participation_by_island = {}
    for island, snapshot in sorted(participation_snapshots.items()):
        rebate_rate = compute_rebate_rate(snapshot, reimbursement_policy)
        participation_score = (
            float(snapshot.get("distinct_funded_hotkeys", 0)) * float(reimbursement_policy.get("distinct_funded_hotkey_weight", 1))
            + float(snapshot.get("paid_loop_count", 0)) * float(reimbursement_policy.get("paid_loop_weight", 1))
            + float(snapshot.get("unique_brief_count", 0)) * float(reimbursement_policy.get("unique_brief_weight", 1))
        )
        high_target = float(reimbursement_policy["high_participation_target"])
        participation_by_island[island] = {
            "participation_score": round(participation_score, 6),
            "participation_fraction": round(min(1.0, max(0.0, participation_score / high_target)), 6),
            "rebate_rate": float(rebate_rate),
            "distinct_funded_hotkeys": int(snapshot.get("distinct_funded_hotkeys", 0)),
            "paid_loop_count": int(snapshot.get("paid_loop_count", 0)),
            "unique_brief_count": int(snapshot.get("unique_brief_count", 0)),
        }

    reimbursement_by_epoch: dict[str, int] = {}
    for schedule in reimbursement_schedules:
        for entry in schedule.get("entries", []):
            epoch_key = str(entry["epoch"])
            reimbursement_by_epoch[epoch_key] = reimbursement_by_epoch.get(epoch_key, 0) + int(entry["amount_microusd"])

    report_without_id = {
        "report_id": "",
        "schema_version": "1.0",
        "shadow_version": PRODUCTION_SHADOW_VERSION,
        "epoch": int(epoch),
        "shadow_only": True,
        "read_only": True,
        "submission_allowed": False,
        "live_mutation_flags": flags.live_mutation_flags(),
        "participation_by_island": participation_by_island,
        "reimbursement_award_count": len(reimbursement_awards),
        "awarded_reimbursement_count": sum(1 for award in reimbursement_awards if award.get("status") == "awarded"),
        "scheduled_reimbursement_microusd_by_epoch": dict(sorted(reimbursement_by_epoch.items(), key=lambda item: int(item[0]))),
        "shadow_live_diff": dict(live_diff),
        "verifier_divergence": dict(verifier_divergence),
    }
    return {**report_without_id, "report_id": "research_shadow_report:" + sha256_json(report_without_id)}


def compute_shadow_live_diff(
    shadow_u16_weights: Mapping[str, Any],
    live_u16_weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    live_u16_weights = live_u16_weights or {}
    shadow = {str(uid): int(weight) for uid, weight in shadow_u16_weights.items()}
    live = {str(uid): int(weight) for uid, weight in live_u16_weights.items()}
    all_uids = sorted(set(shadow) | set(live), key=lambda uid: int(uid))
    by_uid = {}
    for uid in all_uids:
        shadow_weight = shadow.get(uid, 0)
        live_weight = live.get(uid, 0)
        by_uid[uid] = {
            "shadow_u16": shadow_weight,
            "live_u16": live_weight,
            "delta_u16": shadow_weight - live_weight,
            "abs_delta_u16": abs(shadow_weight - live_weight),
        }
    return {
        "uid_count": len(all_uids),
        "changed_uid_count": sum(1 for row in by_uid.values() if row["delta_u16"] != 0),
        "total_abs_delta_u16": sum(row["abs_delta_u16"] for row in by_uid.values()),
        "max_abs_delta_u16": max((row["abs_delta_u16"] for row in by_uid.values()), default=0),
        "by_uid": by_uid,
    }


def compute_verifier_divergence(
    recomputed_weight_vector: Mapping[str, Any],
    *,
    official_weight_vector: Mapping[str, Any] | None = None,
    official_weight_hash: str | None = None,
) -> dict[str, Any]:
    recomputed_hash = sha256_json(recomputed_weight_vector)
    if official_weight_vector is None and not official_weight_hash:
        return {
            "status": "no_official_shadow_bundle_provided",
            "diverged": False,
            "recomputed_weight_hash": recomputed_hash,
            "official_weight_hash": None,
        }

    official_hash = official_weight_hash or sha256_json(official_weight_vector)
    return {
        "status": "match" if official_hash == recomputed_hash else "diverged",
        "diverged": official_hash != recomputed_hash,
        "recomputed_weight_hash": recomputed_hash,
        "official_weight_hash": official_hash,
    }


def assert_shadow_output_read_only(bundle: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    if not bool(bundle.get("shadow_only", False)):
        errors.append("bundle_not_marked_shadow_only")
    if not bool(bundle.get("read_only", False)):
        errors.append("bundle_not_marked_read_only")
    if bool(bundle.get("submission_allowed", True)):
        errors.append("submission_allowed_true")
    if bool(bundle.get("on_chain_submission_allowed", True)):
        errors.append("on_chain_submission_allowed_true")
    enabled_live_flags = [
        name
        for name, enabled in bundle.get("mutation_guards", {}).get("live_mutation_flags", {}).items()
        if bool(enabled)
    ]
    if enabled_live_flags:
        errors.append("live_mutation_flags_enabled")

    return {
        "passed": not errors,
        "errors": errors,
        "shadow_only": bool(bundle.get("shadow_only", False)),
        "read_only": bool(bundle.get("read_only", False)),
        "submission_allowed": bool(bundle.get("submission_allowed", True)),
        "on_chain_submission_allowed": bool(bundle.get("on_chain_submission_allowed", True)),
        "enabled_live_mutation_flags": enabled_live_flags,
    }


def build_on_chain_submission_payload(_bundle: Mapping[str, Any]) -> dict[str, Any]:
    raise RuntimeError("controlled production shadow outputs are read-only and cannot build on-chain submissions")


def load_production_shadow_fixture(path: Path | str | None = None) -> dict[str, Any]:
    fixture_path = Path(path) if path else FIXTURE_PATH
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_controlled_production_shadow(path: Path | str | None = None) -> dict[str, Any]:
    fixture = load_production_shadow_fixture(path)
    result = build_controlled_production_shadow(fixture["shadow_case"])
    bundle = result["bundle"]
    report = result["report"]
    expectations = fixture.get("expectations", {})

    errors = []
    if ProductionShadowFlags().to_dict() != expectations.get("default_flags", ProductionShadowFlags().to_dict()):
        errors.append("default_flags_do_not_match_expected_inert_defaults")
    if not result["read_only_evidence"]["passed"]:
        errors.extend(result["read_only_evidence"]["errors"])
    if any(not item["passed"] for item in result["reimbursement_verifications"]):
        errors.append("reimbursement_schedule_verification_failed")
    if any(not item["passed"] for item in result["grant_verifications"]):
        errors.append("grant_schedule_verification_failed")
    if report["awarded_reimbursement_count"] != expectations.get("awarded_reimbursement_count"):
        errors.append("awarded_reimbursement_count_mismatch")
    if report["scheduled_reimbursement_microusd_by_epoch"] != expectations.get("scheduled_reimbursement_microusd_by_epoch"):
        errors.append("scheduled_reimbursement_totals_mismatch")
    if report["verifier_divergence"]["status"] != expectations.get("verifier_divergence_status"):
        errors.append("verifier_divergence_status_mismatch")
    if report["shadow_live_diff"]["changed_uid_count"] != expectations.get("changed_uid_count"):
        errors.append("shadow_live_changed_uid_count_mismatch")

    tampered = {**bundle["weight_vector"], "weight_sum": int(bundle["weight_vector"]["weight_sum"]) + 1}
    divergence = compute_verifier_divergence(bundle["weight_vector"], official_weight_vector=tampered)
    if not divergence["diverged"] or divergence["status"] != "diverged":
        errors.append("tampered_official_weight_vector_did_not_diverge")

    try:
        build_on_chain_submission_payload(bundle)
        errors.append("shadow_bundle_built_on_chain_payload")
    except RuntimeError:
        pass

    if errors:
        raise AssertionError("; ".join(errors))

    return {
        "bundle_id": bundle["bundle_id"],
        "report_id": report["report_id"],
        "epoch": bundle["epoch"],
        "awarded_reimbursement_count": report["awarded_reimbursement_count"],
        "scheduled_reimbursement_microusd_by_epoch": report["scheduled_reimbursement_microusd_by_epoch"],
        "changed_uid_count": report["shadow_live_diff"]["changed_uid_count"],
        "max_abs_delta_u16": report["shadow_live_diff"]["max_abs_delta_u16"],
        "verifier_divergence_status": report["verifier_divergence"]["status"],
        "read_only": result["read_only_evidence"]["passed"],
    }


def _flag_value(data: Mapping[str, Any], canonical_name: str, env_name: str) -> bool:
    if canonical_name in data:
        return _truthy(data[canonical_name])
    return _truthy(data.get(env_name, False))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUTHY_VALUES
