"""Local alpha reimbursement kernel for Leadpoet Research Lab.

This module is intentionally local-only. It does not call gateways, Supabase,
validators, chain APIs, or provider APIs. Its job is to make the alpha
participation rebate deterministic enough to graduate into open-verifier
fixtures later.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import sha256_json
from .schema_validation import validate_schema_record


SCHEMA_VERSION = "1.0"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "reimbursement_fixtures.json"
MICRO_USD = Decimal("1000000")
RATE_QUANT = Decimal("0.000001")


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def usd_to_microusd(value: Any) -> int:
    return int((_decimal(value) * MICRO_USD).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def microusd_to_usd(value: int) -> float:
    return round(int(value) / 1_000_000, 6)


def _rate_to_float(value: Decimal) -> float:
    return float(value.quantize(RATE_QUANT, rounding=ROUND_HALF_UP))


def _clamp_decimal(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    return max(low, min(high, value))


@dataclass(frozen=True)
class IslandParticipationSnapshot:
    snapshot_id: str
    island: str
    lookback_start: str
    lookback_end: str
    distinct_funded_hotkeys: int
    paid_loop_count: int
    unique_brief_count: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "IslandParticipationSnapshot":
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            island=str(data["island"]),
            lookback_start=str(data["lookback_start"]),
            lookback_end=str(data["lookback_end"]),
            distinct_funded_hotkeys=int(data["distinct_funded_hotkeys"]),
            paid_loop_count=int(data["paid_loop_count"]),
            unique_brief_count=int(data["unique_brief_count"]),
        )


@dataclass(frozen=True)
class ResearchRunCostRecord:
    run_id: str
    miner_hotkey: str
    island: str
    run_day: str
    verified_compute_cost_microusd: int
    miner_openrouter_cost_microusd: int
    loop_start_tao_fee_microusd: int
    paid_research_loop: bool = True
    valid_receipt: bool = True
    verified_loop_start_payment: bool = True
    preserved_loop_start_credit: bool = False
    miner_openrouter_key_present: bool = True
    trusted_cost_ledger: bool = True
    passed_abuse_checks: bool = True
    refunded: bool = False
    voided: bool = False
    duplicate: bool = False
    novelty_rejected: bool = False
    self_cancelled_before_minimum_work: bool = False
    banned_hotkey: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchRunCostRecord":
        return cls(
            run_id=str(data["run_id"]),
            miner_hotkey=str(data["miner_hotkey"]),
            island=str(data["island"]),
            run_day=str(data["run_day"]),
            verified_compute_cost_microusd=usd_to_microusd(
                data.get("funded_compute_budget_usd", data.get("verified_compute_cost_usd", 0))
            ),
            miner_openrouter_cost_microusd=usd_to_microusd(
                data.get("actual_openrouter_cost_usd", data.get("miner_openrouter_cost_usd", 0))
            ),
            loop_start_tao_fee_microusd=usd_to_microusd(data.get("loop_start_tao_fee_usd", 0)),
            paid_research_loop=bool(data.get("paid_research_loop", True)),
            valid_receipt=bool(data.get("valid_receipt", True)),
            verified_loop_start_payment=bool(data.get("verified_loop_start_payment", True)),
            preserved_loop_start_credit=bool(data.get("preserved_loop_start_credit", False)),
            miner_openrouter_key_present=bool(data.get("miner_openrouter_key_present", True)),
            trusted_cost_ledger=bool(data.get("trusted_cost_ledger", True)),
            passed_abuse_checks=bool(data.get("passed_abuse_checks", True)),
            refunded=bool(data.get("refunded", False)),
            voided=bool(data.get("voided", False)),
            duplicate=bool(data.get("duplicate", False)),
            novelty_rejected=bool(data.get("novelty_rejected", False)),
            self_cancelled_before_minimum_work=bool(data.get("self_cancelled_before_minimum_work", False)),
            banned_hotkey=bool(data.get("banned_hotkey", False)),
        )


@dataclass(frozen=True)
class ReimbursementPolicy:
    policy_id: str
    enabled: bool
    min_rebate_rate: Decimal
    max_rebate_rate: Decimal
    base_rebate_rate: Decimal
    high_participation_target: Decimal
    reimbursement_epochs: int
    max_usd_per_run: Decimal
    max_usd_per_hotkey_day: Decimal
    max_usd_per_island_day: Decimal
    global_budget_usd: Decimal
    include_loop_start_fee_in_base: bool = False
    material_spend_ratio: Decimal | None = None
    default_island: str = "generalist"
    usd_per_0_1_percent_epoch: Decimal | None = None
    distinct_funded_hotkey_weight: Decimal = Decimal("1")
    paid_loop_weight: Decimal = Decimal("1")
    unique_brief_weight: Decimal = Decimal("1")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReimbursementPolicy":
        return cls(
            policy_id=str(data["policy_id"]),
            enabled=bool(data.get("enabled", False)),
            min_rebate_rate=_decimal(data["min_rebate_rate"]),
            max_rebate_rate=_decimal(data["max_rebate_rate"]),
            base_rebate_rate=_decimal(data.get("base_rebate_rate", data["max_rebate_rate"])),
            high_participation_target=_decimal(data["high_participation_target"]),
            reimbursement_epochs=int(data["reimbursement_epochs"]),
            max_usd_per_run=_decimal(data["max_usd_per_run"]),
            max_usd_per_hotkey_day=_decimal(data["max_usd_per_hotkey_day"]),
            max_usd_per_island_day=_decimal(data["max_usd_per_island_day"]),
            global_budget_usd=_decimal(data["global_budget_usd"]),
            include_loop_start_fee_in_base=bool(data.get("include_loop_start_fee_in_base", False)),
            material_spend_ratio=(
                _decimal(data["material_spend_ratio"]) if "material_spend_ratio" in data else None
            ),
            default_island=str(data.get("default_island") or "generalist"),
            usd_per_0_1_percent_epoch=(
                _decimal(data["usd_per_0_1_percent_epoch"]) if "usd_per_0_1_percent_epoch" in data else None
            ),
            distinct_funded_hotkey_weight=_decimal(data.get("distinct_funded_hotkey_weight", 1)),
            paid_loop_weight=_decimal(data.get("paid_loop_weight", 1)),
            unique_brief_weight=_decimal(data.get("unique_brief_weight", 1)),
        )


@dataclass(frozen=True)
class ReimbursementCapUsage:
    hotkey_day_awarded_microusd: int = 0
    island_day_awarded_microusd: int = 0
    global_awarded_microusd: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "ReimbursementCapUsage":
        data = data or {}
        return cls(
            hotkey_day_awarded_microusd=usd_to_microusd(data.get("hotkey_day_awarded_usd", 0)),
            island_day_awarded_microusd=usd_to_microusd(data.get("island_day_awarded_usd", 0)),
            global_awarded_microusd=usd_to_microusd(data.get("global_awarded_usd", 0)),
        )


@dataclass(frozen=True)
class ReimbursementAward:
    award_id: str
    run_id: str
    miner_hotkey: str
    island: str
    run_day: str
    status: str
    participation_score: Decimal
    participation_fraction: Decimal
    rebate_rate: Decimal
    eligible_cost_microusd: int
    capped_cost_basis_microusd: int
    raw_reimbursement_microusd: int
    target_reimbursement_microusd: int
    reimbursement_epochs: int
    caps_applied: tuple[str, ...]
    ineligibility_reasons: tuple[str, ...]
    loop_start_fee_included: bool
    input_hash: str
    usd_per_0_1_percent_epoch: Decimal | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "award_id": self.award_id,
            "run_id": self.run_id,
            "miner_hotkey": self.miner_hotkey,
            "island": self.island,
            "run_day": self.run_day,
            "status": self.status,
            "participation_score": _rate_to_float(self.participation_score),
            "participation_fraction": _rate_to_float(self.participation_fraction),
            "rebate_rate": _rate_to_float(self.rebate_rate),
            "eligible_cost_microusd": self.eligible_cost_microusd,
            "eligible_cost_usd": microusd_to_usd(self.eligible_cost_microusd),
            "capped_cost_basis_microusd": self.capped_cost_basis_microusd,
            "capped_cost_basis_usd": microusd_to_usd(self.capped_cost_basis_microusd),
            "raw_reimbursement_microusd": self.raw_reimbursement_microusd,
            "raw_reimbursement_usd": microusd_to_usd(self.raw_reimbursement_microusd),
            "target_reimbursement_microusd": self.target_reimbursement_microusd,
            "target_reimbursement_usd": microusd_to_usd(self.target_reimbursement_microusd),
            "reimbursement_epochs": self.reimbursement_epochs,
            "caps_applied": list(self.caps_applied),
            "ineligibility_reasons": list(self.ineligibility_reasons),
            "loop_start_fee_included": self.loop_start_fee_included,
            "input_hash": self.input_hash,
        }
        if self.usd_per_0_1_percent_epoch is not None:
            payload["usd_per_0_1_percent_epoch"] = microusd_to_usd(
                usd_to_microusd(self.usd_per_0_1_percent_epoch)
            )
        return payload


@dataclass(frozen=True)
class ReimbursementSchedule:
    schedule_id: str
    award_id: str
    status: str
    start_epoch: int
    epoch_count: int
    total_microusd: int
    entries: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "award_id": self.award_id,
            "status": self.status,
            "start_epoch": self.start_epoch,
            "epoch_count": self.epoch_count,
            "total_microusd": self.total_microusd,
            "total_usd": microusd_to_usd(self.total_microusd),
            "entries": list(self.entries),
        }


def compute_participation_score(
    snapshot: IslandParticipationSnapshot | Mapping[str, Any],
    policy: ReimbursementPolicy | Mapping[str, Any],
) -> Decimal:
    if not isinstance(snapshot, IslandParticipationSnapshot):
        snapshot = IslandParticipationSnapshot.from_mapping(snapshot)
    if not isinstance(policy, ReimbursementPolicy):
        policy = ReimbursementPolicy.from_mapping(policy)
    return (
        Decimal(snapshot.distinct_funded_hotkeys) * policy.distinct_funded_hotkey_weight
        + Decimal(snapshot.paid_loop_count) * policy.paid_loop_weight
        + Decimal(snapshot.unique_brief_count) * policy.unique_brief_weight
    )


def compute_rebate_rate(
    snapshot: IslandParticipationSnapshot | Mapping[str, Any],
    policy: ReimbursementPolicy | Mapping[str, Any],
) -> Decimal:
    if not isinstance(snapshot, IslandParticipationSnapshot):
        snapshot = IslandParticipationSnapshot.from_mapping(snapshot)
    if not isinstance(policy, ReimbursementPolicy):
        policy = ReimbursementPolicy.from_mapping(policy)

    errors = validate_reimbursement_policy(policy)
    if errors:
        raise ValueError("; ".join(errors))

    score = compute_participation_score(snapshot, policy)
    fraction = _clamp_decimal(score / policy.high_participation_target, Decimal("0"), Decimal("1"))
    rate_ceiling = policy.base_rebate_rate if snapshot.island == policy.default_island else policy.max_rebate_rate
    rate = rate_ceiling - fraction * (rate_ceiling - policy.min_rebate_rate)
    return rate.quantize(RATE_QUANT, rounding=ROUND_HALF_UP)


def compute_reimbursement_award(
    run: ResearchRunCostRecord | Mapping[str, Any],
    snapshot: IslandParticipationSnapshot | Mapping[str, Any],
    policy: ReimbursementPolicy | Mapping[str, Any],
    cap_usage: ReimbursementCapUsage | Mapping[str, Any] | None = None,
) -> ReimbursementAward:
    if not isinstance(run, ResearchRunCostRecord):
        run = ResearchRunCostRecord.from_mapping(run)
    if not isinstance(snapshot, IslandParticipationSnapshot):
        snapshot = IslandParticipationSnapshot.from_mapping(snapshot)
    if not isinstance(policy, ReimbursementPolicy):
        policy = ReimbursementPolicy.from_mapping(policy)
    if not isinstance(cap_usage, ReimbursementCapUsage):
        cap_usage = ReimbursementCapUsage.from_mapping(cap_usage)

    policy_errors = validate_reimbursement_policy(policy)
    if policy_errors:
        raise ValueError("; ".join(policy_errors))

    score = compute_participation_score(snapshot, policy)
    fraction = _clamp_decimal(score / policy.high_participation_target, Decimal("0"), Decimal("1"))
    rebate_rate = compute_rebate_rate(snapshot, policy)
    eligibility_reasons = _run_ineligibility_reasons(run)

    eligible_cost = _eligible_reimbursement_cost(run, policy)
    if policy.include_loop_start_fee_in_base:
        eligible_cost += run.loop_start_tao_fee_microusd
    if eligible_cost <= 0:
        eligibility_reasons.append("nonpositive_eligible_cost")

    input_hash = _award_input_hash(run, snapshot, policy, cap_usage)
    if not policy.enabled:
        return _zero_award(
            run=run,
            policy=policy,
            score=score,
            fraction=fraction,
            rebate_rate=rebate_rate,
            eligible_cost=eligible_cost,
            status="disabled",
            reasons=("policy_disabled",),
            input_hash=input_hash,
        )
    if eligibility_reasons:
        return _zero_award(
            run=run,
            policy=policy,
            score=score,
            fraction=fraction,
            rebate_rate=rebate_rate,
            eligible_cost=eligible_cost,
            status="ineligible",
            reasons=tuple(eligibility_reasons),
            input_hash=input_hash,
        )

    caps_applied: list[str] = []
    max_run_cost = usd_to_microusd(policy.max_usd_per_run)
    capped_cost_basis = min(eligible_cost, max_run_cost)
    if capped_cost_basis < eligible_cost:
        caps_applied.append("per_run")

    raw_target = _round_microusd(Decimal(capped_cost_basis) * rebate_rate)
    target = raw_target

    for cap_name, remaining in _remaining_caps(policy, cap_usage):
        if target > remaining:
            target = max(0, remaining)
            caps_applied.append(cap_name)

    status = "awarded" if target > 0 else "capped_to_zero"
    award = ReimbursementAward(
        award_id="",
        run_id=run.run_id,
        miner_hotkey=run.miner_hotkey,
        island=run.island,
        run_day=run.run_day,
        status=status,
        participation_score=score,
        participation_fraction=fraction,
        rebate_rate=rebate_rate,
        eligible_cost_microusd=eligible_cost,
        capped_cost_basis_microusd=capped_cost_basis,
        raw_reimbursement_microusd=raw_target,
        target_reimbursement_microusd=target,
        reimbursement_epochs=policy.reimbursement_epochs,
        caps_applied=tuple(caps_applied),
        ineligibility_reasons=(),
        loop_start_fee_included=policy.include_loop_start_fee_in_base,
        input_hash=input_hash,
        usd_per_0_1_percent_epoch=policy.usd_per_0_1_percent_epoch,
    )
    return _with_award_id(award)


def build_reimbursement_schedule(
    award: ReimbursementAward | Mapping[str, Any],
    *,
    start_epoch: int = 0,
) -> ReimbursementSchedule:
    if not isinstance(award, ReimbursementAward):
        award = _award_from_mapping(award)
    if start_epoch < 0:
        raise ValueError("start_epoch must be non-negative")

    total = award.target_reimbursement_microusd
    if award.status != "awarded" or total <= 0:
        return ReimbursementSchedule(
            schedule_id=f"reimbursement_schedule:{award.award_id}",
            award_id=award.award_id,
            status="empty",
            start_epoch=start_epoch,
            epoch_count=0,
            total_microusd=0,
            entries=(),
        )

    if award.reimbursement_epochs <= 0:
        raise ValueError("reimbursement_epochs must be positive for awarded reimbursements")

    base = total // award.reimbursement_epochs
    remainder = total % award.reimbursement_epochs
    entries: list[dict[str, Any]] = []
    for idx in range(award.reimbursement_epochs):
        amount = base + (1 if idx < remainder else 0)
        entry = {
            "epoch": start_epoch + idx,
            "amount_microusd": amount,
            "amount_usd": microusd_to_usd(amount),
        }
        alpha_fields = _alpha_entry_fields(amount, award.usd_per_0_1_percent_epoch)
        if alpha_fields:
            entry.update(alpha_fields)
        entries.append(entry)
    return ReimbursementSchedule(
        schedule_id=f"reimbursement_schedule:{award.award_id}",
        award_id=award.award_id,
        status="scheduled",
        start_epoch=start_epoch,
        epoch_count=award.reimbursement_epochs,
        total_microusd=total,
        entries=tuple(entries),
    )


def build_reimbursement_decision(
    award: ReimbursementAward,
    schedule: ReimbursementSchedule,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "award": award.to_dict(),
        "schedule": schedule.to_dict(),
    }


def validate_reimbursement_policy(policy: ReimbursementPolicy | Mapping[str, Any]) -> list[str]:
    if not isinstance(policy, ReimbursementPolicy):
        policy = ReimbursementPolicy.from_mapping(policy)
    errors: list[str] = []
    if policy.min_rebate_rate < 0 or policy.max_rebate_rate < 0:
        errors.append("rebate rates must be non-negative")
    if policy.base_rebate_rate < 0:
        errors.append("base_rebate_rate must be non-negative")
    if policy.min_rebate_rate > policy.max_rebate_rate:
        errors.append("min_rebate_rate must be <= max_rebate_rate")
    if policy.base_rebate_rate < policy.min_rebate_rate or policy.base_rebate_rate > policy.max_rebate_rate:
        errors.append("base_rebate_rate must satisfy min <= base <= max")
    if policy.max_rebate_rate > 1:
        errors.append("max_rebate_rate must be <= 1")
    if policy.material_spend_ratio is not None and (
        policy.material_spend_ratio < 0 or policy.material_spend_ratio > 1
    ):
        errors.append("material_spend_ratio must be between 0 and 1")
    if policy.usd_per_0_1_percent_epoch is not None and policy.usd_per_0_1_percent_epoch <= 0:
        errors.append("usd_per_0_1_percent_epoch must be positive")
    if policy.high_participation_target <= 0:
        errors.append("high_participation_target must be positive")
    if policy.reimbursement_epochs <= 0:
        errors.append("reimbursement_epochs must be positive")
    for field in (
        "max_usd_per_run",
        "max_usd_per_hotkey_day",
        "max_usd_per_island_day",
        "global_budget_usd",
    ):
        if getattr(policy, field) < 0:
            errors.append(f"{field} must be non-negative")
    return errors


def validate_reimbursement_award(
    award: ReimbursementAward | Mapping[str, Any],
    schedule: ReimbursementSchedule | Mapping[str, Any] | None = None,
) -> list[str]:
    if not isinstance(award, ReimbursementAward):
        award = _award_from_mapping(award)
    if schedule is not None and not isinstance(schedule, ReimbursementSchedule):
        schedule = _schedule_from_mapping(schedule)

    errors: list[str] = []
    if award.status not in {"awarded", "disabled", "ineligible", "capped_to_zero"}:
        errors.append(f"unknown reimbursement status: {award.status}")
    if award.rebate_rate < 0 or award.rebate_rate > 1:
        errors.append("rebate_rate must be between 0 and 1")
    for field in (
        "eligible_cost_microusd",
        "capped_cost_basis_microusd",
        "raw_reimbursement_microusd",
        "target_reimbursement_microusd",
    ):
        if getattr(award, field) < 0:
            errors.append(f"{field} must be non-negative")
    if award.status == "awarded" and award.target_reimbursement_microusd <= 0:
        errors.append("awarded reimbursement must have a positive target")
    if award.status != "awarded" and award.target_reimbursement_microusd != 0:
        errors.append("non-awarded reimbursement must have zero target")
    if award.status == "ineligible" and not award.ineligibility_reasons:
        errors.append("ineligible reimbursement must explain why")
    if award.status == "disabled" and award.ineligibility_reasons != ("policy_disabled",):
        errors.append("disabled reimbursement must use policy_disabled reason")
    if schedule is not None:
        total = sum(int(entry["amount_microusd"]) for entry in schedule.entries)
        if total != schedule.total_microusd:
            errors.append("schedule entries must sum to schedule total")
        if schedule.total_microusd != award.target_reimbursement_microusd:
            errors.append("schedule total must equal award target")
        if award.status == "awarded" and schedule.epoch_count != award.reimbursement_epochs:
            errors.append("awarded schedule epoch_count must match reimbursement_epochs")
        if award.status != "awarded" and schedule.entries:
            errors.append("non-awarded reimbursement must have an empty schedule")
    return errors


def verify_research_lab_reimbursements(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    fixture = _load_fixture(Path(fixture_path))
    policy = ReimbursementPolicy.from_mapping(fixture["policy"])
    disabled_policy = ReimbursementPolicy.from_mapping(fixture["disabled_policy"])
    low_snapshot = IslandParticipationSnapshot.from_mapping(fixture["snapshots"]["low"])
    high_snapshot = IslandParticipationSnapshot.from_mapping(fixture["snapshots"]["high"])
    generalist_snapshot = IslandParticipationSnapshot.from_mapping(fixture["snapshots"]["generalist_default"])
    no_usage = ReimbursementCapUsage.from_mapping(fixture["cap_usage"]["none"])

    _assert(compute_rebate_rate(low_snapshot, policy) == Decimal("1.000000"), "specialized low participation gives max rate")
    _assert(compute_rebate_rate(high_snapshot, policy) == Decimal("0.250000"), "specialized high participation gives min rate")
    _assert(compute_rebate_rate(generalist_snapshot, policy) == Decimal("0.500000"), "generalist starts at base rate")

    low_run = ResearchRunCostRecord.from_mapping(fixture["runs"]["eligible_low"])
    low_award = compute_reimbursement_award(low_run, low_snapshot, policy, no_usage)
    low_schedule = build_reimbursement_schedule(low_award, start_epoch=fixture["schedule_start_epoch"])
    _assert(low_award.status == "awarded", "eligible low-participation run awards")
    _assert(low_award.eligible_cost_microusd == fixture["expectations"]["funded_budget_cost_microusd"], "funded budget is used when actual spend is material")
    _assert(low_award.target_reimbursement_microusd == fixture["expectations"]["low_target_microusd"], "low target matches 100 percent")
    _assert(not validate_reimbursement_award(low_award, low_schedule), "low award validates")
    _assert(_schedule_total(low_schedule) == low_award.target_reimbursement_microusd, "schedule sums exactly")
    _assert("alpha_percent" in low_schedule.entries[0], "schedule includes alpha percent")
    _assert("alpha_share_ppm" in low_schedule.entries[0], "schedule includes alpha share ppm")
    _assert(low_award.to_dict() == compute_reimbursement_award(low_run, low_snapshot, policy, no_usage).to_dict(), "award deterministic")
    _assert(
        low_schedule.to_dict()
        == build_reimbursement_schedule(
            compute_reimbursement_award(low_run, low_snapshot, policy, no_usage),
            start_epoch=fixture["schedule_start_epoch"],
        ).to_dict(),
        "schedule deterministic",
    )

    decision = build_reimbursement_decision(low_award, low_schedule)
    _assert(not validate_schema_record("research_reimbursement.schema.json", decision), "decision schema validates")

    high_award = compute_reimbursement_award(
        fixture["runs"]["eligible_high"],
        high_snapshot,
        policy,
        no_usage,
    )
    _assert(high_award.target_reimbursement_microusd == fixture["expectations"]["high_target_microusd"], "high target matches min rate")

    generalist_award = compute_reimbursement_award(
        fixture["runs"]["generalist_five_dollar"],
        generalist_snapshot,
        policy,
        no_usage,
    )
    _assert(
        generalist_award.target_reimbursement_microusd
        == fixture["expectations"]["generalist_five_dollar_target_microusd"],
        "generalist default reimburses at base rate",
    )

    low_actual_award = compute_reimbursement_award(
        fixture["runs"]["low_actual_spend"],
        generalist_snapshot,
        policy,
        no_usage,
    )
    _assert(
        low_actual_award.eligible_cost_microusd == fixture["expectations"]["low_actual_cost_basis_microusd"],
        "low actual spend lowers the reimbursement cost basis",
    )

    zero_actual_award = compute_reimbursement_award(
        fixture["runs"]["zero_actual_spend"],
        generalist_snapshot,
        policy,
        no_usage,
    )
    _assert(zero_actual_award.status == "ineligible", "zero actual spend is ineligible")
    _assert("nonpositive_eligible_cost" in zero_actual_award.ineligibility_reasons, "zero spend reason is recorded")

    disabled_award = compute_reimbursement_award(low_run, low_snapshot, disabled_policy, no_usage)
    disabled_schedule = build_reimbursement_schedule(disabled_award)
    _assert(disabled_award.status == "disabled", "disabled policy returns disabled status")
    _assert(disabled_award.target_reimbursement_microusd == 0, "disabled policy returns zero target")
    _assert(not disabled_schedule.entries, "disabled policy returns no schedule entries")

    expensive_award = compute_reimbursement_award(
        fixture["runs"]["expensive"],
        low_snapshot,
        policy,
        no_usage,
    )
    _assert(expensive_award.target_reimbursement_microusd == fixture["expectations"]["per_run_cap_target_microusd"], "per-run cap applies")
    _assert("per_run" in expensive_award.caps_applied, "per-run cap is recorded")

    for cap_name, expected_key in (
        ("hotkey_near_cap", "hotkey_cap_target_microusd"),
        ("island_near_cap", "island_cap_target_microusd"),
        ("global_near_cap", "global_cap_target_microusd"),
    ):
        capped = compute_reimbursement_award(
            low_run,
            low_snapshot,
            policy,
            ReimbursementCapUsage.from_mapping(fixture["cap_usage"][cap_name]),
        )
        _assert(capped.target_reimbursement_microusd == fixture["expectations"][expected_key], f"{cap_name} target applies")
        _assert(cap_name.split("_")[0] in capped.caps_applied, f"{cap_name} is recorded")

    for case_name, expected_reasons in fixture["expectations"]["ineligible_cases"].items():
        award = compute_reimbursement_award(
            fixture["runs"][case_name],
            low_snapshot,
            policy,
            no_usage,
        )
        _assert(award.status == "ineligible", f"{case_name} is ineligible")
        for reason in expected_reasons:
            _assert(reason in award.ineligibility_reasons, f"{case_name} includes {reason}")
        _assert(not build_reimbursement_schedule(award).entries, f"{case_name} has no schedule")

    return {
        "low_rebate_rate": _rate_to_float(compute_rebate_rate(low_snapshot, policy)),
        "high_rebate_rate": _rate_to_float(compute_rebate_rate(high_snapshot, policy)),
        "low_award_microusd": low_award.target_reimbursement_microusd,
        "high_award_microusd": high_award.target_reimbursement_microusd,
        "schedule_epochs": len(low_schedule.entries),
        "schedule_total_microusd": _schedule_total(low_schedule),
        "fixture_cases": len(fixture["runs"]),
    }


def _round_microusd(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _eligible_reimbursement_cost(run: ResearchRunCostRecord, policy: ReimbursementPolicy) -> int:
    if policy.material_spend_ratio is None:
        return run.verified_compute_cost_microusd + run.miner_openrouter_cost_microusd
    funded = max(0, run.verified_compute_cost_microusd)
    actual = max(0, run.miner_openrouter_cost_microusd)
    if actual <= 0:
        return 0
    if funded <= 0:
        return actual
    material_threshold = _round_microusd(Decimal(funded) * policy.material_spend_ratio)
    if actual < material_threshold:
        return actual
    return funded


def _alpha_entry_fields(amount_microusd: int, usd_per_0_1_percent_epoch: Decimal | None) -> dict[str, Any]:
    if usd_per_0_1_percent_epoch is None:
        return {}
    amount_usd = Decimal(amount_microusd) / MICRO_USD
    alpha_percent = (amount_usd / usd_per_0_1_percent_epoch) * Decimal("0.1")
    alpha_share_ppm = int((alpha_percent * Decimal("10000")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return {
        "alpha_percent": _rate_to_float(alpha_percent),
        "alpha_share_ppm": alpha_share_ppm,
    }


def _run_ineligibility_reasons(run: ResearchRunCostRecord) -> list[str]:
    reasons: list[str] = []
    if not run.paid_research_loop:
        reasons.append("not_paid_research_loop")
    if not run.valid_receipt:
        reasons.append("missing_valid_receipt")
    if not (run.verified_loop_start_payment or run.preserved_loop_start_credit):
        reasons.append("missing_loop_start_payment_or_credit")
    if not run.miner_openrouter_key_present:
        reasons.append("missing_miner_openrouter_key")
    if not run.trusted_cost_ledger:
        reasons.append("missing_trusted_cost_ledger")
    if not run.passed_abuse_checks:
        reasons.append("abuse_checks_failed")
    if run.refunded:
        reasons.append("refunded")
    if run.voided:
        reasons.append("voided")
    if run.duplicate:
        reasons.append("duplicate_run")
    if run.novelty_rejected:
        reasons.append("novelty_rejected")
    if run.self_cancelled_before_minimum_work:
        reasons.append("self_cancelled_before_minimum_work")
    if run.banned_hotkey:
        reasons.append("banned_hotkey")
    return reasons


def _remaining_caps(
    policy: ReimbursementPolicy,
    usage: ReimbursementCapUsage,
) -> tuple[tuple[str, int], ...]:
    return (
        (
            "hotkey",
            max(0, usd_to_microusd(policy.max_usd_per_hotkey_day) - usage.hotkey_day_awarded_microusd),
        ),
        (
            "island",
            max(0, usd_to_microusd(policy.max_usd_per_island_day) - usage.island_day_awarded_microusd),
        ),
        (
            "global",
            max(0, usd_to_microusd(policy.global_budget_usd) - usage.global_awarded_microusd),
        ),
    )


def _zero_award(
    *,
    run: ResearchRunCostRecord,
    policy: ReimbursementPolicy,
    score: Decimal,
    fraction: Decimal,
    rebate_rate: Decimal,
    eligible_cost: int,
    status: str,
    reasons: Sequence[str],
    input_hash: str,
) -> ReimbursementAward:
    award = ReimbursementAward(
        award_id="",
        run_id=run.run_id,
        miner_hotkey=run.miner_hotkey,
        island=run.island,
        run_day=run.run_day,
        status=status,
        participation_score=score,
        participation_fraction=fraction,
        rebate_rate=rebate_rate,
        eligible_cost_microusd=max(0, eligible_cost),
        capped_cost_basis_microusd=0,
        raw_reimbursement_microusd=0,
        target_reimbursement_microusd=0,
        reimbursement_epochs=policy.reimbursement_epochs,
        caps_applied=(),
        ineligibility_reasons=tuple(reasons),
        loop_start_fee_included=policy.include_loop_start_fee_in_base,
        input_hash=input_hash,
        usd_per_0_1_percent_epoch=policy.usd_per_0_1_percent_epoch,
    )
    return _with_award_id(award)


def _award_input_hash(
    run: ResearchRunCostRecord,
    snapshot: IslandParticipationSnapshot,
    policy: ReimbursementPolicy,
    usage: ReimbursementCapUsage,
) -> str:
    return sha256_json(
        {
            "run": run.__dict__,
            "snapshot": snapshot.__dict__,
            "policy": _policy_hash_dict(policy),
            "usage": usage.__dict__,
        }
    )


def _policy_hash_dict(policy: ReimbursementPolicy) -> dict[str, Any]:
    return {
        "policy_id": policy.policy_id,
        "enabled": policy.enabled,
        "min_rebate_rate": str(policy.min_rebate_rate),
        "max_rebate_rate": str(policy.max_rebate_rate),
        "base_rebate_rate": str(policy.base_rebate_rate),
        "high_participation_target": str(policy.high_participation_target),
        "reimbursement_epochs": policy.reimbursement_epochs,
        "max_usd_per_run": str(policy.max_usd_per_run),
        "max_usd_per_hotkey_day": str(policy.max_usd_per_hotkey_day),
        "max_usd_per_island_day": str(policy.max_usd_per_island_day),
        "global_budget_usd": str(policy.global_budget_usd),
        "include_loop_start_fee_in_base": policy.include_loop_start_fee_in_base,
        "material_spend_ratio": str(policy.material_spend_ratio) if policy.material_spend_ratio is not None else None,
        "default_island": policy.default_island,
        "usd_per_0_1_percent_epoch": (
            str(policy.usd_per_0_1_percent_epoch) if policy.usd_per_0_1_percent_epoch is not None else None
        ),
        "distinct_funded_hotkey_weight": str(policy.distinct_funded_hotkey_weight),
        "paid_loop_weight": str(policy.paid_loop_weight),
        "unique_brief_weight": str(policy.unique_brief_weight),
    }


def _with_award_id(award: ReimbursementAward) -> ReimbursementAward:
    data = award.to_dict()
    data.pop("award_id", None)
    award_id = "reimbursement_award:" + sha256_json(data)
    return ReimbursementAward(**{**award.__dict__, "award_id": award_id})


def _award_from_mapping(data: Mapping[str, Any]) -> ReimbursementAward:
    return ReimbursementAward(
        award_id=str(data["award_id"]),
        run_id=str(data["run_id"]),
        miner_hotkey=str(data["miner_hotkey"]),
        island=str(data["island"]),
        run_day=str(data["run_day"]),
        status=str(data["status"]),
        participation_score=_decimal(data["participation_score"]),
        participation_fraction=_decimal(data["participation_fraction"]),
        rebate_rate=_decimal(data["rebate_rate"]),
        eligible_cost_microusd=int(data["eligible_cost_microusd"]),
        capped_cost_basis_microusd=int(data["capped_cost_basis_microusd"]),
        raw_reimbursement_microusd=int(data["raw_reimbursement_microusd"]),
        target_reimbursement_microusd=int(data["target_reimbursement_microusd"]),
        reimbursement_epochs=int(data["reimbursement_epochs"]),
        caps_applied=tuple(str(item) for item in data.get("caps_applied", [])),
        ineligibility_reasons=tuple(str(item) for item in data.get("ineligibility_reasons", [])),
        loop_start_fee_included=bool(data["loop_start_fee_included"]),
        input_hash=str(data["input_hash"]),
        usd_per_0_1_percent_epoch=(
            _decimal(data["usd_per_0_1_percent_epoch"]) if "usd_per_0_1_percent_epoch" in data else None
        ),
    )


def _schedule_from_mapping(data: Mapping[str, Any]) -> ReimbursementSchedule:
    return ReimbursementSchedule(
        schedule_id=str(data["schedule_id"]),
        award_id=str(data["award_id"]),
        status=str(data["status"]),
        start_epoch=int(data["start_epoch"]),
        epoch_count=int(data["epoch_count"]),
        total_microusd=int(data["total_microusd"]),
        entries=tuple(dict(entry) for entry in data.get("entries", [])),
    )


def _schedule_total(schedule: ReimbursementSchedule) -> int:
    return sum(int(entry["amount_microusd"]) for entry in schedule.entries)


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
