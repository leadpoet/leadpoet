"""SOURCE_ADD two-leg emission rewards (sourceexperiments.md §3, owner-final).

Supersedes the P1.5 USD trial-yield bounty bands. Both legs are fixed-term
emission streams on the existing champion-allocation rails:

- **Leg 1 — credible submission**: 1% of miner emissions ×
  ``RESEARCH_LAB_REWARD_EPOCHS`` (20), created when a miner-submitted source
  reaches ``provenance_precheck_passed``. Flat, one per adapter, ever.
- **Leg 2 — implementation rider**: +5% × 20 epochs to the ADAPTER OWNER,
  created alongside the implementing merge's champion grant, triggered
  mechanically (all four): (1) merged diff routes to the adapter, (2) merge
  cleared the normal score-only bar, (3) the patch survived the shadow-monitor
  window, (4) adapter-ablation attribution ≥ threshold. Never before the
  shadow window elapses; expires N months after max(acceptance, market open);
  paid to the owner even when the house arm wires it. On-chain paid is paid —
  an auto-revert stops the stream going forward only.

Reward records enter the ``champion_allocations`` / ``queued_champion_allocations``
sections with a ``reward_kind`` field, so the validator pays them with zero
code change (it pays whatever the allocation doc says within the lab cap).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from .canonical import sha256_json

# §3.4 config defaults (env-tunable; these are the launch values).
DEFAULT_LEG1_ALPHA_PERCENT = 1.0
DEFAULT_LEG2_ALPHA_PERCENT = 5.0
DEFAULT_REWARD_EPOCHS = 20
DEFAULT_LEG2_EXPIRY_MONTHS = 6
DEFAULT_SHADOW_WINDOW_DAYS = 7
DEFAULT_ABLATION_THRESHOLD_POINTS = 0.5

REWARD_KIND_SOURCE_ACCEPTANCE = "source_acceptance"
REWARD_KIND_SOURCE_IMPLEMENTATION = "source_implementation"

PUBLIC_LABELS = {
    REWARD_KIND_SOURCE_ACCEPTANCE: "Source acceptance reward",
    REWARD_KIND_SOURCE_IMPLEMENTATION: "Source implementation reward",
}


class SourceAddRewardLeg(int, Enum):
    ACCEPTANCE = 1
    IMPLEMENTATION = 2


class SourceAddRewardState(str, Enum):
    ACTIVE = "active"
    QUEUED = "queued"
    PARTIALLY_PAID = "partially_paid"
    STOPPED_FORWARD = "stopped_forward"  # revert semantics: paid is paid


@dataclass(frozen=True)
class SourceAddRewardRecord:
    reward_ref: str
    adapter_id: str
    miner_ref: str
    leg: int
    alpha_percent: float
    reward_epochs: int
    start_epoch: int
    state: str
    reward_kind: str
    allocation_ref: str = ""
    trigger_evidence: dict[str, Any] | None = None
    public_label: str = ""
    stopped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def champion_reward_row(self) -> dict[str, Any]:
        """The row shape for the existing champion-reward rails.

        ``reward_kind`` is MANDATED on the row so the validator needs zero code
        change: allocation building treats this like any champion obligation
        under ``validator_policy_lab_cap_ceiling``; the kind only labels it.
        """

        return {
            "champion_reward_id": self.reward_ref,
            "miner_hotkey": self.miner_ref,
            "candidate_id": "",
            "run_id": "",
            "island": "generalist",
            "current_reward_status": self.state,
            "reward_status": self.state,
            "desired_alpha_percent": float(self.alpha_percent),
            "start_epoch": int(self.start_epoch),
            "epoch_count": int(self.reward_epochs),
            "improvement_points": 0.0,
            "threshold_points": 0.0,
            "reward_kind": self.reward_kind,
            "public_label": self.public_label or PUBLIC_LABELS.get(self.reward_kind, ""),
        }


def validate_source_add_reward_record(record: SourceAddRewardRecord | Mapping[str, Any]) -> list[str]:
    if isinstance(record, Mapping):
        record = SourceAddRewardRecord(
            reward_ref=str(record.get("reward_ref") or ""),
            adapter_id=str(record.get("adapter_id") or ""),
            miner_ref=str(record.get("miner_ref") or ""),
            leg=int(record.get("leg") or 0),
            alpha_percent=float(record.get("alpha_percent") or 0.0),
            reward_epochs=int(record.get("reward_epochs") or 0),
            start_epoch=int(record.get("start_epoch") or 0),
            state=str(record.get("state") or ""),
            reward_kind=str(record.get("reward_kind") or ""),
            allocation_ref=str(record.get("allocation_ref") or ""),
            trigger_evidence=dict(record.get("trigger_evidence") or {}) or None,
            public_label=str(record.get("public_label") or ""),
            stopped_reason=str(record.get("stopped_reason") or ""),
        )
    errors: list[str] = []
    if not record.reward_ref.startswith("source_add_reward:"):
        errors.append("reward_ref must be source_add_reward:-prefixed")
    if not record.adapter_id:
        errors.append("adapter_id is required")
    if not record.miner_ref:
        errors.append("miner_ref is required")
    if record.leg not in (1, 2):
        errors.append("leg must be 1 or 2")
    if record.alpha_percent <= 0:
        errors.append("alpha_percent must be positive")
    if record.reward_epochs <= 0:
        errors.append("reward_epochs must be positive")
    if record.state not in {state.value for state in SourceAddRewardState}:
        errors.append(f"unknown reward state: {record.state}")
    if record.reward_kind not in PUBLIC_LABELS:
        errors.append(f"unknown reward_kind: {record.reward_kind}")
    if record.leg == 1 and record.reward_kind != REWARD_KIND_SOURCE_ACCEPTANCE:
        errors.append("leg 1 must carry reward_kind source_acceptance")
    if record.leg == 2:
        if record.reward_kind != REWARD_KIND_SOURCE_IMPLEMENTATION:
            errors.append("leg 2 must carry reward_kind source_implementation")
        evidence = record.trigger_evidence or {}
        if not evidence.get("shadow_window_passed"):
            errors.append("leg 2 requires shadow_window_passed trigger evidence")
        if not evidence.get("ablation_passed"):
            errors.append("leg 2 requires ablation_passed trigger evidence")
    return errors


def _reward_ref(adapter_id: str, leg: int) -> str:
    return "source_add_reward:" + sha256_json({"adapter_id": adapter_id, "leg": leg}).split(":", 1)[1][:16]


def create_leg1_reward(
    *,
    adapter_id: str,
    miner_ref: str,
    start_epoch: int,
    existing_rewards: Sequence[Mapping[str, Any]] = (),
    alpha_percent: float = DEFAULT_LEG1_ALPHA_PERCENT,
    reward_epochs: int = DEFAULT_REWARD_EPOCHS,
    state: str = SourceAddRewardState.ACTIVE.value,
) -> SourceAddRewardRecord | None:
    """Create the credible-submission leg. One per adapter, ever.

    Queue policy is normal FIFO on the allocation rails: pass
    ``state="queued"`` when the lab cap cannot pay right now, exactly like
    improvement grants. Returns None when this adapter already has Leg 1.
    """

    if _has_leg(existing_rewards, adapter_id, 1):
        return None
    return SourceAddRewardRecord(
        reward_ref=_reward_ref(adapter_id, 1),
        adapter_id=adapter_id,
        miner_ref=miner_ref,
        leg=1,
        alpha_percent=float(alpha_percent),
        reward_epochs=int(reward_epochs),
        start_epoch=int(start_epoch),
        state=state,
        reward_kind=REWARD_KIND_SOURCE_ACCEPTANCE,
        public_label=PUBLIC_LABELS[REWARD_KIND_SOURCE_ACCEPTANCE],
    )


def _has_leg(existing_rewards: Sequence[Mapping[str, Any]], adapter_id: str, leg: int) -> bool:
    for row in existing_rewards:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("adapter_id") or "") == adapter_id and int(row.get("leg") or 0) == leg:
            return True
    return False


def _parse_iso(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def leg2_expired(
    *,
    now: str,
    accepted_at: str,
    market_open_at: str = "",
    expiry_months: int = DEFAULT_LEG2_EXPIRY_MONTHS,
) -> bool:
    """Leg 2 lapses ``expiry_months`` after max(acceptance, paid-loop market open).

    The market-open clock protects early contributors: pre-market acceptances
    do not burn expiry time during the quiet period.
    """

    now_dt = _parse_iso(now)
    anchors = [dt for dt in (_parse_iso(accepted_at), _parse_iso(market_open_at)) if dt is not None]
    if now_dt is None or not anchors:
        return False  # unknown clocks never expire a reward silently
    anchor = max(anchors)
    total_months = anchor.year * 12 + (anchor.month - 1) + max(0, int(expiry_months))
    expiry_year, expiry_month = divmod(total_months, 12)
    expiry_day = min(
        anchor.day,
        [31, 29 if expiry_year % 4 == 0 and (expiry_year % 100 != 0 or expiry_year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][expiry_month],
    )
    expiry = anchor.replace(year=expiry_year, month=expiry_month + 1, day=expiry_day)
    return now_dt >= expiry


def evaluate_leg2_trigger(
    *,
    adapter_id: str,
    catalog_registry_ids: Sequence[str],
    merged: bool,
    merged_diff_routed_registry_ids: Sequence[str],
    merge_cleared_score_bar: bool,
    shadow_monitor_live: bool,
    shadow_window_days_elapsed: float,
    shadow_window_survived: bool,
    ablation_adapter_on_score: float | None,
    ablation_adapter_off_score: float | None,
    now: str,
    accepted_at: str,
    market_open_at: str = "",
    existing_rewards: Sequence[Mapping[str, Any]] = (),
    shadow_window_days_required: float = DEFAULT_SHADOW_WINDOW_DAYS,
    ablation_threshold_points: float = DEFAULT_ABLATION_THRESHOLD_POINTS,
    expiry_months: int = DEFAULT_LEG2_EXPIRY_MONTHS,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Mechanical, binary leg-2 trigger. Returns (armed, blockers, evidence).

    All four §3.2 conditions must hold, the shadow monitor must be LIVE (a
    liveness check at creation time, not just a flag), the reward must be
    unexpired, and no leg-2 may already exist for the adapter.
    """

    blockers: list[str] = []
    if _has_leg(existing_rewards, adapter_id, 2):
        blockers.append("leg2_already_created")
    if leg2_expired(now=now, accepted_at=accepted_at, market_open_at=market_open_at, expiry_months=expiry_months):
        blockers.append("leg2_expired")
    if not merged:
        blockers.append("candidate_not_merged")
    routed = {str(item) for item in merged_diff_routed_registry_ids}
    adapter_registry_ids = {str(item) for item in catalog_registry_ids if item}
    if not adapter_registry_ids or not (routed & adapter_registry_ids):
        blockers.append("diff_does_not_route_to_adapter")
    if not merge_cleared_score_bar:
        blockers.append("merge_below_score_bar")
    if not shadow_monitor_live:
        blockers.append("shadow_monitor_not_live")
    if shadow_window_days_elapsed < float(shadow_window_days_required):
        blockers.append("shadow_window_not_elapsed")
    elif not shadow_window_survived:
        blockers.append("shadow_window_not_survived")
    ablation_delta: float | None = None
    if ablation_adapter_on_score is None or ablation_adapter_off_score is None:
        blockers.append("ablation_not_run")
    else:
        ablation_delta = float(ablation_adapter_on_score) - float(ablation_adapter_off_score)
        if ablation_delta < float(ablation_threshold_points):
            blockers.append("ablation_attribution_below_threshold")
    evidence = {
        "adapter_id": adapter_id,
        "routed_registry_ids": sorted(routed & adapter_registry_ids),
        "merge_cleared_score_bar": bool(merge_cleared_score_bar),
        "shadow_window_days_elapsed": float(shadow_window_days_elapsed),
        "shadow_window_passed": shadow_window_days_elapsed >= float(shadow_window_days_required)
        and bool(shadow_window_survived),
        "ablation_delta_points": ablation_delta,
        "ablation_threshold_points": float(ablation_threshold_points),
        "ablation_passed": ablation_delta is not None and ablation_delta >= float(ablation_threshold_points),
        "evaluated_at": str(now),
    }
    return (not blockers), blockers, evidence


def create_leg2_reward(
    *,
    adapter_id: str,
    adapter_owner_miner_ref: str,
    start_epoch: int,
    trigger_evidence: Mapping[str, Any],
    existing_rewards: Sequence[Mapping[str, Any]] = (),
    alpha_percent: float = DEFAULT_LEG2_ALPHA_PERCENT,
    reward_epochs: int = DEFAULT_REWARD_EPOCHS,
    state: str = SourceAddRewardState.ACTIVE.value,
) -> SourceAddRewardRecord | None:
    """Create the implementation rider for the ADAPTER OWNER.

    ``adapter_owner_miner_ref`` is always the adapter's owner regardless of who
    funded the wiring loop — including the house arm (the house holds no
    stream; the owner's leg 2 still fires). One-time per adapter; idempotent
    first-only. A merge routing to multiple new adapters fires one call per
    adapter.
    """

    if _has_leg(existing_rewards, adapter_id, 2):
        return None
    record = SourceAddRewardRecord(
        reward_ref=_reward_ref(adapter_id, 2),
        adapter_id=adapter_id,
        miner_ref=adapter_owner_miner_ref,
        leg=2,
        alpha_percent=float(alpha_percent),
        reward_epochs=int(reward_epochs),
        start_epoch=int(start_epoch),
        state=state,
        reward_kind=REWARD_KIND_SOURCE_IMPLEMENTATION,
        trigger_evidence=dict(trigger_evidence),
        public_label=PUBLIC_LABELS[REWARD_KIND_SOURCE_IMPLEMENTATION],
    )
    errors = validate_source_add_reward_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    return record


def stop_reward_forward(record: SourceAddRewardRecord, *, reason: str) -> SourceAddRewardRecord:
    """Revert semantics: on-chain paid is paid; the stream stops going forward."""

    return replace(
        record,
        state=SourceAddRewardState.STOPPED_FORWARD.value,
        stopped_reason=str(reason)[:200],
    )
