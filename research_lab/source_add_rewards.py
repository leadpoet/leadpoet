"""SOURCE_ADD two-leg emission rewards (sourceexperiments.md §3, owner-final).

Supersedes the P1.5 USD trial-yield bounty bands. Both legs are fixed-term
emission streams within a separate, first-priority SOURCE_ADD allocation:

- **Leg 1 — functional source submission**: 1% of miner emissions ×
  ``RESEARCH_LAB_REWARD_EPOCHS`` (20), created only after a measured V2
  functional probe passes. Flat, one per adapter, ever.
- **Leg 2 — implementation rider**: +5% × 20 epochs to the ADAPTER OWNER,
  created alongside the implementing merge's champion grant only when the
  LLM final judge decides the already-winning change was helped by a known
  SOURCE_ADD API. Paid to the owner even when the house arm wires it.

Reward records enter the first-priority ``source_add_allocations`` section.
Their paid percentage is deducted from the configured Research Lab cap before
the unchanged reimbursement/champion allocator runs against the remainder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

from .canonical import sha256_json

# §3.4 config defaults (env-tunable; these are the launch values).
DEFAULT_LEG1_ALPHA_PERCENT = 1.0
DEFAULT_LEG2_ALPHA_PERCENT = 5.0
DEFAULT_REWARD_EPOCHS = 20

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
        """Compatibility alias for the SOURCE_ADD allocation obligation shape.

        The method name is retained for callers created before SOURCE_ADD was
        separated from champion rewards. New allocation code does not place
        this row in champion sections.
        """

        return {
            "champion_reward_id": self.reward_ref,
            "source_add_reward_id": self.reward_ref,
            "adapter_id": self.adapter_id,
            "leg": int(self.leg),
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
        if evidence.get("llm_judge_passed") is not True:
            errors.append("leg 2 requires llm_judge_passed=true trigger evidence")
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
    trigger_evidence: Mapping[str, Any] | None = None,
) -> SourceAddRewardRecord | None:
    """Create the measured-functional-source leg. One per adapter, ever.

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
        trigger_evidence=dict(trigger_evidence or {}) or None,
        public_label=PUBLIC_LABELS[REWARD_KIND_SOURCE_ACCEPTANCE],
    )


def _has_leg(existing_rewards: Sequence[Mapping[str, Any]], adapter_id: str, leg: int) -> bool:
    for row in existing_rewards:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("adapter_id") or "") == adapter_id and int(row.get("leg") or 0) == leg:
            return True
    return False


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
