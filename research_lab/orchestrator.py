"""Phase 1 local orchestrator scaffold.

This module models tickets, queue entries, budget caps, novelty decisions, and
private probe schedules. It is deliberately inert: every helper returns local
records only and depends on the P1.0 fail-closed workflow guards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .canonical import sha256_json
from .loop_foundation import (
    ArtifactReleaseState,
    LoopWorkflowGuards,
    ReleasePolicyRecord,
    VisibilityPolicy,
    assert_loop_workflows_disabled,
    default_loop_workflow_guards,
    validate_release_policy,
    verify_loop_foundation,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "orchestrator_fixtures.json"


class TicketStatus(str, Enum):
    DRAFT = "draft"
    FUNDED = "funded"
    QUEUED = "queued"
    STOPPED = "stopped"
    EXHAUSTED = "exhausted"


class NoveltyDecision(str, Enum):
    PASS = "pass"
    REJECT = "reject"


class ProbeScheduleStatus(str, Enum):
    LOCAL_ONLY_SCHEDULED = "local_only_scheduled"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class BudgetCaps:
    loop_unit_cents: int = 1000
    max_loops_per_day: int = 10
    per_node_eval_cap_cents: int = 3000
    loop_batch_cap_cents: int = 10000
    per_node_wall_clock_cap_s: int = 900
    loop_batch_wall_clock_cap_s: int = 3600

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "BudgetCaps":
        data = data or {}
        return cls(
            loop_unit_cents=int(data.get("loop_unit_cents", cls.loop_unit_cents)),
            max_loops_per_day=int(data.get("max_loops_per_day", cls.max_loops_per_day)),
            per_node_eval_cap_cents=int(data.get("per_node_eval_cap_cents", cls.per_node_eval_cap_cents)),
            loop_batch_cap_cents=int(data.get("loop_batch_cap_cents", cls.loop_batch_cap_cents)),
            per_node_wall_clock_cap_s=int(data.get("per_node_wall_clock_cap_s", cls.per_node_wall_clock_cap_s)),
            loop_batch_wall_clock_cap_s=int(data.get("loop_batch_wall_clock_cap_s", cls.loop_batch_wall_clock_cap_s)),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TicketRecord:
    ticket_id: str
    owner_ref: str
    miner_ref: str
    brief_ref: str
    island: str
    target_component: str
    funded_balance_cents: int
    remaining_balance_cents: int
    status: str = TicketStatus.FUNDED.value
    budget_caps: BudgetCaps = BudgetCaps()
    visibility_policy: str = VisibilityPolicy.DEFAULT_PRIVATE.value
    artifact_release_state: str = ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value
    created_seq: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TicketRecord":
        return cls(
            ticket_id=str(data["ticket_id"]),
            owner_ref=str(data["owner_ref"]),
            miner_ref=str(data["miner_ref"]),
            brief_ref=str(data["brief_ref"]),
            island=str(data["island"]),
            target_component=str(data["target_component"]),
            funded_balance_cents=int(data["funded_balance_cents"]),
            remaining_balance_cents=int(data["remaining_balance_cents"]),
            status=str(data.get("status", TicketStatus.FUNDED.value)),
            budget_caps=BudgetCaps.from_mapping(data.get("budget_caps")),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.DEFAULT_PRIVATE.value)),
            artifact_release_state=str(
                data.get("artifact_release_state", ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value)
            ),
            created_seq=int(data.get("created_seq", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["budget_caps"] = self.budget_caps.to_dict()
        return data


@dataclass(frozen=True)
class BudgetReservation:
    ticket_id: str
    requested_loops: int
    requested_loop_cost_cents: int
    estimated_node_eval_cents: int
    estimated_node_wall_clock_s: int
    estimated_wall_clock_s: int
    approved: bool
    remaining_after_cents: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueueEntry:
    queue_id: str
    ticket_id: str
    priority_score: int
    created_seq: int
    lab_only: bool = True
    scheduler_enabled: bool = False
    status: str = "local_only_queued"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "QueueEntry":
        return cls(
            queue_id=str(data["queue_id"]),
            ticket_id=str(data["ticket_id"]),
            priority_score=int(data["priority_score"]),
            created_seq=int(data["created_seq"]),
            lab_only=bool(data.get("lab_only", True)),
            scheduler_enabled=bool(data.get("scheduler_enabled", False)),
            status=str(data.get("status", "local_only_queued")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PriorReceipt:
    receipt_ref: str
    brief_ref: str
    summary_text: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PriorReceipt":
        return cls(
            receipt_ref=str(data["receipt_ref"]),
            brief_ref=str(data["brief_ref"]),
            summary_text=str(data["summary_text"]),
        )


@dataclass(frozen=True)
class NoveltyGateResult:
    decision: str
    similarity_score: float
    nearest_prior_receipt_refs: tuple[str, ...]
    static_corpus_index_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "similarity_score": self.similarity_score,
            "nearest_prior_receipt_refs": list(self.nearest_prior_receipt_refs),
            "static_corpus_index_version": self.static_corpus_index_version,
        }


@dataclass(frozen=True)
class PrivateProbeScheduleRecord:
    probe_id: str
    ticket_id: str
    brief_ref: str
    owner_ref: str
    status: str = ProbeScheduleStatus.LOCAL_ONLY_SCHEDULED.value
    lab_only: bool = True
    public_workflow_side_effects: bool = False
    scheduler_enabled: bool = False
    visibility_policy: str = VisibilityPolicy.DEFAULT_PRIVATE.value
    artifact_release_state: str = ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value
    reason: str = "private probe schedule record only; no job is started"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_ticket_record(ticket: TicketRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(ticket, TicketRecord):
        ticket = TicketRecord.from_mapping(ticket)

    errors: list[str] = []
    if not ticket.ticket_id:
        errors.append("ticket_id is required")
    if not ticket.owner_ref:
        errors.append("owner_ref is required")
    if not ticket.miner_ref:
        errors.append("miner_ref is required")
    if not ticket.brief_ref:
        errors.append("brief_ref is required")
    if ticket.status not in {status.value for status in TicketStatus}:
        errors.append(f"unknown ticket status: {ticket.status}")
    if ticket.funded_balance_cents < 0:
        errors.append("funded_balance_cents must be non-negative")
    if ticket.remaining_balance_cents < 0:
        errors.append("remaining_balance_cents must be non-negative")
    if ticket.remaining_balance_cents > ticket.funded_balance_cents:
        errors.append("remaining_balance_cents cannot exceed funded_balance_cents")

    release_errors = validate_release_policy(
        ReleasePolicyRecord(
            artifact_ref=f"ticket:{ticket.ticket_id}",
            artifact_type="ticket",
            visibility_policy=ticket.visibility_policy,
            artifact_release_state=ticket.artifact_release_state,
            reason="ticket records consume P1.0 visibility policy",
        )
    )
    errors.extend(release_errors)
    errors.extend(validate_budget_caps(ticket.budget_caps))
    return errors


def validate_budget_caps(caps: BudgetCaps | Mapping[str, Any]) -> list[str]:
    if not isinstance(caps, BudgetCaps):
        caps = BudgetCaps.from_mapping(caps)
    errors: list[str] = []
    if caps.loop_unit_cents <= 0:
        errors.append("loop_unit_cents must be positive")
    if caps.max_loops_per_day <= 0:
        errors.append("max_loops_per_day must be positive")
    if caps.per_node_eval_cap_cents <= 0:
        errors.append("per_node_eval_cap_cents must be positive")
    if caps.loop_batch_cap_cents <= 0:
        errors.append("loop_batch_cap_cents must be positive")
    if caps.per_node_wall_clock_cap_s <= 0:
        errors.append("per_node_wall_clock_cap_s must be positive")
    if caps.loop_batch_wall_clock_cap_s <= 0:
        errors.append("loop_batch_wall_clock_cap_s must be positive")
    return errors


def plan_budget_reservation(
    ticket: TicketRecord | Mapping[str, Any],
    *,
    requested_loops: int,
    estimated_node_eval_cents: int,
    estimated_node_wall_clock_s: int,
    estimated_wall_clock_s: int,
) -> BudgetReservation:
    if not isinstance(ticket, TicketRecord):
        ticket = TicketRecord.from_mapping(ticket)

    errors = validate_ticket_record(ticket)
    caps = ticket.budget_caps
    requested_loop_cost_cents = requested_loops * caps.loop_unit_cents

    if requested_loops <= 0:
        errors.append("requested_loops must be positive")
    if requested_loops > caps.max_loops_per_day:
        errors.append("requested_loops exceeds max_loops_per_day")
    if requested_loop_cost_cents > caps.loop_batch_cap_cents:
        errors.append("requested loop spend exceeds loop_batch_cap_cents")
    if requested_loop_cost_cents > ticket.remaining_balance_cents:
        errors.append("requested loop spend exceeds remaining balance")
    if estimated_node_eval_cents > caps.per_node_eval_cap_cents:
        errors.append("estimated_node_eval_cents exceeds per_node_eval_cap_cents")
    if estimated_node_wall_clock_s > caps.per_node_wall_clock_cap_s:
        errors.append("estimated_node_wall_clock_s exceeds per_node_wall_clock_cap_s")
    if estimated_wall_clock_s > caps.loop_batch_wall_clock_cap_s:
        errors.append("estimated_wall_clock_s exceeds loop_batch_wall_clock_cap_s")

    approved = not errors
    remaining_after = ticket.remaining_balance_cents - requested_loop_cost_cents if approved else ticket.remaining_balance_cents
    return BudgetReservation(
        ticket_id=ticket.ticket_id,
        requested_loops=requested_loops,
        requested_loop_cost_cents=requested_loop_cost_cents,
        estimated_node_eval_cents=estimated_node_eval_cents,
        estimated_node_wall_clock_s=estimated_node_wall_clock_s,
        estimated_wall_clock_s=estimated_wall_clock_s,
        approved=approved,
        remaining_after_cents=remaining_after,
        errors=tuple(errors),
    )


def order_queue_entries(entries: Sequence[QueueEntry | Mapping[str, Any]]) -> list[QueueEntry]:
    queue_entries = [entry if isinstance(entry, QueueEntry) else QueueEntry.from_mapping(entry) for entry in entries]
    return sorted(queue_entries, key=lambda entry: (-entry.priority_score, entry.created_seq, entry.ticket_id))


def validate_queue_entries(entries: Sequence[QueueEntry | Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for entry in [entry if isinstance(entry, QueueEntry) else QueueEntry.from_mapping(entry) for entry in entries]:
        if not entry.lab_only:
            errors.append(f"{entry.queue_id}: queue entry must be lab_only")
        if entry.scheduler_enabled:
            errors.append(f"{entry.queue_id}: scheduler_enabled must remain false")
        if not entry.ticket_id:
            errors.append(f"{entry.queue_id}: ticket_id is required")
    return errors


def evaluate_novelty(
    brief_text: str,
    prior_receipts: Sequence[PriorReceipt | Mapping[str, Any]],
    *,
    static_corpus_index_version: str,
    reject_threshold: float = 0.72,
    nearest_count: int = 2,
) -> NoveltyGateResult:
    receipts = [
        receipt if isinstance(receipt, PriorReceipt) else PriorReceipt.from_mapping(receipt)
        for receipt in prior_receipts
    ]
    brief_tokens = _tokens(brief_text)
    scored = sorted(
        (
            (_jaccard(brief_tokens, _tokens(receipt.summary_text)), receipt.receipt_ref)
            for receipt in receipts
        ),
        key=lambda item: (-item[0], item[1]),
    )
    best_score = scored[0][0] if scored else 0.0
    nearest = tuple(ref for score, ref in scored[:nearest_count] if score > 0.0)
    decision = NoveltyDecision.REJECT.value if best_score >= reject_threshold else NoveltyDecision.PASS.value
    if decision == NoveltyDecision.REJECT.value and not nearest:
        nearest = tuple(ref for _, ref in scored[:1])
    return NoveltyGateResult(
        decision=decision,
        similarity_score=round(best_score, 6),
        nearest_prior_receipt_refs=nearest,
        static_corpus_index_version=static_corpus_index_version,
    )


def build_private_probe_schedule(
    ticket: TicketRecord | Mapping[str, Any],
    *,
    guards: Optional[LoopWorkflowGuards | Mapping[str, Any]] = None,
) -> PrivateProbeScheduleRecord:
    if not isinstance(ticket, TicketRecord):
        ticket = TicketRecord.from_mapping(ticket)
    assert_loop_workflows_disabled(guards or default_loop_workflow_guards())
    ticket_errors = validate_ticket_record(ticket)
    if ticket_errors:
        raise ValueError("; ".join(ticket_errors))
    if ticket.status not in {TicketStatus.FUNDED.value, TicketStatus.QUEUED.value}:
        raise ValueError(f"ticket status cannot be scheduled for a private probe: {ticket.status}")

    probe_id = "probe:" + sha256_json(
        {
            "ticket_id": ticket.ticket_id,
            "brief_ref": ticket.brief_ref,
            "owner_ref": ticket.owner_ref,
            "purpose": "loop-private-probe",
        }
    ).split(":", 1)[1][:16]
    return PrivateProbeScheduleRecord(
        probe_id=probe_id,
        ticket_id=ticket.ticket_id,
        brief_ref=ticket.brief_ref,
        owner_ref=ticket.owner_ref,
    )


def verify_research_lab_orchestrator(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    foundation_summary = verify_loop_foundation()
    fixture = _load_fixture(Path(fixture_path))

    ticket = TicketRecord.from_mapping(fixture["ticket"])
    _assert(not validate_ticket_record(ticket), "valid ticket passes")

    reservation = plan_budget_reservation(
        ticket,
        requested_loops=fixture["budget_checks"]["valid"]["requested_loops"],
        estimated_node_eval_cents=fixture["budget_checks"]["valid"]["estimated_node_eval_cents"],
        estimated_node_wall_clock_s=fixture["budget_checks"]["valid"]["estimated_node_wall_clock_s"],
        estimated_wall_clock_s=fixture["budget_checks"]["valid"]["estimated_wall_clock_s"],
    )
    _assert(reservation.approved, "valid reservation is approved")
    _assert(reservation.remaining_after_cents >= 0, "reservation never overdraws")

    overspend = plan_budget_reservation(
        ticket,
        requested_loops=fixture["budget_checks"]["overspend"]["requested_loops"],
        estimated_node_eval_cents=fixture["budget_checks"]["overspend"]["estimated_node_eval_cents"],
        estimated_node_wall_clock_s=fixture["budget_checks"]["overspend"]["estimated_node_wall_clock_s"],
        estimated_wall_clock_s=fixture["budget_checks"]["overspend"]["estimated_wall_clock_s"],
    )
    _assert(not overspend.approved, "overspend is rejected")
    _assert(
        "requested loop spend exceeds remaining balance" in overspend.errors,
        "overspend explains remaining balance failure",
    )

    invalid_cap = plan_budget_reservation(
        ticket,
        requested_loops=fixture["budget_checks"]["node_cap"]["requested_loops"],
        estimated_node_eval_cents=fixture["budget_checks"]["node_cap"]["estimated_node_eval_cents"],
        estimated_node_wall_clock_s=fixture["budget_checks"]["node_cap"]["estimated_node_wall_clock_s"],
        estimated_wall_clock_s=fixture["budget_checks"]["node_cap"]["estimated_wall_clock_s"],
    )
    _assert(not invalid_cap.approved, "per-node cap breach is rejected")
    _assert(
        "estimated_node_eval_cents exceeds per_node_eval_cap_cents" in invalid_cap.errors,
        "per-node cap failure is explicit",
    )

    wall_clock_cap = plan_budget_reservation(
        ticket,
        requested_loops=fixture["budget_checks"]["node_wall_clock_cap"]["requested_loops"],
        estimated_node_eval_cents=fixture["budget_checks"]["node_wall_clock_cap"]["estimated_node_eval_cents"],
        estimated_node_wall_clock_s=fixture["budget_checks"]["node_wall_clock_cap"]["estimated_node_wall_clock_s"],
        estimated_wall_clock_s=fixture["budget_checks"]["node_wall_clock_cap"]["estimated_wall_clock_s"],
    )
    _assert(not wall_clock_cap.approved, "per-node wall-clock cap breach is rejected")
    _assert(
        "estimated_node_wall_clock_s exceeds per_node_wall_clock_cap_s" in wall_clock_cap.errors,
        "per-node wall-clock cap failure is explicit",
    )

    max_loops_cap = plan_budget_reservation(
        ticket,
        requested_loops=fixture["budget_checks"]["max_loops_per_day"]["requested_loops"],
        estimated_node_eval_cents=fixture["budget_checks"]["max_loops_per_day"]["estimated_node_eval_cents"],
        estimated_node_wall_clock_s=fixture["budget_checks"]["max_loops_per_day"]["estimated_node_wall_clock_s"],
        estimated_wall_clock_s=fixture["budget_checks"]["max_loops_per_day"]["estimated_wall_clock_s"],
    )
    _assert(not max_loops_cap.approved, "max-loops-per-day cap breach is rejected")
    _assert(
        "requested_loops exceeds max_loops_per_day" in max_loops_cap.errors,
        "max-loops-per-day failure is explicit",
    )

    loop_batch_fixture = fixture["budget_checks"]["loop_batch_cap"]
    batch_cap_ticket_data = dict(fixture["ticket"])
    batch_cap_ticket_data.update(loop_batch_fixture["ticket_overrides"])
    batch_cap_ticket = TicketRecord.from_mapping(batch_cap_ticket_data)
    loop_batch_cap = plan_budget_reservation(
        batch_cap_ticket,
        requested_loops=loop_batch_fixture["requested_loops"],
        estimated_node_eval_cents=loop_batch_fixture["estimated_node_eval_cents"],
        estimated_node_wall_clock_s=loop_batch_fixture["estimated_node_wall_clock_s"],
        estimated_wall_clock_s=loop_batch_fixture["estimated_wall_clock_s"],
    )
    _assert(not loop_batch_cap.approved, "loop-batch spend cap breach is rejected")
    _assert(
        "requested loop spend exceeds loop_batch_cap_cents" in loop_batch_cap.errors,
        "loop-batch spend cap failure is explicit",
    )

    batch_wall_clock_cap = plan_budget_reservation(
        ticket,
        requested_loops=fixture["budget_checks"]["loop_batch_wall_clock_cap"]["requested_loops"],
        estimated_node_eval_cents=fixture["budget_checks"]["loop_batch_wall_clock_cap"]["estimated_node_eval_cents"],
        estimated_node_wall_clock_s=fixture["budget_checks"]["loop_batch_wall_clock_cap"][
            "estimated_node_wall_clock_s"
        ],
        estimated_wall_clock_s=fixture["budget_checks"]["loop_batch_wall_clock_cap"]["estimated_wall_clock_s"],
    )
    _assert(not batch_wall_clock_cap.approved, "loop-batch wall-clock cap breach is rejected")
    _assert(
        "estimated_wall_clock_s exceeds loop_batch_wall_clock_cap_s" in batch_wall_clock_cap.errors,
        "loop-batch wall-clock cap failure is explicit",
    )

    ordered = order_queue_entries(fixture["queue_entries"])
    _assert(
        [entry.ticket_id for entry in ordered] == fixture["expected_queue_ticket_order"],
        "queue order is deterministic",
    )
    _assert(not validate_queue_entries(fixture["queue_entries"]), "queue entries are local-only")
    _assert(validate_queue_entries(fixture["invalid_queue_entries"]), "invalid queue entries fail")

    novelty_reject = evaluate_novelty(
        fixture["briefs"]["reject"],
        fixture["prior_receipts"],
        static_corpus_index_version=fixture["static_corpus_index_version"],
    )
    _assert(novelty_reject.decision == NoveltyDecision.REJECT.value, "near-duplicate brief is rejected")
    _assert(novelty_reject.nearest_prior_receipt_refs, "reject includes prior receipt links")

    novelty_pass = evaluate_novelty(
        fixture["briefs"]["pass"],
        fixture["prior_receipts"],
        static_corpus_index_version=fixture["static_corpus_index_version"],
    )
    _assert(novelty_pass.decision == NoveltyDecision.PASS.value, "distinct brief passes novelty gate")

    probe = build_private_probe_schedule(ticket)
    _assert(probe.lab_only, "probe schedule is lab-only")
    _assert(not probe.scheduler_enabled, "probe scheduler is disabled")
    _assert(not probe.public_workflow_side_effects, "probe has no public side effects")

    try:
        build_private_probe_schedule(ticket, guards=fixture["unsafe_workflow_guards"])
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe workflow guards should block private probe scheduling")

    return {
        "foundation_invalid_release_records": foundation_summary["invalid_release_records"],
        "queue_entries": len(fixture["queue_entries"]),
        "prior_receipts": len(fixture["prior_receipts"]),
        "novelty_reject_receipts": len(novelty_reject.nearest_prior_receipt_refs),
        "probe_id": probe.probe_id,
    }


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
