"""Phase 1 loop-game and autopilot records.

P1.4 models the public failure board, private probes, loop balances, local
escrow semantics, public receipts, and autopilot v0 recommendations. It remains
inert: helpers return local records only and reject enabled Phase 1 workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .canonical import sha256_json
from .orchestrator import (
    BudgetCaps,
    PriorReceipt,
    TicketRecord,
    evaluate_novelty,
    validate_ticket_record,
    verify_research_lab_orchestrator,
)
from .loop_foundation import (
    ArtifactReleaseState,
    LoopWorkflowGuards,
    ReleasePolicyRecord,
    VisibilityPolicy,
    assert_loop_workflows_disabled,
    default_loop_workflow_guards,
    validate_release_policy,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "loop_game_fixtures.json"
GENERALIST_ISLAND = "generalist"


class FailureBoardStatus(str, Enum):
    OPEN = "open"
    PROBING = "probing"
    FUNDED = "funded"
    CLOSED = "closed"


class LoopEscrowState(str, Enum):
    REJECTED = "rejected"
    RESERVED = "reserved"
    COMMITTED = "committed"
    RELEASED = "released"


class AutopilotRecommendationStatus(str, Enum):
    ADVISORY_LOCAL_ONLY = "advisory_local_only"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class FailureBoardItem:
    item_id: str
    island: str
    target_component: str
    target_metric: str
    failure_summary: str
    fixture_refs: tuple[str, ...]
    bounty_tier_ref: str
    allocator_prediction_ref: str
    expected_yield_score: float
    status: str = FailureBoardStatus.OPEN.value
    visibility_policy: str = VisibilityPolicy.SANITIZED_TRACE.value
    artifact_release_state: str = ArtifactReleaseState.SANITIZED_TRACE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FailureBoardItem":
        return cls(
            item_id=str(data["item_id"]),
            island=str(data["island"]),
            target_component=str(data["target_component"]),
            target_metric=str(data["target_metric"]),
            failure_summary=str(data["failure_summary"]),
            fixture_refs=tuple(str(item) for item in data.get("fixture_refs", [])),
            bounty_tier_ref=str(data["bounty_tier_ref"]),
            allocator_prediction_ref=str(data["allocator_prediction_ref"]),
            expected_yield_score=float(data["expected_yield_score"]),
            status=str(data.get("status", FailureBoardStatus.OPEN.value)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.SANITIZED_TRACE.value)),
            artifact_release_state=str(data.get("artifact_release_state", ArtifactReleaseState.SANITIZED_TRACE.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fixture_refs"] = list(self.fixture_refs)
        return data


@dataclass(frozen=True)
class PrivateProbeReceipt:
    receipt_ref: str
    probe_id: str
    ticket_id: str
    miner_ref: str
    brief_ref: str
    island: str
    cost_cents: int
    result_ref: str
    cost_ledger_ref: str
    dev_delta_lcb: float
    private_to_owner_ref: str
    visibility_policy: str = VisibilityPolicy.DEFAULT_PRIVATE.value
    artifact_release_state: str = ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PrivateProbeReceipt":
        return cls(
            receipt_ref=str(data["receipt_ref"]),
            probe_id=str(data["probe_id"]),
            ticket_id=str(data["ticket_id"]),
            miner_ref=str(data["miner_ref"]),
            brief_ref=str(data["brief_ref"]),
            island=str(data["island"]),
            cost_cents=int(data["cost_cents"]),
            result_ref=str(data["result_ref"]),
            cost_ledger_ref=str(data["cost_ledger_ref"]),
            dev_delta_lcb=float(data["dev_delta_lcb"]),
            private_to_owner_ref=str(data["private_to_owner_ref"]),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.DEFAULT_PRIVATE.value)),
            artifact_release_state=str(
                data.get("artifact_release_state", ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoopBalanceRecord:
    balance_id: str
    owner_ref: str
    miner_ref: str
    currency: str
    total_funded_cents: int
    available_cents: int
    reserved_cents: int = 0
    spent_cents: int = 0
    local_only: bool = True
    payments_enabled: bool = False
    settlement_enabled: bool = False
    external_account_ref: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LoopBalanceRecord":
        return cls(
            balance_id=str(data["balance_id"]),
            owner_ref=str(data["owner_ref"]),
            miner_ref=str(data["miner_ref"]),
            currency=str(data.get("currency", "usd_cents")),
            total_funded_cents=int(data["total_funded_cents"]),
            available_cents=int(data["available_cents"]),
            reserved_cents=int(data.get("reserved_cents", 0)),
            spent_cents=int(data.get("spent_cents", 0)),
            local_only=bool(data.get("local_only", True)),
            payments_enabled=bool(data.get("payments_enabled", False)),
            settlement_enabled=bool(data.get("settlement_enabled", False)),
            external_account_ref=str(data.get("external_account_ref", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoopEscrowReservation:
    reservation_id: str
    balance_id: str
    ticket_id: str
    idempotency_key: str
    loop_count: int
    reserved_cents: int
    state: str
    committed_spend_cents: int = 0
    released_cents: int = 0
    cost_ledger_ref: str = ""
    local_only: bool = True
    atomic_reserve_commit: bool = True
    external_debit_applied: bool = False
    payment_processor_charge_ref: str = ""
    errors: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LoopEscrowReservation":
        return cls(
            reservation_id=str(data["reservation_id"]),
            balance_id=str(data["balance_id"]),
            ticket_id=str(data["ticket_id"]),
            idempotency_key=str(data["idempotency_key"]),
            loop_count=int(data["loop_count"]),
            reserved_cents=int(data["reserved_cents"]),
            state=str(data["state"]),
            committed_spend_cents=int(data.get("committed_spend_cents", 0)),
            released_cents=int(data.get("released_cents", 0)),
            cost_ledger_ref=str(data.get("cost_ledger_ref", "")),
            local_only=bool(data.get("local_only", True)),
            atomic_reserve_commit=bool(data.get("atomic_reserve_commit", True)),
            external_debit_applied=bool(data.get("external_debit_applied", False)),
            payment_processor_charge_ref=str(data.get("payment_processor_charge_ref", "")),
            errors=tuple(str(item) for item in data.get("errors", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["errors"] = list(self.errors)
        return data


@dataclass(frozen=True)
class LoopReceiptRecord:
    receipt_ref: str
    ticket_id: str
    miner_ref: str
    brief_ref: str
    island: str
    summary_text: str
    stopped_reason: str
    loops_consumed: int
    dev_delta_trajectory: tuple[float, ...]
    best_dev_delta_lcb: float
    cost_ledger_ref: str
    result_ref: str
    proof_refs: tuple[str, ...]
    public_trace_ref: str
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.PUBLIC_RECEIPT.value
    artifact_release_state: str = ArtifactReleaseState.PUBLIC_RECEIPT.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LoopReceiptRecord":
        return cls(
            receipt_ref=str(data["receipt_ref"]),
            ticket_id=str(data["ticket_id"]),
            miner_ref=str(data["miner_ref"]),
            brief_ref=str(data["brief_ref"]),
            island=str(data["island"]),
            summary_text=str(data["summary_text"]),
            stopped_reason=str(data["stopped_reason"]),
            loops_consumed=int(data["loops_consumed"]),
            dev_delta_trajectory=tuple(float(item) for item in data.get("dev_delta_trajectory", [])),
            best_dev_delta_lcb=float(data["best_dev_delta_lcb"]),
            cost_ledger_ref=str(data["cost_ledger_ref"]),
            result_ref=str(data["result_ref"]),
            proof_refs=tuple(str(item) for item in data.get("proof_refs", [])),
            public_trace_ref=str(data["public_trace_ref"]),
            contains_live_champion_ip=bool(data.get("contains_live_champion_ip", False)),
            contains_sealed_eval_details=bool(data.get("contains_sealed_eval_details", False)),
            contains_raw_evidence_snapshot=bool(data.get("contains_raw_evidence_snapshot", False)),
            contains_private_customer_data=bool(data.get("contains_private_customer_data", False)),
            contains_judge_prompts=bool(data.get("contains_judge_prompts", False)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.PUBLIC_RECEIPT.value)),
            artifact_release_state=str(data.get("artifact_release_state", ArtifactReleaseState.PUBLIC_RECEIPT.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dev_delta_trajectory"] = list(self.dev_delta_trajectory)
        data["proof_refs"] = list(self.proof_refs)
        return data

    def public_payload(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "miner_ref": self.miner_ref,
            "brief_ref": self.brief_ref,
            "island": self.island,
            "summary_text": self.summary_text,
            "stopped_reason": self.stopped_reason,
            "loops_consumed": self.loops_consumed,
            "dev_delta_trajectory": list(self.dev_delta_trajectory),
            "best_dev_delta_lcb": self.best_dev_delta_lcb,
            "cost_ledger_ref": self.cost_ledger_ref,
            "result_ref": self.result_ref,
            "proof_refs": list(self.proof_refs),
            "public_trace_ref": self.public_trace_ref,
        }


@dataclass(frozen=True)
class AutopilotRecommendationRecord:
    recommendation_id: str
    island: str
    failure_board_item_id: str
    aiming_rule: str
    expected_yield_score: float
    recommended_loop_count: int
    budget_cents: int
    status: str = AutopilotRecommendationStatus.ADVISORY_LOCAL_ONLY.value
    advisory_only: bool = True
    autopilot_enabled: bool = False
    funding_action_enabled: bool = False
    public_workflow_side_effects: bool = False
    visibility_policy: str = VisibilityPolicy.DEFAULT_PRIVATE.value
    artifact_release_state: str = ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AutopilotRecommendationRecord":
        return cls(
            recommendation_id=str(data["recommendation_id"]),
            island=str(data["island"]),
            failure_board_item_id=str(data["failure_board_item_id"]),
            aiming_rule=str(data["aiming_rule"]),
            expected_yield_score=float(data["expected_yield_score"]),
            recommended_loop_count=int(data["recommended_loop_count"]),
            budget_cents=int(data["budget_cents"]),
            status=str(data.get("status", AutopilotRecommendationStatus.ADVISORY_LOCAL_ONLY.value)),
            advisory_only=bool(data.get("advisory_only", True)),
            autopilot_enabled=bool(data.get("autopilot_enabled", False)),
            funding_action_enabled=bool(data.get("funding_action_enabled", False)),
            public_workflow_side_effects=bool(data.get("public_workflow_side_effects", False)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.DEFAULT_PRIVATE.value)),
            artifact_release_state=str(
                data.get("artifact_release_state", ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_failure_board_item(item: FailureBoardItem | Mapping[str, Any]) -> list[str]:
    if not isinstance(item, FailureBoardItem):
        item = FailureBoardItem.from_mapping(item)
    errors: list[str] = []
    if item.island != GENERALIST_ISLAND:
        errors.append("P1.4 failure board is limited to the generalist island")
    if item.status not in {status.value for status in FailureBoardStatus}:
        errors.append(f"unknown failure-board status: {item.status}")
    if not item.failure_summary:
        errors.append("failure_summary is required")
    if not item.fixture_refs:
        errors.append("fixture_refs must not be empty")
    if not item.bounty_tier_ref:
        errors.append("bounty_tier_ref is required")
    if item.expected_yield_score < 0:
        errors.append("expected_yield_score must be non-negative")
    errors.extend(
        validate_release_policy(
            ReleasePolicyRecord(
                artifact_ref=f"failure_board:{item.item_id}",
                artifact_type="map_projection",
                visibility_policy=item.visibility_policy,
                artifact_release_state=item.artifact_release_state,
                reason="failure board exposes sanitized map cells only",
            )
        )
    )
    return errors


def validate_private_probe_receipt(receipt: PrivateProbeReceipt | Mapping[str, Any]) -> list[str]:
    if not isinstance(receipt, PrivateProbeReceipt):
        receipt = PrivateProbeReceipt.from_mapping(receipt)
    errors: list[str] = []
    if receipt.island != GENERALIST_ISLAND:
        errors.append("P1.4 private probes are limited to the generalist island")
    if receipt.cost_cents < 0:
        errors.append("cost_cents must be non-negative")
    if not receipt.result_ref:
        errors.append("result_ref is required")
    if not receipt.cost_ledger_ref:
        errors.append("cost_ledger_ref is required")
    if not receipt.private_to_owner_ref:
        errors.append("private_to_owner_ref is required")
    errors.extend(
        validate_release_policy(
            ReleasePolicyRecord(
                artifact_ref=receipt.receipt_ref,
                artifact_type="private_probe_receipt",
                visibility_policy=receipt.visibility_policy,
                artifact_release_state=receipt.artifact_release_state,
                reason="private probes must not become public receipts",
            )
        )
    )
    if receipt.artifact_release_state != ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value:
        errors.append("private probe receipt must remain private")
    return errors


def validate_loop_balance(balance: LoopBalanceRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(balance, LoopBalanceRecord):
        balance = LoopBalanceRecord.from_mapping(balance)
    errors: list[str] = []
    if balance.currency != "usd_cents":
        errors.append("P1.4 loop balances must be denominated in usd_cents")
    for field in ("total_funded_cents", "available_cents", "reserved_cents", "spent_cents"):
        if getattr(balance, field) < 0:
            errors.append(f"{field} must be non-negative")
    if balance.available_cents + balance.reserved_cents + balance.spent_cents != balance.total_funded_cents:
        errors.append("available + reserved + spent must equal total_funded_cents")
    if not balance.local_only:
        errors.append("loop balance must remain local_only")
    if balance.payments_enabled:
        errors.append("payments_enabled must remain false")
    if balance.settlement_enabled:
        errors.append("settlement_enabled must remain false")
    if balance.external_account_ref:
        errors.append("external_account_ref must remain empty")
    return errors


def validate_loop_escrow_reservation(reservation: LoopEscrowReservation | Mapping[str, Any]) -> list[str]:
    if not isinstance(reservation, LoopEscrowReservation):
        reservation = LoopEscrowReservation.from_mapping(reservation)
    errors: list[str] = []
    if reservation.state not in {state.value for state in LoopEscrowState}:
        errors.append(f"unknown escrow state: {reservation.state}")
    for field in ("loop_count", "reserved_cents", "committed_spend_cents", "released_cents"):
        if getattr(reservation, field) < 0:
            errors.append(f"{field} must be non-negative")
    if reservation.loop_count <= 0:
        errors.append("loop_count must be positive")
    if not reservation.local_only:
        errors.append("escrow reservation must remain local_only")
    if not reservation.atomic_reserve_commit:
        errors.append("escrow reservation must use atomic reserve/commit semantics")
    if reservation.external_debit_applied:
        errors.append("external_debit_applied must remain false")
    if reservation.payment_processor_charge_ref:
        errors.append("payment_processor_charge_ref must remain empty")
    if reservation.state == LoopEscrowState.COMMITTED.value:
        if not reservation.cost_ledger_ref:
            errors.append("committed escrow requires cost_ledger_ref")
        if reservation.committed_spend_cents + reservation.released_cents != reservation.reserved_cents:
            errors.append("committed_spend_cents + released_cents must equal reserved_cents")
    return errors


def reserve_loop_balance(
    balance: LoopBalanceRecord | Mapping[str, Any],
    ticket: TicketRecord | Mapping[str, Any],
    *,
    requested_loops: int,
    idempotency_key: str,
    guards: Optional[LoopWorkflowGuards | Mapping[str, Any]] = None,
) -> tuple[LoopBalanceRecord, LoopEscrowReservation]:
    """Reserve local loop balance or return a rejected reservation.

    Reserve decline is a normal purchase-time outcome: insufficient balance or
    cap failure leaves the balance unchanged and returns a REJECTED record.
    Commit failures are harder state errors and raise instead.
    """
    assert_loop_workflows_disabled(guards or default_loop_workflow_guards())
    if not isinstance(balance, LoopBalanceRecord):
        balance = LoopBalanceRecord.from_mapping(balance)
    if not isinstance(ticket, TicketRecord):
        ticket = TicketRecord.from_mapping(ticket)

    errors = validate_loop_balance(balance) + validate_ticket_record(ticket)
    caps = ticket.budget_caps
    reserved_cents = requested_loops * caps.loop_unit_cents
    if requested_loops <= 0:
        errors.append("requested_loops must be positive")
    if requested_loops > caps.max_loops_per_day:
        errors.append("requested_loops exceeds max_loops_per_day")
    if reserved_cents > caps.loop_batch_cap_cents:
        errors.append("requested loop spend exceeds loop_batch_cap_cents")
    if reserved_cents > balance.available_cents:
        errors.append("requested loop spend exceeds available balance")

    reservation_id = _escrow_id(balance.balance_id, ticket.ticket_id, idempotency_key, requested_loops)
    if errors:
        return balance, LoopEscrowReservation(
            reservation_id=reservation_id,
            balance_id=balance.balance_id,
            ticket_id=ticket.ticket_id,
            idempotency_key=idempotency_key,
            loop_count=max(1, requested_loops),
            reserved_cents=max(0, reserved_cents),
            state=LoopEscrowState.REJECTED.value,
            errors=tuple(errors),
        )

    next_balance = LoopBalanceRecord(
        **{
            **balance.to_dict(),
            "available_cents": balance.available_cents - reserved_cents,
            "reserved_cents": balance.reserved_cents + reserved_cents,
        }
    )
    reservation = LoopEscrowReservation(
        reservation_id=reservation_id,
        balance_id=balance.balance_id,
        ticket_id=ticket.ticket_id,
        idempotency_key=idempotency_key,
        loop_count=requested_loops,
        reserved_cents=reserved_cents,
        state=LoopEscrowState.RESERVED.value,
    )
    return next_balance, reservation


def commit_loop_escrow(
    balance: LoopBalanceRecord | Mapping[str, Any],
    reservation: LoopEscrowReservation | Mapping[str, Any],
    *,
    actual_spend_cents: int,
    cost_ledger_ref: str,
    guards: Optional[LoopWorkflowGuards | Mapping[str, Any]] = None,
) -> tuple[LoopBalanceRecord, LoopEscrowReservation]:
    """Commit reserved local spend and release any unspent reserve.

    Unlike reserve declines, commit failures indicate an invalid reserved-state
    transition, so the helper raises without mutating the balance.
    """
    assert_loop_workflows_disabled(guards or default_loop_workflow_guards())
    if not isinstance(balance, LoopBalanceRecord):
        balance = LoopBalanceRecord.from_mapping(balance)
    if not isinstance(reservation, LoopEscrowReservation):
        reservation = LoopEscrowReservation.from_mapping(reservation)

    errors = validate_loop_balance(balance) + validate_loop_escrow_reservation(reservation)
    if reservation.state != LoopEscrowState.RESERVED.value:
        errors.append("only reserved escrow can be committed")
    if actual_spend_cents < 0:
        errors.append("actual_spend_cents must be non-negative")
    if actual_spend_cents > reservation.reserved_cents:
        errors.append("actual_spend_cents cannot exceed reserved_cents")
    if reservation.reserved_cents > balance.reserved_cents:
        errors.append("reservation exceeds balance reserved_cents")
    if not cost_ledger_ref:
        errors.append("cost_ledger_ref is required")
    if errors:
        raise ValueError("; ".join(errors))

    released_cents = reservation.reserved_cents - actual_spend_cents
    next_balance = LoopBalanceRecord(
        **{
            **balance.to_dict(),
            "available_cents": balance.available_cents + released_cents,
            "reserved_cents": balance.reserved_cents - reservation.reserved_cents,
            "spent_cents": balance.spent_cents + actual_spend_cents,
        }
    )
    committed = LoopEscrowReservation(
        **{
            **reservation.to_dict(),
            "state": LoopEscrowState.COMMITTED.value,
            "committed_spend_cents": actual_spend_cents,
            "released_cents": released_cents,
            "cost_ledger_ref": cost_ledger_ref,
            "errors": (),
        }
    )
    return next_balance, committed


def build_public_loop_receipt(source: Mapping[str, Any]) -> LoopReceiptRecord:
    flags = _protected_material_flags(source)
    provisional = LoopReceiptRecord(
        receipt_ref="receipt:pending",
        ticket_id=str(source["ticket_id"]),
        miner_ref=str(source["miner_ref"]),
        brief_ref=str(source["brief_ref"]),
        island=str(source["island"]),
        summary_text=str(source["summary_text"]),
        stopped_reason=str(source["stopped_reason"]),
        loops_consumed=int(source["loops_consumed"]),
        dev_delta_trajectory=tuple(float(item) for item in source.get("dev_delta_trajectory", [])),
        best_dev_delta_lcb=float(source["best_dev_delta_lcb"]),
        cost_ledger_ref=str(source["cost_ledger_ref"]),
        result_ref=str(source["result_ref"]),
        proof_refs=tuple(str(item) for item in source.get("proof_refs", [])),
        public_trace_ref=str(source["public_trace_ref"]),
        **flags,
    )
    _assert_receipt_release_policy(provisional)
    receipt_ref = "receipt:" + sha256_json(provisional.public_payload())
    receipt = LoopReceiptRecord(**{**provisional.to_dict(), "receipt_ref": receipt_ref})
    errors = validate_loop_receipt(receipt)
    if errors:
        raise ValueError("; ".join(errors))
    return receipt


def validate_loop_receipt(receipt: LoopReceiptRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(receipt, LoopReceiptRecord):
        receipt = LoopReceiptRecord.from_mapping(receipt)
    errors: list[str] = []
    if receipt.island != GENERALIST_ISLAND:
        errors.append("P1.4 loop receipts are limited to the generalist island")
    if not receipt.receipt_ref.startswith("receipt:sha256:"):
        errors.append("receipt_ref must be hash-addressed")
    else:
        expected_ref = "receipt:" + sha256_json(receipt.public_payload())
        if receipt.receipt_ref != expected_ref:
            errors.append("receipt_ref does not match canonical public payload hash")
    if receipt.loops_consumed <= 0:
        errors.append("loops_consumed must be positive")
    if not receipt.dev_delta_trajectory:
        errors.append("dev_delta_trajectory must not be empty")
    if not receipt.cost_ledger_ref:
        errors.append("cost_ledger_ref is required")
    if not receipt.result_ref:
        errors.append("result_ref is required")
    if not receipt.proof_refs:
        errors.append("proof_refs must not be empty")
    if not receipt.public_trace_ref:
        errors.append("public_trace_ref is required")
    try:
        _assert_receipt_release_policy(receipt)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def receipt_to_prior_receipt(receipt: LoopReceiptRecord | Mapping[str, Any]) -> PriorReceipt:
    if not isinstance(receipt, LoopReceiptRecord):
        receipt = LoopReceiptRecord.from_mapping(receipt)
    errors = validate_loop_receipt(receipt)
    if errors:
        raise ValueError("; ".join(errors))
    return PriorReceipt(
        receipt_ref=receipt.receipt_ref,
        brief_ref=receipt.brief_ref,
        summary_text=receipt.summary_text,
    )


def build_autopilot_recommendation(
    items: Sequence[FailureBoardItem | Mapping[str, Any]],
    *,
    budget_caps: BudgetCaps | Mapping[str, Any] | None = None,
    guards: Optional[LoopWorkflowGuards | Mapping[str, Any]] = None,
) -> AutopilotRecommendationRecord:
    assert_loop_workflows_disabled(guards or default_loop_workflow_guards())
    caps = budget_caps if isinstance(budget_caps, BudgetCaps) else BudgetCaps.from_mapping(budget_caps)
    normalized = [item if isinstance(item, FailureBoardItem) else FailureBoardItem.from_mapping(item) for item in items]
    valid_open = [
        item
        for item in normalized
        if not validate_failure_board_item(item)
        and item.status == FailureBoardStatus.OPEN.value
        and item.island == GENERALIST_ISLAND
    ]
    if not valid_open:
        raise ValueError("autopilot requires at least one open generalist failure-board item")
    selected = sorted(valid_open, key=lambda item: (-item.expected_yield_score, item.item_id))[0]
    payload = {
        "island": GENERALIST_ISLAND,
        "failure_board_item_id": selected.item_id,
        "aiming_rule": "highest expected-yield generalist failure-board cell",
        "expected_yield_score": selected.expected_yield_score,
        "recommended_loop_count": 1,
        "budget_cents": caps.loop_unit_cents,
    }
    recommendation = AutopilotRecommendationRecord(
        recommendation_id="autopilot:" + sha256_json(payload).split(":", 1)[1][:16],
        **payload,
    )
    errors = validate_autopilot_recommendation(recommendation)
    if errors:
        raise ValueError("; ".join(errors))
    return recommendation


def validate_autopilot_recommendation(
    record: AutopilotRecommendationRecord | Mapping[str, Any],
) -> list[str]:
    if not isinstance(record, AutopilotRecommendationRecord):
        record = AutopilotRecommendationRecord.from_mapping(record)
    errors: list[str] = []
    if record.island != GENERALIST_ISLAND:
        errors.append("autopilot v0 is limited to the generalist island")
    if record.status not in {status.value for status in AutopilotRecommendationStatus}:
        errors.append(f"unknown autopilot recommendation status: {record.status}")
    if not record.advisory_only:
        errors.append("autopilot recommendation must remain advisory_only")
    if record.autopilot_enabled:
        errors.append("autopilot_enabled must remain false")
    if record.funding_action_enabled:
        errors.append("funding_action_enabled must remain false")
    if record.public_workflow_side_effects:
        errors.append("public_workflow_side_effects must remain false")
    if record.recommended_loop_count <= 0:
        errors.append("recommended_loop_count must be positive")
    if record.budget_cents <= 0:
        errors.append("budget_cents must be positive")
    errors.extend(
        validate_release_policy(
            ReleasePolicyRecord(
                artifact_ref=record.recommendation_id,
                artifact_type="autopilot_recommendation",
                visibility_policy=record.visibility_policy,
                artifact_release_state=record.artifact_release_state,
                reason="autopilot v0 is an advisory private artifact in P1.4",
            )
        )
    )
    return errors


def verify_research_lab_loop_game(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    orchestrator_summary = verify_research_lab_orchestrator()
    fixture = _load_fixture(Path(fixture_path))

    board_items = [FailureBoardItem.from_mapping(item) for item in fixture["failure_board_items"]]
    for item in board_items:
        _assert(not validate_failure_board_item(item), f"valid failure-board item passes: {item.item_id}")
    for item in fixture["invalid_failure_board_items"]:
        _assert(validate_failure_board_item(item), f"invalid failure-board item fails: {item['item_id']}")

    for receipt in fixture["private_probe_receipts"]:
        _assert(not validate_private_probe_receipt(receipt), f"valid probe receipt passes: {receipt['receipt_ref']}")
    for receipt in fixture["invalid_private_probe_receipts"]:
        _assert(validate_private_probe_receipt(receipt), f"invalid probe receipt fails: {receipt['receipt_ref']}")

    balance = LoopBalanceRecord.from_mapping(fixture["loop_balance"])
    ticket = TicketRecord.from_mapping(fixture["ticket"])
    _assert(not validate_loop_balance(balance), "loop balance validates")

    reserved_balance, reservation = reserve_loop_balance(
        balance,
        ticket,
        requested_loops=fixture["escrow"]["requested_loops"],
        idempotency_key=fixture["escrow"]["idempotency_key"],
    )
    _assert(reservation.state == LoopEscrowState.RESERVED.value, "escrow reserve succeeds")
    _assert(not validate_loop_balance(reserved_balance), "reserved balance validates")
    _assert(not validate_loop_escrow_reservation(reservation), "reserved escrow validates")

    committed_balance, committed = commit_loop_escrow(
        reserved_balance,
        reservation,
        actual_spend_cents=fixture["escrow"]["actual_spend_cents"],
        cost_ledger_ref=fixture["escrow"]["cost_ledger_ref"],
    )
    _assert(committed.state == LoopEscrowState.COMMITTED.value, "escrow commit succeeds")
    _assert(committed.released_cents == fixture["escrow"]["expected_released_cents"], "unspent reserve releases")
    _assert(not validate_loop_balance(committed_balance), "committed balance validates")
    _assert(not validate_loop_escrow_reservation(committed), "committed escrow validates")

    low_balance = LoopBalanceRecord.from_mapping(fixture["low_loop_balance"])
    _, rejected = reserve_loop_balance(
        low_balance,
        ticket,
        requested_loops=fixture["escrow"]["requested_loops"],
        idempotency_key="reserve-too-much",
    )
    _assert(rejected.state == LoopEscrowState.REJECTED.value, "insufficient local balance rejects")
    _assert("requested loop spend exceeds available balance" in rejected.errors, "reject explains balance failure")

    try:
        reserve_loop_balance(
            balance,
            ticket,
            requested_loops=1,
            idempotency_key="unsafe-guard",
            guards=fixture["unsafe_workflow_guards"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe workflow guards should block loop reserve")

    try:
        commit_loop_escrow(
            reserved_balance,
            reservation,
            actual_spend_cents=100,
            cost_ledger_ref="cost_ledger:p1.4:unsafe-guard",
            guards=fixture["unsafe_workflow_guards"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe workflow guards should block loop commit")

    receipt = build_public_loop_receipt(fixture["public_loop_receipt_source"])
    _assert(not validate_loop_receipt(receipt), "public loop receipt validates")
    _assert(receipt.receipt_ref.startswith("receipt:sha256:"), "receipt is hash-addressed")
    prior = receipt_to_prior_receipt(receipt)
    novelty = evaluate_novelty(
        fixture["public_loop_receipt_source"]["summary_text"],
        [prior],
        static_corpus_index_version="static-corpus:p1.4:receipt-compat",
    )
    _assert(receipt.receipt_ref in novelty.nearest_prior_receipt_refs, "receipt is novelty-gate compatible")

    try:
        build_public_loop_receipt(fixture["protected_loop_receipt_source"])
    except ValueError as exc:
        _assert("protected material" in str(exc), "protected public receipt is rejected")
    else:
        raise AssertionError("protected public receipt source should fail")

    autopilot = build_autopilot_recommendation(board_items)
    _assert(not validate_autopilot_recommendation(autopilot), "autopilot recommendation validates")
    _assert(autopilot.failure_board_item_id == fixture["expected_autopilot_item_id"], "autopilot picks top yield")
    for record in fixture["invalid_autopilot_recommendations"]:
        _assert(validate_autopilot_recommendation(record), f"invalid autopilot fails: {record['recommendation_id']}")
    try:
        build_autopilot_recommendation(board_items, guards=fixture["unsafe_workflow_guards"])
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe workflow guards should block autopilot recommendation")

    return {
        "orchestrator_queue_entries": orchestrator_summary["queue_entries"],
        "failure_board_items": len(board_items),
        "private_probe_receipts": len(fixture["private_probe_receipts"]),
        "receipt_ref": receipt.receipt_ref,
        "autopilot_item_id": autopilot.failure_board_item_id,
        "committed_spend_cents": committed.committed_spend_cents,
        "released_cents": committed.released_cents,
    }


def _assert_receipt_release_policy(receipt: LoopReceiptRecord) -> None:
    errors = validate_release_policy(
        ReleasePolicyRecord(
            artifact_ref=receipt.receipt_ref,
            artifact_type="receipt",
            visibility_policy=receipt.visibility_policy,
            artifact_release_state=receipt.artifact_release_state,
            contains_live_champion_ip=receipt.contains_live_champion_ip,
            contains_sealed_eval_details=receipt.contains_sealed_eval_details,
            contains_raw_evidence_snapshot=receipt.contains_raw_evidence_snapshot,
            contains_private_customer_data=receipt.contains_private_customer_data,
            contains_judge_prompts=receipt.contains_judge_prompts,
            reason="public receipts expose proof and summary fields only",
        )
    )
    if errors:
        raise ValueError("; ".join(errors))


def _protected_material_flags(source: Mapping[str, Any]) -> dict[str, bool]:
    # Keep this source-key map in sync as the real loop-run source schema grows.
    return {
        "contains_live_champion_ip": bool(source.get("contains_live_champion_ip") or source.get("live_champion_ref")),
        "contains_sealed_eval_details": bool(
            source.get("contains_sealed_eval_details") or source.get("sealed_eval_details_ref")
        ),
        "contains_raw_evidence_snapshot": bool(
            source.get("contains_raw_evidence_snapshot") or source.get("raw_evidence_snapshot_ref")
        ),
        "contains_private_customer_data": bool(
            source.get("contains_private_customer_data") or source.get("private_customer_ref")
        ),
        "contains_judge_prompts": bool(source.get("contains_judge_prompts") or source.get("judge_prompt_ref")),
    }


def _escrow_id(balance_id: str, ticket_id: str, idempotency_key: str, requested_loops: int) -> str:
    return "escrow:" + sha256_json(
        {
            "balance_id": balance_id,
            "ticket_id": ticket_id,
            "idempotency_key": idempotency_key,
            "requested_loops": requested_loops,
        }
    ).split(":", 1)[1][:16]


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
