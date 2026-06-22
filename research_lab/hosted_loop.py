"""Local/staged Hosted Research Loop MVP.

This module composes the existing Research Lab local contracts into the first
hosted-loop shape: loop-start contract, ticket/probe, balance ledger, queue,
sandboxed Engine v1 typed-patch execution, receipt, trajectory, trace, and map
projection. It is disabled by default and has no gateway, Supabase, validator,
or fulfillment side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence
import uuid

from .canonical import sha256_json
from .engine_v1 import (
    ENGINE_V1_ENABLED_PATCH_TYPES,
    ComponentRegistry,
    HypothesisRecord,
    PatchRecord,
    validate_hypothesis,
    validate_patch,
)
from .evidence import build_evidence_bundle, evidence_refs_from_bundle
from .fabric import (
    ResearchSandboxJobRecord,
    SandboxJobStatus,
    SandboxResourceCaps,
    validate_sandbox_job_record,
)
from .loop_foundation import (
    ArtifactReleaseState,
    LoopWorkflowGuards,
    VisibilityPolicy,
    assert_loop_workflows_disabled,
    default_loop_workflow_guards,
)
from .loop_game import (
    GENERALIST_ISLAND,
    LoopBalanceRecord,
    LoopEscrowReservation,
    LoopReceiptRecord,
    PrivateProbeReceipt,
    build_public_loop_receipt,
    commit_loop_escrow,
    reserve_loop_balance,
    validate_loop_balance,
    validate_loop_escrow_reservation,
    validate_loop_receipt,
    validate_private_probe_receipt,
)
from .loop_start_contract import (
    LoopStartDecision,
    LoopStartPaymentEvidence,
    LoopStartPolicy,
    MinerOpenRouterKeyReference,
    ResearchLoopStartRequest,
    evaluate_loop_start_request,
)
from .notary import LocalSnapshotStore, NotarySigner, capture_snapshot
from .orchestrator import (
    BudgetCaps,
    PrivateProbeScheduleRecord,
    QueueEntry,
    TicketRecord,
    build_private_probe_schedule,
    order_queue_entries,
    validate_queue_entries,
    validate_ticket_record,
)
from .research_map import (
    AllocatorPredictionRecord,
    ComponentMapStatus,
    ComponentStatusRecord,
    FrontierPointRecord,
    FrontierSummaryRecord,
    ResearchMapCellRecord,
    ResearchMapProjectionRecord,
    build_research_map_projection,
    validate_research_map_projection_record,
)
from .schema_validation import validate_schema_record
from .trace import build_execution_trace, make_trace_call


HOSTED_LOOP_MVP_VERSION = "hosted-loop-mvp-local-v0.1.0"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "hosted_loop_fixtures.json"
TRUTHY_VALUES = {"1", "true", "yes", "on"}

FORBIDDEN_SECRET_KEYS = {
    "api_key",
    "openrouter_api_key",
    "openrouter_key",
    "raw_key",
    "raw_secret",
    "token",
    "credential",
    "credential_value",
}
FORBIDDEN_SECRET_MARKERS = (
    "sk-or-",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
)


@dataclass(frozen=True)
class HostedLoopRuntimeFlags:
    local_execution_enabled: bool = False
    production_paid_loops_enabled: bool = False
    gateway_api_enabled: bool = False
    supabase_writes_enabled: bool = False
    scheduler_enabled: bool = False
    live_crowning_enabled: bool = False
    validator_weight_writes_enabled: bool = False
    fulfillment_touch_enabled: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "HostedLoopRuntimeFlags":
        data = data or {}
        return cls(
            local_execution_enabled=bool(data.get("local_execution_enabled", False)),
            production_paid_loops_enabled=bool(data.get("production_paid_loops_enabled", False)),
            gateway_api_enabled=bool(data.get("gateway_api_enabled", False)),
            supabase_writes_enabled=bool(data.get("supabase_writes_enabled", False)),
            scheduler_enabled=bool(data.get("scheduler_enabled", False)),
            live_crowning_enabled=bool(data.get("live_crowning_enabled", False)),
            validator_weight_writes_enabled=bool(data.get("validator_weight_writes_enabled", False)),
            fulfillment_touch_enabled=bool(data.get("fulfillment_touch_enabled", False)),
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "HostedLoopRuntimeFlags":
        env = env or {}
        return cls(
            local_execution_enabled=_truthy(env.get("RESEARCH_LAB_HOSTED_LOOP_LOCAL_EXECUTION")),
            production_paid_loops_enabled=_truthy(env.get("RESEARCH_LAB_HOSTED_LOOP_PRODUCTION_PAID")),
            gateway_api_enabled=_truthy(env.get("RESEARCH_LAB_HOSTED_LOOP_GATEWAY_API")),
            supabase_writes_enabled=_truthy(env.get("RESEARCH_LAB_HOSTED_LOOP_SUPABASE_WRITES")),
            scheduler_enabled=_truthy(env.get("RESEARCH_LAB_HOSTED_LOOP_SCHEDULER")),
            live_crowning_enabled=_truthy(env.get("RESEARCH_LAB_HOSTED_LOOP_LIVE_CROWNING")),
            validator_weight_writes_enabled=_truthy(env.get("RESEARCH_LAB_HOSTED_LOOP_WEIGHT_WRITES")),
            fulfillment_touch_enabled=_truthy(env.get("RESEARCH_LAB_HOSTED_LOOP_FULFILLMENT_TOUCH")),
        )

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class HostedLoopBalanceLedgerEntry:
    ledger_entry_id: str
    seq: int
    ticket_id: str
    balance_id: str
    miner_hotkey: str
    event_type: str
    amount_cents: int
    available_cents: int
    reserved_cents: int
    spent_cents: int
    source_ref: str
    local_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HostedRunQueueEntry:
    queue_id: str
    run_id: str
    ticket_id: str
    loop_start_decision_id: str
    reservation_id: str
    priority_score: int
    created_seq: int
    status: str = "local_only_queued"
    local_only: bool = True
    scheduler_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_orchestrator_queue_entry(self) -> QueueEntry:
        return QueueEntry(
            queue_id=self.queue_id,
            ticket_id=self.ticket_id,
            priority_score=self.priority_score,
            created_seq=self.created_seq,
            lab_only=self.local_only,
            scheduler_enabled=self.scheduler_enabled,
            status=self.status,
        )


@dataclass(frozen=True)
class HostedPatchRun:
    node_id: str
    hypothesis: HypothesisRecord
    patch: PatchRecord
    targeted_metric: str
    score_bundle: dict[str, Any]
    score_result: dict[str, Any]
    reflection: dict[str, Any]
    model_used: str
    tokens_in: int
    tokens_out: int
    draft_cost_cents: int
    eval_cost_cents: int
    reflect_cost_cents: int
    fixture_refs: tuple[str, ...]

    @property
    def total_cost_cents(self) -> int:
        return self.draft_cost_cents + self.eval_cost_cents + self.reflect_cost_cents

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "hypothesis": self.hypothesis.to_dict(),
            "patch": self.patch.to_dict(),
            "targeted_metric": self.targeted_metric,
            "score_bundle": dict(self.score_bundle),
            "score_result": dict(self.score_result),
            "reflection": dict(self.reflection),
            "model_used": self.model_used,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "draft_cost_cents": self.draft_cost_cents,
            "eval_cost_cents": self.eval_cost_cents,
            "reflect_cost_cents": self.reflect_cost_cents,
            "fixture_refs": list(self.fixture_refs),
        }


@dataclass(frozen=True)
class HostedSandboxResult:
    run_id: str
    sandbox_job: ResearchSandboxJobRecord
    patch_runs: tuple[HostedPatchRun, ...]
    evidence_bundle: dict[str, Any]
    execution_trace: dict[str, Any]
    provider_usage: tuple[dict[str, Any], ...]
    cost_ledger_ref: str
    actual_spend_cents: int
    best_node_id: str
    best_dev_delta_lcb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sandbox_job": self.sandbox_job.to_dict(),
            "patch_runs": [run.to_dict() for run in self.patch_runs],
            "evidence_bundle": self.evidence_bundle,
            "execution_trace": self.execution_trace,
            "provider_usage": [dict(item) for item in self.provider_usage],
            "cost_ledger_ref": self.cost_ledger_ref,
            "actual_spend_cents": self.actual_spend_cents,
            "best_node_id": self.best_node_id,
            "best_dev_delta_lcb": self.best_dev_delta_lcb,
        }


@dataclass(frozen=True)
class HostedLoopRunResult:
    loop_start_decision: LoopStartDecision
    ticket: TicketRecord
    probe_schedule: PrivateProbeScheduleRecord
    probe_receipt: PrivateProbeReceipt
    initial_balance: LoopBalanceRecord
    reserved_balance: LoopBalanceRecord
    final_balance: LoopBalanceRecord
    reservation: LoopEscrowReservation
    committed_reservation: LoopEscrowReservation
    balance_ledger: tuple[HostedLoopBalanceLedgerEntry, ...]
    queue_entry: HostedRunQueueEntry
    sandbox: HostedSandboxResult
    trajectory: dict[str, Any]
    results_ledger_rows: tuple[dict[str, Any], ...]
    receipt: LoopReceiptRecord
    research_map_projection: ResearchMapProjectionRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop_start_decision": self.loop_start_decision.to_dict(),
            "ticket": self.ticket.to_dict(),
            "probe_schedule": self.probe_schedule.to_dict(),
            "probe_receipt": self.probe_receipt.to_dict(),
            "initial_balance": self.initial_balance.to_dict(),
            "reserved_balance": self.reserved_balance.to_dict(),
            "final_balance": self.final_balance.to_dict(),
            "reservation": self.reservation.to_dict(),
            "committed_reservation": self.committed_reservation.to_dict(),
            "balance_ledger": [entry.to_dict() for entry in self.balance_ledger],
            "queue_entry": self.queue_entry.to_dict(),
            "sandbox": self.sandbox.to_dict(),
            "trajectory": self.trajectory,
            "results_ledger_rows": list(self.results_ledger_rows),
            "receipt": self.receipt.to_dict(),
            "research_map_projection": self.research_map_projection.to_dict(),
        }

    def public_bundle(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory["trajectory_id"],
            "receipt": self.receipt.public_payload(),
            "receipt_ref": self.receipt.receipt_ref,
            "map_projection": self.research_map_projection.to_dict(),
            "cost_ledger": self.sandbox.execution_trace["cost_ledger"],
            "provider_usage": [dict(item) for item in self.sandbox.provider_usage],
            "final_balance": self.final_balance.to_dict(),
        }


def validate_hosted_loop_runtime_flags(flags: HostedLoopRuntimeFlags | Mapping[str, Any]) -> list[str]:
    if not isinstance(flags, HostedLoopRuntimeFlags):
        flags = HostedLoopRuntimeFlags.from_mapping(flags)
    errors: list[str] = []
    if not flags.local_execution_enabled:
        errors.append("local_execution_disabled")
    for field, enabled in flags.to_dict().items():
        if field == "local_execution_enabled":
            continue
        if enabled:
            errors.append(f"{field}_must_remain_false")
    return errors


def assert_hosted_loop_runtime_allowed(flags: HostedLoopRuntimeFlags | Mapping[str, Any]) -> None:
    errors = validate_hosted_loop_runtime_flags(flags)
    if errors:
        raise ValueError("; ".join(errors))


def load_hosted_loop_fixture(path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_hosted_loop_fixture(
    *,
    scenario: str = "winning",
    flags: HostedLoopRuntimeFlags | Mapping[str, Any] | None = None,
    fixture_path: Path | str = FIXTURE_PATH,
) -> HostedLoopRunResult:
    fixture = load_hosted_loop_fixture(fixture_path)
    runtime_flags = flags if isinstance(flags, HostedLoopRuntimeFlags) else HostedLoopRuntimeFlags.from_mapping(flags)
    return execute_hosted_loop_mvp(
        fixture=fixture,
        scenario_name=scenario,
        flags=runtime_flags,
    )


def execute_hosted_loop_mvp(
    *,
    fixture: Mapping[str, Any],
    scenario_name: str,
    flags: HostedLoopRuntimeFlags | Mapping[str, Any],
    workflow_guards: LoopWorkflowGuards | Mapping[str, Any] | None = None,
) -> HostedLoopRunResult:
    """Run one deterministic local hosted-loop fixture.

    `local_execution_enabled` must be explicitly true. Every production-facing
    flag must remain false, and the shared Phase 1 workflow guards must remain
    disabled. The function writes no durable files except temporary snapshot
    store files used to build hash-addressed evidence records.
    """

    runtime_flags = flags if isinstance(flags, HostedLoopRuntimeFlags) else HostedLoopRuntimeFlags.from_mapping(flags)
    assert_hosted_loop_runtime_allowed(runtime_flags)
    assert_loop_workflows_disabled(workflow_guards or default_loop_workflow_guards())

    scenario = dict(fixture["scenarios"][scenario_name])
    registry = ComponentRegistry.from_mapping(fixture["registry"])
    policy = LoopStartPolicy.from_mapping(fixture["loop_start_policy"])
    request = ResearchLoopStartRequest.from_mapping(fixture["loop_start_request"])
    payment = LoopStartPaymentEvidence.from_mapping(fixture["payment_evidence"])
    key_ref = MinerOpenRouterKeyReference.from_mapping(fixture["miner_openrouter_key_ref"])

    decision = evaluate_loop_start_request(request, payment, key_ref, policy)
    if not decision.can_queue:
        raise ValueError("loop-start decision cannot queue: " + ", ".join(decision.reasons))

    ticket = open_hosted_loop_ticket(fixture, scenario, request)
    probe_schedule = build_private_probe_schedule(ticket)
    probe_receipt = build_private_probe_receipt(fixture, ticket, probe_schedule)
    initial_balance = LoopBalanceRecord.from_mapping(
        {
            "balance_id": fixture["balance"]["balance_id"],
            "owner_ref": ticket.owner_ref,
            "miner_ref": ticket.miner_ref,
            "currency": "usd_cents",
            "total_funded_cents": ticket.funded_balance_cents,
            "available_cents": ticket.funded_balance_cents,
            "reserved_cents": 0,
            "spent_cents": 0,
            "local_only": True,
            "payments_enabled": False,
            "settlement_enabled": False,
            "external_account_ref": "",
        }
    )
    _raise_if_errors(validate_loop_balance(initial_balance), "initial balance")

    reserved_balance, reservation = reserve_loop_balance(
        initial_balance,
        ticket,
        requested_loops=request.loop_count,
        idempotency_key=fixture["queue"]["idempotency_key"],
    )
    _raise_if_errors(validate_loop_balance(reserved_balance), "reserved balance")
    _raise_if_errors(validate_loop_escrow_reservation(reservation), "reservation")

    run_id = _uuid(f"{HOSTED_LOOP_MVP_VERSION}:{scenario_name}:{ticket.ticket_id}:run")
    queue_entry = build_hosted_queue_entry(
        ticket=ticket,
        decision=decision,
        reservation=reservation,
        run_id=run_id,
        priority_score=int(fixture["queue"]["priority_score"]),
        created_seq=int(fixture["queue"]["created_seq"]),
    )
    _raise_if_errors(validate_hosted_queue_entry(queue_entry), "hosted queue entry")

    sandbox = run_sandboxed_engine_v1(
        fixture=fixture,
        scenario=scenario,
        registry=registry,
        ticket=ticket,
        request=request,
        key_ref=key_ref,
        run_id=run_id,
    )

    final_balance, committed = commit_loop_escrow(
        reserved_balance,
        reservation,
        actual_spend_cents=sandbox.actual_spend_cents,
        cost_ledger_ref=sandbox.cost_ledger_ref,
    )
    _raise_if_errors(validate_loop_balance(final_balance), "final balance")
    _raise_if_errors(validate_loop_escrow_reservation(committed), "committed escrow")

    balance_ledger = build_balance_ledger(
        request=request,
        ticket=ticket,
        initial_balance=initial_balance,
        reserved_balance=reserved_balance,
        final_balance=final_balance,
        reservation=reservation,
        committed=committed,
        payment_ref=decision.payment_ref,
    )
    _raise_if_errors(validate_balance_ledger(balance_ledger, final_balance=final_balance), "balance ledger")

    trajectory = build_hosted_trajectory(
        fixture=fixture,
        scenario=scenario,
        request=request,
        ticket=ticket,
        decision=decision,
        sandbox=sandbox,
        final_balance=final_balance,
        receipt_ref=None,
    )
    results_rows = build_results_ledger_rows(
        trajectory=trajectory,
        ticket=ticket,
        patch_runs=sandbox.patch_runs,
        created_at=str(fixture["created_at"]),
    )

    receipt = build_hosted_loop_receipt(
        ticket=ticket,
        sandbox=sandbox,
        committed=committed,
        trajectory=trajectory,
        results_rows=results_rows,
    )
    trajectory = {
        **trajectory,
        "final": {
            **trajectory["final"],
            "settlement": {
                **trajectory["final"]["settlement"],
                "receipt_ref": receipt.receipt_ref,
            },
        },
    }

    map_projection = build_hosted_research_map_projection(
        fixture=fixture,
        registry=registry,
        ticket=ticket,
        sandbox=sandbox,
        receipt=receipt,
        results_rows=results_rows,
    )

    result = HostedLoopRunResult(
        loop_start_decision=decision,
        ticket=ticket,
        probe_schedule=probe_schedule,
        probe_receipt=probe_receipt,
        initial_balance=initial_balance,
        reserved_balance=reserved_balance,
        final_balance=final_balance,
        reservation=reservation,
        committed_reservation=committed,
        balance_ledger=tuple(balance_ledger),
        queue_entry=queue_entry,
        sandbox=sandbox,
        trajectory=trajectory,
        results_ledger_rows=tuple(results_rows),
        receipt=receipt,
        research_map_projection=map_projection,
    )
    _raise_if_errors(validate_hosted_loop_result(result), "hosted loop result")
    return result


def open_hosted_loop_ticket(
    fixture: Mapping[str, Any],
    scenario: Mapping[str, Any],
    request: ResearchLoopStartRequest,
) -> TicketRecord:
    ticket_data = dict(fixture["ticket"])
    return TicketRecord(
        ticket_id=str(ticket_data["ticket_id"]),
        owner_ref=str(ticket_data["owner_ref"]),
        miner_ref=str(ticket_data["miner_ref"]),
        brief_ref=request.brief_ref,
        island=request.island,
        target_component=str(scenario["target_component"]),
        funded_balance_cents=int(ticket_data["funded_balance_cents"]),
        remaining_balance_cents=int(ticket_data["funded_balance_cents"]),
        status="funded",
        budget_caps=BudgetCaps.from_mapping(ticket_data.get("budget_caps")),
        visibility_policy=VisibilityPolicy.DEFAULT_PRIVATE.value,
        artifact_release_state=ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value,
        created_seq=int(ticket_data.get("created_seq", 0)),
    )


def build_private_probe_receipt(
    fixture: Mapping[str, Any],
    ticket: TicketRecord,
    probe_schedule: PrivateProbeScheduleRecord,
) -> PrivateProbeReceipt:
    probe = fixture["probe"]
    receipt = PrivateProbeReceipt(
        receipt_ref=str(probe["receipt_ref"]),
        probe_id=probe_schedule.probe_id,
        ticket_id=ticket.ticket_id,
        miner_ref=ticket.miner_ref,
        brief_ref=ticket.brief_ref,
        island=ticket.island,
        cost_cents=int(probe["cost_cents"]),
        result_ref=str(probe["result_ref"]),
        cost_ledger_ref=str(probe["cost_ledger_ref"]),
        dev_delta_lcb=float(probe["dev_delta_lcb"]),
        private_to_owner_ref=ticket.owner_ref,
    )
    _raise_if_errors(validate_private_probe_receipt(receipt), "private probe receipt")
    return receipt


def build_hosted_queue_entry(
    *,
    ticket: TicketRecord,
    decision: LoopStartDecision,
    reservation: LoopEscrowReservation,
    run_id: str,
    priority_score: int,
    created_seq: int,
) -> HostedRunQueueEntry:
    payload = {
        "ticket_id": ticket.ticket_id,
        "decision_id": decision.decision_id,
        "reservation_id": reservation.reservation_id,
        "run_id": run_id,
    }
    return HostedRunQueueEntry(
        queue_id="hosted_queue:" + sha256_json(payload).split(":", 1)[1][:16],
        run_id=run_id,
        ticket_id=ticket.ticket_id,
        loop_start_decision_id=decision.decision_id,
        reservation_id=reservation.reservation_id,
        priority_score=priority_score,
        created_seq=created_seq,
    )


def validate_hosted_queue_entry(entry: HostedRunQueueEntry | Mapping[str, Any]) -> list[str]:
    if not isinstance(entry, HostedRunQueueEntry):
        entry = HostedRunQueueEntry(**dict(entry))
    errors = validate_queue_entries([entry.to_orchestrator_queue_entry()])
    if not entry.run_id:
        errors.append("run_id is required")
    if not entry.loop_start_decision_id:
        errors.append("loop_start_decision_id is required")
    if not entry.reservation_id:
        errors.append("reservation_id is required")
    if entry.status != "local_only_queued":
        errors.append("hosted queue entry must remain local_only_queued")
    if not entry.local_only:
        errors.append("hosted queue entry must remain local_only")
    if entry.scheduler_enabled:
        errors.append("hosted queue entry scheduler must remain disabled")
    return errors


def run_sandboxed_engine_v1(
    *,
    fixture: Mapping[str, Any],
    scenario: Mapping[str, Any],
    registry: ComponentRegistry,
    ticket: TicketRecord,
    request: ResearchLoopStartRequest,
    key_ref: MinerOpenRouterKeyReference,
    run_id: str,
) -> HostedSandboxResult:
    patch_runs: list[HostedPatchRun] = []

    for candidate in scenario["candidates"]:
        patch = PatchRecord.from_mapping(candidate["patch"])
        if patch.patch_type not in ENGINE_V1_ENABLED_PATCH_TYPES:
            raise ValueError(f"hosted MVP does not enable patch type: {patch.patch_type}")
        hypothesis = HypothesisRecord.from_mapping(candidate["hypothesis"])
        _raise_if_errors(validate_hypothesis(hypothesis, registry), f"hypothesis {hypothesis.hypothesis_id}")
        _raise_if_errors(validate_patch(patch, registry), f"patch {patch.patch_id}")
        raise ValueError(
            "real_evaluator_score_bundle_required: hosted Research Lab loops "
            "must score candidate patches with the production evaluator before "
            "emitting receipts, ledger rows, map projections, or reward inputs."
        )

    if not patch_runs:
        raise ValueError("hosted sandbox requires at least one patch run")

    with tempfile.TemporaryDirectory(prefix="leadpoet_hosted_loop_") as tmpdir:
        snapshot = capture_snapshot(
            url=str(fixture["local_snapshot"]["url"]),
            content=str(fixture["local_snapshot"]["content"]),
            store=LocalSnapshotStore(tmpdir),
            signer=NotarySigner(str(fixture["local_snapshot"]["signing_key"]), key_id="hosted-loop-mvp"),
            fetch_ts=str(fixture["local_snapshot"]["fetch_ts"]),
            metadata={"source": "hosted-loop-local-fixture", "scenario": scenario["scenario_id"]},
        )
        evidence_bundle = build_evidence_bundle(
            artifact_hash=sha256_json({"run_id": run_id, "scenario": scenario["scenario_id"]}),
            snapshots=[snapshot],
            run_id=run_id,
            created_at=str(fixture["created_at"]),
            validate=True,
        )

    calls = _build_trace_calls(fixture, scenario, patch_runs, key_ref)
    trace = build_execution_trace(
        run_id=run_id,
        artifact_hash=sha256_json({"ticket_id": ticket.ticket_id, "patches": [run.patch.patch_id for run in patch_runs]}),
        role="candidate",
        rung="L1",
        status="completed",
        icp_set_hash=str(fixture["icp_set_hash"]),
        eval_version={
            "verifier_hash": str(fixture["eval_version"]["verifier_hash"]),
            "judge_version_hash": str(fixture["eval_version"]["judge_version_hash"]),
        },
        calls=calls,
        evidence_refs=evidence_refs_from_bundle(evidence_bundle),
        outputs_payload={
            "scenario_id": scenario["scenario_id"],
            "node_ids": [run.node_id for run in patch_runs],
            "decisions": [run.result.promotion_decision for run in patch_runs],
        },
        score_bundle_payload={
            "target_deltas": [run.result.target_delta for run in patch_runs],
            "statuses": [run.result.status for run in patch_runs],
        },
        attestation_ref="attestation:local-hosted-loop:" + run_id,
        validate=True,
    )

    provider_usage = _provider_usage_from_calls(calls, key_ref, fixture["loop_start_policy"])
    actual_spend_cents = int(round(float(trace["cost_ledger"]["total_usd"]) * 100))
    sandbox_job = ResearchSandboxJobRecord(
        job_id="sandbox_job:" + sha256_json({"run_id": run_id}).split(":", 1)[1][:16],
        ticket_id=ticket.ticket_id,
        image_ref=str(fixture["sandbox"]["image_ref"]),
        sandbox_policy_ref=str(fixture["sandbox"]["sandbox_policy_ref"]),
        resource_caps=SandboxResourceCaps.from_mapping(fixture["sandbox"]["resource_caps"]),
        mounted_artifact_refs=tuple(str(item) for item in fixture["sandbox"]["mounted_artifact_refs"]),
        status=SandboxJobStatus.COMPLETED.value,
        estimated_cost_cents=int(fixture["sandbox"]["estimated_cost_cents"]),
        actual_cost_cents=actual_spend_cents,
        lab_only=True,
        provisioned_host=False,
        network_enabled=False,
        enclave_attested=False,
    )
    _raise_if_errors(validate_sandbox_job_record(sandbox_job), "sandbox job")

    best = sorted(
        patch_runs,
        key=lambda run: (run.result.target_delta, -run.total_cost_cents, run.node_id),
        reverse=True,
    )[0]
    return HostedSandboxResult(
        run_id=run_id,
        sandbox_job=sandbox_job,
        parent_metrics=parent_metrics,
        patch_runs=tuple(patch_runs),
        evidence_bundle=evidence_bundle,
        execution_trace=trace,
        provider_usage=tuple(provider_usage),
        cost_ledger_ref="cost_ledger:" + sha256_json(trace["cost_ledger"]),
        actual_spend_cents=actual_spend_cents,
        best_node_id=best.node_id,
        best_dev_delta_lcb=round(float(best.result.target_delta), 6),
    )


def build_balance_ledger(
    *,
    request: ResearchLoopStartRequest,
    ticket: TicketRecord,
    initial_balance: LoopBalanceRecord,
    reserved_balance: LoopBalanceRecord,
    final_balance: LoopBalanceRecord,
    reservation: LoopEscrowReservation,
    committed: LoopEscrowReservation,
    payment_ref: str,
) -> list[HostedLoopBalanceLedgerEntry]:
    rows = [
        _ledger_entry(
            seq=0,
            ticket=ticket,
            balance=initial_balance,
            miner_hotkey=request.miner_hotkey,
            event_type="deposit",
            amount_cents=initial_balance.total_funded_cents,
            source_ref=payment_ref,
        ),
        _ledger_entry(
            seq=1,
            ticket=ticket,
            balance=reserved_balance,
            miner_hotkey=request.miner_hotkey,
            event_type="reserve",
            amount_cents=reservation.reserved_cents,
            source_ref=reservation.reservation_id,
        ),
        _ledger_entry(
            seq=2,
            ticket=ticket,
            balance=final_balance,
            miner_hotkey=request.miner_hotkey,
            event_type="spend",
            amount_cents=committed.committed_spend_cents,
            source_ref=committed.cost_ledger_ref,
        ),
    ]
    if committed.released_cents:
        rows.append(
            _ledger_entry(
                seq=3,
                ticket=ticket,
                balance=final_balance,
                miner_hotkey=request.miner_hotkey,
                event_type="release",
                amount_cents=committed.released_cents,
                source_ref=committed.reservation_id,
            )
        )
    return rows


def validate_balance_ledger(
    rows: Sequence[HostedLoopBalanceLedgerEntry | Mapping[str, Any]],
    *,
    final_balance: LoopBalanceRecord,
) -> list[str]:
    normalized = [
        row if isinstance(row, HostedLoopBalanceLedgerEntry) else HostedLoopBalanceLedgerEntry(**dict(row))
        for row in rows
    ]
    errors: list[str] = []
    if not normalized:
        return ["balance ledger requires rows"]
    if [row.seq for row in normalized] != list(range(len(normalized))):
        errors.append("balance ledger seq must be contiguous")
    allowed = {"deposit", "reserve", "spend", "release", "tombstone"}
    ids = set()
    for row in normalized:
        if row.ledger_entry_id in ids:
            errors.append(f"duplicate ledger_entry_id: {row.ledger_entry_id}")
        ids.add(row.ledger_entry_id)
        if row.event_type not in allowed:
            errors.append(f"unknown balance ledger event_type: {row.event_type}")
        if row.amount_cents < 0:
            errors.append("balance ledger amount_cents must be non-negative")
        if not row.local_only:
            errors.append("balance ledger rows must remain local_only")
    last = normalized[-1]
    if (
        last.available_cents != final_balance.available_cents
        or last.reserved_cents != final_balance.reserved_cents
        or last.spent_cents != final_balance.spent_cents
    ):
        errors.append("last balance ledger row must match final balance")
    return errors


def build_hosted_trajectory(
    *,
    fixture: Mapping[str, Any],
    scenario: Mapping[str, Any],
    request: ResearchLoopStartRequest,
    ticket: TicketRecord,
    decision: LoopStartDecision,
    sandbox: HostedSandboxResult,
    final_balance: LoopBalanceRecord,
    receipt_ref: Optional[str],
) -> dict[str, Any]:
    created_at = str(fixture["created_at"])
    trajectory_id = _uuid(f"{HOSTED_LOOP_MVP_VERSION}:{scenario['scenario_id']}:{ticket.ticket_id}:trajectory")
    brief_id = _uuid(ticket.brief_ref)
    events: list[dict[str, Any]] = []
    _append_event(
        events,
        created_at,
        "PROBE",
        0.0,
        {
            "probe_id": "probe:" + sha256_json({"ticket_id": ticket.ticket_id}).split(":", 1)[1][:16],
            "result_quantized": round(float(fixture["probe"]["dev_delta_lcb"]), 6),
            "fixtures_used": list(scenario["fixtures_used"]),
        },
    )
    _append_event(
        events,
        created_at,
        "LOOP_FUNDED",
        0.0,
        {
            "loop_n": request.loop_count,
            "balance_before": round(ticket.funded_balance_cents / 100.0, 6),
        },
    )
    for patch_run in sandbox.patch_runs:
        _append_event(
            events,
            created_at,
            "NODE_DRAFTED",
            patch_run.draft_cost_cents / 100.0,
            {
                "node_id": patch_run.node_id,
                "parent_id": "baseline-reference",
                "operator": "draft",
                "component": patch_run.hypothesis.component,
                "patch_type": patch_run.patch.patch_type,
                "hypothesis": _trajectory_hypothesis(patch_run.hypothesis),
                "patch_ref": patch_run.patch.patch_id,
                "model_used": patch_run.model_used,
                "tokens": {"in": patch_run.tokens_in, "out": patch_run.tokens_out},
            },
        )
        _append_event(
            events,
            created_at,
            "NODE_EVALUATED",
            patch_run.eval_cost_cents / 100.0,
            {
                "node_id": patch_run.node_id,
                "status": patch_run.result.status,
                "rung": "L1",
                "score_bundle_ref": patch_run.score_bundle.get("score_bundle_ref"),
                "paired_lcb_vs_parent": patch_run.result.target_delta,
                "fixtures": list(patch_run.fixture_refs),
                "cache_hits": {"snapshot": 1, "verdict": 0},
                "execution_trace_ref": "execution_trace:" + sandbox.execution_trace["run_id"],
            },
        )
        _append_event(
            events,
            created_at,
            "NODE_REFLECTED",
            patch_run.reflect_cost_cents / 100.0,
            {
                "node_id": patch_run.node_id,
                "lesson": {
                    "worked": patch_run.reflection["worked"],
                    "failed": patch_run.reflection["failed"],
                    "why": patch_run.reflection["why"],
                    "next_question": patch_run.reflection["next_question"],
                },
                "lesson_embedding_ref": "lesson_embedding:" + sha256_json(patch_run.reflection).split(":", 1)[1][:16],
                "lesson_provenance": {
                    "champion_base": patch_run.reflection["champion_base"],
                    "component": patch_run.reflection["component"],
                    "eval_version": patch_run.reflection["eval_version"],
                },
            },
        )
    _append_event(
        events,
        created_at,
        "PLATEAU_STOP",
        0.0,
        {"reason": "plateau", "best_node_id": sandbox.best_node_id},
    )
    return {
        "trajectory_id": trajectory_id,
        "schema_version": "1.0",
        "brief_id": brief_id,
        "island": ticket.island,
        "funder_hotkey": request.miner_hotkey,
        "brief_sanitized_ref": ticket.brief_ref,
        "novelty_gate": {
            "result": "pass",
            "similarity": float(scenario["novelty_similarity"]),
            "nearest_prior_receipt": None,
        },
        "engine_version": HOSTED_LOOP_MVP_VERSION,
        "champion_base": str(fixture["registry"]["champion_base"]),
        "created_at": created_at,
        "events": events,
        "final": {
            "settlement": {
                "loops_consumed": request.loop_count,
                "probation_charged": False,
                "balance_returned": round(final_balance.available_cents / 100.0, 6),
                "crown": None,
                "grant_state": "not_applicable_local_mvp",
                "receipt_ref": receipt_ref,
            }
        },
    }


def build_results_ledger_rows(
    *,
    trajectory: Mapping[str, Any],
    ticket: TicketRecord,
    patch_runs: Sequence[HostedPatchRun],
    created_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for patch_run in patch_runs:
        status = _ledger_status(patch_run.result)
        row = {
            "ledger_row_id": _uuid(f"{trajectory['trajectory_id']}:{patch_run.node_id}:ledger"),
            "schema_version": "1.0",
            "trajectory_id": trajectory["trajectory_id"],
            "node_id": patch_run.node_id,
            "commit": patch_run.patch.patch_id,
            "island": ticket.island,
            "brief_id": trajectory["brief_id"],
            "targeted_metric": patch_run.targeted_metric,
            "delta_vs_parent": patch_run.result.target_delta,
            "cost_usd": round(patch_run.total_cost_cents / 100.0, 6),
            "status": status,
            "description": _public_patch_summary(patch_run),
            "created_at": created_at,
        }
        rows.append(row)
    return rows


def build_hosted_loop_receipt(
    *,
    ticket: TicketRecord,
    sandbox: HostedSandboxResult,
    committed: LoopEscrowReservation,
    trajectory: Mapping[str, Any],
    results_rows: Sequence[Mapping[str, Any]],
) -> LoopReceiptRecord:
    source = {
        "ticket_id": ticket.ticket_id,
        "miner_ref": ticket.miner_ref,
        "brief_ref": ticket.brief_ref,
        "island": ticket.island,
        "summary_text": _receipt_summary(sandbox),
        "stopped_reason": "plateau_detector",
        "loops_consumed": trajectory["final"]["settlement"]["loops_consumed"],
        "dev_delta_trajectory": [row["delta_vs_parent"] for row in results_rows],
        "best_dev_delta_lcb": sandbox.best_dev_delta_lcb,
        "cost_ledger_ref": committed.cost_ledger_ref,
        "result_ref": "result:" + sha256_json({"trajectory_id": trajectory["trajectory_id"], "rows": list(results_rows)}),
        "proof_refs": [
            sandbox.evidence_bundle["bundle_hash"],
            "execution_trace:" + sandbox.execution_trace["run_id"],
            committed.reservation_id,
        ],
        "public_trace_ref": "sanitized_trace:" + sandbox.execution_trace["run_id"],
        "contains_live_champion_ip": False,
        "contains_sealed_eval_details": False,
        "contains_raw_evidence_snapshot": False,
        "contains_private_customer_data": False,
        "contains_judge_prompts": False,
    }
    receipt = build_public_loop_receipt(source)
    _raise_if_errors(validate_loop_receipt(receipt), "hosted loop receipt")
    return receipt


def build_hosted_research_map_projection(
    *,
    fixture: Mapping[str, Any],
    registry: ComponentRegistry,
    ticket: TicketRecord,
    sandbox: HostedSandboxResult,
    receipt: LoopReceiptRecord,
    results_rows: Sequence[Mapping[str, Any]],
) -> ResearchMapProjectionRecord:
    component_statuses = tuple(
        ComponentStatusRecord(
            component_ref=f"component:{entry.name}",
            island=ticket.island,
            component_name=entry.name,
            status=ComponentMapStatus.ACTIVE.value,
            registry_ref=registry.manifest_version,
            public_summary=f"{entry.name} accepts local Engine v1 typed patches only.",
            allowed_patch_types=entry.allowed_patch_types,
            current_patch_seq=entry.current_patch_seq,
            public_receipt_refs=(receipt.receipt_ref,),
            sanitized_trace_refs=("sanitized_trace:" + sandbox.execution_trace["run_id"],),
            withheld_private_artifact_count=1,
        )
        for entry in registry.entries
    )
    points = tuple(
        FrontierPointRecord(
            point_id="frontier_point:" + sha256_json({"row": row["ledger_row_id"]}).split(":", 1)[1][:16],
            run_ref=sandbox.run_id,
            receipt_ref=receipt.receipt_ref,
            patch_ref=str(row["commit"]),
            score=round(50.0 + float(row["delta_vs_parent"] or 0.0), 6),
            delta_vs_parent=float(row["delta_vs_parent"] or 0.0),
            cost_cents=int(round(float(row["cost_usd"]) * 100)),
            status=str(row["status"]),
            public_summary=str(row["description"]),
            kept=str(row["status"]) == "keep",
        )
        for row in results_rows
    )
    kept_cost = sum(point.cost_cents for point in points if point.kept)
    kept_count = sum(1 for point in points if point.kept)
    frontier = FrontierSummaryRecord(
        frontier_id="frontier:" + sha256_json({"receipt_ref": receipt.receipt_ref, "ticket_id": ticket.ticket_id}).split(":", 1)[1][:16],
        island=ticket.island,
        target_component=ticket.target_component,
        points=points,
        running_best_score=max(point.score for point in points),
        keep_rate=round(kept_count / len(points), 6),
        cost_per_kept_patch_cents=int(round(kept_cost / kept_count)) if kept_count else 0,
        crash_rate=round(sum(1 for point in points if point.status in {"crash", "timeout"}) / len(points), 6),
        source_ledger_refs=tuple(str(row["ledger_row_id"]) for row in results_rows),
    )
    best_patch = sorted(sandbox.patch_runs, key=lambda run: run.result.target_delta, reverse=True)[0]
    prediction = AllocatorPredictionRecord(
        prediction_id="allocator:" + sha256_json({"ticket_id": ticket.ticket_id, "component": best_patch.hypothesis.component}).split(":", 1)[1][:16],
        cell_ref="cell:" + ticket.ticket_id,
        island=ticket.island,
        target_component=best_patch.hypothesis.component,
        patch_type=best_patch.patch.patch_type,
        predicted_delta=best_patch.hypothesis.predicted_delta,
        confidence=float(fixture["map"]["allocator_confidence"]),
        expected_cost_cents=int(fixture["map"]["expected_cost_cents"]),
        expected_value_score=float(fixture["map"]["expected_value_score"]),
        provenance_refs=(receipt.receipt_ref, "execution_trace:" + sandbox.execution_trace["run_id"]),
        model_version_ref=str(fixture["map"]["model_version_ref"]),
    )
    cell = ResearchMapCellRecord(
        cell_id=prediction.cell_ref,
        island=ticket.island,
        target_component=best_patch.hypothesis.component,
        patch_type=best_patch.patch.patch_type,
        failure_board_item_ref=str(fixture["map"]["failure_board_item_ref"]),
        sanitized_failure_summary=str(fixture["map"]["sanitized_failure_summary"]),
        sanitized_examples=tuple(str(item) for item in fixture["map"]["sanitized_examples"]),
        recent_case_count=int(fixture["map"]["recent_case_count"]),
        run_density_7d=1,
        achieved_delta_mean=round(sum(point.delta_vs_parent for point in points) / len(points), 6),
        frontier_summary_ref=frontier.frontier_id,
        allocator_prediction_ref=prediction.prediction_id,
        source_receipt_refs=(receipt.receipt_ref,),
    )
    projection = build_research_map_projection(
        generated_for_date=str(fixture["map"]["generated_for_date"]),
        map_version=str(fixture["map"]["map_version"]),
        component_statuses=component_statuses,
        cells=(cell,),
        frontier_summaries=(frontier,),
        allocator_predictions=(prediction,),
        source_ledger_refs=tuple(str(row["ledger_row_id"]) for row in results_rows),
    )
    _raise_if_errors(validate_research_map_projection_record(projection), "hosted map projection")
    return projection


def validate_hosted_loop_result(result: HostedLoopRunResult | Mapping[str, Any]) -> list[str]:
    if not isinstance(result, HostedLoopRunResult):
        return ["validate_hosted_loop_result requires HostedLoopRunResult"]
    errors: list[str] = []
    errors.extend(validate_schema_record("research_trajectory.schema.json", result.trajectory))
    for row in result.results_ledger_rows:
        errors.extend(validate_schema_record("results_ledger_row.schema.json", dict(row)))
    errors.extend(validate_loop_receipt(result.receipt))
    errors.extend(validate_research_map_projection_record(result.research_map_projection))
    errors.extend(validate_balance_ledger(result.balance_ledger, final_balance=result.final_balance))
    errors.extend(validate_hosted_queue_entry(result.queue_entry))
    if result.loop_start_decision.status != "ready_to_queue":
        errors.append("loop-start decision must be ready_to_queue")
    if not result.loop_start_decision.miner_openrouter_key_ref:
        errors.append("miner OpenRouter key ref is required")
    if result.ticket.island != GENERALIST_ISLAND:
        errors.append("hosted MVP is limited to the generalist island")
    if result.sandbox.actual_spend_cents > result.reservation.reserved_cents:
        errors.append("sandbox spend cannot exceed reserved loop balance")
    if result.final_balance.available_cents <= 0:
        errors.append("fixture should leave unspent balance available for return/reuse")
    if _contains_secret_material(result.to_dict()):
        errors.append("hosted loop result contains raw provider secret material")
    patch_types = {run.patch.patch_type for run in result.sandbox.patch_runs}
    deferred = patch_types.difference(ENGINE_V1_ENABLED_PATCH_TYPES)
    if deferred:
        errors.append("hosted loop used deferred patch types: " + ", ".join(sorted(deferred)))
    return errors


def verify_research_lab_hosted_loop(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    fixture = load_hosted_loop_fixture(fixture_path)

    try:
        run_hosted_loop_fixture(scenario="winning", fixture_path=fixture_path)
    except ValueError as exc:
        _assert("local_execution_disabled" in str(exc), "default runtime flags block execution")
    else:
        raise AssertionError("default runtime flags should block hosted loop execution")

    unsafe_flags = dict(fixture["unsafe_runtime_flags"])
    try:
        run_hosted_loop_fixture(scenario="winning", flags=unsafe_flags, fixture_path=fixture_path)
    except ValueError as exc:
        _assert("production_paid_loops_enabled_must_remain_false" in str(exc), "production flags block execution")
    else:
        raise AssertionError("production runtime flags should block hosted loop execution")

    flags = HostedLoopRuntimeFlags.from_mapping(fixture["runtime_flags"])
    try:
        run_hosted_loop_fixture(scenario="winning", flags=flags, fixture_path=fixture_path)
    except ValueError as exc:
        _assert("real_evaluator_score_bundle_required" in str(exc), "hosted loop requires real evaluator score bundle")
    else:
        raise AssertionError("hosted loop must not create a scored result without a real evaluator")

    return {
        "real_evaluator_score_bundle_required": True,
        "production_improvement_scoring_enabled": False,
        "required_evaluator": "research_lab_qualification_style_evaluator",
        "required_source_of_truth": "sealed benchmark ICP set plus qualification.scoring.lead_scorer score bundles",
        "runtime_flags_verified": flags.to_dict(),
    }


def _build_trace_calls(
    fixture: Mapping[str, Any],
    scenario: Mapping[str, Any],
    patch_runs: Sequence[HostedPatchRun],
    key_ref: MinerOpenRouterKeyReference,
) -> list[Any]:
    created_at = str(fixture["created_at"])
    calls = []
    seq = 1
    for patch_run in patch_runs:
        calls.append(
            make_trace_call(
                seq=seq,
                ts=_offset_ts(created_at, seq),
                provider="openrouter",
                model=str(fixture["trace"]["openrouter_model"]),
                purpose="engine_internal",
                component=patch_run.hypothesis.component,
                request={
                    "key_ref": key_ref.key_ref,
                    "patch_type": patch_run.patch.patch_type,
                    "node_id": patch_run.node_id,
                },
                response={
                    "patch_ref": patch_run.patch.patch_id,
                    "lesson_ref": patch_run.reflection["lesson_id"],
                },
                tokens_in=patch_run.tokens_in,
                tokens_out=patch_run.tokens_out,
                cost_usd=(patch_run.draft_cost_cents + patch_run.reflect_cost_cents) / 100.0,
                call_emitter="model",
                teacher_model_flag=True,
            )
        )
        seq += 1
        calls.append(
            make_trace_call(
                seq=seq,
                ts=_offset_ts(created_at, seq),
                provider="local_sandbox",
                model=HOSTED_LOOP_MVP_VERSION,
                purpose="engine_internal",
                component=patch_run.hypothesis.component,
                request={"patch_ref": patch_run.patch.patch_id, "targeted_metric": patch_run.targeted_metric},
                response={"status": patch_run.result.status, "target_delta": patch_run.result.target_delta},
                cost_usd=patch_run.eval_cost_cents / 100.0,
            )
        )
        seq += 1
    calls.append(
        make_trace_call(
            seq=seq,
            ts=_offset_ts(created_at, seq),
            provider="exa",
            model="exa-search",
            purpose="search",
            component=str(scenario["target_component"]),
            request={"query_ref": scenario["query_ref"]},
            response={"snapshot_ref": "local_snapshot"},
            cost_usd=int(fixture["trace"]["exa_cost_cents"]) / 100.0,
        )
    )
    seq += 1
    calls.append(
        make_trace_call(
            seq=seq,
            ts=_offset_ts(created_at, seq),
            provider="scrapingdog",
            model="scrapingdog-fetch",
            purpose="scrape",
            component=str(scenario["target_component"]),
            request={"url_ref": scenario["url_ref"]},
            response={"snapshot_ref": "local_snapshot"},
            cost_usd=int(fixture["trace"]["scrapingdog_cost_cents"]) / 100.0,
        )
    )
    return calls


def _provider_usage_from_calls(
    calls: Sequence[Any],
    key_ref: MinerOpenRouterKeyReference,
    policy_mapping: Mapping[str, Any],
) -> list[dict[str, Any]]:
    policy = LoopStartPolicy.from_mapping(policy_mapping)
    cost_by_provider: dict[str, int] = {}
    tokens_by_provider: dict[str, dict[str, int]] = {}
    for call in calls:
        record = call.to_schema_call()
        provider = str(record["provider"])
        cost_by_provider[provider] = cost_by_provider.get(provider, 0) + int(round(float(record["cost_usd"]) * 100))
        tokens = record.get("tokens") or {}
        token_totals = tokens_by_provider.setdefault(provider, {"in": 0, "out": 0})
        token_totals["in"] += int(tokens.get("in", 0))
        token_totals["out"] += int(tokens.get("out", 0))
    return [
        {
            "provider": "openrouter",
            "key_source": "miner",
            "key_ref": key_ref.key_ref,
            "cost_cents": cost_by_provider.get("openrouter", 0),
            "tokens": tokens_by_provider.get("openrouter", {"in": 0, "out": 0}),
            "secret_material_stored": False,
        },
        {
            "provider": "exa",
            "key_source": "leadpoet_server_side",
            "key_ref": policy.leadpoet_exa_key_ref,
            "cost_cents": cost_by_provider.get("exa", 0),
            "secret_material_stored": False,
        },
        {
            "provider": "scrapingdog",
            "key_source": "leadpoet_server_side",
            "key_ref": policy.leadpoet_scrapingdog_key_ref,
            "cost_cents": cost_by_provider.get("scrapingdog", 0),
            "secret_material_stored": False,
        },
        {
            "provider": "local_sandbox",
            "key_source": "leadpoet_local_runtime",
            "key_ref": "local-runtime:no-secret",
            "cost_cents": cost_by_provider.get("local_sandbox", 0),
            "secret_material_stored": False,
        },
    ]


def _ledger_entry(
    *,
    seq: int,
    ticket: TicketRecord,
    balance: LoopBalanceRecord,
    miner_hotkey: str,
    event_type: str,
    amount_cents: int,
    source_ref: str,
) -> HostedLoopBalanceLedgerEntry:
    base = {
        "seq": seq,
        "ticket_id": ticket.ticket_id,
        "balance_id": balance.balance_id,
        "miner_hotkey": miner_hotkey,
        "event_type": event_type,
        "amount_cents": amount_cents,
        "available_cents": balance.available_cents,
        "reserved_cents": balance.reserved_cents,
        "spent_cents": balance.spent_cents,
        "source_ref": source_ref,
    }
    return HostedLoopBalanceLedgerEntry(
        ledger_entry_id="hosted_balance_ledger:" + sha256_json(base).split(":", 1)[1][:16],
        **base,
    )


def _append_event(
    events: list[dict[str, Any]],
    created_at: str,
    event_type: str,
    cost_usd: float,
    payload: Mapping[str, Any],
) -> None:
    seq = len(events)
    event = {
        "seq": seq,
        "ts": _offset_ts(created_at, seq),
        "type": event_type,
        "cost_usd": round(float(cost_usd), 6),
        **dict(payload),
    }
    event["anchored_hash"] = sha256_json(event)
    events.append(event)


def _trajectory_hypothesis(hypothesis: HypothesisRecord) -> dict[str, Any]:
    return {
        "failure_mode": hypothesis.failure_mode,
        "mechanism": hypothesis.mechanism,
        "predicted_delta": hypothesis.predicted_delta,
        "falsifier": hypothesis.falsifier,
    }


def _ledger_status(result: Any) -> str:
    if result.status == "timeout":
        return "timeout"
    if result.status == "crash":
        return "crash"
    if result.promotion_decision in {
        "keep",
        "simplification_keep",
    }:
        return "keep"
    return "discard"


def _public_patch_summary(patch_run: HostedPatchRun) -> str:
    decision = patch_run.result.promotion_decision
    return (
        f"{patch_run.patch.patch_type} on {patch_run.hypothesis.component} "
        f"targeted {patch_run.targeted_metric}; decision={decision}; "
        f"delta={patch_run.result.target_delta:.6f}."
    )


def _receipt_summary(sandbox: HostedSandboxResult) -> str:
    kept = [run for run in sandbox.patch_runs if _ledger_status(run.result) == "keep"]
    if kept:
        return (
            f"Hosted local Engine v1 loop evaluated {len(sandbox.patch_runs)} typed patches; "
            f"best kept node {sandbox.best_node_id} with delta {sandbox.best_dev_delta_lcb:.6f}."
        )
    return (
        f"Hosted local Engine v1 loop evaluated {len(sandbox.patch_runs)} typed patches; "
        "no patch cleared dev-eval, but the run produced a complete receipt."
    )


def _provider_key_source(provider_usage: Sequence[Mapping[str, Any]], provider: str) -> str:
    for item in provider_usage:
        if item.get("provider") == provider:
            return str(item.get("key_source", ""))
    return ""


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_lower = str(key).lower()
            safe_absence_flag = isinstance(nested, bool) and key_lower.startswith("raw_") and key_lower.endswith("_included")
            if (
                key_lower in FORBIDDEN_SECRET_KEYS
                or (key_lower.startswith("raw_") and not safe_absence_flag)
                or key_lower.endswith("_api_key")
            ):
                return True
            if _contains_secret_material(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in FORBIDDEN_SECRET_MARKERS)
    return False


def _offset_ts(created_at: str, seconds: int) -> str:
    dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt.astimezone(timezone.utc) + timedelta(seconds=seconds)).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _uuid(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUTHY_VALUES


def _raise_if_errors(errors: Sequence[str], label: str) -> None:
    if errors:
        raise ValueError(f"{label}: " + "; ".join(errors))


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
