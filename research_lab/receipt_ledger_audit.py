"""Phase 2 receipts-v2 and ledger-audit contracts.

P2.8 defines local record shapes for sanitized receipts and balance/cost ledger
audits against anchored artifacts. The validators do not mutate balances, pay
bounties, publish audit results, write production state, or query live ledgers.
They only validate deterministic local contracts that future live code must
emit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .canonical import sha256_json
from .on_chain_anchor import (
    PROTECTED_ANCHOR_KEYS,
    PROTECTED_ANCHOR_MARKERS,
    validate_anchor_commit_proposal_record,
    verify_research_lab_on_chain_anchor,
)
from .loop_foundation import (
    ArtifactReleaseState,
    ReleasePolicyRecord,
    VisibilityPolicy,
    validate_release_policy,
)
from .market_foundation import (
    MarketWorkflowGuards,
    assert_market_workflows_disabled,
    default_market_workflow_guards,
    verify_market_foundation,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "receipt_ledger_audit_fixtures.json"

RECEIPT_LEDGER_AUDIT_CONTRACT_VERSION = "receipt_ledger_audit:v1:local_contract"
PENDING_LEDGER_AUDIT_PUBLICATION_REF = "ledger_audit_publication:pending"

PROTECTED_LEDGER_AUDIT_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_ANCHOR_KEYS)
    | {
        "account_email",
        "bank_account",
        "billing_address",
        "credit_card",
        "customer_address",
        "customer_phone",
        "funding_private_key",
        "miner_email",
        "payment_secret",
        "raw_balance_entry",
        "raw_cost_entry",
        "raw_invoice",
        "wallet_private_key",
    }
)

PROTECTED_LEDGER_AUDIT_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_ANCHOR_MARKERS)
        | {
            "bank account",
            "billing address",
            "credit card",
            "customer phone",
            "payment secret",
            "raw balance",
            "raw cost",
            "raw invoice",
            "wallet private key",
        }
    )
)


class ReceiptV2State(str, Enum):
    LOCAL_RECEIPT_STUB = "local_receipt_stub"
    BLOCKED = "blocked"
    PUBLISHED_LIVE = "published_live"


class LedgerAuditState(str, Enum):
    LOCAL_MATCHED = "local_matched"
    LOCAL_MISMATCH = "local_mismatch"
    FRAUD_SUSPECTED = "fraud_suspected"
    UNKNOWN = "unknown"
    PRODUCTION_VERIFIED = "production_verified"


@dataclass(frozen=True)
class ReceiptV2Record:
    receipt_ref: str
    trajectory_id: str
    brief_summary_sanitized: str
    credited_funder_ref: str
    island: str
    stop_reason: str
    loops_consumed: int
    dev_delta_curve: tuple[float, ...]
    map_change_refs: tuple[str, ...]
    cost_ledger_ref: str
    cost_ledger_hash: str
    balance_ledger_ref: str
    balance_ledger_hash: str
    cost_audit_ref: str
    balance_audit_ref: str
    anchor_proposal_ref: str
    anchor_merkle_root: str
    anchor_payload_hash: str
    public_trace_ref: str
    proof_refs: tuple[str, ...]
    prior_receipt_refs: tuple[str, ...]
    cost_ledger_attested: bool = True
    contract_version: str = RECEIPT_LEDGER_AUDIT_CONTRACT_VERSION
    state: str = ReceiptV2State.LOCAL_RECEIPT_STUB.value
    local_only: bool = True
    public_payload_sanitized: bool = True
    production_published: bool = False
    live_publication_enabled: bool = False
    audit_publication_ref: str = PENDING_LEDGER_AUDIT_PUBLICATION_REF
    contains_live_champion_ip: bool = False
    contains_sealed_eval_details: bool = False
    contains_raw_evidence_snapshot: bool = False
    contains_private_customer_data: bool = False
    contains_judge_prompts: bool = False
    visibility_policy: str = VisibilityPolicy.PUBLIC_RECEIPT.value
    artifact_release_state: str = ArtifactReleaseState.PUBLIC_RECEIPT.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReceiptV2Record":
        return cls(
            receipt_ref=str(data["receipt_ref"]),
            trajectory_id=str(data["trajectory_id"]),
            brief_summary_sanitized=str(data["brief_summary_sanitized"]),
            credited_funder_ref=str(data["credited_funder_ref"]),
            island=str(data["island"]),
            stop_reason=str(data["stop_reason"]),
            loops_consumed=int(data["loops_consumed"]),
            dev_delta_curve=tuple(float(item) for item in data.get("dev_delta_curve", [])),
            map_change_refs=tuple(str(item) for item in data.get("map_change_refs", [])),
            cost_ledger_ref=str(data["cost_ledger_ref"]),
            cost_ledger_hash=str(data["cost_ledger_hash"]),
            balance_ledger_ref=str(data["balance_ledger_ref"]),
            balance_ledger_hash=str(data["balance_ledger_hash"]),
            cost_audit_ref=str(data["cost_audit_ref"]),
            balance_audit_ref=str(data["balance_audit_ref"]),
            anchor_proposal_ref=str(data["anchor_proposal_ref"]),
            anchor_merkle_root=str(data["anchor_merkle_root"]),
            anchor_payload_hash=str(data["anchor_payload_hash"]),
            public_trace_ref=str(data["public_trace_ref"]),
            proof_refs=tuple(str(item) for item in data.get("proof_refs", [])),
            prior_receipt_refs=tuple(str(item) for item in data.get("prior_receipt_refs", [])),
            cost_ledger_attested=bool(data.get("cost_ledger_attested", True)),
            contract_version=str(data.get("contract_version", RECEIPT_LEDGER_AUDIT_CONTRACT_VERSION)),
            state=str(data.get("state", ReceiptV2State.LOCAL_RECEIPT_STUB.value)),
            local_only=bool(data.get("local_only", True)),
            public_payload_sanitized=bool(data.get("public_payload_sanitized", True)),
            production_published=bool(data.get("production_published", False)),
            live_publication_enabled=bool(data.get("live_publication_enabled", False)),
            audit_publication_ref=str(data.get("audit_publication_ref", PENDING_LEDGER_AUDIT_PUBLICATION_REF)),
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
        data["dev_delta_curve"] = list(self.dev_delta_curve)
        data["map_change_refs"] = list(self.map_change_refs)
        data["proof_refs"] = list(self.proof_refs)
        data["prior_receipt_refs"] = list(self.prior_receipt_refs)
        return data

    def public_payload(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "brief_summary_sanitized": self.brief_summary_sanitized,
            "credited_funder_ref": self.credited_funder_ref,
            "island": self.island,
            "stop_reason": self.stop_reason,
            "loops_consumed": self.loops_consumed,
            "dev_delta_curve": list(self.dev_delta_curve),
            "map_change_refs": list(self.map_change_refs),
            "cost_ledger_ref": self.cost_ledger_ref,
            "cost_ledger_hash": self.cost_ledger_hash,
            "balance_ledger_ref": self.balance_ledger_ref,
            "balance_ledger_hash": self.balance_ledger_hash,
            "cost_audit_ref": self.cost_audit_ref,
            "balance_audit_ref": self.balance_audit_ref,
            "anchor_proposal_ref": self.anchor_proposal_ref,
            "anchor_merkle_root": self.anchor_merkle_root,
            "anchor_payload_hash": self.anchor_payload_hash,
            "public_trace_ref": self.public_trace_ref,
            "proof_refs": list(self.proof_refs),
            "prior_receipt_refs": list(self.prior_receipt_refs),
            "cost_ledger_attested": self.cost_ledger_attested,
        }


@dataclass(frozen=True)
class BalanceLedgerAuditRecord:
    audit_id: str
    balance_ledger_ref: str
    balance_ledger_hash: str
    anchor_proposal_ref: str
    anchor_merkle_root: str
    anchor_payload_hash: str
    receipt_refs: tuple[str, ...]
    entry_hashes: tuple[str, ...]
    row_count: int
    opening_available_cents: int
    credited_cents: int
    debited_cents: int
    returned_cents: int
    closing_available_cents: int
    mismatch_reasons: tuple[str, ...] = ()
    fraud_evidence_refs: tuple[str, ...] = ()
    contract_version: str = RECEIPT_LEDGER_AUDIT_CONTRACT_VERSION
    state: str = LedgerAuditState.LOCAL_MATCHED.value
    local_only: bool = True
    mutable_summary_only: bool = False
    source_summary_ref: str = ""
    audit_publication_ref: str = PENDING_LEDGER_AUDIT_PUBLICATION_REF
    balance_mutation_performed: bool = False
    payment_executed: bool = False
    bounty_paid: bool = False
    production_write_performed: bool = False
    audit_published_live: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BalanceLedgerAuditRecord":
        return cls(
            audit_id=str(data["audit_id"]),
            balance_ledger_ref=str(data["balance_ledger_ref"]),
            balance_ledger_hash=str(data["balance_ledger_hash"]),
            anchor_proposal_ref=str(data["anchor_proposal_ref"]),
            anchor_merkle_root=str(data["anchor_merkle_root"]),
            anchor_payload_hash=str(data["anchor_payload_hash"]),
            receipt_refs=tuple(str(item) for item in data.get("receipt_refs", [])),
            entry_hashes=tuple(str(item) for item in data.get("entry_hashes", [])),
            row_count=int(data["row_count"]),
            opening_available_cents=int(data["opening_available_cents"]),
            credited_cents=int(data["credited_cents"]),
            debited_cents=int(data["debited_cents"]),
            returned_cents=int(data["returned_cents"]),
            closing_available_cents=int(data["closing_available_cents"]),
            mismatch_reasons=tuple(str(item) for item in data.get("mismatch_reasons", [])),
            fraud_evidence_refs=tuple(str(item) for item in data.get("fraud_evidence_refs", [])),
            contract_version=str(data.get("contract_version", RECEIPT_LEDGER_AUDIT_CONTRACT_VERSION)),
            state=str(data.get("state", LedgerAuditState.LOCAL_MATCHED.value)),
            local_only=bool(data.get("local_only", True)),
            mutable_summary_only=bool(data.get("mutable_summary_only", False)),
            source_summary_ref=str(data.get("source_summary_ref", "")),
            audit_publication_ref=str(data.get("audit_publication_ref", PENDING_LEDGER_AUDIT_PUBLICATION_REF)),
            balance_mutation_performed=bool(data.get("balance_mutation_performed", False)),
            payment_executed=bool(data.get("payment_executed", False)),
            bounty_paid=bool(data.get("bounty_paid", False)),
            production_write_performed=bool(data.get("production_write_performed", False)),
            audit_published_live=bool(data.get("audit_published_live", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["receipt_refs"] = list(self.receipt_refs)
        data["entry_hashes"] = list(self.entry_hashes)
        data["mismatch_reasons"] = list(self.mismatch_reasons)
        data["fraud_evidence_refs"] = list(self.fraud_evidence_refs)
        return data


@dataclass(frozen=True)
class CostLedgerAuditRecord:
    audit_id: str
    cost_ledger_ref: str
    cost_ledger_hash: str
    anchor_proposal_ref: str
    anchor_merkle_root: str
    anchor_payload_hash: str
    receipt_refs: tuple[str, ...]
    entry_hashes: tuple[str, ...]
    row_count: int
    provider_cost_cents: int
    fabric_cost_cents: int
    verifier_cost_cents: int
    refund_cents: int
    total_cost_cents: int
    discrepancy_cents: int = 0
    mismatch_reasons: tuple[str, ...] = ()
    fraud_evidence_refs: tuple[str, ...] = ()
    contract_version: str = RECEIPT_LEDGER_AUDIT_CONTRACT_VERSION
    state: str = LedgerAuditState.LOCAL_MATCHED.value
    local_only: bool = True
    mutable_summary_only: bool = False
    source_summary_ref: str = ""
    audit_publication_ref: str = PENDING_LEDGER_AUDIT_PUBLICATION_REF
    balance_mutation_performed: bool = False
    payment_executed: bool = False
    bounty_paid: bool = False
    production_write_performed: bool = False
    audit_published_live: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CostLedgerAuditRecord":
        return cls(
            audit_id=str(data["audit_id"]),
            cost_ledger_ref=str(data["cost_ledger_ref"]),
            cost_ledger_hash=str(data["cost_ledger_hash"]),
            anchor_proposal_ref=str(data["anchor_proposal_ref"]),
            anchor_merkle_root=str(data["anchor_merkle_root"]),
            anchor_payload_hash=str(data["anchor_payload_hash"]),
            receipt_refs=tuple(str(item) for item in data.get("receipt_refs", [])),
            entry_hashes=tuple(str(item) for item in data.get("entry_hashes", [])),
            row_count=int(data["row_count"]),
            provider_cost_cents=int(data["provider_cost_cents"]),
            fabric_cost_cents=int(data["fabric_cost_cents"]),
            verifier_cost_cents=int(data["verifier_cost_cents"]),
            refund_cents=int(data["refund_cents"]),
            total_cost_cents=int(data["total_cost_cents"]),
            discrepancy_cents=int(data.get("discrepancy_cents", 0)),
            mismatch_reasons=tuple(str(item) for item in data.get("mismatch_reasons", [])),
            fraud_evidence_refs=tuple(str(item) for item in data.get("fraud_evidence_refs", [])),
            contract_version=str(data.get("contract_version", RECEIPT_LEDGER_AUDIT_CONTRACT_VERSION)),
            state=str(data.get("state", LedgerAuditState.LOCAL_MATCHED.value)),
            local_only=bool(data.get("local_only", True)),
            mutable_summary_only=bool(data.get("mutable_summary_only", False)),
            source_summary_ref=str(data.get("source_summary_ref", "")),
            audit_publication_ref=str(data.get("audit_publication_ref", PENDING_LEDGER_AUDIT_PUBLICATION_REF)),
            balance_mutation_performed=bool(data.get("balance_mutation_performed", False)),
            payment_executed=bool(data.get("payment_executed", False)),
            bounty_paid=bool(data.get("bounty_paid", False)),
            production_write_performed=bool(data.get("production_write_performed", False)),
            audit_published_live=bool(data.get("audit_published_live", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["receipt_refs"] = list(self.receipt_refs)
        data["entry_hashes"] = list(self.entry_hashes)
        data["mismatch_reasons"] = list(self.mismatch_reasons)
        data["fraud_evidence_refs"] = list(self.fraud_evidence_refs)
        return data


def compute_ledger_hash(*, ledger_ref: str, entry_hashes: Sequence[str], row_count: int) -> str:
    return sha256_json(
        {
            "ledger_ref": ledger_ref,
            "entry_hashes": list(entry_hashes),
            "row_count": int(row_count),
        }
    )


def build_receipt_v2(source: Mapping[str, Any]) -> ReceiptV2Record:
    protected_errors = _protected_ledger_audit_payload_errors(source)
    if protected_errors:
        raise ValueError("; ".join(protected_errors))
    provisional = ReceiptV2Record(
        receipt_ref="receipt_v2:pending",
        trajectory_id=str(source["trajectory_id"]),
        brief_summary_sanitized=str(source["brief_summary_sanitized"]),
        credited_funder_ref=str(source["credited_funder_ref"]),
        island=str(source["island"]),
        stop_reason=str(source["stop_reason"]),
        loops_consumed=int(source["loops_consumed"]),
        dev_delta_curve=tuple(float(item) for item in source.get("dev_delta_curve", [])),
        map_change_refs=tuple(str(item) for item in source.get("map_change_refs", [])),
        cost_ledger_ref=str(source["cost_ledger_ref"]),
        cost_ledger_hash=str(source["cost_ledger_hash"]),
        balance_ledger_ref=str(source["balance_ledger_ref"]),
        balance_ledger_hash=str(source["balance_ledger_hash"]),
        cost_audit_ref=str(source["cost_audit_ref"]),
        balance_audit_ref=str(source["balance_audit_ref"]),
        anchor_proposal_ref=str(source["anchor_proposal_ref"]),
        anchor_merkle_root=str(source["anchor_merkle_root"]),
        anchor_payload_hash=str(source["anchor_payload_hash"]),
        public_trace_ref=str(source["public_trace_ref"]),
        proof_refs=tuple(str(item) for item in source.get("proof_refs", [])),
        prior_receipt_refs=tuple(str(item) for item in source.get("prior_receipt_refs", [])),
        cost_ledger_attested=bool(source.get("cost_ledger_attested", True)),
        contains_live_champion_ip=bool(source.get("contains_live_champion_ip", False)),
        contains_sealed_eval_details=bool(source.get("contains_sealed_eval_details", False)),
        contains_raw_evidence_snapshot=bool(source.get("contains_raw_evidence_snapshot", False)),
        contains_private_customer_data=bool(source.get("contains_private_customer_data", False)),
        contains_judge_prompts=bool(source.get("contains_judge_prompts", False)),
    )
    receipt = ReceiptV2Record.from_mapping(
        {
            **provisional.to_dict(),
            "receipt_ref": "receipt_v2:" + sha256_json(provisional.public_payload()),
        }
    )
    errors = validate_receipt_v2_record(receipt)
    if errors:
        raise ValueError("; ".join(errors))
    return receipt


def build_balance_ledger_audit(
    *,
    balance_ledger_ref: str,
    anchor_proposal_ref: str,
    anchor_merkle_root: str,
    anchor_payload_hash: str,
    receipt_refs: Sequence[str],
    entry_hashes: Sequence[str],
    opening_available_cents: int,
    credited_cents: int,
    debited_cents: int,
    returned_cents: int,
) -> BalanceLedgerAuditRecord:
    row_count = len(entry_hashes)
    closing_available_cents = opening_available_cents + credited_cents - debited_cents - returned_cents
    ledger_hash = compute_ledger_hash(
        ledger_ref=balance_ledger_ref,
        entry_hashes=entry_hashes,
        row_count=row_count,
    )
    payload = {
        "balance_ledger_ref": balance_ledger_ref,
        "balance_ledger_hash": ledger_hash,
        "anchor_proposal_ref": anchor_proposal_ref,
    }
    return BalanceLedgerAuditRecord(
        audit_id="balance_ledger_audit:" + sha256_json(payload).split(":", 1)[1][:16],
        balance_ledger_ref=balance_ledger_ref,
        balance_ledger_hash=ledger_hash,
        anchor_proposal_ref=anchor_proposal_ref,
        anchor_merkle_root=anchor_merkle_root,
        anchor_payload_hash=anchor_payload_hash,
        receipt_refs=tuple(receipt_refs),
        entry_hashes=tuple(entry_hashes),
        row_count=row_count,
        opening_available_cents=opening_available_cents,
        credited_cents=credited_cents,
        debited_cents=debited_cents,
        returned_cents=returned_cents,
        closing_available_cents=closing_available_cents,
    )


def build_cost_ledger_audit(
    *,
    cost_ledger_ref: str,
    anchor_proposal_ref: str,
    anchor_merkle_root: str,
    anchor_payload_hash: str,
    receipt_refs: Sequence[str],
    entry_hashes: Sequence[str],
    provider_cost_cents: int,
    fabric_cost_cents: int,
    verifier_cost_cents: int,
    refund_cents: int,
) -> CostLedgerAuditRecord:
    row_count = len(entry_hashes)
    total_cost_cents = provider_cost_cents + fabric_cost_cents + verifier_cost_cents - refund_cents
    ledger_hash = compute_ledger_hash(
        ledger_ref=cost_ledger_ref,
        entry_hashes=entry_hashes,
        row_count=row_count,
    )
    payload = {
        "cost_ledger_ref": cost_ledger_ref,
        "cost_ledger_hash": ledger_hash,
        "anchor_proposal_ref": anchor_proposal_ref,
    }
    return CostLedgerAuditRecord(
        audit_id="cost_ledger_audit:" + sha256_json(payload).split(":", 1)[1][:16],
        cost_ledger_ref=cost_ledger_ref,
        cost_ledger_hash=ledger_hash,
        anchor_proposal_ref=anchor_proposal_ref,
        anchor_merkle_root=anchor_merkle_root,
        anchor_payload_hash=anchor_payload_hash,
        receipt_refs=tuple(receipt_refs),
        entry_hashes=tuple(entry_hashes),
        row_count=row_count,
        provider_cost_cents=provider_cost_cents,
        fabric_cost_cents=fabric_cost_cents,
        verifier_cost_cents=verifier_cost_cents,
        refund_cents=refund_cents,
        total_cost_cents=total_cost_cents,
    )


def validate_receipt_v2_record(
    record: ReceiptV2Record | Mapping[str, Any],
    *,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_ledger_audit_payload_errors(raw)
    if not isinstance(record, ReceiptV2Record):
        record = ReceiptV2Record.from_mapping(record)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.state not in {state.value for state in ReceiptV2State}:
        errors.append(f"unknown receipt v2 state: {record.state}")
    for field in (
        "receipt_ref",
        "trajectory_id",
        "brief_summary_sanitized",
        "credited_funder_ref",
        "island",
        "stop_reason",
        "cost_ledger_ref",
        "cost_ledger_hash",
        "balance_ledger_ref",
        "balance_ledger_hash",
        "cost_audit_ref",
        "balance_audit_ref",
        "anchor_proposal_ref",
        "anchor_merkle_root",
        "anchor_payload_hash",
        "public_trace_ref",
    ):
        if not getattr(record, field):
            errors.append(f"receipt v2 requires {field}")
    if not record.receipt_ref.startswith("receipt_v2:sha256:"):
        errors.append("receipt_ref must be receipt_v2:sha256:-prefixed")
    else:
        expected_ref = "receipt_v2:" + sha256_json(record.public_payload())
        if record.receipt_ref != expected_ref:
            errors.append("receipt_ref does not match canonical public payload hash")
    _append_common_anchor_errors(errors, record)
    if not record.cost_ledger_ref.startswith("cost_ledger:"):
        errors.append("cost_ledger_ref must be cost_ledger:-prefixed")
    if not record.balance_ledger_ref.startswith("balance_ledger:"):
        errors.append("balance_ledger_ref must be balance_ledger:-prefixed")
    for field in ("cost_ledger_hash", "balance_ledger_hash", "anchor_payload_hash"):
        if not getattr(record, field).startswith("sha256:"):
            errors.append(f"{field} must be sha256:-prefixed")
    if not record.cost_audit_ref.startswith("cost_ledger_audit:"):
        errors.append("cost_audit_ref must be cost_ledger_audit:-prefixed")
    if not record.balance_audit_ref.startswith("balance_ledger_audit:"):
        errors.append("balance_audit_ref must be balance_ledger_audit:-prefixed")
    if record.loops_consumed <= 0:
        errors.append("loops_consumed must be positive")
    if not record.dev_delta_curve:
        errors.append("dev_delta_curve must not be empty")
    if not record.map_change_refs:
        errors.append("map_change_refs must not be empty")
    for map_change_ref in record.map_change_refs:
        if not map_change_ref.startswith(("research_map_cell:", "research_map_projection:", "map_change:")):
            errors.append("map_change_refs must reference research map cells/projections or map_change refs")
            break
    if not record.proof_refs:
        errors.append("proof_refs must not be empty")
    for proof_ref in record.proof_refs:
        if not proof_ref.startswith(
            (
                "anchor_inclusion:",
                "evidence_bundle:",
                "measurement_gate:",
                "open_verifier:",
                "sanitized_trace:",
                "cost_ledger:",
                "balance_ledger:",
            )
        ):
            errors.append("proof_refs must reference anchored proof, verifier, trace, or ledger refs")
            break
    for prior_receipt_ref in record.prior_receipt_refs:
        if not prior_receipt_ref.startswith("receipt_v2:"):
            errors.append("prior_receipt_refs must be receipt_v2:-prefixed")
            break
    if not record.public_trace_ref.startswith("sanitized_trace:"):
        errors.append("public_trace_ref must be sanitized_trace:-prefixed")
    if not record.cost_ledger_attested:
        errors.append("receipt v2 requires attested cost ledger")
    if record.audit_publication_ref != PENDING_LEDGER_AUDIT_PUBLICATION_REF:
        errors.append("audit_publication_ref must remain pending in P2.8 local contracts")
    release_errors = validate_release_policy(
        ReleasePolicyRecord(
            artifact_ref=record.receipt_ref,
            artifact_type="receipt",
            visibility_policy=record.visibility_policy,
            artifact_release_state=record.artifact_release_state,
            contains_live_champion_ip=record.contains_live_champion_ip,
            contains_sealed_eval_details=record.contains_sealed_eval_details,
            contains_raw_evidence_snapshot=record.contains_raw_evidence_snapshot,
            contains_private_customer_data=record.contains_private_customer_data,
            contains_judge_prompts=record.contains_judge_prompts,
            reason="P2.8 receipts expose sanitized proof and anchored ledger refs only",
        )
    )
    errors.extend(release_errors)
    if not record.local_only:
        errors.append("P2.8 receipts must remain local_only")
    if not record.public_payload_sanitized:
        errors.append("receipt v2 public payload must be sanitized")
    if record.production_published:
        errors.append("P2.8 receipts must not claim production publication")
    if record.live_publication_enabled:
        errors.append("P2.8 receipts must not enable live publication")
    if record.state == ReceiptV2State.PUBLISHED_LIVE.value:
        errors.append("published_live receipt state is disabled in P2.8 local contracts")
    return errors


def validate_balance_ledger_audit_record(
    record: BalanceLedgerAuditRecord | Mapping[str, Any],
    *,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_ledger_audit_payload_errors(raw)
    if not isinstance(record, BalanceLedgerAuditRecord):
        record = BalanceLedgerAuditRecord.from_mapping(record)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    _append_common_audit_errors(errors, record)
    if not record.audit_id.startswith("balance_ledger_audit:"):
        errors.append("balance audit_id must be balance_ledger_audit:-prefixed")
    if not record.balance_ledger_ref.startswith("balance_ledger:"):
        errors.append("balance_ledger_ref must be balance_ledger:-prefixed")
    expected_hash = compute_ledger_hash(
        ledger_ref=record.balance_ledger_ref,
        entry_hashes=record.entry_hashes,
        row_count=record.row_count,
    )
    if record.balance_ledger_hash != expected_hash:
        errors.append("balance_ledger_hash does not match entry_hashes and row_count")
    arithmetic_closing = (
        record.opening_available_cents
        + record.credited_cents
        - record.debited_cents
        - record.returned_cents
    )
    arithmetic_mismatch = arithmetic_closing != record.closing_available_cents
    _append_state_consistency_errors(errors, record, arithmetic_mismatch=arithmetic_mismatch)
    for field in (
        "opening_available_cents",
        "credited_cents",
        "debited_cents",
        "returned_cents",
        "closing_available_cents",
    ):
        if getattr(record, field) < 0:
            errors.append(f"{field} must be non-negative")
    return errors


def validate_cost_ledger_audit_record(
    record: CostLedgerAuditRecord | Mapping[str, Any],
    *,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_ledger_audit_payload_errors(raw)
    if not isinstance(record, CostLedgerAuditRecord):
        record = CostLedgerAuditRecord.from_mapping(record)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    _append_common_audit_errors(errors, record)
    if not record.audit_id.startswith("cost_ledger_audit:"):
        errors.append("cost audit_id must be cost_ledger_audit:-prefixed")
    if not record.cost_ledger_ref.startswith("cost_ledger:"):
        errors.append("cost_ledger_ref must be cost_ledger:-prefixed")
    expected_hash = compute_ledger_hash(
        ledger_ref=record.cost_ledger_ref,
        entry_hashes=record.entry_hashes,
        row_count=record.row_count,
    )
    if record.cost_ledger_hash != expected_hash:
        errors.append("cost_ledger_hash does not match entry_hashes and row_count")
    expected_total = (
        record.provider_cost_cents
        + record.fabric_cost_cents
        + record.verifier_cost_cents
        - record.refund_cents
    )
    expected_discrepancy = expected_total - record.total_cost_cents
    arithmetic_mismatch = expected_total != record.total_cost_cents or record.discrepancy_cents != 0
    if record.discrepancy_cents != expected_discrepancy:
        errors.append("cost ledger discrepancy_cents must equal expected total minus reported total_cost_cents")
    if record.state == LedgerAuditState.LOCAL_MATCHED.value and record.discrepancy_cents != 0:
        errors.append("cost ledger discrepancy must be zero for matched audits")
    _append_state_consistency_errors(errors, record, arithmetic_mismatch=arithmetic_mismatch)
    for field in (
        "provider_cost_cents",
        "fabric_cost_cents",
        "verifier_cost_cents",
        "refund_cents",
        "total_cost_cents",
    ):
        if getattr(record, field) < 0:
            errors.append(f"{field} must be non-negative")
    return errors


def verify_research_lab_receipt_ledger_audit(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    market_summary = verify_market_foundation()
    anchor_summary = verify_research_lab_on_chain_anchor()
    fixture = _load_fixture(Path(fixture_path))

    anchor_errors = validate_anchor_commit_proposal_record(fixture["anchor_fixture_proposal"])
    _assert(not anchor_errors, "source anchor proposal validates")
    _assert(
        fixture["receipt_v2_source"]["anchor_proposal_ref"] == anchor_summary["proposal_id"],
        "receipt fixture pins current P2.7 anchor proposal",
    )

    balance_audit = build_balance_ledger_audit(**fixture["balance_audit_input"])
    _assert(balance_audit.to_dict() == fixture["expected_balance_audit"], "balance audit is deterministic")
    _assert(
        balance_audit.anchor_proposal_ref == anchor_summary["proposal_id"],
        "balance audit pins current P2.7 anchor proposal",
    )
    _assert(
        balance_audit.anchor_merkle_root == anchor_summary["merkle_root"],
        "balance audit pins current P2.7 anchor merkle root",
    )
    _assert(not validate_balance_ledger_audit_record(balance_audit), "balance ledger audit validates")
    for invalid in fixture["invalid_balance_audits"]:
        record = _fixture_record(fixture, invalid, "expected_balance_audit")
        errors = validate_balance_ledger_audit_record(record)
        _assert(errors, f"invalid balance audit fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)
    mismatch_balance = BalanceLedgerAuditRecord.from_mapping(fixture["valid_mismatch_balance_audit"])
    _assert(not validate_balance_ledger_audit_record(mismatch_balance), "balance mismatch state is representable")

    cost_audit = build_cost_ledger_audit(**fixture["cost_audit_input"])
    _assert(cost_audit.to_dict() == fixture["expected_cost_audit"], "cost audit is deterministic")
    _assert(
        cost_audit.anchor_proposal_ref == anchor_summary["proposal_id"],
        "cost audit pins current P2.7 anchor proposal",
    )
    _assert(
        cost_audit.anchor_merkle_root == anchor_summary["merkle_root"],
        "cost audit pins current P2.7 anchor merkle root",
    )
    _assert(not validate_cost_ledger_audit_record(cost_audit), "cost ledger audit validates")
    for invalid in fixture["invalid_cost_audits"]:
        record = _fixture_record(fixture, invalid, "expected_cost_audit")
        errors = validate_cost_ledger_audit_record(record)
        _assert(errors, f"invalid cost audit fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)
    unknown_cost = CostLedgerAuditRecord.from_mapping(fixture["valid_unknown_cost_audit"])
    _assert(not validate_cost_ledger_audit_record(unknown_cost), "cost unknown state is representable")
    fraud_cost = CostLedgerAuditRecord.from_mapping(fixture["valid_fraud_cost_audit"])
    _assert(not validate_cost_ledger_audit_record(fraud_cost), "cost fraud_suspected state is representable")

    receipt = build_receipt_v2(fixture["receipt_v2_source"])
    _assert(receipt.to_dict() == fixture["expected_receipt_v2"], "receipt v2 is deterministic")
    _assert(receipt.balance_audit_ref == balance_audit.audit_id, "receipt pins built balance audit id")
    _assert(receipt.cost_audit_ref == cost_audit.audit_id, "receipt pins built cost audit id")
    _assert(receipt.balance_ledger_hash == balance_audit.balance_ledger_hash, "receipt pins balance ledger hash")
    _assert(receipt.cost_ledger_hash == cost_audit.cost_ledger_hash, "receipt pins cost ledger hash")
    _assert(receipt.anchor_proposal_ref == balance_audit.anchor_proposal_ref, "receipt pins balance audit anchor")
    _assert(receipt.anchor_proposal_ref == cost_audit.anchor_proposal_ref, "receipt pins cost audit anchor")
    _assert(not validate_receipt_v2_record(receipt), "receipt v2 validates")
    for invalid in fixture["invalid_receipts_v2"]:
        record = _fixture_record(fixture, invalid, "expected_receipt_v2")
        errors = validate_receipt_v2_record(record)
        _assert(errors, f"invalid receipt v2 fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    unsafe_errors = validate_cost_ledger_audit_record(
        cost_audit,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_errors, "unsafe Phase 2 guards block ledger audit")
    _assert_expected_error(unsafe_errors, fixture["unsafe_workflow_guards"])

    return {
        "market_dependency_gates": market_summary["dependency_gates"],
        "anchor_proposal": anchor_summary["proposal_id"],
        "receipt_ref": receipt.receipt_ref,
        "balance_audit_id": balance_audit.audit_id,
        "cost_audit_id": cost_audit.audit_id,
        "balance_state": balance_audit.state,
        "cost_state": cost_audit.state,
    }


def _append_common_anchor_errors(errors: list[str], record: Any) -> None:
    if not record.anchor_proposal_ref.startswith("anchor_proposal:"):
        errors.append("anchor_proposal_ref must be anchor_proposal:-prefixed")
    if not record.anchor_merkle_root.startswith("merkle_root:"):
        errors.append("anchor_merkle_root must be merkle_root:-prefixed")
    if not record.anchor_payload_hash.startswith("sha256:"):
        errors.append("anchor_payload_hash must be sha256:-prefixed")


def _append_common_audit_errors(
    errors: list[str],
    record: BalanceLedgerAuditRecord | CostLedgerAuditRecord,
) -> None:
    if record.state not in {state.value for state in LedgerAuditState}:
        errors.append(f"unknown ledger audit state: {record.state}")
    _append_common_anchor_errors(errors, record)
    if record.contract_version != RECEIPT_LEDGER_AUDIT_CONTRACT_VERSION:
        errors.append("contract_version must match P2.8 receipt/ledger audit contract")
    if not record.local_only:
        errors.append("P2.8 ledger audits must remain local_only")
    if record.mutable_summary_only:
        errors.append("ledger audits must reference anchored artifacts, not mutable summaries only")
    if record.source_summary_ref and not (record.anchor_proposal_ref and record.anchor_payload_hash):
        errors.append("source_summary_ref requires anchored artifact refs")
    if record.audit_publication_ref != PENDING_LEDGER_AUDIT_PUBLICATION_REF:
        errors.append("audit_publication_ref must remain pending in P2.8 local contracts")
    if record.balance_mutation_performed:
        errors.append("P2.8 ledger audits must not mutate balances")
    if record.payment_executed:
        errors.append("P2.8 ledger audits must not execute payments")
    if record.bounty_paid:
        errors.append("P2.8 ledger audits must not pay bounties")
    if record.production_write_performed:
        errors.append("P2.8 ledger audits must not write production state")
    if record.audit_published_live:
        errors.append("P2.8 ledger audits must not publish live audit results")
    if record.state == LedgerAuditState.PRODUCTION_VERIFIED.value:
        errors.append("production_verified audit state is disabled in P2.8 local contracts")
    if not record.receipt_refs:
        errors.append("ledger audits require receipt_refs")
    if not record.entry_hashes:
        errors.append("ledger audits require entry_hashes")
    if record.row_count != len(record.entry_hashes):
        errors.append("row_count must match entry_hashes length")
    for entry_hash in record.entry_hashes:
        if not entry_hash.startswith("sha256:"):
            errors.append("entry_hashes must be sha256:-prefixed")
            break
    for receipt_ref in record.receipt_refs:
        if not receipt_ref.startswith("receipt_v2:"):
            errors.append("receipt_refs must be receipt_v2:-prefixed")
            break


def _append_state_consistency_errors(
    errors: list[str],
    record: BalanceLedgerAuditRecord | CostLedgerAuditRecord,
    *,
    arithmetic_mismatch: bool,
) -> None:
    if record.state == LedgerAuditState.LOCAL_MATCHED.value:
        if arithmetic_mismatch:
            errors.append("local_matched audit cannot contain arithmetic mismatch")
        if record.mismatch_reasons:
            errors.append("local_matched audit must not contain mismatch_reasons")
        if record.fraud_evidence_refs:
            errors.append("local_matched audit must not contain fraud_evidence_refs")
    if record.state in {
        LedgerAuditState.LOCAL_MISMATCH.value,
        LedgerAuditState.FRAUD_SUSPECTED.value,
        LedgerAuditState.UNKNOWN.value,
    } and not record.mismatch_reasons:
        errors.append(f"{record.state} audit requires mismatch_reasons")
    if record.state == LedgerAuditState.LOCAL_MISMATCH.value and not arithmetic_mismatch:
        errors.append("local_mismatch audit requires an arithmetic mismatch")
    if record.state == LedgerAuditState.FRAUD_SUSPECTED.value and not record.fraud_evidence_refs:
        errors.append("fraud_suspected audit requires fraud_evidence_refs")


def _protected_ledger_audit_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_ledger_audit_material(record))
    if not found:
        return []
    return ["P2.8 receipt/ledger payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_ledger_audit_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_LEDGER_AUDIT_KEYS and not key_text.endswith(("_ref", "_refs", "_hash", "_hashes")):
                found.add(key_path)
            found.update(_find_protected_ledger_audit_material(nested, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_ledger_audit_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lower = value.lower()
        for marker in PROTECTED_LEDGER_AUDIT_MARKERS:
            if marker in lower:
                found.add(path or marker)
    return found


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fixture_record(fixture: Mapping[str, Any], invalid: Mapping[str, Any], default_base: str) -> dict[str, Any]:
    base = dict(fixture[str(invalid.get("base", default_base))])
    return _deep_merge(base, dict(invalid.get("overrides", {})))


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _assert_expected_error(errors: Sequence[str], record: Mapping[str, Any]) -> None:
    expected = record.get("expected_error_contains")
    if not expected:
        return
    expected_values = [str(item) for item in expected] if isinstance(expected, list) else [str(expected)]
    for expected_value in expected_values:
        _assert(any(expected_value in error for error in errors), f"expected error {expected_value!r}")


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
