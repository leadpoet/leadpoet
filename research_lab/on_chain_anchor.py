"""Phase 2 on-chain anchor commit contracts.

P2.7 defines local record shapes for graduating the current transparency-log /
Arweave checkpoint stub into on-chain anchor commits. This module does not
submit transactions, query chains, spend fees, or verify production inclusion.
It only builds and validates local contract records that future live code must
emit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .canonical import sha256_json
from .full_sealing_kms import (
    PROTECTED_SEALING_KEYS,
    PROTECTED_SEALING_MARKERS,
    verify_research_lab_full_sealing_kms,
)
from .market_foundation import (
    MarketWorkflowGuards,
    assert_market_workflows_disabled,
    default_market_workflow_guards,
    validate_market_sealing_posture,
    verify_market_foundation,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "on_chain_anchor_fixtures.json"

ON_CHAIN_ANCHOR_CONTRACT_VERSION = "on_chain_anchor:v1:local_contract"
ANCHOR_MERKLE_CONVENTION = "local_domain_separated_fixture_v1"
LOCAL_ANCHOR_SIGNATURE_PREFIX = "local_anchor_sig:"
PENDING_CHAIN_TX_REF = "chain_tx:pending"
PENDING_INCLUSION_PROOF_REF = "inclusion_proof:pending"

PROTECTED_ANCHOR_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_SEALING_KEYS)
    | {
        "api_key",
        "customer_email",
        "customer_private_data",
        "evidence_content",
        "judge_prompt",
        "live_champion_ip",
        "password",
        "private_customer_data",
        "private_key",
        "raw_content",
        "raw_customer_data",
        "raw_evidence",
        "raw_payload",
        "raw_snapshot",
        "raw_text",
        "sealed_eval",
        "secret",
        "token",
    }
)

PROTECTED_ANCHOR_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_SEALING_MARKERS)
        | {
            "api key",
            "customer email",
            "judge prompt",
            "live champion",
            "private customer",
            "raw content",
            "raw evidence",
            "raw snapshot",
            "sealed eval",
            "sk-live",
        }
    )
)


class AnchorCommitProposalState(str, Enum):
    LOCAL_PROPOSAL = "local_proposal"
    BLOCKED = "blocked"
    SUBMITTED_CHAIN = "submitted_chain"


class ChainTransactionStubState(str, Enum):
    LOCAL_TX_STUB = "local_tx_stub"
    BLOCKED = "blocked"
    SUBMITTED_CHAIN = "submitted_chain"


class InclusionVerificationState(str, Enum):
    LOCAL_NOT_VERIFIED = "local_not_verified"
    FAILED = "failed"
    VERIFIED_CHAIN = "verified_chain"


@dataclass(frozen=True)
class AnchorCommitProposalRecord:
    proposal_id: str
    epoch_ref: str
    transparency_log_ref: str
    arweave_checkpoint_ref: str
    prior_anchor_ref: str
    payload_hashes: tuple[str, ...]
    merkle_root: str
    fabric_signature_ref: str
    proposer_ref: str
    contract_version: str = ON_CHAIN_ANCHOR_CONTRACT_VERSION
    state: str = AnchorCommitProposalState.LOCAL_PROPOSAL.value
    local_only: bool = True
    public_payload_sanitized: bool = True
    on_chain_submission_requested: bool = False
    chain_transaction_submitted: bool = False
    production_anchor_claimed: bool = False
    raw_payload_present: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AnchorCommitProposalRecord":
        return cls(
            proposal_id=str(data["proposal_id"]),
            epoch_ref=str(data["epoch_ref"]),
            transparency_log_ref=str(data["transparency_log_ref"]),
            arweave_checkpoint_ref=str(data["arweave_checkpoint_ref"]),
            prior_anchor_ref=str(data["prior_anchor_ref"]),
            payload_hashes=tuple(str(item) for item in data.get("payload_hashes", [])),
            merkle_root=str(data["merkle_root"]),
            fabric_signature_ref=str(data["fabric_signature_ref"]),
            proposer_ref=str(data["proposer_ref"]),
            contract_version=str(data.get("contract_version", ON_CHAIN_ANCHOR_CONTRACT_VERSION)),
            state=str(data.get("state", AnchorCommitProposalState.LOCAL_PROPOSAL.value)),
            local_only=bool(data.get("local_only", True)),
            public_payload_sanitized=bool(data.get("public_payload_sanitized", True)),
            on_chain_submission_requested=bool(data.get("on_chain_submission_requested", False)),
            chain_transaction_submitted=bool(data.get("chain_transaction_submitted", False)),
            production_anchor_claimed=bool(data.get("production_anchor_claimed", False)),
            raw_payload_present=bool(data.get("raw_payload_present", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["payload_hashes"] = list(self.payload_hashes)
        return data


@dataclass(frozen=True)
class ChainTransactionStubRecord:
    tx_stub_id: str
    proposal_id: str
    chain_ref: str
    extrinsic_payload_hash: str
    signer_ref: str
    signature_ref: str
    merkle_root: str
    anchor_payload_hash: str
    submitted_tx_hash: str = PENDING_CHAIN_TX_REF
    block_hash: str = "block:pending"
    contract_version: str = ON_CHAIN_ANCHOR_CONTRACT_VERSION
    state: str = ChainTransactionStubState.LOCAL_TX_STUB.value
    local_only: bool = True
    transaction_signed: bool = False
    transaction_submitted: bool = False
    broadcast_performed: bool = False
    chain_client_used: bool = False
    fee_spent: bool = False
    production_anchor_claimed: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ChainTransactionStubRecord":
        return cls(
            tx_stub_id=str(data["tx_stub_id"]),
            proposal_id=str(data["proposal_id"]),
            chain_ref=str(data["chain_ref"]),
            extrinsic_payload_hash=str(data["extrinsic_payload_hash"]),
            signer_ref=str(data["signer_ref"]),
            signature_ref=str(data["signature_ref"]),
            merkle_root=str(data["merkle_root"]),
            anchor_payload_hash=str(data["anchor_payload_hash"]),
            submitted_tx_hash=str(data.get("submitted_tx_hash", PENDING_CHAIN_TX_REF)),
            block_hash=str(data.get("block_hash", "block:pending")),
            contract_version=str(data.get("contract_version", ON_CHAIN_ANCHOR_CONTRACT_VERSION)),
            state=str(data.get("state", ChainTransactionStubState.LOCAL_TX_STUB.value)),
            local_only=bool(data.get("local_only", True)),
            transaction_signed=bool(data.get("transaction_signed", False)),
            transaction_submitted=bool(data.get("transaction_submitted", False)),
            broadcast_performed=bool(data.get("broadcast_performed", False)),
            chain_client_used=bool(data.get("chain_client_used", False)),
            fee_spent=bool(data.get("fee_spent", False)),
            production_anchor_claimed=bool(data.get("production_anchor_claimed", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InclusionVerificationRecord:
    verification_id: str
    proposal_id: str
    tx_stub_id: str
    chain_ref: str
    chain_tx_ref: str
    merkle_root: str
    inclusion_proof_ref: str = PENDING_INCLUSION_PROOF_REF
    inclusion_proof_hash: str = "sha256:pending"
    block_hash: str = "block:pending"
    block_number: int = 0
    contract_version: str = ON_CHAIN_ANCHOR_CONTRACT_VERSION
    state: str = InclusionVerificationState.LOCAL_NOT_VERIFIED.value
    local_only: bool = True
    proof_checked: bool = False
    merkle_root_matches: bool = False
    tx_included: bool = False
    finalized: bool = False
    live_chain_query_performed: bool = False
    production_anchor_verified: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "InclusionVerificationRecord":
        return cls(
            verification_id=str(data["verification_id"]),
            proposal_id=str(data["proposal_id"]),
            tx_stub_id=str(data["tx_stub_id"]),
            chain_ref=str(data["chain_ref"]),
            chain_tx_ref=str(data["chain_tx_ref"]),
            merkle_root=str(data["merkle_root"]),
            inclusion_proof_ref=str(data.get("inclusion_proof_ref", PENDING_INCLUSION_PROOF_REF)),
            inclusion_proof_hash=str(data.get("inclusion_proof_hash", "sha256:pending")),
            block_hash=str(data.get("block_hash", "block:pending")),
            block_number=int(data.get("block_number", 0)),
            contract_version=str(data.get("contract_version", ON_CHAIN_ANCHOR_CONTRACT_VERSION)),
            state=str(data.get("state", InclusionVerificationState.LOCAL_NOT_VERIFIED.value)),
            local_only=bool(data.get("local_only", True)),
            proof_checked=bool(data.get("proof_checked", False)),
            merkle_root_matches=bool(data.get("merkle_root_matches", False)),
            tx_included=bool(data.get("tx_included", False)),
            finalized=bool(data.get("finalized", False)),
            live_chain_query_performed=bool(data.get("live_chain_query_performed", False)),
            production_anchor_verified=bool(data.get("production_anchor_verified", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def merkle_payload_hash(payload: Mapping[str, Any]) -> str:
    return sha256_json(payload)


def compute_anchor_merkle_root(payload_hashes: Sequence[str]) -> str:
    """Compute the P2.7 local anchor fixture Merkle root.

    This helper names a local, domain-separated fixture convention. It is not a
    claim of parity with the gateway checkpoint-root helper; selecting one
    canonical production Merkle construction remains a pre-live anchor decision.
    """
    leaves = [_hash_to_bytes(item) for item in payload_hashes]
    if not leaves:
        return ""
    if len(leaves) == 1:
        return "merkle_root:" + hashlib.sha256(b"\x00" + leaves[0]).hexdigest()
    next_power = 1
    while next_power < len(leaves):
        next_power *= 2
    level = leaves + [leaves[-1]] * (next_power - len(leaves))
    while len(level) > 1:
        level = [
            hashlib.sha256(b"\x01" + level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return "merkle_root:" + level[0].hex()


def build_anchor_commit_proposal(
    *,
    epoch_ref: str,
    transparency_log_ref: str,
    arweave_checkpoint_ref: str,
    prior_anchor_ref: str,
    anchor_payloads: Sequence[Mapping[str, Any]],
    fabric_signature_ref: str,
    proposer_ref: str,
) -> AnchorCommitProposalRecord:
    payload_hashes = tuple(merkle_payload_hash(payload) for payload in anchor_payloads)
    merkle_root = compute_anchor_merkle_root(payload_hashes)
    payload = {
        "epoch_ref": epoch_ref,
        "transparency_log_ref": transparency_log_ref,
        "arweave_checkpoint_ref": arweave_checkpoint_ref,
        "prior_anchor_ref": prior_anchor_ref,
        "payload_hashes": payload_hashes,
        "merkle_root": merkle_root,
        "fabric_signature_ref": fabric_signature_ref,
    }
    return AnchorCommitProposalRecord(
        proposal_id="anchor_proposal:" + sha256_json(payload).split(":", 1)[1][:16],
        epoch_ref=epoch_ref,
        transparency_log_ref=transparency_log_ref,
        arweave_checkpoint_ref=arweave_checkpoint_ref,
        prior_anchor_ref=prior_anchor_ref,
        payload_hashes=payload_hashes,
        merkle_root=merkle_root,
        fabric_signature_ref=fabric_signature_ref,
        proposer_ref=proposer_ref,
    )


def build_chain_transaction_stub(
    *,
    proposal: AnchorCommitProposalRecord | Mapping[str, Any],
    chain_ref: str,
    signer_ref: str,
    signature_ref: str,
) -> ChainTransactionStubRecord:
    if not isinstance(proposal, AnchorCommitProposalRecord):
        proposal = AnchorCommitProposalRecord.from_mapping(proposal)
    anchor_payload = {
        "proposal_id": proposal.proposal_id,
        "merkle_root": proposal.merkle_root,
        "payload_hashes": list(proposal.payload_hashes),
    }
    extrinsic_payload = {
        **anchor_payload,
        "chain_ref": chain_ref,
    }
    extrinsic_hash = sha256_json(extrinsic_payload)
    anchor_payload_hash = sha256_json(anchor_payload)
    payload = {
        "proposal_id": proposal.proposal_id,
        "chain_ref": chain_ref,
        "extrinsic_hash": extrinsic_hash,
        "signature_ref": signature_ref,
    }
    return ChainTransactionStubRecord(
        tx_stub_id="chain_tx_stub:" + sha256_json(payload).split(":", 1)[1][:16],
        proposal_id=proposal.proposal_id,
        chain_ref=chain_ref,
        extrinsic_payload_hash=extrinsic_hash,
        signer_ref=signer_ref,
        signature_ref=signature_ref,
        merkle_root=proposal.merkle_root,
        anchor_payload_hash=anchor_payload_hash,
    )


def build_inclusion_verification_record(
    *,
    proposal: AnchorCommitProposalRecord | Mapping[str, Any],
    tx_stub: ChainTransactionStubRecord | Mapping[str, Any],
) -> InclusionVerificationRecord:
    if not isinstance(proposal, AnchorCommitProposalRecord):
        proposal = AnchorCommitProposalRecord.from_mapping(proposal)
    if not isinstance(tx_stub, ChainTransactionStubRecord):
        tx_stub = ChainTransactionStubRecord.from_mapping(tx_stub)
    payload = {
        "proposal_id": proposal.proposal_id,
        "tx_stub_id": tx_stub.tx_stub_id,
        "chain_tx_ref": tx_stub.submitted_tx_hash,
        "merkle_root": proposal.merkle_root,
    }
    return InclusionVerificationRecord(
        verification_id="anchor_inclusion:" + sha256_json(payload).split(":", 1)[1][:16],
        proposal_id=proposal.proposal_id,
        tx_stub_id=tx_stub.tx_stub_id,
        chain_ref=tx_stub.chain_ref,
        chain_tx_ref=tx_stub.submitted_tx_hash,
        merkle_root=proposal.merkle_root,
    )


def validate_anchor_commit_proposal_record(
    record: AnchorCommitProposalRecord | Mapping[str, Any],
    *,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_anchor_payload_errors(raw)
    if not isinstance(record, AnchorCommitProposalRecord):
        record = AnchorCommitProposalRecord.from_mapping(record)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.state not in {state.value for state in AnchorCommitProposalState}:
        errors.append(f"unknown anchor proposal state: {record.state}")
    for field in (
        "proposal_id",
        "epoch_ref",
        "transparency_log_ref",
        "arweave_checkpoint_ref",
        "prior_anchor_ref",
        "merkle_root",
        "fabric_signature_ref",
        "proposer_ref",
    ):
        if not getattr(record, field):
            errors.append(f"anchor proposal requires {field}")
    if not record.proposal_id.startswith("anchor_proposal:"):
        errors.append("proposal_id must be anchor_proposal:-prefixed")
    if not record.payload_hashes:
        errors.append("anchor proposal requires payload_hashes")
    bad_hashes = [item for item in record.payload_hashes if not item.startswith("sha256:")]
    if bad_hashes:
        errors.append("payload_hashes must be sha256:-prefixed")
    if not record.merkle_root.startswith("merkle_root:"):
        errors.append("merkle_root must be merkle_root:-prefixed")
    expected_merkle_root = compute_anchor_merkle_root(record.payload_hashes)
    if record.payload_hashes and record.merkle_root != expected_merkle_root:
        errors.append("merkle_root does not match payload_hashes")
    if not record.fabric_signature_ref.startswith(LOCAL_ANCHOR_SIGNATURE_PREFIX):
        errors.append("local anchor proposal signature must use local_anchor_sig: prefix")
    if record.contract_version != ON_CHAIN_ANCHOR_CONTRACT_VERSION:
        errors.append("contract_version must match P2.7 on-chain anchor contract")
    if not record.local_only:
        errors.append("P2.7 anchor proposals must remain local_only")
    if not record.public_payload_sanitized:
        errors.append("anchor proposal public payload must be sanitized")
    if record.raw_payload_present:
        errors.append("anchor proposal must not contain raw payload")
    if record.on_chain_submission_requested:
        errors.append("P2.7 local contracts must not request on-chain submission")
    if record.chain_transaction_submitted:
        errors.append("P2.7 local contracts must not mark chain transaction submitted")
    if record.production_anchor_claimed:
        errors.append("P2.7 anchor proposals must not claim production anchor")
    if record.state == AnchorCommitProposalState.SUBMITTED_CHAIN.value:
        errors.append("submitted_chain proposal state is disabled in P2.7 local contracts")
    return errors


def validate_chain_transaction_stub_record(
    record: ChainTransactionStubRecord | Mapping[str, Any],
    *,
    proposal: Optional[AnchorCommitProposalRecord | Mapping[str, Any]] = None,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_anchor_payload_errors(raw)
    if not isinstance(record, ChainTransactionStubRecord):
        record = ChainTransactionStubRecord.from_mapping(record)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.state not in {state.value for state in ChainTransactionStubState}:
        errors.append(f"unknown chain transaction stub state: {record.state}")
    for field in (
        "tx_stub_id",
        "proposal_id",
        "chain_ref",
        "extrinsic_payload_hash",
        "signer_ref",
        "signature_ref",
        "merkle_root",
        "anchor_payload_hash",
    ):
        if not getattr(record, field):
            errors.append(f"chain transaction stub requires {field}")
    if not record.tx_stub_id.startswith("chain_tx_stub:"):
        errors.append("tx_stub_id must be chain_tx_stub:-prefixed")
    for field in ("extrinsic_payload_hash", "anchor_payload_hash"):
        if not getattr(record, field).startswith("sha256:"):
            errors.append(f"{field} must be sha256:-prefixed")
    if not record.merkle_root.startswith("merkle_root:"):
        errors.append("merkle_root must be merkle_root:-prefixed")
    if not record.signature_ref.startswith(LOCAL_ANCHOR_SIGNATURE_PREFIX):
        errors.append("local chain transaction signature must use local_anchor_sig: prefix")
    if record.contract_version != ON_CHAIN_ANCHOR_CONTRACT_VERSION:
        errors.append("contract_version must match P2.7 on-chain anchor contract")
    if not record.local_only:
        errors.append("P2.7 chain transaction stubs must remain local_only")
    if record.transaction_signed:
        errors.append("P2.7 local contracts must not mark chain transaction signed")
    if record.transaction_submitted:
        errors.append("P2.7 local contracts must not submit chain transactions")
    if record.broadcast_performed:
        errors.append("P2.7 local contracts must not broadcast chain transactions")
    if record.chain_client_used:
        errors.append("P2.7 local contracts must not use chain clients")
    if record.fee_spent:
        errors.append("P2.7 local contracts must not spend chain fees")
    if record.production_anchor_claimed:
        errors.append("P2.7 chain transaction stubs must not claim production anchor")
    if record.submitted_tx_hash != PENDING_CHAIN_TX_REF:
        errors.append("submitted_tx_hash must remain pending in P2.7 local contracts")
    if record.block_hash != "block:pending":
        errors.append("block_hash must remain pending in P2.7 local contracts")
    if record.state == ChainTransactionStubState.SUBMITTED_CHAIN.value:
        errors.append("submitted_chain transaction state is disabled in P2.7 local contracts")
    if proposal is not None:
        if not isinstance(proposal, AnchorCommitProposalRecord):
            proposal = AnchorCommitProposalRecord.from_mapping(proposal)
        proposal_errors = validate_anchor_commit_proposal_record(proposal)
        if proposal_errors:
            errors.append("source anchor proposal is invalid: " + "; ".join(proposal_errors))
        if record.proposal_id != proposal.proposal_id:
            errors.append("chain transaction proposal_id mismatch")
        if record.merkle_root != proposal.merkle_root:
            errors.append("chain transaction merkle_root mismatch")
        expected_anchor_payload_hash = sha256_json(
            {
                "proposal_id": proposal.proposal_id,
                "merkle_root": proposal.merkle_root,
                "payload_hashes": list(proposal.payload_hashes),
            }
        )
        expected_extrinsic_payload_hash = sha256_json(
            {
                "proposal_id": proposal.proposal_id,
                "merkle_root": proposal.merkle_root,
                "payload_hashes": list(proposal.payload_hashes),
                "chain_ref": record.chain_ref,
            }
        )
        if record.anchor_payload_hash != expected_anchor_payload_hash:
            errors.append("chain transaction anchor_payload_hash mismatch")
        if record.extrinsic_payload_hash != expected_extrinsic_payload_hash:
            errors.append("chain transaction extrinsic_payload_hash mismatch")
    return errors


def validate_inclusion_verification_record(
    record: InclusionVerificationRecord | Mapping[str, Any],
    *,
    proposal: Optional[AnchorCommitProposalRecord | Mapping[str, Any]] = None,
    tx_stub: Optional[ChainTransactionStubRecord | Mapping[str, Any]] = None,
    guards: Optional[MarketWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_anchor_payload_errors(raw)
    if not isinstance(record, InclusionVerificationRecord):
        record = InclusionVerificationRecord.from_mapping(record)
    try:
        assert_market_workflows_disabled(guards or default_market_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.state not in {state.value for state in InclusionVerificationState}:
        errors.append(f"unknown inclusion verification state: {record.state}")
    for field in ("verification_id", "proposal_id", "tx_stub_id", "chain_ref", "chain_tx_ref", "merkle_root"):
        if not getattr(record, field):
            errors.append(f"inclusion verification requires {field}")
    if not record.verification_id.startswith("anchor_inclusion:"):
        errors.append("verification_id must be anchor_inclusion:-prefixed")
    if not record.merkle_root.startswith("merkle_root:"):
        errors.append("merkle_root must be merkle_root:-prefixed")
    if record.contract_version != ON_CHAIN_ANCHOR_CONTRACT_VERSION:
        errors.append("contract_version must match P2.7 on-chain anchor contract")
    if not record.local_only:
        errors.append("P2.7 inclusion verification records must remain local_only")
    if record.live_chain_query_performed:
        errors.append("P2.7 local contracts must not query live chains")
    if record.production_anchor_verified:
        errors.append("P2.7 inclusion verification must not claim production anchor verified")
    if record.chain_tx_ref != PENDING_CHAIN_TX_REF and not record.proof_checked:
        errors.append("non-pending chain_tx_ref requires proof_checked")
    if record.proof_checked and record.inclusion_proof_ref == PENDING_INCLUSION_PROOF_REF:
        errors.append("proof_checked requires real inclusion_proof_ref")
    if record.tx_included and not record.proof_checked:
        errors.append("tx_included requires proof_checked")
    if record.finalized and not record.tx_included:
        errors.append("finalized requires tx_included")
    if record.merkle_root_matches and not record.proof_checked:
        errors.append("merkle_root_matches requires proof_checked")
    if record.state == InclusionVerificationState.VERIFIED_CHAIN.value:
        if record.inclusion_proof_ref == PENDING_INCLUSION_PROOF_REF:
            errors.append("verified_chain inclusion requires inclusion_proof_ref")
        if not record.inclusion_proof_hash.startswith("sha256:") or record.inclusion_proof_hash == "sha256:pending":
            errors.append("verified_chain inclusion requires inclusion_proof_hash")
        if record.block_hash == "block:pending" or record.block_number <= 0:
            errors.append("verified_chain inclusion requires block evidence")
        if not record.proof_checked:
            errors.append("verified_chain inclusion requires proof_checked")
        if not record.merkle_root_matches:
            errors.append("verified_chain inclusion requires merkle_root_matches")
        if not record.tx_included:
            errors.append("verified_chain inclusion requires tx_included")
        if not record.finalized:
            errors.append("verified_chain inclusion requires finalized")
        errors.append("verified_chain inclusion state is disabled in P2.7 local contracts")
    if proposal is not None:
        if not isinstance(proposal, AnchorCommitProposalRecord):
            proposal = AnchorCommitProposalRecord.from_mapping(proposal)
        proposal_errors = validate_anchor_commit_proposal_record(proposal)
        if proposal_errors:
            errors.append("source anchor proposal is invalid: " + "; ".join(proposal_errors))
        if record.proposal_id != proposal.proposal_id:
            errors.append("inclusion verification proposal_id mismatch")
        if record.merkle_root != proposal.merkle_root:
            errors.append("inclusion verification merkle_root mismatch")
    if tx_stub is not None:
        if not isinstance(tx_stub, ChainTransactionStubRecord):
            tx_stub = ChainTransactionStubRecord.from_mapping(tx_stub)
        tx_errors = validate_chain_transaction_stub_record(tx_stub, proposal=proposal)
        if tx_errors:
            errors.append("source chain transaction stub is invalid: " + "; ".join(tx_errors))
        if record.tx_stub_id != tx_stub.tx_stub_id:
            errors.append("inclusion verification tx_stub_id mismatch")
        if record.chain_tx_ref != tx_stub.submitted_tx_hash:
            errors.append("inclusion verification chain_tx_ref mismatch")
    return errors


def verify_research_lab_on_chain_anchor(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    market_summary = verify_market_foundation()
    sealing_summary = verify_research_lab_full_sealing_kms()
    fixture = _load_fixture(Path(fixture_path))

    proposal = build_anchor_commit_proposal(
        epoch_ref=fixture["proposal_input"]["epoch_ref"],
        transparency_log_ref=fixture["proposal_input"]["transparency_log_ref"],
        arweave_checkpoint_ref=fixture["proposal_input"]["arweave_checkpoint_ref"],
        prior_anchor_ref=fixture["proposal_input"]["prior_anchor_ref"],
        anchor_payloads=fixture["proposal_input"]["anchor_payloads"],
        fabric_signature_ref=fixture["proposal_input"]["fabric_signature_ref"],
        proposer_ref=fixture["proposal_input"]["proposer_ref"],
    )
    _assert(proposal.to_dict() == fixture["expected_proposal"], "anchor proposal is deterministic")
    _assert(not validate_anchor_commit_proposal_record(proposal), "anchor proposal validates")
    for invalid in fixture["invalid_proposals"]:
        record = _fixture_record(fixture, invalid, "expected_proposal")
        errors = validate_anchor_commit_proposal_record(record)
        _assert(errors, f"invalid anchor proposal fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)
    unsafe_proposal_errors = validate_anchor_commit_proposal_record(
        proposal,
        guards=fixture["unsafe_workflow_guards"],
    )
    _assert(unsafe_proposal_errors, "unsafe Phase 2 guards block anchor proposal")

    tx_stub = build_chain_transaction_stub(
        proposal=proposal,
        chain_ref=fixture["tx_stub_input"]["chain_ref"],
        signer_ref=fixture["tx_stub_input"]["signer_ref"],
        signature_ref=fixture["tx_stub_input"]["signature_ref"],
    )
    _assert(tx_stub.to_dict() == fixture["expected_tx_stub"], "chain transaction stub is deterministic")
    _assert(not validate_chain_transaction_stub_record(tx_stub, proposal=proposal), "chain transaction stub validates")
    for invalid in fixture["invalid_tx_stubs"]:
        record = _fixture_record(fixture, invalid, "expected_tx_stub")
        errors = validate_chain_transaction_stub_record(record, proposal=proposal)
        _assert(errors, f"invalid chain transaction stub fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    inclusion = build_inclusion_verification_record(proposal=proposal, tx_stub=tx_stub)
    _assert(inclusion.to_dict() == fixture["expected_inclusion"], "inclusion verification is deterministic")
    _assert(
        not validate_inclusion_verification_record(inclusion, proposal=proposal, tx_stub=tx_stub),
        "local inclusion verification validates",
    )
    for invalid in fixture["invalid_inclusions"]:
        record = _fixture_record(fixture, invalid, "expected_inclusion")
        errors = validate_inclusion_verification_record(record, proposal=proposal, tx_stub=tx_stub)
        _assert(errors, f"invalid inclusion verification fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    sealing_errors = validate_market_sealing_posture(fixture["market_sealing_posture_anchor_claim"])
    _assert(sealing_errors, "P2.7 fixture cannot claim P2.0 production on-chain anchor")
    _assert_expected_error(sealing_errors, fixture["market_sealing_posture_anchor_claim"])

    return {
        "market_dependency_gates": market_summary["dependency_gates"],
        "sealing_gate": sealing_summary["gate_id"],
        "proposal_id": proposal.proposal_id,
        "tx_stub_id": tx_stub.tx_stub_id,
        "inclusion_id": inclusion.verification_id,
        "payload_hashes": len(proposal.payload_hashes),
        "merkle_root": proposal.merkle_root,
        "merkle_convention": ANCHOR_MERKLE_CONVENTION,
    }


def _hash_to_bytes(value: str) -> bytes:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return b""
    try:
        return bytes.fromhex(value.split(":", 1)[1])
    except ValueError:
        return b""


def _protected_anchor_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_anchor_material(record))
    if not found:
        return []
    return ["P2.7 anchor payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_anchor_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_ANCHOR_KEYS and not key_text.endswith(("_ref", "_refs", "_hash")):
                found.add(key_path)
            found.update(_find_protected_anchor_material(nested, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_anchor_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lower = value.lower()
        for marker in PROTECTED_ANCHOR_MARKERS:
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
