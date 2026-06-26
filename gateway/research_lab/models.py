"""Pydantic models for the Research Lab gateway API."""

from __future__ import annotations

import re
import hashlib
import json
import time
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from gateway.research_lab.config import DEFAULT_LOOP_START_FEE_USD


SECRET_MARKERS = (
    "sk-or-",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
    "service_role",
)

SECRET_KEY_RE = re.compile(r"(?:api[_-]?key|raw[_-]?secret|raw[_-]?openrouter|token|credential)", re.I)
MODEL_TIER_RE = r"^[A-Za-z0-9_.:/-]+$"


class SignedResearchLabRequest(BaseModel):
    miner_hotkey: str = Field(min_length=16)
    signature: str = Field(min_length=16)
    timestamp: int
    idempotency_key: str = Field(min_length=8, max_length=160)

    @model_validator(mode="after")
    def timestamp_is_fresh(self) -> "SignedResearchLabRequest":
        now = int(time.time())
        if abs(now - self.timestamp) > 300:
            raise ValueError("timestamp must be within 5 minutes")
        return self

    def signed_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude={"signature"}, exclude_unset=True, mode="json")


class ResearchLabTicketCreateRequest(SignedResearchLabRequest):
    island: str = Field(min_length=1, max_length=80)
    brief_sanitized_ref: str = Field(min_length=8, max_length=256)
    brief_public_summary: Optional[str] = Field(default=None, max_length=2000)
    requested_loop_count: int = Field(default=1, gt=0, le=100)
    loop_start_fee_required_usd: float = Field(default=DEFAULT_LOOP_START_FEE_USD, ge=0)
    research_model_tier: str = Field(default="default", min_length=1, max_length=80, pattern=MODEL_TIER_RE)
    requested_compute_budget_usd: float = Field(default=5.0, ge=0)
    max_compute_budget_usd: float = Field(default=25.0, ge=0)
    miner_openrouter_key_ref: Optional[str] = Field(default=None, max_length=256)
    miner_openrouter_key_handling: Optional[str] = Field(default=None)

    @field_validator("brief_sanitized_ref", "brief_public_summary", "miner_openrouter_key_ref")
    @classmethod
    def no_raw_secret_refs(cls, value: Optional[str]) -> Optional[str]:
        if value:
            reject_secret_material(value)
        return value

    @model_validator(mode="after")
    def valid_budget_bounds(self) -> "ResearchLabTicketCreateRequest":
        if self.max_compute_budget_usd and self.requested_compute_budget_usd > self.max_compute_budget_usd:
            raise ValueError("requested_compute_budget_usd cannot exceed max_compute_budget_usd")
        return self


class ResearchLabProbeRequest(SignedResearchLabRequest):
    ticket_id: UUID
    probe_ref: str = Field(min_length=8, max_length=256)

    @field_validator("probe_ref")
    @classmethod
    def no_raw_probe_material(cls, value: str) -> str:
        reject_secret_material(value)
        return value


class ResearchLabOpenRouterKeyRegisterRequest(SignedResearchLabRequest):
    openrouter_api_key: str = Field(min_length=1, max_length=512)
    key_label: Optional[str] = Field(default=None, max_length=120)

    @field_validator("key_label")
    @classmethod
    def label_has_no_secret_material(cls, value: Optional[str]) -> Optional[str]:
        if value:
            reject_secret_material(value)
        return value


class ResearchLabLoopStartRequest(SignedResearchLabRequest):
    ticket_id: UUID
    payment_block_hash: Optional[str] = Field(default=None, min_length=8, max_length=160)
    payment_extrinsic_index: Optional[int] = Field(default=None, ge=0)
    credit_id: Optional[str] = Field(default=None, min_length=8, max_length=256)
    miner_openrouter_key_ref: str = Field(min_length=8, max_length=256)
    miner_openrouter_key_handling: str = Field(pattern="^(encrypted_ref|ephemeral_ref)$")
    miner_openrouter_preflight_status: str = Field(pattern="^(passed|failed|not_run)$")
    requested_loop_count: int = Field(default=1, gt=0, le=100)
    research_model_tier: Optional[str] = Field(default=None, min_length=1, max_length=80, pattern=MODEL_TIER_RE)
    requested_compute_budget_usd: Optional[float] = Field(default=None, ge=0)
    max_compute_budget_usd: Optional[float] = Field(default=None, ge=0)

    @field_validator("miner_openrouter_key_ref", "payment_block_hash", "credit_id")
    @classmethod
    def no_raw_loop_start_secret(cls, value: Optional[str]) -> Optional[str]:
        if value:
            reject_secret_material(value)
        return value

    @model_validator(mode="after")
    def valid_budget_bounds(self) -> "ResearchLabLoopStartRequest":
        using_credit = bool(self.credit_id)
        has_block = bool(self.payment_block_hash)
        has_extrinsic = self.payment_extrinsic_index is not None
        if using_credit and (has_block or has_extrinsic):
            raise ValueError("credit_id cannot be combined with payment block/extrinsic fields")
        if not using_credit and not (has_block and has_extrinsic):
            raise ValueError("loop start requires either credit_id or both payment_block_hash and payment_extrinsic_index")
        if has_block != has_extrinsic:
            raise ValueError("payment_block_hash and payment_extrinsic_index must be provided together")
        if (
            self.requested_compute_budget_usd is not None
            and self.max_compute_budget_usd is not None
            and self.requested_compute_budget_usd > self.max_compute_budget_usd
        ):
            raise ValueError("requested_compute_budget_usd cannot exceed max_compute_budget_usd")
        return self


class ResearchLabLoopTopUpRequest(SignedResearchLabRequest):
    ticket_id: UUID
    continue_from_run_id: Optional[UUID] = None
    payment_block_hash: str = Field(min_length=8, max_length=160)
    payment_extrinsic_index: int = Field(ge=0)
    additional_compute_budget_usd: float = Field(gt=0)
    research_model_tier: Optional[str] = Field(default=None, min_length=1, max_length=80, pattern=MODEL_TIER_RE)
    topup_reason: str = Field(default="promising_needs_topup", pattern="^(promising_needs_topup|manual_budget_increase)$")
    miner_openrouter_key_ref: str = Field(min_length=8, max_length=256)
    miner_openrouter_key_handling: str = Field(pattern="^(encrypted_ref|ephemeral_ref)$")
    miner_openrouter_preflight_status: str = Field(pattern="^(passed|failed|not_run)$")

    @field_validator("miner_openrouter_key_ref", "payment_block_hash")
    @classmethod
    def no_raw_topup_secret(cls, value: str) -> str:
        reject_secret_material(value)
        return value


class ResearchLabReceiptCreateRequest(BaseModel):
    internal_run_ref: str = Field(min_length=8, max_length=256)
    ticket_id: UUID
    trajectory_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    loop_start_payment_id: Optional[UUID] = None
    loop_start_credit_id: Optional[str] = Field(default=None, max_length=256)
    miner_hotkey: str = Field(min_length=16)
    island: str = Field(min_length=1, max_length=80)
    receipt_status: str = Field(pattern="^(queued|completed|failed|cancelled|tombstoned)$")
    loop_count: int = Field(default=1, gt=0)
    miner_openrouter_key_ref: Optional[str] = Field(default=None, max_length=256)
    provider_usage: list[dict[str, Any]] = Field(default_factory=list)
    cost_ledger: dict[str, Any] = Field(default_factory=dict)
    receipt_doc: dict[str, Any] = Field(default_factory=dict)
    public_receipt_ref: Optional[str] = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def no_secret_material(self) -> "ResearchLabReceiptCreateRequest":
        reject_secret_material(self.model_dump())
        return self


class ResearchLabCandidateArtifactCreateRequest(BaseModel):
    run_id: UUID
    ticket_id: UUID
    receipt_id: Optional[UUID] = None
    miner_hotkey: str = Field(min_length=16)
    island: str = Field(min_length=1, max_length=80)
    candidate_kind: str = Field(default="patch", pattern="^(patch|image_build)$")
    private_model_manifest: dict[str, Any]
    candidate_patch_manifest: dict[str, Any]
    candidate_model_manifest: Optional[dict[str, Any]] = None
    candidate_source_diff_hash: Optional[str] = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    candidate_build_doc: dict[str, Any] = Field(default_factory=dict)
    hypothesis_doc: dict[str, Any] = Field(default_factory=dict)
    redacted_public_summary: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def no_secret_material(self) -> "ResearchLabCandidateArtifactCreateRequest":
        reject_secret_material(self.model_dump())
        if self.candidate_kind == "image_build":
            if not self.candidate_model_manifest:
                raise ValueError("image_build candidate requires candidate_model_manifest")
            if not self.candidate_source_diff_hash:
                raise ValueError("image_build candidate requires candidate_source_diff_hash")
        return self


class ResearchLabScoreBundleCreateRequest(BaseModel):
    bundle_status: str = Field(default="scored", pattern="^(scored|failed|rejected|tombstoned)$")
    receipt_id: Optional[UUID] = None
    score_bundle: dict[str, Any]

    @model_validator(mode="after")
    def valid_score_bundle_shape(self) -> "ResearchLabScoreBundleCreateRequest":
        reject_secret_material(self.model_dump())
        bundle = self.score_bundle
        if bundle.get("bundle_type") != "research_lab_evaluation_score_bundle":
            raise ValueError("score_bundle must be a Research Lab evaluation score bundle")
        if bundle.get("schema_version") != "1.0":
            raise ValueError("unsupported score bundle schema version")
        if not bundle.get("signature_ref"):
            raise ValueError("score bundle signature_ref is required")
        if bundle.get("score_bundle_hash") != bundle.get("anchored_hash"):
            raise ValueError("score_bundle_hash must match anchored_hash")
        if bundle.get("score_bundle_hash") != _score_bundle_hash(bundle):
            raise ValueError("score bundle hash mismatch")
        reward_path = bundle.get("reward_path") or {}
        if reward_path.get("eligible_for_crown") or reward_path.get("eligible_for_improvement_grant"):
            raise ValueError("score bundle cannot directly create crown or improvement-grant eligibility")
        return self


class ResearchLabCandidateEvaluationResultRequest(BaseModel):
    candidate_id: str = Field(pattern=r"^candidate:[0-9a-f]{64}$")
    candidate_status: str = Field(pattern="^(evaluating|scored|failed|rejected)$")
    evaluator_ref: Optional[str] = Field(default=None, max_length=256)
    reason: Optional[str] = Field(default=None, max_length=500)
    score_bundle: Optional[dict[str, Any]] = None
    result_doc: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_result_shape(self) -> "ResearchLabCandidateEvaluationResultRequest":
        reject_secret_material(self.model_dump())
        if self.candidate_status == "scored" and not self.score_bundle:
            raise ValueError("scored Research Lab candidate requires score_bundle")
        if self.score_bundle:
            bundle = self.score_bundle
            if bundle.get("bundle_type") != "research_lab_evaluation_score_bundle":
                raise ValueError("score_bundle must be a Research Lab evaluation score bundle")
            if bundle.get("schema_version") != "1.0":
                raise ValueError("unsupported score bundle schema version")
            if not bundle.get("signature_ref"):
                raise ValueError("score bundle signature_ref is required")
            if bundle.get("score_bundle_hash") != bundle.get("anchored_hash"):
                raise ValueError("score_bundle_hash must match anchored_hash")
            if bundle.get("score_bundle_hash") != _score_bundle_hash(bundle):
                raise ValueError("score bundle hash mismatch")
        return self


class ResearchLabTicketResponse(BaseModel):
    ticket_id: str
    status: str
    event_id: str
    event_seq: int
    ticket_hash: str


class ResearchLabLoopStartResponse(BaseModel):
    ticket_id: str
    run_id: str
    payment_id: str
    payment_ref: str
    queued: bool
    credit_preserved: bool = False
    credit_id: Optional[str] = None
    status: str


class ResearchLabLoopTopUpResponse(BaseModel):
    ticket_id: str
    run_id: str
    continued_from_run_id: Optional[str] = None
    topup_payment_id: str
    payment_ref: str
    queued: bool
    status: str


class ResearchLabOpenRouterKeyRegisterResponse(BaseModel):
    key_ref: str
    preflight_status: str
    key_hash: str
    limit_remaining: Optional[Any] = None
    limit_reset: Optional[str] = None


class ResearchLabReceiptResponse(BaseModel):
    receipt_id: str
    receipt_hash: str
    status: str


class ResearchLabCandidateArtifactResponse(BaseModel):
    candidate_id: str
    candidate_artifact_hash: str
    candidate_patch_hash: str
    status: str
    event_id: str
    event_seq: int


class ResearchLabCandidateEvaluationResultResponse(BaseModel):
    candidate_id: str
    status: str
    event_id: str
    event_seq: int
    score_bundle_id: Optional[str] = None
    receipt_finalized: bool = False


class ResearchLabScoreBundleResponse(BaseModel):
    score_bundle_id: str
    score_bundle_hash: str
    status: str
    event_id: str
    event_seq: int


def reject_secret_material(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if SECRET_KEY_RE.search(str(key)):
                raise ValueError(f"raw secret field is not allowed: {key}")
            reject_secret_material(item)
    elif isinstance(value, list):
        for item in value:
            reject_secret_material(item)
    elif isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in SECRET_MARKERS):
            raise ValueError("raw provider secret material is not allowed")


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _score_bundle_hash(bundle: dict[str, Any]) -> str:
    excluded = {"score_bundle_hash", "anchored_hash", "signature", "signature_ref"}
    payload = {key: value for key, value in dict(bundle).items() if key not in excluded}
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
