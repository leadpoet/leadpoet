from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VerificationCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    request_id: str
    run_id: str
    lease_token: str
    attempt: int = Field(ge=1, le=3)
    original_position: int = Field(ge=1, le=50)
    candidate: dict[str, Any]
    normalized_criteria: dict[str, Any]
    request_input: dict[str, Any]
    # Older claim RPCs omit this during a rolling deployment. Defaulting to
    # public API traffic prevents a new worker from accidentally enforcing a
    # canary policy without trusted database provenance.
    request_origin: Literal["api", "admin_preview"] = "api"
    deadline_at: datetime


class SignalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_index: int = Field(ge=0, le=4)
    accepted: bool
    verifier_score: float = Field(ge=0, le=100)
    failure_reason: str | None = Field(default=None, max_length=500)
    detail: dict[str, Any] = Field(default_factory=dict)


class VerificationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["accepted", "rejected", "unavailable"]
    reason_code: str = Field(min_length=1, max_length=120)
    reason_message: str = Field(max_length=2_000)
    verifier_score: float = Field(ge=0, le=100)
    verified_payload: dict[str, Any] | None = None
    signal_decisions: list[SignalDecision] = Field(default_factory=list, max_length=5)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> "VerificationDecision":
        if self.outcome == "accepted" and self.verified_payload is None:
            raise ValueError("accepted decisions require a verified payload")
        if self.outcome != "accepted" and self.verified_payload is not None:
            raise ValueError("non-accepted decisions cannot carry a verified payload")
        encoded = json.dumps(self.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
        if len(encoded) > 128 * 1024:
            raise ValueError("verification decision exceeds the durable storage limit")
        return self
