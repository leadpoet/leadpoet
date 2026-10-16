"""Reviewer turn admission, without grading requests or evidence decisions.

The grading package supplies its existing limits and owns all turn and research
counts. This object only decides when its next existing turn can start. It has
no provider access and cannot build a prompt, select a model, or accept a claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Admission:
    mode: Literal["stop", "ordinary", "submit_only"]
    # Before the deadline the existing judgment reserve forces submit_findings
    # through tool_choice while retaining the original tools in the request.
    # Only the late allowance removes research tools from that request.
    restrict_tools: bool = False
    serialization_expired: bool = False


class ReviewerAdmission:
    def __init__(
        self, *, admission_seconds: float, judgment_reserve_seconds: float,
        serialization_grace_seconds: float, loaded_research: bool,
    ) -> None:
        self._admission_seconds = admission_seconds
        self._judgment_reserve_seconds = judgment_reserve_seconds
        self._serialization_grace_seconds = serialization_grace_seconds
        self._late_submit_pending = loaded_research

    def next_turn(
        self, *, elapsed: float, serialization_retry_pending: bool,
    ) -> Admission:
        late_submit = bool(
            self._late_submit_pending and elapsed >= self._admission_seconds
        )
        self._late_submit_pending = False
        if (
            elapsed >= self._admission_seconds
            and not serialization_retry_pending
            and not late_submit
        ):
            return Admission("stop")
        return Admission(
            "submit_only"
            if elapsed >= self._admission_seconds - self._judgment_reserve_seconds
            else "ordinary",
            restrict_tools=late_submit,
            serialization_expired=bool(
                serialization_retry_pending
                and elapsed >= (
                    self._admission_seconds + self._serialization_grace_seconds
                )
            ),
        )

    def tool_settled(self, *, name: str, has_loaded_evidence: bool) -> None:
        # A selected research tool can settle without a new provider call when
        # the reserve has begun. Existing loaded evidence can still be judged.
        self._late_submit_pending = bool(
            name in {"search_web", "fetch_page"} and has_loaded_evidence
        )
