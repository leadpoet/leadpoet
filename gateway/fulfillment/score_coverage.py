"""Shared score-coverage accounting for fulfillment submissions.

The validator feed and lifecycle finalizer must agree on what "scored" means.
A submission is complete only when every revealed lead has the required
validator coverage; one score row for a multi-lead submission is not enough.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class FulfillmentScoreCoverage:
    """Coverage summary for one fulfillment request."""

    expected_leads: int
    covered_leads: int
    incomplete_submissions: int
    missing_score_slots: int
    malformed_revealed_leads: int
    validator_hotkeys: frozenset[str]
    score_row_count: int = 0
    latest_scored_at: str | None = None

    @property
    def complete(self) -> bool:
        return (
            self.expected_leads > 0
            and self.covered_leads == self.expected_leads
            and self.incomplete_submissions == 0
            and self.malformed_revealed_leads == 0
        )


def _lead_id(entry: Mapping) -> str:
    return str(entry.get("lead_id") or "").strip()


def summarize_score_coverage(
    revealed_submissions: Sequence[Mapping],
    score_rows: Iterable[Mapping],
    *,
    required_validators: int,
) -> FulfillmentScoreCoverage:
    """Return per-lead validator coverage for revealed submissions.

    Scores only count when their ``(submission_id, lead_id)`` pair exists in
    the revealed payload. Duplicate rows from one validator count once.
    Missing or duplicate revealed lead IDs fail closed because the scorer
    cannot correlate them unambiguously.
    """

    quorum = max(1, int(required_validators))
    validators_by_lead: dict[tuple[str, str], set[str]] = {}
    all_validator_hotkeys: set[str] = set()
    score_row_count = 0
    latest_scored_at: str | None = None
    latest_scored_at_value: datetime | None = None

    for row in score_rows:
        score_row_count += 1
        scored_at = str(row.get("scored_at") or "").strip()
        if scored_at:
            try:
                scored_at_value = datetime.fromisoformat(
                    scored_at.replace("Z", "+00:00")
                )
            except ValueError:
                scored_at_value = None
            if scored_at_value is not None and (
                latest_scored_at_value is None
                or scored_at_value > latest_scored_at_value
            ):
                latest_scored_at = scored_at
                latest_scored_at_value = scored_at_value
        submission_id = str(row.get("submission_id") or "").strip()
        lead_id = str(row.get("lead_id") or "").strip()
        validator_hotkey = str(row.get("validator_hotkey") or "").strip()
        if not submission_id or not lead_id or not validator_hotkey:
            continue
        validators_by_lead.setdefault((submission_id, lead_id), set()).add(
            validator_hotkey
        )

    expected_leads = 0
    covered_leads = 0
    incomplete_submissions = 0
    missing_score_slots = 0
    malformed_revealed_leads = 0

    for submission in revealed_submissions:
        submission_id = str(submission.get("submission_id") or "").strip()
        lead_data = submission.get("lead_data") or []
        submission_incomplete = False
        seen_lead_ids: set[str] = set()

        for entry in lead_data:
            expected_leads += 1
            lead_id = _lead_id(entry) if isinstance(entry, Mapping) else ""
            if not submission_id or not lead_id or lead_id in seen_lead_ids:
                malformed_revealed_leads += 1
                missing_score_slots += quorum
                submission_incomplete = True
                continue

            seen_lead_ids.add(lead_id)
            validators = validators_by_lead.get((submission_id, lead_id), set())
            all_validator_hotkeys.update(validators)
            validator_count = len(validators)
            if validator_count >= quorum:
                covered_leads += 1
            else:
                missing_score_slots += quorum - validator_count
                submission_incomplete = True

        if not lead_data:
            submission_incomplete = True
        if submission_incomplete:
            incomplete_submissions += 1

    return FulfillmentScoreCoverage(
        expected_leads=expected_leads,
        covered_leads=covered_leads,
        incomplete_submissions=incomplete_submissions,
        missing_score_slots=missing_score_slots,
        malformed_revealed_leads=malformed_revealed_leads,
        validator_hotkeys=frozenset(all_validator_hotkeys),
        score_row_count=score_row_count,
        latest_scored_at=latest_scored_at,
    )


def missing_lead_data_for_validator(
    submission: Mapping,
    scored_lead_ids: Iterable[str],
) -> list[Mapping]:
    """Return only the revealed leads the validator has not scored yet."""

    scored = {
        str(lead_id or "").strip()
        for lead_id in scored_lead_ids
        if str(lead_id or "").strip()
    }
    missing: list[Mapping] = []
    for entry in submission.get("lead_data") or []:
        if not isinstance(entry, Mapping):
            missing.append(entry)
            continue
        lead_id = _lead_id(entry)
        if not lead_id or lead_id not in scored:
            missing.append(entry)
    return missing
