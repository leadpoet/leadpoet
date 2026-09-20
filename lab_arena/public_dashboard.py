"""Small, allow-listed public projections for the Arena dashboard.

The durable Arena rows contain private source and runtime fields.  This module
is the only place where the dashboard shapes are assembled; it never returns a
raw database row or document.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, Mapping, Optional, Sequence

from lab_arena import code_review_policy, contracts, icp_disclosure, source_disclosure, verify


PUBLIC_BASELINE_REPOSITORY = "https://github.com/leadpoet/leadpoet-sales-agent/tree/lab"
DEFAULT_RECENT_ROUND_LIMIT = 30
MAX_RECENT_ROUND_LIMIT = 100


def company_diagnostic(
    breakdown: Mapping[str, Any], company: Mapping[str, Any], *, icp_position: int
) -> dict:
    """Project accepted checks only; never publish verifier prose or evidence."""

    from qualification.scoring.competition import has_verified_primary_intent

    states = {
        "match": "passed", "verified": "passed", "pass": "passed",
        "mismatch": "failed", "fail": "failed", "unavailable": "unavailable",
        "not_evaluated": "not_evaluated", "not_required": "not_required",
    }
    gates = {
        item.get("gate"): item
        for item in breakdown.get("verifier_gate_receipts") or []
        if isinstance(item, Mapping)
    }
    fit = gates.get("company_fit", {})
    dimensions = fit.get("company_fit_dimensions") or {}
    checks = {
        name: states.get(dimensions.get(name), "unavailable")
        for name in ("identity", "industry", "employee_size", "geography", "stage")
    }
    if fit.get("company_fit_stage_required") is False:
        checks["stage"] = "not_required"
    checks["required_attribute"] = states.get(
        fit.get("required_attribute_decision"), "unavailable"
    )
    details = breakdown.get("intent_signals_detail") or []
    if has_verified_primary_intent(details):
        checks["intent"] = "passed"
    elif gates.get("intent_verification", {}).get("decision") == "unavailable":
        checks["intent"] = "unavailable"
    elif details:
        # A supported claim can still be unresolved. Do not call review a mismatch.
        verdicts = [item.get("judge_verdict") or {} for item in details if isinstance(item, Mapping)]
        checks["intent"] = "unavailable" if any(
            item.get("pipeline_decision") in {"review", "unavailable"}
            or item.get("error_class") for item in verdicts
        ) else "failed"
    else:
        checks["intent"] = "not_evaluated"
    checks["intent_details"] = states.get(
        gates.get("intent_details", {}).get("decision"), "not_evaluated"
    )
    contact = breakdown.get("contact_verification") or {}
    checks["contact"] = states.get(contact.get("decision"), "unavailable")
    subchecks = contact.get("subchecks") or {}
    checks["email"] = states.get(
        (subchecks.get("email_verification") or {}).get("status"), "not_evaluated"
    )
    # Publish the failing check name, never a provider's free-text error or PII.
    contact_failure = next((
        name for name in (
            "claim", "identity", "source", "company", "role", "location",
            "email_attribution", "email_verification",
        ) if (subchecks.get(name) or {}).get("status") in {"fail", "unavailable"}
    ), None)
    if contact.get("reason") == "contact_claim_invalid":
        contact_failure = "claim"
    return {
        "icp_position": icp_position,
        "company_index": breakdown["company_index"],
        "company_name": str(company.get("company_name") or "")[:200],
        "qualified": breakdown.get("company_qualified") is True,
        "duplicate_company": breakdown.get("duplicate_company") is True,
        "missing_contact": company.get("contact") is None,
        "checks": checks,
        "contact_failure": contact_failure,
    }


_COST_REASONS = frozenset(
    {
        "eligible",
        "historical_round",
        "stored_output_invalid",
        "provider_calls_inflight",
        "provider_cost_uncertain",
        "execution_cap_exceeded",
        "cost_per_company_exceeded",
    }
)
_COST_COUNTER_KEYS = (
    "settled_microusd",
    "reserved_or_uncertain_microusd",
    "conservative_microusd",
    "inflight_calls",
    "uncertain_calls",
    "refused_calls",
    "call_count",
)
_SUCCESSFUL_CALL_COST_COUNTER_KEYS = (
    "successful_microusd",
    "successful_calls",
    "success_unresolved_microusd",
    "success_unresolved_calls",
)
_ROUND_COLUMNS = (
    "round_id,status,created_at,configuration_doc,participants,"
    "publication_doc,published_at,cancel_reason,promotion_required,"
    "baseline_promoted_at,icp_set_date,evaluation_date"
)


def _timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if moment.tzinfo is None or moment.utcoffset() is None:
        return None
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _configuration_scope(row: Mapping[str, Any]) -> tuple[str, int, str]:
    configuration = row.get("configuration_doc")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    return (
        str(configuration.get("network_name") or "finney"),
        int(configuration.get("netuid") or 71),
        str(configuration.get("mode") or ""),
    )


def _publication(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("publication_doc")
    return value if isinstance(value, Mapping) else {}


def _score(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 100.0 else None


def _safe_integer(value: Any) -> Optional[int]:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > contracts.JSON_SAFE_INTEGER_MAX
    ):
        return None
    return value


def _cost_bucket(
    value: Any, *, successful_call_policy: bool = False
) -> Optional[dict]:
    if not isinstance(value, Mapping):
        return None
    counter_keys = _COST_COUNTER_KEYS + (
        _SUCCESSFUL_CALL_COST_COUNTER_KEYS if successful_call_policy else ()
    )
    projected = {key: _safe_integer(value.get(key)) for key in counter_keys}
    if any(item is None for item in projected.values()):
        return None
    raw_providers = value.get("providers")
    if not isinstance(raw_providers, list) or len(raw_providers) > len(contracts.PROVIDERS):
        return None
    providers = []
    seen = set()
    for raw in raw_providers:
        if not isinstance(raw, Mapping) or raw.get("provider") not in contracts.PROVIDERS:
            return None
        provider = str(raw["provider"])
        if provider in seen:
            return None
        seen.add(provider)
        row = {"provider": provider}
        for key in counter_keys:
            number = _safe_integer(raw.get(key))
            if number is None:
                return None
            row[key] = number
        providers.append(row)
    projected["providers"] = sorted(providers, key=lambda row: row["provider"])
    return projected


def _cost_projection(
    ranking: Mapping[str, Any], *, sourcing_cost_eligibility_policy: Any = None
) -> dict:
    """Allow-list a final cost result; never pass publication fields through."""

    eligible = ranking.get("eligible")
    reason = ranking.get("eligibility_reason")
    if not isinstance(eligible, bool) or reason not in _COST_REASONS:
        return {}
    summary = ranking.get("cost_summary")
    if summary is None and (
        (reason == "historical_round" and eligible)
        or (reason == "stored_output_invalid" and not eligible)
    ):
        return {
            "cost_summary": None,
            "eligible": eligible,
            "eligibility_reason": str(reason),
        }
    if not isinstance(summary, Mapping):
        return {}
    per_icp_policy = (
        sourcing_cost_eligibility_policy
        == contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
    )
    if per_icp_policy:
        scalar_keys = (
            "returned_company_count",
            "qualified_company_count",
            "eligible_icp_count",
            "competition_sourcing_microusd",
            "execution_icp_cap_microusd",
            "cost_per_company_cap_microusd",
        )
        projected_summary = {
            key: _safe_integer(summary.get(key)) for key in scalar_keys
        }
        raw_rows = summary.get("per_icp")
        execution = _cost_bucket(summary.get("execution"), successful_call_policy=True)
        judge = _cost_bucket(summary.get("judge"), successful_call_policy=True)
        if (
            any(item is None for item in projected_summary.values())
            or not isinstance(raw_rows, list)
            or len(raw_rows) != contracts.BENCHMARK_ICP_COUNT
            or execution is None
            or judge is None
        ):
            return {}
        rows = []
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                return {}
            row_reason = raw.get("eligibility_reason")
            row = {
                key: _safe_integer(raw.get(key))
                for key in (
                    "icp_position", "returned_company_count",
                    "qualified_company_count", "competition_sourcing_microusd",
                    "eligibility_cap_microusd",
                )
            }
            row["eligible"] = raw.get("eligible")
            row["eligibility_reason"] = row_reason
            if (
                any(value is None for key, value in row.items()
                    if key not in ("eligible", "eligibility_reason"))
                or not isinstance(row["eligible"], bool)
                or row_reason not in _COST_REASONS
            ):
                return {}
            rows.append(row)
        if {row["icp_position"] for row in rows} != set(
            range(contracts.BENCHMARK_ICP_COUNT)
        ):
            return {}
        projected_summary.update({
            "sourcing_cost_eligibility_policy": (
                contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
            ),
            "per_icp": sorted(rows, key=lambda row: row["icp_position"]),
            "execution": execution,
            "judge": judge,
        })
        return {
            "cost_summary": projected_summary,
            "eligible": eligible,
            "eligibility_reason": str(reason),
        }
    scalar_keys = (
        "returned_company_count",
        "execution_cap_microusd",
        "cost_per_company_cap_microusd",
        "eligibility_cap_microusd",
    )
    projected_summary = {key: _safe_integer(summary.get(key)) for key in scalar_keys}
    if "qualified_company_count" in summary:
        projected_summary["qualified_company_count"] = _safe_integer(summary["qualified_company_count"])
    successful_call_policy = (
        sourcing_cost_eligibility_policy
        == contracts.SUCCESSFUL_CALLS_COST_POLICY
    )
    execution = _cost_bucket(
        summary.get("execution"),
        successful_call_policy=successful_call_policy,
    )
    judge = _cost_bucket(
        summary.get("judge"),
        successful_call_policy=successful_call_policy,
    )
    if any(item is None for item in projected_summary.values()) or execution is None or judge is None:
        return {}
    projected_summary.update({"execution": execution, "judge": judge})
    if successful_call_policy:
        projected_summary.update(
            {
                "sourcing_cost_eligibility_policy": (
                    contracts.SUCCESSFUL_CALLS_COST_POLICY
                ),
                "competition_sourcing_microusd": (
                    execution["successful_microusd"]
                    + execution["success_unresolved_microusd"]
                ),
            }
        )
    return {
        "cost_summary": projected_summary,
        "eligible": eligible,
        "eligibility_reason": str(reason),
    }


def _participants(row: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    publication = _publication(row)
    raw = publication.get("participants") if row.get("status") == "published" else row.get("participants")
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _rankings(row: Mapping[str, Any], key: str) -> Dict[str, Mapping[str, Any]]:
    raw = _publication(row).get(key)
    if not isinstance(raw, list):
        return {}
    return {
        str(item.get("submission_id")): item
        for item in raw
        if isinstance(item, Mapping) and item.get("submission_id")
    }


def _champion_submission_id(row: Mapping[str, Any]) -> Optional[str]:
    decision = _publication(row).get("king_decision")
    if not isinstance(decision, Mapping):
        return None
    outcome = str(decision.get("outcome") or "")
    if outcome == "crowned":
        value = decision.get("winner_submission_id")
    elif outcome in ("defended", "retained_ineligible"):
        value = decision.get("king_submission_id")
    else:
        return None
    return str(value) if value else None


def _baseline_and_champion(row: Mapping[str, Any]) -> tuple[Optional[dict], Optional[dict]]:
    published = row.get("status") == "published"
    configuration = row.get("configuration_doc")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    final = _rankings(row, "final_ranking") if published else {}
    baseline = None
    champion = None
    champion_id = _champion_submission_id(row) if published else None
    for participant in _participants(row):
        submission_id = str(participant.get("submission_id") or "")
        if not submission_id:
            continue
        is_baseline = bool(
            participant.get("is_baseline", participant.get("is_king", False))
        )
        ranking = final.get(submission_id) or {}
        projected = {
            "submission_id": submission_id,
            "miner_hotkey": str(participant.get("miner_hotkey") or ""),
            "final_score": _score(ranking.get("final_score")),
        }
        if published:
            projected.update(
                _cost_projection(
                    ranking,
                    sourcing_cost_eligibility_policy=configuration.get(
                        "sourcing_cost_eligibility_policy"
                    ),
                )
            )
        if is_baseline:
            baseline = projected
        elif submission_id == champion_id:
            champion = projected
    return baseline, champion


def round_summary(row: Mapping[str, Any]) -> dict:
    network_name, netuid, mode = _configuration_scope(row)
    configuration = row.get("configuration_doc")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    schedule = configuration.get("schedule")
    schedule = schedule if isinstance(schedule, Mapping) else {}
    disclosure = icp_disclosure.disclosure_metadata(row) or {}
    baseline, champion = _baseline_and_champion(row)
    if champion is None or row.get("promotion_required") is not True:
        promotion_status = "not_required"
    elif row.get("baseline_promoted_at"):
        promotion_status = "promoted"
    else:
        promotion_status = "pending"
    return {
        "round_id": str(row.get("round_id") or ""),
        "status": str(row.get("status") or ""),
        "mode": mode,
        "network_name": network_name,
        "netuid": netuid,
        "created_at": _timestamp(row.get("created_at")),
        "submission_open": _timestamp(schedule.get("submission_open")),
        "submission_cutoff": _timestamp(schedule.get("submission_cutoff")),
        "icp_set_date": disclosure.get("icp_set_date"),
        "evaluation_date": disclosure.get("evaluation_date"),
        "public_at": disclosure.get("public_at"),
        "published_at": _timestamp(row.get("published_at")),
        "cancel_reason": str(row.get("cancel_reason")) if row.get("cancel_reason") else None,
        "participant_count": len(_participants(row)),
        "baseline": baseline,
        "champion": champion,
        "promotion_status": promotion_status,
    }


def _is_administrative_archive(row: Mapping[str, Any]) -> bool:
    """Identify immutable evidence copies that are not competition rounds."""
    reason = row.get("cancel_reason")
    return bool(
        row.get("status") == "cancelled"
        and isinstance(reason, str)
        and reason.startswith("authorized_")
        and reason.rstrip("0123456789").endswith("_archive")
    )


def _recent_competition_rounds(
    service: Any, *, network_name: str, netuid: int, limit: int
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    seen_round_ids: set[str] = set()
    offset = 0
    while len(rows) < limit:
        page = service._store.list_rounds(
            mode=service._config.mode,
            network_name=network_name,
            netuid=netuid,
            limit=limit,
            offset=offset,
            columns=_ROUND_COLUMNS,
        )
        if not page:
            break
        unseen = []
        for row in page:
            round_id = str(row.get("round_id") or "")
            if round_id in seen_round_ids:
                continue
            seen_round_ids.add(round_id)
            unseen.append(row)
        if not unseen:
            break
        rows.extend(row for row in unseen if not _is_administrative_archive(row))
        offset += len(page)
        if len(page) < limit:
            break
    return rows[:limit]


def competition_snapshot(service: Any, *, limit: int = DEFAULT_RECENT_ROUND_LIMIT) -> dict:
    bounded_limit = max(1, min(int(limit), MAX_RECENT_ROUND_LIMIT))
    network_name, netuid = service._chain_scope()
    pinned_round_id = getattr(service._config, "pinned_round_id", None)
    if pinned_round_id is not None:
        rows = [service._round(str(pinned_round_id))]
    else:
        rows = _recent_competition_rounds(
            service,
            network_name=network_name,
            netuid=netuid,
            limit=bounded_limit,
        )
    summaries = [round_summary(row) for row in rows]
    open_round = next((row for row in summaries if row["status"] == "open"), None)
    latest_round = next(
        (row for row in summaries if row["status"] != "open"),
        None,
    )
    latest_completed = next(
        (row for row in summaries if row["status"] == "published"), None
    )
    if latest_completed is None and pinned_round_id is None:
        published = service._store.list_rounds(
            status="published",
            mode=service._config.mode,
            network_name=network_name,
            netuid=netuid,
            limit=1,
            columns=_ROUND_COLUMNS,
        )
        latest_completed = round_summary(published[0]) if published else None
    return {
        "mode": service._config.mode,
        "network_name": network_name,
        "netuid": netuid,
        "repo_url": PUBLIC_BASELINE_REPOSITORY,
        "open_round": open_round,
        "latest_round": latest_round,
        "latest_completed_round": latest_completed,
        "rounds": summaries,
    }


def _stage1_scores(service: Any, row: Mapping[str, Any]) -> Dict[str, float]:
    # Intermediate scores must not escape before evaluation is published.
    if row.get("status") != "published":
        return {}
    configuration = row.get("configuration_doc")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    per_icp_policy = (
        configuration.get("sourcing_cost_eligibility_policy")
        == contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
    )
    published_stage1 = _rankings(row, "stage1_ranking") if per_icp_policy else {}
    final_rankings = _rankings(row, "final_ranking") if per_icp_policy else {}
    selected: Dict[tuple[str, int], Mapping[str, Any]] = {}
    for run in service._store.list_runs(str(row["round_id"]), stage=1, kind="execute"):
        if run.get("per_icp_score") is None:
            continue
        submission_id = str(run.get("submission_id") or "")
        position = int(run.get("icp_position") or 0)
        key = (submission_id, position)
        current = selected.get(key)
        if current is None or int(run.get("attempt") or 0) > int(current.get("attempt") or 0):
            selected[key] = run
    result: Dict[str, float] = {}
    positions = contracts.stage_positions(1)
    for participant in _participants(row):
        submission_id = str(participant.get("submission_id") or "")
        is_baseline = bool(
            participant.get("is_baseline", participant.get("is_king", False))
        )
        if per_icp_policy and not is_baseline:
            published_score = _score(
                (published_stage1.get(submission_id) or {}).get("stage1_score")
            )
            if submission_id and published_score is not None:
                result[submission_id] = published_score
            continue
        runs = [selected.get((submission_id, position)) for position in positions]
        if submission_id and all(run is not None for run in runs):
            values = [
                float(run["per_icp_score"]) for run in runs if run is not None
            ]
            if per_icp_policy:
                frozen_cost = _cost_projection(
                    final_rankings.get(submission_id) or {},
                    sourcing_cost_eligibility_policy=(
                        contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
                    ),
                ).get("cost_summary")
                if not isinstance(frozen_cost, Mapping):
                    # A per-ICP score without its frozen publication-time cost
                    # basis would expose the raw score under a different rule.
                    continue
                eligibility = {
                    int(item["icp_position"]): bool(item["eligible"])
                    for item in frozen_cost["per_icp"]
                }
                values = [
                    value if eligibility[position] else 0.0
                    for position, value in zip(positions, values)
                ]
            result[submission_id] = verify.stage_score(
                values, len(positions),
            )
    return result


def _submission_lifecycle(
    *, raw_status: str, round_status: str, is_champion: bool,
    final_score: Optional[float],
) -> str:
    if round_status == "open" and raw_status == "accepted":
        return "queued"
    if round_status == "cancelled":
        return "cancelled"
    if round_status == "published":
        if final_score is None:
            return "scoring_failed"
        return "champion" if is_champion else "scored"
    if round_status == "scored":
        return "scored"
    return "scoring"


def _submitted_at(submission: Mapping[str, Any]) -> Optional[str]:
    accepted_at = _timestamp(submission.get("accepted_at"))
    if accepted_at is not None:
        return accepted_at
    if submission.get("status") == "frozen":
        return _timestamp(submission.get("frozen_at"))
    if submission.get("status") == "accepted":
        return _timestamp(submission.get("updated_at"))
    return None


def code_review_summary(
    row: Mapping[str, Any], *, round_row: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime] = None,
) -> dict:
    """Project review metadata, never source or provider-authored prose."""
    document = row.get("code_review_doc") or {}
    result = {
        "status": row.get("code_review_status") or "pending",
        "model": document.get("model"),
        "file_count": document.get("file_count"),
        "source_bytes": document.get("source_bytes"),
        "cost_microusd": document.get("cost_microusd"),
        "review_cost_microusd": document.get("review_cost_microusd"),
        "cost_status": document.get("cost_status"),
        "error_code": document.get("error_code"),
        "categories": document.get("categories") or [],
    }
    if "code_review_attempts" in row:
        result["attempts"] = row["code_review_attempts"]
    if type(document.get("retryable")) is bool:
        result["retryable"] = (
            document["retryable"]
            and row.get("status") != "rejected"
            and code_review_policy.review_retry_available(row)
        )
        if round_row is not None:
            deadline_text = _timestamp(
                ((round_row.get("configuration_doc") or {}).get("schedule") or {}).get("benchmark_deadline")
            )
            deadline = datetime.fromisoformat(deadline_text.replace("Z", "+00:00")) if deadline_text else None
            ready_at = code_review_policy.retry_ready_at(row)
            result["retryable"] = bool(
                result["retryable"] and round_row.get("status") == "open"
                and deadline and now is not None
                and now < deadline
                and (ready_at is None or ready_at < deadline)
            )
    http_status = document.get("provider_http_status")
    if type(http_status) is int and 100 <= http_status <= 599:
        result["provider_http_status"] = http_status
    return result


def submissions_snapshot(service: Any, round_id: str) -> dict:
    row = service._round(round_id)
    round_status = str(row.get("status") or "")
    stage1_scores = _stage1_scores(service, row)
    final_scores = _rankings(row, "final_ranking") if round_status == "published" else {}
    champion_id = _champion_submission_id(row)
    participants = {
        str(item.get("submission_id")): item for item in _participants(row)
        if item.get("submission_id")
    }
    records = service._store.list_submissions(
        round_id,
        columns=(
            "submission_id,round_id,miner_hotkey,status,is_king,updated_at,"
            "accepted_at,frozen_at,source_ref,consent,rejection_rule,"
            "replaced_by_submission_id,code_review_status,code_review_attempts,"
            "code_review_started_at,code_review_doc"
        ),
    )
    submissions = []
    for submission in records:
        raw_status = str(submission.get("status") or "")
        review_excluded = (
            raw_status == "rejected"
            and _timestamp(submission.get("accepted_at")) is not None
            and submission.get("rejection_rule") in (
                "code_review_incomplete", "code_review_rejected"
            )
            and not submission.get("replaced_by_submission_id")
            and not submission.get("is_king")
        )
        if raw_status not in ("accepted", "frozen") and not review_excluded:
            continue
        submission_id = str(submission.get("submission_id") or "")
        participant = participants.get(submission_id)
        # Preserve the outcome of previously public intake after a review
        # exclusion. Never enumerate superseded or never-admitted uploads.
        if round_status != "open" and participant is None and not review_excluded:
            continue
        is_baseline = bool(
            (participant or {}).get(
                "is_baseline",
                (participant or {}).get("is_king", submission.get("is_king", False)),
            )
        )
        is_champion = bool(
            round_status == "published"
            and not is_baseline
            and champion_id == submission_id
        )
        final = final_scores.get(submission_id) or {}
        final_score = _score(final.get("final_score"))
        projected = {
                "submission_id": submission_id,
                "miner_hotkey": str(submission.get("miner_hotkey") or ""),
                "is_baseline": is_baseline,
                "status": _submission_lifecycle(
                    raw_status=raw_status,
                    round_status=round_status,
                    is_champion=is_champion,
                    final_score=final_score,
                ),
                "submitted_at": _submitted_at(submission),
                "stage1_score": stage1_scores.get(submission_id),
                "final_score": final_score,
                "is_champion": is_champion,
                "code": source_disclosure.disclosure_status(
                    submission, service.now(), round_row=row
                ),
            }
        if submission.get("code_review_status") and not is_baseline:
            projected["code_review"] = code_review_summary(submission, round_row=row, now=service.now())
        if review_excluded:
            projected.update({
                "status": "review_rejected" if submission.get("rejection_rule") == "code_review_rejected" else "review_failed",
                "stage1_score": None,
                "final_score": None,
                "is_champion": False,
            })
        if round_status == "published" and not review_excluded:
            configuration = row.get("configuration_doc")
            configuration = (
                configuration if isinstance(configuration, Mapping) else {}
            )
            projected.update(
                _cost_projection(
                    final,
                    sourcing_cost_eligibility_policy=configuration.get(
                        "sourcing_cost_eligibility_policy"
                    ),
                )
            )
        submissions.append(projected)
    return {"round_id": round_id, "submissions": submissions}


__all__ = [
    "competition_snapshot",
    "round_summary",
    "submissions_snapshot",
]
