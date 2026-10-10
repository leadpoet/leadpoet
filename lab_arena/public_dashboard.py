"""Small, allow-listed public projections for the Arena dashboard.

The durable Arena rows contain private source and runtime fields.  This module
is the only place where the dashboard shapes are assembled; it never returns a
raw database row or document.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
import math
from typing import Any, Dict, Mapping, Optional, Sequence

from lab_arena import code_review_policy, contracts, icp_disclosure, source_disclosure, verify
from lab_arena.store import COMPETITION_CONFIGURATION_FIELDS


PUBLIC_BASELINE_REPOSITORY = "https://github.com/leadpoet/leadpoet-sales-agent/tree/lab"
DEFAULT_RECENT_ROUND_LIMIT = 30
MAX_RECENT_ROUND_LIMIT = 100


def company_diagnostic(
    breakdown: Mapping[str, Any], company: Mapping[str, Any], *, icp_position: int,
    contacts_required: bool = True
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
    result = {
        "icp_position": icp_position,
        "company_index": breakdown["company_index"],
        "company_name": str(company.get("company_name") or "")[:200],
        "qualified": breakdown.get("company_qualified") is True,
        "duplicate_company": breakdown.get("duplicate_company") is True,
        "checks": checks,
    }
    if not contacts_required:
        return result
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
    result.update(
        missing_contact=company.get("contact") is None,
        contact_failure=contact_failure,
    )
    return result


_COST_REASONS = frozenset(
    {
        "eligible",
        "execution_incomplete",
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
_COMPETITION_ROUND_COLUMNS = _ROUND_COLUMNS.replace(
    "configuration_doc",
    ",".join(
        "cfg_%s:configuration_doc->%s::text" % (key, key)
        for key in COMPETITION_CONFIGURATION_FIELDS
    ),
).replace("publication_doc", "publication_doc:lab_arena_competition_publication_v1")


def _competition_round(row: Mapping[str, Any]) -> Mapping[str, Any]:
    if "configuration_doc" in row:
        # Embedded stores can return complete rows regardless of the SELECT.
        return row
    # SQL NULL is an absent key; text 'null' is an explicit JSON null. Keep
    # their distinct legacy defaults and validation behavior.
    configuration = {
        key: json.loads(row["cfg_" + key])
        for key in COMPETITION_CONFIGURATION_FIELDS
        if row.get("cfg_" + key) is not None
    }
    return {**row, "configuration_doc": configuration}


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
    ranking: Mapping[str, Any], *, sourcing_cost_eligibility_policy: Any = None,
    configuration: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Allow-list a final cost result; never pass publication fields through."""

    eligible = ranking.get("eligible")
    reason = ranking.get("eligibility_reason")
    if not isinstance(eligible, bool) or reason not in _COST_REASONS:
        return {}
    summary = ranking.get("cost_summary")
    if summary is None and (
        (reason == "historical_round" and eligible)
        or (reason in ("stored_output_invalid", "execution_incomplete") and not eligible)
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
            or len(raw_rows) != contracts.benchmark_icp_count(configuration)
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
            range(contracts.benchmark_icp_count(configuration))
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
                    configuration=configuration,
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


def round_summary(row: Mapping[str, Any], *, completed_scores: Optional[Mapping[str, Any]] = None) -> dict:
    network_name, netuid, mode = _configuration_scope(row)
    configuration = row.get("configuration_doc")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    schedule = configuration.get("schedule")
    schedule = schedule if isinstance(schedule, Mapping) else {}
    disclosure = icp_disclosure.disclosure_metadata(row) or {}
    baseline, champion = _baseline_and_champion(row)
    if baseline and row.get("status") != "published":
        completed = (completed_scores or {}).get(baseline["submission_id"])
        if completed:
            baseline = {**baseline, "final_score": completed["final_score"], "score_status": "complete"}
    if champion is None or row.get("promotion_required") is not True:
        promotion_status = "not_required"
    elif row.get("baseline_promoted_at"):
        promotion_status = "promoted"
    else:
        promotion_status = "pending"
    return {
        "round_id": str(row.get("round_id") or ""),
        "benchmark_icp_count": contracts.benchmark_icp_count(configuration),
        "promotion_margin": contracts.promotion_margin(configuration),
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
    return icp_disclosure.is_administrative_archive(row)


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
            columns=_COMPETITION_ROUND_COLUMNS,
        )
        if not page:
            break
        unseen = []
        for row in page:
            row = _competition_round(row)
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
    summaries = [round_summary(row, completed_scores=_completed_scores(service, row)) for row in rows]
    summarized_rounds = list(zip(rows, summaries))
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
            columns=_COMPETITION_ROUND_COLUMNS,
        )
        published = [_competition_round(row) for row in published]
        latest_completed = round_summary(published[0]) if published else None
        if latest_completed is not None:
            summarized_rounds.append((published[0], latest_completed))

    pending_promotions = [
        (row, summary) for row, summary in summarized_rounds
        if summary["mode"] == "live" and summary["promotion_status"] == "pending"
    ]
    if pending_promotions:
        # Use the same frozen evaluation-day authority as promotion. Read it
        # once, including when the newer day is outside this page or pinned view.
        latest_day = service._latest_published_day()
        for row, summary in pending_promotions:
            if latest_day is not None and latest_day > service._evaluation_day(row):
                summary["promotion_status"] = "superseded"
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


def _completed_scores(service: Any, row: Mapping[str, Any]) -> Dict[str, Any]:
    configuration = row.get("configuration_doc") or {}
    if (row.get("status") in {"open", "committed", "stage1", "stage1_closed", "published", "cancelled"}
        or configuration.get("execution_sequence_policy") != contracts.BASELINE_SCORED_FIRST_POLICY
        or configuration.get("sourcing_cost_eligibility_policy") != contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY):
        return {}
    # The summary SELECT deliberately excludes private scoring plans.
    full_row = row if "stage1_scoring_plan_doc" in row else service._round(str(row["round_id"]))
    return service.completed_submission_scores(full_row)


def _stage1_scores(service: Any, row: Mapping[str, Any]) -> Dict[str, float]:
    # Legacy stage-one rankings remain private until round publication.
    if row.get("status") != "published":
        return {}
    configuration = row.get("configuration_doc")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    execution_policy = configuration.get("execution_sequence_policy")
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
    positions = contracts.execution_positions(1, execution_policy, configuration)
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
                    configuration=configuration,
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
    if final_score is not None:
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


_EVALUATION_COLUMNS = (
    "run_id,assignment_id,submission_id,kind,status,terminal_cause,runner_hotkey,attempt,"
    "lease_generation,lease_expires_at,"
    "source_commit:result_doc->resource_summary->>validator_source_commit,"
    "source_dirty:result_doc->resource_summary->>validator_source_dirty"
)

_EVALUATION_FAILURE_REASONS = {
    "credential_error": "provider_credentials",
    "stage_closed": "execution_window",
    "judge_error": "review",
    "judge_timeout": "review",
    "provider_error": "provider",
    "model_timeout": "execution",
    "invalid_output": "execution",
    "budget_exhausted": "execution",
    "model_error": "execution",
}


def _recorded_source(record: Mapping[str, Any]) -> tuple:
    commit = record.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        commit = None
    dirty = record.get("source_dirty")
    return commit, dirty if dirty in ("clean", "dirty") else "unknown"


def _has_runtime_identity(run: Mapping[str, Any]) -> bool:
    generation = run.get("lease_generation")
    if type(generation) is not int or generation <= 0:
        return False
    try:
        contracts.require_hotkey(str(run.get("runner_hotkey") or ""))
    except contracts.ArenaContractError:
        return False
    return True


def _runtime_source(run: Mapping[str, Any], starts: Sequence[Mapping[str, Any]]) -> tuple:
    # A completion records the source for that specific historical run. An
    # active lease must use a start receipt with the exact lease generation.
    if run.get("status") in ("accepted", "failed", "submitted"):
        source = _recorded_source(run)
        if source[0] is not None:
            return source
    if not _has_runtime_identity(run):
        return None, "unknown"
    generation = run["lease_generation"]
    sources = set()
    for event in starts:
        event_generation = event.get("start_lease_generation")
        # JSON text projection returns a string; avoid permissive numeric casts.
        if str(event_generation) != str(generation) or isinstance(event_generation, bool):
            continue
        if (event.get("run_id") != run.get("run_id")
                or event.get("runner_hotkey") != run.get("runner_hotkey")
                or event.get("assignment_id") != run.get("assignment_id")
                or event.get("attempt") != run.get("attempt")):
            continue
        sources.add(_recorded_source(event))
    # Conflicting receipts cannot identify one recorded source with certainty.
    return next(iter(sources)) if len(sources) == 1 else (None, "unknown")


def evaluation_progress(
    runs: Sequence[Mapping[str, Any]], now: datetime, *,
    runtime_starts: Sequence[Mapping[str, Any]] = (), outcome: Optional[str] = None,
) -> dict:
    """Project assignment activity and recorded code, never private run data.

    Code versions describe execution/judge processes. Cached judgment authority
    continues to use the separately validated scoring_attribution projection.
    """
    effective = {}
    versions = {}
    starts_by_run: Dict[str, list] = {}
    for event in runtime_starts:
        starts_by_run.setdefault(str(event.get("run_id") or ""), []).append(event)
    for index, run in enumerate(runs):
        if run.get("kind") not in ("execute", "score"):
            continue
        key = run.get("assignment_id") or run.get("run_id") or ("missing", index)
        previous = effective.get(key)
        attempt = run.get("attempt") if type(run.get("attempt")) is int else 0
        priority = (run.get("status") == "accepted", attempt)
        if previous is None or priority > previous[0]:
            effective[key] = (priority, run)
        hotkey = str(run.get("runner_hotkey") or "")
        try:
            contracts.require_hotkey(hotkey)
        except contracts.ArenaContractError:
            continue
        phase = "executing" if run["kind"] == "execute" else "scoring"
        source = _runtime_source(run, starts_by_run.get(str(run.get("run_id") or ""), ()))
        version = (hotkey, phase, *source)
        versions[version] = versions.get(version, 0) + 1
    counts = dict.fromkeys(("queued", "active", "completed", "failed", "retrying"), 0)
    failure_reasons = set()
    validators = set()
    invalid_assignment = False
    submitted = False
    for _, run in effective.values():
        status = run.get("status")
        if status == "accepted":
            counts["completed"] += 1
        elif status == "failed":
            counts["failed"] += 1
            cause = run.get("terminal_cause")
            failure_reasons.add(
                _EVALUATION_FAILURE_REASONS.get(cause, "unknown")
                if isinstance(cause, str) else "unknown"
            )
        elif status == "submitted":
            submitted = True
        else:
            expires = _timestamp(run.get("lease_expires_at"))
            active = status == "leased" and expires and datetime.fromisoformat(expires.replace("Z", "+00:00")) > now
            if active and outcome is None:
                counts["active"] += 1
                hotkey = str(run.get("runner_hotkey") or "")
                try:
                    contracts.require_hotkey(hotkey)
                except contracts.ArenaContractError:
                    invalid_assignment = True
                    continue
                phase = "executing" if run["kind"] == "execute" else "scoring"
                source = _runtime_source(run, starts_by_run.get(str(run.get("run_id") or ""), ()))
                validators.add((hotkey, phase, *source))
            elif outcome is not None:
                counts["failed"] += 1
                failure_reasons.add("unknown")
            else:
                retrying = (
                    type(run.get("attempt")) is int and run["attempt"] > 1
                    or type(run.get("lease_generation")) is int and run["lease_generation"] > 1
                )
                counts["retrying" if retrying else "queued"] += 1
    state = outcome or (
        "unavailable" if invalid_assignment else "evaluating" if counts["active"]
        else "retrying" if counts["retrying"] else "queued" if counts["queued"] or not effective
        else "finalizing" if submitted else "failed" if counts["failed"] else "finalizing"
    )
    # A malformed active identity makes the active identities unavailable.
    if invalid_assignment:
        validators.clear()
    result = {
        "state": state,
        "validators": [
            {"hotkey": key[0], "phase": key[1], "commit": key[2], "working_tree": key[3]}
            for key in sorted(validators, key=lambda key: (key[0], key[1], key[2] or "", key[3]))
        ],
        "active_count": counts["active"],
        "counts": counts,
        "code_versions": [
            {"validator_hotkey": key[0], "phase": key[1], "commit": key[2],
             "working_tree": key[3], "run_count": versions[key]}
            for key in sorted(versions, key=lambda key: (key[0], key[1], key[2] or "", key[3]))
        ],
    }
    if failure_reasons and state != "completed":
        result["failure_reasons"] = sorted(failure_reasons)
    return result


def _attach_evaluations(service: Any, round_id: str, entries: Sequence[dict]) -> None:
    if not entries:
        return
    ids = sorted({entry["submission_id"] for entry in entries})
    runs_by_submission: Dict[str, list] = {sid: [] for sid in ids}
    # Compact reads are limited to public submissions on this returned page.
    for run in service._store.list_runs(round_id, columns=_EVALUATION_COLUMNS, submission_ids=ids):
        if run.get("submission_id") in runs_by_submission:
            runs_by_submission[run["submission_id"]].append(run)
    starts_by_submission: Dict[str, list] = {sid: [] for sid in ids}
    now = service.now()
    live_ids = {entry["submission_id"] for entry in entries if entry["status"] == "scoring"}
    fallback_ids = [
        str(run["run_id"])
        for runs in runs_by_submission.values() for run in runs
        if run.get("run_id") and _has_runtime_identity(run) and (
            run.get("status") == "leased" and run.get("submission_id") in live_ids
            or run.get("status") == "failed" and _recorded_source(run)[0] is None
        )
    ]
    # Accepted historical runs without source metadata remain explicitly
    # unknown. No scan of their provider trajectories can identify a commit.
    if fallback_ids:
        for event in service._store.list_runtime_starts(round_id, run_ids=fallback_ids):
            if event.get("submission_id") in starts_by_submission:
                starts_by_submission[event["submission_id"]].append(event)
    for entry in entries:
        status = entry["status"]
        outcome = ("completed" if status in ("scored", "champion")
                   else "failed" if status in ("scoring_failed", "cancelled", "review_failed", "review_rejected") else None)
        entry["evaluation"] = evaluation_progress(
            runs_by_submission[entry["submission_id"]], now,
            runtime_starts=starts_by_submission[entry["submission_id"]], outcome=outcome,
        )


def submissions_snapshot(service: Any, round_id: str) -> dict:
    row = service._round(round_id)
    round_status = str(row.get("status") or "")
    stage1_scores = _stage1_scores(service, row)
    final_scores = _rankings(row, "final_ranking") if round_status == "published" else {}
    completed_scores = _completed_scores(service, row)
    if round_status != "published":
        final_scores = completed_scores
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
                **({"score_status": "complete"} if submission_id in completed_scores else {}),
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
        if (round_status == "published" or submission_id in completed_scores) and not review_excluded:
            configuration = row.get("configuration_doc")
            configuration = (
                configuration if isinstance(configuration, Mapping) else {}
            )
            projected.update(
                _cost_projection(
                    final,
                    configuration=configuration,
                    sourcing_cost_eligibility_policy=configuration.get(
                        "sourcing_cost_eligibility_policy"
                    ),
                )
            )
        submissions.append(projected)
    if round_status != "open":
        _attach_evaluations(service, round_id, submissions)
    else:
        for item in submissions:
            item["evaluation"] = evaluation_progress([], service.now())
    return {"round_id": round_id, "submissions": submissions}


__all__ = [
    "competition_snapshot",
    "round_summary",
    "submissions_snapshot",
]


def history_snapshot(service: Any, *, cursor: Optional[str] = None, limit: int = 25,
                     day: Optional[str] = None, hotkey: str = "") -> dict:
    """Bounded keyset pagination over immutable, published score records."""
    from lab_arena.service import ServiceError

    network_name, netuid = service._chain_scope()
    limit = max(1, min(int(limit), 50))
    needle = hotkey.strip().lower()
    entries, summaries = [], {}
    before = None
    resume = None
    if cursor:
        try:
            state = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
            if not isinstance(state, list) or len(state) != 5 or not all(isinstance(x, str) for x in state):
                raise ValueError()
            created, round_id, after_submission, cursor_day, cursor_hotkey = state
            if cursor_day != (day or "") or cursor_hotkey != needle:
                raise ValueError()
            if not _timestamp(created) or len(round_id) > 128 or len(after_submission) > 128:
                raise ValueError()
            # Always resolve the scoped round; never trust a cursor's scope or date.
            resume = service._round(round_id)
            if resume.get("status") != "published" or _timestamp(resume.get("created_at")) != created:
                raise ValueError()
            before = (str(resume["created_at"]), round_id)
        except (ValueError, TypeError, KeyError):
            raise ServiceError("history_cursor_invalid", 400)
    else:
        after_submission = ""

    def page(next_row=None, after=""):
        next_cursor = None
        if next_row is not None:
            state = [_timestamp(next_row.get("created_at")), str(next_row["round_id"]), after, day or "", needle]
            next_cursor = base64.urlsafe_b64encode(json.dumps(state).encode()).decode().rstrip("=")
        by_round: Dict[str, list] = {}
        for entry in entries:
            by_round.setdefault(entry["round_id"], []).append(entry)
        for round_id, items in by_round.items():
            _attach_evaluations(service, round_id, items)
        return {"rounds": list(summaries.values()), "submissions": entries, "next_cursor": next_cursor}

    pinned = getattr(service._config, "pinned_round_id", None)
    # At most four compact round queries, even for a miner with no matches.
    # A continuation allows the caller to search older history without rescanning.
    for batch in range(4):
        if pinned is not None:
            rows = [service._round(str(pinned))] if batch == 0 and resume is None else []
        else:
            rows = service._store.list_rounds(
                status="published", mode=service._config.mode,
                network_name=network_name, netuid=netuid, limit=25,
                columns=_ROUND_COLUMNS, history_order=True,
                **({"before_round": before} if before else {}),
                **({"evaluation_date": day} if day else {}),
            )
        fetched = len(rows)
        if resume is not None:
            rows = [resume] + rows
            resume = None
        for row in rows:
            if row.get("status") != "published" or _is_administrative_archive(row):
                continue
            summary = round_summary(row)
            if day and summary.get("evaluation_date") != day:
                continue
            rankings = _rankings(row, "final_ranking")
            champion_id = _champion_submission_id(row)
            last_submission = after_submission
            for participant in sorted(_participants(row), key=lambda item: str(item.get("submission_id") or "")):
                submission_id = str(participant.get("submission_id") or "")
                miner_hotkey = str(participant.get("miner_hotkey") or "")
                if submission_id <= after_submission:
                    continue
                if not submission_id or not miner_hotkey or (needle and needle not in miner_hotkey.lower()):
                    last_submission = submission_id
                    continue
                if len(entries) == limit:
                    return page(row, last_submission)
                final_score = _score((rankings.get(submission_id) or {}).get("final_score"))
                baseline = bool(participant.get("is_baseline", participant.get("is_king", False)))
                champion = not baseline and champion_id == submission_id
                summaries[summary["round_id"]] = summary
                entries.append({
                    "round_id": summary["round_id"], "submission_id": submission_id,
                    "miner_hotkey": miner_hotkey, "is_baseline": baseline,
                    "is_champion": champion, "final_score": final_score,
                    "status": _submission_lifecycle(raw_status="frozen", round_status="published",
                                                    is_champion=champion, final_score=final_score),
                })
                last_submission = submission_id
            after_submission = ""
        if fetched < 25 or pinned is not None:
            return page()
        last = rows[-1]
        before = (str(last["created_at"]), str(last["round_id"]))
    return page(last, max((str(p.get("submission_id") or "") for p in _participants(last)), default=""))
