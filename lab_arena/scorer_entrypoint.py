"""Entrypoint of the Arena-built judge image, run by validators for scoring assignments.

Inside the gVisor sandbox with no network, the shim routes every provider call
the evaluator makes through the validator's worker socket to the gateway
broker. The gateway adds the submission's miner keys, or the organizer's keys
for the public baseline. The validator never receives those keys. This process
reads one scoring input (ICP, output, scorer policy), scores it with
the Lab scorer exactly as the Arena's central path did, and writes the
breakdown list. Placeholder credentials satisfy the evaluator's environment
checks and are stripped by the shim's trusted-scorer mode before matching.

Failures are reported as a judge error document for infrastructure retry.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from lab_arena import contracts, provider_observations, scoring, shim
from qualification.scoring.evaluation_clock import use_evaluation_date

PLACEHOLDER_CREDENTIALS = {name: "arena-placeholder-" + name.lower() for name in scoring.CREDENTIAL_ENV_NAMES}


def score_input(
    document: Dict[str, Any], *,
    on_company_judgments: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    on_terminal_breakdowns: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    if document.get("schema_version") != scoring.SCORING_INPUT_SCHEMA_VERSION:
        raise scoring.ScoringError("scoring input schema is invalid")
    scored_run_id = str(document.get("scored_run_id") or "")
    if not scored_run_id:
        raise scoring.ScoringError("scored_run_id is invalid")
    policy = contracts.validate_scorer_policy(document["scorer_policy"])
    if "provider_observations" in document:
        raise scoring.ScoringError("provider observations use an unsupported transport")
    try:
        icp, observations = provider_observations.extract_scoring_icp(
            document["icp"], policy
        )
    except ValueError as exc:
        raise scoring.ScoringError(str(exc)) from exc
    companies = [dict(item) for item in document["companies"]]
    os.environ[shim.TRUSTED_SCORER_ENV] = "1"
    scoring.apply_policy_to_environment(
        policy,
        environ=os.environ,
        credentials=dict(PLACEHOLDER_CREDENTIALS),
    )
    from lab_arena import company_judgments, contact_policy
    scorer = scoring.lab_scorer(
        policy,
        **(
            {"contact_source_evidence": document.get("contact_source_evidence")}
            if contact_policy.scorer_enabled(policy) else {}
        ),
        **(
            {"provider_observations": observations}
            if observations is not None else {}
        ),
    )
    if policy.get("signal_judgment_policy") == "signal_judgment_v1":
        from lab_arena.signal_cache_control import exchange
        scorer._signal_judgment_exchange = exchange
    item = {"scored_run_id": scored_run_id}
    completed_judgments = []
    completed_breakdowns = []

    def retain_terminal_breakdowns(rows):
        # Preserve runtime scoring semantics for non-resumable rows. Only
        # independently terminal, original-group-valid rows cross the seam.
        durable = []
        for row in rows:
            try:
                durable = scoring.validate_terminal_breakdowns_for_work_item(
                    durable + [row], icp=icp, companies=companies, policy=policy,
                )
            except scoring.ScoringError:
                continue
        completed_breakdowns[:] = durable
        if durable and on_terminal_breakdowns is not None:
            on_terminal_breakdowns(durable)

    def retain_company_judgments(rows):
        completed_judgments[:] = rows
        if on_company_judgments is not None:
            on_company_judgments(rows)

    try:
        with use_evaluation_date(str(document.get("evaluation_date") or "")):
            if company_judgments.enabled(policy):
                cache_context = document.get("company_judgment_cache")
                if not isinstance(cache_context, dict):
                    raise scoring.ScoringError(
                        "company quality cache lease is required"
                    )
                breakdowns, new_judgments = scoring.score_quality_work_item(
                    item,
                    icp=icp,
                    companies=companies,
                    scorer=scorer,
                    cache_context=cache_context,
                    max_scored_companies=int(policy["max_scored_companies"]),
                    on_company_judgments=retain_company_judgments,
                )
            else:
                prior = document.get("prior_terminal_breakdowns")
                if prior is not None:
                    prior = scoring.validate_terminal_breakdowns_for_work_item(
                        prior, icp=icp, companies=companies, policy=policy,
                    )
                breakdowns = scoring.score_work_item(
                    item,
                    icp=icp,
                    companies=companies,
                    scorer=scorer,
                    max_scored_companies=int(policy["max_scored_companies"]),
                    **({"prior_terminal_breakdowns": prior,
                        "on_terminal_breakdowns": retain_terminal_breakdowns}
                       if getattr(scorer, "integrity_policy", False) else {}),
                )
    except (scoring.ScoringError, ValueError) as exc:
        return scoring.build_scoring_failure(
            scored_run_id,
            "judge_error",
            detail=str(exc),
            reason=(
                exc.failure_reason
                if isinstance(exc, scoring.ScoringError)
                else ""
            ),
            **({"company_judgments": completed_judgments}
               if company_judgments.enabled(policy) else {}),
            **({"terminal_breakdowns": completed_breakdowns}
               if completed_breakdowns else {}),
        )
    return scoring.build_scoring_output(
        scored_run_id,
        breakdowns,
        **(
            {"company_judgments": new_judgments}
            if company_judgments.enabled(policy)
            else {}
        ),
    )


def main(argv: Any = None) -> int:
    from arena_reviewer_runtime.checkpoint_io import run_scorer
    from lab_arena import quality_policy
    return run_scorer(
        score_input=score_input, scoring=scoring, quality_policy=quality_policy,
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
