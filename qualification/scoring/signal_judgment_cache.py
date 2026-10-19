"""Optional exchange for exact Arena intent judgments.

The gateway owns the frozen input scope and accepted evidence. The scorer owns
the fresh company-fit context and validates every returned terminal row before
using it. No exchange is installed for rounds without the signal-cache policy.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
from typing import Any, Awaitable, Callable, Iterator, Mapping, Sequence


class SignalJudgmentCacheError(ValueError):
    """The trusted signal cache exchange failed or returned invalid evidence."""


SignalExchange = Callable[[dict[str, Any]], Awaitable[Mapping[str, Any]]]
_exchange: ContextVar[SignalExchange | None] = ContextVar(
    "arena_signal_judgment_exchange", default=None,
)


@contextmanager
def install_exchange(callback: SignalExchange) -> Iterator[None]:
    """Install one score work item's trusted cache exchange."""

    if not callable(callback):
        raise SignalJudgmentCacheError("signal exchange is not callable")
    token = _exchange.set(callback)
    try:
        yield
    finally:
        _exchange.reset(token)


def current_exchange() -> SignalExchange | None:
    return _exchange.get()


async def invoke_exchange(
    callback: SignalExchange, request: dict[str, Any],
) -> Mapping[str, Any]:
    """Keep transport failures distinct from a substantive verifier zero."""

    try:
        return await callback(request)
    except SignalJudgmentCacheError:
        raise
    except Exception as exc:
        raise SignalJudgmentCacheError("signal cache exchange unavailable") from exc


def _finite_number(value: Any, *, minimum: float, maximum: float) -> bool:
    return (
        type(value) in {int, float}
        and math.isfinite(float(value))
        and minimum <= float(value) <= maximum
    )


def _source_grounded_evaluations(
    trace: Mapping[str, Any], expected_urls: Sequence[str],
    allowed_context_urls: Sequence[str],
) -> list[Mapping[str, Any]] | None:
    intent_verdict = trace.get("intent_verdict")
    evaluations = (
        intent_verdict.get("signal_evaluations")
        if isinstance(intent_verdict, Mapping) else None
    )
    contexts = trace.get("verified_source_context")
    allowed_urls = set(expected_urls) | set(allowed_context_urls)
    if (
        not isinstance(evaluations, list) or not evaluations
        or not isinstance(contexts, list) or not contexts
        or any(
            not isinstance(context, Mapping)
            or context.get("url") not in allowed_urls
            or not isinstance(context.get("text"), str)
            or not context["text"]
            for context in contexts
        )
    ):
        return None
    if not any(context.get("url") in expected_urls for context in contexts):
        return None
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            return None
        used = evaluation.get("evidence_urls_used")
        if (
            evaluation.get("verification_mode") != "source_grounded"
            or not isinstance(used, list) or not used
            or any(url not in allowed_urls for url in used)
            or not any(url in expected_urls for url in used)
        ):
            return None
    return evaluations


def _quote_bound_to_context(
    evaluations: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
) -> bool:
    # Use the verifier's own exact quote rule for retained, cited sources.
    from qualification.scoring.intent_verification_three_stage import (
        _grounded_exact_text,
    )

    texts_by_url = {
        str(context["url"]): str(context["text"])
        for context in contexts
    }
    return all(
        any(
            _grounded_exact_text(texts_by_url.get(str(url), ""), quote)
            for quote in [
                *(evaluation.get("supporting_quotes") or []),
                *(evaluation.get("contradicting_quotes") or []),
            ]
            if isinstance(quote, str) and quote.strip()
            for url in evaluation["evidence_urls_used"]
        )
        for evaluation in evaluations
    )


def reusable_terminal_judgment(
    row: Any, *, expected_urls: Sequence[str], expected_index: int,
    expected_evidence_type: str, allowed_context_urls: Sequence[str] = (),
) -> bool:
    """Admit complete source-grounded positives and proved terminal negatives.

    Provider failures, ambiguous judgments, and deterministic pre-gates are
    cheap to retry and never become shared accepted evidence.
    """

    if (
        not isinstance(row, Mapping)
        or not expected_urls
        or "signal_judgment_reference" in row
        or "counted_in_aggregate" in row
        or row.get("matched_icp_signal") != expected_index
        or row.get("evidence_type") != expected_evidence_type
        or row.get("evidence_urls") != list(expected_urls)
        or not _finite_number(row.get("raw"), minimum=0, maximum=100)
        or not _finite_number(row.get("after_decay"), minimum=0, maximum=100)
        or not _finite_number(row.get("decay"), minimum=0, maximum=1)
        or not _finite_number(row.get("confidence"), minimum=0, maximum=100)
    ):
        return False
    verdict = row.get("judge_verdict")
    if not isinstance(verdict, Mapping) or verdict.get("error_class"):
        return False
    if row.get("claim_support_verdict") != verdict.get("claim_support_verdict"):
        return False
    if (
        verdict.get("decision") == "rejected_verifier_error"
        or verdict.get("pipeline_decision") == "unavailable"
    ):
        return False
    trace = verdict.get("verification_trace")
    if (
        not isinstance(trace, Mapping)
        or trace.get("evidence_url") != expected_urls[0]
        or trace.get("evidence_urls") != list(expected_urls)
        or not trace.get("evidence_source")
        or "extracted_signal_date" not in trace
    ):
        return False
    evaluations = _source_grounded_evaluations(
        trace, expected_urls, allowed_context_urls,
    )
    if evaluations is None:
        return False
    contexts = trace["verified_source_context"]
    decision = verdict.get("decision")
    score = float(row["raw"])
    after_decay = float(row["after_decay"])
    if decision == "verified":
        return bool(
            score > 0 and after_decay > 0
            and 0 < float(row["decay"]) <= 1
            and row.get("date_status") in {"in_window", "uncertain"}
            and row.get("date_verdict") == row.get("date_status")
            and verdict.get("pipeline_decision") == "approve"
            and verdict.get("claim_support_verdict") == "supported"
            and verdict.get("client_ready") is True
            and trace.get("final_disposition") == "approve"
            and all(
                item.get("signal_status") == "supported"
                and item.get("same_entity_check") == "pass"
                and item.get("supporting_quotes")
                for item in evaluations
            )
            and _quote_bound_to_context(evaluations, contexts)
        )
    if score != 0 or after_decay != 0 or float(row["decay"]) != 0:
        return False
    if decision == "rejected_three_stage":
        return bool(
            row.get("date_status") == "verified"
            and row.get("date_verdict") == "not_evaluated"
            and verdict.get("pipeline_decision") == "reject"
            and verdict.get("client_ready") is False
            and trace.get("final_disposition") == "reject"
            and all(
                (
                    item.get("signal_status") == "contradicted"
                    and item.get("same_entity_check") == "pass"
                ) or (
                    item.get("signal_status") == "wrong_entity"
                    and item.get("same_entity_check") == "fail"
                )
                for item in evaluations
            )
            and verdict.get("claim_support_verdict")
            == evaluations[0].get("signal_status")
            and _quote_bound_to_context(evaluations, contexts)
        )
    if decision == "rejected_freshness":
        return bool(
            row.get("date_status") == "out_of_window"
            and row.get("date_verdict") == "out_of_window"
            and verdict.get("date_verdict") == "out_of_window"
            and verdict.get("authoritative_date")
            and verdict.get("pipeline_decision") == "approve"
            and verdict.get("client_ready") is False
            and trace.get("final_disposition") == "approve"
            and all(
                item.get("signal_status") == "supported"
                and item.get("same_entity_check") == "pass"
                for item in evaluations
            )
            and _quote_bound_to_context(evaluations, contexts)
        )
    return False
