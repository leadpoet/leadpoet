"""Prospective, scorer-local hooks for freshly revalidated intent evidence.

Installing a callback is the opt-in. The callback owns candidate provenance,
strict claim equivalence and final signal-row admission. This module does not
persist a verifier reply or substitute a provider response.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
from typing import Any, Awaitable, Callable, Iterator, Mapping, Sequence

from qualification.scoring.signal_judgment_cache import SignalJudgmentCacheError


EvidenceReuseCallback = Callable[[dict[str, Any]], Awaitable[Mapping[str, Any] | None]]
_callback: ContextVar[EvidenceReuseCallback | None] = ContextVar(
    "arena_intent_evidence_reuse", default=None,
)
MAX_EVIDENCE_BYTES = 800_000


@contextmanager
def install_evidence_reuse(callback: EvidenceReuseCallback) -> Iterator[None]:
    """Install a trusted callback for one prospective scoring context only."""
    if not callable(callback):
        raise SignalJudgmentCacheError("intent evidence callback is not callable")
    token = _callback.set(callback)
    try:
        yield
    finally:
        _callback.reset(token)


def current_callback() -> EvidenceReuseCallback | None:
    return _callback.get()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def evidence_fingerprint(evidence_input: Mapping[str, Any]) -> str:
    """Hash the complete frozen source input, never its shortened receipt."""
    return "sha256:" + hashlib.sha256(_canonical(evidence_input)).hexdigest()


def snapshot(
    row: Mapping[str, Any], contents: Mapping[str, Any], *,
    verified_identity_context: Mapping[str, Any], model: str,
    company_quality: bool, decision_context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Separate submitted wording from every other exact native judge input.

    Dates remain in both documents. They require exact equality and cannot be
    broadened by a semantic-equivalence verdict. Transport overflow is a miss.
    """
    try:
        native_row = json.loads(_canonical(row))
        claim = {"description": native_row.pop("claim"),
                 "date": native_row.get("signal_date"),
                 "evidence_bundle": native_row.pop("_evidence_bundle", [])}
        if claim["evidence_bundle"]:
            native_row["_evidence_bundle"] = [
                {key: value for key, value in item.items()
                 if key not in {"description", "snippet"}}
                for item in claim["evidence_bundle"]
            ]
        evidence_input = {
            "schema_version": "leadpoet.intent_evidence_input.v1",
            "row": native_row,
            "contents": contents,
            "verified_identity_context": verified_identity_context,
            "model": model,
            "company_quality": bool(company_quality),
            "decision_context": decision_context,
        }
        from qualification.scoring.evaluation_clock import evaluation_date
        evidence_input["evaluation_date"] = evaluation_date().isoformat()
        encoded = _canonical({"evidence_input": evidence_input, "claim": claim})
        if len(encoded) > MAX_EVIDENCE_BYTES:
            return None
        copied = json.loads(encoded)
        return copied["evidence_input"], copied["claim"]
    except (KeyError, TypeError, ValueError):
        return None


async def invoke(
    callback: EvidenceReuseCallback, operation: str,
    evidence_input: Mapping[str, Any], claim: Mapping[str, Any], *,
    judgment: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Keep transport errors separate from a substantive negative verdict."""
    request = {"operation": operation, "evidence_input": evidence_input,
               "claim": claim}
    if operation == "complete":
        request["judgment"] = judgment
    try:
        # Each call receives a copy; callback mutation cannot alter the judge.
        response = await callback(json.loads(_canonical(request)))
        if response is not None and not isinstance(response, Mapping):
            raise SignalJudgmentCacheError("intent evidence response is invalid")
        return response
    except SignalJudgmentCacheError:
        raise
    except Exception as exc:
        raise SignalJudgmentCacheError("intent evidence exchange unavailable") from exc


def terminal_result(
    value: Any, *, contents: Mapping[str, Any], source_url: str,
) -> bool:
    """Admit direct grounded native results, including proved negatives.

    The caller still applies the final signal date/scoring admission. Recovery,
    uncertain results and provider errors cannot become candidate evidence.
    """
    if not isinstance(value, Mapping) or any(value.get(key) is not None for key in (
        "source_resolution", "identity_clarification", "evidence_clarification",
    )):
        return False
    decision = value.get("decision")
    if decision not in {"approve", "reject"} or value.get("client_ready") is not (decision == "approve"):
        return False
    stage1, stage3 = value.get("stage1"), value.get("stage3")
    if (not isinstance(stage1, Mapping) or stage1.get("status") != "skipped_source_grounded"
            or not isinstance(stage3, Mapping) or stage3.get("error")):
        return False
    verdict = value.get("verdict")
    evaluations = verdict.get("signal_evaluations") if isinstance(verdict, Mapping) else None
    if not isinstance(evaluations, list) or len(evaluations) != 1:
        return False
    item = evaluations[0]
    if not isinstance(item, Mapping) or item.get("verification_mode") != "source_grounded":
        return False
    status = item.get("signal_status")
    if (stage3.get("status") != status or stage3.get("decision") != decision
            or item.get("signal_id") != "signal-1" or item.get("evidence_urls_used") != [source_url]
            or (decision == "approve" and (status != "supported" or item.get("same_entity_check") != "pass"))
            or (decision == "reject" and not (
                (status == "contradicted" and item.get("same_entity_check") == "pass")
                or (status == "wrong_entity" and item.get("same_entity_check") == "fail")
            ))):
        return False
    from qualification.scoring.intent_verification_three_stage import _grounded_exact_text
    texts = [page.get("text", "") for page in contents.get("results", [])
             if isinstance(page, Mapping) and page.get("url") == source_url]
    quote_lists = [item.get(field) or [] for field in ("supporting_quotes", "contradicting_quotes")]
    if any(not isinstance(values, list) for values in quote_lists):
        return False
    quotes = [quote for values in quote_lists for quote in values
              if isinstance(quote, str) and quote.strip()]
    return bool(quotes and all(any(_grounded_exact_text(text, quote) for text in texts)
                               for quote in quotes))


def reusable_evidence_terminal_judgment(
    row: Any, *, evidence_input: Mapping[str, Any], expected_urls: Sequence[str],
    expected_index: int, expected_evidence_type: str,
    allowed_context_urls: Sequence[str] = (),
) -> bool:
    """Apply v2 admission to full fresh source text without changing the row.

    A native negative receipt deliberately keeps only a short paragraph window.
    That window can omit the judge's valid quote. The prospective snapshot is
    complete and separately hash-bound, so validate every quote against that
    exact source, then apply the unchanged terminal rules to a validation copy.
    V1 admission and the persisted native trace are never changed.
    """
    from qualification.scoring.signal_judgment_cache import reusable_terminal_judgment
    from qualification.scoring.intent_verification_three_stage import _grounded_exact_text

    if (not isinstance(row, Mapping) or not isinstance(evidence_input, Mapping)
            or evidence_input.get("schema_version") != "leadpoet.intent_evidence_input.v1"
            or not expected_urls):
        return False
    native_row = evidence_input.get("row")
    contents = evidence_input.get("contents")
    if (not isinstance(native_row, Mapping)
            or native_row.get("claimed_source_urls") != list(expected_urls)
            or not isinstance(contents, Mapping)):
        return False
    pages = contents.get("results")
    if not isinstance(pages, list) or not pages:
        return False
    texts = {}
    allowed = set(expected_urls) | set(allowed_context_urls)
    for page in pages:
        if (not isinstance(page, Mapping) or page.get("url") not in allowed
                or page["url"] in texts or not isinstance(page.get("text"), str)
                or not page["text"].strip()):
            return False
        texts[page["url"]] = page["text"]
    if not set(expected_urls).issubset(texts):
        return False
    verdict = row.get("judge_verdict")
    trace = verdict.get("verification_trace") if isinstance(verdict, Mapping) else None
    if (not isinstance(trace, Mapping)
            or trace.get("extracted_signal_date") != native_row.get("signal_date")
            or any(trace.get(key) is not None for key in (
                "source_resolution", "identity_clarification", "evidence_clarification",
            ))):
        return False
    intent = trace.get("intent_verdict")
    evaluations = intent.get("signal_evaluations") if isinstance(intent, Mapping) else None
    if not isinstance(evaluations, list) or not evaluations:
        return False
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            return False
        used = evaluation.get("evidence_urls_used")
        if (not isinstance(used, list) or not used
                or any(not isinstance(url, str) or url not in texts for url in used)):
            return False
        quotes = []
        for field in ("supporting_quotes", "contradicting_quotes"):
            values = evaluation.get(field) or []
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                return False
            quotes.extend(values)
        if not quotes or not all(any(_grounded_exact_text(texts[url], quote) for url in used) for quote in quotes):
            return False
    validation_row = deepcopy(dict(row))
    validation_row["judge_verdict"]["verification_trace"]["verified_source_context"] = [
        {"url": url, "text": text} for url, text in texts.items()
    ]
    return reusable_terminal_judgment(
        validation_row, expected_urls=expected_urls, expected_index=expected_index,
        expected_evidence_type=expected_evidence_type,
        allowed_context_urls=allowed_context_urls,
    )
