from __future__ import annotations

import concurrent.futures
import json
import time
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from gateway.research_lab import scoring_worker as sw
from gateway.research_lab.model_authority_v2 import (
    MODEL_QUALIFICATION_AUTHORITY_SCHEMA_V1,
    QualificationOutcomeCompleteV2,
    QualificationOutcomeIncompleteV2Error,
)
from research_lab.canonical import sha256_json
from research_lab.eval.private_runtime import publish_attested_receipt_hash


MODEL_RECEIPT_HASH = "sha256:" + "1" * 64
SCORER_RECEIPT_HASH = "sha256:" + "6" * 64


def _qualification_authority(
    disposition: str,
    *,
    partial_company_count: int = 0,
) -> dict[str, Any]:
    incomplete = disposition.startswith("incomplete_")
    body = {
        "schema_version": MODEL_QUALIFICATION_AUTHORITY_SCHEMA_V1,
        "source_commit": "a" * 40,
        "git_commit_sha": "a" * 40,
        "source_tree_hash": "sha256:" + "b" * 64,
        "model_artifact_digest": "sha256:" + "b" * 64,
        "manifest_hash": "sha256:" + "c" * 64,
        "model_manifest_sha256": "sha256:" + "c" * 64,
        "image_digest": "registry.example/model@sha256:" + "d" * 64,
        "protocol_major": 2,
        "protocol_minor": 0,
        "contract_sha256": "e" * 64,
        "completion_state": "incomplete" if incomplete else "complete",
        "disposition": disposition,
        "retryable": disposition == "incomplete_retryable",
        "failure_classes": (
            [
                "retryable_provider"
                if disposition == "incomplete_retryable"
                else "terminal_auth"
            ]
            if incomplete
            else []
        ),
        "partial_company_count": partial_company_count,
        "invocation_sha256": "f" * 64,
        "input_hash": "sha256:" + "2" * 64,
        "route_completion_receipt_sha256": "3" * 64,
        "provider_terminal_observation_hash": "sha256:" + "4" * 64,
        "host_provider_observation_root": "sha256:" + "5" * 64,
        "execution_receipt_hash": MODEL_RECEIPT_HASH,
    }
    return {**body, "authority_hash": sha256_json(body)}


def _worker() -> sw.ResearchLabGatewayScoringWorker:
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    worker.worker_ref = "qualification-outcome-test-worker"
    worker.config = SimpleNamespace(private_baseline_provider_retry_rounds=1)
    return worker


def _item() -> dict[str, Any]:
    return {
        "icp": {"industry": "software", "max_companies": 3},
        "icp_ref": "icp:qualification-outcome",
        "icp_hash": "hash-qualification-outcome",
        "set_id": 1,
        "day_index": 1,
        "day_rank": 1,
    }


class _UnusedScorer:
    async def score_with_breakdowns(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        raise AssertionError("an incomplete or confirmed-empty result was scored")


@pytest.mark.asyncio
async def test_retryable_incomplete_is_typed_checkpointed_and_never_scored():
    authority = _qualification_authority(
        "incomplete_retryable",
        partial_company_count=1,
    )
    private_partial = {
        "company_name": "private-partial-sentinel",
        "employee_count": "11-50",
    }

    class Runner:
        async def __call__(
            self,
            _icp: Mapping[str, Any],
            _context: Mapping[str, Any],
        ) -> list[dict[str, Any]]:
            publish_attested_receipt_hash(MODEL_RECEIPT_HASH)
            raise QualificationOutcomeIncompleteV2Error(
                "qualification incomplete",
                model_qualification_authority=authority,
                partial_companies=[private_partial],
            )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        result = await _worker()._run_baseline_icp(
            runner=Runner(),
            scorer=_UnusedScorer(),
            item=_item(),
            item_index=1,
            total_icps=1,
            run_start=time.time(),
            executor=executor,
        )
    finally:
        executor.shutdown(wait=False)

    assert result["_retryable"] is True
    assert result["_nonempty"] is False
    assert result["company_count"] == 0
    assert result["diagnostics"]["funnel"]["sourced"] == 1
    assert result[sw._MODEL_QUALIFICATION_AUTHORITY_FIELD] == authority
    assert result[sw._MODEL_QUALIFICATION_PARTIAL_COUNT_FIELD] == 1
    assert result[sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD] == [
        MODEL_RECEIPT_HASH
    ]
    assert "qualification_incomplete_retryable" in result["diagnostics"][
        "failure_categories"
    ]
    assert "runtime_provider_error" not in result["diagnostics"][
        "failure_categories"
    ]
    assert result["diagnostics"]["runtime_error"] == {
        "error_class": "QualificationOutcomeIncompleteV2Error",
        "provider": "sourcing_model",
        "status": 0,
        "category": "qualification_incomplete_retryable",
    }
    assert result["diagnostics"]["model_qualification_failure_classes"] == [
        "retryable_provider"
    ]
    assert sw._baseline_summary_checkpointable(result) is False

    checkpoint = sw._baseline_attempt_checkpoint_row(result, retry_round=0)
    assert checkpoint[sw._MODEL_QUALIFICATION_AUTHORITY_FIELD] == authority
    assert checkpoint[sw._MODEL_QUALIFICATION_PARTIAL_COUNT_FIELD] == 1
    assert checkpoint["_runtime_error"] == "attempt_failed"
    assert "private-partial-sentinel" not in json.dumps(checkpoint, sort_keys=True)


@pytest.mark.asyncio
async def test_confirmed_empty_requires_typed_joined_authority(monkeypatch):
    authority = _qualification_authority("complete_confirmed_empty")

    class Runner:
        async def __call__(
            self,
            _icp: Mapping[str, Any],
            _context: Mapping[str, Any],
        ) -> QualificationOutcomeCompleteV2:
            publish_attested_receipt_hash(MODEL_RECEIPT_HASH)
            return QualificationOutcomeCompleteV2(
                [],
                model_qualification_authority=authority,
            )

    def legacy_empty_gate(*_args: Any, **_kwargs: Any) -> bool:
        raise AssertionError("typed v2 empty reached the legacy paid-call gate")

    monkeypatch.setattr(sw, "_accept_provider_backed_empty_retry", legacy_empty_gate)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        result = await _worker()._run_baseline_icp(
            runner=Runner(),
            scorer=_UnusedScorer(),
            item=_item(),
            item_index=1,
            total_icps=1,
            run_start=time.time(),
            executor=executor,
            retry_round=0,
        )
    finally:
        executor.shutdown(wait=False)

    assert result["_retryable"] is False
    assert result["_nonempty"] is False
    assert result["diagnostics"]["sourcing_failed"] is False
    assert result["diagnostics"]["empty_result_provider_evidence_validated"] is True
    assert result["diagnostics"]["model_qualification_disposition"] == (
        "complete_confirmed_empty"
    )
    assert result[sw._MODEL_QUALIFICATION_AUTHORITY_FIELD] == authority
    assert sw._baseline_summary_checkpointable(result) is True


@pytest.mark.asyncio
async def test_terminal_incomplete_is_not_retried_or_scored(monkeypatch):
    authority = _qualification_authority("incomplete_terminal")
    stage_events: list[dict[str, Any]] = []

    class Runner:
        async def __call__(
            self,
            _icp: Mapping[str, Any],
            _context: Mapping[str, Any],
        ) -> list[dict[str, Any]]:
            publish_attested_receipt_hash(MODEL_RECEIPT_HASH)
            raise QualificationOutcomeIncompleteV2Error(
                "qualification incomplete",
                model_qualification_authority=authority,
            )

    monkeypatch.setattr(
        sw,
        "_record_private_baseline_stage",
        lambda **fields: stage_events.append(dict(fields)),
    )
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        result = await _worker()._run_baseline_icp(
            runner=Runner(),
            scorer=_UnusedScorer(),
            item=_item(),
            item_index=1,
            total_icps=1,
            run_start=time.time(),
            executor=executor,
        )
    finally:
        executor.shutdown(wait=False)

    assert result["_retryable"] is False
    assert result["_nonempty"] is False
    assert "qualification_incomplete_terminal" in result["diagnostics"][
        "failure_categories"
    ]
    assert "runtime_provider_error" not in result["diagnostics"][
        "failure_categories"
    ]
    assert result["diagnostics"]["model_qualification_failure_classes"] == [
        "terminal_auth"
    ]
    assert stage_events[-1]["reason_code"] == "qualification_incomplete_terminal"
    assert stage_events[-1]["retryable"] is False
    assert stage_events[-1]["result_status"] == "failed"


@pytest.mark.asyncio
async def test_complete_nonempty_scores_and_binds_both_receipt_roots():
    authority = _qualification_authority("complete_nonempty")
    company = {"company_name": "complete-company", "employee_count": "11-50"}

    class Runner:
        async def __call__(
            self,
            _icp: Mapping[str, Any],
            _context: Mapping[str, Any],
        ) -> QualificationOutcomeCompleteV2:
            publish_attested_receipt_hash(MODEL_RECEIPT_HASH)
            return QualificationOutcomeCompleteV2(
                [company],
                model_qualification_authority=authority,
            )

    class Scorer:
        async def score_with_breakdowns(
            self,
            _outputs: list[dict[str, Any]],
            _icp: Mapping[str, Any],
            _is_reference: bool,
        ) -> list[dict[str, Any]]:
            publish_attested_receipt_hash(SCORER_RECEIPT_HASH)
            return [{"final_score": 9.0}]

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        result = await _worker()._run_baseline_icp(
            runner=Runner(),
            scorer=Scorer(),
            item=_item(),
            item_index=1,
            total_icps=1,
            run_start=time.time(),
            executor=executor,
        )
    finally:
        executor.shutdown(wait=False)

    assert result["_retryable"] is False
    assert result["_nonempty"] is True
    assert result["score"] == pytest.approx(9.0)
    assert result[sw._MODEL_QUALIFICATION_AUTHORITY_FIELD] == authority
    assert result[sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD] == sorted(
        [MODEL_RECEIPT_HASH, SCORER_RECEIPT_HASH]
    )
    assert sw._baseline_summary_checkpointable(result) is True


def test_qualification_checkpoint_rejects_authority_count_mismatch():
    row = {
        **_item(),
        "score": 0.0,
        "company_count": 0,
        "sourced_count": 0,
        "diagnostics": {"sourcing_failed": True},
        "_item_index": 1,
        "_retryable": True,
        "_nonempty": False,
        "_runtime_error": "attempt_failed",
        "_retry_backoff_seconds": 0.0,
        sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD: [MODEL_RECEIPT_HASH],
        sw._MODEL_QUALIFICATION_AUTHORITY_FIELD: _qualification_authority(
            "incomplete_retryable",
            partial_company_count=1,
        ),
        sw._MODEL_QUALIFICATION_PARTIAL_COUNT_FIELD: 0,
    }

    with pytest.raises(
        ValueError,
        match="partial count differs from authority",
    ):
        sw._baseline_attempt_checkpoint_row(row, retry_round=0)
