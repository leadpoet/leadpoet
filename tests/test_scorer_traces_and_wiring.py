"""Tests for §5.4 scorer-judge traces, baseline in-container trace collection,
candidate-scoped trace-sink wiring, and the admin confirmation follow-ups.

Covers, with fake boto3 + fake scorers (no live AWS / Supabase / docker):
  * _ScorerTraceRecorder — doc shape (identity-only inputs, full breakdowns),
    key scheme {prefix}/scorer-traces/{context}/{icp}.json, SSE-KMS via the
    score-bundle key, env-prefix override vs manifest-URI derivation,
    flag-off, count-and-drop inertness, and failure containment;
  * _run_baseline_icp — in-container collection through run_in_executor
    (contextvars copied explicitly), pointer-only diagnostics fields, the
    {prefix}/{context}/baseline/{icp}.json key scheme, count-and-drop without
    the S3 prefix, capture-off summaries byte-identical to the legacy shape
    (score_summary_doc hashing safety), and trace-failure containment;
  * the candidate-scoped in-container sink ({prefix}/{candidate_id}/{icp}.json)
    plus the _TraceCapturingCompanyScorer wrapper, end-to-end through the real
    evaluator scoring loop;
  * admin promote-scored-candidate — score-only: bypass_gates stays empty
    (there are no promotion gates left to bypass); --force's remaining role is
    pushing past the already_promoted early-exit to re-drive merge side
    effects, and it logs what it bypassed; dry-run output still surfaces the
    historical confirmation state for operator context.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import sys
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import gateway.research_lab.admin as admin
import gateway.research_lab.promotion as promotion
import gateway.research_lab.scoring_worker as sw
from gateway.research_lab.promotion import (
    CONFIRMATION_HOLD_REASON,
    CONFIRMATION_REJECTED_REASON,
    CONFIRMATION_RESULT_REASON,
)
from research_lab.canonical import sha256_json
from research_lab.eval import evaluator
from research_lab.eval.private_runtime import publish_incontainer_trace_entries


TRACE_ENV_NAMES = (
    "RESEARCH_LAB_SCORER_TRACE_CAPTURE",
    "RESEARCH_LAB_SCORER_TRACE_S3_PREFIX",
    "RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE",
    "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX",
    "RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID",
    "RESEARCH_LAB_BENCHMARK_SCRAPINGDOG_API_KEY",
    "RESEARCH_LAB_BENCHMARK_OPENROUTER_API_KEY",
    "RESEARCH_LAB_BENCHMARK_SCORER_MAX_CONCURRENCY",
)

MANIFEST_URI = "s3://lab-bucket/research-lab/sourcing-model/manifest.json"


@pytest.fixture(autouse=True)
def _clear_trace_env(monkeypatch):
    for name in TRACE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeS3Client:
    def __init__(self, puts: list[dict[str, Any]], fail: dict[str, Any]):
        self._puts = puts
        self._fail = fail

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        if self._fail.get("error") is not None:
            raise self._fail["error"]
        self._puts.append(kwargs)
        return {}


@pytest.fixture
def fake_s3(monkeypatch):
    """Install a fake boto3 whose put_object records calls (or fails on demand)."""
    puts: list[dict[str, Any]] = []
    fail: dict[str, Any] = {"error": None}
    fake_boto3 = types.SimpleNamespace(client=lambda service, **kwargs: _FakeS3Client(puts, fail))
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    return {"puts": puts, "fail": fail}


def _worker(**config_overrides: Any) -> Any:
    values: dict[str, Any] = {
        "score_bundle_kms_key_id": "kms-key-1",
        "private_model_manifest_uri": MANIFEST_URI,
    }
    values.update(config_overrides)
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    worker.config = SimpleNamespace(**values)
    worker.worker_ref = "worker-test"
    worker.proxy_ref_hash = None
    return worker


def _recorder(**config_overrides: Any) -> sw._ScorerTraceRecorder:
    values: dict[str, Any] = {"score_bundle_kms_key_id": "kms-key-1"}
    values.update(config_overrides)
    return sw._ScorerTraceRecorder(SimpleNamespace(**values))


COMPANIES = [
    {
        "company_name": "TraceCo",
        "company_website": "https://traceco.test",
        "company_linkedin": "https://linkedin.com/company/traceco",
        "employee_count": "11-50",
        "industry": "Software",
        "evidence_text": "RAW-PAGE-EVIDENCE-MUST-NOT-PERSIST",
    },
    {"company_name": "Other Co", "employee_count": "51-200"},
]

BREAKDOWNS = [
    {
        "company_name": "TraceCo",
        "final_score": 72.5,
        "icp_fit": 0.9,
        "intent_signal_final": 0.8,
        "intent_signals_detail": [],
        "failure_reason": "",
        "reasoning": "strong hiring signal on the careers page",
    },
    {
        "company_name": "Other Co",
        "final_score": 0.0,
        "icp_fit": 0.2,
        "intent_signal_final": 0.0,
        "intent_signals_detail": [],
        "failure_reason": "intent_fabricated",
        "reasoning": "signal could not be verified",
    },
]


class FakeBreakdownScorer:
    """QualificationStyleCompanyScorer stand-in returning fixed breakdowns."""

    def __init__(self, breakdowns: list[dict[str, Any]] | None = None):
        self.breakdowns = breakdowns if breakdowns is not None else [dict(b) for b in BREAKDOWNS]
        self.calls: list[tuple[Any, Any, bool]] = []

    async def score_with_breakdowns(self, outputs: Any, icp: Any, is_reference: bool) -> list[dict[str, Any]]:
        self.calls.append((outputs, icp, is_reference))
        return [dict(b) for b in self.breakdowns]


def _benchmark_item(tag: str = "a") -> dict[str, Any]:
    return {
        "icp": {"industry": "Software", "employee_count": "11-50", "name": f"icp-{tag}"},
        "icp_ref": f"icp:{tag}",
        "icp_hash": f"hash-{tag}",
        "set_id": 1,
        "day_index": 1,
        "day_rank": 1,
    }


TRACE_ENTRY = {
    "seq": 1,
    "phase": "call",
    "provider_class": "llm",
    "method": "POST",
    "url_redacted": "https://openrouter.ai/api/v1/chat/completions",
    "request_body_b64": "eyJtb2RlbCI6ICJtIn0=",
    "response_status": 200,
    "response_body_b64": "",
    "outcome": "success",
    "error": "",
}


def _publishing_runner(companies: list[dict[str, Any]] | None = None):
    """Sync runner (executed on the baseline executor thread) that publishes
    one in-container trace entry — proving the contextvars copy wiring."""

    def runner(icp: Any, ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
        publish_incontainer_trace_entries([dict(TRACE_ENTRY)])
        return [dict(c) for c in (companies if companies is not None else COMPANIES)]

    return runner


async def _run_icp(worker: Any, *, scorer: Any, runner: Any, trace_context: Any) -> dict[str, Any]:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return await worker._run_baseline_icp(
            runner=runner,
            scorer=scorer,
            item=_benchmark_item(),
            item_index=1,
            total_icps=1,
            run_start=0.0,
            executor=executor,
            trace_context=trace_context,
        )
    finally:
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# _ScorerTraceRecorder
# ---------------------------------------------------------------------------


def test_scorer_trace_uploads_doc_with_sse_kms_and_returns_pointer(fake_s3):
    recorder = _recorder()
    pointer = recorder.capture(
        context_ref="daily-2026-07-02-a0-abc123",
        icp_ref="icp:a",
        icp_hash="hash-a",
        outputs=COMPANIES,
        breakdowns=BREAKDOWNS,
        is_reference_model=True,
        manifest_uri=MANIFEST_URI,
    )
    recorder.flush()

    assert pointer is not None
    assert pointer["s3_ref"] == (
        "s3://lab-bucket/research-lab/sourcing-model/scorer-traces/"
        "daily-2026-07-02-a0-abc123/icp-a.json"
    )
    assert len(fake_s3["puts"]) == 1
    put = fake_s3["puts"][0]
    assert put["Bucket"] == "lab-bucket"
    assert put["Key"] == (
        "research-lab/sourcing-model/scorer-traces/daily-2026-07-02-a0-abc123/icp-a.json"
    )
    assert put["ServerSideEncryption"] == "aws:kms"
    assert put["SSEKMSKeyId"] == "kms-key-1"

    doc = json.loads(put["Body"].decode("utf-8"))
    assert doc["artifact_type"] == "research_lab_scorer_judgment_trace"
    assert doc["context_ref"] == "daily-2026-07-02-a0-abc123"
    assert doc["icp_ref"] == "icp:a"
    assert doc["icp_hash"] == "hash-a"
    assert doc["is_reference_model"] is True
    assert doc["sourced_count"] == 2
    assert doc["scored_count"] == 2
    # Inputs: company identity fields only — evidence/page text never persists.
    assert doc["companies"][0]["company_name"] == "TraceCo"
    assert doc["companies"][0]["company_website"] == "https://traceco.test"
    assert "evidence_text" not in doc["companies"][0]
    assert "RAW-PAGE-EVIDENCE" not in json.dumps(doc["companies"])
    # Outputs: full breakdowns including the scorer's reasoning text.
    assert doc["score_breakdowns"][0]["reasoning"] == "strong hiring signal on the careers page"
    assert doc["score_breakdowns"][0]["final_score"] == 72.5
    # The pointer sha256 matches the uploaded doc.
    assert pointer["sha256"] == sha256_json(doc)


def test_scorer_trace_env_prefix_overrides_manifest_derivation(fake_s3, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_SCORER_TRACE_S3_PREFIX", "s3://trace-bucket/custom/prefix/")
    recorder = _recorder()
    pointer = recorder.capture(
        context_ref="cand-1",
        icp_ref="icp:b",
        outputs=COMPANIES,
        breakdowns=BREAKDOWNS,
        manifest_uri=MANIFEST_URI,
    )
    recorder.flush()
    assert pointer["s3_ref"] == "s3://trace-bucket/custom/prefix/scorer-traces/cand-1/icp-b.json"
    assert fake_s3["puts"][0]["Bucket"] == "trace-bucket"
    assert fake_s3["puts"][0]["Key"] == "custom/prefix/scorer-traces/cand-1/icp-b.json"


def test_scorer_trace_flag_off_is_noop(fake_s3, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_SCORER_TRACE_CAPTURE", "false")
    recorder = _recorder()
    pointer = recorder.capture(
        context_ref="cand-1",
        icp_ref="icp:a",
        outputs=COMPANIES,
        breakdowns=BREAKDOWNS,
        manifest_uri=MANIFEST_URI,
    )
    recorder.flush()
    assert pointer is None
    assert fake_s3["puts"] == []


def test_scorer_trace_inert_without_s3_config(fake_s3, caplog):
    recorder = _recorder()
    with caplog.at_level(logging.INFO, logger="gateway.research_lab.scoring_worker"):
        for _ in range(3):
            assert (
                recorder.capture(
                    context_ref="cand-1",
                    icp_ref="icp:a",
                    outputs=COMPANIES,
                    breakdowns=BREAKDOWNS,
                    manifest_uri="/local/manifest.json",
                )
                is None
            )
    recorder.flush()
    assert fake_s3["puts"] == []
    drops = [
        record
        for record in caplog.records
        if "research_lab_scorer_trace_capture_disabled" in record.getMessage()
    ]
    assert len(drops) == 1  # logged once, further drops silent


def test_scorer_trace_empty_breakdowns_skip(fake_s3):
    recorder = _recorder()
    assert (
        recorder.capture(
            context_ref="cand-1",
            icp_ref="icp:a",
            outputs=COMPANIES,
            breakdowns=[],
            manifest_uri=MANIFEST_URI,
        )
        is None
    )
    recorder.flush()
    assert fake_s3["puts"] == []


def test_scorer_trace_upload_failure_contained_and_logged_once(fake_s3, caplog):
    fake_s3["fail"]["error"] = RuntimeError("s3 exploded for test")
    recorder = _recorder()
    with caplog.at_level(logging.WARNING, logger="gateway.research_lab.scoring_worker"):
        for _ in range(2):
            pointer = recorder.capture(
                context_ref="cand-1",
                icp_ref="icp:a",
                outputs=COMPANIES,
                breakdowns=BREAKDOWNS,
                manifest_uri=MANIFEST_URI,
            )
            # Optimistic pointer: returned while the write is in flight.
            assert pointer is not None
        recorder.flush()
    failures = [
        record
        for record in caplog.records
        if "research_lab_scorer_trace_capture_failed" in record.getMessage()
    ]
    assert len(failures) == 1  # once per context, not per call


# ---------------------------------------------------------------------------
# Baseline batch path: _run_baseline_icp collection + pointer-only summaries
# ---------------------------------------------------------------------------


async def test_run_baseline_icp_records_scorer_and_incontainer_traces(fake_s3, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX", "s3://trace-bucket/incontainer")
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID", "kms-key-42")
    worker = _worker()
    trace_context = worker._baseline_trace_context(
        context_ref="daily-2026-07-02-a0-w1",
        manifest_uri=MANIFEST_URI,
    )
    summary = await _run_icp(
        worker,
        scorer=FakeBreakdownScorer(),
        runner=_publishing_runner(),
        trace_context=trace_context,
    )
    worker._get_scorer_trace_recorder().flush()

    diagnostics = summary["diagnostics"]
    # In-container pointers (§9.1): {prefix}/{context}/baseline/{icp}.json.
    assert diagnostics["incontainer_trace_ref"] == (
        "s3://trace-bucket/incontainer/daily-2026-07-02-a0-w1/baseline/icp-a.json"
    )
    assert diagnostics["incontainer_trace_call_count"] == 1
    assert diagnostics["incontainer_trace_sha256"] == sha256_json([dict(TRACE_ENTRY)])
    # Scorer-judgment pointers (§5.4).
    assert diagnostics["scorer_trace_ref"] == (
        "s3://lab-bucket/research-lab/sourcing-model/scorer-traces/"
        "daily-2026-07-02-a0-w1/icp-a.json"
    )
    assert diagnostics["scorer_trace_sha256"].startswith("sha256:")

    puts_by_key = {put["Key"]: put for put in fake_s3["puts"]}
    incontainer_put = puts_by_key["incontainer/daily-2026-07-02-a0-w1/baseline/icp-a.json"]
    assert incontainer_put["ServerSideEncryption"] == "aws:kms"
    assert incontainer_put["SSEKMSKeyId"] == "kms-key-42"
    payload = json.loads(incontainer_put["Body"].decode("utf-8"))
    assert payload["artifact_type"] == "research_lab_incontainer_trace"
    assert payload["run_ref"] == "daily-2026-07-02-a0-w1"
    assert payload["icp_ref"] == "icp:a"
    assert payload["entries"][0]["seq"] == 1
    scorer_put = puts_by_key[
        "research-lab/sourcing-model/scorer-traces/daily-2026-07-02-a0-w1/icp-a.json"
    ]
    assert json.loads(scorer_put["Body"].decode("utf-8"))["is_reference_model"] is True

    # Pointer-only guard: no trace/judgment content in the summary itself.
    serialized = json.dumps(summary)
    assert "request_body_b64" not in serialized
    assert "eyJtb2RlbCI6" not in serialized
    assert "strong hiring signal" not in serialized
    # Scoring semantics untouched.
    assert summary["score"] == pytest.approx((72.5 + 0.0) / 2)
    assert summary["_nonempty"] is True


async def test_run_baseline_icp_capture_disabled_summary_byte_identical(fake_s3, monkeypatch):
    """Hashing safety: with capture off the summary must match the legacy shape
    key-for-key, so score_summary_doc (hashed at write time) is unchanged."""
    worker = _worker()
    legacy = await _run_icp(
        worker,
        scorer=FakeBreakdownScorer(),
        runner=lambda icp, ctx: [dict(c) for c in COMPANIES],
        trace_context=None,
    )

    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE", "false")
    monkeypatch.setenv("RESEARCH_LAB_SCORER_TRACE_CAPTURE", "false")
    captured_off = await _run_icp(
        worker,
        scorer=FakeBreakdownScorer(),
        runner=_publishing_runner(),
        trace_context=worker._baseline_trace_context(
            context_ref="daily-2026-07-02-a0-w1",
            manifest_uri=MANIFEST_URI,
        ),
    )
    assert fake_s3["puts"] == []
    for row in (legacy, captured_off):
        for key in ("_item_index", "_retryable", "_nonempty", "_runtime_error"):
            row.pop(key)
    assert captured_off == legacy
    assert sw.canonical_hash(captured_off) == sw.canonical_hash(legacy)
    assert "incontainer_trace_ref" not in captured_off["diagnostics"]
    assert "scorer_trace_ref" not in captured_off["diagnostics"]


async def test_run_baseline_icp_no_prefix_counts_and_drops_once(fake_s3, caplog):
    """Capture on but no S3 config: in-container pointers still record that
    capture happened (ref empty, evaluator semantics); the drop is logged once
    per batch; the scorer recorder stays inert."""
    worker = _worker()
    trace_context = worker._baseline_trace_context(
        context_ref="daily-2026-07-02-a0-w1",
        manifest_uri="/local/manifest.json",  # underivable: count-and-drop
    )
    summaries = []
    with caplog.at_level(logging.INFO, logger="gateway.research_lab.scoring_worker"):
        for _ in range(2):
            summaries.append(
                await _run_icp(
                    worker,
                    scorer=FakeBreakdownScorer(),
                    runner=_publishing_runner(),
                    trace_context=trace_context,
                )
            )
    worker._get_scorer_trace_recorder().flush()
    assert fake_s3["puts"] == []
    for summary in summaries:
        diagnostics = summary["diagnostics"]
        assert diagnostics["incontainer_trace_ref"] == ""
        assert diagnostics["incontainer_trace_call_count"] == 1
        assert diagnostics["incontainer_trace_sha256"].startswith("sha256:")
        assert "scorer_trace_ref" not in diagnostics
    drop_logs = [
        record
        for record in caplog.records
        if "research_lab_incontainer_trace_dropped" in record.getMessage()
    ]
    assert len(drop_logs) == 1
    assert trace_context["incontainer_drop_state"]["dropped_entries"] == 2


async def test_run_baseline_icp_trace_failure_never_affects_scoring(fake_s3, monkeypatch, caplog):
    async def _exploding_publish(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("publish exploded for test")

    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX", "s3://trace-bucket/x")
    monkeypatch.setattr(sw, "_publish_baseline_incontainer_trace", _exploding_publish)
    worker = _worker()
    with caplog.at_level(logging.WARNING, logger="gateway.research_lab.scoring_worker"):
        summary = await _run_icp(
            worker,
            scorer=FakeBreakdownScorer(),
            runner=_publishing_runner(),
            trace_context=worker._baseline_trace_context(
                context_ref="ctx",
                manifest_uri=MANIFEST_URI,
            ),
        )
    assert summary["score"] == pytest.approx((72.5 + 0.0) / 2)
    assert summary["_runtime_error"] == ""
    assert any(
        "research_lab_baseline_trace_record_failed" in record.getMessage()
        for record in caplog.records
    )


async def test_incontainer_upload_failure_keeps_sha_count_pointers(fake_s3, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX", "s3://trace-bucket/x")
    fake_s3["fail"]["error"] = RuntimeError("s3 exploded for test")
    monkeypatch.setenv("RESEARCH_LAB_SCORER_TRACE_CAPTURE", "false")
    worker = _worker()
    summary = await _run_icp(
        worker,
        scorer=FakeBreakdownScorer(),
        runner=_publishing_runner(),
        trace_context=worker._baseline_trace_context(context_ref="ctx", manifest_uri=""),
    )
    diagnostics = summary["diagnostics"]
    assert diagnostics["incontainer_trace_ref"] == ""  # failed write, no dangling wrong ref
    assert diagnostics["incontainer_trace_call_count"] == 1
    assert summary["score"] == pytest.approx((72.5 + 0.0) / 2)


def test_confirmation_side_trace_context_derived_from_scope():
    worker = _worker()
    assert worker._confirmation_side_trace_context(mode_label="confirmation_baseline") is None
    worker._confirmation_trace_scope = {
        "candidate_id": "cand-1",
        "attempt": 2,
        "manifest_uri": MANIFEST_URI,
    }
    champion = worker._confirmation_side_trace_context(mode_label="confirmation_baseline")
    candidate = worker._confirmation_side_trace_context(mode_label="confirmation_candidate")
    assert champion["context_ref"] == "confirmation-cand-1-a2-champion"
    assert candidate["context_ref"] == "confirmation-cand-1-a2-candidate"
    assert champion["manifest_uri"] == MANIFEST_URI
    assert champion["incontainer_drop_state"] == {"logged": False, "dropped_entries": 0}


# ---------------------------------------------------------------------------
# Candidate path: sink keying + scorer wrapper (through the real evaluator)
# ---------------------------------------------------------------------------


async def test_candidate_sink_keys_by_candidate(fake_s3, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX", "s3://trace-bucket/lab/traces")
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID", "kms-key-42")
    worker = _worker()
    sink = worker._candidate_incontainer_trace_sink("cand-123")
    assert sink is not None
    ref = await sink("icp-1", [dict(TRACE_ENTRY)])
    assert ref == "s3://trace-bucket/lab/traces/cand-123/icp-1.json"
    put = fake_s3["puts"][0]
    assert put["Key"] == "lab/traces/cand-123/icp-1.json"
    assert put["ServerSideEncryption"] == "aws:kms"
    assert put["SSEKMSKeyId"] == "kms-key-42"
    payload = json.loads(put["Body"].decode("utf-8"))
    assert payload["run_ref"] == "cand-123"
    assert payload["call_count"] == 1


def test_candidate_sink_none_without_prefix(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX", raising=False)
    worker = _worker()
    # None -> the evaluator's default count-and-drop sink applies.
    assert worker._candidate_incontainer_trace_sink("cand-123") is None


async def test_trace_capturing_scorer_delegates_and_captures(fake_s3, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_SCORER_TRACE_S3_PREFIX", "s3://trace-bucket/lab")
    recorder = _recorder()
    pointers: dict[str, dict[str, str]] = {}
    item = _benchmark_item("a")
    wrapper = sw._TraceCapturingCompanyScorer(
        recorder=recorder,
        context_ref="cand-1",
        manifest_uri=MANIFEST_URI,
        benchmark_items=[item],
        pointer_map=pointers,
        inner=FakeBreakdownScorer(),
    )
    scores = await wrapper(COMPANIES, item["icp"], False)
    recorder.flush()
    # __call__ mirrors QualificationStyleCompanyScorer exactly.
    assert scores == [72.5, 0.0]
    assert pointers["icp:a"]["s3_ref"] == "s3://trace-bucket/lab/scorer-traces/cand-1/icp-a.json"
    doc = json.loads(fake_s3["puts"][0]["Body"].decode("utf-8"))
    assert doc["icp_ref"] == "icp:a"
    assert doc["icp_hash"] == "hash-a"
    assert doc["is_reference_model"] is False
    assert doc["score_breakdowns"][0]["reasoning"] == "strong hiring signal on the careers page"


async def test_trace_capturing_scorer_capture_error_contained(monkeypatch):
    recorder = _recorder()

    def _boom(**kwargs: Any):
        raise RuntimeError("capture exploded for test")

    monkeypatch.setattr(recorder, "capture", _boom)
    wrapper = sw._TraceCapturingCompanyScorer(
        recorder=recorder,
        context_ref="cand-1",
        manifest_uri=MANIFEST_URI,
        benchmark_items=[_benchmark_item("a")],
        inner=FakeBreakdownScorer(),
    )
    breakdowns = await wrapper.score_with_breakdowns(COMPANIES, _benchmark_item("a")["icp"], False)
    assert [b["final_score"] for b in breakdowns] == [72.5, 0.0]


async def test_candidate_wiring_through_real_evaluator(fake_s3, monkeypatch):
    """End-to-end shape used at the scoring call site: the wrapper scorer and
    the candidate-scoped sink flow through evaluator.score_private_model_pair_items;
    rows gain the candidate-keyed incontainer_trace_ref and the scorer docs land
    under scorer-traces/{candidate_id}/."""
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX", "s3://trace-bucket/lab/traces")
    monkeypatch.setenv("RESEARCH_LAB_SCORER_TRACE_S3_PREFIX", "s3://trace-bucket/lab/traces")
    worker = _worker()
    items = [_benchmark_item("a")]
    pointers: dict[str, dict[str, str]] = {}
    wrapper = sw._TraceCapturingCompanyScorer(
        recorder=worker._get_scorer_trace_recorder(),
        context_ref="cand-9",
        manifest_uri=MANIFEST_URI,
        benchmark_items=items,
        pointer_map=pointers,
        inner=FakeBreakdownScorer(),
    )

    async def runner(icp: Any, context: Any) -> list[dict[str, Any]]:
        publish_incontainer_trace_entries([dict(TRACE_ENTRY)])
        return [dict(c) for c in COMPANIES]

    rows = await evaluator.score_private_model_pair_items(
        benchmark_items=items,
        base_runner=None,
        candidate_runner=runner,
        company_scorer=wrapper,
        run_context={"run_id": "run-1"},
        image_candidate=True,
        trace_sink=worker._candidate_incontainer_trace_sink("cand-9"),
    )
    worker._get_scorer_trace_recorder().flush()

    row = rows[0]
    assert row["candidate_company_scores"] == [72.5, 0.0]
    assert row["incontainer_trace_ref"] == "s3://trace-bucket/lab/traces/cand-9/icp-a.json"
    assert row["incontainer_trace_call_count"] == 1
    assert pointers["icp:a"]["s3_ref"] == (
        "s3://trace-bucket/lab/traces/scorer-traces/cand-9/icp-a.json"
    )
    keys = {put["Key"] for put in fake_s3["puts"]}
    assert keys == {
        "lab/traces/cand-9/icp-a.json",
        "lab/traces/scorer-traces/cand-9/icp-a.json",
    }
    # Rows stay pointer-only.
    serialized = json.dumps(row)
    assert "request_body_b64" not in serialized
    assert "strong hiring signal" not in serialized


# ---------------------------------------------------------------------------
# Admin: score-only replay (--force = already_promoted re-drive only) +
# historical confirmation state surfacing
# ---------------------------------------------------------------------------


class AdminFakeStore:
    """select_one/select_many fake with real 2-tuple filter semantics for the
    promotion-events table (shared by admin queries and load_confirmation_state)."""

    def __init__(self) -> None:
        self.promotion_events: list[dict[str, Any]] = []
        self.select_one_results: dict[str, Any] = {}
        self._seq = 0
        self._base = datetime.now(timezone.utc)

    def seed_promotion_event(self, **kwargs: Any) -> dict[str, Any]:
        self._seq += 1
        row = {
            "promotion_event_id": f"pe-{self._seq}",
            "created_at": (self._base + timedelta(milliseconds=self._seq)).isoformat(),
            "event_doc": {},
            **kwargs,
        }
        self.promotion_events.append(row)
        return row

    async def select_one(self, table: str, **kwargs: Any) -> dict[str, Any] | None:
        result = self.select_one_results.get(table)
        return dict(result) if isinstance(result, Mapping) else result

    async def select_many(self, table: str, **kwargs: Any) -> list[dict[str, Any]]:
        if table != "research_lab_candidate_promotion_events":
            return []
        rows = list(self.promotion_events)
        for spec in tuple(kwargs.get("filters") or ()):
            if len(spec) == 2:
                field, value = spec
                rows = [r for r in rows if str(r.get(field) or "") == str(value)]
        rows = sorted(rows, key=lambda r: str(r.get("created_at") or ""), reverse=True)
        return [dict(r) for r in rows[: int(kwargs.get("limit") or 100)]]


@pytest.fixture
def admin_store(monkeypatch) -> AdminFakeStore:
    fake = AdminFakeStore()
    monkeypatch.setattr(admin, "select_one", fake.select_one)
    monkeypatch.setattr(admin, "select_many", fake.select_many)
    # load_confirmation_state reads promotion's own store binding.
    monkeypatch.setattr(promotion, "select_many", fake.select_many)
    monkeypatch.delenv(promotion.PROMOTION_CONFIRMATION_RERUN_ENV, raising=False)

    fake.select_one_results["research_lab_candidate_evaluation_current"] = {
        "candidate_id": "cand-1",
        "current_score_bundle_id": "sb-1",
        "current_candidate_status": "scored",
        "current_reason": "gateway_qualification_worker_scored_candidate",
        "candidate_kind": "image_build",
        "parent_artifact_hash": "sha256:" + "a" * 64,
        "candidate_artifact_hash": "",
    }
    fake.select_one_results["research_evaluation_score_bundle_current"] = {
        "score_bundle_id": "sb-1",
        "current_event_status": "scored",
        "candidate_artifact_hash": "sha256:" + "c" * 64,
        "score_bundle_doc": {
            "aggregates": {"mean_delta": 2.5},
            "private_holdout_gate": {"decision": "private_holdout_approved"},
        },
    }

    async def _fake_baseline_doc(gate: Any):
        return {}, "loaded"

    monkeypatch.setattr(admin, "load_baseline_summary_doc_for_gate", _fake_baseline_doc)
    monkeypatch.setattr(
        admin,
        "promotion_improvement_metric",
        lambda bundle, baseline_score_summary_doc=None: SimpleNamespace(
            improvement_points=2.5,
            rejection_status=None,
            event_doc=lambda: {"metric": "fake"},
        ),
    )
    monkeypatch.setattr(
        admin,
        "ResearchLabGatewayConfig",
        SimpleNamespace(from_env=lambda: SimpleNamespace(improvement_threshold_points=1.0)),
    )
    return fake


async def _promote(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "candidate_id": "cand-1",
        "score_bundle_id": None,
        "dry_run": True,
        "force": False,
        "actor_ref": "op-test",
        "reason": "test-replay",
    }
    kwargs.update(overrides)
    return await admin._promote_scored_candidate(**kwargs)


async def test_admin_force_no_longer_bypasses_promotion_gates(admin_store, monkeypatch):
    """Score-only: there are no health/quarantine/confirmation gates left, so
    --force hands the controller an EMPTY bypass set — the score threshold can
    never be waived from the CLI."""
    captured: dict[str, Any] = {}

    class _FakeController:
        def __init__(self, config: Any, *, worker_ref: str):
            captured["worker_ref"] = worker_ref

        async def process_scored_candidate(self, **kwargs: Any) -> dict[str, Any]:
            captured["kwargs"] = kwargs
            return {"status": "merged", "private_model_version_id": "v-2"}

    monkeypatch.setattr(admin, "ResearchLabPromotionController", _FakeController)
    result = await _promote(dry_run=False, force=True)
    assert result["ok"] is True
    assert captured["kwargs"]["bypass_gates"] == frozenset()
    assert captured["worker_ref"] == "op-test"


async def test_admin_force_already_promoted_redrive_logged(admin_store, monkeypatch, caplog):
    """--force's remaining bypass is the already_promoted early-exit (to
    re-drive dropped merge side effects) and it must say so in the log."""
    admin_store.seed_promotion_event(
        candidate_id="cand-1",
        event_type="active_version_created",
        promotion_status="merged",
    )

    class _FakeController:
        def __init__(self, config: Any, *, worker_ref: str):
            pass

        async def process_scored_candidate(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "status": "already_promoted",
                "private_source_status": {"status": "already_recorded"},
                "champion_reward_status": "already_created",
            }

    monkeypatch.setattr(admin, "ResearchLabPromotionController", _FakeController)
    with caplog.at_level(logging.WARNING, logger="gateway.research_lab.admin"):
        result = await _promote(dry_run=False, force=True)
    assert result["ok"] is True
    bypass_logs = [
        record.getMessage()
        for record in caplog.records
        if "bypassing safeguards" in record.getMessage()
    ]
    assert bypass_logs and "already_promoted" in bypass_logs[0]


async def test_admin_without_force_keeps_bypass_gates_empty(admin_store, monkeypatch):
    captured: dict[str, Any] = {}

    class _FakeController:
        def __init__(self, config: Any, *, worker_ref: str):
            pass

        async def process_scored_candidate(self, **kwargs: Any) -> dict[str, Any]:
            captured["kwargs"] = kwargs
            return {"status": "rejected_below_threshold"}

    monkeypatch.setattr(admin, "ResearchLabPromotionController", _FakeController)
    result = await _promote(dry_run=False, force=False)
    assert captured["kwargs"]["bypass_gates"] == frozenset()
    assert result["promotion_result"]["status"] == "rejected_below_threshold"


async def test_admin_dry_run_surfaces_confirmation_hold(admin_store):
    admin_store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="promotion_checked",
        promotion_status="checked",
        event_doc={
            "reason": CONFIRMATION_HOLD_REASON,
            "first_pass_improvement_points": 2.5,
            "confirmation_min_delta": 1.0,
            "baseline_benchmark_bundle_id": "bb-1",
        },
    )
    result = await _promote(dry_run=True)
    assert result["dry_run"] is True
    confirmation = result["planned"]["confirmation"]
    assert confirmation["held_pending_confirmation"] is True
    assert confirmation["latest_reason"] == CONFIRMATION_HOLD_REASON
    assert confirmation["held_event"]["first_pass_improvement_points"] == pytest.approx(2.5)
    assert confirmation["held_event"]["confirmation_min_delta"] == pytest.approx(1.0)
    assert confirmation["held_event"]["baseline_benchmark_bundle_id"] == "bb-1"
    assert "recorded_confirmation" not in confirmation
    assert "rejected_confirmation_failed" not in confirmation


async def test_admin_dry_run_surfaces_recorded_delta(admin_store):
    admin_store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="promotion_checked",
        promotion_status="checked",
        event_doc={"reason": CONFIRMATION_HOLD_REASON, "first_pass_improvement_points": 2.5},
    )
    admin_store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="promotion_checked",
        promotion_status="checked",
        event_doc={
            "reason": CONFIRMATION_RESULT_REASON,
            "confirmation": {
                "confirmation_delta": 1.4,
                "window_match": True,
                "rolling_window_hash": "sha256:" + "3" * 64,
            },
        },
    )
    result = await _promote(dry_run=True)
    confirmation = result["planned"]["confirmation"]
    assert confirmation["held_pending_confirmation"] is False
    assert confirmation["latest_reason"] == CONFIRMATION_RESULT_REASON
    assert confirmation["recorded_confirmation"]["confirmation_delta"] == pytest.approx(1.4)
    assert confirmation["recorded_confirmation"]["window_match"] is True


async def test_admin_dry_run_surfaces_rejection_with_both_deltas(admin_store):
    admin_store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="below_threshold",
        promotion_status="rejected",
        event_doc={
            "reason": CONFIRMATION_REJECTED_REASON,
            "failure_mode": "confirmation_delta_below_min",
            "first_pass_improvement_points": 2.5,
            "confirmation_delta": 0.3,
            "confirmation_min_delta": 1.0,
        },
    )
    result = await _promote(dry_run=True)
    rejected = result["planned"]["confirmation"]["rejected_confirmation_failed"]
    assert rejected["failure_mode"] == "confirmation_delta_below_min"
    # BOTH deltas are surfaced.
    assert rejected["first_pass_improvement_points"] == pytest.approx(2.5)
    assert rejected["confirmation_delta"] == pytest.approx(0.3)
    assert rejected["confirmation_min_delta"] == pytest.approx(1.0)


async def test_admin_unrelated_below_threshold_is_not_confirmation_rejection(admin_store):
    admin_store.seed_promotion_event(
        candidate_id="cand-1",
        source_score_bundle_id="sb-1",
        event_type="below_threshold",
        promotion_status="rejected",
        event_doc={"mean_delta": 0.1},  # plain threshold rejection, no reason
    )
    result = await _promote(dry_run=True)
    confirmation = result["planned"]["confirmation"]
    assert "rejected_confirmation_failed" not in confirmation
    assert confirmation["held_pending_confirmation"] is False
