"""Tests for the Research Lab evaluator fairness/scoring fixes.

Covers bugs #8/#14/#15/#31/#35 from the pre-launch audit:
- timeout retry + 2-consecutive-timeout skip latch (and the legacy latch flag),
- provider-flake retry with symmetric zero-score ICP accounting,
- the local retryable-error classifier (Scrapingdog 400 retryable, 410/auth permanent),
- the `sourced_zero_no_error` silent-zero flag and in-container detection for
  httpx/requests/aiohttp,
- capped top-5 per-ICP scoring matching leadpoet_verifier.aggregation exactly,
- the per-ICP checkpoint/resume interface,
- optional candidate ICP concurrency with deterministic result ordering.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from leadpoet_verifier import aggregation
from leadpoet_verifier.research_evaluation import (
    score_bundle_hash,
    verify_research_evaluation_score_bundle,
)
from research_lab.canonical import sha256_json
from research_lab.eval import evaluator
from research_lab.eval.artifacts import PrivateModelArtifactManifest
from research_lab.eval.benchmark import SealedBenchmarkSet
from research_lab.eval import private_runtime
from research_lab.eval.private_runtime import PrivateModelRuntimeError
from research_lab.validator_integration import (
    verify_research_lab_evaluation_bundle_page,
    write_research_lab_validator_artifact,
)


EVAL_ENV_FLAGS = (
    "RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY",
    "RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY",
    "RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE",
    "RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES",
    "RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY",
    "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS",
)

TIMEOUT_MSG = "docker private model adapter timed out"
PROVIDER_RETRYABLE_MSG = (
    "docker private model provider-backed sourcing failed before returning companies: "
    "HTTPError: too many requests; status=429; url=https://api.example.test/search"
)
PROVIDER_PERMANENT_MSG = (
    "docker private model provider-backed sourcing failed before returning companies: "
    "HTTPError: unauthorized; status=401; url=https://api.example.test/search"
)
SCRAPINGDOG_400_MSG = (
    "private model provider-backed sourcing failed before returning companies: "
    'HTTPError: HTTP Error 400: Bad Request; status=400; body={"success": false, '
    '"message": "Something went wrong or profile not found"}; '
    "url=https://api.scrapingdog.com/linkedin"
)
SCRAPINGDOG_410_MSG = (
    "private model provider-backed sourcing failed before returning companies: "
    "HTTPError: HTTP Error 410: Gone; status=410; url=https://api.scrapingdog.com/linkedin"
)
COST_CAP_402_MSG = (
    "private model provider-backed sourcing failed before returning companies: "
    "HTTPError: HTTP Error 402: Payment Required; status=402; "
    "body={\"error\":\"research_lab_provider_cost_cap_exceeded\"}"
)


@pytest.fixture(autouse=True)
def _clear_eval_env_flags(monkeypatch):
    for name in EVAL_ENV_FLAGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(evaluator, "_PROVIDER_429_RETRY_BACKOFF_SECONDS", 0.0)


def _benchmark_items(count: int) -> list[dict]:
    return [
        {
            "icp_ref": f"icp-{index}",
            "icp_hash": f"hash-{index}",
            "icp": {"name": f"icp-{index}"},
        }
        for index in range(count)
    ]


class ScriptedRunner:
    """Fake model runner: per-ICP scripted outcomes, then a default success."""

    def __init__(self, script=None, default=None):
        self.script = {key: list(value) for key, value in (script or {}).items()}
        self.default = default if default is not None else [{"score": 50.0}]
        self.calls: list[str] = []

    async def __call__(self, icp, context):
        ref = str(icp["name"])
        self.calls.append(ref)
        plan = self.script.get(ref)
        if plan:
            outcome = plan.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return list(self.default)

    def call_count(self, ref: str) -> int:
        return sum(1 for call in self.calls if call == ref)


async def _fake_scorer(companies, icp, is_reference_model):
    return [float(company["score"]) for company in companies]


async def _score_items(runner, items, **kwargs):
    return await evaluator.score_private_model_pair_items(
        benchmark_items=items,
        base_runner=None,
        candidate_runner=runner,
        company_scorer=_fake_scorer,
        run_context={"run_id": "run-1"},
        image_candidate=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Bug #14 — timeout retry + 2-consecutive-timeout latch
# ---------------------------------------------------------------------------


async def test_single_timeout_is_retried_and_recovers():
    runner = ScriptedRunner(
        script={"icp-0": [PrivateModelRuntimeError(TIMEOUT_MSG), [{"score": 40.0}]]}
    )
    rows = await _score_items(runner, _benchmark_items(3))
    assert runner.call_count("icp-0") == 2
    assert rows[0]["candidate_company_scores"] == [40.0]
    assert rows[0]["failure_reason"] == ""
    # No latch: the remaining ICPs still run.
    assert runner.call_count("icp-1") == 1
    assert runner.call_count("icp-2") == 1


async def test_latch_requires_two_consecutive_timeouts():
    timeout = lambda: PrivateModelRuntimeError(TIMEOUT_MSG)  # noqa: E731
    runner = ScriptedRunner(
        script={
            "icp-0": [timeout(), timeout(), timeout()],
            "icp-1": [timeout(), timeout(), timeout()],
        }
    )
    rows = await _score_items(runner, _benchmark_items(4))
    assert "candidate_model_runtime_timeout" in rows[0]["failure_reason"]
    assert "candidate_model_runtime_timeout" in rows[1]["failure_reason"]
    # Latch engaged after the second consecutive timeout: remaining ICPs are
    # skipped without running.
    assert "candidate_model_runtime_skipped_after_timeout" in rows[2]["failure_reason"]
    assert "candidate_model_runtime_skipped_after_timeout" in rows[3]["failure_reason"]
    assert runner.call_count("icp-0") == 3  # first attempt + two retry rounds
    assert runner.call_count("icp-1") == 3
    assert runner.call_count("icp-2") == 0
    assert runner.call_count("icp-3") == 0


async def test_non_consecutive_timeouts_do_not_latch():
    timeout = lambda: PrivateModelRuntimeError(TIMEOUT_MSG)  # noqa: E731
    runner = ScriptedRunner(
        script={
            "icp-0": [timeout(), timeout(), timeout()],
            "icp-2": [timeout(), timeout(), timeout()],
        }
    )
    rows = await _score_items(runner, _benchmark_items(4))
    # A successful ICP between timeouts resets the consecutive counter.
    assert rows[1]["candidate_company_scores"] == [50.0]
    assert rows[3]["candidate_company_scores"] == [50.0]
    assert all("skipped_after" not in row["failure_reason"] for row in rows)


async def test_legacy_timeout_latch_flag_restores_first_timeout_latch(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY", "true")
    runner = ScriptedRunner(script={"icp-0": [PrivateModelRuntimeError(TIMEOUT_MSG)]})
    rows = await _score_items(runner, _benchmark_items(3))
    # Legacy: no retry, latch on the very first timeout.
    assert runner.call_count("icp-0") == 1
    assert "candidate_model_runtime_timeout" in rows[0]["failure_reason"]
    assert "candidate_model_runtime_skipped_after_timeout" in rows[1]["failure_reason"]
    assert "candidate_model_runtime_skipped_after_timeout" in rows[2]["failure_reason"]
    assert runner.call_count("icp-1") == 0
    assert runner.call_count("icp-2") == 0


# ---------------------------------------------------------------------------
# Bug #15 — provider-flake retry + symmetric zero-score accounting
# ---------------------------------------------------------------------------


async def test_retryable_provider_error_is_retried_then_scored_zero():
    runner = ScriptedRunner(
        script={
            "icp-1": [
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
            ]
        }
    )
    rows = await _score_items(runner, _benchmark_items(3))
    assert runner.call_count("icp-1") == 3
    assert rows[1]["provider_excluded"] is False
    assert "candidate_model_runtime_provider_error" in rows[1]["failure_reason"]
    assert rows[1]["candidate_company_scores"] == []
    # Retry-exhausted provider failures stay in the aggregate as zero-score ICPs.
    assert evaluator._benchmark_style_score(rows, "candidate_company_scores") == pytest.approx(
        (50.0 + 0.0 + 50.0) / 3
    )
    assert evaluator._provider_excluded_icp_ids(rows) == []


async def test_provider_error_recovered_on_retry_scores_normally():
    runner = ScriptedRunner(
        script={"icp-0": [PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG), [{"score": 30.0}]]}
    )
    rows = await _score_items(runner, _benchmark_items(2))
    assert runner.call_count("icp-0") == 2
    assert rows[0]["provider_excluded"] is False
    assert rows[0]["candidate_company_scores"] == [30.0]
    assert rows[0]["failure_reason"] == ""


async def test_permanent_provider_error_not_retried_and_scores_zero():
    runner = ScriptedRunner(script={"icp-0": [PrivateModelRuntimeError(PROVIDER_PERMANENT_MSG)]})
    rows = await _score_items(runner, _benchmark_items(2))
    assert runner.call_count("icp-0") == 1
    assert rows[0]["provider_excluded"] is False
    assert evaluator._benchmark_style_score(rows, "candidate_company_scores") == pytest.approx(
        (0.0 + 50.0) / 2
    )


async def test_cost_cap_provider_error_not_retried_and_scores_zero_without_output():
    runner = ScriptedRunner(script={"icp-0": [PrivateModelRuntimeError(COST_CAP_402_MSG)]})
    rows = await _score_items(runner, _benchmark_items(2))

    assert runner.call_count("icp-0") == 1
    assert "candidate_model_runtime_provider_error" in rows[0]["failure_reason"]
    assert evaluator._benchmark_style_score(rows, "candidate_company_scores") == pytest.approx(
        (0.0 + 50.0) / 2
    )


async def test_cost_cap_trace_preserves_returned_candidate_scores():
    class CapSoftStopRunner:
        async def __call__(self, icp, context):
            private_runtime.publish_incontainer_trace_entries(
                [
                    {
                        "phase": "provider_cost",
                        "provider_cost_event": {
                            "provider": "exa",
                            "endpoint": "/search",
                            "request_fingerprint": "c" * 64,
                            "evidence": "budget_soft_stop",
                            "status_code": 200,
                            "billable": False,
                            "cost_usd": 0,
                            "cost_source": "blocked_before_paid_call",
                            "cap_usd": 1.0,
                            "spent_before_usd": 1.0,
                            "spent_after_usd": 1.0,
                            "cap_blocked": True,
                            "tracking_failed": False,
                            "scope": "scope-cap",
                        },
                    }
                ]
            )
            return [{"score": 42.0}]

    rows = await _score_items(
        CapSoftStopRunner(),
        _benchmark_items(1),
        trace_sink=lambda icp_ref, entries: f"s3://trace/{icp_ref}.json",
    )

    assert rows[0]["candidate_company_scores"] == [42.0]
    assert rows[0]["provider_cost_cap_blocked"] is True
    assert "provider_cost_cap_blocked" in rows[0]["failure_reason"]


async def test_provider_flake_retry_flag_off_restores_legacy_zeroing(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY", "false")
    runner = ScriptedRunner(script={"icp-1": [PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG)]})
    rows = await _score_items(runner, _benchmark_items(3))
    assert runner.call_count("icp-1") == 1  # no retry
    assert rows[1]["provider_excluded"] is False
    assert evaluator._provider_excluded_icp_ids(rows) == []
    # Retry flag off skips retries, but the failed ICP still contributes 0.
    assert evaluator._benchmark_style_score(rows, "candidate_company_scores") == pytest.approx(
        (50.0 + 0.0 + 50.0) / 3
    )


async def test_provider_error_never_latches_remaining_icps():
    runner = ScriptedRunner(
        script={
            "icp-0": [
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
            ]
        }
    )
    rows = await _score_items(runner, _benchmark_items(3))
    assert runner.call_count("icp-0") == 3
    assert runner.call_count("icp-1") == 1
    assert runner.call_count("icp-2") == 1
    assert all("skipped_after" not in row["failure_reason"] for row in rows)


async def test_gate_keeps_retry_exhausted_provider_failures_in_totals():
    runner = ScriptedRunner(
        script={
            "icp-1": [
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
            ]
        }
    )
    gate = {
        "public_icp_refs": ["icp-0", "icp-1"],
        "baseline_public_score": 10.0,
        "baseline_aggregate_score": 20.0,
        "baseline_benchmark_bundle_id": "bundle-1",
        "baseline_benchmark_hash": "sha256:baseline",
    }
    all_results, gate_result = await evaluator._score_with_private_holdout_gate(
        benchmark_items=_benchmark_items(4),
        base_runner=None,
        candidate_runner=runner,
        scorer=_fake_scorer,
        run_context={"run_id": "run-1"},
        image_candidate=True,
        runtime_patch=None,
        gate=gate,
    )
    assert gate_result["decision"] == "private_holdout_approved"
    # Public and total scores include the retry-exhausted provider failure as 0.
    assert gate_result["candidate_public_score"] == pytest.approx((50.0 + 0.0) / 2)
    assert gate_result["candidate_total_score"] == pytest.approx((50.0 + 0.0 + 50.0 + 50.0) / 4)
    assert gate_result["provider_excluded_icp_ids"] == []
    assert gate_result["candidate_delta_vs_daily_baseline"] == pytest.approx(17.5)
    assert len(all_results) == 4


# ---------------------------------------------------------------------------
# Bug #15/#37 — local retryable-error classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    (
        (SCRAPINGDOG_400_MSG, True),
        (SCRAPINGDOG_410_MSG, False),
        (PROVIDER_RETRYABLE_MSG, True),  # 429
        ("HTTPError: request timeout; status=408", True),
        ("HTTPError: internal error; status=500", True),
        ("HTTPError: bad gateway; status=502", True),
        ("URLError: connection reset by peer", True),
        ("adapter failed: exit status 137 (killed)", True),
        (COST_CAP_402_MSG, False),
        ("provider_cost_cap_blocked cost_cap_reached", False),
        (PROVIDER_PERMANENT_MSG, False),  # 401 auth
        ("HTTPError: forbidden; status=403", False),
        ("HTTPError: HTTP Error 400: Bad Request; status=400; body=invalid request payload", False),
        ("HTTPError: not found; status=404", False),
        ("ValueError: model returned garbage", False),
    ),
)
def test_candidate_error_classifier_verdicts(message, expected):
    assert evaluator._candidate_error_is_retryable(message) is expected


def test_candidate_429_backoff_only_for_explicit_429(monkeypatch):
    monkeypatch.setattr(evaluator, "_PROVIDER_429_RETRY_BACKOFF_SECONDS", 15.0)

    assert evaluator._candidate_429_retry_backoff_seconds(PROVIDER_RETRYABLE_MSG) == 15.0
    assert evaluator._candidate_429_retry_backoff_seconds("HTTPError: request timeout; status=408") == 0.0
    assert evaluator._candidate_429_retry_backoff_seconds("HTTPError: internal error; status=500") == 0.0
    assert evaluator._candidate_429_retry_backoff_seconds("HTTPError: too many requests without status") == 0.0
    assert evaluator._candidate_429_retry_backoff_seconds(SCRAPINGDOG_410_MSG) == 0.0


# ---------------------------------------------------------------------------
# Bug #35 — sourced_zero_no_error flag
# ---------------------------------------------------------------------------


async def test_sourced_zero_no_error_flag():
    runner = ScriptedRunner(
        script={
            "icp-0": [[]],  # zero companies, no error: the silent-zero blind spot
            "icp-1": [
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
            ],
        }
    )
    rows = await _score_items(runner, _benchmark_items(3))
    assert rows[0]["sourced_zero_no_error"] is True
    assert "candidate_model_zero_companies" in rows[0]["failure_reason"]
    # Zero companies WITH a recorded runtime error is not a silent zero.
    assert rows[1]["sourced_zero_no_error"] is False
    # Non-empty output is not a silent zero.
    assert rows[2]["sourced_zero_no_error"] is False

    health = evaluator.build_scoring_health_doc(rows)
    assert health["sourced_zero_no_error_count"] == 1
    assert health["provider_excluded_icp_count"] == 0
    assert health["health_status"] == "degraded"


# ---------------------------------------------------------------------------
# health_status reflects only measurement-invalidating failures. Legitimate
# zero-company outcomes and candidate-quality crashes stay informational.
# ---------------------------------------------------------------------------


def test_health_status_healthy_for_zero_company_outcomes():
    rows = [
        {"failure_reason": "candidate_model_zero_companies", "sourced_zero_no_error": True},
        {"failure_reason": "reference_model_zero_companies"},
        {"failure_reason": "candidate_model_runtime_invalid_json"},
        {},
    ]
    health = evaluator.build_scoring_health_doc(rows)
    # Counted and rated, but none of these invalidate the measurement.
    assert health["sourced_zero_no_error_count"] == 1
    assert health["candidate_zero_company_count"] == 1
    assert health["candidate_runtime_failure_count"] == 1
    assert health["health_status"] == "healthy"


@pytest.mark.parametrize(
    "row",
    [
        {"failure_reason": "candidate_model_runtime_provider_error"},
        {"failure_reason": "reference_model_runtime_invalid_json"},
        {"failure_reason": "candidate_model_runtime_timeout"},
        {"provider_excluded": True},
        {"provider_cost_cap_blocked": True},
        {"provider_cost_tracking_failed": True},
    ],
)
def test_health_status_degraded_for_critical_failures(row):
    health = evaluator.build_scoring_health_doc([row, {}])
    assert health["health_status"] == "degraded"


# ---------------------------------------------------------------------------
# Loop-ending provider-error gate: only credit/auth/quota rejections and
# provider-infra failures escalate an empty run to a runtime error. Request-
# shaped rejections of model-generated URLs (SD 400/404/410/503) score as-is.
# ---------------------------------------------------------------------------

_MARKER = private_runtime.PROVIDER_ERROR_MARKER


def _stderr(*error_lines: str) -> str:
    return "\n".join(f"{_MARKER} {line}" for line in error_lines)


SD_404_LINE = "HTTPError: HTTP Error 404: Not Found; status=404; url=https://api.scrapingdog.com/scrape"
SD_400_LINE = (
    'HTTPError: HTTP Error 400: Bad Request; status=400; body={"success": false, '
    '"message": "Something went wrong or profile not found"}; url=https://api.scrapingdog.com/linkedin'
)
SD_503_LINE = "HTTPError: HTTP Error 503: Service Unavailable; status=503; url=https://api.scrapingdog.com/scrape"
SD_410_LINE = "HTTPError: HTTP Error 410: Gone; status=410; url=https://api.scrapingdog.com/linkedin"
EXA_402_LINE = "HTTPError: HTTP Error 402: Payment Required; status=402; url=https://api.exa.ai/search"
PROXY_500_LINE = "HTTPError: HTTP Error 500: Internal Server Error; status=500; reason=Internal Server Error"
TIMEOUT_LINE = "TimeoutError: timed out; url=http://172.17.0.1:8791/exa/search"
RATE_429_LINE = "HTTPError: too many requests; status=429; url=https://api.example.test/search"
QUOTA_400_LINE = "HTTPError: HTTP Error 400: Bad Request; status=400; body=request quota exceeded"


@pytest.mark.parametrize(
    "line,loop_ending",
    [
        (SD_404_LINE, False),
        (SD_400_LINE, False),
        (SD_503_LINE, False),
        (SD_410_LINE, False),
        (EXA_402_LINE, True),
        (PROXY_500_LINE, True),
        (TIMEOUT_LINE, True),
        (RATE_429_LINE, True),
        # Credit/quota markers win even under a request-shaped status.
        (QUOTA_400_LINE, True),
    ],
)
def test_provider_error_line_loop_ending_classification(line, loop_ending):
    assert private_runtime._provider_error_line_is_loop_ending(line) is loop_ending


def test_empty_run_with_only_request_errors_does_not_raise():
    # SD rejecting model-generated URLs is model-output behavior: the empty
    # run stands as a legitimate zero-company result.
    private_runtime._raise_on_empty_provider_error(
        [], _stderr(SD_404_LINE, SD_400_LINE, SD_503_LINE), context_label="docker private model"
    )


def test_empty_run_with_credit_exhaustion_raises():
    with pytest.raises(PrivateModelRuntimeError) as excinfo:
        private_runtime._raise_on_empty_provider_error(
            [], _stderr(EXA_402_LINE), context_label="docker private model"
        )
    assert "402" in str(excinfo.value)


def test_empty_run_mixed_errors_raises_with_loop_ending_line():
    # The raise text must carry the loop-ending line so downstream
    # diagnostics parse the outage status, not a trailing per-URL 404.
    with pytest.raises(PrivateModelRuntimeError) as excinfo:
        private_runtime._raise_on_empty_provider_error(
            [], _stderr(SD_404_LINE, EXA_402_LINE, SD_404_LINE), context_label="private model"
        )
    text = str(excinfo.value)
    assert "402" in text
    assert "404" not in text


def test_empty_run_with_transport_failure_raises():
    with pytest.raises(PrivateModelRuntimeError):
        private_runtime._raise_on_empty_provider_error(
            [], _stderr(TIMEOUT_LINE), context_label="docker private model"
        )
    with pytest.raises(PrivateModelRuntimeError):
        private_runtime._raise_on_empty_provider_error(
            [], _stderr(PROXY_500_LINE), context_label="docker private model"
        )


def test_non_empty_run_never_raises_on_provider_errors():
    private_runtime._raise_on_empty_provider_error(
        [{"company_name": "Acme"}], _stderr(EXA_402_LINE), context_label="docker private model"
    )


def test_empty_run_without_provider_errors_does_not_raise():
    private_runtime._raise_on_empty_provider_error([], "", context_label="docker private model")


def test_provider_error_detection_bootstrap_covers_httpx_requests_aiohttp(tmp_path):
    probe = r"""
import asyncio

import httpx

def _boom_handler(request):
    raise httpx.ConnectError("httpx boom for test")

client = httpx.Client(transport=httpx.MockTransport(_boom_handler))
try:
    client.get("https://provider.test/httpx")
except Exception:
    pass

def _status_handler(request):
    return httpx.Response(429, request=request, text="rate limited body")

client_status = httpx.Client(transport=httpx.MockTransport(_status_handler))
try:
    client_status.get("https://provider.test/httpx429").raise_for_status()
except Exception:
    pass

import requests
from requests.adapters import BaseAdapter

class _BoomAdapter(BaseAdapter):
    def send(self, request, **kwargs):
        raise requests.ConnectionError("requests boom for test")

session = requests.Session()
session.mount("https://", _BoomAdapter())
try:
    session.get("https://provider.test/requests")
except Exception:
    pass

response = requests.models.Response()
response.status_code = 503
response.url = "https://provider.test/requests503"
try:
    response.raise_for_status()
except Exception:
    pass

import aiohttp

async def _aio():
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=5)
    ) as aio_session:
        await aio_session.get("http://127.0.0.1:9/aiohttp")

try:
    asyncio.run(_aio())
except Exception:
    pass

print("PROBE_DONE")
"""
    completed = subprocess.run(
        [sys.executable, "-c", private_runtime._PROVIDER_DIAGNOSTICS_BOOTSTRAP + probe],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PROBE_DONE" in completed.stdout
    marker = private_runtime.PROVIDER_ERROR_MARKER
    marker_lines = [line for line in completed.stderr.splitlines() if marker in line]
    assert any("httpx boom for test" in line for line in marker_lines)
    assert any("status=429" in line for line in marker_lines)
    assert any("requests boom for test" in line for line in marker_lines)
    assert any("status=503" in line for line in marker_lines)
    assert any("aiohttp" in line for line in marker_lines)


def test_provider_error_detection_bootstrap_tolerates_missing_libraries(tmp_path):
    blocker = r"""
import importlib.abc
import sys

class _BlockHttpLibs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"httpx", "requests", "aiohttp"}:
            raise ImportError("blocked for test: " + fullname)
        return None

sys.meta_path.insert(0, _BlockHttpLibs())
"""
    probe = r"""
import urllib.request

try:
    urllib.request.urlopen("http://127.0.0.1:9/urllib")
except Exception:
    pass
print("BLOCKED_PROBE_DONE")
"""
    completed = subprocess.run(
        [sys.executable, "-c", blocker + private_runtime._PROVIDER_DIAGNOSTICS_BOOTSTRAP + probe],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert "BLOCKED_PROBE_DONE" in completed.stdout
    # The urllib hook still works when the optional libraries are absent.
    assert private_runtime.PROVIDER_ERROR_MARKER in completed.stderr


# ---------------------------------------------------------------------------
# Bug #8 — capped top-5 scoring + LLM-scored company cap
# ---------------------------------------------------------------------------


def test_capped_top5_matches_verifier_arithmetic(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    per_icp_company_scores = [
        [10.0, 90.0, 50.0, 70.0, 30.0, 20.0, 110.0],  # >5 companies, one above cap
        [40.0, 5.0],  # <5 companies
        [],  # zero companies
    ]
    rows = [{"candidate_company_scores": scores} for scores in per_icp_company_scores]
    expected_per_icp = [
        aggregation.per_icp_normalized_score(sorted(scores, reverse=True)[:5], max_leads=5)
        for scores in per_icp_company_scores
    ]
    expected = sum(expected_per_icp) / len(expected_per_icp)
    assert evaluator._benchmark_style_score(rows, "candidate_company_scores") == pytest.approx(expected)
    # The shared per-ICP helper matches the verifier module exactly on <=5 inputs.
    assert evaluator.benchmark_icp_score_from_company_scores([40.0, 5.0]) == pytest.approx(
        aggregation.per_icp_normalized_score([40.0, 5.0], max_leads=5)
    )
    # Truncating to a single strong company no longer beats five decent ones.
    single = evaluator.benchmark_icp_score_from_company_scores([90.0])
    five = evaluator.benchmark_icp_score_from_company_scores([60.0, 55.0, 50.0, 45.0, 40.0])
    assert five > single


def test_capped_top5_flag_defaults_to_legacy_mean():
    scores = [10.0, 90.0, 50.0, 70.0, 30.0, 20.0, 110.0]
    rows = [{"candidate_company_scores": scores}]
    # Default OFF: unweighted mean of all company scores, no clamping.
    assert evaluator._benchmark_style_score(rows, "candidate_company_scores") == pytest.approx(
        sum(scores) / len(scores)
    )
    assert evaluator.benchmark_icp_score_from_company_scores(scores) == pytest.approx(
        sum(scores) / len(scores)
    )


def test_capped_score_normalizes_against_requested_company_count(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    # One 80-score company against a 3-company goal earns a third of the goal
    # being fully met — quantity is paid, not just quality.
    assert evaluator.benchmark_icp_score_from_company_scores(
        [80.0], requested_count=3
    ) == pytest.approx(80.0 / 3)
    assert evaluator.benchmark_icp_score_from_company_scores(
        [80.0, 80.0, 80.0], requested_count=3
    ) == pytest.approx(80.0)
    # Overshoot beyond the goal is capped at the top-N companies.
    assert evaluator.benchmark_icp_score_from_company_scores(
        [80.0, 70.0, 60.0, 90.0], requested_count=3
    ) == pytest.approx((90.0 + 80.0 + 70.0) / 3)
    # None/invalid falls back to the fixed 5-lead budget (legacy ICP sets).
    assert evaluator.benchmark_icp_score_from_company_scores(
        [80.0], requested_count=None
    ) == pytest.approx(80.0 / 5)
    # Rows built by the evaluator carry the goal; the batch scorer honors it.
    rows = [
        {"candidate_company_scores": [80.0], "icp_company_goal": 1},
        {"candidate_company_scores": [80.0], "icp_company_goal": 4},
    ]
    assert evaluator._benchmark_style_score(rows, "candidate_company_scores") == pytest.approx(
        (80.0 + 80.0 / 4) / 2
    )


def test_icp_company_goal_reads_max_companies():
    assert evaluator._icp_company_goal({"max_companies": 3}) == 3
    assert evaluator._icp_company_goal({"max_companies": "7"}) == 7
    assert evaluator._icp_company_goal({"max_companies": 500}) == 50
    assert evaluator._icp_company_goal({"max_companies": 0}) == 1
    assert evaluator._icp_company_goal({}) == 5
    assert evaluator._icp_company_goal({"max_companies": "junk"}) == 5
    assert evaluator._icp_company_goal(None) == 5


async def test_max_scored_companies_caps_llm_scoring(monkeypatch):
    monkeypatch.delenv("QUALIFICATION_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    class FakeModels:
        class CompanyOutput:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class ICPPrompt:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    scored_calls = []

    class FakeLeadScorer:
        @staticmethod
        async def score_company_autoresearch_intent_v2(
            *, company, icp, run_cost_usd, run_time_seconds, seen_companies, is_reference_model
        ):
            scored_calls.append(company)
            return {"final_score": 10.0}

    def fake_import(name):
        if name == "gateway.qualification.models":
            return FakeModels
        if name == "qualification.scoring.lead_scorer":
            return FakeLeadScorer
        raise ImportError(name)

    monkeypatch.setattr(evaluator, "import_module", fake_import)
    icp = {
        "industry": "Software",
        "intent_signal": "hiring sales reps",
        "employee_count": "51-200",
    }
    companies = [
        {"company_name": f"co-{index}", "employee_count": "51-200"} for index in range(6)
    ]
    scorer = evaluator.QualificationStyleCompanyScorer()

    monkeypatch.setenv("RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES", "2")
    capped = await scorer(companies, icp, False)
    assert len(capped) == 2
    assert len(scored_calls) == 2

    scored_calls.clear()
    monkeypatch.delenv("RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES")
    uncapped = await scorer(companies, icp, False)
    # Default 0 = unlimited (legacy behavior).
    assert len(uncapped) == 6
    assert len(scored_calls) == 6


def test_explicit_candidate_intent_signals_are_schema_normalized():
    company = {
        "company_name": "ExampleCo",
        "company_website": "https://example.com",
        "industry": "Software",
        "employee_count": "51-200",
        "country": "United States",
        "intent_signal": "hiring sales reps",
        "intent_url": "https://example.com/careers",
        "intent_signals": [
            {
                "source": "company_site",
                "url": "https://example.com/careers",
                "signal": "hiring sales reps",
                "matched_icp_signal": 0,
            },
            {
                "source": "news",
                "url": "https://example.com/news",
                "matched_icp_signal": 0,
            },
        ],
    }
    normalized = evaluator._normalize_company_output(
        company,
        {"intent_signals": ["hiring sales reps"]},
    )

    assert len(normalized["intent_signals"]) == 2
    for signal in normalized["intent_signals"]:
        assert signal["description"]
        assert signal["snippet"]
        assert signal["url"].startswith("https://example.com/")
        assert signal["matched_icp_signal"] == 0


# ---------------------------------------------------------------------------
# Defaults — score-affecting flags OFF, fairness fixes ON
# ---------------------------------------------------------------------------


def test_flag_defaults():
    assert evaluator._timeout_latch_legacy_enabled() is False  # bug #14 fix active
    assert evaluator._provider_flake_retry_enabled() is True  # bug #15 fix active
    assert evaluator._capped_top5_score_enabled() is False  # score-scale change OFF
    assert evaluator._max_scored_companies_per_icp() == 0  # unlimited (legacy)
    assert evaluator._candidate_scoring_concurrency() == 1  # serial (legacy)


# ---------------------------------------------------------------------------
# Bug #31 — per-ICP checkpoint / resume interface
# ---------------------------------------------------------------------------


async def test_icp_checkpoint_receives_each_completed_row():
    runner = ScriptedRunner()
    checkpointed = []
    rows = await _score_items(
        runner,
        _benchmark_items(3),
        icp_checkpoint=checkpointed.append,
    )
    assert checkpointed == rows
    assert [row["icp_ref"] for row in checkpointed] == ["icp-0", "icp-1", "icp-2"]


async def test_resume_results_skip_completed_icps():
    resumed_row = {
        "icp_ref": "icp-0",
        "icp_hash": "hash-0",
        "status": "completed",
        "hard_failure": False,
        "base_company_scores": [],
        "candidate_company_scores": [77.0],
        "failure_reason": "",
        "provider_excluded": False,
        "sourced_zero_no_error": False,
        "reference_sourced_zero_no_error": False,
    }
    runner = ScriptedRunner()
    checkpointed = []
    rows = await _score_items(
        runner,
        _benchmark_items(3),
        icp_checkpoint=checkpointed.append,
        resume_results=[resumed_row],
    )
    # The resumed ICP is not re-run and its stored row is reused verbatim.
    assert runner.call_count("icp-0") == 0
    assert rows[0] == resumed_row
    assert runner.call_count("icp-1") == 1
    assert runner.call_count("icp-2") == 1
    # Ordering follows the benchmark items, and resumed rows are not
    # re-checkpointed.
    assert [row["icp_ref"] for row in rows] == ["icp-0", "icp-1", "icp-2"]
    assert [row["icp_ref"] for row in checkpointed] == ["icp-1", "icp-2"]


async def test_async_icp_checkpoint_supported():
    runner = ScriptedRunner()
    checkpointed = []

    async def sink(row):
        checkpointed.append(row["icp_ref"])

    await _score_items(runner, _benchmark_items(2), icp_checkpoint=sink)
    assert checkpointed == ["icp-0", "icp-1"]


# ---------------------------------------------------------------------------
# §5.3 — optional candidate ICP concurrency, deterministic ordering
# ---------------------------------------------------------------------------


async def test_concurrent_candidate_scoring_orders_results_deterministically(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY", "3")
    total = 6

    class DelayedRunner:
        def __init__(self):
            self.calls = []

        async def __call__(self, icp, context):
            index = int(str(icp["name"]).split("-")[1])
            # Later ICPs in a wave finish first to exercise reordering.
            await asyncio.sleep((total - index) * 0.01)
            self.calls.append(index)
            return [{"score": float(index * 10)}]

    runner = DelayedRunner()
    rows = await evaluator.score_private_model_pair_items(
        benchmark_items=_benchmark_items(total),
        base_runner=None,
        candidate_runner=runner,
        company_scorer=_fake_scorer,
        run_context={"run_id": "run-1"},
        image_candidate=True,
    )
    assert [row["icp_ref"] for row in rows] == [f"icp-{index}" for index in range(total)]
    assert [row["candidate_company_scores"] for row in rows] == [
        [float(index * 10)] for index in range(total)
    ]
    # Completion order actually differed from benchmark order within waves.
    assert runner.calls != sorted(runner.calls)


# ---------------------------------------------------------------------------
# Bundle surface — legacy provider_excluded_icp_ids remains empty for new bundles
# ---------------------------------------------------------------------------


def _artifact_manifest(seed: str) -> dict:
    payload = {
        "model_artifact_hash": sha256_json({"artifact": seed}),
        "git_commit_sha": "a" * 12,
        "image_digest": (
            f"123456789012.dkr.ecr.us-east-1.amazonaws.com/private/{seed}@sha256:{'0' * 64}"
        ),
        "config_hash": sha256_json({"config": seed}),
        "component_registry_version": "1.0",
        "scoring_adapter_version": "1.0",
        "manifest_uri": f"s3://research-lab-test/{seed}.json",
        "signature_ref": f"kms-signature:test-key:{seed}",
        "build_id": "",
    }
    return {**payload, "manifest_hash": sha256_json(payload)}


async def test_bundle_keeps_retry_exhausted_provider_failures_in_aggregate_and_health():
    runner = ScriptedRunner(
        script={
            "icp-1": [
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
                PrivateModelRuntimeError(PROVIDER_RETRYABLE_MSG),
            ]
        }
    )
    rows = await _score_items(runner, _benchmark_items(3))
    parent = _artifact_manifest("parent")
    candidate = _artifact_manifest("candidate")
    bundle = evaluator.build_score_bundle_from_scored_icps(
        artifact_manifest=parent,
        benchmark={
            "benchmark_id": "bench-1",
            "icp_set_hash": sha256_json({"icps": 1}),
            "split_ref": "split-1",
            "item_refs": ["icp-0", "icp-1", "icp-2"],
            "scoring_version": "v1",
        },
        patch_manifest={"parent_artifact_hash": parent["model_artifact_hash"]},
        candidate_artifact_manifest=candidate,
        per_icp_results=rows,
        run_context={
            "run_id": "run-1",
            "ticket_id": "ticket-1",
            "miner_hotkey": "hotkey-1",
            "island": "island-1",
            "evaluation_epoch": 1,
            "evaluator_version": "eval-1",
            "execution_trace_ref": "trace-1",
            "cost_ledger_ref": "cost-1",
            "signature_ref": "pending",
        },
    )
    assert bundle["provider_excluded_icp_ids"] == []
    assert bundle["scoring_health"]["provider_excluded_icp_count"] == 0
    assert bundle["scoring_health"]["provider_excluded_icp_rate"] == pytest.approx(0.0)
    assert bundle["scoring_health"]["provider_error_count"] == 1
    expected_clean_icp = aggregation.per_icp_normalized_score([50.0], max_leads=5)
    assert bundle["aggregates"]["candidate_score"] == pytest.approx(
        (expected_clean_icp + 0.0 + expected_clean_icp) / 3
    )
    assert bundle["score_bundle_hash"] == bundle["anchored_hash"]


def test_score_bundle_records_serving_version_and_per_icp_context(tmp_path):
    parent = _artifact_manifest("parent-versioned")
    candidate = _artifact_manifest("candidate-versioned")
    benchmark = SealedBenchmarkSet(
        benchmark_id="rolling:bench",
        icp_set_hash=sha256_json({"window": "2026-07-09"}),
        split_ref="s3://research-lab-test/split.json",
        item_refs=("icp-a",),
        scoring_version="scoring-v1",
    )
    bundle = evaluator.build_score_bundle_from_scored_icps(
        artifact_manifest=PrivateModelArtifactManifest.from_mapping(parent),
        benchmark=benchmark,
        patch_manifest={
            "parent_artifact_hash": parent["model_artifact_hash"],
            "patch_kind": "image_build",
        },
        candidate_artifact_manifest=PrivateModelArtifactManifest.from_mapping(candidate),
        per_icp_results=[
            {
                "icp_ref": "icp-a",
                "icp_hash": sha256_json({"icp": "a"}),
                "base_company_scores": [10.0],
                "candidate_company_scores": [20.0, 30.0],
                "status": "completed",
            }
        ],
        run_context={
            "run_id": "run-versioned",
            "ticket_id": "ticket-versioned",
            "candidate_id": "candidate-row-1",
            "private_model_version_id": "private-version-1",
            "miner_hotkey": "hotkey-versioned",
            "island": "generalist",
            "evaluation_epoch": 77,
            "benchmark_date": "2026-07-09",
            "benchmark_attempt": 2,
            "run_scope": "candidate_scoring",
            "rolling_window_hash": benchmark.icp_set_hash,
            "provider_cache_day": "2026-07-09",
            "evaluator_version": "eval-versioned",
            "execution_trace_ref": "execution_trace:run-versioned",
            "cost_ledger_ref": "cost_ledger:run-versioned",
            "signature_ref": "kms-signature:test",
            "candidate_source_diff_hash": sha256_json({"diff": "x"}),
            "candidate_build_ref": "candidate_build:image",
        },
        policy={"reference_evaluation_mode": "paired_base"},
    )

    verification = verify_research_evaluation_score_bundle(bundle)
    assert verification["passed"], verification["errors"]
    serving = bundle["serving_model_version"]
    assert serving["result_role"] == "candidate_scoring"
    assert serving["candidate_id"] == "candidate-row-1"
    assert serving["parent_model"]["model_artifact_hash"] == parent["model_artifact_hash"]
    assert serving["candidate_model"]["model_artifact_hash"] == candidate["model_artifact_hash"]
    assert serving["parent_model"]["image_ref_hash"].startswith("sha256:")
    assert serving["candidate_model"]["image_ref_hash"].startswith("sha256:")
    assert serving["parent_model"]["manifest_ref_hash"].startswith("sha256:")
    assert serving["candidate_model"]["manifest_ref_hash"].startswith("sha256:")
    assert "image_digest" not in json.dumps(serving, sort_keys=True)
    assert "manifest_uri" not in json.dumps(serving, sort_keys=True)
    assert ".dkr.ecr." not in json.dumps(serving, sort_keys=True)
    assert serving["version_stamp_hash"].startswith("sha256:")
    assert serving["public_stamp_hash"].startswith("sha256:")
    row = bundle["aggregates"]["per_icp_results"][0]
    context = row["evaluation_context"]
    assert context["schema_version"] == "research_lab_evaluation_context.v1"
    assert context["run_id"] == "run-versioned"
    assert context["candidate_id"] == "candidate-row-1"
    assert context["icp_set_hash"] == benchmark.icp_set_hash
    assert context["serving_model_version_hash"] == serving["version_stamp_hash"]
    assert context["result_row_hash"].startswith("sha256:")

    page_verification = verify_research_lab_evaluation_bundle_page(
        {
            "bundle_type": "research_lab_evaluation_score_bundle_page",
            "epoch": 77,
            "score_bundles": [
                {
                    "score_bundle_id": "score_bundle:" + bundle["score_bundle_hash"].split(":", 1)[1],
                    "bundle_status": "scored",
                    "current_event_status": "scored",
                    "score_bundle_doc": bundle,
                }
            ],
            "on_chain_submission_allowed": False,
        },
        flags={"evaluation_verify_enabled": True},
    )
    assert page_verification["passed"], page_verification["errors"]

    legacy_bundle = json.loads(json.dumps(bundle))
    legacy_bundle["serving_model_version"]["parent_model"]["image_digest"] = parent["image_digest"]
    legacy_bundle["serving_model_version"]["parent_model"]["manifest_uri"] = parent["manifest_uri"]
    legacy_bundle["serving_model_version"]["candidate_model"]["image_digest"] = (
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/model@sha256:" + "f" * 64
    )
    legacy_bundle["serving_model_version"]["candidate_model"]["manifest_uri"] = candidate["manifest_uri"]
    legacy_bundle["score_bundle_hash"] = score_bundle_hash(legacy_bundle)
    legacy_bundle["anchored_hash"] = legacy_bundle["score_bundle_hash"]
    legacy_verification = verify_research_evaluation_score_bundle(legacy_bundle)
    assert legacy_verification["passed"], legacy_verification["errors"]

    legacy_page = {
        "bundle_type": "research_lab_evaluation_score_bundle_page",
        "epoch": 77,
        "score_bundles": [
            {
                "score_bundle_id": "score_bundle:" + legacy_bundle["score_bundle_hash"].split(":", 1)[1],
                "bundle_status": "scored",
                "current_event_status": "scored",
                "score_bundle_doc": legacy_bundle,
            }
        ],
        "on_chain_submission_allowed": False,
    }
    legacy_page_verification = verify_research_lab_evaluation_bundle_page(
        legacy_page,
        flags={"evaluation_verify_enabled": True},
    )
    assert legacy_page_verification["passed"], legacy_page_verification["errors"]
    artifact_path = write_research_lab_validator_artifact(
        output_dir=tmp_path,
        epoch=77,
        bundle=legacy_page,
        verification=legacy_page_verification,
        component={"verified_weight_inputs": legacy_page_verification["verified_weight_inputs"]},
        artifact_kind="evaluation",
    )
    rendered_artifact = artifact_path.read_text(encoding="utf-8")
    assert "image_digest" not in rendered_artifact
    assert "manifest_uri" not in rendered_artifact
    assert ".dkr.ecr." not in rendered_artifact
    assert parent["manifest_uri"] not in rendered_artifact
    assert candidate["manifest_uri"] not in rendered_artifact

    mutable_image_bundle = json.loads(json.dumps(bundle))
    mutable_image_bundle["serving_model_version"]["candidate_model"]["image_digest"] = (
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/model:latest"
    )
    mutable_image_verification = verify_research_evaluation_score_bundle(mutable_image_bundle)
    assert "score_bundle_contains_raw_secret_material" in mutable_image_verification["errors"]

    misplaced_image_bundle = json.loads(json.dumps(bundle))
    misplaced_image_bundle["image_digest"] = candidate["image_digest"]
    misplaced_image_verification = verify_research_evaluation_score_bundle(misplaced_image_bundle)
    assert "score_bundle_contains_raw_secret_material" in misplaced_image_verification["errors"]

    secret_named_manifest_bundle = json.loads(json.dumps(legacy_bundle))
    secret_named_manifest_bundle["serving_model_version"]["candidate_model"]["manifest_uri"] = (
        "s3://research-lab-test/private_repo/candidate.json"
    )
    secret_named_manifest_verification = verify_research_evaluation_score_bundle(secret_named_manifest_bundle)
    assert "score_bundle_contains_raw_secret_material" in secret_named_manifest_verification["errors"]

    unsafe_manifest_bundle = json.loads(json.dumps(bundle))
    unsafe_manifest_bundle["serving_model_version"]["candidate_model"]["manifest_uri"] = (
        "s3://private-model-artifacts/candidate.json"
    )
    unsafe_manifest_verification = verify_research_evaluation_score_bundle(unsafe_manifest_bundle)
    assert "score_bundle_contains_raw_secret_material" in unsafe_manifest_verification["errors"]
