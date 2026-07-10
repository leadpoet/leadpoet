"""Additive scoring telemetry invariants.

These tests deliberately avoid a live Supabase project. They prove the local
writer/evaluator contracts that must hold before migration 83 is reviewed and
applied: deterministic identities/events, physical retry separation, cost
ordering, score/hash invariance, and failure containment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.research_lab import scoring_telemetry as telemetry
from gateway.research_lab import global_icp_queue
from gateway.research_lab import store as research_store
from gateway.research_lab.scoring_worker import _persist_provider_cost_events
from research_lab.canonical import sha256_json
from research_lab.eval import evaluator, private_runtime
from research_lab.eval.private_runtime import PrivateModelRuntimeError


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
CANDIDATE_ID = "candidate:" + "c" * 64


def _run_context() -> telemetry.ScoringRunContext:
    return telemetry.ScoringRunContext(
        scoring_id="scoring:" + HASH_A,
        scoring_run_id="11111111-1111-4111-8111-111111111111",
        run_type="candidate_scoring",
        run_attempt=0,
        worker_ref="worker:test",
        expected_icp_count=1,
        candidate_id=CANDIDATE_ID,
        source_run_id="22222222-2222-4222-8222-222222222222",
    )


def test_scoring_identity_is_stable_and_domain_separated():
    first = telemetry.scoring_identity({"candidate_id": CANDIDATE_ID, "window": HASH_A})
    second = telemetry.scoring_identity({"window": HASH_A, "candidate_id": CANDIDATE_ID})

    assert first == second
    assert first.startswith("scoring:sha256:")
    assert len(first) == len("scoring:sha256:") + 64
    assert first != telemetry.scoring_identity({"candidate_id": CANDIDATE_ID, "window": HASH_B})


@pytest.mark.asyncio
async def test_allocate_run_uses_monotonic_attempt_and_anchored_hash(monkeypatch):
    inserted: list[dict] = []

    async def fake_select_many(*args, **kwargs):
        return [{"run_attempt": 4}]

    async def fake_insert(table: str, row: dict):
        assert table == "research_lab_scoring_runs"
        inserted.append(dict(row))
        return row

    monkeypatch.setattr(telemetry, "select_many", fake_select_many)
    monkeypatch.setattr(telemetry, "insert_row", fake_insert)

    run = await telemetry.allocate_scoring_run(
        identity_doc={"candidate_id": CANDIDATE_ID, "window": HASH_A},
        run_type="candidate_scoring",
        worker_ref="worker:test",
        expected_icp_count=20,
        minimum_run_attempt=2,
        candidate_id=CANDIDATE_ID,
        source_run_id="22222222-2222-4222-8222-222222222222",
    )

    assert run is not None
    assert run.run_attempt == 5
    payload = {key: value for key, value in inserted[0].items() if key != "anchored_hash"}
    assert inserted[0]["anchored_hash"] == telemetry.canonical_hash(payload)


@pytest.mark.asyncio
async def test_allocate_run_links_and_closes_stale_predecessor(monkeypatch):
    old_run_id = "99999999-9999-4999-8999-999999999999"
    inserted: list[tuple[str, dict]] = []

    async def fake_select_many(*args, **kwargs):
        return [
            {
                "scoring_id": "scoring:" + HASH_A,
                "scoring_run_id": old_run_id,
                "run_type": "candidate_scoring",
                "run_attempt": 2,
                "worker_ref": "worker:dead",
                "expected_icp_count": 20,
                "current_run_status": "heartbeat",
                "current_status_at": "2020-01-01T00:00:00Z",
            }
        ]

    async def fake_insert(table: str, row: dict):
        inserted.append((table, dict(row)))
        return row

    monkeypatch.setattr(telemetry, "select_many", fake_select_many)
    monkeypatch.setattr(telemetry, "insert_row", fake_insert)

    run = await telemetry.allocate_scoring_run(
        identity_doc={"candidate_id": CANDIDATE_ID, "window": HASH_A},
        run_type="candidate_scoring",
        worker_ref="worker:new",
        expected_icp_count=20,
        candidate_id=CANDIDATE_ID,
        source_run_id="22222222-2222-4222-8222-222222222222",
    )

    assert run is not None
    assert run.run_attempt == 3
    assert run.resumed_from_scoring_run_id == old_run_id
    new_row = next(row for table, row in inserted if table == "research_lab_scoring_runs")
    assert new_row["resumed_from_scoring_run_id"] == old_run_id
    restarted = next(
        row for table, row in inserted
        if table == "research_lab_scoring_run_events" and row["event_type"] == "restarted"
    )
    assert restarted["scoring_run_id"] == old_run_id
    assert restarted["event_doc"]["resumed_by_scoring_run_id"] == run.scoring_run_id


def test_fresh_incomplete_run_is_not_misclassified_as_restart():
    assert telemetry._incomplete_run_is_stale(
        {
            "current_run_status": None,
            "created_at": telemetry.datetime.now(telemetry.timezone.utc).isoformat(),
        }
    ) is False


@pytest.mark.asyncio
async def test_retry_attempts_get_unique_executions_and_terminal_events(monkeypatch):
    rows: list[tuple[str, dict]] = []

    async def capture(table: str, row: dict, **kwargs):
        rows.append((table, dict(row)))
        return dict(row)

    monkeypatch.setattr(telemetry, "_best_effort_insert", capture)
    session = telemetry.ScoringTelemetrySession(_run_context())
    await session.plan(
        icp_ref="icp:1",
        icp_hash=HASH_B,
        icp_ordinal=0,
        model_role="candidate",
    )
    await session.attempt_started(
        icp_ref="icp:1",
        icp_hash=HASH_B,
        icp_ordinal=0,
        model_role="candidate",
        retry_round=0,
    )
    await session.lifecycle(
        "attempt_failed",
        {
            "icp_ref": "icp:1",
            "model_role": "candidate",
            "retry_round": 0,
            "retryable": True,
            "failure_category": "provider_rate_limit",
            "error": "status=429",
        },
    )
    await session.attempt_started(
        icp_ref="icp:1",
        icp_hash=HASH_B,
        icp_ordinal=0,
        model_role="candidate",
        retry_round=1,
    )
    await session.complete_result(
        {"icp_ref": "icp:1", "candidate_company_scores": [20.0, 40.0]},
        model_role="candidate",
        checkpoint_ref=telemetry.opaque_checkpoint_ref("bucket", "key"),
        checkpoint_hash=HASH_A,
    )

    executions = [row for table, row in rows if table.endswith("icp_executions")]
    events = [row for table, row in rows if table.endswith("icp_events")]
    assert len(executions) == 2
    assert executions[0]["icp_execution_id"] != executions[1]["icp_execution_id"]
    assert [row["retry_round"] for row in executions] == [0, 1]
    assert any(row["event_type"] == "failed" for row in events)
    completed = next(row for row in events if row["event_type"] == "completed")
    assert completed["score"] == pytest.approx(30.0)
    assert completed["checkpoint_hash"] == HASH_A


class _TraceRunner:
    def __init__(self):
        self.calls = 0

    async def __call__(self, icp, context):
        self.calls += 1
        private_runtime.publish_incontainer_trace_entries(
            [{"seq": self.calls, "phase": "provider_cost", "marker": f"call-{self.calls}"}]
        )
        if self.calls == 1:
            raise PrivateModelRuntimeError(
                "docker private model provider-backed sourcing failed before returning "
                "companies: HTTPError: too many requests; status=429; "
                "url=https://api.example.test/search"
            )
        return [{"score": 60.0}]


async def _score_with_optional_telemetry(*, enabled: bool):
    runner = _TraceRunner()
    trace_docs: list[list[dict]] = []
    ordering: list[str] = []

    async def scorer(companies, icp, is_reference):
        return [float(company["score"]) for company in companies]

    async def trace_sink(icp_ref: str, entries: list[dict]):
        trace_docs.append(list(entries))
        return "s3://trace/test.json"

    async def lifecycle(action: str, payload: dict):
        ordering.append(f"event:{action}:{payload.get('retry_round', 0)}")
        if action == "attempt_started":
            return {
                "scoring_id": "scoring:" + HASH_A,
                "scoring_run_id": "11111111-1111-4111-8111-111111111111",
                "icp_execution_id": f"33333333-3333-4333-8333-33333333333{payload.get('retry_round', 0)}",
            }
        return None

    async def cost_sink(icp_ref: str, entries: list[dict], context: dict):
        ordering.append(f"cost:{len(entries)}:{context.get('icp_execution_id', '')[-1:]}")

    rows = await evaluator.score_private_model_pair_items(
        benchmark_items=[{"icp_ref": "icp:1", "icp_hash": HASH_B, "icp": {"name": "one"}}],
        base_runner=None,
        candidate_runner=runner,
        company_scorer=scorer,
        run_context={"run_id": "run:test"},
        image_candidate=True,
        trace_sink=trace_sink,
        scoring_telemetry_hook=lifecycle if enabled else None,
        attempt_cost_sink=cost_sink if enabled else None,
    )
    return rows, trace_docs, ordering


@pytest.mark.asyncio
async def test_attempt_costs_precede_retry_terminal_and_do_not_change_result_hash(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE", "true")
    monkeypatch.setenv("RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS", "1")
    monkeypatch.setattr(evaluator, "_PROVIDER_429_RETRY_BACKOFF_SECONDS", 0.0)

    legacy_rows, legacy_traces, _ = await _score_with_optional_telemetry(enabled=False)
    v2_rows, v2_traces, ordering = await _score_with_optional_telemetry(enabled=True)

    assert v2_rows == legacy_rows
    assert sha256_json(v2_rows) == sha256_json(legacy_rows)
    assert v2_traces == legacy_traces
    assert ordering.index("cost:1:0") < ordering.index("event:attempt_failed:0")
    assert "cost:1:1" in ordering


@pytest.mark.asyncio
async def test_telemetry_callback_outage_cannot_change_scoring(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE", "true")
    monkeypatch.setenv("RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS", "1")
    monkeypatch.setattr(evaluator, "_PROVIDER_429_RETRY_BACKOFF_SECONDS", 0.0)
    expected, _, _ = await _score_with_optional_telemetry(enabled=False)

    runner = _TraceRunner()

    async def scorer(companies, icp, is_reference):
        return [float(company["score"]) for company in companies]

    async def broken_hook(action, payload):
        raise RuntimeError("telemetry db unavailable")

    async def broken_cost_sink(icp_ref, entries, context):
        raise RuntimeError("telemetry cost insert unavailable")

    async def trace_sink(icp_ref, entries):
        return "s3://trace/test.json"

    actual = await evaluator.score_private_model_pair_items(
        benchmark_items=[{"icp_ref": "icp:1", "icp_hash": HASH_B, "icp": {"name": "one"}}],
        base_runner=None,
        candidate_runner=runner,
        company_scorer=scorer,
        run_context={"run_id": "run:test"},
        image_candidate=True,
        trace_sink=trace_sink,
        scoring_telemetry_hook=broken_hook,
        attempt_cost_sink=broken_cost_sink,
    )
    assert actual == expected


@pytest.mark.asyncio
async def test_provider_cost_row_carries_all_or_none_telemetry_ids(monkeypatch):
    inserted: list[dict] = []

    async def capture(table: str, row: dict):
        inserted.append(dict(row))
        return row

    monkeypatch.setattr("gateway.research_lab.scoring_worker.insert_row", capture)
    entry = {
        "seq": 1,
        "provider_cost_event": {
            "provider": "exa",
            "endpoint": "/search",
            "request_fingerprint": "d" * 64,
            "status_code": 200,
            "billable": True,
            "cost_usd": 0.01,
            "cost_source": "test",
            "cap_usd": 1.0,
            "scope": "unchanged-provider-scope",
        },
    }
    await _persist_provider_cost_events(
        entries=[entry],
        run_type="candidate_scoring",
        icp_ref="icp:1",
        scoring_id="scoring:" + HASH_A,
        scoring_run_id="11111111-1111-4111-8111-111111111111",
        icp_execution_id="33333333-3333-4333-8333-333333333333",
    )

    assert inserted[0]["run_scope"] == "unchanged-provider-scope"
    assert {
        inserted[0]["scoring_id"],
        inserted[0]["scoring_run_id"],
        inserted[0]["icp_execution_id"],
    } == {
        "scoring:" + HASH_A,
        "11111111-1111-4111-8111-111111111111",
        "33333333-3333-4333-8333-333333333333",
    }


def test_migration_has_unified_v2_legacy_and_generation_contracts():
    sql = Path("scripts/83-research-lab-scoring-execution-telemetry.sql").read_text()

    required = {
        "research_lab_scoring_runs",
        "research_lab_scoring_run_events",
        "research_lab_scoring_icp_executions",
        "research_lab_scoring_icp_events",
        "research_lab_scoring_dashboard_telemetry_v2",
        "research_lab_scoring_dashboard_telemetry_legacy",
        "research_lab_scoring_dashboard_telemetry",
        "queue_generation_id",
        "attempt_count",
        "service_role",
        "telemetry_degraded",
    }
    for token in required:
        assert token in sql
    assert "scoring queue must be drained" in sql
    assert "private_baseline_rebenchmark" in sql
    assert "candidate_scoring" in sql
    assert "promotion_confirmation" in sql
    banned_payload_names = ("provider_output JSONB", "request_body JSONB", "response_body JSONB")
    assert all(name not in sql for name in banned_payload_names)


def test_checkpoint_index_is_observation_only():
    session = telemetry.ScoringTelemetrySession(_run_context())
    execution = telemetry.IcpExecutionContext(
        scoring_id=session.run.scoring_id,
        scoring_run_id=session.run.scoring_run_id,
        icp_execution_id="33333333-3333-4333-8333-333333333333",
        icp_ref="icp:1",
        icp_hash=HASH_B,
        icp_ordinal=0,
        model_role="candidate",
        retry_round=0,
        attempt_ordinal=0,
        execution_kind="model_invocation",
        phase="all",
        worker_ref="worker:test",
    )
    session.executions[("candidate", "icp:1", 0)] = execution
    session.latest_attempt[("candidate", "icp:1")] = 0
    rows = [{"icp_ref": "icp:1", "candidate_company_scores": [25.0]}]
    before = json.dumps(rows, sort_keys=True)

    index = telemetry.checkpoint_telemetry_index(session, rows, model_role="candidate")

    assert json.dumps(rows, sort_keys=True) == before
    assert index["executions"]["icp:1"]["icp_execution_id"] == execution.icp_execution_id
    assert "telemetry_index" not in rows[0]


@pytest.mark.asyncio
async def test_dispatch_anchor_is_invariant_to_additive_telemetry_link(monkeypatch):
    rows: list[dict] = []

    async def capture(table: str, row: dict):
        rows.append(dict(row))
        return row

    monkeypatch.setattr(research_store, "insert_row", capture)
    monkeypatch.setattr(
        research_store,
        "uuid4",
        lambda: "77777777-7777-4777-8777-777777777777",
    )
    common = {
        "dispatch_type": "candidate_scoring",
        "dispatch_status": "assigned",
        "worker_ref": "worker:test",
        "candidate_id": CANDIDATE_ID,
        "event_doc": {"phase": "test"},
    }
    await research_store.create_scoring_dispatch_event(**common)
    await research_store.create_scoring_dispatch_event(
        **common,
        scoring_id="scoring:" + HASH_A,
        scoring_run_id="11111111-1111-4111-8111-111111111111",
    )

    assert rows[0]["anchored_hash"] == rows[1]["anchored_hash"]
    assert "scoring_id" not in rows[0]
    assert rows[1]["scoring_id"] == "scoring:" + HASH_A


@pytest.mark.asyncio
async def test_global_queue_enqueue_scopes_every_row_to_one_generation(monkeypatch):
    inserted: list[tuple[str, dict]] = []

    async def no_existing(*args, **kwargs):
        return None

    async def capture(table: str, row: dict):
        inserted.append((table, dict(row)))
        return row

    monkeypatch.setattr(global_icp_queue, "select_one", no_existing)
    monkeypatch.setattr(global_icp_queue, "insert_row", capture)
    scoring_run_id = "11111111-1111-4111-8111-111111111111"
    generation = await global_icp_queue.enqueue_candidate(
        candidate_id=CANDIDATE_ID,
        window_hash=HASH_A,
        public_items=[{"icp_ref": "public:1"}],
        private_items=[{"icp_ref": "private:1"}],
        baseline_public_score=40.0,
        worker_ref="worker:test",
        seq_base=100,
        scoring_run_id=scoring_run_id,
    )

    assert generation
    assert len(inserted) == 3
    assert {row["queue_generation_id"] for _, row in inserted} == {generation}
    assert {row["scoring_run_id"] for _, row in inserted} == {scoring_run_id}
    jobs = [row for table, row in inserted if table == global_icp_queue.JOB_TABLE]
    assert {row["status"] for row in jobs} == {"queued", "held"}


@pytest.mark.asyncio
async def test_global_queue_completion_cas_binds_claimant_and_attempt(monkeypatch):
    observed_filters = None

    async def lose_cas(table, values, *, filters):
        nonlocal observed_filters
        observed_filters = filters
        return None

    monkeypatch.setattr(global_icp_queue, "_cas_update", lose_cas)
    committed = await global_icp_queue.complete_job(
        job={
            "job_id": "33333333-3333-4333-8333-333333333333",
            "claimed_by": "worker:old",
            "attempt_count": 2,
        },
        result_doc={"score": 50.0},
    )

    assert committed is False
    assert ("claimed_by", "worker:old") in observed_filters
    assert ("attempt_count", 2) in observed_filters


@pytest.mark.asyncio
async def test_stale_queue_execution_is_cancelled_before_requeue(monkeypatch):
    order: list[str] = []
    stale_job = {
        "job_id": "33333333-3333-4333-8333-333333333333",
        "status": "claimed",
        "attempt_count": 1,
    }

    async def select_stale(*args, **kwargs):
        return [stale_job]

    async def cas(table, values, *, filters):
        order.append(f"status:{values['status']}")
        return {**stale_job, **values}

    async def cancelled(job):
        order.append("telemetry:cancelled")

    monkeypatch.setattr(global_icp_queue, "select_many", select_stale)
    monkeypatch.setattr(global_icp_queue, "_cas_update", cas)

    recovered = await global_icp_queue.recover_stale_leases(
        stale_job_recovered=cancelled
    )

    assert recovered == 1
    assert order == ["status:held", "telemetry:cancelled", "status:queued"]


@pytest.mark.asyncio
async def test_queue_telemetry_callback_failure_cannot_reclassify_scored_job(monkeypatch):
    claimed = [
        {
            "job_id": "33333333-3333-4333-8333-333333333333",
            "candidate_id": CANDIDATE_ID,
            "queue_generation_id": "44444444-4444-4444-8444-444444444444",
            "phase": "private",
        },
        None,
    ]

    async def no_recovery(**kwargs):
        return 0

    async def claim(**kwargs):
        return claimed.pop(0)

    async def complete(**kwargs):
        return True

    async def not_ready(*args, **kwargs):
        return None

    async def score(job):
        return {"icp_ref": "icp:1", "score": 50.0}

    async def callback_outage(*args):
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr(global_icp_queue, "recover_stale_leases", no_recovery)
    monkeypatch.setattr(global_icp_queue, "claim_next_job", claim)
    monkeypatch.setattr(global_icp_queue, "complete_job", complete)
    monkeypatch.setattr(global_icp_queue, "candidate_ready_to_assemble", not_ready)

    counters = await global_icp_queue.run_queue_scoring_pass(
        worker_ref="worker:test",
        lease_seconds=60,
        score_icp=score,
        compute_public_score=lambda rows: 0.0,
        assemble_candidate=lambda generation, candidate, docs: None,
        job_completed=callback_outage,
    )

    assert counters["scored"] == 1
    assert counters["failed"] == 0
