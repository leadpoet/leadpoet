"""Section 18.4: one complete round through a real PostgREST container.

Every production Arena write goes through PostgREST RPC with the
``lab_arena_service`` JWT. The other round tests use the psycopg transport;
this one proves each service function's parameter coercion and result shape
on the PostgREST path, then probes every function the round did not reach
with production-shaped parameters so a routing or coercion failure (an
unknown function, a bad cast) is distinguished from the function's own
domain refusal.
"""

from __future__ import annotations

import collections
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, ArenaStoreError, FUNCTION_SIGNATURES
from tests.lab_arena import test_lab_arena_service_round as rt
from tests.lab_arena.test_lab_arena_postgrest_route import make_transport, stack  # noqa: F401  (module fixture)

psycopg2 = pytest.importorskip("psycopg2")

ROUTING_OR_COERCION_MARKERS = ("code=42883", "code=42P01", "code=22P02", "code=22003", "code=22007", "PGRST", "HTTP 404", "HTTP 400 code=None")


class RestHarness(rt.Harness):
    def __init__(self, stack, tmp_path, **kwargs):
        self.stack = stack
        self.calls = collections.Counter()
        info = stack["connection"].info
        connect = lambda: psycopg2.connect(host=info.host, port=info.port, user="postgres", password="postgres", dbname="postgres")
        super().__init__(connect, tmp_path, **kwargs)

    def objects_key(self) -> str:
        return 'postgrest-' + str(self.tmp)

    def make_store(self) -> ArenaStore:
        transport = make_transport(self.stack, "lab_arena_service")
        original_post, original_get = transport._client.post, transport._client.get
        transport._client.post = lambda url, **kw: original_post(url.replace("/rest/v1", ""), **kw)
        transport._client.get = lambda url, **kw: original_get(url.replace("/rest/v1", ""), **kw)
        original_rpc = transport.rpc
        calls = self.calls

        def counted(function, *args, **kwargs):
            calls[function] += 1
            return original_rpc(function, *args, **kwargs)

        transport.rpc = counted
        self.transport = transport
        return ArenaStore(transport, lease_ttl_seconds=420)


def placeholder(param: str, sql_type: str):
    lowered = sql_type.lower()
    if lowered.endswith("[]"):
        return []
    if lowered in ("jsonb", "json"):
        return {}
    if lowered in ("bigint", "integer", "int", "numeric", "smallint"):
        return 0
    if lowered == "boolean":
        return False
    if lowered == "timestamptz":
        return "2026-09-02T00:00:00+00:00"
    if lowered == "date":
        return "2026-09-02"
    return "missing-" + param.replace("p_", "").replace("_", "-")


@pytest.mark.parametrize('stack', ['current'], indirect=True)
def test_full_round_through_postgrest_reaches_every_service_function(stack, tmp_path):
    harness = RestHarness(stack, tmp_path, challengers=["Rest-A", "Rest-B", "Rest-C"], runners=["alpha", "beta"])
    service = harness.service
    identity = service.store.require_service_role()
    assert identity["current_user"] == "lab_arena_service" and identity["jwt_role"] == "lab_arena_service"
    for row in service.store.list_rounds():
        if row["status"] not in ("published", "cancelled"):
            service.store.cancel_round(row["round_id"], "operator_abort")
    # The operator's first command (--check-only) parses PostgREST error bodies for the function probes.
    checks = service.startup_checks()
    assert checks["database_identity"]["current_user"] == "lab_arena_service" and checks["current_round"] is None
    cutoff = datetime.now(timezone.utc) + timedelta(minutes=30)
    configuration = service.create_round(cutoff, round_id="arena-2026-09-03-rest")
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    # With a current round, startup also compares the pinned identities of this build.
    assert harness.build_service().startup_checks()["current_round"] == round_id
    submissions = {flavor: harness.submit(flavor, round_id) for flavor in harness.challengers}
    participant_count = len(harness.challengers) + 1  # challengers plus the daily public baseline
    assert service.advance_round(round_id)["status"] == "waiting"
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    committed = service.advance_round(round_id)
    assert committed["status"] == "ok" and committed["participants"] == participant_count and harness.status() == "committed"
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = service.advance_round(round_id)
    assert opened["status"] == "ok" and opened["assignments"] == contracts.STAGE_1_ICP_COUNT * participant_count and harness.status() == "stage1"
    harness.run_stage_with_runners(2)
    runs = service.store.list_runs(round_id, stage=1)
    assert len(runs) == contracts.STAGE_1_ICP_COUNT * participant_count and all(run["status"] == "accepted" for run in runs)
    assert service.advance_round(round_id)["work_items"] == contracts.STAGE_1_ICP_COUNT * participant_count and harness.status() == "stage1_closed"
    assert service.advance_round(round_id)["work_items"] == contracts.STAGE_1_ICP_COUNT * participant_count and harness.status() == "stage1_scoring"
    harness.run_stage_with_runners(2)
    assert service.advance_round(round_id)["status"] == "closed" and harness.status() == "stage1_judged"
    # A restarted service (a fresh PostgREST client) continues the same round.
    harness.service = harness.build_service()
    service = harness.service
    assert service.advance_round(round_id)["judge_executions"] == contracts.STAGE_1_ICP_COUNT * participant_count and harness.status() == "stage1_scored"
    harness.advance_until("published", runners=2)
    assert harness.status() == "published"
    rt.assert_canary_absent(harness, harness.connect)
    row = service.store.get_round(round_id)
    publication = row["publication_doc"]
    assert len(publication["participants"]) == participant_count
    assert len(publication["stage1_ranking"]) == len(harness.challengers)
    assert len(publication["final_ranking"]) == participant_count
    assert publication["king_decision"]["outcome"] in ("crowned", "no_king")
    execution_runs = service.store.list_runs(round_id, kind="execute")
    assert len(execution_runs) == (
        contracts.STAGE_1_ICP_COUNT + contracts.STAGE_2_ICP_COUNT
    ) * participant_count
    assert all(run["status"] == "accepted" and run["per_icp_score"] is not None for run in execution_runs)
    # Lease expiry runs on every driver tick; the round exercised it with nothing to expire.
    # The service-only read RPC returns real participation, not a domain error.
    runner_hotkey = harness.runner_keys[0]
    assert service.store.has_recent_participation("finney", 71, runner_hotkey)
    assert not service.store.has_recent_participation("test", 71, runner_hotkey)
    reached = set(harness.calls)
    assert {"lab_arena_create_round", "lab_arena_transition_round", "lab_arena_update_submission", "lab_arena_open_stage", "lab_arena_close_scoring", "lab_arena_claim_assignment", "lab_arena_reserve_call", "lab_arena_mark_dispatched", "lab_arena_settle_call", "lab_arena_close_stage", "lab_arena_record_run_scores", "lab_arena_whoami"} <= reached, sorted(reached)
    assert "lab_arena_register_submission" in reached or "lab_arena_register_submission_v2" in reached
    assert "lab_arena_open_scoring" in reached or "lab_arena_open_scoring_v2" in reached
    assert "lab_arena_complete_attempt" in reached or "lab_arena_complete_attempt_v2" in reached
    # Every function the round did not reach is routed and coerced by PostgREST; only its own domain check refuses.
    capability_versions = {
        "lab_arena_schema_version_v1": 197,
        "lab_arena_weight_state_schema_v1": 202,
        "lab_arena_code_review_schema_v1": 207,
        "lab_arena_validator_scoring_authority_schema_v1": 208,
        "lab_arena_integrity_schema_v1": 213,
        "lab_arena_contact_schema_v1": 215,
        "lab_arena_participation_schema_v1": 221,
        "lab_arena_champion_funding_schema_v1": 227,
        "lab_arena_successful_call_cost_schema_v1": 230,
        "lab_arena_twenty_icp_promotion_schema_v1": 251,
        "lab_arena_baseline_cost_eligibility_schema_v1": 254,
        "lab_arena_submission_replacement_schema_v1": 258,
        "lab_arena_per_icp_cost_schema_v1": 289,
        "lab_arena_dynamic_benchmark_schema_v1": 353,
        "lab_arena_submission_duplicate_schema_v1": 405,
        "lab_arena_deepline_catalog_schema_v1": 415,
        "lab_arena_deepline_response_schema_v1": 426,
        "lab_arena_infrastructure_retry_cost_schema_v1": 478,
        "lab_arena_company_quality_schema_v1": 1,
    }
    empty_responses = {
        "lab_arena_current_daily_icp_set": {"status": "unavailable", "set_id": 0},
        "lab_arena_next_closed_deepline_reconciliation_v1": {"status": "none"},
        "lab_arena_next_closed_provider_reconciliation_v1": {"status": "none"},
        "lab_arena_list_deepline_cost_reconciliations_v1": {"status": "ok", "items": []},
        "lab_arena_list_deepline_cost_reconciliations_v2": {"status": "ok", "items": []},
        "lab_arena_list_deepline_cost_reconciliations_v3": {"status": "ok", "items": []},
        "lab_arena_list_openrouter_cost_reconciliations_v1": {"status": "ok", "items": []},
        "lab_arena_latest_promoted_champion_v1": None,
        "lab_arena_signal_authority_exclusions_v1": None,
        "lab_arena__miner_credit_judgment_terminal_v1": False,
        "lab_arena_operator_hold_active_v1": False,
        "lab_arena_has_recent_participation_v1": {"eligible": False},
        "lab_arena_retry_credit_failures_v1": {
            "status": "no_eligible", "reason": "invalid_request_hash", "requeued_count": 0},
        "lab_arena_scoring_policy_history_v1": {"round_id": "missing-round-id", "revisions": []},
        "lab_arena_review_guard_state_v1": {
            "schema_version": "leadpoet.lab_arena.review_guard_state.v1", "guard_present": False},
    }
    for function in sorted(set(FUNCTION_SIGNATURES) - reached):
        params = {name: placeholder(name, sql_type) for name, sql_type in FUNCTION_SIGNATURES[function]}
        if function in capability_versions:
            assert harness.transport.rpc(function, params)["version"] == capability_versions[function], function
            continue
        if function in empty_responses:
            assert harness.transport.rpc(function, params) == empty_responses[function], function
            continue
        if function in ("lab_arena_parallel_execution_schema_v1", "lab_arena_parallel_execution_schema_v2"):
            expanded = function.endswith("v2")
            assert harness.transport.rpc(function, params) == {
                "schema_version": "leadpoet.lab_arena.parallel_execution_schema.v" + ("2" if expanded else "1"),
                "version": 401 if expanded else 255,
                "max_parallel_icps": 251 if expanded else 20,
            }
            continue
        if function == "lab_arena_restart_guard_state_v1":
            guard = harness.transport.rpc(function, params)
            assert guard["schema_version"] == "leadpoet.lab_arena.restart_guard_state.v1"
            assert all(guard[field] is False for field in ("guard_present", "guard_active", "operator_paused", "paused"))
            assert guard["drain"]["preserved"] is True
            assert all(value == 0 for key, value in guard["drain"].items() if key.endswith("_count"))
            continue
        with pytest.raises(ArenaStoreError) as excinfo:
            harness.transport.rpc(function, params)
        message = str(excinfo.value)
        assert not any(marker in message for marker in ROUTING_OR_COERCION_MARKERS), (function, message)
        assert "lab_arena_" in message or "code=P0" in message, (function, message)
    # The cancel path is real, not a probe: a fresh round is cancelled through PostgREST.
    second = service.create_round(cutoff, round_id="arena-2026-09-03-restcancel")
    assert service.store.cancel_round(second["round_id"], "operator_abort")["status"] in ("ok", "cancelled")
    assert service.store.get_round(second["round_id"])["status"] == "cancelled"
