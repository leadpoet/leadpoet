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
import json
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, ArenaStoreError, FUNCTION_SIGNATURES, PsycopgTransport
from tests.lab_arena import test_lab_arena_service_round as rt
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena.test_lab_arena_postgrest_route import make_transport, stack as compatibility_stack

psycopg2 = pytest.importorskip("psycopg2")

ROUTING_OR_COERCION_MARKERS = ("code=42883", "code=42P01", "code=22P02", "code=22003", "code=22007", "PGRST", "HTTP 404", "HTTP 400 code=None")


@pytest.fixture()
def stack(tmp_path_factory):
    # Each sequence variant owns its publication/reward history.
    yield from compatibility_stack.__wrapped__(tmp_path_factory)


class RestHarness(IntegrityHarness):
    advance_until = rt.Harness.advance_until

    def __init__(self, stack, tmp_path, *, baseline_first=False, **kwargs):
        self.stack = stack
        self.baseline_first = baseline_first
        self.calls = collections.Counter()
        info = stack["connection"].info
        connect = lambda: psycopg2.connect(host=info.host, port=info.port, user="postgres", password="postgres", dbname="postgres")
        super().__init__(connect, tmp_path, **kwargs)

    def build_service(self):
        service = super().build_service()
        if self.baseline_first:
            service.config.defaults = replace(service.config.defaults,
                execution_sequence_from="2000-01-01T00:00:00Z", per_icp_cost_policy=True)
        return service

    def objects_key(self) -> str:
        return "postgrest"

    def make_store(self) -> ArenaStore:
        transport = make_transport(self.stack, "lab_arena_service")
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


@pytest.mark.parametrize("baseline_first", [False, True])
def test_full_round_through_postgrest_reaches_every_service_function(stack, tmp_path, monkeypatch, baseline_first):
    from tests.lab_arena.champion_funding_round_test import ChampionTransport
    from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge
    # Current billing requires the final request-bound Deepline receipt that
    # the current native champion fixture supplies. No paid service is used.
    monkeypatch.setattr(rt, "FakeProviderTransport", ChampionTransport)
    monkeypatch.setattr(rt, "deterministic_scorer", _baseline_first_judge)
    harness = RestHarness(stack, tmp_path, baseline_first=baseline_first, challengers=["Rest-A", "Rest-B", "Rest-C"], runners=["alpha", "beta"])
    if baseline_first:
        from tests.lab_arena.baseline_scored_first_postgres_test import _enable_verified_proxy_runtime
        _enable_verified_proxy_runtime(harness)
    service = harness.service
    identity = service.store.require_service_role()
    assert identity["current_user"] == "lab_arena_service" and identity["jwt_role"] == "lab_arena_service"
    for row in service.store.list_rounds():
        if row["status"] not in ("published", "cancelled"):
            service.store.cancel_round(row["round_id"], "operator_abort")
    # The operator's first command (--check-only) parses PostgREST error bodies for the function probes.
    checks = service.startup_checks()
    assert checks["database_identity"]["current_user"] == "lab_arena_service" and checks["current_round"] is None
    # Current review admission starts after the replacement freeze. Match
    # the existing native full-round helper's real SQL/signed-clock window.
    harness.clock.now = datetime.now(timezone.utc)
    cutoff = datetime.now(timezone.utc) + timedelta(minutes=30)
    configuration = service.create_round(cutoff, round_id="arena-2026-10-28-rest" + str(int(baseline_first)))
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    # With a current round, startup also compares the pinned identities of this build.
    assert harness.build_service().startup_checks()["current_round"] == round_id
    harness.clock.now = cutoff - timedelta(minutes=15)
    submissions = {flavor: harness.submit(flavor, round_id) for flavor in harness.challengers}
    participant_count = len(harness.challengers) + 1  # challengers plus the daily public baseline
    stage_one_work = contracts.BENCHMARK_ICP_COUNT if baseline_first else contracts.STAGE_1_ICP_COUNT * participant_count
    assert service.advance_round(round_id)["status"] == "waiting"
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    committed = service.advance_round(round_id)
    assert committed["status"] == "ok" and committed["participants"] == participant_count and harness.status() == "committed"
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = service.advance_round(round_id)
    assert opened["status"] == "ok" and opened["assignments"] == stage_one_work and harness.status() == "stage1"
    harness.run_stage_with_runners(2)
    runs = service.store.list_runs(round_id, stage=1)
    assert len(runs) == stage_one_work and all(run["status"] == "accepted" for run in runs)
    assert service.advance_round(round_id)["work_items"] == stage_one_work and harness.status() == "stage1_closed"
    assert service.advance_round(round_id)["work_items"] == stage_one_work and harness.status() == "stage1_scoring"
    harness.run_stage_with_runners(2)
    after_baseline = service.advance_round(round_id)
    if baseline_first:
        # Current continuation SQL opens miner execution directly once every
        # baseline judgment is accepted; legacy rounds retain the old phases.
        assert after_baseline["status"] == "ok"
        assert after_baseline["assignments"] == contracts.BENCHMARK_ICP_COUNT * len(harness.challengers)
        assert harness.status() == "stage2"
    else:
        assert after_baseline["status"] == "closed" and harness.status() == "stage1_judged"
    # A restarted service (a fresh PostgREST client) continues the same round.
    harness.service = harness.build_service()
    service = harness.service
    if not baseline_first:
        assert service.advance_round(round_id)["judge_executions"] == stage_one_work and harness.status() == "stage1_scored"
    else:
        assert service.store.get_round(round_id)["status"] == "stage2"
    harness.advance_until("scored", runners=2)
    with harness.connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT public.lab_arena__baseline_final_ready_v1(%s),"
                       "public.lab_arena__all_judgments_ready_v1(%s)", (round_id, round_id))
        assert cursor.fetchone() == (True, True)
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
    # Compare every remaining registered RPC against the same native SQL
    # signature/role. Current capability reads and missing-item predicates may
    # return normal JSON/booleans instead of a domain error; both are valid.
    native = PsycopgTransport(harness.connect, pool_size=1)
    try:
        for function in sorted(set(FUNCTION_SIGNATURES) - reached):
            params = {name: placeholder(name, sql_type) for name, sql_type in FUNCTION_SIGNATURES[function]}
            try:
                expected = native.rpc(function, params)
            except ArenaStoreError as expected_error:
                with pytest.raises(ArenaStoreError) as received:
                    harness.transport.rpc(function, params)
                assert received.value.store_error.get("code") == expected_error.store_error.get("code"), function
                assert not any(marker in str(received.value) for marker in ROUTING_OR_COERCION_MARKERS), function
            else:
                assert harness.transport.rpc(function, params) == expected, function
    finally:
        native.close()
    # The cancel path is real, not a probe: a fresh round is cancelled through PostgREST.
    second = service.create_round(harness.clock() + timedelta(hours=1), round_id="arena-2026-10-28-restcancel" + str(int(baseline_first)))
    assert service.store.cancel_round(second["round_id"], "operator_abort")["status"] in ("ok", "cancelled")
    assert service.store.get_round(second["round_id"])["status"] == "cancelled"
