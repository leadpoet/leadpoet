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
from datetime import datetime, timezone

import pytest

from lab_arena import verify
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
        return "postgrest"

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
    return "missing-" + param.replace("p_", "").replace("_", "-")


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
    configuration = service.create_round(datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc))
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    # With a current round, startup also compares the pinned identities of this build.
    assert harness.build_service().startup_checks()["current_round"] == round_id
    submissions = {flavor: harness.submit(flavor, round_id) for flavor in harness.challengers}
    assert service.advance_round(round_id)["status"] == "waiting"
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    committed = service.advance_round(round_id)
    assert committed["status"] == "ok" and committed["participants"] == 3 and harness.status() == "committed"
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = service.advance_round(round_id)
    assert opened["status"] == "ok" and opened["assignments"] == 60 and harness.status() == "stage1"
    harness.run_stage_with_runners(2)
    runs = service.store.list_runs(round_id, stage=1)
    assert len(runs) == 60 and all(run["status"] == "accepted" for run in runs)
    assert service.advance_round(round_id)["work_items"] == 60 and harness.status() == "stage1_closed"
    assert service.advance_round(round_id)["judge_executions"] == 60 and harness.status() == "stage1_scored"
    assert sorted(service.store.get_round(round_id)["finalists"]) == sorted(submissions.values())
    # A restarted service (a fresh PostgREST client) continues the same round.
    harness.service = harness.build_service()
    service = harness.service
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    assert service.advance_round(round_id)["assignments"] == 90 and harness.status() == "stage2"
    harness.run_stage_with_runners(2)
    assert service.advance_round(round_id)["status"] == "ok" and harness.status() == "stage2_closed"
    assert service.advance_round(round_id)["status"] == "ok" and harness.status() == "scored"
    published = service.advance_round(round_id)
    assert published["status"] == "ok" and published["king_outcome"] == "crowned" and harness.status() == "published"
    rt.assert_canary_absent(harness, harness.connect)
    row = service.store.get_round(round_id)
    bundle = json.loads(harness.objects.get(row["publication_doc"]["result_bundle_ref"]).decode("utf-8"))
    benchmark = service.public_benchmark(round_id)
    outputs = {}
    for entry in bundle["outputs"]:
        outputs[entry["output_hash"]] = json.loads(harness.objects.get(entry["output_ref"]).decode())
    verifier_bundle = {
        "round_configuration": bundle["round_configuration"], "benchmark_commitment": bundle["benchmark_commitment"], "benchmark": benchmark["icps"],
        "participants": bundle["participants"], "scorer_policy": bundle["scorer_policy"], "stage_plans": {"1": bundle["stage_plans"]["stage_1"], "2": bundle["stage_plans"]["stage_2"]},
        "score_bundles": {"1": bundle["score_bundles"]["stage_1"], "2": bundle["score_bundles"]["final"]}, "outputs": outputs,
        "stage1_ranking": bundle["stage1_ranking"], "finalists": bundle["finalists"], "final_ranking": bundle["final_ranking"], "king_decision": bundle["king_decision"], "reward_basis": row["reward_basis_doc"],
    }
    report = verify.rebuild_round(verifier_bundle, service.signing_key_document())
    assert report["ok"], report
    assert bundle["king_decision"]["outcome"] == "crowned" and bundle["king_decision"]["king_hotkey"] == row["king_hotkey"]
    # Lease expiry runs on every driver tick; the round exercised it with nothing to expire.
    reached = set(harness.calls)
    assert {"lab_arena_create_round", "lab_arena_transition_round", "lab_arena_append_journal_entry", "lab_arena_register_submission", "lab_arena_update_submission", "lab_arena_upsert_account_credential", "lab_arena_credit_deposit", "lab_arena_open_stage", "lab_arena_claim_assignment", "lab_arena_reserve_call", "lab_arena_mark_dispatched", "lab_arena_settle_call", "lab_arena_append_events", "lab_arena_complete_attempt", "lab_arena_close_stage", "lab_arena_record_run_scores", "lab_arena_whoami"} <= reached, sorted(reached)
    # Every function the round did not reach is routed and coerced by PostgREST; only its own domain check refuses.
    for function in sorted(set(FUNCTION_SIGNATURES) - reached):
        params = {name: placeholder(name, sql_type) for name, sql_type in FUNCTION_SIGNATURES[function]}
        with pytest.raises(ArenaStoreError) as excinfo:
            harness.transport.rpc(function, params)
        message = str(excinfo.value)
        assert not any(marker in message for marker in ROUTING_OR_COERCION_MARKERS), (function, message)
        assert "lab_arena_" in message or "code=P0" in message, (function, message)
    # The cancel path is real, not a probe: a fresh round is cancelled through PostgREST.
    second = service.create_round(datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc))
    assert service.store.cancel_round(second["round_id"], "operator_abort")["status"] in ("ok", "cancelled")
    assert service.store.get_round(second["round_id"])["status"] == "cancelled"
