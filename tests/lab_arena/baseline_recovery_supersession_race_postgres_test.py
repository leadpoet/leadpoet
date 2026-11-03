"""A validated no-op cannot publish old authority across a model supersession."""
from copy import deepcopy
from dataclasses import replace

import pytest

from lab_arena import scoring
from lab_arena.store import ArenaStoreError
from tests.lab_arena import test_lab_arena_service_round as flows
from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge, _enable_verified_proxy_runtime
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.model_policy_supersession517_postgres_test import invoke, operator_hold
from tests.lab_arena.test_integrity_round import IntegrityHarness


@pytest.fixture
def database():
    yield from current_service_database()


def test_supersession_after_validated_noop_cannot_publish_old_projection(database, tmp_path, monkeypatch):
    pg, dsn = database
    h = IntegrityHarness(lambda: pg.connect(**dsn), tmp_path, challengers=["Control"], runners=["alpha"])
    h.service.config.defaults = replace(h.service.config.defaults,
        execution_sequence_from="2000-01-01T00:00:00Z", company_judgment_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=True, rewards_enabled=False, benchmark_icp_count=2)
    h.service.config.daily_icp_source = lambda **kwargs: {"status": "ready", "set_id": kwargs["set_id"], "icps": daily_icps()[:2]}
    _enable_verified_proxy_runtime(h)
    monkeypatch.setattr(flows, "deterministic_scorer", _baseline_first_judge)
    monkeypatch.setitem(scoring.DEFAULT_JUDGE_MODELS, "role_batch_check", "google/gemini-2.5-flash")
    prices = flows.price_table
    monkeypatch.setattr(flows, "price_table", lambda: prices((*flows.PRICED_MODELS, "google/gemini-2.5-flash")))
    h.sandbox.empty_outputs.update((flavor, position) for flavor in ("Control", "PublicBaseline") for position in (0, 1))
    try:
        assert flows._start_round(h, day=25, epoch=62525) == 2
        h.clock.advance_to(h.schedule()["stage_1_start"])
        h.advance_until("scored", runners=1)
        row = h.service.store.get_round(h.round_id)
        pending = h.service.store.list_runs(h.round_id)
        assert all(run["per_icp_score"] is None for run in pending if run["kind"] == "execute" and run["stage"] == 1)
        captured = {}
        transition = h.service.store.transition_round

        class Draft(Exception):
            pass

        def capture(round_id, expected, status, patch):
            assert (round_id, expected, status) == (h.round_id, "scored", "published")
            captured.update(deepcopy(patch))
            raise Draft

        with monkeypatch.context() as patch:
            patch.setattr(h.service.store, "transition_round", capture)
            with pytest.raises(Draft):
                h.service.publish(h.round_id)
        before = h.service.store.list_runs(h.round_id)
        assert all(isinstance(run["per_icp_score"], (int, float)) for run in before if run["kind"] == "execute")
        # Publication materialized missing scores; a second read never rewrites
        # accepted projections before the supersession race.
        h.service._record_recovered_judgment_scores(row)
        assert h.service.store.list_runs(h.round_id) == before
        revised = deepcopy(row)
        revised["configuration_doc"]["scorer_policy"]["judge_models"]["role_batch_check"] = "google/gemini-3.5-flash-lite"
        revised["configuration_doc"]["scorer_policy"]["env_bindings"]["ARENA_ROLE_BATCH_MODEL"] = "google/gemini-3.5-flash-lite"
        items = [item for stage in (1, 2) for item in h.service._scoring_work_items(
            revised, stage, row[f"stage{stage}_scoring_plan_doc"])]
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton")
            generation = cur.fetchone()[0]
            assert operator_hold(cur, row, generation, True)["status"] == "held"
            superseded = invoke(cur, row, items)
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s", (h.round_id,))
            fresh = cur.fetchone()[0]
            assert operator_hold(cur, fresh, generation, False)["status"] == "released"
        assert superseded["queued"] == 4
        stale = transition(h.round_id, "stage2_judged", "scored", {})
        assert stale["status"] == "stale" and stale["round_status"] == "stage2_scoring"
        assert type(stale["status_generation"]) is int
        after = h.service.store.list_runs(h.round_id)
        assert all(run["per_icp_score"] is None and run["qualification_doc"] is None for run in after if run["kind"] == "execute")
        assert [run for run in after if run["kind"] == "score" and run["attempt"] == 1] == [
            run for run in before if run["kind"] == "score"]
        assert h.service.store.close_scoring(h.round_id, 2)["status"] == "closed"
        # Scored is a continuation phase; final publication must recheck the
        # current generation independently, even after the phase CAS succeeds.
        assert transition(h.round_id, "stage2_judged", "scored", {})["status"] == "ok"
        with pytest.raises(ArenaStoreError, match="baseline_incomplete"):
            transition(h.round_id, "scored", "published", captured)
        final = h.service.store.get_round(h.round_id)
        assert final["status"] == "scored" and final["publication_doc"] is None
        assert final["baseline_promoted_at"] is None and final["reward_activated_at"] is None
    finally:
        h.service.store.close()
