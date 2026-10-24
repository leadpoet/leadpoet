"""The SQL budget authority admits $6 only for frozen full-thinking rounds."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from tests.lab_arena import flat_per_icp_cost500_postgres_test as flat


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/555-lab-arena-full-thinking-six-dollar-cap.sql"
prior = flat.prior
infra = flat.infra


@pytest.fixture(scope="module")
def database():
    fixture = flat.database_with_flat_cost_migration(seed_existing=True)
    db = next(fixture)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            # The flat-cost fixture predates the guarded recovery table that
            # migration 539 reads. Its empty local shape keeps that guard idle.
            cur.execute("CREATE TABLE public.lab_arena_scoring_supersessions ("
                        "round_id text, generation_floor bigint, recovery_doc jsonb, new_round_doc jsonb)")
            cur.execute("GRANT SELECT ON public.lab_arena_scoring_supersessions TO lab_arena_owner")
            cur.execute("ALTER TABLE public.lab_arena_rounds ADD COLUMN "
                        "scoring_generation_floor bigint NOT NULL DEFAULT 0")
            cur.execute((ROOT / "scripts/539-lab-arena-held-recovery-provider-admission.sql").read_text())
            cur.execute(MIGRATION.read_text())
            cur.execute(MIGRATION.read_text())
        yield db
    finally:
        fixture.close()


def seed(conn, *, full=True):
    config = flat.seed(conn)
    if full:
        config["reasoning_capture_policy"] = "full_generated_thinking_v1"
        config["execution_icp_cap_microusd"] = 6_000_000
        config["cost_per_company_microusd"] = 6_000_000
        config["execution_cap_microusd"] = 120_000_000
        with conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s",
                        (json.dumps(config), prior.ROUND))
            cur.execute("SET LOCAL session_replication_role=origin")
        conn.commit()
    return config


@pytest.mark.parametrize("full,cap", [(True, 6_000_000), (False, 4_000_000)])
def test_execution_and_eligibility_share_frozen_boundary(database, full, cap):
    with database[0].connect(**database[1]) as conn:
        seed(conn, full=full)
        lease = prior.claim(conn)
        assert prior.pay(conn, lease, amount=cap)["status"] == "settled"
        for qualified in (0, 1, 5):
            result = infra.eligible(conn, qualified)
            assert result["eligible"] is True
            assert result["eligibility_cap_microusd"] == cap
            assert result["competition_sourcing_microusd"] == cap
        quota = infra.quota(conn, lease, token="a")
        assert quota["admission_cap_microusd"] == cap
        for provider, operation, amount, doc in (
            ("deepline", "deepline.execute", 0, {"reserve_remaining_budget": True, "tool": "exa_search"}),
            ("openrouter", "openrouter.responses", 1, {"reserve_remaining_budget": True}),
            ("scrapingdog", "scrapingdog.scrape", 1, {}),
        ):
            result = prior.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,4500)",
                               (lease["run_id"], "sha256:" + "a" * 64,
                                "sha256:" + {"deepline": "2", "openrouter": "3", "scrapingdog": "4"}[provider] * 64,
                                operation, provider, "miner_key", amount, json.dumps(doc)))
            assert (result["status"], result["reason"]) == ("refused", "money_cap")
        assert infra.state(conn)["inflight_calls"] == 0


def test_full_round_successful_cost_overrun_is_ineligible_but_actual_bill_is_kept(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        lease = prior.claim(conn)
        assert prior.pay(conn, lease, amount=6_000_001)["status"] == "settled"
        result = infra.eligible(conn, 5)
        assert result["eligible"] is False
        assert result["eligibility_reason"] == "execution_cap_exceeded"
        assert result["competition_sourcing_microusd"] == 6_000_001
        assert infra.state(conn)["settled_microusd"] == 6_000_001


def test_failed_bill_consumes_admission_without_changing_successful_cost(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        lease = prior.claim(conn)
        assert infra.failed_call(conn, lease, amount=6_000_000)["status"] == "settled"
        assert infra.eligible(conn, 0)["competition_sourcing_microusd"] == 0
        assert infra.eligible(conn, 0)["eligible"] is True
        result = prior.reserve(conn, lease, call="2")
        assert (result["status"], result["reason"]) == ("refused", "money_cap")


def test_retry_and_tool_calls_share_cap_after_prior_model_failure(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        first = prior.claim(conn)
        assert prior.pay(conn, first, amount=4_000_000)["status"] == "settled"
        assert prior.finish(conn, first, "model_error", token="a")["status"] == "failed"
        retry = prior.claim(conn, "b")
        assert retry["attempt"] == 2
        assert prior.pay(conn, retry, token="b", call="f", amount=2_000_000)["status"] == "settled"
        refused = prior.reserve(conn, retry, token="b", call="c")
        assert (refused["status"], refused["reason"]) == ("refused", "money_cap")
        assert infra.state(conn)["settled_microusd"] == 6_000_000
        assert infra.eligible(conn, 0)["competition_sourcing_microusd"] == 6_000_000


def test_concurrent_reservations_hold_budget_and_release_unused_estimate(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        lease = prior.claim(conn)
        assert prior.pay(conn, lease, amount=4_000_000)["status"] == "settled"
    barrier = Barrier(2)

    def reserve(call_id):
        with database[0].connect(**database[1]) as conn:
            barrier.wait(timeout=10)
            return prior.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,4500)",
                             (lease["run_id"], "sha256:" + "a" * 64,
                              "sha256:" + call_id * 64, "openrouter.chat", "openrouter",
                              "miner_key", 1_500_000, json.dumps({})))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, ("2", "3")))
    assert sorted(result["status"] for result in results) == ["budget_busy", "reserved"]
    assert next(result for result in results if result["status"] == "budget_busy")["reason"] == "provider_calls_inflight"
    admitted = "2" if results[0]["status"] == "reserved" else "3"
    waiting = "3" if admitted == "2" else "2"
    with database[0].connect(**database[1]) as conn:
        state = infra.state(conn)
        assert state["inflight_calls"] == 1
        assert state["reserved_or_uncertain_microusd"] == 1_500_000
        assert infra.eligible(conn, 5)["eligible"] is False
        assert prior.rpc(conn, "lab_arena_mark_dispatched(%s,%s,%s)",
                         (lease["run_id"], "sha256:" + "a" * 64,
                          "sha256:" + admitted * 64))["status"] == "dispatched"
        assert prior.rpc(conn, "lab_arena_settle_call(%s,%s,%s,%s,%s::jsonb,4500)",
                         (lease["run_id"], "sha256:" + "a" * 64,
                          "sha256:" + admitted * 64, 500_000,
                          json.dumps({"status": 200, "call_succeeded": True})))["status"] == "settled"
        assert infra.state(conn)["reserved_or_uncertain_microusd"] == 0
        retry = prior.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,4500)",
                          (lease["run_id"], "sha256:" + "a" * 64,
                           "sha256:" + waiting * 64, "openrouter.chat", "openrouter",
                           "miner_key", 1_500_000, json.dumps({})))
        assert retry["status"] == "reserved"
        assert infra.state(conn)["reserved_or_uncertain_microusd"] == 1_500_000


def test_parallel_holds_fit_below_cap(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        lease = prior.claim(conn)
        assert prior.pay(conn, lease, amount=1_000_000)["status"] == "settled"
    barrier = Barrier(2)

    def reserve(call_id):
        with database[0].connect(**database[1]) as conn:
            barrier.wait(timeout=10)
            return prior.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,4500)",
                             (lease["run_id"], "sha256:" + "a" * 64,
                              "sha256:" + call_id * 64, "openrouter.chat", "openrouter",
                              "miner_key", 1_500_000, json.dumps({})))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, ("2", "3")))
    assert [result["status"] for result in results] == ["reserved", "reserved"]
    with database[0].connect(**database[1]) as conn:
        assert infra.state(conn)["reserved_or_uncertain_microusd"] == 3_000_000


def test_dynamic_tool_holds_remaining_budget_and_unknown_bill_stays_held(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        lease = prior.claim(conn)
        assert prior.pay(conn, lease, amount=4_000_000)["status"] == "settled"
        reserved = prior.reserve(conn, lease, call="2")
        assert reserved["status"] == "reserved"
        assert reserved["amount_microusd"] == 2_000_000
        assert infra.state(conn)["reserved_or_uncertain_microusd"] == 2_000_000
        blocked = prior.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,4500)",
                            (lease["run_id"], "sha256:" + "a" * 64,
                             "sha256:" + "3" * 64, "openrouter.chat", "openrouter",
                             "miner_key", 1_000_000, json.dumps({})))
        assert (blocked["status"], blocked["reason"]) == ("budget_busy", "provider_calls_inflight")
        assert prior.rpc(conn, "lab_arena_mark_dispatched(%s,%s,%s)",
                         (lease["run_id"], "sha256:" + "a" * 64,
                          "sha256:" + "2" * 64))["status"] == "dispatched"
        uncertain = prior.rpc(conn, "lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)",
                              (lease["run_id"], "sha256:" + "a" * 64,
                               "sha256:" + "2" * 64,
                               json.dumps({"reason": "missing_provider_cost", "call_succeeded": True,
                                           "provider_status": 200})))
        assert uncertain["status"] == "uncertain"
        assert infra.state(conn)["reserved_or_uncertain_microusd"] == 2_000_000
        blocked = prior.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,4500)",
                            (lease["run_id"], "sha256:" + "a" * 64,
                             "sha256:" + "3" * 64, "openrouter.chat", "openrouter",
                             "miner_key", 1_000_000, json.dumps({})))
        assert (blocked["status"], blocked["reason"]) == ("budget_busy", "provider_cost_uncertain")


def test_trusted_free_tool_keeps_zero_cost_path_at_full_cap(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        lease = prior.claim(conn)
        assert prior.pay(conn, lease, amount=6_000_000)["status"] == "settled"
        result = prior.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,4500)",
                           (lease["run_id"], "sha256:" + "a" * 64,
                            "sha256:" + "2" * 64, "deepline.execute", "deepline",
                            "miner_key", 0, json.dumps({"tool": "free_simple_company_search"})))
        assert result["status"] == "reserved"
        assert result["amount_microusd"] == 0


def test_host_loss_bill_still_consumes_full_thinking_execution_admission(database):
    with database[0].connect(**database[1]) as conn:
        seed(conn)
        prior.host_loss(conn, amount=4_000_000)
        retry = prior.claim(conn, "b")
        state = infra.state(conn)
        assert state["settled_microusd"] == 4_000_000
        assert state["competition_settled_microusd"] == 0
        reserved = prior.reserve(conn, retry, token="b", call="f")
        assert reserved["status"] == "reserved"
        assert reserved["amount_microusd"] == 2_000_000
        assert infra.eligible(conn, 0)["competition_sourcing_microusd"] == 0
