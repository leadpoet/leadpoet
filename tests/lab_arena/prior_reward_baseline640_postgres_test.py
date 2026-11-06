"""Prior reward authority gates only new baseline work, without changing rows."""
from copy import deepcopy
import json
from pathlib import Path
import uuid

import pytest

from tests.lab_arena import champion_baseline_freshness627_postgres_test as prior

SQL = Path(__file__).parents[2] / "scripts/640-lab-arena-prior-reward-baseline-gate.sql"
SIGNATURE = "public.lab_arena__baseline_source_current_v1(public.lab_arena_rounds)"
database = prior.database
scalar = prior.scalar


@pytest.fixture
def prepared(database):
    fixture = prior.prepared.__wrapped__(database)
    p = next(fixture)
    prior.change(p, "UPDATE public.lab_arena_rounds SET rewards_enabled=true WHERE round_id=%s",
                 (prior.ROUND,))
    with p.conn.cursor() as q:
        p.original_helper = scalar(q, "SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
    try:
        yield p
    finally:
        p.conn.rollback()
        with p.conn.cursor() as q:
            q.execute(p.original_helper)
            q.execute("REVOKE ALL ON FUNCTION " + SIGNATURE +
                      " FROM PUBLIC,anon,authenticated,service_role,lab_arena_service")
        p.conn.commit()
        fixture.close()


def apply(p):
    with p.conn.cursor() as q:
        q.execute(SQL.read_text())
    p.conn.commit()


def data_snapshot(p):
    result = {}
    with p.conn.cursor() as q:
        for table, key in (("lab_arena_rounds", "round_id"),
                           ("lab_arena_submissions", "submission_id"),
                           ("lab_arena_runs", "run_id"), ("lab_arena_ledger", "entry_id")):
            result[table] = scalar(q, "SELECT jsonb_object_agg(" + key +
                ",encode(extensions.digest(to_jsonb(a)::text,'sha256'),'hex')) FROM public." + table + " a")
    return result


def pending_predecessor(p):
    # The current baseline matches the last promoted November3 champion.
    # November4 is an ordinary unresolved day, never a reopened published receipt.
    prior.change(p, "UPDATE public.lab_arena_rounds SET status='stage2_judged',"
        "published_at=NULL,publication_doc=NULL,baseline_promoted_at=NULL,promotion_doc=NULL "
        "WHERE round_id=%s", (prior.prior.ROUND,))
    with p.conn.cursor() as q:
        champion = scalar(q, "SELECT to_jsonb(r) FROM public.lab_arena_rounds r "
            "WHERE round_id='arena-2026-11-03'")
    config = deepcopy(p.store.get_round(prior.ROUND)["configuration_doc"])
    config["baseline_source_url"] = ("https://github.com/leadpoet/leadpoet-sales-agent/archive/" +
                                   champion["promotion_doc"]["commit"] + ".tar.gz")
    config["baseline_reasoning_configuration"] = {
        k: v for k, v in champion["configuration_doc"].items()
        if k in ("reasoning_capture_policy", "reasoning_route_registry")}
    winner = champion["publication_doc"]["king_decision"]
    prior.change(p, "UPDATE public.lab_arena_rounds SET champion_submission_id=%s,"
        "champion_hotkey=%s,configuration_doc=%s::jsonb WHERE round_id=%s",
        (winner["winner_submission_id"], winner["king_hotkey"], json.dumps(config), prior.ROUND))


def test_original_gap_then_baseline_claim_replay_and_reservation_wait(prepared):
    p = prepared
    pending_predecessor(p)
    assert prior.current(p) is True  # Regression: original627 misses ordinary unresolved days.
    baseline = prior.run(p)
    request = uuid.uuid4().hex
    assert prior.claim(p, request)["run_id"] == baseline
    before = data_snapshot(p)
    apply(p)
    assert prior.current(p) is False
    assert prior.freeze(p) == {"status": "promotion_pending"}
    assert prior.claim(p, request) == {"status": "promotion_pending"}
    assert prior.reserve(p, baseline, "sha256:" + "6" * 64) == {"status": "promotion_pending"}
    assert data_snapshot(p) == before


def test_pending_baseline_waits_while_unrelated_miner_claim_still_works(prepared):
    p = prepared
    pending_predecessor(p)
    prior.run(p)
    apply(p)
    assert prior.claim(p) == {"status": "promotion_pending"}
    miner = prior.run(p, baseline=False)
    assert prior.claim(p)["run_id"] == miner


def test_original_freeze_waits_without_freezing_or_rebinding_source(prepared):
    p = prepared
    pending_predecessor(p)
    prior.change(p, "UPDATE public.lab_arena_rounds SET status='open',"
        "champion_funding_frozen=false,champion_submission_id=NULL,champion_hotkey=NULL "
        "WHERE round_id=%s", (prior.ROUND,))
    assert prior.current(p) is True
    before = data_snapshot(p)
    apply(p)
    assert prior.freeze(p) == {"status": "promotion_pending"}
    assert data_snapshot(p) == before
    assert p.store.get_round(prior.ROUND)["champion_funding_frozen"] is False


def test_prior_terminal_without_new_crown_unblocks_same_frozen_source(prepared):
    p = prepared
    pending_predecessor(p)
    before = p.store.get_round(prior.ROUND)
    apply(p)
    assert prior.current(p) is False
    prior.change(p, "UPDATE public.lab_arena_rounds SET status='published',published_at=now(),"
        "publication_doc=%s::jsonb WHERE round_id=%s",
        (json.dumps({"king_decision": {"outcome": "defended"}}), prior.prior.ROUND))
    assert prior.current(p) is True
    assert p.store.get_round(prior.ROUND) == before


def test_only_new_dispatch_waits_existing_dispatch_and_settlement_survive(prepared):
    p = prepared
    pending_predecessor(p)
    baseline = prior.run(p)
    assert prior.claim(p)["run_id"] == baseline
    dispatched, pending = "sha256:" + "7" * 64, "sha256:" + "8" * 64
    assert prior.reserve(p, baseline, dispatched)["status"] == "reserved"
    assert p.store.mark_dispatched(run_id=baseline,
        lease_token_hash=prior.hash_lease_token("freshness627"),
        call_identity=dispatched)["status"] == "dispatched"
    assert prior.reserve(p, baseline, pending)["status"] == "reserved"
    apply(p)
    assert p.store.mark_dispatched(run_id=baseline,
        lease_token_hash=prior.hash_lease_token("freshness627"),
        call_identity=dispatched)["status"] == "dispatched"
    assert p.store.settle_call(run_id=baseline,
        lease_token_hash=prior.hash_lease_token("freshness627"),
        call_identity=dispatched,actual_microusd=123,
        terminal_response={"status": 200})["status"] == "settled"
    assert p.store.mark_dispatched(run_id=baseline,
        lease_token_hash=prior.hash_lease_token("freshness627"),
        call_identity=pending) == {"status": "promotion_pending"}
    assert p.store.mark_dispatched(run_id=baseline,
        lease_token_hash=prior.hash_lease_token("freshness627"),
        call_identity=pending)["status"] == "recovered"
    with p.conn.cursor() as q:
        assert scalar(q, "SELECT array_agg(entry_kind ORDER BY entry_id) "
            "FROM public.lab_arena_ledger WHERE call_identity=%s", (pending,)) == ["reservation", "recovery"]


@pytest.mark.parametrize("mutation,args", [
    ("evaluation_date='2026-11-05'", ()),
    ("evaluation_date='2026-11-06'", ()),
    ("configuration_doc=jsonb_set(configuration_doc,'{mode}','\"shadow\"')", ()),
    ("rewards_enabled=false", ()),
    ("configuration_doc=jsonb_set(configuration_doc,'{network_name}','\"other\"')", ()),
    ("configuration_doc=jsonb_set(configuration_doc,'{netuid}','72')", ()),
    ("status='cancelled'", ()),
])
def test_irrelevant_or_terminal_prior_round_does_not_block(prepared, mutation, args):
    p = prepared
    pending_predecessor(p)
    prior.change(p, "UPDATE public.lab_arena_rounds SET " + mutation + " WHERE round_id=%s",
                 args + (prior.prior.ROUND,))
    apply(p)
    assert prior.current(p) is True


def test_published_prior_and_superseded_older_nonterminal_do_not_block(prepared):
    p = prepared
    assert prior.current(p) is True
    apply(p)
    assert prior.current(p) is True
    prior.change(p, "UPDATE public.lab_arena_rounds SET status='stage2_judged',"
        "publication_doc=NULL,published_at=NULL,baseline_promoted_at=NULL "
        "WHERE round_id='arena-2026-11-03'")
    # Published November4 already superseded the older November3 authority.
    assert prior.current(p) is True


def test_null_evaluation_day_uses_existing_utc_cutoff_authority(prepared):
    p = prepared
    pending_predecessor(p)
    prior.change(p, "UPDATE public.lab_arena_rounds SET evaluation_date=NULL WHERE round_id=%s",
                 (prior.prior.ROUND,))
    assert prior.current(p) is True
    apply(p)
    assert prior.current(p) is False
    # The current round also uses the original627 NULL-date fallback.
    row = p.store.get_round(prior.ROUND)
    config = row["configuration_doc"]
    config["schedule"]["submission_cutoff"] = "2026-11-05T00:00:00Z"
    prior.change(p, "UPDATE public.lab_arena_rounds SET evaluation_date=NULL,"
        "configuration_doc=%s::jsonb WHERE round_id=%s", (json.dumps(config), prior.ROUND))
    assert prior.current(p) is False


@pytest.mark.parametrize("mutation", [
    "status='published'", "status='cancelled'", "rewards_enabled=false",
    "configuration_doc=configuration_doc-'reasoning_capture_policy'",
])
def test_existing_current_round_exemptions_remain(prepared, mutation):
    p = prepared
    pending_predecessor(p)
    prior.change(p, "UPDATE public.lab_arena_rounds SET " + mutation + " WHERE round_id=%s",
                 (prior.ROUND,))
    apply(p)
    assert prior.current(p) is True


def test_function_only_replay_preserves_rows_oid_and_security(prepared):
    p = prepared
    before = data_snapshot(p)
    with p.conn.cursor() as q:
        metadata = scalar(q, "SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure",
                          (SIGNATURE,))
    apply(p)
    with p.conn.cursor() as q:
        assert scalar(q, "SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure",
                      (SIGNATURE,)) == metadata
        definition = scalar(q, "SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
    apply(p)
    with p.conn.cursor() as q:
        assert scalar(q, "SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,)) == definition
    assert data_snapshot(p) == before


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("drift", ["body", "security"])
def test_source_and_security_drift_are_rejected_before_replay(prepared, installed, drift):
    p = prepared
    if installed:
        apply(p)
    with p.conn.cursor() as q:
        if drift == "body":
            definition = scalar(q, "SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
            q.execute(definition.replace("evaluation_day IS NULL", "evaluation_day IS NOT NULL"))
        else:
            q.execute("GRANT EXECUTE ON FUNCTION " + SIGNATURE + " TO lab_arena_service")
    p.conn.commit()
    before = data_snapshot(p)
    with pytest.raises(p.pg.Error, match="baseline640 (preimage|security) differs"):
        apply(p)
    p.conn.rollback()
    assert data_snapshot(p) == before
