"""Test actual process isolation, funding-aware dispatch and provider safety at the worker entry point."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
from lab_arena import collection_governor as gov
from lab_arena.collection_prices import CatalogError
from scripts import run_locked_collection as worker
from scripts.prepare_collection_experiment import store_private_json
from tests.lab_arena.collection_pool_worker_test import pool_raw, SMALL, EXPERT, PROMPT
from lab_arena.collection_experiment import load_experiment


def isolated_tick(args, lane, deadline):
    """Each lane waits until all four independent processes have started."""
    root = Path(args.directory)
    (root / f"lane-{lane}").write_text(str(os.getpid()))
    until = time.monotonic() + 12
    while len(list(root.glob("lane-*"))) < 4:
        if time.monotonic() > until:
            raise AssertionError("parallel processes did not overlap")
        time.sleep(0.01)
    return 1


def test_four_independent_workers_overlap_without_shared_harness_globals(tmp_path):
    assert (
        worker.run_governed_lanes(
            argparse.Namespace(max_jobs=1, directory=tmp_path),
            4,
            tick=isolated_tick,
            timer_active=lambda: True,
        )
        == 4
    )
    assert len({p.read_text() for p in tmp_path.glob("lane-*")}) == 4


def test_only_healthy_lanes_continue_after_ambiguous_claim_and_refill_empty_lane():
    seen = []

    def tick(args, lane, deadline):
        seen.append(lane)
        if lane == 0:
            raise IOError("unknown claim")
        if lane == 1 and seen.count(1) == 1:
            return 0
        return 1

    with pytest.raises(RuntimeError, match="unconfirmed operation"):
        worker.run_governed_lanes(
            argparse.Namespace(max_jobs=4),
            4,
            executor=ThreadPoolExecutor(max_workers=4),
            tick=tick,
            timer_active=lambda: True,
        )
    assert seen.count(0) == 1 and all(seen.count(i) >= 4 for i in (1, 2, 3))


def test_tick_uses_only_a_confirmed_terminal_failure_for_further_work(
    monkeypatch, tmp_path
):
    prompt = tmp_path / "prompt.json"
    prompt.write_text("{}")
    args = argparse.Namespace(
        prompt=prompt, offer_directory=tmp_path, successor_first_model=None
    )
    seen = []

    def run_options(args, parser, raw, *, before_claim, on_pool_settled, epoch_for_tick):
        before_claim()
        if raw.get("confirmed"):
            on_pool_settled("failed")
        raise RuntimeError("experiment interrupted")

    db = SimpleNamespace(rpc=lambda *a: 100, close=lambda: None)
    monkeypatch.setattr(worker, "evidence_stores", lambda: (db, None))
    monkeypatch.setattr(worker, "_worker_timer_active", lambda: True)
    monkeypatch.setattr(worker, "_run_options", run_options)
    confirmed = [True, False]

    def tick(p, d, db, run, **kwargs):
        return run(
            {"confirmed": confirmed.pop(0)}, before_claim=lambda: seen.append("claimed")
        )

    monkeypatch.setattr(worker, "poll_pool_experiments", tick)
    assert worker.governed_tick(args, 0, time.monotonic() + 5) == 1
    with pytest.raises(RuntimeError, match="interrupted"):
        worker.governed_tick(args, 0, time.monotonic() + 5)
    assert seen == ["claimed", "claimed"]
    monkeypatch.setattr(worker, "_worker_timer_active", lambda: False)
    confirmed.append(True)
    assert worker.governed_tick(args, 0, time.monotonic() + 5) == 0 and len(seen) == 2


def test_catalog_hold_rechecks_another_model_in_the_same_frozen_program(tmp_path):
    raw = pool_raw(repeats=1)
    raw["arms"].append(dict(raw["arms"][0], id="expert", model=EXPERT))
    raw["routes"][EXPERT] = "coreweave/fp8"
    raw.update(max_slots=2, budget_microusd=200000)
    pack = load_experiment(raw, PROMPT)
    store_private_json(tmp_path / (pack.batch_id + ".json"), raw)
    seen = []
    db = SimpleNamespace(rpc=lambda *a: "open")

    def run(pack, *, before_claim):
        if not seen:
            seen.append(SMALL)
            raise gov.HeldRoute(
                SMALL, CatalogError("strict_retention_unavailable", "fixture")
            )
        before_claim()
        seen.append(EXPERT)
        return 1

    assert (
        worker.poll_pool_experiments(PROMPT, tmp_path, db, run, lane=1, fundable=100000)
        == 1
    )
    assert seen == [SMALL, EXPERT]
    seen.clear()
    assert (
        worker.poll_pool_experiments(PROMPT, tmp_path, db, run, lane=1, fundable=99999)
        == 0
        and not seen
    )
    assert (tmp_path / ".last-pool-program-1").is_file()


def test_epoch_source_is_native_or_cached_and_refuses_missing_measurement(monkeypatch):
    graph = SimpleNamespace(finalized_block_hash=lambda n: "0x" + "ab" * 32)
    snap = SimpleNamespace(last_epoch_block=900, settlement_epoch_id=lambda cut: 199)
    monkeypatch.setattr(gov.chain, "finalized_epoch_snapshot", lambda g: snap)
    values = []

    def rpc(op, params):
        assert op == "collection_epoch_capacity_v1"
        assert params["p_epoch"] == 199 and params["p_boundary"] == 900
        if params["p_valuation"] is not None:
            values.append(params["p_valuation"])
        return bool(values)

    db = SimpleNamespace(rpc=rpc)
    cost = lambda g, b: {"block": b}
    assert gov.prepare_epoch(graph, None, db, nullcontext, value=cost) == 199
    assert (
        gov.prepare_epoch(
            graph,
            None,
            db,
            nullcontext,
            value=lambda *a: pytest.fail("duplicate oracle access"),
        )
        == 199
    )
    assert values == [{"block": 900}]
    with pytest.raises(ValueError, match="no measured"):
        gov.prepare_epoch(
            graph, None, SimpleNamespace(rpc=lambda *a: False), nullcontext, value=cost
        )


def test_release_template_enables_only_governed_service_and_30_second_backoff():
    repo = Path(__file__).resolve().parents[2] / "deploy"
    assert (
        "--governed-workers --productive-burst"
        in (repo / "leadpoet-collection-paid.conf").read_text()
    )
    timer = (repo / "leadpoet-collection-paid-timer.conf").read_text()
    assert (
        "OnUnitInactiveSec=\nOnUnitInactiveSec=30s" in timer
        and "AccuracySec=1s" in timer
    )
    assert (
        "OnUnitInactiveSec=5min"
        in (repo / "leadpoet-collection-pilot.timer").read_text()
    )


def test_catalog_and_provider_retry_after_extend_only_one_pinned_route(monkeypatch):
    sent = []
    db = SimpleNamespace(rpc=lambda op, args: sent.append((op, args)))
    client = gov.GovernedIntake(db, None, 200, {EXPERT: "coreweave/fp8"}, lambda: 200)
    err = CatalogError("provider_catalog_throttled", "private")
    monkeypatch.setattr(
        gov, "endpoint_prices", lambda *a, **k: (_ for _ in ()).throw(err)
    )
    with pytest.raises(gov.HeldRoute) as caught:
        client.prices(EXPERT)
    assert caught.value.model == EXPERT and caught.value.cause is err
    client.respect_retry_after(EXPERT, "125")
    client.respect_retry_after(EXPERT, "not a delay")
    assert sent == [
        (
            "collection_route_backoff_v1",
            {"p_model": EXPERT, "p_route": "coreweave/fp8", "p_wait": v},
        )
        for v in [0, 125]
    ]


def test_empty_funding_never_runs_providers_or_processes(tmp_path, monkeypatch):
    cfg = tmp_path / "prompt.json"
    cfg.write_text("{}")
    spec = pool_raw(repeats=1)
    path = tmp_path / (load_experiment(spec, PROMPT).batch_id + ".json")
    store_private_json(path, spec)
    seen = []

    def rpc(op, params):
        seen.append(op)
        if op == "collection_pool_active_v1":
            return True
        if op == "collection_worker_funding_v1":
            return 99999
        pytest.fail("unfunded worker tried to read additional services")

    monkeypatch.setattr(
        worker, "evidence_stores", lambda: (SimpleNamespace(rpc=rpc), None)
    )
    monkeypatch.setattr(
        worker,
        "run_governed_lanes",
        lambda *a: pytest.fail("started a worker without signed capacity"),
    )
    assert (
        worker.main(
            [
                "--batch",
                str(cfg),
                "--prompt",
                str(cfg),
                "--offer-directory",
                str(tmp_path),
                "--productive-burst",
                "--max-jobs",
                "64",
                "--governed-workers",
            ]
        )
        == 0
    )
    assert seen == ["collection_pool_active_v1", "collection_worker_funding_v1"]


def test_fundable_program_skips_completed_finite_and_running_packs(tmp_path):
    from types import SimpleNamespace
    spec = pool_raw(repeats=1)
    spec.pop("successor_seeds")
    name = load_experiment(spec, PROMPT).batch_id
    store_private_json(tmp_path / (name+".json"), spec)
    states = []
    def db(state):
        def rpc(name, args):
            assert name == "collection_pool_pack_state_v1" and args["p_slots"] == 1
            states.append(state)
            return state
        return SimpleNamespace(rpc=rpc)
    assert not worker._fundable_program(tmp_path,db("open"),0)
    assert not states
    assert not worker._fundable_program(tmp_path,db("complete"),100000)
    assert not worker._fundable_program(tmp_path,db("running"),100000)
    assert worker._fundable_program(tmp_path,db("open"),100000)
