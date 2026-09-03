"""Capacity: one round at the challenger cap on disposable PostgreSQL.

Every registered miner may enter one agent per day, up to ``MAX_CHALLENGERS``
(256). This test admits that many, runs both stages with fake sandboxes and
in-process validators, and reports the wall time of each service step so the
floor and the scoring workers can be sized (RUNBOOK section 8). It is slow,
so it runs only when ``LAB_ARENA_SCALE_TEST=1``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import pytest

from lab_arena import contracts

from tests.lab_arena.test_lab_arena_service_round import Harness, connect, database  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(os.environ.get("LAB_ARENA_SCALE_TEST") != "1", reason="set LAB_ARENA_SCALE_TEST=1 to run the challenger-cap round")

CHALLENGERS = int(os.environ.get("LAB_ARENA_SCALE_CHALLENGERS", str(contracts.MAX_CHALLENGERS)))
RUNNERS = ["alpha", "beta", "gamma"]


def test_round_at_the_challenger_cap_scores_every_participant_and_publishes(connect, tmp_path):
    timings = {}

    def timed(label, call):
        started = time.monotonic()
        result = call()
        timings[label] = round(time.monotonic() - started, 2)
        return result

    flavors = ["Scale-%03d" % index for index in range(CHALLENGERS)]
    harness = Harness(connect, tmp_path, challengers=flavors, runners=RUNNERS)
    harness.max_challengers = contracts.MAX_CHALLENGERS
    harness.service = harness.build_service()
    service = harness.service
    harness.chain.epoch = 30500
    configuration = service.create_round(datetime(2026, 10, 20, 0, 0, tzinfo=timezone.utc))
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    timed("submissions", lambda: [harness.submit(flavor, round_id) for flavor in flavors])
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    freeze = timed("freeze_and_benchmark", lambda: service.advance_round(round_id))
    assert freeze["status"] == "ok", freeze
    participants = service.store.get_round(round_id)["participants"]
    assert len(participants) == CHALLENGERS
    for participant in participants:
        harness.flavors.setdefault(participant["image_digest"], "King")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = timed("stage1_open", lambda: service.advance_round(round_id))
    assert opened["assignments"] == 20 * CHALLENGERS
    timed("stage1_executions", lambda: harness.run_stage_with_runners(len(RUNNERS)))
    assert timed("stage1_close", lambda: service.advance_round(round_id))["status"] == "ok"
    assert timed("stage1_scoring_open", lambda: service.advance_round(round_id))["assignments"] == 20 * CHALLENGERS
    timed("stage1_scorings", lambda: harness.run_stage_with_runners(len(RUNNERS)))
    assert timed("stage1_scoring_close", lambda: service.advance_round(round_id))["status"] == "closed"
    assert timed("stage1_score_stage", lambda: service.advance_round(round_id))["status"] == "ok"
    assert harness.status() == "stage1_scored"
    finalists = service.store.get_round(round_id)["finalists"]
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    stage2 = timed("stage2_open", lambda: service.advance_round(round_id))
    assert stage2["status"] == "ok" and harness.status() == "stage2"
    timed("stage2_executions", lambda: harness.run_stage_with_runners(len(RUNNERS)))
    timed("stage2_to_published", lambda: harness.advance_until("published", runners=len(RUNNERS), max_steps=200))
    round_row = service.store.get_round(round_id)
    assert round_row["status"] == "published" and round_row["king_outcome"] in ("crowned", "defended")
    stage1 = json.loads(harness.objects.get(round_row["stage1_scores_ref"]).decode())
    assert len(stage1["submission_scores"]) == CHALLENGERS and len(stage1["rows"]) == 20 * CHALLENGERS
    timing = json.loads(harness.objects.get("arena/%s/timing/stage1_scoring.json" % round_id).decode())
    assert timing["judge_executions"] == 20 * CHALLENGERS and timing["work_items"] == 20 * CHALLENGERS
    report = {"challengers": CHALLENGERS, "finalists": len(finalists), "runs_stage1": 40 * CHALLENGERS, "timings_seconds": timings}
    (tmp_path / "scale_report.json").write_text(json.dumps(report, indent=2))
    print("\nSCALE REPORT " + json.dumps(report))
